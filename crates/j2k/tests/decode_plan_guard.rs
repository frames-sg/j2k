// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode guard: every route a decode takes writes the bytes of the planes
//! route at an unlimited planes budget, and the heap each decode holds at
//! its peak is recorded, so that a change to how decodes are planned can be
//! checked for "bytes identical, peak not higher".
//!
//! One test decodes generated images (see `fixtures`) for every request
//! (full image, region, reduced resolution, region at half resolution,
//! bounded rows), output (8-bit, 16-bit, native components), parallelism
//! (serial, and pools of 4 and 11 threads built here) and planes budget
//! (default, and the decoder's test hook at 0, 1 MiB and unlimited) that
//! applies. Per decode it records the hash of the output or the class of
//! the error, and the most heap bytes live at once during the decode above
//! those live before it, from the counting allocator of `j2k_alloc_probe`
//! (pool threads included). The cases run one after another in this test
//! alone, so nothing else allocates while one is measured; a case on a pool
//! runs five times or more in an optimized build, whose highest peak is
//! kept.
//!
//! The test fails when a decode writes other bytes than the planes route
//! (cropped to the region for regions and rows), changes outcome or typed
//! error class, exceeds its accepted peak, or drifts from the baseline's
//! exact case set. It prints the table and writes it to
//! `decode_plan_guard/table.txt` under the target's test directory, beside a
//! summary. `decode_plan_guard/baseline-<arch>-<os>.txt` holds the table last
//! accepted for that target, measured with `cargo test --release`: a peak may
//! exceed its accepted value only by that value's accepted spread plus
//! 64 KiB plus 1/64 of the accepted peak. Copy the table over the baseline
//! only after reviewing and explaining an accepted change; peaks compare
//! only between runs of the same target and profile.

#[path = "decode_plan_guard/decode.rs"]
mod decode;
#[path = "decode_plan_guard/fixtures.rs"]
mod fixtures;
#[path = "decode_plan_guard/report.rs"]
mod report;

use std::collections::HashMap;
use std::time::Instant;

use decode::{
    error_class, hash, native_hash, native_planes, output_rect, row_format, Budget, Decode,
    Measured, NativePlane, Outcome, Output, Request,
};
use fixtures::{Extent, Fixture};
use j2k::{CpuDecodeParallelism, J2kDecoder, J2kScratchPool};
use j2k_core::{Downscale, PixelFormat, Rect};
use rayon::{ThreadPool, ThreadPoolBuilder};
use report::{compare_with_baseline, table, write_report, Findings, Report, Row};

const POOL_THREADS: usize = 4;
const WIDE_POOL_THREADS: usize = 11;
const SCALES: [Downscale; 3] = [Downscale::Half, Downscale::Quarter, Downscale::Eighth];
const MIB: usize = 1 << 20;

/// A peak passes its baseline when it exceeds it by more than the
/// baseline's spread plus this plus 1/64 of the baseline peak (allocator
/// rounding, pool job boxes, warning vectors and other small allocations no
/// decode plans).
const SLACK_BYTES: usize = 64 * 1024;

/// Times a decode on a pool runs in an optimized build: how many tiles or
/// bands a pool holds at once depends on scheduling, so its peak varies from
/// run to run, and the table keeps the highest and the spread. A debug build
/// is not compared with the baseline's peaks and decodes each case once.
const POOL_REPEATS: usize = if cfg!(debug_assertions) { 1 } else { 5 };

/// Times in all a pool decode of a large fixture runs, and one whose peaks
/// spread by more than [`SLACK_BYTES`] over [`POOL_REPEATS`]: tiles
/// decoding at once on the planes route move the peak by up to a fifth, and
/// some cases have two peaks 3-6% apart that a few runs can miss.
const NOISY_REPEATS: usize = 20;

#[derive(Clone, Copy)]
struct Case {
    fixture: usize,
    request: Request,
    output: Output,
    threads: usize,
    budget: Budget,
}

impl Case {
    fn id(&self, fixtures: &[Fixture]) -> String {
        let scale = |scale: Downscale| match scale {
            Downscale::None => "full",
            Downscale::Half => "half",
            Downscale::Quarter => "quarter",
            Downscale::Eighth => "eighth",
            _ => "other",
        };
        let request = match self.request {
            Request::Full => "full".to_string(),
            Request::Region => "region".to_string(),
            Request::Scaled(s) => format!("scaled-{}", scale(s)),
            Request::RegionScaled(s) => format!("region-{}", scale(s)),
            Request::Rows => "rows".to_string(),
        };
        let output = match self.output {
            Output::Packed(fmt) => format!("{fmt:?}").to_lowercase(),
            Output::Native => "native".to_string(),
        };
        let parallelism = match self.threads {
            1 => "serial".to_string(),
            threads => format!("pool{threads}"),
        };
        let budget = match self.budget {
            Budget::Default => "default".to_string(),
            Budget::Planes(usize::MAX) => "planes-max".to_string(),
            Budget::Planes(bytes) if bytes >= MIB => format!("planes-{}m", bytes / MIB),
            Budget::Planes(bytes) => format!("planes-{bytes}"),
        };
        let name = fixtures[self.fixture].name;
        format!("{name}/{request}/{output}/{parallelism}/{budget}")
    }
}

/// A region crossing the middle of the image (and its inner tile edges).
fn region(fixture: &Fixture) -> Rect {
    Rect {
        x: fixture.width / 4,
        y: fixture.height / 4,
        w: (fixture.width / 2).max(1),
        h: (fixture.height / 2).max(1),
    }
}

fn components(fixture: &Fixture) -> u16 {
    J2kDecoder::inspect(&fixture.bytes)
        .expect("inspect guard fixture")
        .components
}

/// The requests the guard runs on a fixture, each with the planes budgets
/// that apply to it (the test hook holds only full-image decodes, at full
/// or reduced resolution, packed or native).
fn requests(fixture: &Fixture) -> Vec<(Request, Vec<Budget>)> {
    let mut budgets = vec![
        Budget::Default,
        Budget::Planes(0),
        Budget::Planes(usize::MAX),
    ];
    let default = vec![Budget::Default];
    match fixture.extent {
        Extent::Every => {
            let mut requests = vec![(Request::Full, budgets.clone())];
            requests.extend(SCALES.map(|s| (Request::Scaled(s), budgets.clone())));
            requests.extend([
                (Request::Region, default.clone()),
                (Request::RegionScaled(Downscale::Half), default.clone()),
                (Request::Rows, default),
            ]);
            requests
        }
        Extent::Large => {
            budgets.push(Budget::Planes(MIB));
            vec![
                (Request::Full, budgets.clone()),
                (Request::Scaled(Downscale::Half), budgets),
                (Request::Region, default),
            ]
        }
    }
}

fn cases(fixtures: &[Fixture]) -> Vec<Case> {
    let mut cases = Vec::new();
    for (index, fixture) in fixtures.iter().enumerate() {
        let components = components(fixture);
        let mut pools = vec![1, POOL_THREADS];
        if fixture.wide_pool {
            pools.push(WIDE_POOL_THREADS);
        }
        for (request, budgets) in requests(fixture) {
            let outputs: Vec<Output> = match request {
                Request::Rows => [false, true]
                    .into_iter()
                    .filter_map(|sixteen| row_format(components, sixteen))
                    .map(Output::Packed)
                    .collect(),
                Request::Full | Request::Region => fixture
                    .formats
                    .map(Output::Packed)
                    .into_iter()
                    .chain([Output::Native])
                    .collect(),
                Request::Scaled(_) | Request::RegionScaled(_) => {
                    fixture.formats.map(Output::Packed).to_vec()
                }
            };
            for output in outputs {
                for &threads in &pools {
                    if threads == WIDE_POOL_THREADS && request != Request::Full {
                        continue;
                    }
                    for &budget in &budgets {
                        cases.push(Case {
                            fixture: index,
                            request,
                            output,
                            threads,
                            budget,
                        });
                    }
                }
            }
        }
    }
    cases
}

/// What the planes route writes at an unlimited planes budget, serially:
/// every packed format and scale a case reads, and the native planes.
#[derive(Default)]
struct References {
    packed: HashMap<(usize, PixelFormat, Downscale), Result<Vec<u8>, String>>,
    native: HashMap<usize, Result<Vec<NativePlane>, String>>,
}

impl References {
    fn decode(fixtures: &[Fixture], cases: &[Case]) -> Self {
        let mut references = Self::default();
        for case in cases {
            let fixture = &fixtures[case.fixture];
            match (case.output, case.request) {
                (Output::Native, _) => {
                    references
                        .native
                        .entry(case.fixture)
                        .or_insert_with(|| reference_native(fixture));
                }
                (Output::Packed(fmt), request) => {
                    let scale = match request {
                        Request::Scaled(scale) | Request::RegionScaled(scale) => scale,
                        _ => Downscale::None,
                    };
                    references
                        .packed
                        .entry((case.fixture, fmt, scale))
                        .or_insert_with(|| reference_packed(fixture, fmt, scale));
                }
            }
        }
        references
    }

    /// The outcome `case` must have: the reference, cropped to the case's
    /// output rectangle.
    fn expected(&self, case: &Case, fixture: &Fixture) -> Outcome {
        let size = (fixture.width, fixture.height);
        let region = region(fixture);
        let outcome = match case.output {
            Output::Native => self.native[&case.fixture].as_ref().map(|planes| {
                if case.request == Request::Region {
                    native_hash(&crop_native(planes, region))
                } else {
                    native_hash(planes)
                }
            }),
            Output::Packed(fmt) => {
                let (scale, rect) = match case.request {
                    Request::Scaled(s) | Request::RegionScaled(s) => {
                        let image = output_rect(Request::Scaled(s), size, region);
                        (s, (image, output_rect(case.request, size, region)))
                    }
                    request => (
                        Downscale::None,
                        (Rect::full(size), output_rect(request, size, region)),
                    ),
                };
                self.packed[&(case.fixture, fmt, scale)]
                    .as_ref()
                    .map(|bytes| hash(&crop(bytes, rect.0.w, rect.1, fmt.bytes_per_pixel())))
            }
        };
        match outcome {
            Ok(hash) => Outcome::Bytes(hash),
            Err(class) => Outcome::Error(class.clone()),
        }
    }
}

fn reference_decoder(fixture: &Fixture) -> Result<J2kDecoder<'_>, String> {
    let mut decoder = J2kDecoder::new(&fixture.bytes).map_err(|error| error_class(&error))?;
    decoder.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
    decoder.set_full_decode_planes_budget(usize::MAX);
    Ok(decoder)
}

fn reference_packed(
    fixture: &Fixture,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<Vec<u8>, String> {
    let mut decoder = reference_decoder(fixture)?;
    let size = (fixture.width, fixture.height);
    let rect = output_rect(Request::Scaled(scale), size, Rect::full(size));
    let stride = rect.w as usize * fmt.bytes_per_pixel();
    let mut out = vec![0; stride * rect.h as usize];
    decoder
        .decode_scaled_into(&mut J2kScratchPool::new(), &mut out, stride, fmt, scale)
        .map_err(|error| error_class(&error))?;
    Ok(out)
}

fn reference_native(fixture: &Fixture) -> Result<Vec<NativePlane>, String> {
    let decoded = reference_decoder(fixture)?
        .decode_native_components()
        .map_err(|error| error_class(&error))?;
    Ok(native_planes(&decoded))
}

/// The rows of `rect` from rows of `width` pixels of `pixel` bytes.
fn crop(bytes: &[u8], width: u32, rect: Rect, pixel: usize) -> Vec<u8> {
    let stride = width as usize * pixel;
    let row = rect.w as usize * pixel;
    (rect.y..rect.y + rect.h)
        .flat_map(|y| {
            let start = y as usize * stride + rect.x as usize * pixel;
            &bytes[start..start + row]
        })
        .copied()
        .collect()
}

fn crop_native(planes: &[NativePlane], rect: Rect) -> Vec<NativePlane> {
    planes
        .iter()
        .map(|plane| NativePlane {
            dimensions: (rect.w, rect.h),
            data: crop(
                &plane.data,
                plane.dimensions.0,
                rect,
                usize::from(plane.bytes_per_sample),
            ),
            ..*plane
        })
        .collect()
}

struct Pools {
    pools: Vec<(usize, ThreadPool)>,
}

impl Pools {
    fn new() -> Self {
        let pools = [1, POOL_THREADS, WIDE_POOL_THREADS]
            .into_iter()
            .map(|threads| {
                let pool = ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(move |index| format!("guard-{threads}-{index}"))
                    .build()
                    .expect("guard thread pool");
                (threads, pool)
            })
            .collect();
        Self { pools }
    }

    fn get(&self, threads: usize) -> &ThreadPool {
        &self
            .pools
            .iter()
            .find(|(count, _)| *count == threads)
            .expect("guard pool of that size")
            .1
    }
}

fn run(case: &Case, fixtures: &[Fixture], pools: &Pools) -> Measured {
    let fixture = &fixtures[case.fixture];
    let decode = Decode {
        bytes: &fixture.bytes,
        size: (fixture.width, fixture.height),
        region: region(fixture),
        request: case.request,
        output: case.output,
        parallelism: if case.threads == 1 {
            CpuDecodeParallelism::Serial
        } else {
            CpuDecodeParallelism::Auto
        },
        budget: case.budget,
    };
    pools.get(case.threads).install(|| decode.run())
}

/// `case`'s outcome and highest peak over its repeats, with the spread of
/// the peaks; a repeat that writes other bytes is a mismatch.
fn measure(
    case: &Case,
    fixtures: &[Fixture],
    pools: &Pools,
    id: &str,
    mismatches: &mut Vec<String>,
) -> (Measured, usize) {
    let mut measured = run(case, fixtures, pools);
    let mut lowest = measured.peak;
    let mut repeats = match (case.threads, fixtures[case.fixture].extent) {
        (1, _) => 1,
        (_, Extent::Large) if POOL_REPEATS > 1 => NOISY_REPEATS,
        _ => POOL_REPEATS,
    };
    let mut runs = 1;
    while runs < repeats {
        runs += 1;
        let again = run(case, fixtures, pools);
        if again.outcome != measured.outcome {
            mismatches.push(format!(
                "{id}: wrote {} and then {}",
                measured.outcome, again.outcome
            ));
        }
        lowest = lowest.min(again.peak);
        measured.peak = measured.peak.max(again.peak);
        if runs == POOL_REPEATS && measured.peak - lowest > SLACK_BYTES {
            repeats = repeats.max(NOISY_REPEATS);
        }
    }
    let spread = measured.peak - lowest;
    (measured, spread)
}

#[test]
fn every_route_writes_the_planes_bytes_and_records_its_peak_heap() {
    let started = Instant::now();
    // Decodes run in the pools built here. Code that asks for the global
    // pool's threads from outside a pool builds it, so build it now, before
    // any measurement, at one thread.
    ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .expect("the global pool is built once, by this test");
    let pools = Pools::new();
    let fixtures = pools.get(POOL_THREADS).install(fixtures::fixtures);
    let cases = cases(&fixtures);
    let references = References::decode(&fixtures, &cases);
    // Warm each pool, so first-use allocations of its threads are not in a
    // measured peak.
    for (threads, _) in &pools.pools {
        let warm = Case {
            threads: *threads,
            ..cases[0]
        };
        let _ = run(&warm, &fixtures, &pools);
    }

    let mut rows = Vec::with_capacity(cases.len());
    let mut mismatches = Vec::new();
    let mut findings = Findings::default();
    for case in &cases {
        let id = case.id(&fixtures);
        let (measured, spread) = measure(case, &fixtures, &pools, &id, &mut mismatches);
        let expected = references.expected(case, &fixtures[case.fixture]);
        match (&measured.outcome, &expected) {
            (Outcome::Bytes(got), Outcome::Bytes(want)) if got != want => {
                mismatches.push(format!("{id}: wrote {got:016x}, planes {want:016x}"));
            }
            (got, want) if got != want => findings
                .outcome_changes
                .push(format!("{id}: {got}, planes {want}")),
            _ => {}
        }
        rows.push(Row {
            id,
            measured,
            spread,
            expected,
        });
    }

    let table = table(&rows);
    let baseline = compare_with_baseline(&rows, &mut findings);
    let report = Report {
        rows: &rows,
        fixtures: &fixtures,
        cases: &cases,
        mismatches: &mismatches,
        findings: &findings,
    };
    let summary = report.summary(&baseline, started.elapsed().as_secs_f64());
    write_report(&table, &summary);

    assert!(
        mismatches.is_empty(),
        "decodes wrote other bytes than the planes route:\n{}",
        mismatches.join("\n")
    );
    assert!(
        findings.is_empty(),
        "decode guard findings:\n{}",
        findings.failures().join("\n")
    );
}
