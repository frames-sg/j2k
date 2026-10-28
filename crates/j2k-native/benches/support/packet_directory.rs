// SPDX-License-Identifier: MIT OR Apache-2.0

//! Repeated regions through the existing component-output API.

use criterion::{measurement::WallTime, BatchSize, BenchmarkGroup, Criterion, Throughput};
use j2k_native::{
    encode, CpuDecodeParallelism, DecodeSettings, DecodedComponents, DecoderContext, EncodeOptions,
    EncodeProgressionOrder, Image,
};
#[cfg(feature = "parallel")]
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

type Region = (u32, u32, u32, u32);

fn source_bytes() -> Vec<u8> {
    if let Some(path) = std::env::var_os("J2K_PACKET_DIRECTORY_SOURCE") {
        eprintln!("packet directory: source {}", path.to_string_lossy());
        return std::fs::read(path).expect("read region source");
    }
    eprintln!("packet directory: generated 1536x1024 approximation");
    let (width, height) = (1536, 1024);
    encode(
        &super::patterned_rgb8(width, height),
        width,
        height,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            progression_order: EncodeProgressionOrder::Rpcl,
            num_layers: 7,
            precinct_exponents: vec![(7, 7), (7, 7), (7, 7), (7, 7), (8, 8), (8, 8)],
            ..EncodeOptions::default()
        },
    )
    .unwrap()
}

fn context<'a>(mode: CpuDecodeParallelism) -> DecoderContext<'a> {
    let mut ctx = DecoderContext::default();
    ctx.set_cpu_decode_parallelism(mode);
    ctx
}

// Each control Image is constructed outside the timer and used once, so it
// follows the existing packet walk without charging header setup to decoding.
fn control<'a, 'ctx>(
    image: &Image<'a>,
    roi: Region,
    ctx: &'ctx mut DecoderContext<'a>,
) -> DecodedComponents<'ctx> {
    image
        .decode_region_components_for_integer_output_with_context(roi, ctx)
        .unwrap()
}

/// One mode and region: the image every automatic decode reuses, and the
/// pool the decodes run on.
struct Case<'s, 'a> {
    name: String,
    bytes: &'a [u8],
    image: &'s Image<'a>,
    roi: Region,
    mode: CpuDecodeParallelism,
    #[cfg(feature = "parallel")]
    pool: &'s ThreadPool,
}

impl<'a> Case<'_, 'a> {
    fn install(&self, operation: &mut (dyn FnMut() + Send)) {
        #[cfg(feature = "parallel")]
        self.pool.install(operation);
        #[cfg(not(feature = "parallel"))]
        operation();
    }

    fn control_image(&self) -> Image<'a> {
        Image::new(self.bytes, &DecodeSettings::default()).unwrap()
    }

    fn automatic<'ctx>(&self, ctx: &'ctx mut DecoderContext<'a>) -> DecodedComponents<'ctx> {
        self.image
            .decode_region_components_for_integer_output_with_context(self.roi, ctx)
            .unwrap()
    }
}

pub(super) fn bench(criterion: &mut Criterion) {
    let bytes = source_bytes();
    let image = Image::new(&bytes, &DecodeSettings::default()).unwrap();
    let original_metadata = image.retained_allocation_bytes().unwrap();
    let (width, height) = (image.width(), image.height());
    let side = std::env::var("J2K_PACKET_DIRECTORY_REGION_SIDE")
        .map_or(256, |side| side.parse::<u32>().unwrap());
    let threads = std::env::var("J2K_PACKET_DIRECTORY_THREADS")
        .map_or(4, |threads| threads.parse::<usize>().unwrap());
    assert!(side > 0 && side <= width && side <= height && threads > 0);
    let label = std::env::var_os("J2K_PACKET_DIRECTORY_SOURCE").map_or_else(
        || "generated".to_owned(),
        |path| {
            std::path::Path::new(&path)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        },
    );
    let mut group = criterion.benchmark_group(format!(
        "packet_directory/{label}_{width}x{height}_roi{side}_pool{threads}"
    ));
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("image_setup", |bencher| {
        bencher.iter(|| black_box(Image::new(&bytes, &DecodeSettings::default()).unwrap()));
    });
    let regions = [
        (
            "middle",
            ((width - side) / 2, (height - side) / 2, side, side),
        ),
        ("edge", (width - side, height - side, side, side)),
    ];
    for (mode_name, mode) in [
        ("serial", CpuDecodeParallelism::Serial),
        ("auto", CpuDecodeParallelism::Auto),
    ] {
        #[cfg(feature = "parallel")]
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for (region_name, roi) in regions {
            let case = Case {
                name: format!("{mode_name}/{region_name}"),
                bytes: &bytes,
                image: &image,
                roi,
                mode,
                #[cfg(feature = "parallel")]
                pool: &pool,
            };
            if std::env::var("J2K_PACKET_DIRECTORY_CASE")
                .is_ok_and(|selected| selected != case.name)
            {
                continue;
            }
            let mut control_ctx = context(mode);
            let mut automatic_ctx = context(mode);
            check_parity(&case, &mut control_ctx, &mut automatic_ctx);
            report_latency(&case, &mut control_ctx, &mut automatic_ctx);
            eprintln!(
                "region {} cache_bytes={} retained_context_bytes={}",
                case.name,
                image.retained_allocation_bytes().unwrap() - original_metadata,
                automatic_ctx.workspace_stats().retained_component_bytes()
                    + automatic_ctx.workspace_stats().retained_scratch_bytes()
            );
            #[cfg(feature = "parallel")]
            report_outer8(&case);
            bench_routes(&mut group, &case);
        }
    }
    group.finish();
}

fn check_parity<'a>(
    case: &Case<'_, 'a>,
    control_ctx: &mut DecoderContext<'a>,
    automatic_ctx: &mut DecoderContext<'a>,
) {
    for call in 0..2 {
        let control_image = case.control_image();
        let start = Instant::now();
        case.install(&mut || {
            let expected = control(&control_image, case.roi, control_ctx);
            let actual = case.automatic(automatic_ctx);
            assert_eq!(actual.dimensions(), expected.dimensions());
            assert_eq!(actual.planes().len(), expected.planes().len());
            for (actual_plane, expected_plane) in actual.planes().iter().zip(expected.planes()) {
                assert!(actual_plane
                    .samples()
                    .iter()
                    .map(|sample| sample.to_bits())
                    .eq(expected_plane
                        .samples()
                        .iter()
                        .map(|sample| sample.to_bits())));
            }
        });
        eprintln!(
            "region {} parity call={call} pair_ms={:.3}",
            case.name,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn report_latency<'a>(
    case: &Case<'_, 'a>,
    control_ctx: &mut DecoderContext<'a>,
    automatic_ctx: &mut DecoderContext<'a>,
) {
    let mut control_first = Vec::new();
    let mut control_second = Vec::new();
    let mut automatic = Vec::new();
    for run in 0..33 {
        for (route, samples) in [
            (0, &mut control_first),
            (0, &mut control_second),
            (1, &mut automatic),
        ] {
            let control_image = (route == 0).then(|| case.control_image());
            let start = Instant::now();
            case.install(&mut || {
                if route == 0 {
                    black_box(control(
                        control_image.as_ref().unwrap(),
                        case.roi,
                        control_ctx,
                    ));
                } else {
                    black_box(case.automatic(automatic_ctx));
                }
            });
            if run > 0 {
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    for (name, samples) in [
        ("A1", &mut control_first),
        ("A2", &mut control_second),
        ("B", &mut automatic),
    ] {
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "region {} {name} p50_ms={:.3} p95_ms={:.3}",
            case.name, samples[16], samples[30]
        );
    }
}

#[cfg(feature = "parallel")]
fn report_outer8(case: &Case<'_, '_>) {
    let mut control_contexts: Vec<_> = (0..8).map(|_| context(case.mode)).collect();
    let mut automatic_contexts: Vec<_> = (0..8).map(|_| context(case.mode)).collect();
    let mut batches = [Vec::new(), Vec::new(), Vec::new()];
    for round in 0..17 {
        let order = if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
        for route in order {
            let control_images: Vec<_> = if route == 2 {
                Vec::new()
            } else {
                (0..8).map(|_| case.control_image()).collect()
            };
            let start = Instant::now();
            case.pool.install(|| {
                if route == 2 {
                    automatic_contexts.par_iter_mut().for_each(|ctx| {
                        black_box(case.automatic(ctx));
                    });
                } else {
                    control_contexts
                        .par_iter_mut()
                        .zip(&control_images)
                        .for_each(|(ctx, control_image)| {
                            black_box(control(control_image, case.roi, ctx));
                        });
                }
            });
            if round > 0 {
                batches[route].push(start.elapsed().as_secs_f64());
            }
        }
    }
    for (name, samples) in ["A1", "A2", "B"].into_iter().zip(&mut batches) {
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "outer8 {} {name} p50_ms={:.3} requests_per_s={:.3}",
            case.name,
            samples[8] * 1000.0,
            8.0 / samples[8]
        );
    }
}

fn bench_routes(group: &mut BenchmarkGroup<'_, WallTime>, case: &Case<'_, '_>) {
    for automatic in [false, true] {
        let route = if automatic { "automatic" } else { "control" };
        let mut reused = context(case.mode);
        group.throughput(Throughput::Elements(1));
        for fresh in [false, true] {
            let reuse = if fresh { "fresh" } else { "reuse" };
            group.bench_function(format!("{}/{route}_{reuse}", case.name), |bencher| {
                bencher.iter_batched_ref(
                    || (!automatic).then(|| case.control_image()),
                    |control_image| {
                        case.install(&mut || {
                            let mut new_ctx = context(case.mode);
                            let ctx = if fresh { &mut new_ctx } else { &mut reused };
                            if automatic {
                                black_box(case.automatic(ctx));
                            } else {
                                black_box(control(control_image.as_ref().unwrap(), case.roi, ctx));
                            }
                        });
                    },
                    BatchSize::PerIteration,
                );
            });
        }
        #[cfg(feature = "parallel")]
        {
            let mut contexts: Vec<_> = (0..8).map(|_| context(case.mode)).collect();
            group.throughput(Throughput::Elements(8));
            group.bench_function(format!("{}/{route}_outer8", case.name), |bencher| {
                bencher.iter_batched_ref(
                    || {
                        if automatic {
                            Vec::new()
                        } else {
                            (0..8).map(|_| case.control_image()).collect::<Vec<_>>()
                        }
                    },
                    |control_images| {
                        case.pool.install(|| {
                            contexts
                                .par_iter_mut()
                                .enumerate()
                                .for_each(|(index, ctx)| {
                                    if automatic {
                                        black_box(case.automatic(ctx));
                                    } else {
                                        black_box(control(&control_images[index], case.roi, ctx));
                                    }
                                });
                        });
                    },
                    BatchSize::PerIteration,
                );
            });
        }
    }
}
