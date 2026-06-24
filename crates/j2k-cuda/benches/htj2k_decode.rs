// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use j2k_core::{
    BackendKind, BackendRequest, DecoderContext, DeviceSubmission, DeviceSurface, Downscale,
    ImageDecode, ImageDecodeSubmit, PixelFormat, Rect, TileBatchDecodeManyDevice,
};
use j2k_cuda::{Codec, CudaSession, J2kDecoder, SurfaceResidency};
use j2k_native::{encode_htj2k, EncodeOptions};

const TILE_DIM: u32 = 512;
const BATCH_SIZES: &[usize] = &[8, 16, 32, 64];

struct DecodeBenchCase {
    id: String,
    fixture: Vec<u8>,
    fmt: PixelFormat,
    dimensions: (u32, u32),
    input_source: String,
    cuda_available: bool,
}

struct DecodeBenchCorpus {
    cases: Vec<DecodeBenchCase>,
    external_stats: ExternalDecodeStats,
}

#[derive(Default)]
struct ExternalDecodeStats {
    fixtures_seen: usize,
    skipped_non_htj2k: usize,
    skipped_unsupported_shape: usize,
    skipped_format_disabled: usize,
}

struct ExternalDecodeCases {
    cases: Vec<DecodeBenchCase>,
    stats: ExternalDecodeStats,
}

fn bench_htj2k_decode(c: &mut Criterion) {
    let corpus = all_decode_cases();
    emit_input_metadata(&corpus);
    let scale = Downscale::Half;

    bench_full_tile(c, &corpus.cases);
    bench_roi(c, &corpus.cases);
    bench_scaled(c, &corpus.cases, scale);
    bench_roi_scaled(c, &corpus.cases, scale);
    bench_tile_batch(c, &corpus.cases);
    bench_mixed_external_tile_batch(c, &corpus.cases);
}

fn bench_full_tile(c: &mut Criterion, cases: &[DecodeBenchCase]) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_full_tile_decode");
    for case in cases {
        let cpu_id = cpu_benchmark_id(case);
        group.bench_with_input(
            BenchmarkId::new(cpu_id, dimensions_label(case.dimensions)),
            case,
            |b, case| {
                b.iter(|| {
                    let mut decoder =
                        J2kDecoder::new(std::hint::black_box(case.fixture.as_slice()))
                            .expect("decoder");
                    let stride = case.dimensions.0 as usize * case.fmt.bytes_per_pixel();
                    let mut out = vec![0u8; stride * case.dimensions.1 as usize];
                    decoder
                        .decode_into(&mut out, stride, case.fmt)
                        .expect("CPU HTJ2K decode");
                    std::hint::black_box(out)
                });
            },
        );
        if case.cuda_available {
            let cuda_id = cuda_benchmark_id(case);
            group.bench_with_input(
                BenchmarkId::new(cuda_id, dimensions_label(case.dimensions)),
                case,
                |b, case| {
                    let mut session = CudaSession::default();
                    b.iter(|| {
                        let mut decoder =
                            J2kDecoder::new(std::hint::black_box(case.fixture.as_slice()))
                                .expect("decoder");
                        let surface = decoder
                            .submit_to_device(&mut session, case.fmt, BackendRequest::Cuda)
                            .expect("strict CUDA HTJ2K decode submission")
                            .wait()
                            .expect("strict CUDA HTJ2K decode");
                        assert_cuda_resident_decode(&surface);
                        std::hint::black_box(surface)
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_roi(c: &mut Criterion, cases: &[DecodeBenchCase]) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_roi_decode");
    for case in cases {
        let roi = roi_for_case(case);
        let cpu_id = cpu_benchmark_id(case);
        group.bench_with_input(BenchmarkId::new(cpu_id, roi.w), case, |b, case| {
            b.iter(|| {
                let mut decoder = J2kDecoder::new(std::hint::black_box(case.fixture.as_slice()))
                    .expect("decoder");
                let mut pool = j2k_cuda::J2kScratchPool::new();
                let stride = roi.w as usize * case.fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * roi.h as usize];
                decoder
                    .decode_region_into(&mut pool, &mut out, stride, case.fmt, roi)
                    .expect("CPU HTJ2K ROI decode");
                std::hint::black_box(out)
            });
        });
        if case.cuda_available {
            let cuda_id = cuda_benchmark_id(case);
            group.bench_with_input(BenchmarkId::new(cuda_id, roi.w), case, |b, case| {
                let mut session = CudaSession::default();
                b.iter(|| {
                    let mut decoder =
                        J2kDecoder::new(std::hint::black_box(case.fixture.as_slice()))
                            .expect("decoder");
                    let surface = decoder
                        .submit_region_to_device(&mut session, case.fmt, roi, BackendRequest::Cuda)
                        .expect("strict CUDA HTJ2K ROI decode submission")
                        .wait()
                        .expect("strict CUDA HTJ2K ROI decode");
                    assert_cuda_resident_decode(&surface);
                    std::hint::black_box(surface)
                });
            });
        }
    }
    group.finish();
}

fn bench_scaled(c: &mut Criterion, cases: &[DecodeBenchCase], scale: Downscale) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_scaled_decode");
    for case in cases {
        let scaled = Rect::full(case.dimensions).scaled_covering(scale);
        let cpu_id = cpu_benchmark_id(case);
        group.bench_with_input(BenchmarkId::new(cpu_id, scaled.w), case, |b, case| {
            b.iter(|| {
                let mut decoder = J2kDecoder::new(std::hint::black_box(case.fixture.as_slice()))
                    .expect("decoder");
                let mut pool = j2k_cuda::J2kScratchPool::new();
                let stride = scaled.w as usize * case.fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * scaled.h as usize];
                decoder
                    .decode_scaled_into(&mut pool, &mut out, stride, case.fmt, scale)
                    .expect("CPU HTJ2K scaled decode");
                std::hint::black_box(out)
            });
        });
        if case.cuda_available {
            let cuda_id = cuda_benchmark_id(case);
            group.bench_with_input(BenchmarkId::new(cuda_id, scaled.w), case, |b, case| {
                let mut session = CudaSession::default();
                b.iter(|| {
                    let mut decoder =
                        J2kDecoder::new(std::hint::black_box(case.fixture.as_slice()))
                            .expect("decoder");
                    let surface = decoder
                        .submit_scaled_to_device(
                            &mut session,
                            case.fmt,
                            scale,
                            BackendRequest::Cuda,
                        )
                        .expect("strict CUDA HTJ2K scaled decode submission")
                        .wait()
                        .expect("strict CUDA HTJ2K scaled decode");
                    assert_cuda_resident_decode(&surface);
                    std::hint::black_box(surface)
                });
            });
        }
    }
    group.finish();
}

fn bench_roi_scaled(c: &mut Criterion, cases: &[DecodeBenchCase], scale: Downscale) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_roi_scaled_decode");
    for case in cases {
        let roi = roi_for_case(case);
        let scaled = roi.scaled_covering(scale);
        let cpu_id = cpu_benchmark_id(case);
        group.bench_with_input(BenchmarkId::new(cpu_id, scaled.w), case, |b, case| {
            b.iter(|| {
                let mut decoder = J2kDecoder::new(std::hint::black_box(case.fixture.as_slice()))
                    .expect("decoder");
                let mut pool = j2k_cuda::J2kScratchPool::new();
                let stride = scaled.w as usize * case.fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * scaled.h as usize];
                decoder
                    .decode_region_scaled_into(&mut pool, &mut out, stride, case.fmt, roi, scale)
                    .expect("CPU HTJ2K ROI+scaled decode");
                std::hint::black_box(out)
            });
        });
        if case.cuda_available {
            let cuda_id = cuda_benchmark_id(case);
            group.bench_with_input(BenchmarkId::new(cuda_id, scaled.w), case, |b, case| {
                let mut session = CudaSession::default();
                b.iter(|| {
                    let mut decoder =
                        J2kDecoder::new(std::hint::black_box(case.fixture.as_slice()))
                            .expect("decoder");
                    let surface = decoder
                        .submit_region_scaled_to_device(
                            &mut session,
                            case.fmt,
                            roi,
                            scale,
                            BackendRequest::Cuda,
                        )
                        .expect("strict CUDA HTJ2K ROI+scaled decode submission")
                        .wait()
                        .expect("strict CUDA HTJ2K ROI+scaled decode");
                    assert_cuda_resident_decode(&surface);
                    std::hint::black_box(surface)
                });
            });
        }
    }
    group.finish();
}

fn bench_tile_batch(c: &mut Criterion, cases: &[DecodeBenchCase]) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_tile_batch_decode");
    let batch_sizes = decode_batch_sizes();
    for case in cases {
        for &batch_size in &batch_sizes {
            let fixtures = vec![case.fixture.clone(); batch_size];
            let inputs = fixtures.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let fmt = case.fmt;
            let cpu_id = cpu_benchmark_id(case);
            group.bench_with_input(
                BenchmarkId::new(cpu_id, batch_size),
                &inputs,
                |b, inputs| {
                    b.iter(|| {
                        let mut ctx = DecoderContext::<j2k_cuda::J2kContext>::new();
                        let mut pool = j2k_cuda::J2kScratchPool::new();
                        let surfaces = Codec::decode_tiles_to_device(
                            &mut ctx,
                            &mut pool,
                            std::hint::black_box(inputs),
                            fmt,
                            BackendRequest::Cpu,
                        )
                        .expect("CPU HTJ2K batch decode");
                        std::hint::black_box(surfaces)
                    });
                },
            );
            if case.cuda_available && cuda_batch_decode_supported(fmt) {
                let cuda_id = cuda_benchmark_id(case);
                group.bench_with_input(
                    BenchmarkId::new(cuda_id, batch_size),
                    &inputs,
                    |b, inputs| {
                        let mut session = CudaSession::default();
                        b.iter(|| {
                            let surfaces = J2kDecoder::decode_batch_to_device_with_session(
                                std::hint::black_box(inputs),
                                fmt,
                                &mut session,
                            )
                            .expect("strict CUDA HTJ2K real batch decode");
                            assert_eq!(surfaces.len(), inputs.len());
                            assert_cuda_resident_batch_decode(&surfaces);
                            std::hint::black_box(surfaces)
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

fn bench_mixed_external_tile_batch(c: &mut Criterion, cases: &[DecodeBenchCase]) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_external_mixed_tile_batch_decode");
    let batch_sizes = decode_batch_sizes();
    for fmt in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8] {
        let external_cases = cases
            .iter()
            .filter(|case| case.input_source.starts_with("external:") && case.fmt == fmt)
            .collect::<Vec<_>>();
        if external_cases.len() < 2 {
            continue;
        }
        for &batch_size in &batch_sizes {
            let inputs = (0..batch_size)
                .map(|index| {
                    external_cases[index % external_cases.len()]
                        .fixture
                        .as_slice()
                })
                .collect::<Vec<_>>();
            group.bench_with_input(
                BenchmarkId::new(
                    format!("cpu_external_mixed_{}", pixel_format_label(fmt)),
                    batch_size,
                ),
                &inputs,
                |b, inputs| {
                    b.iter(|| {
                        let mut ctx = DecoderContext::<j2k_cuda::J2kContext>::new();
                        let mut pool = j2k_cuda::J2kScratchPool::new();
                        let surfaces = Codec::decode_tiles_to_device(
                            &mut ctx,
                            &mut pool,
                            std::hint::black_box(inputs),
                            fmt,
                            BackendRequest::Cpu,
                        )
                        .expect("CPU HTJ2K mixed external batch decode");
                        std::hint::black_box(surfaces)
                    });
                },
            );
            if cuda_batch_decode_supported(fmt)
                && external_cases.iter().all(|case| case.cuda_available)
            {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("cuda_external_mixed_{}", pixel_format_label(fmt)),
                        batch_size,
                    ),
                    &inputs,
                    |b, inputs| {
                        let mut session = CudaSession::default();
                        b.iter(|| {
                            let surfaces = J2kDecoder::decode_batch_to_device_with_session(
                                std::hint::black_box(inputs),
                                fmt,
                                &mut session,
                            )
                            .expect("strict CUDA HTJ2K mixed external batch decode");
                            assert_eq!(surfaces.len(), inputs.len());
                            assert_cuda_resident_batch_decode(&surfaces);
                            std::hint::black_box(surfaces)
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

fn all_decode_cases() -> DecodeBenchCorpus {
    let enabled_cases = enabled_decode_cases();
    let mut cases = Vec::new();

    if include_generated_decode_cases() {
        cases.extend(generated_decode_cases(&enabled_cases));
    }
    let external = external_decode_cases(&enabled_cases);
    cases.extend(external.cases);
    assert!(
        !cases.is_empty(),
        "no CUDA HTJ2K decode bench cases available; enable generated cases or set J2K_CUDA_DECODE_INPUT_DIRS"
    );
    DecodeBenchCorpus {
        cases,
        external_stats: external.stats,
    }
}

fn generated_decode_cases(enabled_cases: &[&str]) -> Vec<DecodeBenchCase> {
    let mut cases = Vec::new();

    if enabled_cases.contains(&"gray8") {
        let gray_fixture = htj2k_gray8_fixture(TILE_DIM, TILE_DIM);
        cases.push(decode_case(
            "gray8",
            "j2k-generated-cuda-htj2k",
            gray_fixture,
            PixelFormat::Gray8,
            (TILE_DIM, TILE_DIM),
        ));
    }
    if enabled_cases
        .iter()
        .any(|id| matches!(*id, "rgb8" | "rgba8"))
    {
        let rgb_fixture = htj2k_rgb8_fixture(TILE_DIM, TILE_DIM);
        if enabled_cases.contains(&"rgb8") {
            cases.push(decode_case(
                "rgb8",
                "j2k-generated-cuda-htj2k",
                rgb_fixture.clone(),
                PixelFormat::Rgb8,
                (TILE_DIM, TILE_DIM),
            ));
        }
        if enabled_cases.contains(&"rgba8") {
            cases.push(decode_case(
                "rgba8",
                "j2k-generated-cuda-htj2k",
                rgb_fixture,
                PixelFormat::Rgba8,
                (TILE_DIM, TILE_DIM),
            ));
        }
    }
    cases
}

fn decode_case(
    id: impl Into<String>,
    input_source: impl Into<String>,
    fixture: Vec<u8>,
    fmt: PixelFormat,
    dimensions: (u32, u32),
) -> DecodeBenchCase {
    let id = id.into();
    let cuda_available = cuda_decode_available(&id, &fixture, fmt);
    DecodeBenchCase {
        id,
        fixture,
        fmt,
        dimensions,
        input_source: input_source.into(),
        cuda_available,
    }
}

fn include_generated_decode_cases() -> bool {
    !env_falsey("J2K_CUDA_DECODE_INCLUDE_GENERATED")
}

fn env_falsey(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off"))
}

fn external_decode_cases(enabled_cases: &[&str]) -> ExternalDecodeCases {
    let dirs = external_input_dirs();
    if dirs.is_empty() {
        return ExternalDecodeCases {
            cases: Vec::new(),
            stats: ExternalDecodeStats::default(),
        };
    }
    let manifest = cuda_decode_manifest().unwrap_or_else(|error| panic!("{error}"));
    let mut cases = Vec::new();
    let mut stats = ExternalDecodeStats::default();
    for dir in dirs {
        let mut paths = Vec::new();
        collect_j2k_paths(&dir, &mut paths).unwrap_or_else(|error| panic!("{error}"));
        paths.sort();
        assert!(
            !paths.is_empty(),
            "J2K_CUDA_DECODE_INPUT_DIRS entry {} contains no .j2k/.j2c/.jp2/.jph/.jhc fixtures",
            dir.display()
        );
        for path in paths {
            stats.fixtures_seen += 1;
            let bytes = fs::read(&path).unwrap_or_else(|error| {
                panic!(
                    "read external CUDA decode fixture {}: {error}",
                    path.display()
                )
            });
            if codec_from_bytes(&bytes) != Some("htj2k") {
                eprintln!(
                    "skipping external CUDA decode fixture {}: CUDA adoption decode bench only accepts HTJ2K fixtures",
                    path.display()
                );
                stats.skipped_non_htj2k += 1;
                continue;
            }
            validate_manifest_entry(&path, &bytes, manifest.as_ref())
                .unwrap_or_else(|error| panic!("{error}"));
            let info = j2k::J2kDecoder::inspect(&bytes).unwrap_or_else(|error| {
                panic!(
                    "inspect external CUDA decode fixture {}: {error}",
                    path.display()
                )
            });
            let stem = sanitized_stem(&path);
            let input_source = format!("external:{}", path.display());
            match (info.components, info.bit_depth) {
                (1, 8) if enabled_cases.contains(&"gray8") => {
                    cases.push(decode_case(
                        format!("external_{stem}_gray8"),
                        input_source,
                        bytes,
                        PixelFormat::Gray8,
                        info.dimensions,
                    ));
                }
                (3, 8) => {
                    let mut pushed = false;
                    if enabled_cases.contains(&"rgb8") {
                        cases.push(decode_case(
                            format!("external_{stem}_rgb8"),
                            input_source.clone(),
                            bytes.clone(),
                            PixelFormat::Rgb8,
                            info.dimensions,
                        ));
                        pushed = true;
                    }
                    if enabled_cases.contains(&"rgba8") {
                        cases.push(decode_case(
                            format!("external_{stem}_rgba8"),
                            input_source,
                            bytes,
                            PixelFormat::Rgba8,
                            info.dimensions,
                        ));
                        pushed = true;
                    }
                    if !pushed {
                        stats.skipped_format_disabled += 1;
                        eprintln!(
                            "skipping external CUDA decode fixture {}: decoded shape components={} bit_depth={} is disabled by J2K_CUDA_DECODE_FORMATS",
                            path.display(),
                            info.components,
                            info.bit_depth
                        );
                    }
                }
                (1, 8) => {
                    stats.skipped_format_disabled += 1;
                    eprintln!(
                        "skipping external CUDA decode fixture {}: decoded shape components={} bit_depth={} is disabled by J2K_CUDA_DECODE_FORMATS",
                        path.display(),
                        info.components,
                        info.bit_depth
                    );
                }
                _ => {
                    stats.skipped_unsupported_shape += 1;
                    eprintln!(
                        "skipping external CUDA decode fixture {}: unsupported benchmark shape components={} bit_depth={}",
                        path.display(),
                        info.components,
                        info.bit_depth
                    );
                }
            }
        }
    }
    ExternalDecodeCases { cases, stats }
}

fn emit_input_metadata(corpus: &DecodeBenchCorpus) {
    let external_count = corpus
        .cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .count();
    let batch_sizes = decode_batch_sizes()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "j2k_cuda_decode_generated_included\t{}",
        include_generated_decode_cases()
    );
    println!("j2k_cuda_decode_batch_sizes\t{batch_sizes}");
    println!(
        "j2k_cuda_decode_io_policy\thost-memory-fixture-bytes-preloaded-no-filesystem-io-in-timed-loop;cuda-rows-return-device-resident-surfaces"
    );
    println!(
        "j2k_cuda_decode_input_dirs\t{}",
        std::env::var("J2K_CUDA_DECODE_INPUT_DIRS").unwrap_or_else(|_| "not set".to_string())
    );
    println!(
        "j2k_cuda_decode_manifest\t{}",
        std::env::var("J2K_CUDA_DECODE_MANIFEST").unwrap_or_else(|_| "not set".to_string())
    );
    println!("j2k_cuda_decode_case_count\t{}", corpus.cases.len());
    println!("j2k_cuda_decode_external_case_count\t{external_count}");
    println!(
        "j2k_cuda_decode_external_fixture_count\t{}",
        corpus.external_stats.fixtures_seen
    );
    println!(
        "j2k_cuda_decode_external_skipped_non_htj2k_count\t{}",
        corpus.external_stats.skipped_non_htj2k
    );
    println!(
        "j2k_cuda_decode_external_skipped_unsupported_shape_count\t{}",
        corpus.external_stats.skipped_unsupported_shape
    );
    println!(
        "j2k_cuda_decode_external_skipped_format_disabled_count\t{}",
        corpus.external_stats.skipped_format_disabled
    );
}

fn external_input_dirs() -> Vec<PathBuf> {
    std::env::var_os("J2K_CUDA_DECODE_INPUT_DIRS")
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default()
}

fn collect_j2k_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!(
            "J2K_CUDA_DECODE_INPUT_DIRS entry is not a directory: {}",
            dir.display()
        ));
    }
    for entry in fs::read_dir(dir).map_err(|error| format!("read {}: {error}", dir.display()))? {
        let path = entry
            .map_err(|error| format!("read {} entry: {error}", dir.display()))?
            .path();
        if path.is_dir() {
            collect_j2k_paths(&path, paths)?;
        } else if is_j2k_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn is_j2k_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "j2k" | "j2c" | "jp2" | "jph" | "jhc"
            )
        })
}

#[derive(Clone)]
struct CudaDecodeManifest {
    entries: HashMap<PathBuf, CudaDecodeManifestEntry>,
}

#[derive(Clone)]
struct CudaDecodeManifestEntry {
    input_fnv1a64: Option<String>,
    codec: Option<String>,
    container: Option<String>,
}

fn cuda_decode_manifest() -> Result<Option<CudaDecodeManifest>, String> {
    let Some(path) = std::env::var_os("J2K_CUDA_DECODE_MANIFEST").map(PathBuf::from) else {
        return Ok(None);
    };
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("read J2K_CUDA_DECODE_MANIFEST {}: {error}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| format!("CUDA decode manifest {} is empty", path.display()))?;
    let headers = header.split('\t').collect::<Vec<_>>();
    let path_index = manifest_column(&headers, "path")?;
    let hash_index = optional_manifest_column(&headers, "input_fnv1a64");
    let codec_index = optional_manifest_column(&headers, "codec");
    let container_index = optional_manifest_column(&headers, "container");
    let mut entries = HashMap::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim_start().starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        let row_number = line_index + 2;
        let raw_path = manifest_field(&fields, path_index, "path", row_number)?;
        let resolved_path = if Path::new(raw_path).is_absolute() {
            PathBuf::from(raw_path)
        } else {
            base.join(raw_path)
        };
        let canonical_path = resolved_path.canonicalize().map_err(|error| {
            format!(
                "CUDA decode manifest {} row {row_number} path {} cannot be canonicalized: {error}",
                path.display(),
                resolved_path.display()
            )
        })?;
        let entry = CudaDecodeManifestEntry {
            input_fnv1a64: manifest_optional_value(
                &fields,
                hash_index,
                "input_fnv1a64",
                row_number,
            )?,
            codec: manifest_optional_value(&fields, codec_index, "codec", row_number)?,
            container: manifest_optional_value(&fields, container_index, "container", row_number)?,
        };
        if entries.insert(canonical_path, entry).is_some() {
            return Err(format!(
                "CUDA decode manifest {} row {row_number} duplicates path {raw_path}",
                path.display()
            ));
        }
    }
    Ok(Some(CudaDecodeManifest { entries }))
}

fn validate_manifest_entry(
    path: &Path,
    bytes: &[u8],
    manifest: Option<&CudaDecodeManifest>,
) -> Result<(), String> {
    let Some(manifest) = manifest else {
        return Ok(());
    };
    let canonical_path = path.canonicalize().map_err(|error| {
        format!(
            "canonicalize external CUDA decode fixture {}: {error}",
            path.display()
        )
    })?;
    let Some(entry) = manifest.entries.get(&canonical_path) else {
        return Err(format!(
            "external CUDA decode fixture {} is not covered by J2K_CUDA_DECODE_MANIFEST",
            path.display()
        ));
    };
    let expected_hash = entry.input_fnv1a64.as_deref().ok_or_else(|| {
        format!(
            "external CUDA decode fixture {} manifest row is missing input_fnv1a64",
            path.display()
        )
    })?;
    let actual_hash = fnv1a64_hex(bytes);
    if actual_hash != expected_hash {
        return Err(format!(
            "external CUDA decode fixture {} hash mismatch: manifest {expected_hash} != actual {actual_hash}",
            path.display()
        ));
    }
    let expected_codec = entry.codec.as_deref().ok_or_else(|| {
        format!(
            "external CUDA decode fixture {} manifest row is missing codec",
            path.display()
        )
    })?;
    let actual_codec = codec_from_bytes(bytes).unwrap_or("unknown");
    if actual_codec != expected_codec {
        return Err(format!(
            "external CUDA decode fixture {} codec mismatch: manifest {expected_codec} != detected {actual_codec}",
            path.display()
        ));
    }
    let expected_container = entry.container.as_deref().ok_or_else(|| {
        format!(
            "external CUDA decode fixture {} manifest row is missing container",
            path.display()
        )
    })?;
    let actual_container = container_from_path_and_bytes(path, bytes);
    if actual_container != expected_container {
        return Err(format!(
            "external CUDA decode fixture {} container mismatch: manifest {expected_container} != detected {actual_container}",
            path.display()
        ));
    }
    Ok(())
}

fn manifest_column(headers: &[&str], name: &str) -> Result<usize, String> {
    optional_manifest_column(headers, name)
        .ok_or_else(|| format!("CUDA decode manifest is missing required {name:?} column"))
}

fn optional_manifest_column(headers: &[&str], name: &str) -> Option<usize> {
    headers.iter().position(|header| *header == name)
}

fn manifest_field<'a>(
    fields: &'a [&str],
    index: usize,
    name: &str,
    row_number: usize,
) -> Result<&'a str, String> {
    fields
        .get(index)
        .copied()
        .ok_or_else(|| format!("CUDA decode manifest row {row_number} is missing {name:?} field"))
}

fn manifest_optional_value(
    fields: &[&str],
    index: Option<usize>,
    name: &str,
    row_number: usize,
) -> Result<Option<String>, String> {
    let Some(index) = index else {
        return Ok(None);
    };
    let value = manifest_field(fields, index, name, row_number)?.trim();
    if value.is_empty() {
        return Ok(None);
    }
    if value.chars().any(char::is_control) {
        return Err(format!(
            "CUDA decode manifest row {row_number} field {name:?} contains a control character"
        ));
    }
    Ok(Some(value.to_string()))
}

fn codec_from_bytes(bytes: &[u8]) -> Option<&'static str> {
    let codestream = codestream_payload(bytes)?;
    match j2k_native::inspect_j2k_codestream_header(codestream) {
        Ok(header) if header.high_throughput => Some("htj2k"),
        Ok(_) => Some("j2k"),
        Err(_) => Some("unknown"),
    }
}

fn container_from_path_and_bytes(path: &Path, bytes: &[u8]) -> &'static str {
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        match extension.to_ascii_lowercase().as_str() {
            "jph" => return "jph",
            "jhc" => return "jhc",
            _ => {}
        }
    }
    if bytes.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']) {
        "jp2"
    } else {
        "raw-codestream"
    }
}

fn codestream_payload(bytes: &[u8]) -> Option<&[u8]> {
    if j2k_native::looks_like_j2k_codestream(bytes) {
        return Some(bytes);
    }
    if !bytes.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']) {
        return None;
    }
    let mut offset = 0_usize;
    while offset.checked_add(8)? <= bytes.len() {
        let lbox = u32::from_be_bytes(bytes[offset..offset + 4].try_into().ok()?) as usize;
        let box_type = &bytes[offset + 4..offset + 8];
        let (header_len, box_len) = match lbox {
            0 => (8, bytes.len() - offset),
            1 => {
                if offset.checked_add(16)? > bytes.len() {
                    return None;
                }
                let xlbox = u64::from_be_bytes(bytes[offset + 8..offset + 16].try_into().ok()?);
                let box_len = usize::try_from(xlbox).ok()?;
                (16, box_len)
            }
            value => (8, value),
        };
        if box_len < header_len || offset.checked_add(box_len)? > bytes.len() {
            return None;
        }
        if box_type == b"jp2c" {
            return Some(&bytes[offset + header_len..offset + box_len]);
        }
        offset += box_len;
    }
    None
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn sanitized_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("unnamed")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn dimensions_label(dimensions: (u32, u32)) -> String {
    format!("{}x{}", dimensions.0, dimensions.1)
}

fn pixel_format_label(fmt: PixelFormat) -> &'static str {
    match fmt {
        PixelFormat::Gray8 => "gray8",
        PixelFormat::Rgb8 => "rgb8",
        PixelFormat::Rgba8 => "rgba8",
        PixelFormat::Gray16 => "gray16",
        PixelFormat::Rgb16 => "rgb16",
        PixelFormat::Rgba16 => "rgba16",
        _ => "other",
    }
}

fn roi_for_case(case: &DecodeBenchCase) -> Rect {
    Rect {
        x: case.dimensions.0 / 4,
        y: case.dimensions.1 / 5,
        w: (case.dimensions.0 / 2).max(1),
        h: (case.dimensions.1 / 2).max(1),
    }
}

fn cuda_batch_decode_supported(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16
    )
}

fn enabled_decode_cases() -> Vec<&'static str> {
    let Some(value) = std::env::var_os("J2K_CUDA_DECODE_FORMATS") else {
        return vec!["gray8", "rgb8", "rgba8"];
    };
    let value = value.to_string_lossy();
    let mut cases = Vec::new();
    for raw in value.split(',') {
        let id = raw.trim();
        if id.is_empty() {
            continue;
        }
        let id = match id {
            "gray8" => "gray8",
            "rgb8" => "rgb8",
            "rgba8" => "rgba8",
            other => panic!(
                "unsupported J2K_CUDA_DECODE_FORMATS entry `{other}`; expected gray8,rgb8,rgba8"
            ),
        };
        if !cases.contains(&id) {
            cases.push(id);
        }
    }
    assert!(
        !cases.is_empty(),
        "J2K_CUDA_DECODE_FORMATS did not contain any decode formats"
    );
    cases
}

fn decode_batch_sizes() -> Vec<usize> {
    let Some(value) = std::env::var_os("J2K_CUDA_DECODE_BATCH_SIZES") else {
        return BATCH_SIZES.to_vec();
    };
    let value = value.to_string_lossy();
    let mut batch_sizes = Vec::new();
    for raw in value.split(',') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let batch_size = raw.parse::<usize>().unwrap_or_else(|error| {
            panic!("invalid J2K_CUDA_DECODE_BATCH_SIZES entry `{raw}`: {error}")
        });
        assert!(
            batch_size > 0,
            "J2K_CUDA_DECODE_BATCH_SIZES entries must be greater than zero"
        );
        if !batch_sizes.contains(&batch_size) {
            batch_sizes.push(batch_size);
        }
    }
    assert!(
        !batch_sizes.is_empty(),
        "J2K_CUDA_DECODE_BATCH_SIZES did not contain any batch sizes"
    );
    batch_sizes
}

fn cpu_benchmark_id(case: &DecodeBenchCase) -> String {
    match case.id.as_str() {
        "gray8" => "cpu_gray8".to_string(),
        "rgb8" => "cpu_rgb8".to_string(),
        "rgba8" => "cpu_rgba8".to_string(),
        other => format!("cpu_{other}"),
    }
}

fn cuda_benchmark_id(case: &DecodeBenchCase) -> String {
    match case.id.as_str() {
        "gray8" => "cuda_gray8".to_string(),
        "rgb8" => "cuda_rgb8".to_string(),
        "rgba8" => "cuda_rgba8".to_string(),
        other => format!("cuda_{other}"),
    }
}

fn assert_cuda_resident_decode(surface: &j2k_cuda::Surface) {
    let cuda = assert_cuda_resident_surface(surface);
    assert!(cuda.stats().decode_kernel_dispatches() > 0);
}

fn assert_cuda_resident_batch_decode(surfaces: &[j2k_cuda::Surface]) {
    assert!(!surfaces.is_empty());
    let decode_dispatches = surfaces
        .iter()
        .map(assert_cuda_resident_surface)
        .map(|cuda| cuda.stats().decode_kernel_dispatches())
        .sum::<usize>();
    assert!(decode_dispatches > 0);
}

fn assert_cuda_resident_surface(surface: &j2k_cuda::Surface) -> j2k_cuda::CudaSurface<'_> {
    assert_eq!(surface.backend_kind(), BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert!(surface.as_host_bytes().is_none());
    let cuda = surface.cuda_surface().expect("cuda surface");
    assert_ne!(cuda.device_ptr(), 0);
    assert_eq!(cuda.stats().copy_kernel_dispatches(), 0);
    cuda
}

fn cuda_decode_available(label: &str, fixture: &[u8], fmt: PixelFormat) -> bool {
    let mut session = CudaSession::default();
    let result = J2kDecoder::new(fixture)
        .and_then(|mut decoder| decoder.decode_to_device_with_session(fmt, &mut session));
    match result {
        Ok(surface) if surface.residency() == SurfaceResidency::CudaResidentDecode => true,
        Ok(_) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but {label} decode was not CUDA resident")
        }
        Ok(_) => {
            eprintln!(
                "skipping CUDA HTJ2K {label} decode benches: strict CUDA resident path unavailable"
            );
            false
        }
        Err(error) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but {label} CUDA decode failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA HTJ2K {label} decode benches: {error}");
            false
        }
    }
}

fn htj2k_gray8_fixture(width: u32, height: u32) -> Vec<u8> {
    let pixels = (0..width * height)
        .map(|idx| u8::try_from((idx * 17 + idx / 3) & 0xff).expect("masked sample fits in u8"))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 1, 8, false, &options).expect("encode HTJ2K fixture")
}

fn htj2k_rgb8_fixture(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for idx in 0..width * height {
        pixels.push(u8::try_from((idx * 17 + idx / 3) & 0xff).expect("masked red fits"));
        pixels.push(u8::try_from((idx * 29 + 7) & 0xff).expect("masked green fits"));
        pixels.push(u8::try_from((idx * 43 + 19) & 0xff).expect("masked blue fits"));
    }
    let options = EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 3, 8, false, &options).expect("encode RGB HTJ2K fixture")
}

criterion_group!(benches, bench_htj2k_decode);
criterion_main!(benches);
