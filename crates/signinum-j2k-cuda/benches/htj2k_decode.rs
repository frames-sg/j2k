// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use signinum_core::{
    BackendKind, BackendRequest, DecoderContext, DeviceSubmission, DeviceSurface, Downscale,
    ImageDecode, ImageDecodeSubmit, PixelFormat, Rect, TileBatchDecodeManyDevice,
};
use signinum_j2k_cuda::{Codec, CudaSession, J2kDecoder, SurfaceResidency};
use signinum_j2k_native::{encode_htj2k, EncodeOptions};

const TILE_DIM: u32 = 512;
const BATCH_SIZES: &[usize] = &[8, 16, 32, 64];

struct DecodeBenchCase {
    id: &'static str,
    fixture: Vec<u8>,
    fmt: PixelFormat,
    cuda_available: bool,
}

fn bench_htj2k_decode(c: &mut Criterion) {
    let enabled_cases = enabled_decode_cases();
    let mut cases = Vec::new();

    if enabled_cases.contains(&"gray8") {
        let gray_fixture = htj2k_gray8_fixture(TILE_DIM, TILE_DIM);
        cases.push(DecodeBenchCase {
            id: "gray8",
            cuda_available: cuda_decode_available("gray8", &gray_fixture, PixelFormat::Gray8),
            fixture: gray_fixture,
            fmt: PixelFormat::Gray8,
        });
    }
    if enabled_cases
        .iter()
        .any(|id| matches!(*id, "rgb8" | "rgba8"))
    {
        let rgb_fixture = htj2k_rgb8_fixture(TILE_DIM, TILE_DIM);
        if enabled_cases.contains(&"rgb8") {
            cases.push(DecodeBenchCase {
                id: "rgb8",
                cuda_available: cuda_decode_available("rgb8", &rgb_fixture, PixelFormat::Rgb8),
                fixture: rgb_fixture.clone(),
                fmt: PixelFormat::Rgb8,
            });
        }
        if enabled_cases.contains(&"rgba8") {
            cases.push(DecodeBenchCase {
                id: "rgba8",
                cuda_available: cuda_decode_available("rgba8", &rgb_fixture, PixelFormat::Rgba8),
                fixture: rgb_fixture,
                fmt: PixelFormat::Rgba8,
            });
        }
    }

    let roi = Rect {
        x: TILE_DIM / 4,
        y: TILE_DIM / 5,
        w: TILE_DIM / 2,
        h: TILE_DIM / 2,
    };
    let scale = Downscale::Half;

    bench_full_tile(c, &cases);
    bench_roi(c, &cases, roi);
    bench_scaled(c, &cases, scale);
    bench_roi_scaled(c, &cases, roi, scale);
    bench_tile_batch(c, &cases);
}

fn bench_full_tile(c: &mut Criterion, cases: &[DecodeBenchCase]) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_full_tile_decode");
    for case in cases {
        let cpu_id = cpu_benchmark_id(case);
        group.bench_with_input(BenchmarkId::new(cpu_id, TILE_DIM), case, |b, case| {
            b.iter(|| {
                let mut decoder =
                    J2kDecoder::new(black_box(case.fixture.as_slice())).expect("decoder");
                let stride = TILE_DIM as usize * case.fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * TILE_DIM as usize];
                decoder
                    .decode_into(&mut out, stride, case.fmt)
                    .expect("CPU HTJ2K decode");
                black_box(out)
            });
        });
        if case.cuda_available {
            let cuda_id = cuda_benchmark_id(case);
            group.bench_with_input(BenchmarkId::new(cuda_id, TILE_DIM), case, |b, case| {
                let mut session = CudaSession::default();
                b.iter(|| {
                    let mut decoder =
                        J2kDecoder::new(black_box(case.fixture.as_slice())).expect("decoder");
                    let surface = decoder
                        .submit_to_device(&mut session, case.fmt, BackendRequest::Cuda)
                        .expect("strict CUDA HTJ2K decode submission")
                        .wait()
                        .expect("strict CUDA HTJ2K decode");
                    assert_cuda_resident_decode(&surface);
                    black_box(surface)
                });
            });
        }
    }
    group.finish();
}

fn bench_roi(c: &mut Criterion, cases: &[DecodeBenchCase], roi: Rect) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_roi_decode");
    for case in cases {
        let cpu_id = cpu_benchmark_id(case);
        group.bench_with_input(BenchmarkId::new(cpu_id, roi.w), case, |b, case| {
            b.iter(|| {
                let mut decoder =
                    J2kDecoder::new(black_box(case.fixture.as_slice())).expect("decoder");
                let mut pool = signinum_j2k_cuda::J2kScratchPool::new();
                let stride = roi.w as usize * case.fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * roi.h as usize];
                decoder
                    .decode_region_into(&mut pool, &mut out, stride, case.fmt, roi)
                    .expect("CPU HTJ2K ROI decode");
                black_box(out)
            });
        });
        if case.cuda_available {
            let cuda_id = cuda_benchmark_id(case);
            group.bench_with_input(BenchmarkId::new(cuda_id, roi.w), case, |b, case| {
                let mut session = CudaSession::default();
                b.iter(|| {
                    let mut decoder =
                        J2kDecoder::new(black_box(case.fixture.as_slice())).expect("decoder");
                    let surface = decoder
                        .submit_region_to_device(&mut session, case.fmt, roi, BackendRequest::Cuda)
                        .expect("strict CUDA HTJ2K ROI decode submission")
                        .wait()
                        .expect("strict CUDA HTJ2K ROI decode");
                    assert_cuda_resident_decode(&surface);
                    black_box(surface)
                });
            });
        }
    }
    group.finish();
}

fn bench_scaled(c: &mut Criterion, cases: &[DecodeBenchCase], scale: Downscale) {
    let scaled = Rect::full((TILE_DIM, TILE_DIM)).scaled_covering(scale);
    let mut group = c.benchmark_group("j2k_cuda_htj2k_scaled_decode");
    for case in cases {
        let cpu_id = cpu_benchmark_id(case);
        group.bench_with_input(BenchmarkId::new(cpu_id, scaled.w), case, |b, case| {
            b.iter(|| {
                let mut decoder =
                    J2kDecoder::new(black_box(case.fixture.as_slice())).expect("decoder");
                let mut pool = signinum_j2k_cuda::J2kScratchPool::new();
                let stride = scaled.w as usize * case.fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * scaled.h as usize];
                decoder
                    .decode_scaled_into(&mut pool, &mut out, stride, case.fmt, scale)
                    .expect("CPU HTJ2K scaled decode");
                black_box(out)
            });
        });
        if case.cuda_available {
            let cuda_id = cuda_benchmark_id(case);
            group.bench_with_input(BenchmarkId::new(cuda_id, scaled.w), case, |b, case| {
                let mut session = CudaSession::default();
                b.iter(|| {
                    let mut decoder =
                        J2kDecoder::new(black_box(case.fixture.as_slice())).expect("decoder");
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
                    black_box(surface)
                });
            });
        }
    }
    group.finish();
}

fn bench_roi_scaled(c: &mut Criterion, cases: &[DecodeBenchCase], roi: Rect, scale: Downscale) {
    let scaled = roi.scaled_covering(scale);
    let mut group = c.benchmark_group("j2k_cuda_htj2k_roi_scaled_decode");
    for case in cases {
        let cpu_id = cpu_benchmark_id(case);
        group.bench_with_input(BenchmarkId::new(cpu_id, scaled.w), case, |b, case| {
            b.iter(|| {
                let mut decoder =
                    J2kDecoder::new(black_box(case.fixture.as_slice())).expect("decoder");
                let mut pool = signinum_j2k_cuda::J2kScratchPool::new();
                let stride = scaled.w as usize * case.fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * scaled.h as usize];
                decoder
                    .decode_region_scaled_into(&mut pool, &mut out, stride, case.fmt, roi, scale)
                    .expect("CPU HTJ2K ROI+scaled decode");
                black_box(out)
            });
        });
        if case.cuda_available {
            let cuda_id = cuda_benchmark_id(case);
            group.bench_with_input(BenchmarkId::new(cuda_id, scaled.w), case, |b, case| {
                let mut session = CudaSession::default();
                b.iter(|| {
                    let mut decoder =
                        J2kDecoder::new(black_box(case.fixture.as_slice())).expect("decoder");
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
                    black_box(surface)
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
                        let mut ctx = DecoderContext::<signinum_j2k_cuda::J2kContext>::new();
                        let mut pool = signinum_j2k_cuda::J2kScratchPool::new();
                        let surfaces = Codec::decode_tiles_to_device(
                            &mut ctx,
                            &mut pool,
                            black_box(inputs),
                            fmt,
                            BackendRequest::Cpu,
                        )
                        .expect("CPU HTJ2K batch decode");
                        black_box(surfaces)
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
                                black_box(inputs),
                                fmt,
                                &mut session,
                            )
                            .expect("strict CUDA HTJ2K real batch decode");
                            assert_eq!(surfaces.len(), inputs.len());
                            surfaces.iter().for_each(assert_cuda_resident_decode);
                            black_box(surfaces)
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

fn cuda_batch_decode_supported(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16
    )
}

fn enabled_decode_cases() -> Vec<&'static str> {
    let Some(value) = std::env::var_os("SIGNINUM_J2K_CUDA_DECODE_FORMATS") else {
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
                "unsupported SIGNINUM_J2K_CUDA_DECODE_FORMATS entry `{other}`; expected gray8,rgb8,rgba8"
            ),
        };
        if !cases.contains(&id) {
            cases.push(id);
        }
    }
    assert!(
        !cases.is_empty(),
        "SIGNINUM_J2K_CUDA_DECODE_FORMATS did not contain any decode formats"
    );
    cases
}

fn decode_batch_sizes() -> Vec<usize> {
    let Some(value) = std::env::var_os("SIGNINUM_J2K_CUDA_DECODE_BATCH_SIZES") else {
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
            panic!("invalid SIGNINUM_J2K_CUDA_DECODE_BATCH_SIZES entry `{raw}`: {error}")
        });
        assert!(
            batch_size > 0,
            "SIGNINUM_J2K_CUDA_DECODE_BATCH_SIZES entries must be greater than zero"
        );
        if !batch_sizes.contains(&batch_size) {
            batch_sizes.push(batch_size);
        }
    }
    assert!(
        !batch_sizes.is_empty(),
        "SIGNINUM_J2K_CUDA_DECODE_BATCH_SIZES did not contain any batch sizes"
    );
    batch_sizes
}

fn cpu_benchmark_id(case: &DecodeBenchCase) -> &'static str {
    match case.id {
        "gray8" => "cpu_gray8",
        "rgb8" => "cpu_rgb8",
        "rgba8" => "cpu_rgba8",
        other => panic!("unknown CPU decode bench case `{other}`"),
    }
}

fn cuda_benchmark_id(case: &DecodeBenchCase) -> &'static str {
    match case.id {
        "gray8" => "cuda_gray8",
        "rgb8" => "cuda_rgb8",
        "rgba8" => "cuda_rgba8",
        other => panic!("unknown CUDA decode bench case `{other}`"),
    }
}

fn assert_cuda_resident_decode(surface: &signinum_j2k_cuda::Surface) {
    assert_eq!(surface.backend_kind(), BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert!(surface.as_host_bytes().is_none());
    let cuda = surface.cuda_surface().expect("cuda surface");
    assert_ne!(cuda.device_ptr(), 0);
    assert_eq!(cuda.stats().copy_kernel_dispatches(), 0);
    assert!(cuda.stats().decode_kernel_dispatches() > 0);
}

fn cuda_decode_available(label: &str, fixture: &[u8], fmt: PixelFormat) -> bool {
    let mut session = CudaSession::default();
    let result = J2kDecoder::new(fixture)
        .and_then(|mut decoder| decoder.decode_to_device_with_session(fmt, &mut session));
    match result {
        Ok(surface) if surface.residency() == SurfaceResidency::CudaResidentDecode => true,
        Ok(_) if std::env::var_os("SIGNINUM_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("SIGNINUM_REQUIRE_CUDA_BENCH is set but {label} decode was not CUDA resident")
        }
        Ok(_) => {
            eprintln!(
                "skipping CUDA HTJ2K {label} decode benches: strict CUDA resident path unavailable"
            );
            false
        }
        Err(error) if std::env::var_os("SIGNINUM_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("SIGNINUM_REQUIRE_CUDA_BENCH is set but {label} CUDA decode failed: {error}")
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
