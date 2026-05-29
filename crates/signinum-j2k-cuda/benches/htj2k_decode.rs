// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use signinum_core::{
    BackendRequest, DecoderContext, Downscale, ImageDecode, ImageDecodeDevice, PixelFormat, Rect,
    TileBatchDecodeManyDevice,
};
use signinum_j2k_cuda::{Codec, CudaSession, J2kDecoder, SurfaceResidency};
use signinum_j2k_native::{encode_htj2k, EncodeOptions};

const TILE_DIM: u32 = 512;
const BATCH_SIZE: usize = 8;

fn bench_htj2k_decode(c: &mut Criterion) {
    let fixture = htj2k_gray8_fixture(TILE_DIM, TILE_DIM);
    let roi = Rect {
        x: TILE_DIM / 4,
        y: TILE_DIM / 5,
        w: TILE_DIM / 2,
        h: TILE_DIM / 2,
    };
    let scale = Downscale::Half;
    let cuda_available = cuda_decode_available(&fixture);

    bench_full_tile(c, &fixture, cuda_available);
    bench_roi(c, &fixture, roi, cuda_available);
    bench_scaled(c, &fixture, scale, cuda_available);
    bench_roi_scaled(c, &fixture, roi, scale, cuda_available);
    bench_tile_batch(c, &fixture, cuda_available);
}

fn bench_full_tile(c: &mut Criterion, fixture: &[u8], cuda_available: bool) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_full_tile_decode");
    group.bench_with_input(
        BenchmarkId::new("cpu_gray8", TILE_DIM),
        fixture,
        |b, bytes| {
            b.iter(|| {
                let mut decoder = J2kDecoder::new(black_box(bytes)).expect("decoder");
                let mut out = vec![0u8; TILE_DIM as usize * TILE_DIM as usize];
                decoder
                    .decode_into(&mut out, TILE_DIM as usize, PixelFormat::Gray8)
                    .expect("CPU HTJ2K decode");
                black_box(out)
            });
        },
    );
    if cuda_available {
        group.bench_with_input(
            BenchmarkId::new("cuda_gray8", TILE_DIM),
            fixture,
            |b, bytes| {
                b.iter(|| {
                    let mut decoder = J2kDecoder::new(black_box(bytes)).expect("decoder");
                    let surface = decoder
                        .decode_to_device(PixelFormat::Gray8, BackendRequest::Cuda)
                        .expect("strict CUDA HTJ2K decode");
                    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
                    black_box(surface)
                });
            },
        );
    }
    group.finish();
}

fn bench_roi(c: &mut Criterion, fixture: &[u8], roi: Rect, cuda_available: bool) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_roi_decode");
    group.bench_with_input(BenchmarkId::new("cpu_gray8", roi.w), fixture, |b, bytes| {
        b.iter(|| {
            let mut decoder = J2kDecoder::new(black_box(bytes)).expect("decoder");
            let mut pool = signinum_j2k_cuda::J2kScratchPool::new();
            let mut out = vec![0u8; roi.w as usize * roi.h as usize];
            decoder
                .decode_region_into(&mut pool, &mut out, roi.w as usize, PixelFormat::Gray8, roi)
                .expect("CPU HTJ2K ROI decode");
            black_box(out)
        });
    });
    if cuda_available {
        group.bench_with_input(
            BenchmarkId::new("cuda_gray8", roi.w),
            fixture,
            |b, bytes| {
                b.iter(|| {
                    let mut decoder = J2kDecoder::new(black_box(bytes)).expect("decoder");
                    let surface = decoder
                        .decode_region_to_device(PixelFormat::Gray8, roi, BackendRequest::Cuda)
                        .expect("strict CUDA HTJ2K ROI decode");
                    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
                    black_box(surface)
                });
            },
        );
    }
    group.finish();
}

fn bench_scaled(c: &mut Criterion, fixture: &[u8], scale: Downscale, cuda_available: bool) {
    let scaled = Rect::full((TILE_DIM, TILE_DIM)).scaled_covering(scale);
    let mut group = c.benchmark_group("j2k_cuda_htj2k_scaled_decode");
    group.bench_with_input(
        BenchmarkId::new("cpu_gray8", scaled.w),
        fixture,
        |b, bytes| {
            b.iter(|| {
                let mut decoder = J2kDecoder::new(black_box(bytes)).expect("decoder");
                let mut pool = signinum_j2k_cuda::J2kScratchPool::new();
                let mut out = vec![0u8; scaled.w as usize * scaled.h as usize];
                decoder
                    .decode_scaled_into(
                        &mut pool,
                        &mut out,
                        scaled.w as usize,
                        PixelFormat::Gray8,
                        scale,
                    )
                    .expect("CPU HTJ2K scaled decode");
                black_box(out)
            });
        },
    );
    if cuda_available {
        group.bench_with_input(
            BenchmarkId::new("cuda_gray8", scaled.w),
            fixture,
            |b, bytes| {
                b.iter(|| {
                    let mut decoder = J2kDecoder::new(black_box(bytes)).expect("decoder");
                    let surface = decoder
                        .decode_scaled_to_device(PixelFormat::Gray8, scale, BackendRequest::Cuda)
                        .expect("strict CUDA HTJ2K scaled decode");
                    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
                    black_box(surface)
                });
            },
        );
    }
    group.finish();
}

fn bench_roi_scaled(
    c: &mut Criterion,
    fixture: &[u8],
    roi: Rect,
    scale: Downscale,
    cuda_available: bool,
) {
    let scaled = roi.scaled_covering(scale);
    let mut group = c.benchmark_group("j2k_cuda_htj2k_roi_scaled_decode");
    group.bench_with_input(
        BenchmarkId::new("cpu_gray8", scaled.w),
        fixture,
        |b, bytes| {
            b.iter(|| {
                let mut decoder = J2kDecoder::new(black_box(bytes)).expect("decoder");
                let mut pool = signinum_j2k_cuda::J2kScratchPool::new();
                let mut out = vec![0u8; scaled.w as usize * scaled.h as usize];
                decoder
                    .decode_region_scaled_into(
                        &mut pool,
                        &mut out,
                        scaled.w as usize,
                        PixelFormat::Gray8,
                        roi,
                        scale,
                    )
                    .expect("CPU HTJ2K ROI+scaled decode");
                black_box(out)
            });
        },
    );
    if cuda_available {
        group.bench_with_input(
            BenchmarkId::new("cuda_gray8", scaled.w),
            fixture,
            |b, bytes| {
                b.iter(|| {
                    let mut decoder = J2kDecoder::new(black_box(bytes)).expect("decoder");
                    let surface = decoder
                        .decode_region_scaled_to_device(
                            PixelFormat::Gray8,
                            roi,
                            scale,
                            BackendRequest::Cuda,
                        )
                        .expect("strict CUDA HTJ2K ROI+scaled decode");
                    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
                    black_box(surface)
                });
            },
        );
    }
    group.finish();
}

fn bench_tile_batch(c: &mut Criterion, fixture: &[u8], cuda_available: bool) {
    let fixtures = vec![fixture.to_vec(); BATCH_SIZE];
    let inputs = fixtures.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut group = c.benchmark_group("j2k_cuda_htj2k_tile_batch_decode");
    group.bench_with_input(
        BenchmarkId::new("cpu_gray8", BATCH_SIZE),
        &inputs,
        |b, inputs| {
            b.iter(|| {
                let mut ctx = DecoderContext::<signinum_j2k_cuda::J2kContext>::new();
                let mut pool = signinum_j2k_cuda::J2kScratchPool::new();
                let surfaces = Codec::decode_tiles_to_device(
                    &mut ctx,
                    &mut pool,
                    black_box(inputs),
                    PixelFormat::Gray8,
                    BackendRequest::Cpu,
                )
                .expect("CPU HTJ2K batch decode");
                black_box(surfaces)
            });
        },
    );
    if cuda_available {
        group.bench_with_input(
            BenchmarkId::new("cuda_gray8", BATCH_SIZE),
            &inputs,
            |b, inputs| {
                b.iter(|| {
                    let mut ctx = DecoderContext::<signinum_j2k_cuda::J2kContext>::new();
                    let mut pool = signinum_j2k_cuda::J2kScratchPool::new();
                    let surfaces = Codec::decode_tiles_to_device(
                        &mut ctx,
                        &mut pool,
                        black_box(inputs),
                        PixelFormat::Gray8,
                        BackendRequest::Cuda,
                    )
                    .expect("strict CUDA HTJ2K batch decode");
                    assert!(
                        surfaces
                            .iter()
                            .all(|surface| surface.residency()
                                == SurfaceResidency::CudaResidentDecode)
                    );
                    black_box(surfaces)
                });
            },
        );
    }
    group.finish();
}

fn cuda_decode_available(fixture: &[u8]) -> bool {
    let mut session = CudaSession::default();
    let result = J2kDecoder::new(fixture).and_then(|mut decoder| {
        decoder.decode_to_device_with_session(PixelFormat::Gray8, &mut session)
    });
    match result {
        Ok(surface) if surface.residency() == SurfaceResidency::CudaResidentDecode => true,
        Ok(_) if std::env::var_os("SIGNINUM_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("SIGNINUM_REQUIRE_CUDA_BENCH is set but decode was not CUDA resident")
        }
        Ok(_) => {
            eprintln!("skipping CUDA HTJ2K decode benches: strict CUDA resident path unavailable");
            false
        }
        Err(error) if std::env::var_os("SIGNINUM_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("SIGNINUM_REQUIRE_CUDA_BENCH is set but CUDA decode failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA HTJ2K decode benches: {error}");
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

criterion_group!(benches, bench_htj2k_decode);
criterion_main!(benches);
