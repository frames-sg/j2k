// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
#[cfg(target_os = "macos")]
use j2k_core::PixelFormat;
use j2k_jpeg::{
    encode_jpeg_baseline, encode_jpeg_lossless, JpegBackend, JpegEncodeOptions,
    JpegLosslessEncodeOptions, JpegLosslessSamples, JpegSamples, JpegSubsampling,
};
#[cfg(target_os = "macos")]
use j2k_jpeg_metal::{
    encode_jpeg_baseline_batch_from_metal_buffers, encode_jpeg_baseline_from_metal_buffer,
    encode_jpeg_lossless_from_metal_buffer, JpegBaselineMetalEncodeTile,
    JpegLosslessMetalEncodeTile, MetalBackendSession,
};
#[cfg(target_os = "macos")]
use sha2::{Digest, Sha256};
use std::time::Duration;

const DEFAULT_DIM: u32 = 512;
const DEFAULT_BATCH_SIZE: usize = 8;
const DEFAULT_QUALITY: u8 = 90;

fn bench_single_encode(c: &mut Criterion, dim: u32, tile_bytes: usize, rgb: &[u8]) {
    let cpu_options = options(JpegBackend::Cpu);
    #[cfg(target_os = "macos")]
    let metal_options = options(JpegBackend::Metal);

    let mut single = c.benchmark_group("jpeg_baseline_encode_single");
    single.sample_size(10);
    single.warm_up_time(Duration::from_secs(1));
    single.measurement_time(Duration::from_secs(3));
    single.throughput(Throughput::Bytes(tile_bytes as u64));
    single.bench_function(format!("cpu_rgb8_422_{dim}x{dim}"), |b| {
        let tile = &rgb[..tile_bytes];
        b.iter(|| {
            encode_jpeg_baseline(
                JpegSamples::Rgb8 {
                    data: std::hint::black_box(tile),
                    width: dim,
                    height: dim,
                },
                cpu_options,
            )
            .expect("CPU JPEG Baseline encode")
        });
    });

    #[cfg(target_os = "macos")]
    if let Ok(session) = MetalBackendSession::system_default() {
        let Ok(buffer) = j2k_metal_support::checked_shared_buffer_with_slice(
            session.device(),
            &rgb[..tile_bytes],
        ) else {
            eprintln!("skipping Metal single benchmark: buffer allocation failed");
            return;
        };
        // SAFETY: the benchmark buffer is initialized before construction and
        // remains immutable while the tile descriptor is alive.
        let tile = unsafe {
            JpegBaselineMetalEncodeTile::new(
                &buffer,
                0,
                (dim, dim),
                dim as usize * 3,
                (dim, dim),
                PixelFormat::Rgb8,
            )
        };
        single.bench_function(format!("metal_rgb8_422_{dim}x{dim}"), |b| {
            b.iter(|| {
                encode_jpeg_baseline_from_metal_buffer(
                    std::hint::black_box(tile),
                    metal_options,
                    &session,
                )
                .expect("Metal JPEG Baseline encode")
            });
        });
    }
    single.finish();
}

fn bench_batch_encode(
    c: &mut Criterion,
    dim: u32,
    batch_size: usize,
    tile_bytes: usize,
    rgb: &[u8],
) {
    let cpu_options = options(JpegBackend::Cpu);
    #[cfg(target_os = "macos")]
    let metal_options = options(JpegBackend::Metal);
    let mut batch = c.benchmark_group("jpeg_baseline_encode_batch");
    batch.sample_size(10);
    batch.warm_up_time(Duration::from_secs(1));
    batch.measurement_time(Duration::from_secs(3));
    batch.throughput(Throughput::Bytes((tile_bytes * batch_size) as u64));
    batch.bench_function(format!("cpu_rgb8_422_{dim}x{dim}_batch{batch_size}"), |b| {
        b.iter(|| {
            let mut total = 0usize;
            for tile in rgb.chunks_exact(tile_bytes).take(batch_size) {
                let encoded = encode_jpeg_baseline(
                    JpegSamples::Rgb8 {
                        data: std::hint::black_box(tile),
                        width: dim,
                        height: dim,
                    },
                    cpu_options,
                )
                .expect("CPU JPEG Baseline batch encode");
                total = total.saturating_add(encoded.data.len());
                std::hint::black_box(encoded);
            }
            total
        });
    });

    #[cfg(target_os = "macos")]
    if let Ok(session) = MetalBackendSession::system_default() {
        let Ok(buffer) = j2k_metal_support::checked_shared_buffer_with_slice(session.device(), rgb)
        else {
            eprintln!("skipping Metal batch benchmark: buffer allocation failed");
            return;
        };
        // SAFETY: the benchmark buffer is initialized before construction and
        // remains immutable while all tile descriptors are alive.
        let tiles = (0..batch_size)
            .map(|tile| unsafe {
                JpegBaselineMetalEncodeTile::new(
                    &buffer,
                    tile * tile_bytes,
                    (dim, dim),
                    dim as usize * 3,
                    (dim, dim),
                    PixelFormat::Rgb8,
                )
            })
            .collect::<Vec<_>>();
        let probe = encode_jpeg_baseline_batch_from_metal_buffers(&tiles, metal_options, &session)
            .expect("Metal JPEG Baseline batch probe");
        let mut hasher = Sha256::new();
        for frame in &probe {
            let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&frame.data));
            let pixels = decoder.decode().expect("decode Metal batch probe");
            assert_eq!(pixels.len(), tile_bytes);
            hasher.update((frame.data.len() as u64).to_le_bytes());
            hasher.update(&frame.data);
        }
        let mcus = dim.div_ceil(16) as usize * dim.div_ceil(8) as usize;
        let coefficient_scratch_bytes = mcus * 4 * 64 * std::mem::size_of::<i32>() * batch_size;
        eprintln!(
            "jpeg_metal_staged_encode_probe path=staged dimensions={dim}x{dim} batch={batch_size} coefficient_scratch_bytes={coefficient_scratch_bytes} input_sha256={:x} output_sha256={:x}",
            Sha256::digest(rgb),
            hasher.finalize(),
        );
        batch.bench_function(
            format!("metal_rgb8_422_{dim}x{dim}_batch{batch_size}"),
            |b| {
                b.iter(|| {
                    encode_jpeg_baseline_batch_from_metal_buffers(
                        std::hint::black_box(&tiles),
                        metal_options,
                        &session,
                    )
                    .expect("Metal JPEG Baseline batch encode")
                });
            },
        );
    }
    batch.finish();
}

fn bench_encode_baseline(c: &mut Criterion) {
    let dim = bench_dim();
    let batch_size = bench_batch_size();
    let tile_bytes = dim as usize * dim as usize * 3;
    let rgb = j2k_test_support::patterned_rgb8_tiles(dim, dim, batch_size);

    bench_single_encode(c, dim, tile_bytes, &rgb);
    bench_batch_encode(c, dim, batch_size, tile_bytes, &rgb);
}

/// A 16-bit CT/CR-like phantom: smooth rings and gradients plus low-bit
/// noise, at `precision` bits.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "phantom values are clamped to the sample range before narrowing"
)]
fn phantom16(width: u32, height: u32, precision: u8) -> Vec<u16> {
    let max = f64::from((1_u32 << precision) - 1);
    let mut state = 0x9e37_79b9_u32;
    let mut samples = Vec::new();
    for y in 0..height {
        for x in 0..width {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let dx = f64::from(x) / f64::from(width) - 0.5;
            let dy = f64::from(y) / f64::from(height) - 0.5;
            let r = (dx * dx + dy * dy).sqrt();
            let v = 0.5 + 0.3 * (r * 40.0).cos() * (-r * 3.0).exp() + 0.15 * dx;
            let noise = f64::from(state % 16) - 7.5;
            samples.push((v.clamp(0.0, 1.0) * max + noise).clamp(0.0, max) as u16);
        }
    }
    samples
}

fn bench_lossless_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("jpeg_lossless_encode");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    #[cfg(target_os = "macos")]
    let session = MetalBackendSession::system_default().ok();
    for (width, height, precision, predictor) in
        [(512, 512, 16, 1), (2048, 2500, 16, 1), (512, 512, 12, 7)]
    {
        let samples = phantom16(width, height, precision);
        let mut options = JpegLosslessEncodeOptions::new(precision);
        options.predictor = predictor;
        let name = format!("gray{precision}_sv{predictor}_{width}x{height}");
        group.throughput(Throughput::Bytes(samples.len() as u64 * 2));
        group.bench_function(format!("cpu_{name}"), |b| {
            b.iter(|| {
                encode_jpeg_lossless(
                    JpegLosslessSamples::Gray16 {
                        data: std::hint::black_box(&samples),
                        width,
                        height,
                    },
                    options,
                )
                .expect("CPU lossless encode")
            });
        });
        #[cfg(target_os = "macos")]
        if let Some(session) = &session {
            let bytes: Vec<u8> = samples
                .iter()
                .flat_map(|sample| sample.to_ne_bytes())
                .collect();
            let Ok(buffer) =
                j2k_metal_support::checked_shared_buffer_with_slice(session.device(), &bytes)
            else {
                eprintln!("skipping Metal lossless benchmark: buffer allocation failed");
                continue;
            };
            // SAFETY: the buffer is initialized before construction and stays
            // immutable while the tile descriptor is alive.
            let tile = unsafe {
                JpegLosslessMetalEncodeTile::new(
                    &buffer,
                    0,
                    (width, height),
                    width as usize * 2,
                    PixelFormat::Gray16,
                )
            };
            group.bench_function(format!("metal_{name}"), |b| {
                b.iter(|| {
                    encode_jpeg_lossless_from_metal_buffer(
                        std::hint::black_box(tile),
                        options,
                        session,
                    )
                    .expect("Metal lossless encode")
                });
            });
        }
    }
    group.finish();
}

fn options(backend: JpegBackend) -> JpegEncodeOptions {
    JpegEncodeOptions {
        quality: bench_quality(),
        subsampling: JpegSubsampling::Ybr422,
        restart_interval: None,
        backend,
    }
}

fn bench_dim() -> u32 {
    let Some(value) = std::env::var_os("J2K_JPEG_ENCODE_BENCH_DIM") else {
        return DEFAULT_DIM;
    };
    let value = value
        .to_string_lossy()
        .parse::<u32>()
        .expect("J2K_JPEG_ENCODE_BENCH_DIM must be a u32");
    assert!(
        (64..=4096).contains(&value),
        "J2K_JPEG_ENCODE_BENCH_DIM must be between 64 and 4096"
    );
    value
}

fn bench_batch_size() -> usize {
    let Some(value) = std::env::var_os("J2K_JPEG_ENCODE_BENCH_BATCH") else {
        return DEFAULT_BATCH_SIZE;
    };
    let value = value
        .to_string_lossy()
        .parse::<usize>()
        .expect("J2K_JPEG_ENCODE_BENCH_BATCH must be a usize");
    assert!(
        (1..=128).contains(&value),
        "J2K_JPEG_ENCODE_BENCH_BATCH must be between 1 and 128"
    );
    value
}

fn bench_quality() -> u8 {
    let Some(value) = std::env::var_os("J2K_JPEG_ENCODE_BENCH_QUALITY") else {
        return DEFAULT_QUALITY;
    };
    let value = value
        .to_string_lossy()
        .parse::<u8>()
        .expect("J2K_JPEG_ENCODE_BENCH_QUALITY must be a u8");
    assert!(
        (1..=100).contains(&value),
        "J2K_JPEG_ENCODE_BENCH_QUALITY must be between 1 and 100"
    );
    value
}

criterion_group!(benches, bench_encode_baseline, bench_lossless_encode);
criterion_main!(benches);
