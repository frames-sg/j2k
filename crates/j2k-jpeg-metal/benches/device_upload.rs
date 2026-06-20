// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use j2k_core::{
    BackendRequest, DecoderContext, DeviceSubmission, ImageDecodeDevice, PixelFormat,
    TileBatchDecodeSubmit,
};
use j2k_jpeg::{Decoder as CpuDecoder, DecoderContext as JpegDecoderContext};
use j2k_jpeg_metal::{Codec, Decoder as MetalDecoder, MetalSession, ScratchPool};
use jpeg_encoder::{ColorType, Encoder, SamplingFactor};

const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
const DEFAULT_GENERATED_DIM: u16 = 2048;
const DEFAULT_BATCH_DIM: u16 = 1024;
const DEFAULT_BATCH_SIZE: usize = 64;

fn bench_device_upload(c: &mut Criterion) {
    let input = bench_input();
    let mut group = c.benchmark_group("jpeg_metal_device");

    group.bench_function("cpu_decode_rgb8", |b| {
        let decoder = CpuDecoder::new(&input).expect("cpu decoder");
        b.iter(|| decoder.decode(PixelFormat::Rgb8).expect("cpu decode"));
    });

    if metal_decode_available() {
        group.bench_function("metal_surface_rgb8", |b| {
            b.iter(|| {
                let mut decoder = MetalDecoder::new(&input).expect("metal decoder");
                decoder
                    .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
                    .expect("device decode")
            });
        });
    }

    group.finish();

    bench_batch_decode(c);
}

fn metal_decode_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        metal::Device::system_default().is_some()
    }
    #[cfg(not(target_os = "macos"))]
    {
        assert!(
            std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_none(),
            "J2K_REQUIRE_METAL_BENCH is set but this is not a Metal host"
        );
        false
    }
}

fn bench_input() -> Vec<u8> {
    match std::env::var_os("J2K_GPU_BENCH_JPEG") {
        Some(path) => std::fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "failed to read J2K_GPU_BENCH_JPEG={}: {error}",
                path.to_string_lossy()
            )
        }),
        None if std::env::var_os("J2K_GPU_BENCH_SMALL_FIXTURE").is_some() => BASELINE_420.to_vec(),
        None => {
            let (width, height) = generated_dimensions();
            generated_jpeg(width, height)
        }
    }
}

fn generated_jpeg(width: u16, height: u16) -> Vec<u8> {
    let rgb = j2k_test_support::gpu_bench_rgb8(u32::from(width), u32::from(height));

    let mut jpeg = Vec::new();
    let mut encoder = Encoder::new(&mut jpeg, 90);
    encoder.set_sampling_factor(SamplingFactor::F_2_2);
    if let Some(interval) = restart_interval() {
        encoder.set_restart_interval(interval);
    }
    encoder
        .encode(&rgb, width, height, ColorType::Rgb)
        .expect("encode generated benchmark JPEG");
    jpeg
}

fn generated_dimensions() -> (u16, u16) {
    let Some(value) = std::env::var_os("J2K_GPU_BENCH_DIM") else {
        return (DEFAULT_GENERATED_DIM, DEFAULT_GENERATED_DIM);
    };
    parse_dimensions(&value.to_string_lossy())
}

fn parse_dimensions(value: &str) -> (u16, u16) {
    if let Some((width, height)) = value.split_once('x') {
        parse_dimensions_pair(width, height)
    } else {
        let square = value
            .parse::<u16>()
            .expect("J2K_GPU_BENCH_DIM must be a u16 or WIDTHxHEIGHT");
        assert_in_bench_bounds(square);
        (square, square)
    }
}

fn parse_dimensions_pair(width: &str, height: &str) -> (u16, u16) {
    let width = width
        .trim()
        .parse::<u16>()
        .expect("J2K_GPU_BENCH_DIM must be a u16 or WIDTHxHEIGHT");
    let height = height
        .trim()
        .parse::<u16>()
        .expect("J2K_GPU_BENCH_DIM must be a u16 or WIDTHxHEIGHT");
    assert_in_bench_bounds(width);
    assert_in_bench_bounds(height);
    (width, height)
}

fn assert_in_bench_bounds(value: u16) {
    assert!(
        (256..=8192).contains(&value),
        "J2K_GPU_BENCH_DIM dimensions must be between 256 and 8192"
    );
}

fn restart_interval() -> Option<u16> {
    let value = std::env::var_os("J2K_GPU_BENCH_RESTART_INTERVAL")?;
    let value = value
        .to_string_lossy()
        .parse::<u16>()
        .expect("J2K_GPU_BENCH_RESTART_INTERVAL must be a u16");
    if value == 0 {
        None
    } else {
        Some(value)
    }
}

fn bench_batch_decode(c: &mut Criterion) {
    let dim = batch_dim();
    let input = generated_jpeg(dim, dim);
    let batch_size = batch_size();

    let mut group = c.benchmark_group("jpeg_metal_batch_decode");
    group.sample_size(10);

    group.bench_function(format!("cpu_decode_rgb8_batch{batch_size}"), |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..batch_size {
                let decoder = CpuDecoder::new(&input).expect("cpu decoder");
                let decoded_rgb = decoder.decode(PixelFormat::Rgb8).expect("cpu decode");
                total = total.saturating_add(decoded_rgb.0.len());
                std::hint::black_box(decoded_rgb);
            }
            total
        });
    });

    if metal_decode_available() {
        group.bench_function(format!("metal_rgb8_batch{batch_size}_surfaces"), |b| {
            b.iter(|| {
                device_decode_tile_batch(&input, batch_size, BackendRequest::Metal);
            });
        });
    }

    group.bench_function(format!("auto_rgb8_batch{batch_size}_surfaces"), |b| {
        b.iter(|| {
            device_decode_tile_batch(&input, batch_size, BackendRequest::Auto);
        });
    });

    group.finish();
}

fn device_decode_tile_batch(input: &[u8], batch_size: usize, backend: BackendRequest) {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let submissions = (0..batch_size)
        .map(|_| {
            <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                input,
                PixelFormat::Rgb8,
                backend,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();
    for submission in submissions {
        std::hint::black_box(submission.wait().expect("surface"));
    }
}

fn batch_size() -> usize {
    let Some(value) = std::env::var_os("J2K_GPU_BENCH_BATCH") else {
        return DEFAULT_BATCH_SIZE;
    };
    let value = value
        .to_string_lossy()
        .parse::<usize>()
        .expect("J2K_GPU_BENCH_BATCH must be a usize");
    assert!(
        (1..=512).contains(&value),
        "J2K_GPU_BENCH_BATCH must be between 1 and 512"
    );
    value
}

fn batch_dim() -> u16 {
    let Some(value) = std::env::var_os("J2K_GPU_BENCH_BATCH_DIM") else {
        return DEFAULT_BATCH_DIM;
    };
    let value = value
        .to_string_lossy()
        .parse::<u16>()
        .expect("J2K_GPU_BENCH_BATCH_DIM must be a u16");
    assert!(
        (128..=4096).contains(&value),
        "J2K_GPU_BENCH_BATCH_DIM must be between 128 and 4096"
    );
    value
}

criterion_group!(benches, bench_device_upload);
criterion_main!(benches);
