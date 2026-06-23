// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use j2k_core::{
    BackendRequest, DeviceSubmission, DeviceSurface, ImageDecodeDevice, ImageDecodeSubmit,
    PixelFormat,
};
#[cfg(feature = "cuda-runtime")]
use j2k_core::{DecoderContext, TileBatchDecodeManyDevice};
use j2k_jpeg::{
    encode_jpeg_baseline, Decoder as CpuDecoder, JpegBackend, JpegEncodeOptions, JpegSamples,
    JpegSubsampling,
};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg_cuda::Codec as CudaCodec;
use j2k_jpeg_cuda::{CudaSession, Decoder as CudaDecoder};

const DEFAULT_JPEG: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
const DEFAULT_GENERATED_DIM: u16 = 2048;
#[cfg(feature = "cuda-runtime")]
const DEFAULT_BATCH_DIM: u16 = 1024;
#[cfg(feature = "cuda-runtime")]
const DEFAULT_BATCH_SIZE: usize = 64;

fn bench_device_decode(c: &mut Criterion) {
    let input = bench_input();
    let mut group = c.benchmark_group("jpeg_cuda_device_decode");

    group.bench_function("cpu_decode_rgb8", |b| {
        b.iter(|| {
            let decoder = CpuDecoder::new(&input).expect("cpu decoder");
            decoder.decode(PixelFormat::Rgb8).expect("cpu decode")
        });
    });

    match cuda_probe(&input) {
        Some(probe) => {
            let label = if probe.used_owned_cuda_decode {
                "cuda_owned_rgb8_surface"
            } else {
                "cuda_upload_fallback_rgb8_surface"
            };
            group.bench_function(label, |b| {
                let mut session = CudaSession::default();
                b.iter(|| {
                    let mut decoder = CudaDecoder::new(&input).expect("cuda decoder");
                    <CudaDecoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
                        &mut decoder,
                        &mut session,
                        PixelFormat::Rgb8,
                        BackendRequest::Cuda,
                    )
                    .expect("cuda submit")
                    .wait()
                    .expect("cuda decode")
                });
            });

            let label = if probe.used_owned_cuda_decode {
                "cuda_owned_rgb8_download"
            } else {
                "cuda_upload_fallback_rgb8_download"
            };
            group.bench_function(label, |b| {
                let mut session = CudaSession::default();
                b.iter(|| {
                    let mut decoder = CudaDecoder::new(&input).expect("cuda decoder");
                    let surface = <CudaDecoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
                        &mut decoder,
                        &mut session,
                        PixelFormat::Rgb8,
                        BackendRequest::Cuda,
                    )
                    .expect("cuda submit")
                    .wait()
                    .expect("cuda decode");
                    let mut out = vec![0u8; surface.byte_len()];
                    surface
                        .download_into(&mut out, surface.pitch_bytes())
                        .expect("cuda download");
                    out
                });
            });
        }
        None if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA decode is unavailable")
        }
        None => {
            eprintln!("skipping CUDA decode benches: CUDA runtime is unavailable");
        }
    }

    group.finish();

    bench_batch_decode(c);
    bench_chunked_entropy_diagnostic(c);
}

fn bench_input() -> Vec<u8> {
    let path =
        std::env::var_os("J2K_CUDA_BENCH_JPEG").or_else(|| std::env::var_os("J2K_GPU_BENCH_JPEG"));
    match path {
        Some(path) => std::fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "failed to read J2K_CUDA_BENCH_JPEG={}: {error}",
                path.to_string_lossy()
            )
        }),
        None if std::env::var_os("J2K_GPU_BENCH_SMALL_FIXTURE").is_some() => DEFAULT_JPEG.to_vec(),
        None => {
            let (width, height) = generated_dimensions();
            generated_jpeg(width, height)
        }
    }
}

fn generated_jpeg(width: u16, height: u16) -> Vec<u8> {
    let rgb = j2k_test_support::gpu_bench_rgb8(u32::from(width), u32::from(height));
    encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: u32::from(width),
            height: u32::from(height),
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: bench_subsampling(),
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode generated benchmark JPEG")
    .data
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

fn bench_subsampling() -> JpegSubsampling {
    let value = std::env::var("J2K_CUDA_BENCH_SUBSAMPLING")
        .or_else(|_| std::env::var("J2K_GPU_BENCH_SUBSAMPLING"))
        .unwrap_or_else(|_| "420".to_string());
    parse_subsampling(&value)
}

fn parse_subsampling(value: &str) -> JpegSubsampling {
    match value.trim().to_ascii_lowercase().as_str() {
        "420" | "4:2:0" | "ybr420" => JpegSubsampling::Ybr420,
        "422" | "4:2:2" | "ybr422" => JpegSubsampling::Ybr422,
        "444" | "4:4:4" | "ybr444" => JpegSubsampling::Ybr444,
        other => panic!("unsupported JPEG bench subsampling {other}; expected 420, 422, or 444"),
    }
}

#[cfg(feature = "cuda-runtime")]
fn bench_chunked_entropy_diagnostic(c: &mut Criterion) {
    let (width, height) = generated_dimensions();
    let input = generated_chunked_entropy_jpeg(width, height);

    if let Err(error) = probe_chunked_entropy_diagnostic(&input) {
        assert!(
            std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_none(),
            "J2K_REQUIRE_CUDA_BENCH is set but CUDA JPEG chunked entropy diagnostic is unavailable: {error}"
        );
        eprintln!("skipping CUDA JPEG chunked entropy diagnostic bench: {error}");
        return;
    }

    let mut group = c.benchmark_group("jpeg_cuda_chunked_entropy");
    group.sample_size(10);

    group.bench_function("cpu_fast_packet_planning", |b| {
        b.iter(|| {
            let packet = j2k_jpeg::adapter::build_fast420_packet(&input).expect("fast420 packet");
            std::hint::black_box(packet.entropy_checkpoints.len())
        });
    });

    group.bench_function("cuda_chunked_entropy_sync", |b| {
        let mut session = CudaSession::default();
        b.iter(|| {
            let report = CudaCodec::diagnose_tile_rgb8_chunked_entropy_with_session(
                &input,
                j2k_cuda_runtime::CudaJpegChunkedEntropyConfig::default(),
                &mut session,
            )
            .expect("chunked entropy diagnostic");
            std::hint::black_box(report.synchronized_overflow_count())
        });
    });

    group.finish();
}

#[cfg(not(feature = "cuda-runtime"))]
fn bench_chunked_entropy_diagnostic(_c: &mut Criterion) {}

#[cfg(feature = "cuda-runtime")]
fn generated_chunked_entropy_jpeg(width: u16, height: u16) -> Vec<u8> {
    assert_eq!(
        bench_subsampling(),
        JpegSubsampling::Ybr420,
        "jpeg_cuda_chunked_entropy requires generated 4:2:0 JPEG input; unset J2K_CUDA_BENCH_SUBSAMPLING/J2K_GPU_BENCH_SUBSAMPLING or set it to 420"
    );
    generated_jpeg(width, height)
}

#[cfg(feature = "cuda-runtime")]
fn probe_chunked_entropy_diagnostic(input: &[u8]) -> Result<(), j2k_jpeg_cuda::Error> {
    let mut session = CudaSession::default();
    CudaCodec::diagnose_tile_rgb8_chunked_entropy_with_session(
        input,
        j2k_cuda_runtime::CudaJpegChunkedEntropyConfig::default(),
        &mut session,
    )
    .map(|_| ())
}

#[cfg(feature = "cuda-runtime")]
fn bench_batch_decode(c: &mut Criterion) {
    let dim = batch_dim();
    let input = generated_jpeg(dim, dim);
    let batch_size = batch_size();
    let batch_refs = vec![input.as_slice(); batch_size];

    let mut group = c.benchmark_group("jpeg_cuda_batch_decode");
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

    let mut probe_ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut probe_pool = j2k_jpeg::ScratchPool::new();
    if let Err(error) = CudaCodec::decode_tiles_to_device(
        &mut probe_ctx,
        &mut probe_pool,
        &batch_refs[..1],
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    ) {
        assert!(
            std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_none(),
            "J2K_REQUIRE_CUDA_BENCH is set but owned CUDA JPEG batch decode is unavailable: {error}"
        );
        eprintln!("skipping CUDA adapter batch decode bench: {error}");
        group.finish();
        return;
    }

    group.bench_function(
        format!("cuda_adapter_rgb8_batch{batch_size}_surfaces"),
        |b| {
            let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
            let mut pool = j2k_jpeg::ScratchPool::new();
            b.iter(|| {
                let outputs = CudaCodec::decode_tiles_to_device(
                    &mut ctx,
                    &mut pool,
                    &batch_refs,
                    PixelFormat::Rgb8,
                    BackendRequest::Cuda,
                )
                .expect("cuda adapter batch decode");
                std::hint::black_box(outputs)
            });
        },
    );

    group.finish();
}

#[cfg(not(feature = "cuda-runtime"))]
fn bench_batch_decode(_c: &mut Criterion) {}

#[cfg(feature = "cuda-runtime")]
fn batch_size() -> usize {
    let Some(value) = std::env::var_os("J2K_GPU_BENCH_BATCH") else {
        return DEFAULT_BATCH_SIZE;
    };
    let value = value
        .to_string_lossy()
        .parse::<usize>()
        .expect("J2K_GPU_BENCH_BATCH must be a usize");
    assert!(
        (1..=256).contains(&value),
        "J2K_GPU_BENCH_BATCH must be between 1 and 256"
    );
    value
}

#[cfg(feature = "cuda-runtime")]
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

struct CudaProbe {
    used_owned_cuda_decode: bool,
}

fn cuda_probe(input: &[u8]) -> Option<CudaProbe> {
    let mut decoder = CudaDecoder::new(input).expect("cuda decoder");
    let surface = match decoder.decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda) {
        Ok(surface) => surface,
        Err(error) => {
            eprintln!("skipping CUDA decode benches: {error}");
            return None;
        }
    };
    let stats = surface.cuda_surface().expect("cuda surface").stats();
    if std::env::var_os("J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE").is_some()
        && !stats.used_owned_cuda_decode()
    {
        panic!(
            "J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE is set but owned CUDA JPEG decode was not used"
        );
    }
    Some(CudaProbe {
        used_owned_cuda_decode: stats.used_owned_cuda_decode(),
    })
}

criterion_group!(benches, bench_device_decode);
criterion_main!(benches);
