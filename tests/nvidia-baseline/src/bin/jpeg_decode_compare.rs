// SPDX-License-Identifier: Apache-2.0

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
use std::time::Instant;

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
use j2k_core::PixelFormat;
#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
use j2k_jpeg::{
    encode_jpeg_baseline, Decoder as JpegDecoder, JpegBackend, JpegEncodeOptions, JpegSamples,
    JpegSubsampling,
};
#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
use j2k_jpeg_cuda::{Codec as JpegCudaCodec, CudaSession};
#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
use j2k_nvidia_baseline::{nvidia_decode_jpeg_rgb, NvBaselineSession, NvJpegDecodeTiming};

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
const DEFAULT_DIM: u32 = 2048;
#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
const DEFAULT_ITERS: usize = 100;
#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
const DEFAULT_WARMUP: usize = 10;

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = Config::from_env();
    let (jpeg, input_label) = if let Some(path) = std::env::var_os("J2K_CUDA_BENCH_JPEG") {
        let jpeg = std::fs::read(&path)?;
        let info = JpegDecoder::inspect(&jpeg)?;
        config.width = info.dimensions.0;
        config.height = info.dimensions.1;
        (jpeg, format!("file={}", path.to_string_lossy()))
    } else {
        let rgb = patterned_rgb8(config.width, config.height, config.pattern);
        let jpeg = encode_jpeg_baseline(
            JpegSamples::Rgb8 {
                data: &rgb,
                width: config.width,
                height: config.height,
            },
            JpegEncodeOptions {
                quality: config.quality,
                subsampling: config.subsampling,
                restart_interval: None,
                backend: JpegBackend::Cpu,
            },
        )?
        .data;
        (
            jpeg,
            format!(
                "generated_pattern={} subsampling={}",
                config.pattern.name(),
                subsampling_name(config.subsampling)
            ),
        )
    };

    let owned = time_owned_cuda(&jpeg, config)?;
    let nvidia = time_nvjpeg(&jpeg, config)?;
    let (nvjpeg_pixels, nvjpeg_max_delta_vs_cpu) = nvjpeg_pixels_and_max_delta_vs_cpu(&jpeg)?;
    let owned_max_delta_vs_nvjpeg = max_delta(&owned.verification_pixels, &nvjpeg_pixels);

    println!("JPEG decode comparison");
    println!(
        "input: {}x{}  {}  jpeg_bytes={}  warmup={}  iterations={}",
        config.width,
        config.height,
        input_label,
        jpeg.len(),
        config.warmup,
        config.iterations
    );
    println!();
    print_summary("owned_cuda_wall_ms", &owned.wall_ms);
    print_summary("nvjpeg_wall_ms", &nvidia.wall_ms);
    print_summary("nvjpeg_event_ms", &nvidia.event_ms);
    println!();
    println!(
        "owned_vs_nvjpeg_wall_mean: {:.2}x",
        mean(&owned.wall_ms) / mean(&nvidia.wall_ms)
    );
    println!(
        "owned_vs_nvjpeg_event_mean: {:.2}x",
        mean(&owned.wall_ms) / mean(&nvidia.event_ms)
    );
    println!("owned_cuda_max_delta_vs_cpu: {}", owned.max_delta_vs_cpu);
    println!("nvjpeg_max_delta_vs_cpu: {}", nvjpeg_max_delta_vs_cpu);
    println!(
        "owned_cuda_max_delta_vs_nvjpeg: {}",
        owned_max_delta_vs_nvjpeg
    );
    println!(
        "note: owned_cuda_wall_ms includes Rust JPEG inspect/cache lookup, CUDA driver launch, and status readback; nvjpeg_event_ms is CUDA event time for the nvJPEG decode call only."
    );

    Ok(())
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
#[derive(Clone, Copy)]
struct Config {
    width: u32,
    height: u32,
    quality: u8,
    iterations: usize,
    warmup: usize,
    pattern: Pattern,
    subsampling: JpegSubsampling,
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
impl Config {
    fn from_env() -> Self {
        let (width, height) = std::env::var("J2K_GPU_BENCH_DIM")
            .ok()
            .map_or((DEFAULT_DIM, DEFAULT_DIM), |value| parse_dimensions(&value));
        Self {
            width,
            height,
            quality: std::env::var("J2K_JPEG_COMPARE_QUALITY")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(90),
            iterations: std::env::var("J2K_JPEG_COMPARE_ITERS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_ITERS),
            warmup: std::env::var("J2K_JPEG_COMPARE_WARMUP")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_WARMUP),
            pattern: Pattern::from_env(),
            subsampling: subsampling_from_env(),
        }
    }
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn subsampling_from_env() -> JpegSubsampling {
    let value = std::env::var("J2K_JPEG_COMPARE_SUBSAMPLING")
        .or_else(|_| std::env::var("J2K_CUDA_BENCH_SUBSAMPLING"))
        .unwrap_or_else(|_| "420".to_string());
    match value.trim().to_ascii_lowercase().as_str() {
        "420" | "4:2:0" | "ybr420" => JpegSubsampling::Ybr420,
        "422" | "4:2:2" | "ybr422" => JpegSubsampling::Ybr422,
        "444" | "4:4:4" | "ybr444" => JpegSubsampling::Ybr444,
        other => {
            panic!("unsupported J2K_JPEG_COMPARE_SUBSAMPLING={other}; expected 420, 422, or 444")
        }
    }
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
const fn subsampling_name(subsampling: JpegSubsampling) -> &'static str {
    match subsampling {
        JpegSubsampling::Gray => "gray",
        JpegSubsampling::Ybr444 => "444",
        JpegSubsampling::Ybr422 => "422",
        JpegSubsampling::Ybr420 => "420",
    }
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
#[derive(Clone, Copy)]
enum Pattern {
    Stress,
    Smooth,
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
impl Pattern {
    fn from_env() -> Self {
        match std::env::var("J2K_JPEG_COMPARE_PATTERN")
            .unwrap_or_else(|_| "stress".to_string())
            .as_str()
        {
            "smooth" => Self::Smooth,
            "stress" => Self::Stress,
            other => panic!("unsupported J2K_JPEG_COMPARE_PATTERN={other}"),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Stress => "stress",
            Self::Smooth => "smooth",
        }
    }
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
struct OwnedTiming {
    wall_ms: Vec<f64>,
    max_delta_vs_cpu: u8,
    verification_pixels: Vec<u8>,
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
struct NvidiaTiming {
    wall_ms: Vec<f64>,
    event_ms: Vec<f64>,
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn time_owned_cuda(jpeg: &[u8], config: Config) -> Result<OwnedTiming, Box<dyn std::error::Error>> {
    let mut session = CudaSession::default();
    let pitch = config.width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let byte_len = pitch * config.height as usize;
    let output = session.take_owned_cuda_output_buffer(byte_len)?;
    for _ in 0..config.warmup {
        let stats = JpegCudaCodec::decode_tile_rgb8_into_cuda_buffer_with_session(
            jpeg,
            &output,
            pitch,
            &mut session,
        )?;
        assert!(stats.used_owned_cuda_decode());
    }

    let mut wall_ms = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        let start = Instant::now();
        let stats = JpegCudaCodec::decode_tile_rgb8_into_cuda_buffer_with_session(
            jpeg,
            &output,
            pitch,
            &mut session,
        )?;
        assert!(stats.used_owned_cuda_decode());
        wall_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let mut downloaded = vec![0u8; byte_len];
    output.copy_to_host(&mut downloaded)?;
    let (expected, _) = JpegDecoder::new(jpeg)?.decode(PixelFormat::Rgb8)?;
    let max_delta_vs_cpu = downloaded
        .iter()
        .zip(expected)
        .map(|(actual, expected)| actual.abs_diff(expected))
        .max()
        .unwrap_or(0);
    Ok(OwnedTiming {
        wall_ms,
        max_delta_vs_cpu,
        verification_pixels: downloaded,
    })
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn time_nvjpeg(jpeg: &[u8], config: Config) -> Result<NvidiaTiming, Box<dyn std::error::Error>> {
    let mut session = NvBaselineSession::new()
        .map_err(|error| format!("NVIDIA baseline unavailable: {error:?}"))?;
    for _ in 0..config.warmup {
        let timing = session
            .decode_jpeg_rgb_interleaved_timed(jpeg)
            .map_err(|error| format!("nvJPEG warmup failed: {error:?}"))?;
        assert_decode_dimensions(timing, config);
    }

    let mut wall_ms = Vec::with_capacity(config.iterations);
    let mut event_ms = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        let start = Instant::now();
        let timing = session
            .decode_jpeg_rgb_interleaved_timed(jpeg)
            .map_err(|error| format!("nvJPEG decode failed: {error:?}"))?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_decode_dimensions(timing, config);
        wall_ms.push(elapsed_ms);
        event_ms.push(timing.decode_ms);
    }
    Ok(NvidiaTiming { wall_ms, event_ms })
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn nvjpeg_pixels_and_max_delta_vs_cpu(
    jpeg: &[u8],
) -> Result<(Vec<u8>, u8), Box<dyn std::error::Error>> {
    let (actual, width, height) =
        nvidia_decode_jpeg_rgb(jpeg).map_err(|error| format!("nvJPEG decode failed: {error:?}"))?;
    let (expected, outcome) = JpegDecoder::new(jpeg)?.decode(PixelFormat::Rgb8)?;
    assert_eq!((width, height), (outcome.decoded.w, outcome.decoded.h));
    let delta = max_delta(&actual, &expected);
    Ok((actual, delta))
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn assert_decode_dimensions(timing: NvJpegDecodeTiming, config: Config) {
    assert_eq!((timing.width, timing.height), (config.width, config.height));
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn patterned_rgb8(width: u32, height: u32, pattern: Pattern) -> Vec<u8> {
    let mut out = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            match pattern {
                Pattern::Stress => {
                    out.push(((x * 13 + y * 3) & 0xFF) as u8);
                    out.push(((x * 5 + y * 17) & 0xFF) as u8);
                    out.push(((x * 11 + y * 7 + (x ^ y)) & 0xFF) as u8);
                }
                Pattern::Smooth => {
                    let sx = x.saturating_mul(255) / width.max(1);
                    let sy = y.saturating_mul(255) / height.max(1);
                    let tissue = (((x / 64) * 19 + (y / 64) * 37) & 0x3F) as u32;
                    out.push(((sx * 3 + sy + tissue) / 5).min(255) as u8);
                    out.push(((sx + sy * 2 + tissue * 2) / 5).min(255) as u8);
                    out.push(((sx + sy + 128 + tissue) / 4).min(255) as u8);
                }
            }
        }
    }
    out
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn parse_dimensions(value: &str) -> (u32, u32) {
    if let Some((width, height)) = value.split_once('x') {
        (
            width
                .trim()
                .parse()
                .expect("J2K_GPU_BENCH_DIM width must be u32"),
            height
                .trim()
                .parse()
                .expect("J2K_GPU_BENCH_DIM height must be u32"),
        )
    } else {
        let square = value
            .trim()
            .parse()
            .expect("J2K_GPU_BENCH_DIM must be u32 or WIDTHxHEIGHT");
        (square, square)
    }
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn print_summary(label: &str, values: &[f64]) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{label}: mean={:.4} min={:.4} p50={:.4} p90={:.4}",
        mean(values),
        sorted[0],
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.90)
    );
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[index]
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn max_delta(actual: &[u8], expected: &[u8]) -> u8 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| actual.abs_diff(*expected))
        .max()
        .unwrap_or(0)
}

#[cfg(any(target_os = "macos", not(feature = "nvjpeg2000")))]
fn main() {
    eprintln!("jpeg_decode_compare requires --features nvjpeg2000 on a CUDA/NVIDIA baseline host");
}
