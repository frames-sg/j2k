// SPDX-License-Identifier: Apache-2.0
//
// JPEG -> HTJ2K transcode throughput comparison:
//   j2k  — coefficient-domain batch transcode (CUDA transform with CPU or CUDA HT encode)
//   NVIDIA    — reused-session serial nvJPEG decode + nvJPEG2000 HT encode
//
// Reports the four metric families requested: end-to-end throughput (MP/s),
// per-stage breakdown, output size + PSNR, and GPU-only vs wall-clock.
//
// Usage:
//   transcode_compare <file.jpg> [more.jpg ...]
//   transcode_compare              # falls back to J2K_BENCH_JPEG_DIR/*.jpg
//
// The NVIDIA columns show "n/a (not built)" unless this crate was compiled with
// --features nvjpeg2000 on a host with nvcc + libnvjpeg + libnvjpeg2k.

mod report_format;

use std::{path::PathBuf, time::Instant};

use j2k::adapter::encode_stage::IrreversibleQuantizationSubbandScales;
#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
use j2k::adapter::encode_stage::J2kEncodeStageAccelerator;
#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
use j2k_cuda::CudaEncodeStageAccelerator;
use j2k_native::{DecodeSettings, Image};
use j2k_nvidia_baseline::{
    nvidia_decode_jpeg_rgb, psnr_u8, write_text_artifact, ycbcr_to_rgb_round_nearest,
    NvBaselineError, NvBaselineSession,
};
use j2k_transcode::{
    EncodedTranscodeBatch, JpegTileBatchInput, JpegToHtj2kError, JpegToHtj2kOptions,
    JpegToHtj2kTranscoder, JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE,
};
use report_format::{
    csv_f64_or_blank, escape_csv as csv_escape, escape_json as json_escape, json_f64_or_null,
};

// The j2k GPU backend is platform-selected: Metal on macOS, CUDA elsewhere.
#[cfg(not(target_os = "macos"))]
use j2k_transcode_cuda::CudaDctToWaveletStageAccelerator as BenchAccelerator;
#[cfg(target_os = "macos")]
use j2k_transcode_metal::MetalDctToWaveletStageAccelerator as BenchAccelerator;

const BACKEND_NAME: &str = if cfg!(target_os = "macos") {
    "Metal"
} else {
    "CUDA"
};

const WARMUP: usize = 2;
const ITERATIONS: usize = 10;

#[allow(clippy::too_many_lines)]
fn main() {
    let config = match BenchmarkConfig::from_env_args() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let jpegs = match load_inputs(&config) {
        Ok(jpegs) if !jpegs.is_empty() => jpegs,
        Ok(_) => {
            eprintln!("no JPEG inputs found; pass file paths or set J2K_BENCH_JPEG_DIR");
            std::process::exit(2);
        }
        Err(error) => {
            eprintln!("failed to load inputs: {error}");
            std::process::exit(2);
        }
    };
    if let Some(min_tiles) = config.min_tiles {
        if jpegs.len() < min_tiles {
            eprintln!(
                "input corpus has {} tile(s), below required --min-tiles {min_tiles}",
                jpegs.len()
            );
            std::process::exit(2);
        }
    }

    let total_pixels: u64 = jpegs
        .iter()
        .map(|j| u64::from(j.width) * u64::from(j.height))
        .sum();
    let megapixels = total_pixels as f64 / 1.0e6;
    println!(
        "inputs: {} tile(s), {:.2} MP total\n",
        jpegs.len(),
        megapixels
    );

    if config.profile_j2k_cuda_only {
        let scale = config
            .quant_scales
            .first()
            .copied()
            .unwrap_or(JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE);
        let options = lossy_options_for_config(&config, scale);
        let j2k_cuda_ht = run_j2k(&jpegs, &options, &config);
        print_j2k_cuda_profile_report(
            &jpegs,
            megapixels,
            corpus_hash(&jpegs),
            &config,
            scale,
            &j2k_cuda_ht,
        );
        if let Err(error) =
            write_j2k_profile_artifacts(&config, &jpegs, megapixels, scale, &j2k_cuda_ht)
        {
            eprintln!("failed to write benchmark artifacts: {error}");
            std::process::exit(2);
        }
        enforce_required_j2k_result(&j2k_cuda_ht);
        return;
    }

    let nvidia = run_nvidia(&jpegs, &config);
    let rd_points = run_j2k_rd_sweep(&jpegs, &config, &nvidia);
    let selected_index = select_rd_point(&rd_points, &config, &nvidia);
    let j2k_cpu_ht = rd_points
        .get(selected_index)
        .map_or_else(J2KResult::default, |point| point.result.clone());
    let selected_scale = rd_points
        .get(selected_index)
        .map_or(JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE, |point| {
            point.scale
        });
    let selected_options = lossy_options_for_config(&config, selected_scale);
    let j2k_cuda_ht = run_j2k(&jpegs, &selected_options, &config);

    print_report(
        &jpegs,
        megapixels,
        corpus_hash(&jpegs),
        &config,
        &rd_points,
        selected_index,
        &j2k_cpu_ht,
        &j2k_cuda_ht,
        &nvidia,
    );
    if let Err(error) = write_artifacts(
        &config,
        &jpegs,
        megapixels,
        &rd_points,
        selected_index,
        &j2k_cuda_ht,
        &nvidia,
    ) {
        eprintln!("failed to write benchmark artifacts: {error}");
        std::process::exit(2);
    }
    if let Err(error) = validate_rate_match(&config, &j2k_cpu_ht, &j2k_cuda_ht, &nvidia) {
        eprintln!("{error}");
        std::process::exit(1);
    }
    enforce_required_results(&j2k_cuda_ht, &nvidia);
}

#[derive(Debug, Clone)]
struct BenchmarkConfig {
    input_paths: Vec<PathBuf>,
    quant_scales: Vec<f32>,
    subband_scales: IrreversibleQuantizationSubbandScales,
    decomposition_levels: u8,
    match_nvidia_bytes: bool,
    match_tolerance: f64,
    profile_j2k_cuda_only: bool,
    warmup: usize,
    iterations: usize,
    min_tiles: Option<usize>,
    json_path: Option<PathBuf>,
    csv_path: Option<PathBuf>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            input_paths: Vec::new(),
            quant_scales: vec![JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE],
            subband_scales: IrreversibleQuantizationSubbandScales::default(),
            decomposition_levels: 1,
            match_nvidia_bytes: false,
            match_tolerance: 0.02,
            profile_j2k_cuda_only: false,
            warmup: WARMUP,
            iterations: ITERATIONS,
            min_tiles: None,
            json_path: None,
            csv_path: None,
        }
    }
}

impl BenchmarkConfig {
    fn from_env_args() -> Result<Self, String> {
        Self::parse(std::env::args_os().skip(1).map(PathBuf::from))
    }

    #[allow(clippy::similar_names)]
    fn parse(args: impl IntoIterator<Item = PathBuf>) -> Result<Self, String> {
        let mut config = Self::default();
        let mut iter = args.into_iter();
        while let Some(arg) = iter.next() {
            let arg_s = arg.to_string_lossy();
            match arg_s.as_ref() {
                "--match-nvidia-bytes" => config.match_nvidia_bytes = true,
                "--profile-j2k-cuda-only" => config.profile_j2k_cuda_only = true,
                "--quant-scales" => {
                    let value = iter
                        .next()
                        .ok_or("--quant-scales requires a comma-separated value")?;
                    config.quant_scales = parse_f32_list(&value.to_string_lossy())?;
                    if config.quant_scales.is_empty() {
                        return Err("--quant-scales must contain at least one scale".to_string());
                    }
                }
                "--subband-scales" => {
                    let value = iter
                        .next()
                        .ok_or("--subband-scales requires ll,hl,lh,hh values")?;
                    config.subband_scales = parse_subband_scales(&value.to_string_lossy())?;
                }
                "--decomposition-levels" => {
                    let value = iter
                        .next()
                        .ok_or("--decomposition-levels requires a value")?;
                    config.decomposition_levels =
                        parse_positive_u8(&value.to_string_lossy(), "--decomposition-levels")?;
                }
                "--match-tolerance" => {
                    let value = iter.next().ok_or("--match-tolerance requires a value")?;
                    config.match_tolerance =
                        parse_positive_f64(&value.to_string_lossy(), "--match-tolerance")?;
                }
                "--warmup" => {
                    let value = iter.next().ok_or("--warmup requires a value")?;
                    config.warmup = parse_positive_usize(&value.to_string_lossy(), "--warmup")?;
                }
                "--iterations" => {
                    let value = iter.next().ok_or("--iterations requires a value")?;
                    config.iterations =
                        parse_positive_usize(&value.to_string_lossy(), "--iterations")?;
                }
                "--min-tiles" => {
                    let value = iter.next().ok_or("--min-tiles requires a value")?;
                    config.min_tiles = Some(parse_positive_usize(
                        &value.to_string_lossy(),
                        "--min-tiles",
                    )?);
                }
                "--json" => {
                    config.json_path = Some(iter.next().ok_or("--json requires a path")?);
                }
                "--csv" => {
                    config.csv_path = Some(iter.next().ok_or("--csv requires a path")?);
                }
                "--help" | "-h" => return Err(usage()),
                _ if arg_s.starts_with("--") => {
                    return Err(format!("unknown option: {arg_s}\n{}", usage()));
                }
                _ => config.input_paths.push(arg),
            }
        }
        if config.profile_j2k_cuda_only && config.match_nvidia_bytes {
            return Err(
                "--profile-j2k-cuda-only cannot be combined with --match-nvidia-bytes".to_string(),
            );
        }
        Ok(config)
    }
}

fn usage() -> String {
    "usage: transcode_compare [--profile-j2k-cuda-only] [--quant-scales a,b,c] [--subband-scales ll,hl,lh,hh] [--decomposition-levels n] [--match-nvidia-bytes] [--match-tolerance frac] [--warmup n] [--iterations n] [--min-tiles n] [--json path] [--csv path] [file.jpg ...]".to_string()
}

fn parse_f32_list(value: &str) -> Result<Vec<f32>, String> {
    value
        .split(',')
        .map(|part| {
            let parsed = part
                .trim()
                .parse::<f32>()
                .map_err(|_| format!("invalid f32 value: {part}"))?;
            if parsed.is_finite() && parsed > 0.0 {
                Ok(parsed)
            } else {
                Err(format!(
                    "scale must be finite and greater than zero: {part}"
                ))
            }
        })
        .collect()
}

fn parse_positive_f64(value: &str, flag: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("{flag} must be a finite positive number"))?;
    if parsed.is_finite() && parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(format!("{flag} must be finite and greater than zero"))
    }
}

fn parse_positive_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{flag} must be a positive integer"))
        .and_then(|value| {
            (value > 0)
                .then_some(value)
                .ok_or_else(|| format!("{flag} must be greater than zero"))
        })
}

fn parse_positive_u8(value: &str, flag: &str) -> Result<u8, String> {
    let parsed = parse_positive_usize(value, flag)?;
    u8::try_from(parsed).map_err(|_| format!("{flag} must be <= {}", u8::MAX))
}

fn parse_subband_scales(value: &str) -> Result<IrreversibleQuantizationSubbandScales, String> {
    let values = parse_f32_list(value)?;
    if values.len() != 4 {
        return Err("--subband-scales expects exactly four values: ll,hl,lh,hh".to_string());
    }
    Ok(IrreversibleQuantizationSubbandScales {
        low_low: values[0],
        high_low: values[1],
        low_high: values[2],
        high_high: values[3],
    })
}

struct JpegInput {
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    label: String,
}

fn load_inputs(config: &BenchmarkConfig) -> std::io::Result<Vec<JpegInput>> {
    let mut paths = config.input_paths.clone();
    if paths.is_empty() {
        if let Some(dir) = std::env::var_os("J2K_BENCH_JPEG_DIR") {
            for entry in std::fs::read_dir(dir)? {
                let path = entry?.path();
                if path.extension().is_some_and(|ext| {
                    ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg")
                }) {
                    paths.push(path);
                }
            }
            paths.sort();
        }
    }
    if paths.is_empty() {
        // Bundled sanity fixture (tiny/unrepresentative — for a smoke test only).
        let bytes = include_bytes!(
            "../../../../crates/j2k-transcode-cuda/tests/fixtures/conformance/baseline_420_16x16.jpg"
        )
        .to_vec();
        eprintln!("warning: no inputs given; using the bundled 16x16 fixture (not representative)");
        return Ok(vec![JpegInput {
            width: 16,
            height: 16,
            label: "baseline_420_16x16".to_string(),
            bytes,
        }]);
    }

    let mut inputs = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = std::fs::read(&path)?;
        let (width, height) = jpeg_dimensions(&bytes).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("could not parse JPEG dimensions from {}", path.display()),
            )
        })?;
        inputs.push(JpegInput {
            bytes,
            width,
            height,
            label: path.file_name().map_or_else(
                || path.display().to_string(),
                |name| name.to_string_lossy().into_owned(),
            ),
        });
    }
    Ok(inputs)
}

/// Parse a JPEG's pixel dimensions from its SOF marker (no decode).
fn jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut i = 2; // skip SOI
    while i + 9 < bytes.len() {
        if bytes[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = bytes[i + 1];
        // SOF0..SOF3, SOF5..SOF7, SOF9..SOF11, SOF13..SOF15 carry dimensions.
        let is_sof = matches!(marker, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF);
        let len = (u16::from(bytes[i + 2]) << 8 | u16::from(bytes[i + 3])) as usize;
        if is_sof {
            let height = u32::from(bytes[i + 5]) << 8 | u32::from(bytes[i + 6]);
            let width = u32::from(bytes[i + 7]) << 8 | u32::from(bytes[i + 8]);
            return (width != 0 && height != 0).then_some((width, height));
        }
        if marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
            i += 2;
        } else {
            i += 2 + len;
        }
    }
    None
}

#[derive(Clone, Default)]
struct J2KResult {
    ran: bool,
    used_gpu: bool,
    best_wall_s: f64,
    extract_us: u128,
    repack_us: u128,
    transform_wall_us: u128,
    encode_wall_us: u128,
    transform_gpu_stage_us: u128,
    encode_cuda_stage_us: u128,
    encode_cuda_ht_kernel_us: u128,
    encode_cuda_ht_status_readback_us: u128,
    encode_cuda_ht_compact_us: u128,
    encode_cuda_ht_output_readback_us: u128,
    pack_upload_us: u128,
    idct_row_lift_us: u128,
    column_lift_us: u128,
    quantize_us: u128,
    readback_us: u128,
    transform_dispatches: usize,
    transform_dispatched_jobs: usize,
    transform_cpu_fallback_jobs: usize,
    encode_dispatches: usize,
    encode_ht_code_block_dispatches: usize,
    encode_packetization_dispatches: usize,
    output_bytes: usize,
    codestreams: Vec<Vec<u8>>,
}

#[derive(Clone)]
struct RdPoint {
    scale: f32,
    result: J2KResult,
    quality: Option<QualitySummary>,
}

#[derive(Clone, Copy, Default)]
struct EncodeBenchMetrics {
    cuda_stage_us: u128,
    total_dispatches: usize,
    ht_code_block_dispatches: usize,
    packetization_dispatches: usize,
}

fn lossy_options_for_config(config: &BenchmarkConfig, scale: f32) -> JpegToHtj2kOptions {
    let mut options = JpegToHtj2kOptions::lossy_97();
    options.encode_options.irreversible_quantization_scale = scale;
    options.encode_options.num_decomposition_levels = config.decomposition_levels;
    options
        .encode_options
        .irreversible_quantization_subband_scales = config.subband_scales;
    options
}

fn run_j2k_rd_sweep(
    jpegs: &[JpegInput],
    config: &BenchmarkConfig,
    nvidia: &NvidiaResult,
) -> Vec<RdPoint> {
    config
        .quant_scales
        .iter()
        .copied()
        .map(|scale| {
            let options = lossy_options_for_config(config, scale);
            let result = run_j2k_transform_cpu_encode(jpegs, &options, config);
            let quality = quality_summary(jpegs, &result.codestreams);
            if config.match_nvidia_bytes && nvidia.ran {
                let delta = byte_delta_fraction(result.output_bytes, nvidia.output_bytes);
                println!(
                    "RD point: scale {scale:.4}  bytes {}  delta vs NVIDIA {:+.2}%  PSNR {}",
                    result.output_bytes,
                    delta * 100.0,
                    fmt_psnr(quality.as_ref().and_then(|q| q.mean_psnr)),
                );
            }
            RdPoint {
                scale,
                result,
                quality,
            }
        })
        .collect()
}

fn select_rd_point(points: &[RdPoint], config: &BenchmarkConfig, nvidia: &NvidiaResult) -> usize {
    if points.is_empty() {
        return 0;
    }
    if !(config.match_nvidia_bytes && nvidia.ran) {
        return 0;
    }
    let selected = points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            byte_delta_fraction(a.result.output_bytes, nvidia.output_bytes)
                .abs()
                .partial_cmp(&byte_delta_fraction(b.result.output_bytes, nvidia.output_bytes).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(0, |(idx, _)| idx);
    let delta =
        byte_delta_fraction(points[selected].result.output_bytes, nvidia.output_bytes).abs();
    if delta > config.match_tolerance {
        eprintln!(
            "warning: closest J2K RD point is {:.2}% from NVIDIA bytes, outside {:.2}% tolerance",
            delta * 100.0,
            config.match_tolerance * 100.0
        );
    }
    selected
}

fn validate_rate_match(
    config: &BenchmarkConfig,
    selected_cpu_ht: &J2KResult,
    cuda_ht: &J2KResult,
    nvidia: &NvidiaResult,
) -> Result<(), String> {
    if !(config.match_nvidia_bytes && nvidia.ran) {
        return Ok(());
    }

    let selected_delta =
        byte_delta_fraction(selected_cpu_ht.output_bytes, nvidia.output_bytes).abs();
    if selected_delta > config.match_tolerance {
        return Err(format!(
            "selected J2K RD point is {:.2}% from NVIDIA bytes, outside {:.2}% tolerance",
            selected_delta * 100.0,
            config.match_tolerance * 100.0
        ));
    }

    if cuda_ht.ran {
        let cuda_delta = byte_delta_fraction(cuda_ht.output_bytes, nvidia.output_bytes).abs();
        if cuda_delta > config.match_tolerance {
            return Err(format!(
                "J2K CUDA HT row is {:.2}% from NVIDIA bytes, outside {:.2}% tolerance",
                cuda_delta * 100.0,
                config.match_tolerance * 100.0
            ));
        }
    }

    Ok(())
}

fn byte_delta_fraction(candidate: usize, target: usize) -> f64 {
    if target == 0 {
        return 0.0;
    }
    (candidate as f64 - target as f64) / target as f64
}

fn run_j2k(
    jpegs: &[JpegInput],
    options: &JpegToHtj2kOptions,
    config: &BenchmarkConfig,
) -> J2KResult {
    let inputs: Vec<JpegTileBatchInput<'_>> = jpegs
        .iter()
        .map(|j| JpegTileBatchInput { bytes: &j.bytes })
        .collect();
    // Warm up (and detect whether the GPU path is available).
    let used_gpu = true;
    let mut session = J2KBenchSession::new(true, true);
    for iteration in 0..config.warmup.max(1) {
        match session.transcode_batch(&inputs, options) {
            Ok((batch, encode_metrics)) => validate_j2k_cuda_dispatch(&batch, encode_metrics),
            Err(error) => {
                assert!(
                    ! j2k_cuda_required(),
                    "j2k: explicit {BACKEND_NAME} path failed under J2K_REQUIRE_CUDA_RUNTIME=1: {error:?}"
                );
                if iteration == 0 {
                    eprintln!(
                        "j2k: explicit {BACKEND_NAME} CUDA HT path unavailable; reporting CUDA HT row as n/a"
                    );
                }
                return J2KResult::default();
            }
        }
    }

    let mut best_wall_s = f64::INFINITY;
    let mut last = J2KResult::default();
    for _ in 0..config.iterations {
        let start = Instant::now();
        let batch = session.transcode_batch(&inputs, options);
        let elapsed = start.elapsed().as_secs_f64();
        let (batch, encode_metrics) = match batch {
            Ok(batch) => batch,
            Err(error) => {
                assert!(
                    ! j2k_cuda_required(),
                    "j2k: explicit {BACKEND_NAME} path failed under J2K_REQUIRE_CUDA_RUNTIME=1: {error:?}"
                );
                return J2KResult::default();
            }
        };
        validate_j2k_cuda_dispatch(&batch, encode_metrics);
        if elapsed < best_wall_s {
            best_wall_s = elapsed;
            last = j2k_result_from_batch(&batch, used_gpu, elapsed, encode_metrics);
        }
    }
    last
}

fn run_j2k_transform_cpu_encode(
    jpegs: &[JpegInput],
    options: &JpegToHtj2kOptions,
    config: &BenchmarkConfig,
) -> J2KResult {
    let inputs: Vec<JpegTileBatchInput<'_>> = jpegs
        .iter()
        .map(|j| JpegTileBatchInput { bytes: &j.bytes })
        .collect();
    let mut used_gpu = true;
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = BenchAccelerator::new_explicit();
    for iteration in 0..config.warmup.max(1) {
        match transcoder
            .transcode_batch_with_accelerator(&inputs, options, &mut accelerator)
            .and_then(reject_failed_j2k_tiles)
        {
            Ok(batch) => validate_j2k_transform_dispatch(&batch),
            Err(error) => {
                assert!(
                    ! j2k_cuda_required(),
                    "j2k: explicit {BACKEND_NAME} transform path failed under J2K_REQUIRE_CUDA_RUNTIME=1: {error:?}"
                );
                used_gpu = false;
                if iteration == 0 {
                    eprintln!(
                        "j2k: explicit {BACKEND_NAME} transform unavailable; measuring scalar CPU fallback"
                    );
                }
                break;
            }
        }
    }

    let mut best_wall_s = f64::INFINITY;
    let mut last = J2KResult::default();
    for _ in 0..config.iterations {
        let start = Instant::now();
        let batch = if used_gpu {
            transcoder
                .transcode_batch_with_accelerator(&inputs, options, &mut accelerator)
                .and_then(reject_failed_j2k_tiles)
        } else {
            transcoder
                .transcode_batch(&inputs, options)
                .and_then(reject_failed_j2k_tiles)
        };
        let elapsed = start.elapsed().as_secs_f64();
        let batch = match batch {
            Ok(batch) => batch,
            Err(error) => {
                assert!(
                    ! j2k_cuda_required(),
                    "j2k: explicit {BACKEND_NAME} transform path failed under J2K_REQUIRE_CUDA_RUNTIME=1: {error:?}"
                );
                return J2KResult::default();
            }
        };
        if used_gpu {
            validate_j2k_transform_dispatch(&batch);
        }
        if elapsed < best_wall_s {
            best_wall_s = elapsed;
            last = j2k_result_from_batch(&batch, used_gpu, elapsed, EncodeBenchMetrics::default());
        }
    }
    last
}

fn j2k_result_from_batch(
    batch: &EncodedTranscodeBatch,
    used_gpu: bool,
    elapsed: f64,
    encode_metrics: EncodeBenchMetrics,
) -> J2KResult {
    let t = &batch.report.timings;
    J2KResult {
        ran: true,
        used_gpu,
        best_wall_s: elapsed,
        extract_us: batch.report.extract_us,
        repack_us: t.jpeg_dct_repack_us,
        transform_wall_us: batch.report.transform_us,
        encode_wall_us: batch.report.encode_us,
        pack_upload_us: t.dwt97_batch_pack_upload_us,
        idct_row_lift_us: t.dwt97_batch_idct_row_lift_us,
        column_lift_us: t.dwt97_batch_column_lift_us,
        quantize_us: t.dwt97_batch_quantize_codeblock_us,
        readback_us: t.dwt97_batch_readback_us,
        transform_gpu_stage_us: t.dwt97_batch_pack_upload_us
            + t.dwt97_batch_idct_row_lift_us
            + t.dwt97_batch_column_lift_us
            + t.dwt97_batch_quantize_codeblock_us
            + t.dwt97_batch_readback_us,
        encode_cuda_stage_us: encode_metrics
            .cuda_stage_us
            .saturating_add(t.dwt97_batch_ht_encode_us),
        encode_cuda_ht_kernel_us: t.dwt97_batch_ht_kernel_us,
        encode_cuda_ht_status_readback_us: t.dwt97_batch_ht_status_readback_us,
        encode_cuda_ht_compact_us: t.dwt97_batch_ht_compact_us,
        encode_cuda_ht_output_readback_us: t.dwt97_batch_ht_output_readback_us,
        transform_dispatches: t.accelerator_dispatches,
        transform_dispatched_jobs: t.accelerator_dispatched_jobs,
        transform_cpu_fallback_jobs: t.cpu_fallback_jobs,
        encode_dispatches: encode_metrics
            .total_dispatches
            .saturating_add(t.dwt97_batch_ht_codeblock_dispatches),
        encode_ht_code_block_dispatches: encode_metrics
            .ht_code_block_dispatches
            .saturating_add(t.dwt97_batch_ht_codeblock_dispatches),
        encode_packetization_dispatches: encode_metrics.packetization_dispatches,
        output_bytes: batch
            .tiles
            .iter()
            .flatten()
            .map(|tile| tile.codestream.len())
            .sum(),
        codestreams: batch
            .tiles
            .iter()
            .flatten()
            .map(|tile| tile.codestream.clone())
            .collect(),
    }
}

struct J2KBenchSession {
    use_gpu: bool,
    transcoder: JpegToHtj2kTranscoder,
    transform_accelerator: Option<BenchAccelerator>,
    #[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
    encode_accelerator: Option<CudaEncodeStageAccelerator>,
}

impl J2KBenchSession {
    fn new(use_gpu: bool, resident_ht_encode: bool) -> Self {
        Self {
            use_gpu,
            transcoder: JpegToHtj2kTranscoder::default(),
            transform_accelerator: use_gpu.then(|| new_bench_accelerator(resident_ht_encode)),
            #[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
            encode_accelerator: use_gpu.then(|| {
                CudaEncodeStageAccelerator::with_profile_collection(true)
                    .prefer_cpu_ht_subband(true)
                    .prefer_cpu_quantize_subband(true)
                    .prefer_cpu_packetization(true)
            }),
        }
    }

    fn transcode_batch(
        &mut self,
        inputs: &[JpegTileBatchInput<'_>],
        options: &JpegToHtj2kOptions,
    ) -> Result<(EncodedTranscodeBatch, EncodeBenchMetrics), JpegToHtj2kError> {
        if !self.use_gpu {
            return self
                .transcoder
                .transcode_batch(inputs, options)
                .and_then(reject_failed_j2k_tiles)
                .map(|batch| (batch, EncodeBenchMetrics::default()));
        }

        self.transcode_gpu_batch(inputs, options)
    }

    #[cfg(target_os = "macos")]
    fn transcode_gpu_batch(
        &mut self,
        inputs: &[JpegTileBatchInput<'_>],
        options: &JpegToHtj2kOptions,
    ) -> Result<(EncodedTranscodeBatch, EncodeBenchMetrics), JpegToHtj2kError> {
        let accelerator = self
            .transform_accelerator
            .as_mut()
            .expect("GPU j2k session has a transform accelerator");
        self.transcoder
            .transcode_batch_with_accelerator(inputs, options, accelerator)
            .and_then(reject_failed_j2k_tiles)
            .map(|batch| (batch, EncodeBenchMetrics::default()))
    }

    #[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
    fn transcode_gpu_batch(
        &mut self,
        inputs: &[JpegTileBatchInput<'_>],
        options: &JpegToHtj2kOptions,
    ) -> Result<(EncodedTranscodeBatch, EncodeBenchMetrics), JpegToHtj2kError> {
        let transform_accelerator = self
            .transform_accelerator
            .as_mut()
            .expect("GPU j2k session has a transform accelerator");
        let encode_accelerator = self
            .encode_accelerator
            .as_mut()
            .expect("CUDA j2k session has an encode accelerator");
        let before = encode_accelerator.dispatch_report();
        encode_accelerator.reset_collected_stage_timings();
        let batch = self.transcoder.transcode_batch_with_accelerators(
            inputs,
            options,
            transform_accelerator,
            encode_accelerator,
        )?;
        let batch = reject_failed_j2k_tiles(batch)?;
        let encode_timings = encode_accelerator.collected_stage_timings();
        let dispatch = encode_accelerator
            .dispatch_report()
            .saturating_delta(before);
        Ok((
            batch,
            EncodeBenchMetrics {
                cuda_stage_us: encode_timings.total_us(),
                total_dispatches: dispatch.total(),
                ht_code_block_dispatches: dispatch.ht_code_block,
                packetization_dispatches: dispatch.packetization,
            },
        ))
    }

    #[cfg(all(not(target_os = "macos"), not(feature = "nvjpeg2000")))]
    fn transcode_gpu_batch(
        &mut self,
        inputs: &[JpegTileBatchInput<'_>],
        options: &JpegToHtj2kOptions,
    ) -> Result<(EncodedTranscodeBatch, EncodeBenchMetrics), JpegToHtj2kError> {
        let accelerator = self
            .transform_accelerator
            .as_mut()
            .expect("GPU j2k session has a transform accelerator");
        self.transcoder
            .transcode_batch_with_accelerator(inputs, options, accelerator)
            .and_then(reject_failed_j2k_tiles)
            .map(|batch| (batch, EncodeBenchMetrics::default()))
    }
}

#[cfg(not(target_os = "macos"))]
fn new_bench_accelerator(resident_ht_encode: bool) -> BenchAccelerator {
    if resident_ht_encode {
        BenchAccelerator::new_explicit_resident_ht_encode()
    } else {
        BenchAccelerator::new_explicit()
    }
}

#[cfg(target_os = "macos")]
fn new_bench_accelerator(_resident_ht_encode: bool) -> BenchAccelerator {
    BenchAccelerator::new_explicit()
}

fn reject_failed_j2k_tiles(
    batch: EncodedTranscodeBatch,
) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
    if batch.report.failed_tiles == 0 {
        Ok(batch)
    } else {
        Err(JpegToHtj2kError::Validation(
            "j2k benchmark produced one or more failed tiles",
        ))
    }
}

fn j2k_cuda_required() -> bool {
    std::env::var_os("J2K_REQUIRE_CUDA_RUNTIME").is_some()
}

#[cfg(not(target_os = "macos"))]
fn j2k_cuda_stage_timings_disabled() -> bool {
    std::env::var_os("J2K_CUDA_DISABLE_STAGE_TIMINGS").is_some()
}

fn nvidia_baseline_required() -> bool {
    std::env::var_os("J2K_REQUIRE_NV_BASELINE_BUILD").is_some()
}

fn enforce_required_j2k_result(j2k: &J2KResult) {
    if j2k_cuda_required() && !(j2k.ran && j2k.used_gpu) {
        eprintln!("j2k: required CUDA benchmark did not produce a GPU result");
        std::process::exit(1);
    }
}

fn enforce_required_results(j2k: &J2KResult, nvidia: &NvidiaResult) {
    let mut failed = false;
    if j2k_cuda_required() && !(j2k.ran && j2k.used_gpu) {
        eprintln!("j2k: required CUDA benchmark did not produce a GPU result");
        failed = true;
    }
    if nvidia_baseline_required() && !nvidia.ran {
        eprintln!(
            "NVIDIA baseline: required nvJPEG/nvJPEG2000 benchmark did not run: {}",
            nv_status(nvidia)
        );
        failed = true;
    }
    if failed {
        std::process::exit(1);
    }
}

fn validate_j2k_cuda_dispatch(batch: &EncodedTranscodeBatch, encode: EncodeBenchMetrics) {
    #[cfg(not(target_os = "macos"))]
    {
        validate_j2k_transform_dispatch(batch);
        if !j2k_cuda_required() {
            return;
        }
        assert_eq!(
            batch.report.failed_tiles, 0,
            "j2k: CUDA benchmark produced failed tiles under J2K_REQUIRE_CUDA_RUNTIME=1"
        );
        let ht_code_block_dispatches = encode
            .ht_code_block_dispatches
            .saturating_add(batch.report.timings.dwt97_batch_ht_codeblock_dispatches);
        assert!(
            ht_code_block_dispatches != 0,
            "j2k: CUDA HT encode dispatched zero code-block batches under J2K_REQUIRE_CUDA_RUNTIME=1"
        );
    }

    #[cfg(target_os = "macos")]
    let _ = (batch, encode);
}

fn validate_j2k_transform_dispatch(batch: &EncodedTranscodeBatch) {
    #[cfg(not(target_os = "macos"))]
    {
        if !j2k_cuda_required() {
            return;
        }
        assert_eq!(
            batch.report.failed_tiles, 0,
            "j2k: CUDA transform benchmark produced failed tiles under J2K_REQUIRE_CUDA_RUNTIME=1"
        );
        assert!(
            batch.report.timings.accelerator_dispatches != 0,
            "j2k: CUDA transform dispatched zero batches under J2K_REQUIRE_CUDA_RUNTIME=1"
        );
        assert_eq!(
            batch.report.timings.cpu_fallback_jobs, 0,
            "j2k: CUDA transform used CPU fallback jobs under J2K_REQUIRE_CUDA_RUNTIME=1"
        );
        assert_eq!(
            batch.report.timings.accelerator_dispatched_jobs,
            batch.report.transformed_components,
            "j2k: CUDA transform dispatched jobs do not cover all transformed components under J2K_REQUIRE_CUDA_RUNTIME=1"
        );
        if !j2k_cuda_stage_timings_disabled() {
            assert!(
                batch.report.timings.dwt97_batch_quantize_codeblock_us != 0,
                "j2k: CUDA fused 9/7 code-block quantize stage was not timed under J2K_REQUIRE_CUDA_RUNTIME=1"
            );
        }
    }

    #[cfg(target_os = "macos")]
    let _ = batch;
}

#[derive(Default)]
struct NvidiaResult {
    ran: bool,
    best_wall_s: f64,
    decode_ms: f64,
    encode_ms: f64,
    output_bytes: usize,
    codestreams: Vec<Vec<u8>>,
    error: Option<NvBaselineError>,
}

fn run_nvidia(jpegs: &[JpegInput], config: &BenchmarkConfig) -> NvidiaResult {
    let mut session = match NvBaselineSession::new() {
        Ok(session) => session,
        Err(error) => {
            return NvidiaResult {
                error: Some(error),
                ..NvidiaResult::default()
            };
        }
    };

    // Warm up with the same reused session that will be measured.
    for _ in 0..config.warmup {
        for jpeg in jpegs {
            let _ = session.transcode_jpeg_to_htj2k(&jpeg.bytes);
        }
    }

    let mut best_wall_s = f64::INFINITY;
    let mut best = NvidiaResult::default();
    for _ in 0..config.iterations {
        let start = Instant::now();
        let mut decode_ms = 0f64;
        let mut encode_ms = 0f64;
        let mut output_bytes = 0usize;
        let mut codestreams = Vec::with_capacity(jpegs.len());
        let mut failed = None;
        for jpeg in jpegs {
            match session.transcode_jpeg_to_htj2k(&jpeg.bytes) {
                Ok(result) => {
                    decode_ms += result.decode_ms;
                    encode_ms += result.encode_ms;
                    output_bytes += result.codestream.len();
                    codestreams.push(result.codestream);
                }
                Err(error) => {
                    failed = Some(error);
                    break;
                }
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        if let Some(error) = failed {
            return NvidiaResult {
                error: Some(error),
                ..NvidiaResult::default()
            };
        }
        if elapsed < best_wall_s {
            best_wall_s = elapsed;
            best = NvidiaResult {
                ran: true,
                best_wall_s: elapsed,
                decode_ms,
                encode_ms,
                output_bytes,
                codestreams,
                error: None,
            };
        }
    }
    best
}

fn native_decode_rgb(codestream: &[u8]) -> Option<Vec<u8>> {
    let image = Image::new(codestream, &DecodeSettings::default()).ok()?;
    let bitmap = image.decode_native().ok()?;
    (bitmap.num_components == 3 && bitmap.bytes_per_sample == 1).then_some(bitmap.data)
}

#[allow(clippy::struct_field_names)]
#[derive(Clone)]
struct QualitySummary {
    mean_psnr: Option<f64>,
    aggregate_psnr: Option<f64>,
    channel_psnr: [Option<f64>; 3],
    per_tile_psnr: Vec<Option<f64>>,
}

#[derive(Clone, Copy, Debug)]
struct RgbMseSummary {
    sum_sq: f64,
    samples: usize,
    channel_sum_sq: [f64; 3],
    channel_samples: [usize; 3],
    #[cfg(test)]
    channel_psnr: [Option<f64>; 3],
}

fn quality_summary(jpegs: &[JpegInput], codestreams: &[Vec<u8>]) -> Option<QualitySummary> {
    (codestreams.len() == jpegs.len()).then_some(())?;
    let mut total = 0f64;
    let mut counted = 0usize;
    let mut total_sum_sq = 0f64;
    let mut total_samples = 0usize;
    let mut channel_sum_sq = [0f64; 3];
    let mut channel_samples = [0usize; 3];
    let mut per_tile_psnr = Vec::with_capacity(jpegs.len());
    for (jpeg, codestream) in jpegs.iter().zip(codestreams.iter()) {
        let Ok((source, _, _)) = nvidia_decode_jpeg_rgb(&jpeg.bytes) else {
            return None;
        };
        let recon = native_decode_rgb(codestream)?;
        let (psnr, mse) = best_psnr_and_mse(&recon, &source)?;
        total += psnr;
        counted += 1;
        total_sum_sq += mse.sum_sq;
        total_samples = total_samples.saturating_add(mse.samples);
        for channel in 0..3 {
            channel_sum_sq[channel] += mse.channel_sum_sq[channel];
            channel_samples[channel] =
                channel_samples[channel].saturating_add(mse.channel_samples[channel]);
        }
        per_tile_psnr.push(Some(psnr));
    }
    (counted == jpegs.len()).then(|| QualitySummary {
        mean_psnr: Some(total / counted as f64),
        aggregate_psnr: psnr_from_mse(total_sum_sq, total_samples),
        channel_psnr: [
            psnr_from_mse(channel_sum_sq[0], channel_samples[0]),
            psnr_from_mse(channel_sum_sq[1], channel_samples[1]),
            psnr_from_mse(channel_sum_sq[2], channel_samples[2]),
        ],
        per_tile_psnr,
    })
}

fn best_psnr_and_mse(recon: &[u8], source_rgb: &[u8]) -> Option<(f64, RgbMseSummary)> {
    let direct = psnr_u8(recon, source_rgb);
    let converted_rgb = ycbcr_to_rgb_round_nearest(recon);
    let converted = psnr_u8(&converted_rgb, source_rgb);
    match (direct, converted) {
        (Some(a), Some(b)) if a >= b => rgb_mse_summary(recon, source_rgb).map(|mse| (a, mse)),
        (Some(_), Some(b)) => rgb_mse_summary(&converted_rgb, source_rgb).map(|mse| (b, mse)),
        (Some(a), None) => rgb_mse_summary(recon, source_rgb).map(|mse| (a, mse)),
        (None, Some(b)) => rgb_mse_summary(&converted_rgb, source_rgb).map(|mse| (b, mse)),
        (None, None) => None,
    }
}

fn rgb_mse_summary(a: &[u8], b: &[u8]) -> Option<RgbMseSummary> {
    if a.len() != b.len() || a.is_empty() || !a.len().is_multiple_of(3) {
        return None;
    }
    let mut sum_sq = 0f64;
    let mut channel_sum_sq = [0f64; 3];
    let mut channel_samples = [0usize; 3];
    for (left, right) in a.chunks_exact(3).zip(b.chunks_exact(3)) {
        for channel in 0..3 {
            let diff = f64::from(left[channel]) - f64::from(right[channel]);
            let squared = diff * diff;
            sum_sq += squared;
            channel_sum_sq[channel] += squared;
            channel_samples[channel] += 1;
        }
    }
    Some(RgbMseSummary {
        sum_sq,
        samples: a.len(),
        channel_sum_sq,
        channel_samples,
        #[cfg(test)]
        channel_psnr: [
            psnr_from_mse(channel_sum_sq[0], channel_samples[0]),
            psnr_from_mse(channel_sum_sq[1], channel_samples[1]),
            psnr_from_mse(channel_sum_sq[2], channel_samples[2]),
        ],
    })
}

fn psnr_from_mse(sum_sq: f64, samples: usize) -> Option<f64> {
    if samples == 0 {
        return None;
    }
    if sum_sq == 0.0 {
        return Some(f64::INFINITY);
    }
    let mse = sum_sq / samples as f64;
    Some(10.0 * (255.0f64 * 255.0 / mse).log10())
}

#[allow(
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn print_report(
    jpegs: &[JpegInput],
    megapixels: f64,
    corpus_hash: u64,
    config: &BenchmarkConfig,
    rd_points: &[RdPoint],
    selected_index: usize,
    j2k_cpu_ht: &J2KResult,
    j2k_cuda_ht: &J2KResult,
    nvidia: &NvidiaResult,
) {
    let labels: Vec<&str> = jpegs.iter().map(|j| j.label.as_str()).take(4).collect();
    println!(
        "tiles: {}{}",
        labels.join(", "),
        if jpegs.len() > 4 { ", ..." } else { "" }
    );
    println!("corpus hash: {corpus_hash:016x}");
    println!(
        "lossy profile: selected scale {:.4}, decomposition levels {}, subband scales LL {:.3} HL {:.3} LH {:.3} HH {:.3}",
        rd_points
            .get(selected_index)
            .map_or(JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE, |point| point
                .scale),
        config.decomposition_levels,
        config.subband_scales.low_low,
        config.subband_scales.high_low,
        config.subband_scales.low_high,
        config.subband_scales.high_high,
    );
    println!(
        "iterations: {} (best wall-clock reported)\n",
        config.iterations
    );

    if rd_points.len() > 1 || config.match_nvidia_bytes {
        println!("J2K RD sweep (CUDA transform + CPU HT, PSNR not rate matched):");
        for (idx, point) in rd_points.iter().enumerate() {
            let selected = if idx == selected_index { "*" } else { " " };
            let delta = if nvidia.ran {
                format!(
                    "{:+.2}%",
                    byte_delta_fraction(point.result.output_bytes, nvidia.output_bytes) * 100.0
                )
            } else {
                "n/a".to_string()
            };
            println!(
                " {selected} scale {:.4}  bytes {}  vs NVIDIA {}  mean PSNR {}  agg PSNR {}  wall {:.3} ms",
                point.scale,
                point.result.output_bytes,
                delta,
                fmt_psnr(point.quality.as_ref().and_then(|q| q.mean_psnr)),
                fmt_psnr(point.quality.as_ref().and_then(|q| q.aggregate_psnr)),
                point.result.best_wall_s * 1000.0,
            );
        }
        println!();
    }

    println!(
        "{:<24}{:>18}{:>18}{:>16}",
        "metric", "sig xform+CPU HT", "sig xform+CUDA HT", "NVIDIA reused"
    );
    println!("{}", "-".repeat(76));

    // End-to-end throughput.
    let sig_cpu_mps = if j2k_cpu_ht.ran && j2k_cpu_ht.best_wall_s > 0.0 {
        megapixels / j2k_cpu_ht.best_wall_s
    } else {
        0.0
    };
    let sig_cuda_mps = if j2k_cuda_ht.ran && j2k_cuda_ht.best_wall_s > 0.0 {
        megapixels / j2k_cuda_ht.best_wall_s
    } else {
        0.0
    };
    let nv_mps = if nvidia.ran && nvidia.best_wall_s > 0.0 {
        megapixels / nvidia.best_wall_s
    } else {
        0.0
    };
    println!(
        "{:<24}{:>18}{:>18}{:>16}",
        "end-to-end MP/s",
        fmt_mps_ran(j2k_cpu_ht.ran, sig_cpu_mps),
        fmt_mps_ran(j2k_cuda_ht.ran, sig_cuda_mps),
        fmt_mps(nvidia, nv_mps),
    );

    // Wall-clock totals.
    println!(
        "{:<24}{:>18}{:>18}{:>16}",
        "wall-clock (ms)",
        fmt_ms(j2k_cpu_ht.ran, j2k_cpu_ht.best_wall_s * 1000.0),
        fmt_ms(j2k_cuda_ht.ran, j2k_cuda_ht.best_wall_s * 1000.0),
        fmt_ms(nvidia.ran, nvidia.best_wall_s * 1000.0),
    );

    // GPU-only.
    let sig_cpu_gpu_ms =
        (j2k_cpu_ht.transform_gpu_stage_us + j2k_cpu_ht.encode_cuda_stage_us) as f64 / 1000.0;
    let sig_cuda_gpu_ms =
        (j2k_cuda_ht.transform_gpu_stage_us + j2k_cuda_ht.encode_cuda_stage_us) as f64 / 1000.0;
    let nv_gpu_ms = nvidia.decode_ms + nvidia.encode_ms;
    println!(
        "{:<24}{:>18}{:>18}{:>16}",
        "GPU-only (ms)",
        fmt_ms(j2k_cpu_ht.ran && j2k_cpu_ht.used_gpu, sig_cpu_gpu_ms),
        fmt_ms(j2k_cuda_ht.ran && j2k_cuda_ht.used_gpu, sig_cuda_gpu_ms,),
        fmt_ms(nvidia.ran, nv_gpu_ms),
    );

    // Per-stage breakdown.
    println!("\nper-stage (ms):");
    print_j2k_stages("j2k CUDA transform + CPU HT encode", j2k_cpu_ht);
    print_j2k_stages(
        "j2k CUDA transform + CUDA HT block encode + CPU packetization",
        j2k_cuda_ht,
    );
    if nvidia.ran {
        println!(
            "  NVIDIA reused-session serial: nvJPEG decode {:.3}  nvJPEG2000 HT encode {:.3}",
            nvidia.decode_ms, nvidia.encode_ms
        );
    } else {
        println!("  NVIDIA reused-session serial: {}", nv_status(nvidia));
    }

    // Output size.
    println!("\noutput size + quality:");
    println!(
        "  bytes:  sig CPU HT {}   sig CUDA HT {}   NVIDIA {}",
        j2k_cpu_ht.output_bytes,
        j2k_cuda_ht.output_bytes,
        if nvidia.ran {
            nvidia.output_bytes.to_string()
        } else {
            nv_status(nvidia)
        },
    );

    // PSNR vs the nvJPEG-decoded source (best-effort; needs the NVIDIA baseline).
    let sig_cpu_quality = quality_summary(jpegs, &j2k_cpu_ht.codestreams);
    let sig_cuda_quality = quality_summary(jpegs, &j2k_cuda_ht.codestreams);
    let nv_quality = if nvidia.ran {
        quality_summary(jpegs, &nvidia.codestreams)
    } else {
        None
    };
    println!(
        "  mean PSNR vs source (dB, best color interp, not rate matched):  sig CPU HT {}   sig CUDA HT {}   NVIDIA {}",
        fmt_psnr(sig_cpu_quality.as_ref().and_then(|q| q.mean_psnr)),
        fmt_psnr(sig_cuda_quality.as_ref().and_then(|q| q.mean_psnr)),
        fmt_psnr(nv_quality.as_ref().and_then(|q| q.mean_psnr)),
    );
    println!(
        "  aggregate PSNR vs source (dB, best color interp, not rate matched):  sig CPU HT {}   sig CUDA HT {}   NVIDIA {}",
        fmt_psnr(sig_cpu_quality.as_ref().and_then(|q| q.aggregate_psnr)),
        fmt_psnr(sig_cuda_quality.as_ref().and_then(|q| q.aggregate_psnr)),
        fmt_psnr(nv_quality.as_ref().and_then(|q| q.aggregate_psnr)),
    );
    println!(
        "  RGB channel PSNR vs source (R/G/B):  sig CPU HT {}   sig CUDA HT {}   NVIDIA {}",
        fmt_channel_psnr(sig_cpu_quality.as_ref()),
        fmt_channel_psnr(sig_cuda_quality.as_ref()),
        fmt_channel_psnr(nv_quality.as_ref()),
    );
}

fn print_j2k_cuda_profile_report(
    jpegs: &[JpegInput],
    megapixels: f64,
    corpus_hash: u64,
    config: &BenchmarkConfig,
    scale: f32,
    j2k: &J2KResult,
) {
    let labels: Vec<&str> = jpegs.iter().map(|j| j.label.as_str()).take(4).collect();
    let mp_s = if j2k.ran && j2k.best_wall_s > 0.0 {
        megapixels / j2k.best_wall_s
    } else {
        0.0
    };
    println!(
        "tiles: {}{}",
        labels.join(", "),
        if jpegs.len() > 4 { ", ..." } else { "" }
    );
    println!("corpus hash: {corpus_hash:016x}");
    println!(
        "profile: j2k CUDA HT only, scale {:.4}, decomposition levels {}, subband scales LL {:.3} HL {:.3} LH {:.3} HH {:.3}",
        scale,
        config.decomposition_levels,
        config.subband_scales.low_low,
        config.subband_scales.high_low,
        config.subband_scales.low_high,
        config.subband_scales.high_high,
    );
    println!(
        "iterations: {} (best wall-clock reported)\n",
        config.iterations
    );
    println!(
        "PROFILE_RESULT j2k_cuda_ht_only mp_s={:.3} wall_ms={:.3} gpu_ms={:.3} bytes={} transform_dispatches={} transform_jobs={} cpu_fallback_jobs={} encode_dispatches={} ht_codeblock_dispatches={} packetization_dispatches={}",
        mp_s,
        j2k.best_wall_s * 1000.0,
        result_gpu_ms( j2k),
        j2k.output_bytes,
        j2k.transform_dispatches,
        j2k.transform_dispatched_jobs,
        j2k.transform_cpu_fallback_jobs,
        j2k.encode_dispatches,
        j2k.encode_ht_code_block_dispatches,
        j2k.encode_packetization_dispatches,
    );
    print_j2k_stages(
        "j2k CUDA transform + CUDA HT block encode + CPU packetization",
        j2k,
    );
}

fn print_j2k_stages(label: &str, j2k: &J2KResult) {
    println!(
        "  {label}: extract {:.3}  repack {:.3}  transform wall {:.3}  encode wall {:.3}",
        us_ms(j2k.extract_us),
        us_ms(j2k.repack_us),
        us_ms(j2k.transform_wall_us),
        us_ms(j2k.encode_wall_us),
    );
    println!(
        "    GPU transform: pack/upload {:.3}  idct+row {:.3}  column {:.3}  quantize {:.3}  readback {:.3}",
        us_ms( j2k.pack_upload_us),
        us_ms( j2k.idct_row_lift_us),
        us_ms( j2k.column_lift_us),
        us_ms( j2k.quantize_us),
        us_ms( j2k.readback_us),
    );
    println!(
        "    transform dispatches: {}  jobs: {}  CPU fallback jobs: {}",
        j2k.transform_dispatches, j2k.transform_dispatched_jobs, j2k.transform_cpu_fallback_jobs,
    );
    if j2k.encode_dispatches > 0 {
        println!(
            "    CUDA HT encode: total {:.3}  dispatches {}  HT code-block {}  packetization {}",
            us_ms(j2k.encode_cuda_stage_us),
            j2k.encode_dispatches,
            j2k.encode_ht_code_block_dispatches,
            j2k.encode_packetization_dispatches,
        );
        if j2k.encode_cuda_ht_kernel_us
            + j2k.encode_cuda_ht_status_readback_us
            + j2k.encode_cuda_ht_compact_us
            + j2k.encode_cuda_ht_output_readback_us
            > 0
        {
            println!(
                "      resident split: kernel {:.3}  status {:.3}  compact {:.3}  output {:.3}",
                us_ms(j2k.encode_cuda_ht_kernel_us),
                us_ms(j2k.encode_cuda_ht_status_readback_us),
                us_ms(j2k.encode_cuda_ht_compact_us),
                us_ms(j2k.encode_cuda_ht_output_readback_us),
            );
        }
    } else {
        println!("    CUDA HT encode: n/a (CPU encode path)");
    }
}

fn fmt_mps(nvidia: &NvidiaResult, mps: f64) -> String {
    if nvidia.ran {
        format!("{mps:>16.1}")
    } else {
        format!("{:>16}", nv_status(nvidia))
    }
}

fn fmt_mps_ran(ran: bool, mps: f64) -> String {
    if ran {
        format!("{mps:.1}")
    } else {
        "n/a".to_string()
    }
}

fn fmt_ms(ran: bool, ms: f64) -> String {
    if ran {
        format!("{ms:.3}")
    } else {
        "n/a".to_string()
    }
}

fn us_ms(us: u128) -> f64 {
    us as f64 / 1000.0
}

fn fmt_psnr(psnr: Option<f64>) -> String {
    psnr.map_or_else(|| "n/a".to_string(), |value| format!("{value:.2}"))
}

fn fmt_channel_psnr(quality: Option<&QualitySummary>) -> String {
    quality.map_or_else(
        || "n/a".to_string(),
        |quality| {
            format!(
                "{}/{}/{}",
                fmt_psnr(quality.channel_psnr[0]),
                fmt_psnr(quality.channel_psnr[1]),
                fmt_psnr(quality.channel_psnr[2])
            )
        },
    )
}

fn nv_status(nvidia: &NvidiaResult) -> String {
    if nvidia.ran {
        "ok".to_string()
    } else {
        match nvidia.error {
            Some(NvBaselineError::NotBuilt) => "n/a (not built)".to_string(),
            Some(NvBaselineError::Stage(code)) => format!("n/a (err {code})"),
            None => "n/a".to_string(),
        }
    }
}

fn result_gpu_ms(result: &J2KResult) -> f64 {
    (result.transform_gpu_stage_us + result.encode_cuda_stage_us) as f64 / 1000.0
}

fn corpus_hash(jpegs: &[JpegInput]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for jpeg in jpegs {
        fnv1a_update(&mut hash, jpeg.label.as_bytes());
        fnv1a_update(&mut hash, &jpeg.width.to_le_bytes());
        fnv1a_update(&mut hash, &jpeg.height.to_le_bytes());
        fnv1a_update(&mut hash, &jpeg.bytes);
    }
    hash
}

fn fnv1a_update(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

fn write_artifacts(
    config: &BenchmarkConfig,
    jpegs: &[JpegInput],
    megapixels: f64,
    rd_points: &[RdPoint],
    selected_index: usize,
    j2k_cuda_ht: &J2KResult,
    nvidia: &NvidiaResult,
) -> std::io::Result<()> {
    if let Some(path) = &config.csv_path {
        write_text_artifact(
            path,
            csv_report(
                jpegs,
                megapixels,
                rd_points,
                selected_index,
                j2k_cuda_ht,
                nvidia,
            ),
        )?;
    }
    if let Some(path) = &config.json_path {
        write_text_artifact(
            path,
            json_report(
                config,
                jpegs,
                megapixels,
                rd_points,
                selected_index,
                j2k_cuda_ht,
                nvidia,
            ),
        )?;
    }
    Ok(())
}

fn write_j2k_profile_artifacts(
    config: &BenchmarkConfig,
    jpegs: &[JpegInput],
    megapixels: f64,
    scale: f32,
    j2k: &J2KResult,
) -> std::io::Result<()> {
    if let Some(path) = &config.csv_path {
        write_text_artifact(path, j2k_profile_csv_report(jpegs, megapixels, scale, j2k))?;
    }
    if let Some(path) = &config.json_path {
        write_text_artifact(
            path,
            j2k_profile_json_report(config, jpegs, megapixels, scale, j2k),
        )?;
    }
    Ok(())
}

fn j2k_profile_csv_report(
    jpegs: &[JpegInput],
    megapixels: f64,
    scale: f32,
    j2k: &J2KResult,
) -> String {
    let mp_s = if j2k.ran && j2k.best_wall_s > 0.0 {
        megapixels / j2k.best_wall_s
    } else {
        0.0
    };
    format!(
        "row,ran,used_gpu,scale,tiles,megapixels,mp_s,bytes,wall_ms,gpu_ms,extract_ms,repack_ms,transform_wall_ms,encode_wall_ms,encode_cuda_ht_kernel_ms,encode_cuda_ht_status_readback_ms,encode_cuda_ht_compact_ms,encode_cuda_ht_output_readback_ms,pack_upload_ms,idct_row_lift_ms,column_lift_ms,quantize_ms,readback_ms,transform_dispatches,transform_jobs,cpu_fallback_jobs,encode_dispatches,ht_codeblock_dispatches,packetization_dispatches\nj2k_cuda_ht_only,{},{},{:.6},{},{:.8},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{}\n",
        j2k.ran,
        j2k.used_gpu,
        scale,
        jpegs.len(),
        megapixels,
        mp_s,
        j2k.output_bytes,
        j2k.best_wall_s * 1000.0,
        result_gpu_ms( j2k),
        us_ms( j2k.extract_us),
        us_ms( j2k.repack_us),
        us_ms( j2k.transform_wall_us),
        us_ms( j2k.encode_wall_us),
        us_ms( j2k.encode_cuda_ht_kernel_us),
        us_ms( j2k.encode_cuda_ht_status_readback_us),
        us_ms( j2k.encode_cuda_ht_compact_us),
        us_ms( j2k.encode_cuda_ht_output_readback_us),
        us_ms( j2k.pack_upload_us),
        us_ms( j2k.idct_row_lift_us),
        us_ms( j2k.column_lift_us),
        us_ms( j2k.quantize_us),
        us_ms( j2k.readback_us),
        j2k.transform_dispatches,
        j2k.transform_dispatched_jobs,
        j2k.transform_cpu_fallback_jobs,
        j2k.encode_dispatches,
        j2k.encode_ht_code_block_dispatches,
        j2k.encode_packetization_dispatches,
    )
}

#[allow(clippy::format_push_string)]
fn j2k_profile_json_report(
    config: &BenchmarkConfig,
    jpegs: &[JpegInput],
    megapixels: f64,
    scale: f32,
    j2k: &J2KResult,
) -> String {
    let mp_s = if j2k.ran && j2k.best_wall_s > 0.0 {
        megapixels / j2k.best_wall_s
    } else {
        0.0
    };
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"mode\": \"j2k_cuda_ht_only\",\n");
    out.push_str(&format!("  \"tile_count\": {},\n", jpegs.len()));
    out.push_str(&format!("  \"megapixels\": {megapixels:.8},\n"));
    out.push_str(&format!(
        "  \"corpus_hash\": \"{:016x}\",\n",
        corpus_hash(jpegs)
    ));
    out.push_str(&format!("  \"scale\": {scale:.6},\n"));
    out.push_str(&format!(
        "  \"decomposition_levels\": {},\n",
        config.decomposition_levels
    ));
    out.push_str(&format!(
        "  \"subband_scales\": {{\"ll\": {:.6}, \"hl\": {:.6}, \"lh\": {:.6}, \"hh\": {:.6}}},\n",
        config.subband_scales.low_low,
        config.subband_scales.high_low,
        config.subband_scales.low_high,
        config.subband_scales.high_high,
    ));
    out.push_str("  \"inputs\": [");
    for (idx, jpeg) in jpegs.iter().enumerate() {
        if idx != 0 {
            out.push_str(", ");
        }
        out.push_str(&format!(
            "{{\"label\": \"{}\", \"width\": {}, \"height\": {}, \"bytes\": {}}}",
            json_escape(&jpeg.label),
            jpeg.width,
            jpeg.height,
            jpeg.bytes.len()
        ));
    }
    out.push_str("],\n");
    out.push_str(&format!(
        "  \"result\": {{\"ran\": {}, \"used_gpu\": {}, \"mp_s\": {:.6}, \"bytes\": {}, \"wall_ms\": {:.6}, \"gpu_ms\": {:.6}, \"extract_ms\": {:.6}, \"repack_ms\": {:.6}, \"transform_wall_ms\": {:.6}, \"encode_wall_ms\": {:.6}, \"encode_cuda_ht_kernel_ms\": {:.6}, \"encode_cuda_ht_status_readback_ms\": {:.6}, \"encode_cuda_ht_compact_ms\": {:.6}, \"encode_cuda_ht_output_readback_ms\": {:.6}, \"pack_upload_ms\": {:.6}, \"idct_row_lift_ms\": {:.6}, \"column_lift_ms\": {:.6}, \"quantize_ms\": {:.6}, \"readback_ms\": {:.6}, \"transform_dispatches\": {}, \"transform_jobs\": {}, \"cpu_fallback_jobs\": {}, \"encode_dispatches\": {}, \"ht_codeblock_dispatches\": {}, \"packetization_dispatches\": {}}}\n",
        j2k.ran,
        j2k.used_gpu,
        mp_s,
        j2k.output_bytes,
        j2k.best_wall_s * 1000.0,
        result_gpu_ms( j2k),
        us_ms( j2k.extract_us),
        us_ms( j2k.repack_us),
        us_ms( j2k.transform_wall_us),
        us_ms( j2k.encode_wall_us),
        us_ms( j2k.encode_cuda_ht_kernel_us),
        us_ms( j2k.encode_cuda_ht_status_readback_us),
        us_ms( j2k.encode_cuda_ht_compact_us),
        us_ms( j2k.encode_cuda_ht_output_readback_us),
        us_ms( j2k.pack_upload_us),
        us_ms( j2k.idct_row_lift_us),
        us_ms( j2k.column_lift_us),
        us_ms( j2k.quantize_us),
        us_ms( j2k.readback_us),
        j2k.transform_dispatches,
        j2k.transform_dispatched_jobs,
        j2k.transform_cpu_fallback_jobs,
        j2k.encode_dispatches,
        j2k.encode_ht_code_block_dispatches,
        j2k.encode_packetization_dispatches,
    ));
    out.push_str("}\n");
    out
}

#[allow(clippy::format_push_string)]
fn csv_report(
    jpegs: &[JpegInput],
    megapixels: f64,
    rd_points: &[RdPoint],
    selected_index: usize,
    j2k_cuda_ht: &J2KResult,
    nvidia: &NvidiaResult,
) -> String {
    let mut out = String::from(
        "row,selected,ran,used_gpu,nvidia_ran,nvidia_status,scale,bytes,byte_delta_vs_nvidia,wall_ms,gpu_ms,mean_psnr,aggregate_psnr,psnr_r,psnr_g,psnr_b,transform_dispatches,transform_jobs,cpu_fallback_jobs,encode_dispatches,ht_codeblock_dispatches,packetization_dispatches\n",
    );
    for (idx, point) in rd_points.iter().enumerate() {
        append_csv_result(
            &mut out,
            "j2k_cuda_transform_cpu_ht",
            idx == selected_index,
            Some(point.scale),
            &point.result,
            point.quality.as_ref(),
            nvidia,
        );
    }
    let cuda_quality = quality_summary(jpegs, &j2k_cuda_ht.codestreams);
    append_csv_result(
        &mut out,
        "j2k_cuda_transform_cuda_ht_block_cpu_packet",
        false,
        rd_points.get(selected_index).map(|point| point.scale),
        j2k_cuda_ht,
        cuda_quality.as_ref(),
        nvidia,
    );
    let nvidia_wall = nvidia
        .ran
        .then(|| format!("{:.6}", nvidia.best_wall_s * 1000.0));
    let nvidia_gpu = nvidia
        .ran
        .then(|| format!("{:.6}", nvidia.decode_ms + nvidia.encode_ms));
    let nvidia_quality = if nvidia.ran {
        quality_summary(jpegs, &nvidia.codestreams)
    } else {
        None
    };
    let _ = megapixels;
    out.push_str(&format!(
        "nvidia_reused_session_serial,false,{},{},{},{},,{},{},{},{},{},{},{},{},{},,,,,,\n",
        nvidia.ran,
        nvidia.ran,
        nvidia.ran,
        csv_escape(&nv_status(nvidia)),
        if nvidia.ran {
            nvidia.output_bytes.to_string()
        } else {
            String::new()
        },
        "",
        nvidia_wall.unwrap_or_default(),
        nvidia_gpu.unwrap_or_default(),
        csv_optional_f64(
            nvidia_quality
                .as_ref()
                .and_then(|quality| quality.mean_psnr)
        ),
        csv_optional_f64(
            nvidia_quality
                .as_ref()
                .and_then(|quality| quality.aggregate_psnr)
        ),
        csv_optional_f64(
            nvidia_quality
                .as_ref()
                .and_then(|quality| quality.channel_psnr[0])
        ),
        csv_optional_f64(
            nvidia_quality
                .as_ref()
                .and_then(|quality| quality.channel_psnr[1])
        ),
        csv_optional_f64(
            nvidia_quality
                .as_ref()
                .and_then(|quality| quality.channel_psnr[2])
        ),
    ));
    out
}

#[allow(clippy::format_push_string)]
fn append_csv_result(
    out: &mut String,
    row: &str,
    selected: bool,
    scale: Option<f32>,
    result: &J2KResult,
    quality: Option<&QualitySummary>,
    nvidia: &NvidiaResult,
) {
    let byte_delta = if nvidia.ran {
        Some(byte_delta_fraction(
            result.output_bytes,
            nvidia.output_bytes,
        ))
    } else {
        None
    };
    out.push_str(&format!(
        "{row},{selected},{},{},{},{},{},{},{},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{}\n",
        result.ran,
        result.used_gpu,
        nvidia.ran,
        csv_escape(&nv_status(nvidia)),
        scale.map_or_else(String::new, |scale| format!("{scale:.6}")),
        result.output_bytes,
        byte_delta.map_or_else(String::new, |delta| format!("{delta:.8}")),
        result.best_wall_s * 1000.0,
        result_gpu_ms(result),
        csv_optional_f64(quality.and_then(|quality| quality.mean_psnr)),
        csv_optional_f64(quality.and_then(|quality| quality.aggregate_psnr)),
        csv_optional_f64(quality.and_then(|quality| quality.channel_psnr[0])),
        csv_optional_f64(quality.and_then(|quality| quality.channel_psnr[1])),
        csv_optional_f64(quality.and_then(|quality| quality.channel_psnr[2])),
        result.transform_dispatches,
        result.transform_dispatched_jobs,
        result.transform_cpu_fallback_jobs,
        result.encode_dispatches,
        result.encode_ht_code_block_dispatches,
        result.encode_packetization_dispatches,
    ));
}

#[allow(clippy::format_push_string)]
fn json_report(
    config: &BenchmarkConfig,
    jpegs: &[JpegInput],
    megapixels: f64,
    rd_points: &[RdPoint],
    selected_index: usize,
    j2k_cuda_ht: &J2KResult,
    nvidia: &NvidiaResult,
) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"tile_count\": {},\n", jpegs.len()));
    out.push_str(&format!("  \"megapixels\": {megapixels:.8},\n"));
    out.push_str(&format!(
        "  \"corpus_hash\": \"{:016x}\",\n",
        corpus_hash(jpegs)
    ));
    out.push_str(&format!(
        "  \"match_nvidia_bytes\": {},\n  \"match_tolerance\": {:.8},\n",
        config.match_nvidia_bytes, config.match_tolerance
    ));
    out.push_str(&format!(
        "  \"decomposition_levels\": {},\n",
        config.decomposition_levels
    ));
    out.push_str(&format!(
        "  \"subband_scales\": {{\"ll\": {:.6}, \"hl\": {:.6}, \"lh\": {:.6}, \"hh\": {:.6}}},\n",
        config.subband_scales.low_low,
        config.subband_scales.high_low,
        config.subband_scales.low_high,
        config.subband_scales.high_high,
    ));
    out.push_str("  \"inputs\": [");
    for (idx, jpeg) in jpegs.iter().enumerate() {
        if idx != 0 {
            out.push_str(", ");
        }
        out.push_str(&format!(
            "{{\"label\": \"{}\", \"width\": {}, \"height\": {}, \"bytes\": {}}}",
            json_escape(&jpeg.label),
            jpeg.width,
            jpeg.height,
            jpeg.bytes.len()
        ));
    }
    out.push_str("],\n");
    out.push_str(&format!("  \"selected_rd_index\": {selected_index},\n"));
    out.push_str("  \"rd_points\": [\n");
    for (idx, point) in rd_points.iter().enumerate() {
        if idx != 0 {
            out.push_str(",\n");
        }
        out.push_str("    ");
        append_json_j2k_result(
            &mut out,
            point.scale,
            &point.result,
            point.quality.as_ref(),
            nvidia,
        );
    }
    out.push_str("\n  ],\n");
    out.push_str("  \"j2k_cuda_ht_experimental\": ");
    let cuda_quality = quality_summary(jpegs, &j2k_cuda_ht.codestreams);
    append_json_j2k_result(
        &mut out,
        rd_points
            .get(selected_index)
            .map_or(JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE, |point| {
                point.scale
            }),
        j2k_cuda_ht,
        cuda_quality.as_ref(),
        nvidia,
    );
    out.push_str(",\n");
    let nvidia_quality = if nvidia.ran {
        quality_summary(jpegs, &nvidia.codestreams)
    } else {
        None
    };
    out.push_str(&format!(
        "  \"nvidia_reused_session_serial\": {{\"ran\": {}, \"status\": \"{}\", \"bytes\": {}, \"wall_ms\": {:.6}, \"gpu_ms\": {:.6}, \"decode_ms\": {:.6}, \"encode_ms\": {:.6}, \"mean_psnr\": {}, \"aggregate_psnr\": {}, \"channel_psnr\": {{\"r\": {}, \"g\": {}, \"b\": {}}}}}\n",
        nvidia.ran,
        json_escape(&nv_status(nvidia)),
        nvidia.output_bytes,
        nvidia.best_wall_s * 1000.0,
        if nvidia.ran { nvidia.decode_ms + nvidia.encode_ms } else { 0.0 },
        nvidia.decode_ms,
        nvidia.encode_ms,
        json_optional_f64(nvidia_quality.as_ref().and_then(|quality| quality.mean_psnr)),
        json_optional_f64(nvidia_quality.as_ref().and_then(|quality| quality.aggregate_psnr)),
        json_optional_f64(nvidia_quality.as_ref().and_then(|quality| quality.channel_psnr[0])),
        json_optional_f64(nvidia_quality.as_ref().and_then(|quality| quality.channel_psnr[1])),
        json_optional_f64(nvidia_quality.as_ref().and_then(|quality| quality.channel_psnr[2])),
    ));
    out.push_str("}\n");
    out
}

#[allow(clippy::format_push_string)]
fn append_json_j2k_result(
    out: &mut String,
    scale: f32,
    result: &J2KResult,
    quality: Option<&QualitySummary>,
    nvidia: &NvidiaResult,
) {
    out.push_str(&format!(
        "{{\"scale\": {:.6}, \"ran\": {}, \"used_gpu\": {}, \"bytes\": {}, \"byte_delta_vs_nvidia\": {:.8}, \"wall_ms\": {:.6}, \"gpu_ms\": {:.6}, \"mean_psnr\": {}, \"aggregate_psnr\": {}, \"channel_psnr\": {{\"r\": {}, \"g\": {}, \"b\": {}}}, \"transform_dispatches\": {}, \"transform_jobs\": {}, \"cpu_fallback_jobs\": {}, \"encode_dispatches\": {}, \"ht_codeblock_dispatches\": {}, \"packetization_dispatches\": {}, \"per_tile_psnr\": [",
        scale,
        result.ran,
        result.used_gpu,
        result.output_bytes,
        if nvidia.ran { byte_delta_fraction(result.output_bytes, nvidia.output_bytes) } else { 0.0 },
        result.best_wall_s * 1000.0,
        result_gpu_ms(result),
        json_optional_f64(quality.and_then(|quality| quality.mean_psnr)),
        json_optional_f64(quality.and_then(|quality| quality.aggregate_psnr)),
        json_optional_f64(quality.and_then(|quality| quality.channel_psnr[0])),
        json_optional_f64(quality.and_then(|quality| quality.channel_psnr[1])),
        json_optional_f64(quality.and_then(|quality| quality.channel_psnr[2])),
        result.transform_dispatches,
        result.transform_dispatched_jobs,
        result.transform_cpu_fallback_jobs,
        result.encode_dispatches,
        result.encode_ht_code_block_dispatches,
        result.encode_packetization_dispatches,
    ));
    if let Some(quality) = quality {
        for (idx, psnr) in quality.per_tile_psnr.iter().enumerate() {
            if idx != 0 {
                out.push_str(", ");
            }
            out.push_str(&json_optional_f64(*psnr));
        }
    }
    out.push_str("]}");
}

fn json_optional_f64(value: Option<f64>) -> String {
    json_f64_or_null(value, 8)
}

fn csv_optional_f64(value: Option<f64>) -> String {
    csv_f64_or_blank(value, 6)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn path(value: &str) -> PathBuf {
        PathBuf::from(value)
    }

    #[test]
    fn parse_benchmark_config_accepts_rd_and_artifact_flags() {
        let error = BenchmarkConfig::parse([
            path("--quant-scales"),
            path("1.5,1.9,2.3"),
            path("--subband-scales"),
            path("0.9,1.0,1.1,1.3"),
            path("--match-nvidia-bytes"),
            path("--profile-j2k-cuda-only"),
        ])
        .expect_err("profile-only mode rejects NVIDIA byte matching");
        assert!(error.contains("--profile-j2k-cuda-only"));

        let config = BenchmarkConfig::parse([
            path("--quant-scales"),
            path("1.5,1.9,2.3"),
            path("--subband-scales"),
            path("0.9,1.0,1.1,1.3"),
            path("--decomposition-levels"),
            path("3"),
            path("--profile-j2k-cuda-only"),
            path("--match-tolerance"),
            path("0.015"),
            path("--warmup"),
            path("3"),
            path("--iterations"),
            path("75"),
            path("--min-tiles"),
            path("100"),
            path("--json"),
            path("target/report.json"),
            path("--csv"),
            path("target/report.csv"),
            path("a.jpg"),
        ])
        .expect("config parses");

        assert_eq!(config.quant_scales, vec![1.5, 1.9, 2.3]);
        assert_eq!(config.subband_scales.low_low.to_bits(), 0.9f32.to_bits());
        assert_eq!(config.subband_scales.low_high.to_bits(), 1.1f32.to_bits());
        assert_eq!(config.decomposition_levels, 3);
        assert!(!config.match_nvidia_bytes);
        assert!(config.profile_j2k_cuda_only);
        assert_eq!(config.match_tolerance, 0.015);
        assert_eq!(config.warmup, 3);
        assert_eq!(config.iterations, 75);
        assert_eq!(config.min_tiles, Some(100));
        assert_eq!(config.json_path, Some(path("target/report.json")));
        assert_eq!(config.csv_path, Some(path("target/report.csv")));
        assert_eq!(config.input_paths, vec![path("a.jpg")]);
    }

    #[test]
    fn select_rd_point_chooses_closest_nvidia_bytes() {
        let config = BenchmarkConfig {
            match_nvidia_bytes: true,
            ..BenchmarkConfig::default()
        };
        let points = vec![
            RdPoint {
                scale: 1.5,
                result: J2KResult {
                    output_bytes: 900,
                    ..J2KResult::default()
                },
                quality: None,
            },
            RdPoint {
                scale: 1.9,
                result: J2KResult {
                    output_bytes: 1010,
                    ..J2KResult::default()
                },
                quality: None,
            },
        ];
        let nvidia = NvidiaResult {
            ran: true,
            output_bytes: 1000,
            ..NvidiaResult::default()
        };

        assert_eq!(select_rd_point(&points, &config, &nvidia), 1);
    }

    #[test]
    fn validate_rate_match_rejects_selected_rd_point_outside_tolerance() {
        let config = BenchmarkConfig {
            match_nvidia_bytes: true,
            match_tolerance: 0.02,
            ..BenchmarkConfig::default()
        };
        let nvidia = NvidiaResult {
            ran: true,
            output_bytes: 1_000,
            ..NvidiaResult::default()
        };
        let selected = J2KResult {
            ran: true,
            output_bytes: 900,
            ..J2KResult::default()
        };
        let cuda_ht = J2KResult {
            ran: true,
            used_gpu: true,
            output_bytes: 1_000,
            ..J2KResult::default()
        };

        let error = validate_rate_match(&config, &selected, &cuda_ht, &nvidia)
            .expect_err("selected RD point outside tolerance must fail");
        assert!(error.contains("selected J2K RD point"));
    }

    #[test]
    fn validate_rate_match_rejects_cuda_ht_row_outside_tolerance() {
        let config = BenchmarkConfig {
            match_nvidia_bytes: true,
            match_tolerance: 0.02,
            ..BenchmarkConfig::default()
        };
        let nvidia = NvidiaResult {
            ran: true,
            output_bytes: 1_000,
            ..NvidiaResult::default()
        };
        let selected = J2KResult {
            ran: true,
            output_bytes: 1_000,
            ..J2KResult::default()
        };
        let cuda_ht = J2KResult {
            ran: true,
            used_gpu: true,
            output_bytes: 950,
            ..J2KResult::default()
        };

        let error = validate_rate_match(&config, &selected, &cuda_ht, &nvidia)
            .expect_err("CUDA HT row outside tolerance must fail");
        assert!(error.contains("J2K CUDA HT row"));
    }

    #[test]
    fn json_optional_f64_emits_null_for_non_finite_values() {
        assert_eq!(json_optional_f64(Some(f64::INFINITY)), "null");
        assert_eq!(json_optional_f64(Some(f64::NEG_INFINITY)), "null");
        assert_eq!(json_optional_f64(Some(f64::NAN)), "null");
        assert_eq!(json_optional_f64(Some(42.25)), "42.25000000");
    }

    #[test]
    fn csv_report_marks_missing_nvidia_status_without_zero_delta() {
        let point = RdPoint {
            scale: 1.9,
            result: J2KResult {
                ran: true,
                used_gpu: true,
                output_bytes: 123,
                ..J2KResult::default()
            },
            quality: None,
        };
        let csv = csv_report(
            &[],
            0.0,
            &[point],
            0,
            &J2KResult::default(),
            &NvidiaResult {
                error: Some(NvBaselineError::NotBuilt),
                ..NvidiaResult::default()
            },
        );

        assert!(csv.starts_with(
            "row,selected,ran,used_gpu,nvidia_ran,nvidia_status,scale,bytes,byte_delta_vs_nvidia"
        ));
        assert!(csv.contains(
            "j2k_cuda_transform_cpu_ht,true,true,true,false,n/a (not built),1.900000,123,"
        ));
        assert!(!csv.contains(",123,0.00000000,"));
    }

    #[test]
    fn csv_report_marks_successful_nvidia_status_as_ok() {
        let csv = csv_report(
            &[],
            0.0,
            &[],
            0,
            &J2KResult::default(),
            &NvidiaResult {
                ran: true,
                output_bytes: 623,
                ..NvidiaResult::default()
            },
        );

        assert!(csv.contains("nvidia_reused_session_serial,false,true,true,true,ok,,623,"));
    }

    #[test]
    fn rgb_mse_summary_reports_per_channel_error() {
        let summary = rgb_mse_summary(&[10, 20, 30, 20, 40, 70], &[10, 30, 50, 30, 40, 40])
            .expect("matching RGB buffers produce channel stats");

        assert_eq!(summary.sum_sq, 1500.0);
        assert_eq!(summary.samples, 6);
        assert_eq!(summary.channel_sum_sq, [100.0, 100.0, 1300.0]);
        assert_eq!(summary.channel_samples, [2, 2, 2]);
        assert_eq!(summary.channel_psnr[0], psnr_from_mse(100.0, 2));
        assert_eq!(summary.channel_psnr[1], psnr_from_mse(100.0, 2));
        assert_eq!(summary.channel_psnr[2], psnr_from_mse(1300.0, 2));
    }

    #[test]
    fn csv_report_includes_rgb_channel_psnr_columns() {
        let csv = csv_report(
            &[],
            0.0,
            &[],
            0,
            &J2KResult::default(),
            &NvidiaResult::default(),
        );

        assert!(csv.starts_with(
            "row,selected,ran,used_gpu,nvidia_ran,nvidia_status,scale,bytes,byte_delta_vs_nvidia,wall_ms,gpu_ms,mean_psnr,aggregate_psnr,psnr_r,psnr_g,psnr_b,"
        ));
    }

    #[test]
    fn json_report_includes_rgb_channel_psnr_fields() {
        let json = json_report(
            &BenchmarkConfig::default(),
            &[],
            0.0,
            &[],
            0,
            &J2KResult::default(),
            &NvidiaResult::default(),
        );

        assert!(json.contains("\"channel_psnr\": {\"r\": null, \"g\": null, \"b\": null}"));
    }
}
