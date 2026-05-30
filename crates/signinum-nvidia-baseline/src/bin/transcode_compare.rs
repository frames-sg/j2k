// SPDX-License-Identifier: Apache-2.0
//
// JPEG -> HTJ2K transcode throughput comparison:
//   signinum  — coefficient-domain batch transcode (the CUDA code-block path)
//   NVIDIA    — nvJPEG decode (JPEG -> RGB) + nvJPEG2000 HT encode (RGB -> HTJ2K)
//
// Reports the four metric families requested: end-to-end throughput (MP/s),
// per-stage breakdown, output size + PSNR, and GPU-only vs wall-clock.
//
// Usage:
//   transcode_compare <file.jpg> [more.jpg ...]
//   transcode_compare              # falls back to SIGNINUM_BENCH_JPEG_DIR/*.jpg
//
// The NVIDIA columns show "n/a (not built)" unless this crate was compiled with
// --features nvjpeg2000 on a host with nvcc + libnvjpeg + libnvjpeg2k.

use std::time::Instant;

use signinum_j2k_native::{DecodeSettings, Image};
use signinum_nvidia_baseline::{
    nvidia_baseline_available, nvidia_decode_jpeg_rgb, nvidia_transcode_jpeg_to_htj2k, psnr_u8,
    NvBaselineError,
};
use signinum_transcode::{
    JpegTileBatchInput, JpegToHtj2kOptions, JpegToHtj2kTranscoder,
};
use signinum_transcode_cuda::CudaDctToWaveletStageAccelerator;

const WARMUP: usize = 2;
const ITERATIONS: usize = 10;

fn main() {
    let jpegs = match load_inputs() {
        Ok(jpegs) if !jpegs.is_empty() => jpegs,
        Ok(_) => {
            eprintln!("no JPEG inputs found; pass file paths or set SIGNINUM_BENCH_JPEG_DIR");
            std::process::exit(2);
        }
        Err(error) => {
            eprintln!("failed to load inputs: {error}");
            std::process::exit(2);
        }
    };

    let total_pixels: u64 = jpegs.iter().map(|j| u64::from(j.width) * u64::from(j.height)).sum();
    let megapixels = total_pixels as f64 / 1.0e6;
    println!(
        "inputs: {} tile(s), {:.2} MP total\n",
        jpegs.len(),
        megapixels
    );

    let signinum = run_signinum(&jpegs);
    let nvidia = run_nvidia(&jpegs);

    print_report(&jpegs, megapixels, &signinum, &nvidia);
}

struct JpegInput {
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    label: String,
}

fn load_inputs() -> std::io::Result<Vec<JpegInput>> {
    let mut paths: Vec<std::path::PathBuf> = std::env::args_os().skip(1).map(Into::into).collect();
    if paths.is_empty() {
        if let Some(dir) = std::env::var_os("SIGNINUM_BENCH_JPEG_DIR") {
            for entry in std::fs::read_dir(dir)? {
                let path = entry?.path();
                if path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg")) {
                    paths.push(path);
                }
            }
            paths.sort();
        }
    }
    if paths.is_empty() {
        // Bundled sanity fixture (tiny/unrepresentative — for a smoke test only).
        let bytes =
            include_bytes!("../../../signinum-transcode-cuda/tests/fixtures/conformance/baseline_420_16x16.jpg")
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
        let (width, height) = jpeg_dimensions(&bytes).unwrap_or((0, 0));
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
            return Some((width, height));
        }
        if marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
            i += 2;
        } else {
            i += 2 + len;
        }
    }
    None
}

#[derive(Default)]
struct SigninumResult {
    ran: bool,
    used_gpu: bool,
    best_wall_s: f64,
    gpu_stage_us: u128,
    pack_upload_us: u128,
    idct_row_lift_us: u128,
    column_lift_us: u128,
    quantize_us: u128,
    readback_us: u128,
    output_bytes: usize,
    codestreams: Vec<Vec<u8>>,
}

fn run_signinum(jpegs: &[JpegInput]) -> SigninumResult {
    let inputs: Vec<JpegTileBatchInput<'_>> =
        jpegs.iter().map(|j| JpegTileBatchInput { bytes: &j.bytes }).collect();
    let options = JpegToHtj2kOptions::lossy_97();

    // Warm up (and detect whether the GPU path is available).
    let mut used_gpu = true;
    for iteration in 0..WARMUP.max(1) {
        let mut accelerator = CudaDctToWaveletStageAccelerator::new_explicit();
        let mut transcoder = JpegToHtj2kTranscoder::default();
        if transcoder
            .transcode_batch_with_accelerator(&inputs, &options, &mut accelerator)
            .is_err()
        {
            used_gpu = false;
            if iteration == 0 {
                eprintln!("signinum: explicit CUDA path unavailable; measuring scalar CPU fallback");
            }
            break;
        }
    }

    let mut best_wall_s = f64::INFINITY;
    let mut last = SigninumResult::default();
    for _ in 0..ITERATIONS {
        let mut accelerator = CudaDctToWaveletStageAccelerator::new_explicit();
        let mut transcoder = JpegToHtj2kTranscoder::default();
        let start = Instant::now();
        let batch = if used_gpu {
            transcoder
                .transcode_batch_with_accelerator(&inputs, &options, &mut accelerator)
                .ok()
        } else {
            transcoder.transcode_batch(&inputs, &options).ok()
        };
        let elapsed = start.elapsed().as_secs_f64();
        let Some(batch) = batch else {
            return SigninumResult::default();
        };
        if elapsed < best_wall_s {
            best_wall_s = elapsed;
            let t = &batch.report.timings;
            last = SigninumResult {
                ran: true,
                used_gpu,
                best_wall_s: elapsed,
                pack_upload_us: t.dwt97_batch_pack_upload_us,
                idct_row_lift_us: t.dwt97_batch_idct_row_lift_us,
                column_lift_us: t.dwt97_batch_column_lift_us,
                quantize_us: t.dwt97_batch_quantize_codeblock_us,
                readback_us: t.dwt97_batch_readback_us,
                gpu_stage_us: t.dwt97_batch_pack_upload_us
                    + t.dwt97_batch_idct_row_lift_us
                    + t.dwt97_batch_column_lift_us
                    + t.dwt97_batch_quantize_codeblock_us
                    + t.dwt97_batch_readback_us,
                output_bytes: batch.tiles.iter().flatten().map(|tile| tile.codestream.len()).sum(),
                codestreams: batch
                    .tiles
                    .iter()
                    .flatten()
                    .map(|tile| tile.codestream.clone())
                    .collect(),
            };
        }
    }
    last
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

fn run_nvidia(jpegs: &[JpegInput]) -> NvidiaResult {
    if !nvidia_baseline_available() {
        return NvidiaResult {
            error: Some(NvBaselineError::NotBuilt),
            ..NvidiaResult::default()
        };
    }

    // Warm up.
    for _ in 0..WARMUP {
        for jpeg in jpegs {
            let _ = nvidia_transcode_jpeg_to_htj2k(&jpeg.bytes);
        }
    }

    let mut best_wall_s = f64::INFINITY;
    let mut best = NvidiaResult::default();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let mut decode_ms = 0f64;
        let mut encode_ms = 0f64;
        let mut output_bytes = 0usize;
        let mut codestreams = Vec::with_capacity(jpegs.len());
        let mut failed = None;
        for jpeg in jpegs {
            match nvidia_transcode_jpeg_to_htj2k(&jpeg.bytes) {
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

/// Mean PSNR (dB) of each codestream's reconstruction vs the nvJPEG source RGB.
fn mean_psnr(jpegs: &[JpegInput], codestreams: &[Vec<u8>]) -> Option<f64> {
    if codestreams.len() != jpegs.len() {
        return None;
    }
    let mut total = 0f64;
    let mut counted = 0usize;
    for (jpeg, codestream) in jpegs.iter().zip(codestreams.iter()) {
        let Ok((source, _, _)) = nvidia_decode_jpeg_rgb(&jpeg.bytes) else {
            return None;
        };
        let Some(recon) = native_decode_rgb(codestream) else {
            continue;
        };
        if let Some(psnr) = psnr_u8(&recon, &source) {
            if psnr.is_finite() {
                total += psnr;
                counted += 1;
            }
        }
    }
    (counted > 0).then(|| total / counted as f64)
}

#[allow(clippy::similar_names)]
fn print_report(
    jpegs: &[JpegInput],
    megapixels: f64,
    signinum: &SigninumResult,
    nvidia: &NvidiaResult,
) {
    let labels: Vec<&str> = jpegs.iter().map(|j| j.label.as_str()).take(4).collect();
    println!("tiles: {}{}", labels.join(", "), if jpegs.len() > 4 { ", ..." } else { "" });
    println!("iterations: {ITERATIONS} (best wall-clock reported)\n");

    println!("{:<22}{:>16}{:>16}", "metric", "signinum", "NVIDIA");
    println!("{}", "-".repeat(54));

    // End-to-end throughput.
    let sig_mps = if signinum.ran && signinum.best_wall_s > 0.0 {
        megapixels / signinum.best_wall_s
    } else {
        0.0
    };
    let nv_mps = if nvidia.ran && nvidia.best_wall_s > 0.0 {
        megapixels / nvidia.best_wall_s
    } else {
        0.0
    };
    let sig_role = if signinum.used_gpu { "GPU" } else { "CPU" };
    println!(
        "{:<22}{:>13.1} ({}){:>16}",
        "end-to-end MP/s",
        sig_mps,
        sig_role,
        fmt_mps(nvidia, nv_mps),
    );

    // Wall-clock totals.
    println!(
        "{:<22}{:>16}{:>16}",
        "wall-clock (ms)",
        fmt_ms(signinum.ran, signinum.best_wall_s * 1000.0),
        fmt_ms(nvidia.ran, nvidia.best_wall_s * 1000.0),
    );

    // GPU-only.
    let sig_gpu_ms = signinum.gpu_stage_us as f64 / 1000.0;
    let nv_gpu_ms = nvidia.decode_ms + nvidia.encode_ms;
    println!(
        "{:<22}{:>16}{:>16}",
        "GPU-only (ms)",
        fmt_ms(signinum.ran && signinum.used_gpu, sig_gpu_ms),
        fmt_ms(nvidia.ran, nv_gpu_ms),
    );

    // Per-stage breakdown.
    println!("\nper-stage (ms):");
    println!(
        "  signinum: pack/upload {:.3}  idct+row {:.3}  column {:.3}  quantize {:.3}  readback {:.3}",
        us_ms(signinum.pack_upload_us),
        us_ms(signinum.idct_row_lift_us),
        us_ms(signinum.column_lift_us),
        us_ms(signinum.quantize_us),
        us_ms(signinum.readback_us),
    );
    if nvidia.ran {
        println!(
            "  NVIDIA:   nvJPEG decode {:.3}  nvJPEG2000 HT encode {:.3}",
            nvidia.decode_ms, nvidia.encode_ms
        );
    } else {
        println!("  NVIDIA:   {}", nv_status(nvidia));
    }

    // Output size.
    println!("\noutput size + quality:");
    println!(
        "  bytes:  signinum {}   NVIDIA {}",
        signinum.output_bytes,
        if nvidia.ran { nvidia.output_bytes.to_string() } else { nv_status(nvidia) },
    );

    // PSNR vs the nvJPEG-decoded source (best-effort; needs the NVIDIA baseline).
    let sig_psnr = mean_psnr(jpegs, &signinum.codestreams);
    let nv_psnr = if nvidia.ran { mean_psnr(jpegs, &nvidia.codestreams) } else { None };
    println!(
        "  PSNR vs source (dB):  signinum {}   NVIDIA {}",
        fmt_psnr(sig_psnr),
        fmt_psnr(nv_psnr),
    );
}

fn fmt_mps(nvidia: &NvidiaResult, mps: f64) -> String {
    if nvidia.ran {
        format!("{mps:>16.1}")
    } else {
        format!("{:>16}", nv_status(nvidia))
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

fn nv_status(nvidia: &NvidiaResult) -> String {
    match nvidia.error {
        Some(NvBaselineError::NotBuilt) => "n/a (not built)".to_string(),
        Some(NvBaselineError::Stage(code)) => format!("n/a (err {code})"),
        None => "n/a".to_string(),
    }
}
