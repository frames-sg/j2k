// SPDX-License-Identifier: Apache-2.0
//
// JPEG -> HTJ2K transcode throughput comparison:
//   signinum  — coefficient-domain batch transcode (transform + optional CUDA HT encode)
//   NVIDIA    — reused-session serial nvJPEG decode + nvJPEG2000 HT encode
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

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
use signinum_j2k_cuda::CudaEncodeStageAccelerator;
#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
use signinum_j2k_native::J2kEncodeStageAccelerator;
use signinum_j2k_native::{DecodeSettings, Image};
use signinum_nvidia_baseline::{
    nvidia_decode_jpeg_rgb, psnr_u8, NvBaselineError, NvBaselineSession,
};
use signinum_transcode::{
    EncodedTranscodeBatch, JpegTileBatchInput, JpegToHtj2kError, JpegToHtj2kOptions,
    JpegToHtj2kTranscoder,
};

// The signinum GPU backend is platform-selected: Metal on macOS, CUDA elsewhere.
#[cfg(not(target_os = "macos"))]
use signinum_transcode_cuda::CudaDctToWaveletStageAccelerator as BenchAccelerator;
#[cfg(target_os = "macos")]
use signinum_transcode_metal::MetalDctToWaveletStageAccelerator as BenchAccelerator;

const BACKEND_NAME: &str = if cfg!(target_os = "macos") {
    "Metal"
} else {
    "CUDA"
};

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

    let signinum = run_signinum(&jpegs);
    let nvidia = run_nvidia(&jpegs);

    print_report(&jpegs, megapixels, &signinum, &nvidia);
    enforce_required_results(&signinum, &nvidia);
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
            "../../../signinum-transcode-cuda/tests/fixtures/conformance/baseline_420_16x16.jpg"
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

#[derive(Default)]
struct SigninumResult {
    ran: bool,
    used_gpu: bool,
    best_wall_s: f64,
    transform_gpu_stage_us: u128,
    encode_cuda_stage_us: u128,
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

#[derive(Clone, Copy, Default)]
struct EncodeBenchMetrics {
    cuda_stage_us: u128,
    total_dispatches: usize,
    ht_code_block_dispatches: usize,
    packetization_dispatches: usize,
}

fn run_signinum(jpegs: &[JpegInput]) -> SigninumResult {
    let inputs: Vec<JpegTileBatchInput<'_>> = jpegs
        .iter()
        .map(|j| JpegTileBatchInput { bytes: &j.bytes })
        .collect();
    let options = JpegToHtj2kOptions::lossy_97();

    // Warm up (and detect whether the GPU path is available).
    let mut used_gpu = true;
    let mut session = SigninumBenchSession::new(true);
    for iteration in 0..WARMUP.max(1) {
        match session.transcode_batch(&inputs, &options) {
            Ok((batch, encode_metrics)) => validate_signinum_cuda_dispatch(&batch, encode_metrics),
            Err(error) => {
                assert!(
                    !signinum_cuda_required(),
                    "signinum: explicit {BACKEND_NAME} path failed under SIGNINUM_REQUIRE_CUDA_RUNTIME=1: {error:?}"
                );
                used_gpu = false;
                if iteration == 0 {
                    eprintln!(
                        "signinum: explicit {BACKEND_NAME} path unavailable; measuring scalar CPU fallback"
                    );
                }
                session = SigninumBenchSession::new(false);
                break;
            }
        }
    }

    let mut best_wall_s = f64::INFINITY;
    let mut last = SigninumResult::default();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let batch = session.transcode_batch(&inputs, &options);
        let elapsed = start.elapsed().as_secs_f64();
        let (batch, encode_metrics) = match batch {
            Ok(batch) => batch,
            Err(error) => {
                assert!(
                    !signinum_cuda_required(),
                    "signinum: explicit {BACKEND_NAME} path failed under SIGNINUM_REQUIRE_CUDA_RUNTIME=1: {error:?}"
                );
                return SigninumResult::default();
            }
        };
        validate_signinum_cuda_dispatch(&batch, encode_metrics);
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
                transform_gpu_stage_us: t.dwt97_batch_pack_upload_us
                    + t.dwt97_batch_idct_row_lift_us
                    + t.dwt97_batch_column_lift_us
                    + t.dwt97_batch_quantize_codeblock_us
                    + t.dwt97_batch_readback_us,
                encode_cuda_stage_us: encode_metrics.cuda_stage_us,
                transform_dispatches: t.accelerator_dispatches,
                transform_dispatched_jobs: t.accelerator_dispatched_jobs,
                transform_cpu_fallback_jobs: t.cpu_fallback_jobs,
                encode_dispatches: encode_metrics.total_dispatches,
                encode_ht_code_block_dispatches: encode_metrics.ht_code_block_dispatches,
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
            };
        }
    }
    last
}

struct SigninumBenchSession {
    use_gpu: bool,
    transcoder: JpegToHtj2kTranscoder,
    transform_accelerator: Option<BenchAccelerator>,
    #[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
    encode_accelerator: Option<CudaEncodeStageAccelerator>,
}

impl SigninumBenchSession {
    fn new(use_gpu: bool) -> Self {
        Self {
            use_gpu,
            transcoder: JpegToHtj2kTranscoder::default(),
            transform_accelerator: use_gpu.then(BenchAccelerator::new_explicit),
            #[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
            encode_accelerator: use_gpu
                .then(|| CudaEncodeStageAccelerator::with_profile_collection(true)),
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
                .and_then(reject_failed_signinum_tiles)
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
            .expect("GPU signinum session has a transform accelerator");
        self.transcoder
            .transcode_batch_with_accelerator(inputs, options, accelerator)
            .and_then(reject_failed_signinum_tiles)
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
            .expect("GPU signinum session has a transform accelerator");
        let encode_accelerator = self
            .encode_accelerator
            .as_mut()
            .expect("CUDA signinum session has an encode accelerator");
        let before = encode_accelerator.dispatch_report();
        encode_accelerator.reset_collected_stage_timings();
        let batch = self.transcoder.transcode_batch_with_accelerators(
            inputs,
            options,
            transform_accelerator,
            encode_accelerator,
        )?;
        let batch = reject_failed_signinum_tiles(batch)?;
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
            .expect("GPU signinum session has a transform accelerator");
        self.transcoder
            .transcode_batch_with_accelerator(inputs, options, accelerator)
            .and_then(reject_failed_signinum_tiles)
            .map(|batch| (batch, EncodeBenchMetrics::default()))
    }
}

fn reject_failed_signinum_tiles(
    batch: EncodedTranscodeBatch,
) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
    if batch.report.failed_tiles == 0 {
        Ok(batch)
    } else {
        Err(JpegToHtj2kError::Validation(
            "signinum benchmark produced one or more failed tiles",
        ))
    }
}

fn signinum_cuda_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
}

fn nvidia_baseline_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_NV_BASELINE_BUILD").is_some()
}

fn enforce_required_results(signinum: &SigninumResult, nvidia: &NvidiaResult) {
    let mut failed = false;
    if signinum_cuda_required() && !(signinum.ran && signinum.used_gpu) {
        eprintln!("signinum: required CUDA benchmark did not produce a GPU result");
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

fn validate_signinum_cuda_dispatch(batch: &EncodedTranscodeBatch, encode: EncodeBenchMetrics) {
    #[cfg(not(target_os = "macos"))]
    {
        if !signinum_cuda_required() {
            return;
        }
        assert!(
            batch.report.timings.accelerator_dispatches != 0,
            "signinum: CUDA transform dispatched zero batches under SIGNINUM_REQUIRE_CUDA_RUNTIME=1"
        );
        assert!(
            encode.ht_code_block_dispatches != 0,
            "signinum: CUDA HT encode dispatched zero code-block batches under SIGNINUM_REQUIRE_CUDA_RUNTIME=1"
        );
    }

    #[cfg(target_os = "macos")]
    let _ = (batch, encode);
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
    for _ in 0..WARMUP {
        for jpeg in jpegs {
            let _ = session.transcode_jpeg_to_htj2k(&jpeg.bytes);
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
        let recon = native_decode_rgb(codestream)?;
        if let Some(psnr) = best_psnr(&recon, &source) {
            total += psnr;
            counted += 1;
        } else {
            return None;
        }
    }
    (counted == jpegs.len()).then(|| total / counted as f64)
}

/// PSNR of a reconstruction vs the RGB source, taking the consistent color
/// interpretation. NVIDIA's codestream is RGB (MCT); signinum's coefficient-
/// domain transcode keeps the JPEG's YCbCr — so try both and keep the higher,
/// which corrects the color-space mismatch without hiding real quality loss
/// (genuine degradation lowers both interpretations).
fn best_psnr(recon: &[u8], source_rgb: &[u8]) -> Option<f64> {
    let direct = psnr_u8(recon, source_rgb);
    let converted = psnr_u8(&ycbcr_to_rgb(recon), source_rgb);
    match (direct, converted) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

/// JFIF full-range YCbCr -> RGB, interleaved.
fn ycbcr_to_rgb(ycbcr: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(ycbcr.len());
    for px in ycbcr.chunks_exact(3) {
        let y = f32::from(px[0]);
        let cb = f32::from(px[1]) - 128.0;
        let cr = f32::from(px[2]) - 128.0;
        rgb.push((y + 1.402 * cr).clamp(0.0, 255.0).round() as u8);
        rgb.push(
            (y - 0.344_136 * cb - 0.714_136 * cr)
                .clamp(0.0, 255.0)
                .round() as u8,
        );
        rgb.push((y + 1.772 * cb).clamp(0.0, 255.0).round() as u8);
    }
    rgb
}

#[allow(clippy::similar_names)]
fn print_report(
    jpegs: &[JpegInput],
    megapixels: f64,
    signinum: &SigninumResult,
    nvidia: &NvidiaResult,
) {
    let labels: Vec<&str> = jpegs.iter().map(|j| j.label.as_str()).take(4).collect();
    println!(
        "tiles: {}{}",
        labels.join(", "),
        if jpegs.len() > 4 { ", ..." } else { "" }
    );
    println!("iterations: {ITERATIONS} (best wall-clock reported)\n");

    println!("{:<22}{:>16}{:>16}", "metric", "signinum", "NVIDIA reused");
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
    let sig_gpu_ms =
        (signinum.transform_gpu_stage_us + signinum.encode_cuda_stage_us) as f64 / 1000.0;
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
        "  signinum {BACKEND_NAME} transform: pack/upload {:.3}  idct+row {:.3}  column {:.3}  quantize {:.3}  readback {:.3}",
        us_ms(signinum.pack_upload_us),
        us_ms(signinum.idct_row_lift_us),
        us_ms(signinum.column_lift_us),
        us_ms(signinum.quantize_us),
        us_ms(signinum.readback_us),
    );
    println!(
        "    transform dispatches: {}  jobs: {}  CPU fallback jobs: {}",
        signinum.transform_dispatches,
        signinum.transform_dispatched_jobs,
        signinum.transform_cpu_fallback_jobs,
    );
    if signinum.encode_dispatches > 0 {
        println!(
            "  signinum CUDA HT encode: total {:.3}  dispatches {}  HT code-block {}  packetization {}",
            us_ms(signinum.encode_cuda_stage_us),
            signinum.encode_dispatches,
            signinum.encode_ht_code_block_dispatches,
            signinum.encode_packetization_dispatches,
        );
    } else {
        println!("  signinum CUDA HT encode: n/a (CPU encode path)");
    }
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
        "  bytes:  signinum {}   NVIDIA {}",
        signinum.output_bytes,
        if nvidia.ran {
            nvidia.output_bytes.to_string()
        } else {
            nv_status(nvidia)
        },
    );

    // PSNR vs the nvJPEG-decoded source (best-effort; needs the NVIDIA baseline).
    let sig_psnr = mean_psnr(jpegs, &signinum.codestreams);
    let nv_psnr = if nvidia.ran {
        mean_psnr(jpegs, &nvidia.codestreams)
    } else {
        None
    };
    println!(
        "  PSNR vs source (dB, best color interp, not rate matched):  signinum {}   NVIDIA {}",
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
