// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::format_push_string,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::print_stdout,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
use signinum_core::{BackendKind, DeviceSurface, PixelFormat};
use signinum_j2k_native::{encode_htj2k, DecodeSettings, EncodeOptions, Image};
use signinum_nvidia_baseline::{
    nvidia_j2k_decode_available, psnr_u8, NvBaselineError, NvBaselineSession, NvJ2kDecodeFormat,
};

const DEFAULT_FIXTURE_DIM: u32 = 512;
const DEFAULT_WARMUP: usize = 2;
const DEFAULT_ITERATIONS: usize = 10;

fn main() {
    let config = match Config::from_env_args() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };

    let inputs = match load_inputs(&config) {
        Ok(inputs) if !inputs.is_empty() => inputs,
        Ok(_) => {
            eprintln!("no JPEG 2000 inputs found");
            std::process::exit(2);
        }
        Err(error) => {
            eprintln!("failed to load inputs: {error}");
            std::process::exit(2);
        }
    };
    if inputs.len() < config.min_inputs {
        eprintln!(
            "input corpus has {} codestream(s), below required --min-inputs {}",
            inputs.len(),
            config.min_inputs
        );
        std::process::exit(2);
    }

    let require_nvidia = std::env::var_os("SIGNINUM_REQUIRE_NV_BASELINE_BUILD").is_some();
    if require_nvidia && !nvidia_j2k_decode_available() {
        eprintln!("SIGNINUM_REQUIRE_NV_BASELINE_BUILD set, but nvJPEG2000 decode is unavailable");
        std::process::exit(1);
    }

    let rows = run_comparison(&inputs, &config);
    print_report(&rows, &config);
    if let Err(error) = write_artifacts(&rows, &config) {
        eprintln!("failed to write decode comparison artifacts: {error}");
        std::process::exit(2);
    }

    if require_nvidia && rows.iter().any(Row::has_required_failure) {
        eprintln!("required direct nvJPEG2000 decode comparison failed");
        std::process::exit(1);
    }
}

#[derive(Debug, Clone)]
struct Config {
    inputs: Vec<PathBuf>,
    jpeg_dir: Option<PathBuf>,
    json: Option<PathBuf>,
    csv: Option<PathBuf>,
    fixture_dim: u32,
    warmup: usize,
    iterations: usize,
    min_inputs: usize,
    max_inputs: Option<usize>,
}

impl Config {
    fn from_env_args() -> Result<Self, String> {
        Self::from_args(std::env::args().skip(1))
    }

    fn from_args<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut config = Self {
            inputs: Vec::new(),
            jpeg_dir: None,
            json: None,
            csv: None,
            fixture_dim: DEFAULT_FIXTURE_DIM,
            warmup: DEFAULT_WARMUP,
            iterations: DEFAULT_ITERATIONS,
            min_inputs: 1,
            max_inputs: None,
        };
        let mut iter = args.into_iter().map(Into::into);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--json" => config.json = Some(next_path(&mut iter, "--json")?),
                "--csv" => config.csv = Some(next_path(&mut iter, "--csv")?),
                "--jpeg-dir" => config.jpeg_dir = Some(next_path(&mut iter, "--jpeg-dir")?),
                "--fixture-dim" => {
                    config.fixture_dim = next_parse(&mut iter, "--fixture-dim")?;
                    if config.fixture_dim == 0 {
                        return Err("--fixture-dim must be > 0".to_string());
                    }
                }
                "--warmup" => config.warmup = next_parse(&mut iter, "--warmup")?,
                "--iterations" => {
                    config.iterations = next_parse(&mut iter, "--iterations")?;
                    if config.iterations == 0 {
                        return Err("--iterations must be > 0".to_string());
                    }
                }
                "--min-inputs" => config.min_inputs = next_parse(&mut iter, "--min-inputs")?,
                "--max-inputs" => {
                    let max_inputs = next_parse(&mut iter, "--max-inputs")?;
                    if max_inputs == 0 {
                        return Err("--max-inputs must be > 0".to_string());
                    }
                    config.max_inputs = Some(max_inputs);
                }
                "-h" | "--help" => return Err(usage()),
                other if other.starts_with('-') => {
                    return Err(format!("unknown flag `{other}`\n{}", usage()))
                }
                path => config.inputs.push(PathBuf::from(path)),
            }
        }
        if config.min_inputs == 0 {
            return Err("--min-inputs must be > 0".to_string());
        }
        Ok(config)
    }
}

fn next_path(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf, String> {
    iter.next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{flag} requires a path"))
}

fn next_parse<T>(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    let value = iter
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))?;
    value
        .parse()
        .map_err(|_| format!("{flag} has invalid value `{value}`"))
}

fn usage() -> String {
    "usage: decode_compare [--fixture-dim n] [--jpeg-dir path] [--warmup n] [--iterations n] [--min-inputs n] [--max-inputs n] [--json path] [--csv path] [file.j2k ...]".to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeCaseFormat {
    Gray8,
    Rgb8,
}

impl DecodeCaseFormat {
    const fn components(self) -> usize {
        match self {
            Self::Gray8 => 1,
            Self::Rgb8 => 3,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Gray8 => "gray8",
            Self::Rgb8 => "rgb8",
        }
    }

    const fn nvidia(self) -> NvJ2kDecodeFormat {
        match self {
            Self::Gray8 => NvJ2kDecodeFormat::Gray8,
            Self::Rgb8 => NvJ2kDecodeFormat::Rgb8,
        }
    }

    #[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
    const fn signinum(self) -> PixelFormat {
        match self {
            Self::Gray8 => PixelFormat::Gray8,
            Self::Rgb8 => PixelFormat::Rgb8,
        }
    }
}

#[derive(Debug, Clone)]
struct DecodeInput {
    label: String,
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    format: DecodeCaseFormat,
}

fn load_inputs(config: &Config) -> Result<Vec<DecodeInput>, String> {
    if let Some(jpeg_dir) = &config.jpeg_dir {
        return load_nvidia_htj2k_from_jpeg_dir(jpeg_dir, config.max_inputs);
    }
    if config.inputs.is_empty() {
        return generated_inputs(config.fixture_dim);
    }
    let paths = config
        .inputs
        .iter()
        .take(config.max_inputs.unwrap_or(usize::MAX));
    paths.map(|path| load_input_path(path)).collect()
}

fn load_nvidia_htj2k_from_jpeg_dir(
    jpeg_dir: &Path,
    max_inputs: Option<usize>,
) -> Result<Vec<DecodeInput>, String> {
    let mut jpeg_paths = fs::read_dir(jpeg_dir)
        .map_err(|error| format!("{}: {error}", jpeg_dir.display()))?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let is_jpeg = path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg")
                });
            is_jpeg.then_some(path)
        })
        .collect::<Vec<_>>();
    jpeg_paths.sort();
    if let Some(max_inputs) = max_inputs {
        jpeg_paths.truncate(max_inputs);
    }
    if jpeg_paths.is_empty() {
        return Err(format!("{}: no JPEG inputs found", jpeg_dir.display()));
    }

    let mut session = NvBaselineSession::new()
        .map_err(|error| format!("NVIDIA baseline session required for --jpeg-dir: {error:?}"))?;
    let mut inputs = Vec::with_capacity(jpeg_paths.len());
    for path in jpeg_paths {
        let jpeg = fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let encoded = session
            .transcode_jpeg_to_htj2k(&jpeg)
            .map_err(|error| format!("{} NVIDIA HTJ2K transcode: {error:?}", path.display()))?;
        if encoded.num_components != 3 {
            return Err(format!(
                "{} NVIDIA HTJ2K transcode returned {} component(s), expected RGB",
                path.display(),
                encoded.num_components
            ));
        }
        let label = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("input.jpg");
        inputs.push(DecodeInput {
            label: format!("nvidia_htj2k:{label}"),
            bytes: encoded.codestream,
            width: encoded.width,
            height: encoded.height,
            format: DecodeCaseFormat::Rgb8,
        });
    }
    Ok(inputs)
}

fn generated_inputs(dim: u32) -> Result<Vec<DecodeInput>, String> {
    let gray = (0..dim * dim)
        .map(|idx| u8::try_from((idx * 17 + idx / 3) & 0xff).expect("masked sample fits"))
        .collect::<Vec<_>>();
    let mut rgb = Vec::with_capacity(dim as usize * dim as usize * 3);
    for idx in 0..dim * dim {
        rgb.push(u8::try_from((idx * 17 + idx / 3) & 0xff).expect("masked red fits"));
        rgb.push(u8::try_from((idx * 29 + 7) & 0xff).expect("masked green fits"));
        rgb.push(u8::try_from((idx * 43 + 19) & 0xff).expect("masked blue fits"));
    }
    Ok(vec![
        encode_input("generated_gray8", dim, dim, DecodeCaseFormat::Gray8, &gray)?,
        encode_input("generated_rgb8", dim, dim, DecodeCaseFormat::Rgb8, &rgb)?,
    ])
}

fn encode_input(
    label: &str,
    width: u32,
    height: u32,
    format: DecodeCaseFormat,
    pixels: &[u8],
) -> Result<DecodeInput, String> {
    let options = EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(
        pixels,
        width,
        height,
        u8::try_from(format.components()).expect("component count fits"),
        8,
        false,
        &options,
    )
    .map_err(|error| format!("encode {label}: {error}"))?;
    Ok(DecodeInput {
        label: label.to_string(),
        bytes,
        width,
        height,
        format,
    })
}

fn load_input_path(path: &Path) -> Result<DecodeInput, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let image = Image::new(&bytes, &DecodeSettings::default())
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let bitmap = image
        .decode_native()
        .map_err(|error| format!("{} decode: {error}", path.display()))?;
    let format = match (bitmap.num_components, bitmap.bytes_per_sample) {
        (1, 1) => DecodeCaseFormat::Gray8,
        (3, 1) => DecodeCaseFormat::Rgb8,
        _ => {
            return Err(format!(
                "{}: only 8-bit gray/RGB direct decode comparison is supported, got {} components and {} byte(s)/sample",
                path.display(),
                bitmap.num_components,
                bitmap.bytes_per_sample
            ));
        }
    };
    Ok(DecodeInput {
        label: path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("input")
            .to_string(),
        bytes,
        width: bitmap.width,
        height: bitmap.height,
        format,
    })
}

#[derive(Debug, Clone)]
struct TimedPixels {
    wall_ms: f64,
    gpu_ms: Option<f64>,
    stage_ms: Option<f64>,
    pixels: Vec<u8>,
}

#[derive(Debug, Clone)]
struct TimedStatus {
    status: String,
    result: Option<TimedPixels>,
}

impl TimedStatus {
    fn ok(result: TimedPixels) -> Self {
        Self {
            status: "ok".to_string(),
            result: Some(result),
        }
    }

    fn failed(status: impl Into<String>) -> Self {
        Self {
            status: status.into(),
            result: None,
        }
    }

    fn has_failure(&self) -> bool {
        self.status != "ok"
    }
}

#[derive(Debug, Clone)]
struct Row {
    label: String,
    format: DecodeCaseFormat,
    width: u32,
    height: u32,
    codestream_bytes: usize,
    cpu: TimedStatus,
    signinum_cuda: TimedStatus,
    nvidia: TimedStatus,
    signinum_cuda_psnr_vs_cpu: Option<f64>,
    nvidia_psnr_vs_cpu: Option<f64>,
}

impl Row {
    fn megapixels(&self) -> f64 {
        f64::from(self.width) * f64::from(self.height) / 1.0e6
    }

    fn has_required_failure(&self) -> bool {
        self.cpu.has_failure() || self.signinum_cuda.has_failure() || self.nvidia.has_failure()
    }
}

fn run_comparison(inputs: &[DecodeInput], config: &Config) -> Vec<Row> {
    let mut rows = Vec::with_capacity(inputs.len());
    let cpu_results = inputs
        .iter()
        .map(|input| timed_best(config, || decode_cpu(input)))
        .collect::<Vec<_>>();

    // Keep the nvJPEG2000 decode session isolated from Signinum's CUDA Driver
    // context. Some CUDA runtime/nvJPEG2000 stacks fail decode after another
    // Driver API context has been made current in the same process.
    let mut nvidia_session = NvBaselineSession::new().ok();
    let nvidia_results = inputs
        .iter()
        .map(|input| match nvidia_session.as_mut() {
            Some(session) => timed_best(config, || decode_nvidia(session, input)),
            None => TimedStatus::failed(nvidia_unavailable_status()),
        })
        .collect::<Vec<_>>();

    #[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
    let mut signinum_cuda_session = signinum_j2k_cuda::CudaSession::default();
    for ((input, cpu), nvidia) in inputs.iter().zip(cpu_results).zip(nvidia_results) {
        #[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
        let signinum_cuda = timed_best(config, || {
            decode_signinum_cuda(&mut signinum_cuda_session, input)
        });
        #[cfg(any(target_os = "macos", not(feature = "nvjpeg2000")))]
        let signinum_cuda = timed_best(config, || decode_signinum_cuda(input));
        let cpu_pixels = cpu.result.as_ref().map(|result| result.pixels.as_slice());
        let signinum_cuda_psnr_vs_cpu = cpu_pixels
            .zip(
                signinum_cuda
                    .result
                    .as_ref()
                    .map(|result| result.pixels.as_slice()),
            )
            .and_then(|(cpu, cuda)| psnr_u8(cpu, cuda));
        let nvidia_psnr_vs_cpu = cpu_pixels
            .zip(
                nvidia
                    .result
                    .as_ref()
                    .map(|result| result.pixels.as_slice()),
            )
            .and_then(|(cpu, nv)| psnr_u8(cpu, nv));
        rows.push(Row {
            label: input.label.clone(),
            format: input.format,
            width: input.width,
            height: input.height,
            codestream_bytes: input.bytes.len(),
            cpu,
            signinum_cuda,
            nvidia,
            signinum_cuda_psnr_vs_cpu,
            nvidia_psnr_vs_cpu,
        });
    }
    rows
}

fn nvidia_unavailable_status() -> String {
    match NvBaselineSession::new() {
        Err(NvBaselineError::NotBuilt) => "not-built".to_string(),
        Err(error) => format!("session-error:{error:?}"),
        Ok(_) => "session-unavailable".to_string(),
    }
}

fn timed_best<F>(config: &Config, mut f: F) -> TimedStatus
where
    F: FnMut() -> Result<TimedPixels, String>,
{
    let mut best: Option<TimedPixels> = None;
    for index in 0..config.warmup.saturating_add(config.iterations) {
        match f() {
            Ok(mut result) => {
                if index >= config.warmup
                    && best
                        .as_ref()
                        .is_none_or(|current| result.wall_ms < current.wall_ms)
                {
                    result.wall_ms = round6(result.wall_ms);
                    result.gpu_ms = result.gpu_ms.map(round6);
                    result.stage_ms = result.stage_ms.map(round6);
                    best = Some(result);
                }
            }
            Err(error) => return TimedStatus::failed(error),
        }
    }
    best.map_or_else(|| TimedStatus::failed("no-measurements"), TimedStatus::ok)
}

fn decode_cpu(input: &DecodeInput) -> Result<TimedPixels, String> {
    let started = Instant::now();
    let image = Image::new(&input.bytes, &DecodeSettings::default())
        .map_err(|error| format!("cpu parse: {error}"))?;
    let bitmap = image
        .decode_native()
        .map_err(|error| format!("cpu decode: {error}"))?;
    if bitmap.width != input.width
        || bitmap.height != input.height
        || usize::from(bitmap.num_components) != input.format.components()
        || bitmap.bytes_per_sample != 1
    {
        return Err("cpu decode returned unexpected shape".to_string());
    }
    Ok(TimedPixels {
        wall_ms: elapsed_ms(started),
        gpu_ms: None,
        stage_ms: None,
        pixels: bitmap.data,
    })
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn decode_signinum_cuda(
    session: &mut signinum_j2k_cuda::CudaSession,
    input: &DecodeInput,
) -> Result<TimedPixels, String> {
    let started = Instant::now();
    let mut decoder =
        signinum_j2k_cuda::J2kDecoder::new(&input.bytes).map_err(|error| error.to_string())?;
    let (surface, report) = decoder
        .decode_to_device_with_session_and_profile(input.format.signinum(), session)
        .map_err(|error| format!("signinum cuda decode: {error}"))?;
    if surface.backend_kind() != BackendKind::Cuda
        || surface.residency() != signinum_j2k_cuda::SurfaceResidency::CudaResidentDecode
    {
        return Err("signinum cuda decode did not return a CUDA-resident surface".to_string());
    }
    let stride = input.width as usize * input.format.components();
    let mut pixels = vec![0u8; stride * input.height as usize];
    surface
        .download_into(&mut pixels, stride)
        .map_err(|error| format!("signinum cuda download: {error}"))?;
    Ok(TimedPixels {
        wall_ms: elapsed_ms(started),
        gpu_ms: Some(us_to_ms(signinum_cuda_kernel_us(&report))),
        stage_ms: Some(us_to_ms(report.total_us)),
        pixels,
    })
}

#[cfg(any(target_os = "macos", not(feature = "nvjpeg2000")))]
fn decode_signinum_cuda(input: &DecodeInput) -> Result<TimedPixels, String> {
    let _ = input;
    Err("not-built".to_string())
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn signinum_cuda_kernel_us(report: &signinum_j2k_cuda::CudaHtj2kProfileReport) -> u128 {
    [
        report.ht_cleanup_us,
        report.ht_refine_us,
        report.dequant_us,
        report.idwt_us,
        report.mct_us,
        report.store_us,
    ]
    .into_iter()
    .fold(0u128, u128::saturating_add)
}

fn decode_nvidia(
    session: &mut NvBaselineSession,
    input: &DecodeInput,
) -> Result<TimedPixels, String> {
    let started = Instant::now();
    let decoded = session
        .decode_j2k_interleaved(&input.bytes, input.format.nvidia())
        .map_err(|error| format!("nvidia decode: {error:?}"))?;
    if decoded.width != input.width
        || decoded.height != input.height
        || decoded.num_components as usize != input.format.components()
        || decoded.bytes_per_sample != 1
    {
        return Err("nvidia decode returned unexpected shape".to_string());
    }
    Ok(TimedPixels {
        wall_ms: elapsed_ms(started),
        gpu_ms: Some(decoded.decode_ms),
        stage_ms: None,
        pixels: decoded.pixels,
    })
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn us_to_ms(micros: u128) -> f64 {
    micros as f64 / 1_000.0
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn print_report(rows: &[Row], config: &Config) {
    let megapixels = rows.iter().map(Row::megapixels).sum::<f64>();
    println!(
        "inputs: {} codestream(s), {:.2} MP total, iterations {}",
        rows.len(),
        megapixels,
        config.iterations
    );
    println!(
        "{:<24} {:>7} {:>10} {:>12} {:>12} {:>12} {:>14} {:>14} {:>12}",
        "input",
        "format",
        "CPU ms",
        "sig wall",
        "sig gpu",
        "sig stage",
        "NVIDIA wall",
        "NVIDIA gpu",
        "NV PSNR"
    );
    for row in rows {
        println!(
            "{:<24} {:>7} {:>10} {:>12} {:>12} {:>12} {:>14} {:>14} {:>12}",
            row.label,
            row.format.label(),
            status_ms(&row.cpu),
            status_ms(&row.signinum_cuda),
            status_gpu_ms(&row.signinum_cuda),
            status_stage_ms(&row.signinum_cuda),
            status_ms(&row.nvidia),
            status_gpu_ms(&row.nvidia),
            fmt_optional(row.nvidia_psnr_vs_cpu)
        );
    }
}

fn status_ms(status: &TimedStatus) -> String {
    status.result.as_ref().map_or_else(
        || status.status.clone(),
        |result| format!("{:.3}", result.wall_ms),
    )
}

fn status_gpu_ms(status: &TimedStatus) -> String {
    status
        .result
        .as_ref()
        .and_then(|result| result.gpu_ms)
        .map_or_else(|| "n/a".to_string(), |value| format!("{value:.3}"))
}

fn status_stage_ms(status: &TimedStatus) -> String {
    status
        .result
        .as_ref()
        .and_then(|result| result.stage_ms)
        .map_or_else(|| "n/a".to_string(), |value| format!("{value:.3}"))
}

fn fmt_optional(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.3}"),
        Some(_) => "inf".to_string(),
        None => "n/a".to_string(),
    }
}

fn write_artifacts(rows: &[Row], config: &Config) -> std::io::Result<()> {
    if let Some(path) = &config.json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, json_report(rows, config))?;
    }
    if let Some(path) = &config.csv {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, csv_report(rows))?;
    }
    Ok(())
}

fn json_report(rows: &[Row], config: &Config) -> String {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"input_count\": {},\n", rows.len()));
    json.push_str(&format!(
        "  \"megapixels\": {:.6},\n",
        rows.iter().map(Row::megapixels).sum::<f64>()
    ));
    json.push_str(&format!("  \"warmup\": {},\n", config.warmup));
    json.push_str(&format!("  \"iterations\": {},\n", config.iterations));
    json.push_str(&format!(
        "  \"max_inputs\": {},\n",
        config
            .max_inputs
            .map_or_else(|| "null".to_string(), |value| value.to_string())
    ));
    json.push_str("  \"rows\": [\n");
    for (index, row) in rows.iter().enumerate() {
        if index > 0 {
            json.push_str(",\n");
        }
        json.push_str("    {\n");
        json.push_str(&format!(
            "      \"label\": \"{}\",\n",
            escape_json(&row.label)
        ));
        json.push_str(&format!("      \"format\": \"{}\",\n", row.format.label()));
        json.push_str(&format!("      \"width\": {},\n", row.width));
        json.push_str(&format!("      \"height\": {},\n", row.height));
        json.push_str(&format!(
            "      \"codestream_bytes\": {},\n",
            row.codestream_bytes
        ));
        json.push_str(&json_status("cpu", &row.cpu, true));
        json.push_str(",\n");
        json.push_str(&json_status("signinum_cuda", &row.signinum_cuda, true));
        json.push_str(",\n");
        json.push_str(&json_status("nvidia_nvjpeg2000", &row.nvidia, true));
        json.push_str(",\n");
        json.push_str(&format!(
            "      \"signinum_cuda_psnr_vs_cpu\": {},\n",
            json_optional(row.signinum_cuda_psnr_vs_cpu)
        ));
        json.push_str(&format!(
            "      \"nvidia_psnr_vs_cpu\": {}\n",
            json_optional(row.nvidia_psnr_vs_cpu)
        ));
        json.push_str("    }");
    }
    json.push_str("\n  ]\n}\n");
    json
}

fn json_status(name: &str, status: &TimedStatus, indent: bool) -> String {
    let prefix = if indent { "      " } else { "" };
    let mut json = String::new();
    json.push_str(&format!("{prefix}\"{name}\": {{"));
    json.push_str(&format!("\"status\": \"{}\"", escape_json(&status.status)));
    if let Some(result) = &status.result {
        json.push_str(&format!(", \"wall_ms\": {:.6}", result.wall_ms));
        json.push_str(&format!(", \"gpu_ms\": {}", json_optional(result.gpu_ms)));
        json.push_str(&format!(
            ", \"stage_ms\": {}",
            json_optional(result.stage_ms)
        ));
        json.push_str(&format!(", \"bytes\": {}", result.pixels.len()));
    }
    json.push('}');
    json
}

fn json_optional(value: Option<f64>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| {
            if value.is_finite() {
                format!("{value:.6}")
            } else {
                "\"inf\"".to_string()
            }
        },
    )
}

fn csv_report(rows: &[Row]) -> String {
    let mut csv = String::from(
        "label,format,width,height,codestream_bytes,cpu_status,cpu_wall_ms,signinum_cuda_status,signinum_cuda_wall_ms,signinum_cuda_gpu_ms,signinum_cuda_stage_ms,nvidia_status,nvidia_wall_ms,nvidia_gpu_ms,signinum_cuda_psnr_vs_cpu,nvidia_psnr_vs_cpu\n",
    );
    for row in rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            escape_csv(&row.label),
            row.format.label(),
            row.width,
            row.height,
            row.codestream_bytes,
            escape_csv(&row.cpu.status),
            csv_ms(&row.cpu),
            escape_csv(&row.signinum_cuda.status),
            csv_ms(&row.signinum_cuda),
            csv_gpu_ms(&row.signinum_cuda),
            csv_stage_ms(&row.signinum_cuda),
            escape_csv(&row.nvidia.status),
            csv_ms(&row.nvidia),
            csv_gpu_ms(&row.nvidia),
            csv_optional(row.signinum_cuda_psnr_vs_cpu),
            csv_optional(row.nvidia_psnr_vs_cpu),
        ));
    }
    csv
}

fn csv_ms(status: &TimedStatus) -> String {
    status
        .result
        .as_ref()
        .map_or_else(String::new, |result| format!("{:.6}", result.wall_ms))
}

fn csv_gpu_ms(status: &TimedStatus) -> String {
    status
        .result
        .as_ref()
        .and_then(|result| result.gpu_ms)
        .map_or_else(String::new, |value| format!("{value:.6}"))
}

fn csv_stage_ms(status: &TimedStatus) -> String {
    status
        .result
        .as_ref()
        .and_then(|result| result.stage_ms)
        .map_or_else(String::new, |value| format!("{value:.6}"))
}

fn csv_optional(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| {
        if value.is_finite() {
            format!("{value:.6}")
        } else {
            "inf".to_string()
        }
    })
}

fn escape_json(value: &str) -> String {
    let mut escaped = String::new();
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn escape_csv(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{csv_report, Config, DecodeCaseFormat, Row, TimedPixels, TimedStatus};

    #[test]
    fn config_parses_artifact_flags() {
        let config = Config::from_args([
            "--fixture-dim",
            "256",
            "--jpeg-dir",
            "crates/signinum-nvidia-baseline/benchtiles/pancreas",
            "--warmup",
            "1",
            "--iterations",
            "3",
            "--min-inputs",
            "2",
            "--json",
            "target/decode.json",
            "--csv",
            "target/decode.csv",
        ])
        .expect("config parses");

        assert_eq!(config.fixture_dim, 256);
        assert_eq!(
            config.jpeg_dir.as_deref(),
            Some(std::path::Path::new(
                "crates/signinum-nvidia-baseline/benchtiles/pancreas"
            ))
        );
        assert_eq!(config.warmup, 1);
        assert_eq!(config.iterations, 3);
        assert_eq!(config.min_inputs, 2);
        assert_eq!(config.max_inputs, None);
        assert!(config.json.is_some());
        assert!(config.csv.is_some());
    }

    #[test]
    fn config_parses_max_inputs_separately_from_minimum_floor() {
        let config = Config::from_args(["--min-inputs", "100", "--max-inputs", "128"])
            .expect("config parses");

        assert_eq!(config.min_inputs, 100);
        assert_eq!(config.max_inputs, Some(128));
    }

    #[test]
    fn csv_report_marks_unavailable_nvidia_without_zero_times() {
        let row = Row {
            label: "tile,1".to_string(),
            format: DecodeCaseFormat::Rgb8,
            width: 512,
            height: 512,
            codestream_bytes: 100,
            cpu: TimedStatus::ok(TimedPixels {
                wall_ms: 1.0,
                gpu_ms: None,
                stage_ms: None,
                pixels: vec![0],
            }),
            signinum_cuda: TimedStatus::failed("not-built"),
            nvidia: TimedStatus::failed("not-built"),
            signinum_cuda_psnr_vs_cpu: None,
            nvidia_psnr_vs_cpu: None,
        };

        let csv = csv_report(&[row]);
        assert!(csv.contains("\"tile,1\",rgb8,512,512,100,ok,1.000000,not-built,,,,not-built,,"));
    }
}
