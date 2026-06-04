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

    let report = run_comparison(&inputs, &config);
    print_report(&report, &config);
    if let Err(error) = write_artifacts(&report, &config) {
        eprintln!("failed to write decode comparison artifacts: {error}");
        std::process::exit(2);
    }

    if should_exit_for_failed_required_report(&report, &config, require_nvidia) {
        eprintln!("required decode comparison/profile failed");
        std::process::exit(1);
    }
}

fn should_exit_for_failed_required_report(
    report: &DecodeReport,
    config: &Config,
    require_nvidia: bool,
) -> bool {
    if config.is_signinum_cuda_profile() {
        return report
            .profile
            .as_ref()
            .is_some_and(|profile| profile.status.has_failure());
    }
    require_nvidia && report.rows.iter().any(Row::has_required_failure)
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
    profile_signinum_cuda_only: bool,
    profile_signinum_cuda_batch: bool,
    collect_signinum_stage_timings: bool,
    skip_signinum_download: bool,
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
            profile_signinum_cuda_only: false,
            profile_signinum_cuda_batch: false,
            collect_signinum_stage_timings: false,
            skip_signinum_download: false,
        };
        let mut iter = args.into_iter().map(Into::into);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--profile-signinum-cuda-only" => config.profile_signinum_cuda_only = true,
                "--profile-signinum-cuda-batch" => config.profile_signinum_cuda_batch = true,
                "--collect-signinum-stage-timings" => {
                    config.collect_signinum_stage_timings = true;
                }
                "--skip-signinum-download" => config.skip_signinum_download = true,
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
        if config.skip_signinum_download && !config.is_signinum_cuda_profile() {
            return Err(
                "--skip-signinum-download requires --profile-signinum-cuda-only or --profile-signinum-cuda-batch"
                    .to_string(),
            );
        }
        if config.profile_signinum_cuda_only && config.profile_signinum_cuda_batch {
            return Err(
                "--profile-signinum-cuda-only conflicts with --profile-signinum-cuda-batch"
                    .to_string(),
            );
        }
        Ok(config)
    }

    fn is_signinum_cuda_profile(&self) -> bool {
        self.profile_signinum_cuda_only || self.profile_signinum_cuda_batch
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
    "usage: decode_compare [--profile-signinum-cuda-only] [--profile-signinum-cuda-batch] [--collect-signinum-stage-timings] [--skip-signinum-download] [--fixture-dim n] [--jpeg-dir path] [--warmup n] [--iterations n] [--min-inputs n] [--max-inputs n] [--json path] [--csv path] [file.j2k ...]".to_string()
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
    download_ms: Option<f64>,
    cuda_profile: Option<CudaStageBreakdown>,
    pixels: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
struct CudaStageBreakdown {
    parse_us: u128,
    plan_us: u128,
    flatten_us: u128,
    h2d_us: u128,
    ht_cleanup_us: u128,
    ht_refine_us: u128,
    dequant_us: u128,
    idwt_us: u128,
    mct_us: u128,
    store_us: u128,
    total_us: u128,
    wall_total_us: u128,
    table_upload_us: u128,
    payload_upload_us: u128,
    status_d2h_us: u128,
    output_d2h_us: u128,
    block_count: usize,
    payload_bytes: usize,
    dispatch_count: usize,
    ht_dispatch_count: usize,
    dequant_dispatch_count: usize,
    idwt_dispatch_count: usize,
    mct_dispatch_count: usize,
    store_dispatch_count: usize,
}

impl CudaStageBreakdown {
    #[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
    fn from_report(report: &signinum_j2k_cuda::CudaHtj2kProfileReport) -> Self {
        Self {
            parse_us: report.parse_us,
            plan_us: report.plan_us,
            flatten_us: report.flatten_us,
            h2d_us: report.h2d_us,
            ht_cleanup_us: report.ht_cleanup_us,
            ht_refine_us: report.ht_refine_us,
            dequant_us: report.dequant_us,
            idwt_us: report.idwt_us,
            mct_us: report.mct_us,
            store_us: report.store_us,
            total_us: report.total_us,
            wall_total_us: report.detail.wall_total_us,
            table_upload_us: report.detail.table_upload_us,
            payload_upload_us: report.detail.payload_upload_us,
            status_d2h_us: report.detail.status_d2h_us,
            output_d2h_us: report.detail.output_d2h_us,
            block_count: report.block_count,
            payload_bytes: report.payload_bytes,
            dispatch_count: report.dispatch_count,
            ht_dispatch_count: report.detail.ht_dispatch_count,
            dequant_dispatch_count: report.detail.dequant_dispatch_count,
            idwt_dispatch_count: report.detail.idwt_dispatch_count,
            mct_dispatch_count: report.detail.mct_dispatch_count,
            store_dispatch_count: report.detail.store_dispatch_count,
        }
    }

    fn add_assign(&mut self, other: &Self) {
        self.parse_us = self.parse_us.saturating_add(other.parse_us);
        self.plan_us = self.plan_us.saturating_add(other.plan_us);
        self.flatten_us = self.flatten_us.saturating_add(other.flatten_us);
        self.h2d_us = self.h2d_us.saturating_add(other.h2d_us);
        self.ht_cleanup_us = self.ht_cleanup_us.saturating_add(other.ht_cleanup_us);
        self.ht_refine_us = self.ht_refine_us.saturating_add(other.ht_refine_us);
        self.dequant_us = self.dequant_us.saturating_add(other.dequant_us);
        self.idwt_us = self.idwt_us.saturating_add(other.idwt_us);
        self.mct_us = self.mct_us.saturating_add(other.mct_us);
        self.store_us = self.store_us.saturating_add(other.store_us);
        self.total_us = self.total_us.saturating_add(other.total_us);
        self.wall_total_us = self.wall_total_us.saturating_add(other.wall_total_us);
        self.table_upload_us = self.table_upload_us.saturating_add(other.table_upload_us);
        self.payload_upload_us = self
            .payload_upload_us
            .saturating_add(other.payload_upload_us);
        self.status_d2h_us = self.status_d2h_us.saturating_add(other.status_d2h_us);
        self.output_d2h_us = self.output_d2h_us.saturating_add(other.output_d2h_us);
        self.block_count = self.block_count.saturating_add(other.block_count);
        self.payload_bytes = self.payload_bytes.saturating_add(other.payload_bytes);
        self.dispatch_count = self.dispatch_count.saturating_add(other.dispatch_count);
        self.ht_dispatch_count = self
            .ht_dispatch_count
            .saturating_add(other.ht_dispatch_count);
        self.dequant_dispatch_count = self
            .dequant_dispatch_count
            .saturating_add(other.dequant_dispatch_count);
        self.idwt_dispatch_count = self
            .idwt_dispatch_count
            .saturating_add(other.idwt_dispatch_count);
        self.mct_dispatch_count = self
            .mct_dispatch_count
            .saturating_add(other.mct_dispatch_count);
        self.store_dispatch_count = self
            .store_dispatch_count
            .saturating_add(other.store_dispatch_count);
    }
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

#[derive(Debug, Clone)]
struct DecodeReport {
    rows: Vec<Row>,
    profile: Option<ProfileMeasurement>,
}

impl DecodeReport {
    fn rows(rows: Vec<Row>) -> Self {
        Self {
            rows,
            profile: None,
        }
    }

    fn profile(rows: Vec<Row>, profile: ProfileMeasurement) -> Self {
        Self {
            rows,
            profile: Some(profile),
        }
    }

    fn input_count(&self) -> usize {
        self.profile
            .as_ref()
            .map_or(self.rows.len(), |profile| profile.input_count)
    }

    fn megapixels(&self) -> f64 {
        self.profile.as_ref().map_or_else(
            || self.rows.iter().map(Row::megapixels).sum(),
            |profile| profile.megapixels,
        )
    }
}

#[derive(Debug, Clone)]
struct ProfileMeasurement {
    label: &'static str,
    execution_mode: &'static str,
    timing_scope: &'static str,
    download_policy: &'static str,
    input_count: usize,
    megapixels: f64,
    codestream_bytes: usize,
    status: TimedStatus,
}

impl ProfileMeasurement {
    fn mp_s(&self) -> Option<f64> {
        let result = self.status.result.as_ref()?;
        Some(if result.wall_ms > 0.0 {
            self.megapixels / (result.wall_ms / 1_000.0)
        } else {
            f64::INFINITY
        })
    }
}

fn run_comparison(inputs: &[DecodeInput], config: &Config) -> DecodeReport {
    if config.profile_signinum_cuda_batch {
        return DecodeReport::profile(Vec::new(), run_signinum_cuda_batch(inputs, config));
    }
    if config.profile_signinum_cuda_only {
        let rows = run_signinum_cuda_only(inputs, config);
        let profile = serial_profile_measurement(&rows, inputs, config);
        return DecodeReport::profile(rows, profile);
    }

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
            decode_signinum_cuda(&mut signinum_cuda_session, input, true, false)
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
    DecodeReport::rows(rows)
}

fn run_signinum_cuda_batch(inputs: &[DecodeInput], config: &Config) -> ProfileMeasurement {
    #[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
    let signinum_cuda = {
        let mut signinum_cuda_session = signinum_j2k_cuda::CudaSession::default();
        timed_best(config, || {
            decode_signinum_cuda_batch(
                &mut signinum_cuda_session,
                inputs,
                config.collect_signinum_stage_timings,
                config.skip_signinum_download,
            )
        })
    };
    #[cfg(any(target_os = "macos", not(feature = "nvjpeg2000")))]
    let signinum_cuda = timed_best(config, || decode_signinum_cuda_batch(inputs));
    ProfileMeasurement {
        label: "signinum_cuda_decode_real_batch",
        execution_mode: "signinum_cuda_batch",
        timing_scope: "aggregate_batch",
        download_policy: signinum_download_policy(config),
        input_count: inputs.len(),
        megapixels: input_megapixels(inputs),
        codestream_bytes: input_codestream_bytes(inputs),
        status: signinum_cuda,
    }
}

fn run_signinum_cuda_only(inputs: &[DecodeInput], config: &Config) -> Vec<Row> {
    let mut rows = Vec::with_capacity(inputs.len());
    #[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
    let mut signinum_cuda_session = signinum_j2k_cuda::CudaSession::default();
    for input in inputs {
        #[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
        let signinum_cuda = timed_best(config, || {
            decode_signinum_cuda(
                &mut signinum_cuda_session,
                input,
                config.collect_signinum_stage_timings,
                config.skip_signinum_download,
            )
        });
        #[cfg(any(target_os = "macos", not(feature = "nvjpeg2000")))]
        let signinum_cuda = timed_best(config, || decode_signinum_cuda(input));
        rows.push(Row {
            label: input.label.clone(),
            format: input.format,
            width: input.width,
            height: input.height,
            codestream_bytes: input.bytes.len(),
            cpu: TimedStatus::failed("skipped"),
            signinum_cuda,
            nvidia: TimedStatus::failed("skipped"),
            signinum_cuda_psnr_vs_cpu: None,
            nvidia_psnr_vs_cpu: None,
        });
    }
    rows
}

fn serial_profile_measurement(
    rows: &[Row],
    inputs: &[DecodeInput],
    config: &Config,
) -> ProfileMeasurement {
    ProfileMeasurement {
        label: "signinum_cuda_decode_serial_reused_session",
        execution_mode: "signinum_cuda_serial_reused_session",
        timing_scope: "sum_of_per_input_best",
        download_policy: signinum_download_policy(config),
        input_count: inputs.len(),
        megapixels: input_megapixels(inputs),
        codestream_bytes: input_codestream_bytes(inputs),
        status: sum_signinum_cuda_rows(rows),
    }
}

fn sum_signinum_cuda_rows(rows: &[Row]) -> TimedStatus {
    let mut wall_ms = 0.0;
    let mut gpu_ms = None;
    let mut stage_ms = None;
    let mut download_ms = None;
    let mut cuda_profile = None;
    for row in rows {
        let Some(result) = row.signinum_cuda.result.as_ref() else {
            return TimedStatus::failed(format!("{}: {}", row.label, row.signinum_cuda.status));
        };
        wall_ms += result.wall_ms;
        gpu_ms = optional_sum(gpu_ms, result.gpu_ms);
        stage_ms = optional_sum(stage_ms, result.stage_ms);
        download_ms = optional_sum(download_ms, result.download_ms);
        cuda_profile = optional_profile_sum(cuda_profile, result.cuda_profile.as_ref());
    }
    TimedStatus::ok(TimedPixels {
        wall_ms,
        gpu_ms,
        stage_ms,
        download_ms,
        cuda_profile,
        pixels: Vec::new(),
    })
}

fn optional_sum(current: Option<f64>, next: Option<f64>) -> Option<f64> {
    match (current, next) {
        (Some(current), Some(next)) => Some(current + next),
        (None, Some(next)) => Some(next),
        (current, None) => current,
    }
}

fn optional_profile_sum(
    current: Option<CudaStageBreakdown>,
    next: Option<&CudaStageBreakdown>,
) -> Option<CudaStageBreakdown> {
    match (current, next) {
        (Some(mut current), Some(next)) => {
            current.add_assign(next);
            Some(current)
        }
        (None, Some(next)) => Some(next.clone()),
        (current, None) => current,
    }
}

fn input_megapixels(inputs: &[DecodeInput]) -> f64 {
    inputs
        .iter()
        .map(|input| f64::from(input.width) * f64::from(input.height) / 1.0e6)
        .sum()
}

fn input_codestream_bytes(inputs: &[DecodeInput]) -> usize {
    inputs.iter().map(|input| input.bytes.len()).sum()
}

fn signinum_download_policy(config: &Config) -> &'static str {
    if config.skip_signinum_download {
        "no_download"
    } else {
        "download"
    }
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
                    result.download_ms = result.download_ms.map(round6);
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
        download_ms: None,
        cuda_profile: None,
        pixels: bitmap.data,
    })
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn decode_signinum_cuda(
    session: &mut signinum_j2k_cuda::CudaSession,
    input: &DecodeInput,
    collect_stage_timings: bool,
    skip_download: bool,
) -> Result<TimedPixels, String> {
    let started = Instant::now();
    let mut decoder =
        signinum_j2k_cuda::J2kDecoder::new(&input.bytes).map_err(|error| error.to_string())?;
    let (surface, report) = if collect_stage_timings {
        let (surface, report) = decoder
            .decode_to_device_with_session_and_profile(input.format.signinum(), session)
            .map_err(|error| format!("signinum cuda decode: {error}"))?;
        (surface, Some(report))
    } else {
        let surface = decoder
            .decode_to_device_with_session(input.format.signinum(), session)
            .map_err(|error| format!("signinum cuda decode: {error}"))?;
        (surface, None)
    };
    if surface.backend_kind() != BackendKind::Cuda
        || surface.residency() != signinum_j2k_cuda::SurfaceResidency::CudaResidentDecode
    {
        return Err("signinum cuda decode did not return a CUDA-resident surface".to_string());
    }
    let stride = input.width as usize * input.format.components();
    if skip_download {
        return Ok(TimedPixels {
            wall_ms: elapsed_ms(started),
            gpu_ms: report
                .as_ref()
                .map(|report| us_to_ms(signinum_cuda_kernel_us(report))),
            stage_ms: report.as_ref().map(|report| us_to_ms(report.total_us)),
            download_ms: None,
            cuda_profile: report.as_ref().map(CudaStageBreakdown::from_report),
            pixels: Vec::new(),
        });
    }
    let mut pixels = vec![0u8; stride * input.height as usize];
    let download_started = Instant::now();
    surface
        .download_into(&mut pixels, stride)
        .map_err(|error| format!("signinum cuda download: {error}"))?;
    let download_ms = elapsed_ms(download_started);
    Ok(TimedPixels {
        wall_ms: elapsed_ms(started),
        gpu_ms: report
            .as_ref()
            .map(|report| us_to_ms(signinum_cuda_kernel_us(report))),
        stage_ms: report.as_ref().map(|report| us_to_ms(report.total_us)),
        download_ms: Some(download_ms),
        cuda_profile: report.as_ref().map(CudaStageBreakdown::from_report),
        pixels,
    })
}

#[cfg(all(not(target_os = "macos"), feature = "nvjpeg2000"))]
fn decode_signinum_cuda_batch(
    session: &mut signinum_j2k_cuda::CudaSession,
    inputs: &[DecodeInput],
    collect_stage_timings: bool,
    skip_download: bool,
) -> Result<TimedPixels, String> {
    let started = Instant::now();
    let Some(first) = inputs.first() else {
        return Ok(TimedPixels {
            wall_ms: elapsed_ms(started),
            gpu_ms: collect_stage_timings.then_some(0.0),
            stage_ms: collect_stage_timings.then_some(0.0),
            download_ms: None,
            cuda_profile: None,
            pixels: Vec::new(),
        });
    };
    if inputs.iter().any(|input| input.format != first.format) {
        return Err("signinum cuda batch decode requires a single output format".to_string());
    }
    let input_bytes = inputs
        .iter()
        .map(|input| input.bytes.as_slice())
        .collect::<Vec<_>>();
    let (surfaces, report) = if collect_stage_timings {
        let (surfaces, report) =
            signinum_j2k_cuda::J2kDecoder::decode_batch_to_device_with_session_and_profile(
                &input_bytes,
                first.format.signinum(),
                session,
            )
            .map_err(|error| format!("signinum cuda batch decode: {error}"))?;
        (surfaces, Some(report))
    } else {
        let surfaces = signinum_j2k_cuda::J2kDecoder::decode_batch_to_device_with_session(
            &input_bytes,
            first.format.signinum(),
            session,
        )
        .map_err(|error| format!("signinum cuda batch decode: {error}"))?;
        (surfaces, None)
    };
    if surfaces.len() != inputs.len() {
        return Err("signinum cuda batch decode returned unexpected surface count".to_string());
    }
    let mut pixels = Vec::new();
    let mut download_ms = None;
    for (surface, input) in surfaces.iter().zip(inputs) {
        if surface.backend_kind() != BackendKind::Cuda
            || surface.residency() != signinum_j2k_cuda::SurfaceResidency::CudaResidentDecode
        {
            return Err(
                "signinum cuda batch decode did not return CUDA-resident surfaces".to_string(),
            );
        }
        if surface.dimensions() != (input.width, input.height) {
            return Err("signinum cuda batch decode returned unexpected shape".to_string());
        }
    }
    if !skip_download {
        let download_started = Instant::now();
        pixels = signinum_j2k_cuda::Surface::download_batch_tight(&surfaces)
            .map_err(|error| format!("signinum cuda batch download: {error}"))?;
        download_ms = Some(elapsed_ms(download_started));
    }
    Ok(TimedPixels {
        wall_ms: elapsed_ms(started),
        gpu_ms: report
            .as_ref()
            .map(|report| us_to_ms(signinum_cuda_kernel_us(report))),
        stage_ms: report.as_ref().map(|report| us_to_ms(report.total_us)),
        download_ms,
        cuda_profile: report.as_ref().map(CudaStageBreakdown::from_report),
        pixels,
    })
}

#[cfg(any(target_os = "macos", not(feature = "nvjpeg2000")))]
fn decode_signinum_cuda(input: &DecodeInput) -> Result<TimedPixels, String> {
    let _ = input;
    Err("not-built".to_string())
}

#[cfg(any(target_os = "macos", not(feature = "nvjpeg2000")))]
fn decode_signinum_cuda_batch(inputs: &[DecodeInput]) -> Result<TimedPixels, String> {
    let _ = inputs;
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
        download_ms: None,
        cuda_profile: None,
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

fn clean_profile_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn print_report(report: &DecodeReport, config: &Config) {
    if let Some(profile) = &report.profile {
        print_signinum_cuda_profile_report(profile, config);
        return;
    }

    let rows = &report.rows;
    let megapixels = rows.iter().map(Row::megapixels).sum::<f64>();
    println!(
        "inputs: {} codestream(s), {:.2} MP total, iterations {}",
        rows.len(),
        megapixels,
        config.iterations
    );
    println!(
        "{:<24} {:>7} {:>10} {:>12} {:>12} {:>12} {:>12} {:>14} {:>14} {:>12}",
        "input",
        "format",
        "CPU ms",
        "sig serial wall",
        "sig serial gpu",
        "sig serial stage",
        "sig serial dl",
        "NVIDIA wall",
        "NVIDIA gpu",
        "NV PSNR"
    );
    for row in rows {
        println!(
            "{:<24} {:>7} {:>10} {:>12} {:>12} {:>12} {:>12} {:>14} {:>14} {:>12}",
            row.label,
            row.format.label(),
            status_ms(&row.cpu),
            status_ms(&row.signinum_cuda),
            status_gpu_ms(&row.signinum_cuda),
            status_stage_ms(&row.signinum_cuda),
            status_download_ms(&row.signinum_cuda),
            status_ms(&row.nvidia),
            status_gpu_ms(&row.nvidia),
            fmt_optional(row.nvidia_psnr_vs_cpu)
        );
    }
}

fn print_signinum_cuda_profile_report(profile: &ProfileMeasurement, config: &Config) {
    println!(
        "inputs: {} codestream(s), {:.2} MP total",
        profile.input_count, profile.megapixels
    );
    println!(
        "profile: {} execution_mode={} timing_scope={} download_policy={} iterations {}",
        profile.label,
        profile.execution_mode,
        profile.timing_scope,
        profile.download_policy,
        config.iterations
    );
    let Some(result) = profile.status.result.as_ref() else {
        println!("{}: {}", profile.label, profile.status.status);
        return;
    };
    println!(
        "PROFILE_RESULT {} execution_mode={} timing_scope={} download_policy={} input_count={} mp_s={:.3} wall_ms={:.3} gpu_ms={:.3} stage_ms={:.3} download_ms={:.3} bytes={}",
        profile.label,
        profile.execution_mode,
        profile.timing_scope,
        profile.download_policy,
        profile.input_count,
        profile.mp_s().unwrap_or(f64::NAN),
        result.wall_ms,
        clean_profile_zero(result.gpu_ms.unwrap_or(0.0)),
        clean_profile_zero(result.stage_ms.unwrap_or(0.0)),
        clean_profile_zero(result.download_ms.unwrap_or(0.0)),
        profile.codestream_bytes
    );
    if let Some(cuda_profile) = &result.cuda_profile {
        println!(
            "PROFILE_BREAKDOWN {} parse_us={} plan_us={} flatten_us={} h2d_us={} ht_cleanup_us={} ht_refine_us={} dequant_us={} idwt_us={} mct_us={} store_us={} total_us={} wall_total_us={} table_upload_us={} payload_upload_us={} status_d2h_us={} output_d2h_us={} block_count={} payload_bytes={} dispatch_count={} ht_dispatch_count={} dequant_dispatch_count={} idwt_dispatch_count={} mct_dispatch_count={} store_dispatch_count={}",
            profile.label,
            cuda_profile.parse_us,
            cuda_profile.plan_us,
            cuda_profile.flatten_us,
            cuda_profile.h2d_us,
            cuda_profile.ht_cleanup_us,
            cuda_profile.ht_refine_us,
            cuda_profile.dequant_us,
            cuda_profile.idwt_us,
            cuda_profile.mct_us,
            cuda_profile.store_us,
            cuda_profile.total_us,
            cuda_profile.wall_total_us,
            cuda_profile.table_upload_us,
            cuda_profile.payload_upload_us,
            cuda_profile.status_d2h_us,
            cuda_profile.output_d2h_us,
            cuda_profile.block_count,
            cuda_profile.payload_bytes,
            cuda_profile.dispatch_count,
            cuda_profile.ht_dispatch_count,
            cuda_profile.dequant_dispatch_count,
            cuda_profile.idwt_dispatch_count,
            cuda_profile.mct_dispatch_count,
            cuda_profile.store_dispatch_count
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

fn status_download_ms(status: &TimedStatus) -> String {
    status
        .result
        .as_ref()
        .and_then(|result| result.download_ms)
        .map_or_else(|| "n/a".to_string(), |value| format!("{value:.3}"))
}

fn fmt_optional(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.3}"),
        Some(_) => "inf".to_string(),
        None => "n/a".to_string(),
    }
}

fn write_artifacts(report: &DecodeReport, config: &Config) -> std::io::Result<()> {
    if let Some(path) = &config.json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, json_report(report, config))?;
    }
    if let Some(path) = &config.csv {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, csv_report(report))?;
    }
    Ok(())
}

fn json_report(report: &DecodeReport, config: &Config) -> String {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"input_count\": {},\n", report.input_count()));
    json.push_str(&format!("  \"megapixels\": {:.6},\n", report.megapixels()));
    json.push_str(&format!("  \"warmup\": {},\n", config.warmup));
    json.push_str(&format!("  \"iterations\": {},\n", config.iterations));
    json.push_str(&format!(
        "  \"max_inputs\": {},\n",
        config
            .max_inputs
            .map_or_else(|| "null".to_string(), |value| value.to_string())
    ));
    json.push_str("  \"profile\": ");
    if let Some(profile) = &report.profile {
        json.push_str(&json_profile(profile));
        json.push_str(",\n");
    } else {
        json.push_str("null,\n");
    }
    json.push_str("  \"rows\": [\n");
    for (index, row) in report.rows.iter().enumerate() {
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

fn json_profile(profile: &ProfileMeasurement) -> String {
    let mut json = String::new();
    json.push_str("{");
    json.push_str(&format!("\"label\": \"{}\"", escape_json(profile.label)));
    json.push_str(&format!(
        ", \"execution_mode\": \"{}\"",
        escape_json(profile.execution_mode)
    ));
    json.push_str(&format!(
        ", \"timing_scope\": \"{}\"",
        escape_json(profile.timing_scope)
    ));
    json.push_str(&format!(
        ", \"download_policy\": \"{}\"",
        escape_json(profile.download_policy)
    ));
    json.push_str(&format!(", \"input_count\": {}", profile.input_count));
    json.push_str(&format!(", \"megapixels\": {:.6}", profile.megapixels));
    json.push_str(&format!(
        ", \"codestream_bytes\": {}",
        profile.codestream_bytes
    ));
    json.push_str(&format!(", \"mp_s\": {}", json_optional(profile.mp_s())));
    json.push_str(", ");
    json.push_str(&json_status("status", &profile.status, false));
    json.push('}');
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
        json.push_str(&format!(
            ", \"download_ms\": {}",
            json_optional(result.download_ms)
        ));
        json.push_str(&format!(", \"bytes\": {}", result.pixels.len()));
        if let Some(cuda_profile) = &result.cuda_profile {
            json.push_str(", \"cuda_profile\": ");
            json.push_str(&json_cuda_profile(cuda_profile));
        }
    }
    json.push('}');
    json
}

fn json_cuda_profile(profile: &CudaStageBreakdown) -> String {
    format!(
        "{{\"parse_us\": {}, \"plan_us\": {}, \"flatten_us\": {}, \"h2d_us\": {}, \"ht_cleanup_us\": {}, \"ht_refine_us\": {}, \"dequant_us\": {}, \"idwt_us\": {}, \"mct_us\": {}, \"store_us\": {}, \"total_us\": {}, \"wall_total_us\": {}, \"table_upload_us\": {}, \"payload_upload_us\": {}, \"status_d2h_us\": {}, \"output_d2h_us\": {}, \"block_count\": {}, \"payload_bytes\": {}, \"dispatch_count\": {}, \"ht_dispatch_count\": {}, \"dequant_dispatch_count\": {}, \"idwt_dispatch_count\": {}, \"mct_dispatch_count\": {}, \"store_dispatch_count\": {}}}",
        profile.parse_us,
        profile.plan_us,
        profile.flatten_us,
        profile.h2d_us,
        profile.ht_cleanup_us,
        profile.ht_refine_us,
        profile.dequant_us,
        profile.idwt_us,
        profile.mct_us,
        profile.store_us,
        profile.total_us,
        profile.wall_total_us,
        profile.table_upload_us,
        profile.payload_upload_us,
        profile.status_d2h_us,
        profile.output_d2h_us,
        profile.block_count,
        profile.payload_bytes,
        profile.dispatch_count,
        profile.ht_dispatch_count,
        profile.dequant_dispatch_count,
        profile.idwt_dispatch_count,
        profile.mct_dispatch_count,
        profile.store_dispatch_count
    )
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

fn csv_report(report: &DecodeReport) -> String {
    if let Some(profile) = &report.profile {
        return csv_profile_report(profile);
    }

    let mut csv = String::from(
        "label,format,width,height,codestream_bytes,cpu_status,cpu_wall_ms,signinum_cuda_status,signinum_cuda_wall_ms,signinum_cuda_gpu_ms,signinum_cuda_stage_ms,signinum_cuda_download_ms,nvidia_status,nvidia_wall_ms,nvidia_gpu_ms,signinum_cuda_psnr_vs_cpu,nvidia_psnr_vs_cpu\n",
    );
    for row in &report.rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
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
            csv_download_ms(&row.signinum_cuda),
            escape_csv(&row.nvidia.status),
            csv_ms(&row.nvidia),
            csv_gpu_ms(&row.nvidia),
            csv_optional(row.signinum_cuda_psnr_vs_cpu),
            csv_optional(row.nvidia_psnr_vs_cpu),
        ));
    }
    csv
}

fn csv_profile_report(profile: &ProfileMeasurement) -> String {
    let mut csv = format!(
        "row_type,timing_scope,execution_mode,download_policy,input_count,label,status,megapixels,codestream_bytes,mp_s,wall_ms,gpu_ms,stage_ms,download_ms,{}\n",
        cuda_profile_csv_header()
    );
    let cuda_profile = profile
        .status
        .result
        .as_ref()
        .and_then(|result| result.cuda_profile.as_ref());
    csv.push_str(&format!(
        "aggregate,{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{}\n",
        profile.timing_scope,
        profile.execution_mode,
        profile.download_policy,
        profile.input_count,
        escape_csv(profile.label),
        escape_csv(&profile.status.status),
        profile.megapixels,
        profile.codestream_bytes,
        csv_optional(profile.mp_s()),
        csv_ms(&profile.status),
        csv_gpu_ms(&profile.status),
        csv_stage_ms(&profile.status),
        csv_download_ms(&profile.status),
        csv_cuda_profile_values(cuda_profile),
    ));
    csv
}

fn cuda_profile_csv_header() -> &'static str {
    "cuda_parse_us,cuda_plan_us,cuda_flatten_us,cuda_h2d_us,cuda_ht_cleanup_us,cuda_ht_refine_us,cuda_dequant_us,cuda_idwt_us,cuda_mct_us,cuda_store_us,cuda_total_us,cuda_wall_total_us,cuda_table_upload_us,cuda_payload_upload_us,cuda_status_d2h_us,cuda_output_d2h_us,cuda_block_count,cuda_payload_bytes,cuda_dispatch_count,cuda_ht_dispatch_count,cuda_dequant_dispatch_count,cuda_idwt_dispatch_count,cuda_mct_dispatch_count,cuda_store_dispatch_count"
}

fn csv_cuda_profile_values(profile: Option<&CudaStageBreakdown>) -> String {
    const FIELD_COUNT: usize = 24;
    let Some(profile) = profile else {
        return vec![""; FIELD_COUNT].join(",");
    };
    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        profile.parse_us,
        profile.plan_us,
        profile.flatten_us,
        profile.h2d_us,
        profile.ht_cleanup_us,
        profile.ht_refine_us,
        profile.dequant_us,
        profile.idwt_us,
        profile.mct_us,
        profile.store_us,
        profile.total_us,
        profile.wall_total_us,
        profile.table_upload_us,
        profile.payload_upload_us,
        profile.status_d2h_us,
        profile.output_d2h_us,
        profile.block_count,
        profile.payload_bytes,
        profile.dispatch_count,
        profile.ht_dispatch_count,
        profile.dequant_dispatch_count,
        profile.idwt_dispatch_count,
        profile.mct_dispatch_count,
        profile.store_dispatch_count
    )
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

fn csv_download_ms(status: &TimedStatus) -> String {
    status
        .result
        .as_ref()
        .and_then(|result| result.download_ms)
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
    use super::{
        clean_profile_zero, csv_report, json_report, should_exit_for_failed_required_report,
        Config, CudaStageBreakdown, DecodeCaseFormat, DecodeReport, ProfileMeasurement, Row,
        TimedPixels, TimedStatus,
    };

    #[test]
    fn config_parses_artifact_flags() {
        let config = Config::from_args([
            "--profile-signinum-cuda-batch",
            "--collect-signinum-stage-timings",
            "--skip-signinum-download",
            "--fixture-dim",
            "256",
            "--jpeg-dir",
            "tests/nvidia-baseline/benchtiles/pancreas",
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

        assert!(!config.profile_signinum_cuda_only);
        assert!(config.profile_signinum_cuda_batch);
        assert!(config.collect_signinum_stage_timings);
        assert!(config.skip_signinum_download);
        assert_eq!(config.fixture_dim, 256);
        assert_eq!(
            config.jpeg_dir.as_deref(),
            Some(std::path::Path::new(
                "tests/nvidia-baseline/benchtiles/pancreas"
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
    fn profile_modes_conflict() {
        let error = Config::from_args([
            "--profile-signinum-cuda-only",
            "--profile-signinum-cuda-batch",
        ])
        .expect_err("profile modes conflict");

        assert!(error.contains("conflicts"));
    }

    #[test]
    fn config_parses_max_inputs_separately_from_minimum_floor() {
        let config = Config::from_args(["--min-inputs", "100", "--max-inputs", "128"])
            .expect("config parses");

        assert_eq!(config.min_inputs, 100);
        assert_eq!(config.max_inputs, Some(128));
    }

    #[test]
    fn skip_signinum_download_requires_profile_only_mode() {
        let error = Config::from_args(["--skip-signinum-download"])
            .expect_err("skip download is only valid in profile-only mode");

        assert!(error.contains("--profile-signinum-cuda"));
    }

    #[test]
    fn skip_signinum_download_accepts_batch_profile_mode() {
        let config =
            Config::from_args(["--profile-signinum-cuda-batch", "--skip-signinum-download"])
                .expect("batch profile can skip download");

        assert!(config.profile_signinum_cuda_batch);
        assert!(config.skip_signinum_download);
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
                download_ms: None,
                cuda_profile: None,
                pixels: vec![0],
            }),
            signinum_cuda: TimedStatus::failed("not-built"),
            nvidia: TimedStatus::failed("not-built"),
            signinum_cuda_psnr_vs_cpu: None,
            nvidia_psnr_vs_cpu: None,
        };

        let csv = csv_report(&DecodeReport::rows(vec![row]));
        assert!(csv.contains("\"tile,1\",rgb8,512,512,100,ok,1.000000,not-built,,,,,not-built,,"));
    }

    #[test]
    fn artifacts_include_signinum_cuda_download_time_when_recorded() {
        let row = Row {
            label: "tile-1".to_string(),
            format: DecodeCaseFormat::Rgb8,
            width: 256,
            height: 256,
            codestream_bytes: 100,
            cpu: TimedStatus::failed("skipped"),
            signinum_cuda: TimedStatus::ok(TimedPixels {
                wall_ms: 1.0,
                gpu_ms: Some(0.25),
                stage_ms: Some(0.5),
                download_ms: Some(0.125),
                cuda_profile: None,
                pixels: vec![0],
            }),
            nvidia: TimedStatus::failed("skipped"),
            signinum_cuda_psnr_vs_cpu: None,
            nvidia_psnr_vs_cpu: None,
        };
        let config = Config::from_args(std::iter::empty::<&str>()).expect("empty config parses");

        let report = DecodeReport::rows(vec![row.clone()]);
        let json = json_report(&report, &config);
        assert!(json.contains("\"download_ms\": 0.125000"));

        let csv = csv_report(&DecodeReport::rows(vec![row]));
        assert!(csv.contains("signinum_cuda_download_ms"));
        assert!(csv.contains(",0.125000,"));
    }

    #[test]
    fn batch_profile_report_uses_single_aggregate_csv_row() {
        let report = DecodeReport::profile(
            Vec::new(),
            ProfileMeasurement {
                label: "signinum_cuda_decode_real_batch",
                execution_mode: "signinum_cuda_batch",
                timing_scope: "aggregate_batch",
                download_policy: "no_download",
                input_count: 2,
                megapixels: 0.5,
                codestream_bytes: 200,
                status: TimedStatus::ok(TimedPixels {
                    wall_ms: 1.0,
                    gpu_ms: Some(0.25),
                    stage_ms: Some(0.5),
                    download_ms: None,
                    cuda_profile: None,
                    pixels: Vec::new(),
                }),
            },
        );

        let csv = csv_report(&report);
        assert_eq!(csv.lines().count(), 2);
        assert!(csv.contains("aggregate_batch,signinum_cuda_batch,no_download"));
        assert!(csv.contains("signinum_cuda_decode_real_batch"));
    }

    #[test]
    fn batch_profile_json_does_not_emit_per_input_timings() {
        let config =
            Config::from_args(["--profile-signinum-cuda-batch", "--skip-signinum-download"])
                .expect("config parses");
        let report = DecodeReport::profile(
            Vec::new(),
            ProfileMeasurement {
                label: "signinum_cuda_decode_real_batch",
                execution_mode: "signinum_cuda_batch",
                timing_scope: "aggregate_batch",
                download_policy: "no_download",
                input_count: 2,
                megapixels: 0.5,
                codestream_bytes: 200,
                status: TimedStatus::ok(TimedPixels {
                    wall_ms: 1.0,
                    gpu_ms: Some(0.25),
                    stage_ms: Some(0.5),
                    download_ms: None,
                    cuda_profile: None,
                    pixels: Vec::new(),
                }),
            },
        );

        let json = json_report(&report, &config);
        assert!(json.contains("\"profile\": {"));
        assert!(json.contains("\"execution_mode\": \"signinum_cuda_batch\""));
        assert!(json.contains("\"timing_scope\": \"aggregate_batch\""));
        assert!(json.contains("\"download_policy\": \"no_download\""));
        assert!(!json.contains("\"signinum_cuda\": {\"status\": \"ok\", \"wall_ms\""));
    }

    #[test]
    fn serial_profile_is_labeled_serial_reused_session() {
        let report = DecodeReport::profile(
            Vec::new(),
            ProfileMeasurement {
                label: "signinum_cuda_decode_serial_reused_session",
                execution_mode: "signinum_cuda_serial_reused_session",
                timing_scope: "sum_of_per_input_best",
                download_policy: "download",
                input_count: 2,
                megapixels: 0.5,
                codestream_bytes: 200,
                status: TimedStatus::ok(TimedPixels {
                    wall_ms: 2.0,
                    gpu_ms: Some(0.5),
                    stage_ms: Some(1.0),
                    download_ms: Some(0.25),
                    cuda_profile: None,
                    pixels: Vec::new(),
                }),
            },
        );

        let csv = csv_report(&report);
        assert!(csv.contains("signinum_cuda_decode_serial_reused_session"));
        assert!(csv.contains("sum_of_per_input_best"));
        assert!(csv.contains("signinum_cuda_serial_reused_session"));
    }

    #[test]
    fn profile_result_zero_timings_do_not_render_negative_zero() {
        assert_eq!(format!("{:.3}", clean_profile_zero(-0.0)), "0.000");
    }

    #[test]
    fn failed_profile_report_requires_nonzero_exit_without_nvidia_requirement() {
        let config = Config::from_args(["--profile-signinum-cuda-batch"]).expect("config parses");
        let report = DecodeReport::profile(
            Vec::new(),
            ProfileMeasurement {
                label: "signinum_cuda_decode_real_batch",
                execution_mode: "signinum_cuda_batch",
                timing_scope: "aggregate_batch",
                download_policy: "download",
                input_count: 2,
                megapixels: 0.5,
                codestream_bytes: 200,
                status: TimedStatus::failed("not-built"),
            },
        );

        assert!(should_exit_for_failed_required_report(
            &report, &config, false
        ));
    }

    #[test]
    fn profile_artifacts_include_cuda_stage_breakdown() {
        let config = Config::from_args([
            "--profile-signinum-cuda-batch",
            "--collect-signinum-stage-timings",
        ])
        .expect("config parses");
        let report = DecodeReport::profile(
            Vec::new(),
            ProfileMeasurement {
                label: "signinum_cuda_decode_real_batch",
                execution_mode: "signinum_cuda_batch",
                timing_scope: "aggregate_batch",
                download_policy: "download",
                input_count: 2,
                megapixels: 0.5,
                codestream_bytes: 200,
                status: TimedStatus::ok(TimedPixels {
                    wall_ms: 1.0,
                    gpu_ms: Some(0.25),
                    stage_ms: Some(0.5),
                    download_ms: Some(0.125),
                    cuda_profile: Some(fake_cuda_profile()),
                    pixels: Vec::new(),
                }),
            },
        );

        let json = json_report(&report, &config);
        assert!(json.contains("\"cuda_profile\": {"));
        assert!(json.contains("\"idwt_us\": 8"));
        assert!(json.contains("\"dispatch_count\": 19"));

        let csv = csv_report(&report);
        assert!(csv.contains("cuda_idwt_us"));
        assert!(csv.contains("cuda_dispatch_count"));
        assert!(csv.contains(",7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24"));
    }

    fn fake_cuda_profile() -> CudaStageBreakdown {
        CudaStageBreakdown {
            parse_us: 1,
            plan_us: 2,
            flatten_us: 3,
            h2d_us: 4,
            ht_cleanup_us: 5,
            ht_refine_us: 6,
            dequant_us: 7,
            idwt_us: 8,
            mct_us: 9,
            store_us: 10,
            total_us: 11,
            wall_total_us: 12,
            table_upload_us: 13,
            payload_upload_us: 14,
            status_d2h_us: 15,
            output_d2h_us: 16,
            block_count: 17,
            payload_bytes: 18,
            dispatch_count: 19,
            ht_dispatch_count: 20,
            dequant_dispatch_count: 21,
            idwt_dispatch_count: 22,
            mct_dispatch_count: 23,
            store_dispatch_count: 24,
        }
    }
}
