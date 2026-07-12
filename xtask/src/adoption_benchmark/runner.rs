use std::{
    env,
    ffi::OsString,
    fs,
    path::Path,
    process::{Command, Stdio},
};

use crate::process::cargo;

use super::options::AdoptionBenchmarkOptions;
use super::summary::{AdoptionStep, StepStatus};
use super::support::{benchmark_env_path, benchmark_env_path_list, criterion_target_dir};

#[cfg(all(test, unix))]
use std::sync::{Mutex, MutexGuard};

#[cfg(all(test, unix))]
static TEST_CARGO_PROGRAM: Mutex<Option<OsString>> = Mutex::new(None);
#[cfg(all(test, unix))]
static TEST_CARGO_SERIAL: Mutex<()> = Mutex::new(());

fn runner_cargo() -> OsString {
    #[cfg(all(test, unix))]
    if let Some(program) = TEST_CARGO_PROGRAM
        .lock()
        .expect("test Cargo program lock")
        .clone()
    {
        return program;
    }
    cargo()
}

#[cfg(all(test, unix))]
pub(super) struct TestCargoProgramGuard {
    _serial: MutexGuard<'static, ()>,
}

#[cfg(all(test, unix))]
pub(super) fn use_test_cargo_program(program: OsString) -> TestCargoProgramGuard {
    let serial = TEST_CARGO_SERIAL.lock().expect("test Cargo serial lock");
    *TEST_CARGO_PROGRAM.lock().expect("test Cargo program lock") = Some(program);
    TestCargoProgramGuard { _serial: serial }
}

#[cfg(all(test, unix))]
impl Drop for TestCargoProgramGuard {
    fn drop(&mut self) {
        *TEST_CARGO_PROGRAM.lock().expect("test Cargo program lock") = None;
    }
}

pub(super) const SCRUBBED_BENCH_ENV_VARS: &[&str] = &[
    "J2K_FIXTURE_COMPARE_MODE",
    "J2K_FIXTURE_COMPARE_REPEATS",
    "J2K_FIXTURE_COMPARE_BATCH_SIZE",
    "J2K_FIXTURE_COMPARE_BATCH_SIZES",
    "J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES",
    "J2K_FIXTURE_COMPARE_MIXED_BATCH_SIZES",
    "J2K_FIXTURE_COMPARE_THREADS",
    "J2K_FIXTURE_COMPARE_INPUT_DIR",
    "J2K_FIXTURE_COMPARE_INPUT_DIRS",
    "J2K_FIXTURE_COMPARE_MANIFEST",
    "J2K_FIXTURE_COMPARE_INCLUDE_GENERATED",
    "J2K_INCLUDE_OPENJPH",
    "J2K_REQUIRE_OPENJPH",
    "J2K_OPENJPH_EXPAND_BIN",
    "J2K_INCLUDE_KAKADU",
    "J2K_REQUIRE_KAKADU",
    "J2K_KDU_EXPAND_BIN",
    "J2K_KDU_COMPRESS_BIN",
    "J2K_ENCODE_COMPARE_REPEATS",
    "J2K_ENCODE_COMPARE_BATCH_SIZES",
    "J2K_ENCODE_COMPARE_CASE_BATCH_SIZES",
    "J2K_ENCODE_COMPARE_MIXED_BATCH_SIZES",
    "J2K_ENCODE_COMPARE_INPUT_DIRS",
    "J2K_ENCODE_COMPARE_MANIFEST",
    "J2K_ENCODE_COMPARE_INCLUDE_GENERATED",
    "J2K_ENCODE_COMPARE_ENCODERS",
    "J2K_REQUIRE_OPENJPEG",
    "J2K_REQUIRE_GROK",
    "J2K_REQUIRE_CUDA_BENCH",
    "J2K_REQUIRE_METAL_BENCH",
    "J2K_CUDA_DECODE_FORMATS",
    "J2K_CUDA_DECODE_BATCH_SIZES",
    "J2K_CUDA_DECODE_INPUT_DIRS",
    "J2K_CUDA_DECODE_MANIFEST",
    "J2K_CUDA_DECODE_INCLUDE_GENERATED",
    "J2K_CUDA_ENCODE_INPUT_DIRS",
    "J2K_CUDA_ENCODE_MANIFEST",
    "J2K_CUDA_ENCODE_INCLUDE_GENERATED",
    "J2K_METAL_DECODE_INPUT_DIRS",
    "J2K_METAL_DECODE_MANIFEST",
    "J2K_METAL_DECODE_INCLUDE_GENERATED",
    "J2K_METAL_ENCODE_INPUT_DIRS",
    "J2K_METAL_ENCODE_MANIFEST",
    "J2K_METAL_ENCODE_INCLUDE_GENERATED",
    "J2K_METAL_ENCODE_RESIDENT_MAX_ESTIMATED_OUTPUT_BYTES",
    "J2K_TRANSCODE_METAL_PROFILE_STAGES",
];

pub(super) const METAL_TRANSCODE_BENCH_FILTER: &str =
    "jpeg_to_htj2k_wsi_integer_53_tile_batch/srgb_ybr420_224_batch_128";

pub(super) fn run_cpu_encode_compare(
    options: &AdoptionBenchmarkOptions,
) -> Result<AdoptionStep, String> {
    let mut envs = vec![
        ("J2K_REQUIRE_OPENJPEG".to_string(), "1".to_string()),
        ("J2K_REQUIRE_GROK".to_string(), "1".to_string()),
    ];
    if options.quick {
        envs.push(("J2K_ENCODE_COMPARE_REPEATS".to_string(), "1".to_string()));
        envs.push((
            "J2K_ENCODE_COMPARE_BATCH_SIZES".to_string(),
            "1".to_string(),
        ));
    }
    if let Some(input_dirs) = &options.encode_input_dirs {
        envs.push((
            "J2K_ENCODE_COMPARE_INPUT_DIRS".to_string(),
            benchmark_env_path_list(input_dirs, "--encode-fixtures")?,
        ));
    }
    if let Some(manifest) = &options.encode_manifest {
        envs.push((
            "J2K_ENCODE_COMPARE_MANIFEST".to_string(),
            benchmark_env_path(manifest, "--encode-manifest")?,
        ));
    }
    if !options.include_generated {
        envs.push((
            "J2K_ENCODE_COMPARE_INCLUDE_GENERATED".to_string(),
            "0".to_string(),
        ));
    }
    if options.kakadu {
        envs.push(("J2K_INCLUDE_KAKADU".to_string(), "1".to_string()));
        if let Some(bin) = env::var_os("J2K_KDU_COMPRESS_BIN") {
            let bin = bin.into_string().map_err(|_| {
                "J2K_KDU_COMPRESS_BIN contains non-Unicode data; set it to a valid path".to_string()
            })?;
            envs.push(("J2K_KDU_COMPRESS_BIN".to_string(), bin));
        }
    }
    if options.require_kakadu {
        envs.push(("J2K_REQUIRE_KAKADU".to_string(), "1".to_string()));
    }

    run_logged(
        "cpu-encode-compare",
        runner_cargo(),
        &[
            "run",
            "-p",
            "j2k-compare",
            "--release",
            "--bin",
            "jp2k_encode_compare",
        ],
        &envs,
        &options.out_dir,
    )
}

pub(super) fn run_cpu_fixture_compare(
    options: &AdoptionBenchmarkOptions,
) -> Result<AdoptionStep, String> {
    let mut envs = vec![
        ("J2K_REQUIRE_OPENJPEG".to_string(), "1".to_string()),
        ("J2K_REQUIRE_GROK".to_string(), "1".to_string()),
    ];
    if options.quick {
        envs.push(("J2K_FIXTURE_COMPARE_REPEATS".to_string(), "1".to_string()));
        envs.push((
            "J2K_FIXTURE_COMPARE_BATCH_SIZES".to_string(),
            "1".to_string(),
        ));
    }
    if let Some(input_dirs) = &options.input_dirs {
        envs.push((
            "J2K_FIXTURE_COMPARE_INPUT_DIRS".to_string(),
            benchmark_env_path_list(input_dirs, "--fixtures")?,
        ));
    }
    if let Some(manifest) = &options.manifest {
        envs.push((
            "J2K_FIXTURE_COMPARE_MANIFEST".to_string(),
            benchmark_env_path(manifest, "--manifest")?,
        ));
    }
    if !options.include_generated {
        envs.push((
            "J2K_FIXTURE_COMPARE_INCLUDE_GENERATED".to_string(),
            "0".to_string(),
        ));
    }
    if options.openjph {
        envs.push(("J2K_INCLUDE_OPENJPH".to_string(), "1".to_string()));
        if let Some(bin) = env::var_os("J2K_OPENJPH_EXPAND_BIN") {
            let bin = bin.into_string().map_err(|_| {
                "J2K_OPENJPH_EXPAND_BIN contains non-Unicode data; set it to a valid path"
                    .to_string()
            })?;
            envs.push(("J2K_OPENJPH_EXPAND_BIN".to_string(), bin));
        }
    }
    if options.require_openjph {
        envs.push(("J2K_REQUIRE_OPENJPH".to_string(), "1".to_string()));
    }
    if options.kakadu {
        envs.push(("J2K_INCLUDE_KAKADU".to_string(), "1".to_string()));
        if let Some(bin) = env::var_os("J2K_KDU_EXPAND_BIN") {
            let bin = bin.into_string().map_err(|_| {
                "J2K_KDU_EXPAND_BIN contains non-Unicode data; set it to a valid path".to_string()
            })?;
            envs.push(("J2K_KDU_EXPAND_BIN".to_string(), bin));
        }
    }
    if options.require_kakadu {
        envs.push(("J2K_REQUIRE_KAKADU".to_string(), "1".to_string()));
    }

    run_logged(
        "cpu-fixture-compare",
        runner_cargo(),
        &[
            "run",
            "-p",
            "j2k-compare",
            "--release",
            "--bin",
            "jp2k_fixture_compare",
        ],
        &envs,
        &options.out_dir,
    )
}

pub(super) fn run_cpu_public_api_encode(
    options: &AdoptionBenchmarkOptions,
) -> Result<AdoptionStep, String> {
    let mut args = vec![
        "bench".to_string(),
        "-p".to_string(),
        "j2k".to_string(),
        "--bench".to_string(),
        "public_api".to_string(),
        "--".to_string(),
        "j2k_public_cpu_encode_matrix".to_string(),
    ];
    if options.quick {
        args.push("--quick".to_string());
    }
    run_logged_owned(
        "cpu-public-api-encode",
        runner_cargo(),
        &args,
        &[],
        Some(&criterion_target_dir(options, "cpu-public-api-encode")),
        &options.out_dir,
    )
}

pub(super) fn run_cpu_public_api_decode(
    options: &AdoptionBenchmarkOptions,
) -> Result<AdoptionStep, String> {
    let mut args = vec![
        "bench".to_string(),
        "-p".to_string(),
        "j2k".to_string(),
        "--bench".to_string(),
        "public_api".to_string(),
        "--".to_string(),
        "j2k_public_decode".to_string(),
    ];
    if options.quick {
        args.push("--quick".to_string());
    }
    run_logged_owned(
        "cpu-public-api-decode",
        runner_cargo(),
        &args,
        &[],
        Some(&criterion_target_dir(options, "cpu-public-api-decode")),
        &options.out_dir,
    )
}

pub(super) fn run_cuda_htj2k_decode(
    options: &AdoptionBenchmarkOptions,
) -> Result<AdoptionStep, String> {
    let mut args = vec![
        "bench".to_string(),
        "-p".to_string(),
        "j2k-cuda".to_string(),
        "--bench".to_string(),
        "htj2k_decode".to_string(),
        "--features".to_string(),
        "cuda-runtime".to_string(),
        "--".to_string(),
        "j2k_cuda_htj2k_".to_string(),
    ];
    if options.quick {
        args.push("--quick".to_string());
    }
    let mut envs = vec![
        (
            "J2K_CUDA_DECODE_FORMATS".to_string(),
            "gray8,rgb8,rgba8".to_string(),
        ),
        (
            "J2K_CUDA_DECODE_BATCH_SIZES".to_string(),
            options
                .cuda_decode_batch_sizes
                .clone()
                .unwrap_or_else(|| "8,16,32,64".to_string()),
        ),
    ];
    if let Some(input_dirs) = &options.input_dirs {
        envs.push((
            "J2K_CUDA_DECODE_INPUT_DIRS".to_string(),
            benchmark_env_path_list(input_dirs, "--fixtures")?,
        ));
    }
    if let Some(manifest) = &options.manifest {
        envs.push((
            "J2K_CUDA_DECODE_MANIFEST".to_string(),
            benchmark_env_path(manifest, "--manifest")?,
        ));
    }
    if !options.include_generated {
        envs.push((
            "J2K_CUDA_DECODE_INCLUDE_GENERATED".to_string(),
            "0".to_string(),
        ));
    }
    if options.require_cuda {
        envs.push(("J2K_REQUIRE_CUDA_BENCH".to_string(), "1".to_string()));
    }
    run_logged_owned(
        "cuda-htj2k-decode",
        runner_cargo(),
        &args,
        &envs,
        Some(&criterion_target_dir(options, "cuda-htj2k-decode")),
        &options.out_dir,
    )
}

pub(super) fn run_cuda_htj2k_encode(
    options: &AdoptionBenchmarkOptions,
) -> Result<AdoptionStep, String> {
    let mut args = vec![
        "bench".to_string(),
        "-p".to_string(),
        "j2k-cuda".to_string(),
        "--bench".to_string(),
        "htj2k_encode".to_string(),
        "--features".to_string(),
        "cuda-runtime".to_string(),
        "--".to_string(),
        "j2k_cuda_htj2k_".to_string(),
    ];
    if options.quick {
        args.push("--quick".to_string());
    }
    let mut envs = Vec::new();
    if let Some(input_dirs) = &options.encode_input_dirs {
        envs.push((
            "J2K_CUDA_ENCODE_INPUT_DIRS".to_string(),
            benchmark_env_path_list(input_dirs, "--encode-fixtures")?,
        ));
    }
    if let Some(manifest) = &options.encode_manifest {
        envs.push((
            "J2K_CUDA_ENCODE_MANIFEST".to_string(),
            benchmark_env_path(manifest, "--encode-manifest")?,
        ));
    }
    if !options.include_generated {
        envs.push((
            "J2K_CUDA_ENCODE_INCLUDE_GENERATED".to_string(),
            "0".to_string(),
        ));
    }
    if options.require_cuda {
        envs.push(("J2K_REQUIRE_CUDA_BENCH".to_string(), "1".to_string()));
        envs.push(("J2K_REQUIRE_CUDA_OXIDE_BUILD".to_string(), "1".to_string()));
    }
    run_logged_owned(
        "cuda-htj2k-encode",
        runner_cargo(),
        &args,
        &envs,
        Some(&criterion_target_dir(options, "cuda-htj2k-encode")),
        &options.out_dir,
    )
}

pub(super) fn run_metal_decode_benchmark(
    options: &AdoptionBenchmarkOptions,
) -> Result<AdoptionStep, String> {
    if env::consts::OS != "macos" && !options.require_metal {
        return Ok(skipped_step(
            "metal-decode-benchmark",
            "not macOS; Metal benchmark is macOS-only",
            &options.out_dir,
        ));
    }
    let args = vec![
        "test".to_string(),
        "-p".to_string(),
        "j2k-metal".to_string(),
        "--release".to_string(),
        "--test".to_string(),
        "metal_decode_benchmark".to_string(),
        "metal_decode_benchmark".to_string(),
        "--".to_string(),
        "--ignored".to_string(),
        "--nocapture".to_string(),
    ];
    let mut envs = Vec::new();
    if let Some(input_dirs) = &options.input_dirs {
        envs.push((
            "J2K_METAL_DECODE_INPUT_DIRS".to_string(),
            benchmark_env_path_list(input_dirs, "--fixtures")?,
        ));
    }
    if let Some(manifest) = &options.manifest {
        envs.push((
            "J2K_METAL_DECODE_MANIFEST".to_string(),
            benchmark_env_path(manifest, "--manifest")?,
        ));
    }
    if !options.include_generated {
        envs.push((
            "J2K_METAL_DECODE_INCLUDE_GENERATED".to_string(),
            "0".to_string(),
        ));
    }
    if options.require_metal {
        envs.push(("J2K_REQUIRE_METAL_BENCH".to_string(), "1".to_string()));
    }
    run_logged_owned(
        "metal-decode-benchmark",
        runner_cargo(),
        &args,
        &envs,
        None,
        &options.out_dir,
    )
}

pub(super) fn run_metal_encode_auto_routing(
    options: &AdoptionBenchmarkOptions,
) -> Result<AdoptionStep, String> {
    if env::consts::OS != "macos" && !options.require_metal {
        return Ok(skipped_step(
            "metal-encode-auto-routing",
            "not macOS; Metal benchmark is macOS-only",
            &options.out_dir,
        ));
    }
    let args = vec![
        "test".to_string(),
        "-p".to_string(),
        "j2k-metal".to_string(),
        "--release".to_string(),
        "--test".to_string(),
        "encode_auto_routing_benchmark".to_string(),
        "encode_auto_routing_benchmark".to_string(),
        "--".to_string(),
        "--ignored".to_string(),
        "--nocapture".to_string(),
    ];
    let mut envs = Vec::new();
    if let Some(input_dirs) = &options.encode_input_dirs {
        envs.push((
            "J2K_METAL_ENCODE_INPUT_DIRS".to_string(),
            benchmark_env_path_list(input_dirs, "--encode-fixtures")?,
        ));
    }
    if let Some(manifest) = &options.encode_manifest {
        envs.push((
            "J2K_METAL_ENCODE_MANIFEST".to_string(),
            benchmark_env_path(manifest, "--encode-manifest")?,
        ));
    }
    if !options.include_generated {
        envs.push((
            "J2K_METAL_ENCODE_INCLUDE_GENERATED".to_string(),
            "0".to_string(),
        ));
    }
    if options.require_metal {
        envs.push(("J2K_REQUIRE_METAL_BENCH".to_string(), "1".to_string()));
    }
    run_logged_owned(
        "metal-encode-auto-routing",
        runner_cargo(),
        &args,
        &envs,
        None,
        &options.out_dir,
    )
}

pub(super) fn run_metal_transcode_benchmark(
    options: &AdoptionBenchmarkOptions,
) -> Result<AdoptionStep, String> {
    if env::consts::OS != "macos" && !options.require_metal {
        return Ok(skipped_step(
            "metal-transcode-benchmark",
            "not macOS; Metal transcode benchmark is macOS-only",
            &options.out_dir,
        ));
    }
    let args = vec![
        "bench".to_string(),
        "-p".to_string(),
        "j2k-transcode-metal".to_string(),
        "--bench".to_string(),
        "dct97".to_string(),
        "--".to_string(),
        METAL_TRANSCODE_BENCH_FILTER.to_string(),
    ];
    let mut envs = vec![(
        "J2K_TRANSCODE_METAL_PROFILE_STAGES".to_string(),
        "1".to_string(),
    )];
    if options.require_metal {
        envs.push(("J2K_REQUIRE_METAL_BENCH".to_string(), "1".to_string()));
    }
    run_logged_owned(
        "metal-transcode-benchmark",
        runner_cargo(),
        &args,
        &envs,
        Some(&criterion_target_dir(options, "metal-transcode-benchmark")),
        &options.out_dir,
    )
}

pub(super) fn run_logged(
    name: &'static str,
    program: OsString,
    args: &[&str],
    envs: &[(String, String)],
    out_dir: &Path,
) -> Result<AdoptionStep, String> {
    let args = args
        .iter()
        .map(|arg| (*arg).to_string())
        .collect::<Vec<_>>();
    run_logged_owned(name, program, &args, envs, None, out_dir)
}

pub(super) fn run_logged_owned(
    name: &'static str,
    program: impl AsRef<std::ffi::OsStr>,
    args: &[String],
    envs: &[(String, String)],
    target_dir: Option<&Path>,
    out_dir: &Path,
) -> Result<AdoptionStep, String> {
    let program = program.as_ref();
    let stdout = out_dir.join(format!("{name}.out"));
    let stderr = out_dir.join(format!("{name}.err"));
    let stdout_file = fs::File::create(&stdout)
        .map_err(|err| format!("failed to create {}: {err}", stdout.display()))?;
    let stderr_file = fs::File::create(&stderr)
        .map_err(|err| format!("failed to create {}: {err}", stderr.display()))?;

    let display = display_command(program, args, envs, target_dir);
    eprintln!("+ {display}");
    let mut command = Command::new(program);
    command
        .args(args)
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));
    for key in SCRUBBED_BENCH_ENV_VARS {
        command.env_remove(key);
    }
    for (key, value) in envs {
        command.env(key, value);
    }
    if let Some(target_dir) = target_dir {
        command.env("CARGO_TARGET_DIR", target_dir);
    }
    let status = command
        .status()
        .map_err(|err| format!("failed to start `{}`: {err}", program.to_string_lossy()))?;
    if !status.success() {
        return Err(format!(
            "`{}` exited with {status}; stdout={} stderr={}",
            program.to_string_lossy(),
            stdout.display(),
            stderr.display()
        ));
    }
    Ok(AdoptionStep {
        name,
        command: display,
        stdout,
        stderr,
        criterion_root: target_dir.map(|path| path.join("criterion")),
        status: StepStatus::Ran,
    })
}

pub(super) fn skipped_step(name: &'static str, reason: &str, out_dir: &Path) -> AdoptionStep {
    AdoptionStep {
        name,
        command: "not run".to_string(),
        stdout: out_dir.join(format!("{name}.out")),
        stderr: out_dir.join(format!("{name}.err")),
        criterion_root: None,
        status: StepStatus::Skipped {
            reason: reason.to_string(),
        },
    }
}

pub(super) fn display_command(
    program: &std::ffi::OsStr,
    args: &[String],
    envs: &[(String, String)],
    target_dir: Option<&Path>,
) -> String {
    let mut parts = vec!["env".to_string()];
    parts.extend(
        SCRUBBED_BENCH_ENV_VARS
            .iter()
            .map(|key| format!("-u {key}")),
    );
    parts.extend(
        envs.iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>(),
    );
    if let Some(target_dir) = target_dir {
        parts.push(format!("CARGO_TARGET_DIR={}", target_dir.display()));
    }
    parts.push(program.to_string_lossy().into_owned());
    parts.extend(args.iter().cloned());
    parts.join(" ")
}

#[cfg(all(test, unix))]
mod tests;
