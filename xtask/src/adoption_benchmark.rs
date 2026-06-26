use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::perf_guard::{discover_estimates, BenchEstimate};

const SCRUBBED_BENCH_ENV_VARS: &[&str] = &[
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
    "J2K_METAL_ENCODE_INPUT_DIRS",
    "J2K_METAL_ENCODE_MANIFEST",
    "J2K_METAL_ENCODE_INCLUDE_GENERATED",
    "J2K_METAL_ENCODE_RESIDENT_MAX_ESTIMATED_OUTPUT_BYTES",
];

#[derive(Debug, Clone)]
pub(crate) struct AdoptionBenchmarkOptions {
    out_dir: PathBuf,
    input_dirs: Option<String>,
    manifest: Option<PathBuf>,
    encode_input_dirs: Option<String>,
    encode_manifest: Option<PathBuf>,
    cuda_decode_batch_sizes: Option<String>,
    include_generated: bool,
    quick: bool,
    cuda: bool,
    metal: bool,
    openjph: bool,
    kakadu: bool,
    require_cuda: bool,
    require_metal: bool,
    require_openjph: bool,
    require_kakadu: bool,
    finalize_existing: bool,
}

#[derive(Debug)]
struct AdoptionStep {
    name: &'static str,
    command: String,
    stdout: PathBuf,
    stderr: PathBuf,
    criterion_root: Option<PathBuf>,
    status: StepStatus,
}

#[derive(Debug)]
enum StepStatus {
    Ran,
    Skipped { reason: String },
}

pub(crate) fn adoption_benchmark(args: impl Iterator<Item = String>) -> Result<(), String> {
    let args = args.collect::<Vec<_>>();
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        println!("{}", help_text());
        return Ok(());
    }
    let options = AdoptionBenchmarkOptions::parse(args.into_iter())?;
    fs::create_dir_all(&options.out_dir)
        .map_err(|err| format!("failed to create {}: {err}", options.out_dir.display()))?;

    if options.finalize_existing {
        let steps = existing_steps(&options)?;
        write_summary(&options, &steps)?;
        write_readme(&options, &steps)?;
        enforce_publication_gate(&options)?;
        eprintln!(
            "finalized existing adoption benchmark artifacts under {}",
            options.out_dir.display()
        );
        return Ok(());
    }

    let mut steps = vec![
        run_cpu_fixture_compare(&options)?,
        run_cpu_encode_compare(&options)?,
        run_cpu_public_api_encode(&options)?,
        run_cpu_public_api_decode(&options)?,
    ];

    if options.cuda {
        steps.push(run_cuda_htj2k_decode(&options)?);
        steps.push(run_cuda_htj2k_encode(&options)?);
    } else {
        steps.push(skipped_step(
            "cuda-htj2k-decode",
            "not requested; pass --cuda for CUDA decode/encode Criterion benches",
            &options.out_dir,
        ));
        steps.push(skipped_step(
            "cuda-htj2k-encode",
            "not requested; pass --cuda for CUDA decode/encode Criterion benches",
            &options.out_dir,
        ));
    }

    if options.metal {
        steps.push(run_metal_encode_auto_routing(&options)?);
    } else {
        steps.push(skipped_step(
            "metal-encode-auto-routing",
            "not requested; pass --metal for Metal hybrid encode routing benchmark",
            &options.out_dir,
        ));
    }

    write_summary(&options, &steps)?;
    write_readme(&options, &steps)?;
    enforce_publication_gate(&options)?;
    eprintln!(
        "wrote adoption benchmark artifacts under {}",
        options.out_dir.display()
    );
    Ok(())
}

fn run_cpu_encode_compare(options: &AdoptionBenchmarkOptions) -> Result<AdoptionStep, String> {
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
        cargo(),
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

fn run_cpu_fixture_compare(options: &AdoptionBenchmarkOptions) -> Result<AdoptionStep, String> {
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
        cargo(),
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

fn run_cpu_public_api_encode(options: &AdoptionBenchmarkOptions) -> Result<AdoptionStep, String> {
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
        cargo(),
        &args,
        &[],
        Some(&criterion_target_dir(options, "cpu-public-api-encode")),
        &options.out_dir,
    )
}

fn run_cpu_public_api_decode(options: &AdoptionBenchmarkOptions) -> Result<AdoptionStep, String> {
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
        cargo(),
        &args,
        &[],
        Some(&criterion_target_dir(options, "cpu-public-api-decode")),
        &options.out_dir,
    )
}

fn run_cuda_htj2k_decode(options: &AdoptionBenchmarkOptions) -> Result<AdoptionStep, String> {
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
        cargo(),
        &args,
        &envs,
        Some(&criterion_target_dir(options, "cuda-htj2k-decode")),
        &options.out_dir,
    )
}

fn run_cuda_htj2k_encode(options: &AdoptionBenchmarkOptions) -> Result<AdoptionStep, String> {
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
        envs.push(("J2K_REQUIRE_CUDA_KERNEL_BUILD".to_string(), "1".to_string()));
    }
    run_logged_owned(
        "cuda-htj2k-encode",
        cargo(),
        &args,
        &envs,
        Some(&criterion_target_dir(options, "cuda-htj2k-encode")),
        &options.out_dir,
    )
}

fn run_metal_encode_auto_routing(
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
        cargo(),
        &args,
        &envs,
        None,
        &options.out_dir,
    )
}

fn run_logged(
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

fn run_logged_owned(
    name: &'static str,
    program: OsString,
    args: &[String],
    envs: &[(String, String)],
    target_dir: Option<&Path>,
    out_dir: &Path,
) -> Result<AdoptionStep, String> {
    let stdout = out_dir.join(format!("{name}.out"));
    let stderr = out_dir.join(format!("{name}.err"));
    let stdout_file = fs::File::create(&stdout)
        .map_err(|err| format!("failed to create {}: {err}", stdout.display()))?;
    let stderr_file = fs::File::create(&stderr)
        .map_err(|err| format!("failed to create {}: {err}", stderr.display()))?;

    let display = display_command(&program, args, envs, target_dir);
    eprintln!("+ {display}");
    let mut command = Command::new(&program);
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

fn skipped_step(name: &'static str, reason: &str, out_dir: &Path) -> AdoptionStep {
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

fn existing_steps(options: &AdoptionBenchmarkOptions) -> Result<Vec<AdoptionStep>, String> {
    let mut steps = vec![
        existing_ran_step("cpu-fixture-compare", None, &options.out_dir)?,
        existing_ran_step("cpu-encode-compare", None, &options.out_dir)?,
        existing_ran_step(
            "cpu-public-api-encode",
            Some(&criterion_target_dir(options, "cpu-public-api-encode")),
            &options.out_dir,
        )?,
        existing_ran_step(
            "cpu-public-api-decode",
            Some(&criterion_target_dir(options, "cpu-public-api-decode")),
            &options.out_dir,
        )?,
    ];

    if options.cuda {
        steps.push(existing_ran_step(
            "cuda-htj2k-decode",
            Some(&criterion_target_dir(options, "cuda-htj2k-decode")),
            &options.out_dir,
        )?);
        steps.push(existing_ran_step(
            "cuda-htj2k-encode",
            Some(&criterion_target_dir(options, "cuda-htj2k-encode")),
            &options.out_dir,
        )?);
    } else {
        steps.push(skipped_step(
            "cuda-htj2k-decode",
            "not requested; pass --cuda for CUDA decode/encode Criterion benches",
            &options.out_dir,
        ));
        steps.push(skipped_step(
            "cuda-htj2k-encode",
            "not requested; pass --cuda for CUDA decode/encode Criterion benches",
            &options.out_dir,
        ));
    }

    if options.metal {
        steps.push(existing_ran_step(
            "metal-encode-auto-routing",
            None,
            &options.out_dir,
        )?);
    } else {
        steps.push(skipped_step(
            "metal-encode-auto-routing",
            "not requested; pass --metal for Metal hybrid encode routing benchmark",
            &options.out_dir,
        ));
    }

    Ok(steps)
}

fn existing_ran_step(
    name: &'static str,
    target_dir: Option<&Path>,
    out_dir: &Path,
) -> Result<AdoptionStep, String> {
    let stdout = out_dir.join(format!("{name}.out"));
    let stderr = out_dir.join(format!("{name}.err"));
    if !stdout.is_file() {
        return Err(format!(
            "--finalize-existing requires completed {name} stdout at {}",
            stdout.display()
        ));
    }
    let stdout_len = stdout
        .metadata()
        .map_err(|err| format!("stat {}: {err}", stdout.display()))?
        .len();
    if stdout_len == 0 {
        return Err(format!(
            "--finalize-existing found empty {name} stdout at {}",
            stdout.display()
        ));
    }
    if !stderr.exists() {
        return Err(format!(
            "--finalize-existing requires {name} stderr at {}",
            stderr.display()
        ));
    }
    Ok(AdoptionStep {
        name,
        command: "existing artifact reused by --finalize-existing".to_string(),
        stdout,
        stderr,
        criterion_root: target_dir.map(|path| path.join("criterion")),
        status: StepStatus::Ran,
    })
}

fn write_summary(options: &AdoptionBenchmarkOptions, steps: &[AdoptionStep]) -> Result<(), String> {
    let cpu_fixture_metadata = read_tsv_metadata(
        &options.out_dir.join("cpu-fixture-compare.out"),
        &[
            "benchmark_mode",
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "selected_cases",
            "external_case_count",
            "external_native_case_count",
            "external_materialized_case_count",
            "external_unique_input_count",
            "external_native_unique_input_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "generated_case_count",
            "mode_excluded_case_count",
            "skipped_comparators",
            "publication_gate_skipped_comparators",
            "build_profile",
            "debug_assertions",
            "git_revision",
            "git_dirty",
            "host_hardware",
            "openjpeg_version",
            "grok_version",
            "openjph_included",
            "openjph_available",
            "openjph_expand_command",
            "openjph_version",
            "kakadu_included",
            "kakadu_available",
            "kakadu_expand_command",
            "kakadu_version",
        ],
    )
    .unwrap_or_else(|error| {
        serde_json::json!({
            "metadata_error": error,
        })
    });
    let cpu_encode_metadata = read_tsv_metadata(
        &options.out_dir.join("cpu-encode-compare.out"),
        &[
            "benchmark_mode",
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "selected_encoders",
            "selected_cases",
            "external_case_count",
            "external_unique_input_count",
            "external_manifest_covered_case_count",
            "external_manifest_missing_case_count",
            "external_component_group_count",
            "external_dimension_count",
            "external_source_format_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "generated_case_count",
            "encode_manifest",
            "openjpeg_compress_available",
            "grok_compress_available",
            "build_profile",
            "debug_assertions",
            "git_revision",
            "git_dirty",
            "host_hardware",
            "openjpeg_version",
            "openjpeg_linked_library_version",
            "grok_version",
            "grok_linked_library_version",
            "kakadu_included",
            "kakadu_compress_available",
            "kakadu_compress_command",
            "kakadu_version",
        ],
    )
    .unwrap_or_else(|error| {
        serde_json::json!({
            "metadata_error": error,
        })
    });
    let criterion_estimates = criterion_summary_json(steps);
    let cuda_decode_metadata = read_tsv_metadata(
        &options.out_dir.join("cuda-htj2k-decode.out"),
        &[
            "j2k_cuda_decode_generated_included",
            "j2k_cuda_decode_batch_sizes",
            "j2k_cuda_decode_io_policy",
            "j2k_cuda_decode_input_dirs",
            "j2k_cuda_decode_manifest",
            "j2k_cuda_decode_case_count",
            "j2k_cuda_decode_external_case_count",
            "j2k_cuda_decode_external_fixture_count",
            "j2k_cuda_decode_external_skipped_non_htj2k_count",
            "j2k_cuda_decode_external_skipped_unsupported_shape_count",
            "j2k_cuda_decode_external_skipped_format_disabled_count",
        ],
    )
    .unwrap_or_else(|error| {
        serde_json::json!({
            "metadata_error": error,
        })
    });
    let metal_summary = read_metal_encode_summary(
        &options.out_dir.join("metal-encode-auto-routing.out"),
        steps,
    );
    let cuda_encode_metadata = read_tsv_metadata(
        &options.out_dir.join("cuda-htj2k-encode.out"),
        &[
            "j2k_cuda_encode_generated_host_input_included",
            "j2k_cuda_encode_io_policy",
            "j2k_cuda_encode_input_dirs",
            "j2k_cuda_encode_manifest",
            "j2k_cuda_encode_external_case_count",
            "j2k_cuda_encode_external_input_format",
            "j2k_cuda_encode_external_case_sources",
        ],
    )
    .unwrap_or_else(|error| {
        serde_json::json!({
            "metadata_error": error,
        })
    });
    let value = serde_json::json!({
        "version": 1,
        "created_unix_seconds": unix_seconds(),
        "mode": if options.quick { "quick" } else { "full" },
        "input_dirs": options.input_dirs,
        "manifest": options.manifest.as_ref().map(|path| path.display().to_string()),
        "encode_input_dirs": options.encode_input_dirs,
        "encode_manifest": options.encode_manifest.as_ref().map(|path| path.display().to_string()),
        "cuda_decode_batch_sizes": options.cuda_decode_batch_sizes,
        "include_generated": options.include_generated,
        "cuda_requested": options.cuda,
        "metal_requested": options.metal,
        "openjph_requested": options.openjph,
        "kakadu_requested": options.kakadu,
        "require_cuda": options.require_cuda,
        "require_metal": options.require_metal,
        "require_openjph": options.require_openjph,
        "require_kakadu": options.require_kakadu,
        "cpu_fixture_compare": cpu_fixture_metadata,
        "cpu_encode_compare": cpu_encode_metadata,
        "cuda_htj2k_decode": cuda_decode_metadata,
        "cuda_htj2k_encode": cuda_encode_metadata,
        "criterion": criterion_estimates,
        "metal_encode_auto_routing": metal_summary,
        "steps": steps.iter().map(step_json).collect::<Vec<_>>(),
        "scrubbed_env_vars": SCRUBBED_BENCH_ENV_VARS,
        "fixture_comparability_scope": "cpu-fixture-compare uses external encoded fixtures and requires independently sourced native compressed J2K and HTJ2K fixtures for publishable decode claims; repo-materialized natural-image codestreams are useful workload diagnostics but do not satisfy native compressed codec coverage by themselves. Optional OpenJPH rows are opt-in HTJ2K/JPH-compatible CLI rows labeled by decode_method and are not part of the default J2K/OpenJPEG/Grok in-process matrix; optional Kakadu rows are opt-in CLI/file-output context rows labeled by decode_method or encode_method and are not part of the default in-process/publication matrix. cuda-htj2k-decode consumes the same external fixture dirs and manifest when --fixtures/--manifest are provided but measures only the supported HTJ2K subset and reports skipped fixture counts. cpu-encode-compare is classic lossless J2K-in-JP2 only and uses external source images staged to identical PNM bytes outside the timed loop; it is not an HTJ2K encode benchmark. cuda-htj2k-encode and metal-encode-auto-routing consume staged PGM/PPM source images from --encode-fixtures/--encode-manifest when supplied and label external host-input rows separately from generated component rows. Metal resident rows are HTJ2K lossless host-output comparisons only when packetization_used=true and codestream_assembly_used=true; resident_buffer_ms is GPU-pipeline context and not a direct CPU codec comparison. CPU public API rows remain component microbenchmarks",
        "publication_note": "CPU fixture compare and CPU encode compare must both report publication_eligible=true, publication_blockers=none, and benchmark_complete=true before use in adoption claims; CPU decode publishability also requires independent native compressed classic J2K and HTJ2K coverage, not only codestreams generated by this repo. CPU encode rows are classic lossless JP2 only. CUDA decode hardware rows must be run with --require-cuda and the same pinned fixture manifest for supported-HTJ2K-subset claims; CUDA encode hardware rows must be run with --require-cuda and J2K_CUDA_ENCODE_MANIFEST-backed staged PNM sources before they are described as using the same encode source matrix; Metal encode hardware rows must be run with --require-metal and J2K_METAL_ENCODE_MANIFEST-backed staged PNM sources before they are described as using the same encode source matrix. Use resident_host_ms vs cpu_ms for resident Metal speed claims; keep j2k_metal_encode_auto_bench and resident_buffer_ms labeled separately."
    });
    let data = serde_json::to_string_pretty(&value)
        .map_err(|err| format!("failed to serialize adoption benchmark summary: {err}"))?;
    let path = options.out_dir.join("summary.json");
    fs::write(&path, format!("{data}\n"))
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn criterion_estimate_json(estimate: &BenchEstimate) -> serde_json::Value {
    serde_json::json!({
        "id": estimate.id,
        "median_ns": estimate.median_ns,
        "median_lower_ns": estimate.median_lower_ns,
        "median_upper_ns": estimate.median_upper_ns,
    })
}

fn criterion_summary_json(steps: &[AdoptionStep]) -> serde_json::Value {
    let mut total_count = 0_usize;
    let mut step_summaries = Vec::new();
    let mut all_estimates = Vec::new();
    for step in steps {
        let Some(root) = &step.criterion_root else {
            continue;
        };
        if !matches!(&step.status, StepStatus::Ran) {
            continue;
        }
        if !root.exists() {
            step_summaries.push(serde_json::json!({
                "step": step.name,
                "root": root.display().to_string(),
                "count": 0,
                "note": "no Criterion output produced",
            }));
            continue;
        }
        match discover_estimates(root) {
            Ok(estimates) => {
                total_count += estimates.len();
                all_estimates.extend(estimates.iter().map(criterion_estimate_json));
                step_summaries.push(serde_json::json!({
                    "step": step.name,
                    "root": root.display().to_string(),
                    "count": estimates.len(),
                    "estimates": estimates.iter().map(criterion_estimate_json).collect::<Vec<_>>(),
                }));
            }
            Err(error) => step_summaries.push(serde_json::json!({
                "step": step.name,
                "root": root.display().to_string(),
                "error": error,
            })),
        }
    }

    serde_json::json!({
        "count": total_count,
        "steps": step_summaries,
        "estimates": all_estimates,
    })
}

fn read_metal_encode_summary(path: &Path, steps: &[AdoptionStep]) -> serde_json::Value {
    let Some(step) = steps
        .iter()
        .find(|step| step.name == "metal-encode-auto-routing")
    else {
        return serde_json::json!({
            "output": path.display().to_string(),
            "status": "missing-step",
        });
    };
    if let StepStatus::Skipped { reason } = &step.status {
        return serde_json::json!({
            "output": path.display().to_string(),
            "status": "skipped",
            "reason": reason,
        });
    }
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            return serde_json::json!({
                "output": path.display().to_string(),
                "status": "error",
                "error": format!("failed to read Metal benchmark output: {error}"),
            });
        }
    };

    let mut auto_benches = Vec::new();
    let mut auto_probes = Vec::new();
    let mut stage_benches = Vec::new();
    let mut resident_benches = Vec::new();
    let mut metadata = serde_json::Map::new();
    for line in text.lines() {
        if let Some(row) = parse_metal_auto_bench_line(line) {
            auto_benches.push(row);
        } else if let Some(row) = parse_metal_auto_probe_line(line) {
            auto_probes.push(row);
        } else if let Some(row) = parse_metal_stage_bench_line(line) {
            stage_benches.push(row);
        } else if let Some(row) = parse_metal_resident_bench_line(line) {
            resident_benches.push(row);
        } else if let Some((key, value)) = line.split_once('\t') {
            if key.starts_with("j2k_metal_encode_") {
                metadata.insert(
                    key.to_string(),
                    serde_json::Value::String(value.to_string()),
                );
            }
        }
    }

    let skipped_auto_benches = auto_benches
        .iter()
        .filter(|row| row.get("auto_ms").and_then(serde_json::Value::as_str) == Some("skipped"))
        .count();
    let skipped_stage_benches = stage_benches
        .iter()
        .filter(|row| row.get("metal_ms").and_then(serde_json::Value::as_str) == Some("skipped"))
        .count();
    let skipped_resident_benches = resident_benches
        .iter()
        .filter(|row| {
            row.get("resident_host_ms")
                .and_then(serde_json::Value::as_str)
                == Some("skipped")
                || row
                    .get("resident_buffer_ms")
                    .and_then(serde_json::Value::as_str)
                    == Some("skipped")
        })
        .count();
    let resident_verified_benches = resident_benches
        .iter()
        .filter(|row| {
            row.get("packetization_used")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
                && row
                    .get("codestream_assembly_used")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
                && row
                    .get("resident_host_ms")
                    .and_then(serde_json::Value::as_f64)
                    .is_some()
                && row
                    .get("resident_buffer_ms")
                    .and_then(serde_json::Value::as_f64)
                    .is_some()
        })
        .count();
    let probe_errors = auto_probes
        .iter()
        .filter(|row| row.get("error").is_some())
        .count();

    serde_json::json!({
        "output": path.display().to_string(),
        "status": "ran",
        "auto_bench_count": auto_benches.len(),
        "auto_probe_count": auto_probes.len(),
        "stage_bench_count": stage_benches.len(),
        "resident_bench_count": resident_benches.len(),
        "skipped_auto_bench_count": skipped_auto_benches,
        "skipped_stage_bench_count": skipped_stage_benches,
        "skipped_resident_bench_count": skipped_resident_benches,
        "resident_verified_bench_count": resident_verified_benches,
        "probe_error_count": probe_errors,
        "metadata": metadata,
        "auto_benches": auto_benches,
        "auto_probes": auto_probes,
        "stage_benches": stage_benches,
        "resident_benches": resident_benches,
    })
}

fn parse_metal_auto_bench_line(line: &str) -> Option<serde_json::Value> {
    const PREFIX: &str = "j2k_metal_encode_auto_bench ";
    let fields = parse_key_value_fields(line.strip_prefix(PREFIX)?);
    let auto_ms = required_field(&fields, "auto_ms")?;
    Some(serde_json::json!({
        "mode": required_field(&fields, "mode")?,
        "codec": required_field(&fields, "codec")?,
        "components": required_field(&fields, "components")?,
        "size": required_field(&fields, "size")?,
        "cpu_ms": parse_decimal_field(&fields, "cpu_ms")?,
        "auto_ms": parse_optional_decimal(auto_ms)?,
    }))
}

fn parse_metal_auto_probe_line(line: &str) -> Option<serde_json::Value> {
    const PREFIX: &str = "j2k_metal_encode_auto_probe ";
    let rest = line.strip_prefix(PREFIX)?;
    let fields = parse_key_value_fields(rest);
    let mut row = serde_json::json!({
        "mode": required_field(&fields, "mode")?,
        "codec": required_field(&fields, "codec")?,
        "components": required_field(&fields, "components")?,
        "size": required_field(&fields, "size")?,
    });
    if let Some(dispatch) = suffix_after_key(rest, "dispatch") {
        row["dispatch"] = serde_json::Value::String(dispatch.to_string());
    }
    if let Some(error) = suffix_after_key(rest, "error") {
        row["error"] = serde_json::Value::String(error.to_string());
    }
    Some(row)
}

fn parse_metal_stage_bench_line(line: &str) -> Option<serde_json::Value> {
    const PREFIX: &str = "j2k_metal_encode_stage_bench ";
    let rest = line.strip_prefix(PREFIX)?;
    let fields = parse_key_value_fields(rest);
    let metal_ms = required_field(&fields, "metal_ms")?;
    let mut row = serde_json::json!({
        "stage": required_field(&fields, "stage")?,
        "size": required_field(&fields, "size")?,
        "cpu_ms": parse_decimal_field(&fields, "cpu_ms")?,
        "metal_ms": parse_optional_decimal(metal_ms)?,
    });
    if let Some(dispatch) = suffix_after_key(rest, "dispatch") {
        row["dispatch"] = serde_json::Value::String(dispatch.to_string());
    }
    if let Some(error) = suffix_after_key(rest, "error") {
        row["error"] = serde_json::Value::String(error.to_string());
    }
    Some(row)
}

fn parse_metal_resident_bench_line(line: &str) -> Option<serde_json::Value> {
    const PREFIX: &str = "j2k_metal_encode_resident_bench ";
    let rest = line.strip_prefix(PREFIX)?;
    let fields = parse_key_value_fields(rest);
    let mut row = serde_json::json!({
        "mode": required_field(&fields, "mode")?,
        "codec": required_field(&fields, "codec")?,
        "components": required_field(&fields, "components")?,
        "size": required_field(&fields, "size")?,
        "batch_size": parse_integer_field(&fields, "batch_size")?,
        "fixture_count": parse_integer_field(&fields, "fixture_count")?,
        "cpu_ms": parse_decimal_or_label_field(&fields, "cpu_ms")?,
        "hybrid_cpu_packet_ms": parse_decimal_or_label_field(&fields, "hybrid_cpu_packet_ms")?,
        "resident_host_ms": parse_decimal_or_label_field(&fields, "resident_host_ms")?,
        "resident_buffer_ms": parse_decimal_or_label_field(&fields, "resident_buffer_ms")?,
        "packetization_used": parse_bool_field(&fields, "packetization_used")?,
        "codestream_assembly_used": parse_bool_field(&fields, "codestream_assembly_used")?,
        "host_readback_ms": parse_decimal_or_label_field(&fields, "host_readback_ms")?,
        "gpu_ms": parse_decimal_or_label_field(&fields, "gpu_ms")?,
        "encoded_host_bytes": parse_integer_or_label_field(&fields, "encoded_host_bytes")?,
        "encoded_buffer_bytes": parse_integer_or_label_field(&fields, "encoded_buffer_bytes")?,
    });
    if let Some(error) = suffix_after_key(rest, "error") {
        row["error"] = serde_json::Value::String(error.to_string());
    }
    if let Some(value) = optional_field(&fields, "resident_input_storage") {
        row["resident_input_storage"] = serde_json::Value::String(value.to_string());
    }
    if let Some(value) = optional_field(&fields, "resident_staging") {
        row["resident_staging"] = serde_json::Value::String(value.to_string());
    }
    Some(row)
}

fn parse_key_value_fields(text: &str) -> Vec<(&str, &str)> {
    text.split_whitespace()
        .filter_map(|part| part.split_once('='))
        .collect()
}

fn required_field(fields: &[(&str, &str)], key: &str) -> Option<String> {
    fields
        .iter()
        .find_map(|(field_key, value)| (*field_key == key).then(|| (*value).to_string()))
}

fn optional_field<'a>(fields: &'a [(&str, &str)], key: &str) -> Option<&'a str> {
    fields
        .iter()
        .find_map(|(field_key, value)| (*field_key == key).then_some(*value))
}

fn parse_decimal_field(fields: &[(&str, &str)], key: &str) -> Option<f64> {
    required_field(fields, key)?.parse().ok()
}

fn parse_integer_field(fields: &[(&str, &str)], key: &str) -> Option<u64> {
    required_field(fields, key)?.parse().ok()
}

fn parse_bool_field(fields: &[(&str, &str)], key: &str) -> Option<bool> {
    match required_field(fields, key)?.as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn parse_decimal_or_label_field(fields: &[(&str, &str)], key: &str) -> Option<serde_json::Value> {
    parse_decimal_or_label(required_field(fields, key)?)
}

fn parse_integer_or_label_field(fields: &[(&str, &str)], key: &str) -> Option<serde_json::Value> {
    let value = required_field(fields, key)?;
    if let Ok(number) = value.parse::<u64>() {
        return Some(serde_json::Value::Number(number.into()));
    }
    Some(serde_json::Value::String(value))
}

fn parse_decimal_or_label(value: String) -> Option<serde_json::Value> {
    if let Ok(number) = value.parse::<f64>() {
        return serde_json::Number::from_f64(number).map(serde_json::Value::Number);
    }
    Some(serde_json::Value::String(value))
}

fn parse_optional_decimal(value: String) -> Option<serde_json::Value> {
    if value == "skipped" {
        return Some(serde_json::Value::String(value));
    }
    serde_json::Number::from_f64(value.parse().ok()?).map(serde_json::Value::Number)
}

fn suffix_after_key<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!(" {key}=");
    let start = text.find(&needle)? + needle.len();
    Some(&text[start..])
}

fn read_tsv_metadata(path: &Path, keys: &[&str]) -> Result<serde_json::Value, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let mut map = serde_json::Map::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('\t') else {
            continue;
        };
        if keys.contains(&key) {
            map.insert(
                key.to_string(),
                serde_json::Value::String(value.to_string()),
            );
        }
    }
    if map.is_empty() {
        return Err(format!("{} contained no fixture metadata", path.display()));
    }
    Ok(serde_json::Value::Object(map))
}

fn enforce_publication_gate(options: &AdoptionBenchmarkOptions) -> Result<(), String> {
    if options.quick || options.include_generated {
        return Ok(());
    }
    let fixture_metadata = read_tsv_metadata(
        &options.out_dir.join("cpu-fixture-compare.out"),
        &[
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
        ],
    )?;
    let encode_metadata = read_tsv_metadata(
        &options.out_dir.join("cpu-encode-compare.out"),
        &[
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
        ],
    )?;
    let mut issues = Vec::new();
    collect_publication_gate_issues("cpu-fixture-compare", &fixture_metadata, &mut issues);
    collect_publication_gate_issues("cpu-encode-compare", &encode_metadata, &mut issues);
    if issues.is_empty() {
        return Ok(());
    }
    Err(format!(
        "adoption benchmark is not publishable: {}; artifacts were written under {}. Use --quick or --include-generated only for smoke/diagnostic runs.",
        issues.join("; "),
        options.out_dir.display()
    ))
}

fn collect_publication_gate_issues(
    label: &str,
    metadata: &serde_json::Value,
    issues: &mut Vec<String>,
) {
    if metadata
        .get("publication_eligible")
        .and_then(serde_json::Value::as_str)
        != Some("true")
    {
        let blockers = metadata
            .get("publication_blockers")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("not-recorded");
        issues.push(format!(
            "{label} publication_eligible=false blockers={blockers}"
        ));
    }
    if metadata
        .get("publication_blockers")
        .and_then(serde_json::Value::as_str)
        != Some("none")
    {
        let blockers = metadata
            .get("publication_blockers")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("not-recorded");
        issues.push(format!("{label} publication_blockers={blockers}"));
    }
    if metadata
        .get("benchmark_complete")
        .and_then(serde_json::Value::as_str)
        != Some("true")
    {
        issues.push(format!("{label} benchmark_complete is not true"));
    }
}

fn step_json(step: &AdoptionStep) -> serde_json::Value {
    match &step.status {
        StepStatus::Ran => serde_json::json!({
            "name": step.name,
            "status": "ran",
            "command": step.command,
            "stdout": step.stdout.display().to_string(),
            "stderr": step.stderr.display().to_string(),
            "criterion_root": step.criterion_root.as_ref().map(|path| path.display().to_string()),
        }),
        StepStatus::Skipped { reason } => serde_json::json!({
            "name": step.name,
            "status": "skipped",
            "reason": reason,
            "command": step.command,
            "stdout": step.stdout.display().to_string(),
            "stderr": step.stderr.display().to_string(),
            "criterion_root": step.criterion_root.as_ref().map(|path| path.display().to_string()),
        }),
    }
}

fn write_readme(options: &AdoptionBenchmarkOptions, steps: &[AdoptionStep]) -> Result<(), String> {
    let cpu_metadata = read_tsv_metadata(
        &options.out_dir.join("cpu-fixture-compare.out"),
        &[
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "external_unique_input_count",
            "external_component_group_count",
            "external_dimension_count",
            "external_source_format_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "publication_gate_skipped_comparators",
            "openjph_included",
            "openjph_available",
            "openjph_expand_command",
            "openjph_version",
            "kakadu_included",
            "kakadu_available",
            "kakadu_expand_command",
            "kakadu_version",
        ],
    )
    .ok();
    let encode_metadata = read_tsv_metadata(
        &options.out_dir.join("cpu-encode-compare.out"),
        &[
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "external_unique_input_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "kakadu_included",
            "kakadu_compress_available",
            "kakadu_compress_command",
            "kakadu_version",
        ],
    )
    .ok();
    let mut out = String::new();
    out.push_str("# J2K Adoption Benchmark Run\n\n");
    out.push_str("This directory is a benchmark artifact bundle. Treat `summary.json` as the machine-readable index.\n\n");
    out.push_str("## Inputs\n\n");
    out.push_str(&format!(
        "- Fixture dirs: `{}`\n",
        options.input_dirs.as_deref().unwrap_or("not set")
    ));
    out.push_str(&format!(
        "- Fixture manifest: `{}`\n",
        options
            .manifest
            .as_ref()
            .map_or_else(|| "not set".to_string(), |path| path.display().to_string())
    ));
    out.push_str(&format!(
        "- Encode source dirs: `{}`\n",
        options.encode_input_dirs.as_deref().unwrap_or("not set")
    ));
    out.push_str(&format!(
        "- Encode manifest: `{}`\n",
        options
            .encode_manifest
            .as_ref()
            .map_or_else(|| "not set".to_string(), |path| path.display().to_string())
    ));
    out.push_str(&format!(
        "- Generated fixtures included: `{}`\n",
        options.include_generated
    ));
    out.push_str(&format!(
        "- OpenJPH comparator requested: `{}`\n",
        options.openjph
    ));
    out.push_str(&format!(
        "- Kakadu comparator requested: `{}`\n",
        options.kakadu
    ));
    out.push_str(&format!("- Quick mode: `{}`\n\n", options.quick));
    out.push_str("## Steps\n\n");
    out.push_str("| Step | Status | Output | Error log |\n");
    out.push_str("| --- | --- | --- | --- |\n");
    for step in steps {
        let status = match &step.status {
            StepStatus::Ran => "ran".to_string(),
            StepStatus::Skipped { reason } => format!("skipped: {reason}"),
        };
        out.push_str(&format!(
            "| `{}` | {} | `{}` | `{}` |\n",
            step.name,
            status,
            step.stdout.display(),
            step.stderr.display()
        ));
    }
    out.push_str("\n## Publication Gate\n\n");
    out.push_str("Do not publish this bundle unless `cpu-fixture-compare.out` and `cpu-encode-compare.out` both contain `publication_eligible\ttrue`, `publication_blockers\tnone`, `benchmark_complete\ttrue`, and mixed external batch rows. CPU decode publication requires independent native compressed classic J2K and HTJ2K coverage; repo-materialized natural-image codestreams are diagnostic workload rows, not enough by themselves. CPU encode rows compare the same staged PNM bytes for classic lossless JP2 only. Optional OpenJPH rows are CLI/file-output HTJ2K/JPH-compatible context rows and must be labeled separately from the default in-process decoder matrix. Optional Kakadu rows are proprietary CLI/file-output context rows and must be labeled separately from the default matrix. CUDA decode hardware rows must be run with `--require-cuda` and the same pinned fixture manifest for supported-HTJ2K-subset claims. CUDA and Metal encode hardware rows must be run with `--require-cuda` or `--require-metal` and manifest-backed staged PGM/PPM sources before they are described as using the same encode source matrix. For Metal resident encode claims, compare `resident_host_ms` with `cpu_ms` only on rows where `packetization_used=true` and `codestream_assembly_used=true`; `resident_buffer_ms` is GPU-pipeline context, not a host-codec apples-to-apples number.\n");
    if let Some(metadata) = cpu_metadata {
        out.push_str("\nCurrent CPU fixture status:\n\n");
        for key in [
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "external_unique_input_count",
            "external_component_group_count",
            "external_dimension_count",
            "external_source_format_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "publication_gate_skipped_comparators",
            "openjph_included",
            "openjph_available",
            "openjph_expand_command",
            "openjph_version",
            "kakadu_included",
            "kakadu_available",
            "kakadu_expand_command",
            "kakadu_version",
        ] {
            if let Some(value) = metadata.get(key).and_then(serde_json::Value::as_str) {
                out.push_str(&format!("- `{key}`: `{value}`\n"));
            }
        }
    }
    if let Some(metadata) = encode_metadata {
        out.push_str("\nCurrent CPU encode status:\n\n");
        for key in [
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
            "case_batch_sizes",
            "mixed_batch_sizes",
            "external_unique_input_count",
            "mixed_external_batch_group_count",
            "mixed_external_min_distinct_inputs",
            "mixed_external_max_distinct_inputs",
            "mixed_external_group_distinct_inputs",
            "kakadu_included",
            "kakadu_compress_available",
            "kakadu_compress_command",
            "kakadu_version",
        ] {
            if let Some(value) = metadata.get(key).and_then(serde_json::Value::as_str) {
                out.push_str(&format!("- `{key}`: `{value}`\n"));
            }
        }
    }
    let mut criterion_rows = 0_usize;
    for step in steps {
        let Some(root) = &step.criterion_root else {
            continue;
        };
        if !matches!(&step.status, StepStatus::Ran) || !root.exists() {
            continue;
        }
        match discover_estimates(root) {
            Ok(estimates) => {
                criterion_rows += estimates.len();
            }
            Err(error) => {
                out.push_str("\nCriterion estimate parsing failed for `");
                out.push_str(step.name);
                out.push_str("`: `");
                out.push_str(&error);
                out.push_str("`.\n");
            }
        }
    }
    if criterion_rows > 0 {
        out.push_str(&format!(
            "\nCriterion estimates are summarized in `summary.json` ({} rows across current-run steps).\n",
            criterion_rows
        ));
    }
    if steps.iter().any(|step| {
        step.name == "metal-encode-auto-routing" && matches!(&step.status, StepStatus::Ran)
    }) {
        out.push_str("\nMetal encode auto-routing rows are summarized in `summary.json` from `metal-encode-auto-routing.out`.\n");
    }

    let path = options.out_dir.join("README.md");
    fs::write(&path, out).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn display_command(
    program: &OsString,
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

fn criterion_target_dir(options: &AdoptionBenchmarkOptions, step_name: &str) -> PathBuf {
    absolute_path(&options.out_dir)
        .join("cargo-target")
        .join(step_name)
}

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn benchmark_env_path(path: &Path, label: &str) -> Result<String, String> {
    let path = canonical_benchmark_path(path, label)?;
    let display = path.display().to_string();
    path.into_os_string()
        .into_string()
        .map_err(|_| format!("{label} path contains non-Unicode data: {display}"))
}

fn benchmark_env_path_list(path_list: &str, label: &str) -> Result<String, String> {
    let paths = env::split_paths(path_list)
        .map(|path| canonical_benchmark_path(&path, label))
        .collect::<Result<Vec<_>, _>>()?;
    if paths.is_empty() {
        return Err(format!("{label} must include at least one path"));
    }
    let joined = env::join_paths(paths)
        .map_err(|error| format!("{label} path-list cannot be represented: {error}"))?;
    let display = joined.to_string_lossy().into_owned();
    joined
        .into_string()
        .map_err(|_| format!("{label} path-list contains non-Unicode data: {display}"))
}

fn canonical_benchmark_path(path: &Path, label: &str) -> Result<PathBuf, String> {
    absolute_path(path).canonicalize().map_err(|error| {
        format!(
            "{label} path {} cannot be canonicalized: {error}",
            path.display()
        )
    })
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn cargo() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

impl AdoptionBenchmarkOptions {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut options = Self {
            out_dir: PathBuf::from("target/j2k-adoption-benchmark"),
            input_dirs: None,
            manifest: None,
            encode_input_dirs: None,
            encode_manifest: None,
            cuda_decode_batch_sizes: None,
            include_generated: false,
            quick: false,
            cuda: false,
            metal: false,
            openjph: false,
            kakadu: false,
            require_cuda: false,
            require_metal: false,
            require_openjph: false,
            require_kakadu: false,
            finalize_existing: false,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => {
                    options.out_dir = PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--out-dir requires a value".to_string())?,
                    );
                }
                "--fixtures" | "--input-dirs" => {
                    options.input_dirs = Some(
                        args.next()
                            .ok_or_else(|| format!("{arg} requires a platform path-list value"))?,
                    );
                }
                "--manifest" => {
                    options.manifest = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--manifest requires a value".to_string())?,
                    ));
                }
                "--encode-fixtures" | "--encode-input-dirs" => {
                    options.encode_input_dirs = Some(
                        args.next()
                            .ok_or_else(|| format!("{arg} requires a platform path-list value"))?,
                    );
                }
                "--encode-manifest" => {
                    options.encode_manifest =
                        Some(PathBuf::from(args.next().ok_or_else(|| {
                            "--encode-manifest requires a value".to_string()
                        })?));
                }
                "--cuda-decode-batch-sizes" => {
                    options.cuda_decode_batch_sizes = Some(parse_batch_size_list(
                        &args.next().ok_or_else(|| {
                            "--cuda-decode-batch-sizes requires a comma-separated value".to_string()
                        })?,
                        "--cuda-decode-batch-sizes",
                    )?);
                }
                "--include-generated" => options.include_generated = true,
                "--quick" => options.quick = true,
                "--cuda" => options.cuda = true,
                "--metal" => options.metal = true,
                "--openjph" => options.openjph = true,
                "--kakadu" => options.kakadu = true,
                "--require-cuda" => {
                    options.cuda = true;
                    options.require_cuda = true;
                }
                "--require-metal" => {
                    options.metal = true;
                    options.require_metal = true;
                }
                "--require-openjph" => {
                    options.openjph = true;
                    options.require_openjph = true;
                }
                "--require-kakadu" => {
                    options.kakadu = true;
                    options.require_kakadu = true;
                }
                "--finalize-existing" => {
                    options.finalize_existing = true;
                }
                "--help" | "-h" => unreachable!("help handled before option parsing"),
                other => {
                    return Err(format!(
                        "unknown adoption-benchmark argument `{other}`\n{}",
                        help_text()
                    ))
                }
            }
        }
        if options.manifest.is_some() && options.input_dirs.is_none() {
            return Err("--manifest requires --fixtures/--input-dirs".to_string());
        }
        if options.encode_manifest.is_some() && options.encode_input_dirs.is_none() {
            return Err(
                "--encode-manifest requires --encode-fixtures/--encode-input-dirs".to_string(),
            );
        }
        if !options.include_generated
            && (options.input_dirs.is_none() || options.encode_input_dirs.is_none())
        {
            return Err(
                "external-only benchmark requires --fixtures and --encode-fixtures, or pass --include-generated for smoke runs"
                    .to_string(),
            );
        }
        Ok(options)
    }
}

fn help_text() -> String {
    "usage: cargo xtask adoption-benchmark [--fixtures PATHS --manifest FILE] [--encode-fixtures PATHS --encode-manifest FILE] [--include-generated] [--quick] [--cuda|--require-cuda] [--cuda-decode-batch-sizes LIST] [--metal|--require-metal] [--openjph|--require-openjph] [--kakadu|--require-kakadu] [--finalize-existing] [--out-dir DIR]".to_string()
}

fn parse_batch_size_list(value: &str, label: &str) -> Result<String, String> {
    let mut sizes = Vec::new();
    for raw in value.split(',') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let size = raw
            .parse::<usize>()
            .map_err(|error| format!("{label} has invalid batch size {raw:?}: {error}"))?;
        if size == 0 {
            return Err(format!("{label} entries must be greater than zero"));
        }
        if !sizes.contains(&size) {
            sizes.push(size);
        }
    }
    if sizes.is_empty() {
        return Err(format!("{label} must include at least one batch size"));
    }
    Ok(sizes
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(","))
}

#[cfg(test)]
mod tests {
    use super::{
        display_command, enforce_publication_gate, parse_metal_auto_bench_line,
        parse_metal_auto_probe_line, parse_metal_resident_bench_line, parse_metal_stage_bench_line,
        read_metal_encode_summary, AdoptionBenchmarkOptions, AdoptionStep, StepStatus,
    };
    use std::ffi::OsString;

    #[test]
    fn generated_smoke_requires_explicit_flag() {
        let error = AdoptionBenchmarkOptions::parse(std::iter::empty())
            .expect_err("default external-only run must require fixtures");

        assert!(error.contains("external-only benchmark requires --fixtures and --encode-fixtures"));
    }

    #[test]
    fn manifest_requires_fixture_dirs() {
        let error = AdoptionBenchmarkOptions::parse(
            ["--manifest", "fixtures.tsv", "--include-generated"]
                .map(str::to_string)
                .into_iter(),
        )
        .expect_err("manifest without fixture dirs must fail");

        assert!(error.contains("--manifest requires --fixtures"));
    }

    #[test]
    fn encode_manifest_requires_encode_fixture_dirs() {
        let error = AdoptionBenchmarkOptions::parse(
            ["--encode-manifest", "encode.tsv", "--include-generated"]
                .map(str::to_string)
                .into_iter(),
        )
        .expect_err("encode manifest without encode dirs must fail");

        assert!(error.contains("--encode-manifest requires --encode-fixtures"));
    }

    #[test]
    fn external_only_requires_decode_and_encode_fixture_dirs() {
        let error = AdoptionBenchmarkOptions::parse(
            ["--fixtures", "decode-fixtures"]
                .map(str::to_string)
                .into_iter(),
        )
        .expect_err("decode-only external run must fail");

        assert!(error.contains("--fixtures and --encode-fixtures"));
    }

    #[test]
    fn parses_external_decode_and_encode_fixture_dirs() {
        let options = AdoptionBenchmarkOptions::parse(
            [
                "--fixtures",
                "decode-fixtures",
                "--encode-fixtures",
                "source-images",
                "--manifest",
                "decode.tsv",
                "--encode-manifest",
                "encode.tsv",
            ]
            .map(str::to_string)
            .into_iter(),
        )
        .expect("valid external options");

        assert_eq!(options.input_dirs.as_deref(), Some("decode-fixtures"));
        assert_eq!(options.encode_input_dirs.as_deref(), Some("source-images"));
        assert_eq!(
            options.manifest.as_deref(),
            Some(std::path::Path::new("decode.tsv"))
        );
        assert_eq!(
            options.encode_manifest.as_deref(),
            Some(std::path::Path::new("encode.tsv"))
        );
    }

    #[test]
    fn full_external_run_fails_when_comparator_publication_gate_fails() {
        let out_dir = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("j2k-adoption-gate-test")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&out_dir).expect("create out dir");
        for name in ["cpu-fixture-compare.out", "cpu-encode-compare.out"] {
            std::fs::write(
                out_dir.join(name),
                "publication_eligible\tfalse\npublication_blockers\tgenerated-fixtures-included\nbenchmark_complete\ttrue\n",
            )
            .expect("write metadata");
        }
        let options = AdoptionBenchmarkOptions {
            out_dir,
            input_dirs: Some("decode-fixtures".to_string()),
            manifest: Some("fixtures.tsv".into()),
            encode_input_dirs: Some("source-images".to_string()),
            encode_manifest: Some("encode.tsv".into()),
            cuda_decode_batch_sizes: None,
            include_generated: false,
            quick: false,
            cuda: false,
            metal: false,
            openjph: false,
            kakadu: false,
            require_cuda: false,
            require_metal: false,
            require_openjph: false,
            require_kakadu: false,
            finalize_existing: false,
        };

        let error = enforce_publication_gate(&options).expect_err("gate must fail");

        assert!(error.contains("adoption benchmark is not publishable"));
        assert!(error.contains("cpu-fixture-compare publication_eligible=false"));
        assert!(error.contains("cpu-encode-compare publication_blockers"));
    }

    #[test]
    fn displayed_commands_show_benchmark_env_scrub() {
        let command = display_command(
            &OsString::from("cargo"),
            &["run".to_string()],
            &[("J2K_FIXTURE_COMPARE_REPEATS".to_string(), "1".to_string())],
            None,
        );

        assert!(command.starts_with("env -u J2K_FIXTURE_COMPARE_MODE"));
        assert!(command.contains("-u J2K_INCLUDE_OPENJPH"));
        assert!(command.contains("-u J2K_REQUIRE_OPENJPH"));
        assert!(command.contains("-u J2K_OPENJPH_EXPAND_BIN"));
        assert!(command.contains("-u J2K_INCLUDE_KAKADU"));
        assert!(command.contains("-u J2K_REQUIRE_KAKADU"));
        assert!(command.contains("-u J2K_KDU_EXPAND_BIN"));
        assert!(command.contains("-u J2K_KDU_COMPRESS_BIN"));
        assert!(command.contains("-u J2K_ENCODE_COMPARE_ENCODERS"));
        assert!(command.contains("-u J2K_CUDA_DECODE_INPUT_DIRS"));
        assert!(command.contains("-u J2K_CUDA_ENCODE_INPUT_DIRS"));
        assert!(command.contains("-u J2K_CUDA_ENCODE_MANIFEST"));
        assert!(command.contains("-u J2K_METAL_ENCODE_INPUT_DIRS"));
        assert!(command.contains("-u J2K_METAL_ENCODE_MANIFEST"));
        assert!(command.contains("J2K_FIXTURE_COMPARE_REPEATS=1"));
        assert!(command.ends_with("cargo run"));
    }

    #[test]
    fn require_cuda_enables_cuda_benches() {
        let options = AdoptionBenchmarkOptions::parse(
            ["--include-generated", "--require-cuda"]
                .map(str::to_string)
                .into_iter(),
        )
        .expect("valid generated CUDA smoke options");

        assert!(options.cuda);
        assert!(options.require_cuda);
        assert!(!options.metal);
    }

    #[test]
    fn parses_cuda_decode_huge_batch_sizes() {
        let options = AdoptionBenchmarkOptions::parse(
            [
                "--include-generated",
                "--cuda",
                "--cuda-decode-batch-sizes",
                "1,16,256,1024,256",
            ]
            .map(str::to_string)
            .into_iter(),
        )
        .expect("valid generated CUDA huge-batch smoke options");

        assert!(options.cuda);
        assert_eq!(
            options.cuda_decode_batch_sizes.as_deref(),
            Some("1,16,256,1024")
        );
    }

    #[test]
    fn rejects_invalid_cuda_decode_batch_sizes() {
        let error = AdoptionBenchmarkOptions::parse(
            [
                "--include-generated",
                "--cuda",
                "--cuda-decode-batch-sizes",
                "8,0",
            ]
            .map(str::to_string)
            .into_iter(),
        )
        .expect_err("zero CUDA batch size must fail");

        assert!(error.contains("--cuda-decode-batch-sizes entries must be greater than zero"));
    }

    #[test]
    fn require_openjph_enables_openjph_comparator() {
        let options = AdoptionBenchmarkOptions::parse(
            ["--include-generated", "--require-openjph"]
                .map(str::to_string)
                .into_iter(),
        )
        .expect("valid generated OpenJPH smoke options");

        assert!(options.openjph);
        assert!(options.require_openjph);
        assert!(!options.cuda);
        assert!(!options.metal);
    }

    #[test]
    fn require_kakadu_enables_kakadu_comparator() {
        let options = AdoptionBenchmarkOptions::parse(
            ["--include-generated", "--require-kakadu"]
                .map(str::to_string)
                .into_iter(),
        )
        .expect("valid generated Kakadu smoke options");

        assert!(options.kakadu);
        assert!(options.require_kakadu);
        assert!(!options.cuda);
        assert!(!options.metal);
    }

    #[test]
    fn parses_metal_auto_bench_row() {
        let row = parse_metal_auto_bench_line(
            "j2k_metal_encode_auto_bench mode=lossless codec=htj2k components=rgb8 size=1024x1024 cpu_ms=12.345 auto_ms=6.789",
        )
        .expect("valid auto bench row");

        assert_eq!(row["mode"], "lossless");
        assert_eq!(row["codec"], "htj2k");
        assert_eq!(row["components"], "rgb8");
        assert_eq!(row["size"], "1024x1024");
        assert_eq!(row["cpu_ms"], 12.345);
        assert_eq!(row["auto_ms"], 6.789);
    }

    #[test]
    fn parses_metal_stage_skip_with_error() {
        let row = parse_metal_stage_bench_line(
            "j2k_metal_encode_stage_bench stage=forward_dwt97 size=512x512 cpu_ms=1.250 metal_ms=skipped error=Metal device unavailable",
        )
        .expect("valid stage bench row");

        assert_eq!(row["stage"], "forward_dwt97");
        assert_eq!(row["metal_ms"], "skipped");
        assert_eq!(row["error"], "Metal device unavailable");
    }

    #[test]
    fn parses_metal_resident_bench_row() {
        let row = parse_metal_resident_bench_line(
            "j2k_metal_encode_resident_bench mode=lossless_external codec=htj2k components=rgb8 size=1024x768 batch_size=256 fixture_count=24 resident_input_storage=private resident_staging=already_padded_contiguous cpu_ms=120.000 hybrid_cpu_packet_ms=81.250 resident_host_ms=44.500 resident_buffer_ms=39.250 packetization_used=true codestream_assembly_used=true host_readback_ms=5.125 gpu_ms=33.750 encoded_host_bytes=123456 encoded_buffer_bytes=123456",
        )
        .expect("valid resident bench row");

        assert_eq!(row["mode"], "lossless_external");
        assert_eq!(row["codec"], "htj2k");
        assert_eq!(row["batch_size"], 256);
        assert_eq!(row["fixture_count"], 24);
        assert_eq!(row["resident_host_ms"], 44.5);
        assert_eq!(row["resident_buffer_ms"], 39.25);
        assert_eq!(row["packetization_used"], true);
        assert_eq!(row["codestream_assembly_used"], true);
        assert_eq!(row["encoded_host_bytes"], 123456);
        assert_eq!(row["resident_input_storage"], "private");
        assert_eq!(row["resident_staging"], "already_padded_contiguous");
    }

    #[test]
    fn metal_resident_summary_counts_only_full_resident_rows_as_verified() {
        let out_dir = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("j2k-metal-resident-summary-test")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&out_dir).expect("create out dir");
        let stdout = out_dir.join("metal-encode-auto-routing.out");
        std::fs::write(
            &stdout,
            concat!(
                "j2k_metal_encode_resident_bench mode=lossless_external codec=htj2k components=rgb8 size=64x64 batch_size=16 fixture_count=1 cpu_ms=1.000 hybrid_cpu_packet_ms=skipped resident_host_ms=0.500 resident_buffer_ms=0.400 packetization_used=true codestream_assembly_used=true host_readback_ms=0.050 gpu_ms=not-recorded encoded_host_bytes=128 encoded_buffer_bytes=128\n",
                "j2k_metal_encode_resident_bench mode=lossless_external codec=htj2k components=rgb8 size=64x64 batch_size=256 fixture_count=1 cpu_ms=10.000 hybrid_cpu_packet_ms=skipped resident_host_ms=4.500 resident_buffer_ms=3.900 packetization_used=true codestream_assembly_used=false host_readback_ms=0.300 gpu_ms=not-recorded encoded_host_bytes=2048 encoded_buffer_bytes=2048\n",
                "j2k_metal_encode_resident_bench mode=lossless_external codec=htj2k components=rgb8 size=64x64 batch_size=1024 fixture_count=1 cpu_ms=skipped hybrid_cpu_packet_ms=skipped resident_host_ms=skipped resident_buffer_ms=skipped packetization_used=false codestream_assembly_used=false host_readback_ms=skipped gpu_ms=skipped encoded_host_bytes=skipped encoded_buffer_bytes=skipped error=memory budget prevented resident batch\n",
                "j2k_metal_encode_resident_batch_sizes\t1,16,256,1024\n",
            ),
        )
        .expect("write Metal stdout");
        let step = AdoptionStep {
            name: "metal-encode-auto-routing",
            command: "cargo test".to_string(),
            stdout: stdout.clone(),
            stderr: out_dir.join("metal-encode-auto-routing.err"),
            criterion_root: None,
            status: StepStatus::Ran,
        };

        let summary = read_metal_encode_summary(&stdout, &[step]);

        assert_eq!(summary["resident_bench_count"], 3);
        assert_eq!(summary["skipped_resident_bench_count"], 1);
        assert_eq!(summary["resident_verified_bench_count"], 1);
        assert_eq!(
            summary["metadata"]["j2k_metal_encode_resident_batch_sizes"],
            "1,16,256,1024"
        );
    }

    #[test]
    fn parses_metal_probe_dispatch_suffix() {
        let row = parse_metal_auto_probe_line(
            "j2k_metal_encode_auto_probe mode=lossy codec=htj2k components=gray8 size=512x512 dispatch=J2kEncodeDispatchReport { forward_dwt97: Some(1) }",
        )
        .expect("valid probe row");

        assert_eq!(row["mode"], "lossy");
        assert_eq!(
            row["dispatch"],
            "J2kEncodeDispatchReport { forward_dwt97: Some(1) }"
        );
    }
}
