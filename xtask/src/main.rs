use std::collections::BTreeSet;
use std::env;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

#[cfg(feature = "adoption")]
mod adoption_benchmark;
#[cfg(feature = "adoption")]
mod adoption_corpus;
#[cfg(feature = "adoption")]
mod adoption_curate;
#[cfg(not(feature = "adoption"))]
mod adoption_disabled;
#[cfg(feature = "adoption")]
mod adoption_manifest;
#[cfg(feature = "adoption")]
mod adoption_materialize;
#[cfg(feature = "adoption")]
mod adoption_report;
mod coverage;
#[cfg(feature = "adoption")]
mod markdown;
mod metal;
mod perf_guard;
mod process;
mod public_support;
#[cfg(feature = "adoption")]
mod publication_gate;
mod release_status;
mod semver;

use process::cargo;

const PUBLISHABLE_PACKAGES: &[&str] = &[
    "j2k-core",
    "j2k-profile",
    "j2k-types",
    "j2k-codec-math",
    "j2k-cuda-runtime",
    "j2k-metal-support",
    "j2k-native",
    "j2k-jpeg",
    "j2k-tilecodec",
    "j2k",
    "j2k-transcode",
    "j2k-transcode-cuda",
    "j2k-jpeg-metal",
    "j2k-metal",
    "j2k-transcode-metal",
    "j2k-jpeg-cuda",
    "j2k-cuda",
    "j2k-cli",
];

const REGISTRY_INDEPENDENT_PACKAGES: &[&str] =
    &["j2k-core", "j2k-profile", "j2k-types", "j2k-codec-math"];

const STAGED_DEPENDENCY_PACKAGES: &[&str] = &[
    "j2k-cuda-runtime",
    "j2k-metal-support",
    "j2k-native",
    "j2k-jpeg",
    "j2k-tilecodec",
    "j2k",
    "j2k-transcode",
    "j2k-transcode-cuda",
    "j2k-jpeg-metal",
    "j2k-metal",
    "j2k-transcode-metal",
    "j2k-jpeg-cuda",
    "j2k-cuda",
    "j2k-cli",
];

const CPU_RELEASE_PACKAGES: &[&str] = &[
    "j2k-core",
    "j2k-codec-math",
    "j2k-jpeg",
    "j2k-types",
    "j2k-native",
    "j2k",
    "j2k-tilecodec",
    "j2k-cli",
];

const STABLE_SEMVER_PACKAGES: &[&str] = &[
    "j2k",
    "j2k-core",
    "j2k-codec-math",
    "j2k-jpeg",
    "j2k-tilecodec",
    "j2k-jpeg-metal",
    "j2k-metal",
    "j2k-jpeg-cuda",
    "j2k-cuda",
    "j2k-transcode",
    "j2k-transcode-cuda",
    "j2k-metal-support",
    "j2k-transcode-metal",
    "j2k-native",
    "j2k-types",
    "j2k-cuda-runtime",
    "j2k-profile",
];

const STABLE_DOC_LIBRARY_PACKAGES: &[&str] = &[
    "j2k",
    "j2k-core",
    "j2k-codec-math",
    "j2k-jpeg",
    "j2k-tilecodec",
    "j2k-jpeg-metal",
    "j2k-metal",
    "j2k-jpeg-cuda",
    "j2k-cuda",
    "j2k-transcode",
    "j2k-transcode-cuda",
    "j2k-metal-support",
    "j2k-transcode-metal",
    "j2k-native",
    "j2k-types",
    "j2k-cuda-runtime",
    "j2k-profile",
];

const STABLE_API_SNAPSHOT: &str = "docs/stable-api-1.0.public-api.txt";
const CARGO_PUBLIC_API_VERSION: &str = "0.52.0";
const PANIC_SURFACE_UNWRAP_USED_BASELINE: usize = 17;
const PANIC_SURFACE_EXPECT_USED_BASELINE: usize = 106;
const CODEC_MATH_DWT97_METAL_FRAGMENT: &str =
    "crates/j2k-codec-math/generated/dwt97_constants.metal";
const CODEC_MATH_DWT97_RUST_FRAGMENT: &str = "crates/j2k-codec-math/generated/dwt97_constants.rs";

const NO_STD_TARGET: &str = "aarch64-unknown-none";
const NO_STD_CORE_PORTABLE_TARGET: &str = "wasm32-unknown-unknown";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("xtask failed: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let task = env::args().nth(1).unwrap_or_else(|| "help".to_string());
    match task.as_str() {
        "fmt" => fmt(),
        "clippy" => clippy(),
        "clippy-strict" => clippy_strict(),
        "test" => test(),
        "nextest" => nextest(),
        "doc" | "docs" => doc(),
        "typos" => typos(),
        "bench-build" => bench_build(),
        "bench-report" => bench_report(env::args().skip(2)),
        #[cfg(feature = "adoption")]
        "adoption-benchmark" => adoption_benchmark::adoption_benchmark(env::args().skip(2)),
        #[cfg(not(feature = "adoption"))]
        "adoption-benchmark" => adoption_disabled::adoption_benchmark(env::args().skip(2)),
        #[cfg(feature = "adoption")]
        "adoption-curate" => adoption_curate::adoption_curate(env::args().skip(2)),
        #[cfg(not(feature = "adoption"))]
        "adoption-curate" => adoption_disabled::adoption_curate(env::args().skip(2)),
        #[cfg(feature = "adoption")]
        "adoption-manifest" => adoption_manifest::adoption_manifest(env::args().skip(2)),
        #[cfg(not(feature = "adoption"))]
        "adoption-manifest" => adoption_disabled::adoption_manifest(env::args().skip(2)),
        #[cfg(feature = "adoption")]
        "adoption-materialize" => adoption_materialize::adoption_materialize(env::args().skip(2)),
        #[cfg(not(feature = "adoption"))]
        "adoption-materialize" => adoption_disabled::adoption_materialize(env::args().skip(2)),
        #[cfg(feature = "adoption")]
        "adoption-report" => adoption_report::adoption_report(env::args().skip(2)),
        #[cfg(not(feature = "adoption"))]
        "adoption-report" => adoption_disabled::adoption_report(env::args().skip(2)),
        "public-support" => public_support::public_support(env::args().skip(2)),
        "j2k-bench-signoff" => j2k_bench_signoff(),
        "j2k-perf-guard" => perf_guard::j2k_perf_guard(env::args().skip(2)),
        "codec-math-codegen" => codec_math_codegen(env::args().skip(2)),
        "fuzz-build" => fuzz_build(),
        "fuzz-run" => fuzz_run(),
        "stable-api" => stable_api(env::args().skip(2)),
        "semver" => semver::semver(
            env::args().skip(2),
            STABLE_SEMVER_PACKAGES,
            CARGO_PUBLIC_API_VERSION,
        ),
        "deny" => deny(),
        "miri" => miri(),
        "machete" => machete(),
        "panic-surface" => panic_surface(),
        "no-std" => no_std(),
        "unsafe-audit" => verify_unsafe_audit(),
        "downstream-smoke" => downstream_smoke(),
        "repo-lint" => repo_lint(env::args().skip(2)),
        "release-integrity" => release_integrity(),
        "release-status" => release_status::release_status(env::args().skip(2)),
        "release-cpu" => release_cpu(),
        "metal-compile" => metal::metal_compile(),
        "release-metal" => metal::release_metal(),
        "coverage" => coverage::coverage(env::args().skip(2)),
        "package" => package(),
        "ci" => ci(),
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown task `{other}`")),
    }
}

fn ci() -> Result<(), String> {
    fmt()?;
    codec_math_codegen(std::iter::empty::<String>())?;
    clippy()?;
    panic_surface()?;
    test()?;
    doc()?;
    verify_unsafe_audit()
}

fn repo_lint(args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut strict = false;
    for arg in args {
        match arg.as_str() {
            "--strict" => strict = true,
            other => return Err(format!("unknown repo-lint argument `{other}`")),
        }
    }

    run_cargo(&[
        "test",
        "-p",
        "xtask",
        "--test",
        "repo_lint",
        "--",
        "--nocapture",
    ])?;

    if strict {
        run_cargo(&[
            "test",
            "-p",
            "xtask",
            "--test",
            "repo_lint",
            "--",
            "--nocapture",
            "--ignored",
        ])?;
    }

    Ok(())
}

fn fmt() -> Result<(), String> {
    run_cargo(&["fmt", "--all", "--", "--check"])
}

fn clippy() -> Result<(), String> {
    run_cargo(&[
        "clippy",
        "--workspace",
        "--all-targets",
        "--all-features",
        "--",
        "-D",
        "warnings",
    ])
}

fn clippy_strict() -> Result<(), String> {
    let mut args = vec![
        "clippy",
        "-p",
        "j2k-native",
        "-p",
        "j2k",
        "--all-targets",
        "--all-features",
        "--no-deps",
        "--",
        "-W",
        "clippy::pedantic",
        "-W",
        "clippy::nursery",
        "-D",
        "warnings",
    ];

    // Keep the strict gate useful as a ratchet: enable pedantic/nursery, but
    // baseline high-noise codec-math lints so new lint classes still fail.
    for lint in STRICT_CLIPPY_BASELINE_ALLOWED_LINTS {
        args.extend(["-A", lint]);
    }

    run_cargo(&args)
}

const STRICT_CLIPPY_BASELINE_ALLOWED_LINTS: &[&str] = &[
    "clippy::bool_to_int_with_if",
    "clippy::branches_sharing_code",
    "clippy::cast_lossless",
    "clippy::cast_possible_truncation",
    "clippy::cast_possible_wrap",
    "clippy::cast_precision_loss",
    "clippy::cast_sign_loss",
    "clippy::checked_conversions",
    "clippy::cognitive_complexity",
    "clippy::doc_markdown",
    "clippy::elidable_lifetime_names",
    "clippy::explicit_deref_methods",
    "clippy::explicit_iter_loop",
    "clippy::float_cmp",
    "clippy::if_not_else",
    "clippy::inconsistent_struct_constructor",
    "clippy::inline_always",
    "clippy::items_after_statements",
    "clippy::manual_let_else",
    "clippy::map_unwrap_or",
    "clippy::match_same_arms",
    "clippy::missing_const_for_fn",
    "clippy::missing_errors_doc",
    "clippy::must_use_candidate",
    "clippy::needless_collect",
    "clippy::needless_pass_by_ref_mut",
    "clippy::needless_pass_by_value",
    "clippy::no_effect_underscore_binding",
    "clippy::or_fun_call",
    "clippy::redundant_clone",
    "clippy::redundant_closure_for_method_calls",
    "clippy::redundant_else",
    "clippy::redundant_pub_crate",
    "clippy::similar_names",
    "clippy::struct_excessive_bools",
    "clippy::struct_field_names",
    "clippy::suboptimal_flops",
    "clippy::suspicious_operation_groupings",
    "clippy::too_many_lines",
    "clippy::trivially_copy_pass_by_ref",
    "clippy::unnecessary_wraps",
    "clippy::unreadable_literal",
    "clippy::used_underscore_binding",
    "clippy::useless_let_if_seq",
];

fn test() -> Result<(), String> {
    if env::consts::OS != "macos" {
        return test_workspace_without_benches(&[]);
    }

    test_workspace_without_benches(&["--exclude", "j2k-metal"])?;
    test_j2k_metal_without_benches()
}

fn test_workspace_without_benches(extra_args: &[&str]) -> Result<(), String> {
    let mut test_args = vec![
        "test",
        "--workspace",
        "--all-features",
        "--lib",
        "--bins",
        "--tests",
    ];
    test_args.extend_from_slice(extra_args);
    run_cargo(&test_args)?;
    test_facade_cuda_stub()?;

    let mut doc_args = vec!["test", "--workspace", "--all-features", "--doc"];
    doc_args.extend_from_slice(extra_args);
    run_cargo(&doc_args)
}

fn test_facade_cuda_stub() -> Result<(), String> {
    run_cargo(&[
        "test",
        "-p",
        "j2k",
        "--test",
        "encode_lossless",
        "accelerator_facade_auto_falls_back_when_no_stage_dispatches",
    ])
}

fn test_j2k_metal_without_benches() -> Result<(), String> {
    run_cargo_with_env(
        &[
            "test",
            "-p",
            "j2k-metal",
            "--all-features",
            "--lib",
            "--bins",
            "--tests",
        ],
        &[("RUST_TEST_THREADS", "1")],
    )?;
    run_cargo(&["test", "-p", "j2k-metal", "--all-features", "--doc"])
}

fn nextest() -> Result<(), String> {
    run_cargo(&[
        "nextest",
        "run",
        "--workspace",
        "--all-features",
        "--lib",
        "--bins",
        "--tests",
    ])
}

fn doc() -> Result<(), String> {
    run_cargo_with_env(
        &["doc", "--workspace", "--all-features", "--no-deps"],
        &[("RUSTDOCFLAGS", "-D warnings")],
    )?;

    for package in STABLE_DOC_LIBRARY_PACKAGES {
        run_cargo_with_env(
            &["doc", "-p", package, "--lib", "--no-deps"],
            &[("RUSTDOCFLAGS", "-D warnings -D missing_docs")],
        )?;
    }

    run_cargo_with_env(
        &["doc", "-p", "j2k-cli", "--no-deps"],
        &[("RUSTDOCFLAGS", "-D warnings -D missing_docs")],
    )
}

fn typos() -> Result<(), String> {
    run_program(OsString::from("typos"), &[], &[])
}

fn bench_build() -> Result<(), String> {
    run_cargo(&["bench", "-p", "j2k", "--bench", "public_api", "--no-run"])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-native",
        "--bench",
        "tier1_bitplane",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-native",
        "--bench",
        "htj2k_sigprop_phase",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-native",
        "--bench",
        "direct_cpu",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-jpeg",
        "--bench",
        "encode_cpu",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-jpeg",
        "--features",
        "bench-libjpeg-turbo",
        "--no-run",
    ])?;
    run_cargo(&["bench", "-p", "j2k-jpeg-metal", "--no-run"])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-jpeg-cuda",
        "--bench",
        "device_decode",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-cuda",
        "--bench",
        "encode_stages",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-cuda",
        "--bench",
        "htj2k_decode",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-cuda",
        "--bench",
        "htj2k_encode",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-tilecodec",
        "--bench",
        "compare",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-transcode",
        "--bench",
        "dct53",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "j2k-transcode-metal",
        "--bench",
        "dct97",
        "--no-run",
    ])
}

fn j2k_bench_signoff() -> Result<(), String> {
    run_cargo_test_with_pass_floor(
        &["test", "-p", "j2k-compare", "--test", "in_process_parity"],
        &[("J2K_REQUIRE_OPENJPEG", "1"), ("J2K_REQUIRE_GROK", "1")],
        8,
        "in-process OpenJPEG/Grok parity",
    )?;
    run_cargo_test_with_pass_floor(
        &["test", "-p", "j2k", "--test", "openjpeg_parity"],
        &[("J2K_REQUIRE_OPENJPEG", "1")],
        7,
        "OpenJPEG CLI parity",
    )?;
    run_cargo_test_with_pass_floor(
        &["test", "-p", "j2k", "--test", "grok_parity"],
        &[("J2K_REQUIRE_GROK", "1")],
        12,
        "Grok CLI parity",
    )?;
    run_cargo_test_with_pass_floor(
        &[
            "test",
            "-p",
            "j2k-jpeg",
            "--features",
            "bench-libjpeg-turbo",
            "--test",
            "libjpeg_turbo_compare",
        ],
        &[("J2K_REQUIRE_LIBJPEG_TURBO", "1")],
        1,
        "libjpeg-turbo JPEG parity",
    )
}

#[derive(Debug)]
struct BenchmarkReport {
    command: String,
    host: String,
    rustc: String,
    cargo: String,
    git_revision: String,
    workspace_version: String,
    input_source: String,
    compare_threads: String,
    comparator_versions: Vec<(String, String)>,
    skipped_rows: Vec<String>,
}

fn bench_report(args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut command = env::var("J2K_BENCH_COMMAND").unwrap_or_else(|_| "not recorded".into());
    let mut input_source = env::var("J2K_BENCH_INPUT_SOURCE")
        .or_else(|_| env::var("J2K_BENCH_INPUTS"))
        .unwrap_or_else(|_| "not recorded".into());
    let mut out_path = None::<PathBuf>;
    let mut skipped_rows = env::var("J2K_BENCH_SKIPPED_ROWS")
        .ok()
        .map(|rows| split_semicolon_list(&rows))
        .unwrap_or_default();

    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--command" => {
                command = args
                    .next()
                    .ok_or_else(|| "--command requires a value".to_string())?;
            }
            "--input-source" => {
                input_source = args
                    .next()
                    .ok_or_else(|| "--input-source requires a value".to_string())?;
            }
            "--skipped-row" => {
                skipped_rows.push(
                    args.next()
                        .ok_or_else(|| "--skipped-row requires a value".to_string())?,
                );
            }
            "--out" => {
                out_path = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--out requires a value".to_string())?,
                ));
            }
            "--help" | "-h" => {
                print_bench_report_help();
                return Ok(());
            }
            other => return Err(format!("unknown bench-report argument `{other}`")),
        }
    }

    let report = BenchmarkReport {
        command,
        host: host_description(),
        rustc: command_output("rustc", &["-Vv"])
            .unwrap_or_else(|err| format!("unavailable: {err}")),
        cargo: command_output_os(cargo(), &["-V"])
            .unwrap_or_else(|err| format!("unavailable: {err}")),
        git_revision: command_output("git", &["rev-parse", "HEAD"])
            .unwrap_or_else(|err| format!("unavailable: {err}")),
        workspace_version: workspace_version()?,
        input_source,
        compare_threads: env::var("J2K_COMPARE_THREADS").unwrap_or_else(|_| "not set".to_string()),
        comparator_versions: comparator_versions(),
        skipped_rows,
    };
    let rendered = render_benchmark_report(&report);

    if let Some(path) = out_path {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .map_err(|err| format!("failed to write {}: {err}", path.display()))
    } else {
        print!("{rendered}");
        Ok(())
    }
}

fn fuzz_build() -> Result<(), String> {
    run_cargo(&["check", "--manifest-path", "crates/j2k/fuzz/Cargo.toml"])?;
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/j2k-jpeg/fuzz/Cargo.toml",
    ])?;
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/j2k-tilecodec/fuzz/Cargo.toml",
    ])?;
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/j2k-transcode/fuzz/Cargo.toml",
    ])
}

const FUZZ_TARGETS: &[(&str, &str)] = &[
    ("crates/j2k", "decode_fuzz"),
    ("crates/j2k", "jp2_box_fuzz"),
    ("crates/j2k", "jp2_metadata_fuzz"),
    ("crates/j2k", "parse_fuzz"),
    ("crates/j2k", "region_scaled_fuzz"),
    ("crates/j2k-jpeg", "decode_fuzz"),
    ("crates/j2k-jpeg", "parse_fuzz"),
    ("crates/j2k-jpeg", "region_scaled_fuzz"),
    ("crates/j2k-jpeg", "row_stream_fuzz"),
    ("crates/j2k-tilecodec", "decompress_fuzz"),
    ("crates/j2k-transcode", "jpeg_to_htj2k_fuzz"),
];

fn fuzz_run() -> Result<(), String> {
    let runs = env::var("J2K_FUZZ_RUNS").unwrap_or_else(|_| "1000".to_string());
    let max_total_time = env::var("J2K_FUZZ_MAX_TOTAL_TIME_SECONDS").ok();
    let fuzz_target = fuzz_target_triple()?;

    for (crate_dir, target) in FUZZ_TARGETS {
        let mut args = vec![
            "fuzz".to_string(),
            "run".to_string(),
            "--target".to_string(),
            fuzz_target.clone(),
            (*target).to_string(),
            "--".to_string(),
            format!("-runs={runs}"),
        ];
        if let Some(seconds) = &max_total_time {
            args.push(format!("-max_total_time={seconds}"));
        }
        run_nightly_cargo_in_dir_owned(crate_dir, &args)?;
    }
    Ok(())
}

fn fuzz_target_triple() -> Result<String, String> {
    if let Ok(target) = env::var("J2K_FUZZ_TARGET") {
        if !target.trim().is_empty() {
            return Ok(target);
        }
    }

    let version = command_output_os(
        OsString::from("rustup"),
        &["run", "nightly", "rustc", "-vV"],
    )
    .map_err(|err| format!("failed to detect nightly host target for fuzz-run: {err}"))?;
    version
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::to_string)
        .ok_or_else(|| "failed to parse nightly host target from `rustc -vV`".to_string())
}

fn stable_api(args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut write_snapshot = false;
    for arg in args {
        match arg.as_str() {
            "--write" => write_snapshot = true,
            "--help" | "-h" => {
                print_stable_api_help();
                return Ok(());
            }
            other => return Err(format!("unknown stable-api argument `{other}`")),
        }
    }

    let rendered = render_stable_api_snapshot()?;
    if write_snapshot {
        fs::write(STABLE_API_SNAPSHOT, rendered)
            .map_err(|err| format!("failed to write {STABLE_API_SNAPSHOT}: {err}"))?;
        return Ok(());
    }

    let committed = fs::read_to_string(STABLE_API_SNAPSHOT)
        .map_err(|err| format!("failed to read {STABLE_API_SNAPSHOT}: {err}"))?;
    if committed == rendered {
        Ok(())
    } else {
        Err(format!(
            "{STABLE_API_SNAPSHOT} is stale; run `cargo xtask stable-api --write` and review the public API diff"
        ))
    }
}

fn codec_math_codegen(args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut write_fragments = false;
    for arg in args {
        match arg.as_str() {
            "--write" => write_fragments = true,
            "--help" | "-h" => {
                print_codec_math_codegen_help();
                return Ok(());
            }
            other => return Err(format!("unknown codec-math-codegen argument `{other}`")),
        }
    }

    let fragments = [
        (
            CODEC_MATH_DWT97_METAL_FRAGMENT,
            render_codec_math_dwt97_metal_fragment(),
        ),
        (
            CODEC_MATH_DWT97_RUST_FRAGMENT,
            render_codec_math_dwt97_rust_fragment(),
        ),
    ];

    if write_fragments {
        for (path, rendered) in fragments {
            let path = Path::new(path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
            }
            fs::write(path, rendered)
                .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
        }
        return Ok(());
    }

    let mut stale = Vec::new();
    for (path, rendered) in fragments {
        let committed =
            fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))?;
        if committed != rendered {
            stale.push(path);
        }
    }
    if stale.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "codec math generated fragments are stale: {}; run `cargo xtask codec-math-codegen --write` and review the diff",
            stale.join(", ")
        ))
    }
}

fn render_codec_math_dwt97_metal_fragment() -> String {
    use j2k_codec_math::dwt;

    [
        "// Generated from crates/j2k-codec-math/src/dwt.rs.".to_string(),
        format!(
            "constant float CODEC_MATH_DWT97_ALPHA = {}f;",
            compact_f32(dwt::DWT97_ALPHA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_BETA = {}f;",
            compact_f32(dwt::DWT97_BETA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_GAMMA = {}f;",
            compact_f32(dwt::DWT97_GAMMA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_DELTA = {}f;",
            compact_f32(dwt::DWT97_DELTA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_KAPPA = {}f;",
            compact_f32(dwt::DWT97_KAPPA_F32)
        ),
        "constant float CODEC_MATH_DWT97_INV_KAPPA = 1.0f / CODEC_MATH_DWT97_KAPPA;".to_string(),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_ALPHA = {}f;",
            compact_f32(dwt::IDWT97_NEG_ALPHA_F32)
        ),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_BETA = {}f;",
            compact_f32(dwt::IDWT97_NEG_BETA_F32)
        ),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_GAMMA = {}f;",
            compact_f32(dwt::IDWT97_NEG_GAMMA_F32)
        ),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_DELTA = {}f;",
            compact_f32(dwt::IDWT97_NEG_DELTA_F32)
        ),
    ]
    .join("\n")
        + "\n"
}

fn render_codec_math_dwt97_rust_fragment() -> String {
    use j2k_codec_math::dwt;

    [
        "// Generated from crates/j2k-codec-math/src/dwt.rs.".to_string(),
        format!(
            "pub const CODEC_MATH_DWT97_ALPHA: f32 = {};",
            compact_f32(dwt::DWT97_ALPHA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_BETA: f32 = {};",
            compact_f32(dwt::DWT97_BETA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_GAMMA: f32 = {};",
            compact_f32(dwt::DWT97_GAMMA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_DELTA: f32 = {};",
            compact_f32(dwt::DWT97_DELTA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_KAPPA: f32 = {};",
            compact_f32(dwt::DWT97_KAPPA_F32)
        ),
        "pub const CODEC_MATH_DWT97_INV_KAPPA: f32 = 1.0 / CODEC_MATH_DWT97_KAPPA;".to_string(),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_ALPHA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_ALPHA_F32)
        ),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_BETA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_BETA_F32)
        ),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_GAMMA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_GAMMA_F32)
        ),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_DELTA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_DELTA_F32)
        ),
    ]
    .join("\n")
        + "\n"
}

fn compact_f32(value: f32) -> String {
    format!("{value:?}")
}

fn render_stable_api_snapshot() -> Result<String, String> {
    if !cfg!(target_os = "macos") {
        return Err(
            "stable-api snapshot must be generated on macOS so target-gated Metal APIs are included"
                .to_string(),
        );
    }

    let tool_version =
        command_output_os_detailed(cargo(), &["public-api", "--version"]).map_err(|err| {
            format!(
                "failed to detect cargo-public-api: {err}; \
                 install cargo-public-api with `cargo install cargo-public-api --version {CARGO_PUBLIC_API_VERSION} --locked`"
            )
        })?;
    if !tool_version.contains(CARGO_PUBLIC_API_VERSION) {
        return Err(format!(
            "cargo-public-api version must be {CARGO_PUBLIC_API_VERSION}; found `{tool_version}`"
        ));
    }

    let mut out = String::new();
    writeln!(
        &mut out,
        "# J2K 1.0 Public API Snapshot\n\n\
         This file is generated by `cargo xtask stable-api --write` from \
         `cargo public-api -p <package> --all-features -sss --color never`.\n\
         It is generated on macOS so target-gated Metal APIs are included; \
         non-macOS builds expose a smaller cfg-gated subset.\n\n\
         Generator: `{tool_version}`.\n\n\
         It is the item-level companion to `docs/stable-api-1.0.md`: every \
         public module, type, trait, function, method, constant, variant, and \
         field reported here is semver-visible unless moved private before 1.0.\n"
    )
    .unwrap();

    for package in STABLE_DOC_LIBRARY_PACKAGES {
        let api = command_output_os_detailed(
            cargo(),
            &[
                "public-api",
                "-p",
                package,
                "--all-features",
                "-sss",
                "--color",
                "never",
            ],
        )
        .map_err(|err| {
            format!(
                "failed to generate public API for {package}: {err}; \
                 install cargo-public-api with `cargo install cargo-public-api --version {CARGO_PUBLIC_API_VERSION} --locked`"
            )
        })?;
        writeln!(&mut out, "## `{package}`\n\n```text").unwrap();
        writeln!(&mut out, "{api}").unwrap();
        writeln!(&mut out, "```\n").unwrap();
    }

    writeln!(
        &mut out,
        "## `j2k-cli`\n\n\
         `j2k-cli` is a binary package. Its stable command, stdout/stderr, \
         and exit-code contract is documented in `docs/stable-api-1.0.md`.\n"
    )
    .unwrap();

    Ok(out)
}

fn deny() -> Result<(), String> {
    run_cargo(&["deny", "check", "licenses", "advisories", "bans", "sources"])
}

fn miri() -> Result<(), String> {
    run_nightly_cargo(&["miri", "test", "-p", "j2k-core"])?;
    run_nightly_cargo(&["miri", "test", "-p", "j2k-tilecodec"])?;
    run_nightly_cargo(&[
        "miri",
        "test",
        "-p",
        "j2k-native",
        "--no-default-features",
        "inspect::",
    ])
}

fn machete() -> Result<(), String> {
    run_program(OsString::from("cargo-machete"), &["--with-metadata"], &[])
}

fn panic_surface() -> Result<(), String> {
    let output = Command::new(cargo())
        .args([
            "clippy",
            "--workspace",
            "--lib",
            "--all-features",
            "--message-format=json",
            "--",
            "-A",
            "clippy::all",
            "-W",
            "clippy::unwrap_used",
            "-W",
            "clippy::expect_used",
        ])
        .output()
        .map_err(|err| format!("failed to run cargo clippy panic-surface gate: {err}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        return Err(format!(
            "cargo clippy panic-surface gate failed with status {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        ));
    }

    let mut unwrap_used_count = 0usize;
    let mut expect_used_count = 0usize;
    for message in stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
    {
        if message
            .get("reason")
            .and_then(serde_json::Value::as_str)
            .is_none_or(|reason| reason != "compiler-message")
        {
            continue;
        }
        if let Some(code) = message
            .get("message")
            .and_then(|message| message.get("code"))
            .and_then(|code| code.get("code"))
            .and_then(serde_json::Value::as_str)
        {
            match code {
                "clippy::unwrap_used" => unwrap_used_count += 1,
                "clippy::expect_used" => expect_used_count += 1,
                _ => {}
            }
        }
    }

    if unwrap_used_count > PANIC_SURFACE_UNWRAP_USED_BASELINE {
        return Err(format!(
            "panic-surface ratchet exceeded: clippy::unwrap_used reported {unwrap_used_count}, baseline is {PANIC_SURFACE_UNWRAP_USED_BASELINE}"
        ));
    }
    if expect_used_count > PANIC_SURFACE_EXPECT_USED_BASELINE {
        return Err(format!(
            "panic-surface ratchet exceeded: clippy::expect_used reported {expect_used_count}, baseline is {PANIC_SURFACE_EXPECT_USED_BASELINE}"
        ));
    }

    println!(
        "panic-surface ratchet: clippy::unwrap_used {unwrap_used_count}/{PANIC_SURFACE_UNWRAP_USED_BASELINE}, clippy::expect_used {expect_used_count}/{PANIC_SURFACE_EXPECT_USED_BASELINE}"
    );
    Ok(())
}

fn no_std() -> Result<(), String> {
    run_program(
        OsString::from("rustup"),
        &["target", "add", NO_STD_TARGET],
        &[],
    )?;
    run_cargo(&["check", "-p", "j2k-core", "--target", NO_STD_TARGET])?;
    run_cargo(&["check", "-p", "j2k-codec-math", "--target", NO_STD_TARGET])?;
    run_cargo(&[
        "check",
        "-p",
        "j2k-profile",
        "--no-default-features",
        "--target",
        NO_STD_TARGET,
    ])?;
    run_cargo(&[
        "check",
        "-p",
        "j2k-native",
        "--no-default-features",
        "--target",
        NO_STD_TARGET,
    ])?;
    run_program(
        OsString::from("rustup"),
        &["target", "add", NO_STD_CORE_PORTABLE_TARGET],
        &[],
    )?;
    run_cargo(&[
        "check",
        "-p",
        "j2k-core",
        "--target",
        NO_STD_CORE_PORTABLE_TARGET,
    ])?;
    run_cargo(&[
        "check",
        "-p",
        "j2k-codec-math",
        "--target",
        NO_STD_CORE_PORTABLE_TARGET,
    ])
}

fn verify_unsafe_audit() -> Result<(), String> {
    let audit_path = Path::new("docs/unsafe-audit.md");
    let audit = fs::read_to_string(audit_path)
        .map_err(|err| format!("failed to read {}: {err}", audit_path.display()))?;
    if !audit.contains("| Path | Scope | Invariants | Regression guards |") {
        return Err(
            "docs/unsafe-audit.md must include Path/Scope/Invariants/Regression guards columns"
                .to_string(),
        );
    }
    let mut malformed_rows = Vec::new();
    let mut documented_paths = BTreeSet::new();
    for line in audit.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("| `crates/") {
            continue;
        }
        let cells = trimmed.split('|').map(str::trim).collect::<Vec<_>>();
        if let Some(path) = cells.get(1).and_then(|cell| {
            cell.strip_prefix('`')
                .and_then(|cell| cell.strip_suffix('`'))
        }) {
            documented_paths.insert(path.to_string());
        }
        if cells.len() < 6
            || cells[1].is_empty()
            || cells[2].is_empty()
            || cells[3].is_empty()
            || cells[4].is_empty()
            || cells[3].eq_ignore_ascii_case("tbd")
            || cells[4].eq_ignore_ascii_case("tbd")
        {
            malformed_rows.push(trimmed.to_string());
        }
    }
    if !malformed_rows.is_empty() {
        return Err(format!(
            "docs/unsafe-audit.md has unsafe rows missing invariants or regression guards: {malformed_rows:?}"
        ));
    }
    let mut missing = Vec::new();
    let mut current_unsafe = BTreeSet::new();
    for path in rust_sources(Path::new("crates"))? {
        let source = fs::read_to_string(&path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        if source.contains("unsafe ") || source.contains("unsafe{") {
            let relative = path.to_string_lossy().replace('\\', "/");
            current_unsafe.insert(relative);
        }
    }
    for relative in &current_unsafe {
        if !documented_paths.contains(relative) {
            missing.push(relative.clone());
        }
    }
    let stale = documented_paths
        .difference(&current_unsafe)
        .cloned()
        .collect::<Vec<_>>();
    if !stale.is_empty() {
        return Err(format!(
            "docs/unsafe-audit.md has stale unsafe source entries: {stale:?}"
        ));
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "docs/unsafe-audit.md is missing unsafe source entries: {missing:?}"
        ))
    }
}

fn downstream_smoke() -> Result<(), String> {
    run_cargo(&["test", "-p", "j2k", "--examples"])?;
    run_cargo(&["test", "-p", "j2k-transcode", "--examples"])
}

fn release_integrity() -> Result<(), String> {
    let metadata = cargo_metadata()?;
    let workspace_version = workspace_version()?;
    let publishable_set = str_set(PUBLISHABLE_PACKAGES);
    let docs_set = str_set(STABLE_DOC_LIBRARY_PACKAGES);
    let semver_set = str_set(STABLE_SEMVER_PACKAGES);
    let mut errors = Vec::new();

    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata did not contain a packages array".to_string())?;
    let workspace_members = metadata
        .get("workspace_members")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata did not contain a workspace_members array".to_string())?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    let mut workspace_names = BTreeSet::new();

    let unpublished_members = packages
        .iter()
        .filter(|package| {
            package
                .get("id")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|id| workspace_members.contains(id))
                && publish_false(package)
        })
        .filter_map(|package| package.get("name").and_then(serde_json::Value::as_str))
        .collect::<BTreeSet<_>>();

    for package in packages {
        let id = package
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if !workspace_members.contains(id) {
            continue;
        }

        let name = package_name(package)?;
        workspace_names.insert(name.to_string());
        let listed_publishable = publishable_set.contains(name);
        let explicitly_unpublished = publish_false(package);

        if listed_publishable && explicitly_unpublished {
            errors.push(format!(
                "`{name}` is listed as publishable but has `publish = false`"
            ));
            continue;
        }
        if !listed_publishable && !explicitly_unpublished {
            errors.push(format!(
                "`{name}` is neither in PUBLISHABLE_PACKAGES nor explicitly `publish = false`"
            ));
            continue;
        }
        if !listed_publishable {
            continue;
        }

        validate_unpublished_dependencies(name, package, &unpublished_members, &mut errors);

        let version = package
            .get("version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if version != workspace_version {
            errors.push(format!(
                "`{name}` version {version} does not match workspace version {workspace_version}"
            ));
        }
        if package
            .get("readme")
            .and_then(serde_json::Value::as_str)
            .is_none()
        {
            errors.push(format!("`{name}` is publishable but has no package README"));
        }
        if !has_docs_rs_metadata(package) {
            errors.push(format!(
                "`{name}` is publishable but missing [package.metadata.docs.rs] with all-features and empty targets"
            ));
        }
        if has_lib_target(package) {
            if !docs_set.contains(name) {
                errors.push(format!(
                    "`{name}` has a library target but is missing from STABLE_DOC_LIBRARY_PACKAGES"
                ));
            }
            if !semver_set.contains(name) {
                errors.push(format!(
                    "`{name}` has a library target but is missing from STABLE_SEMVER_PACKAGES"
                ));
            }
        } else if name != "j2k-cli" {
            errors.push(format!(
                "`{name}` is publishable but has no library target and no explicit release-integrity exemption"
            ));
        }
    }

    for package in PUBLISHABLE_PACKAGES {
        if !workspace_names.contains(*package) {
            errors.push(format!(
                "`{package}` is listed in PUBLISHABLE_PACKAGES but is not a workspace member"
            ));
        }
    }
    for package in STABLE_DOC_LIBRARY_PACKAGES {
        if !publishable_set.contains(package) {
            errors.push(format!(
                "`{package}` is in STABLE_DOC_LIBRARY_PACKAGES but is not publishable"
            ));
        }
    }
    for package in STABLE_SEMVER_PACKAGES {
        if !publishable_set.contains(package) {
            errors.push(format!(
                "`{package}` is in STABLE_SEMVER_PACKAGES but is not publishable"
            ));
        }
    }

    validate_publish_workflow(&mut errors)?;
    validate_publish_script(&mut errors)?;
    validate_release_docs(&mut errors)?;

    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "release integrity violations:\n- {}",
            errors.join("\n- ")
        ))
    }
}

fn validate_unpublished_dependencies(
    name: &str,
    package: &serde_json::Value,
    unpublished_members: &BTreeSet<&str>,
    errors: &mut Vec<String>,
) {
    let dependencies = package
        .get("dependencies")
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for dependency in dependencies {
        let dep_name = dependency
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if !unpublished_members.contains(dep_name) {
            continue;
        }
        let kind = dependency
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("normal");
        let req = dependency
            .get("req")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("*");
        if kind != "dev" {
            errors.push(format!(
                "`{name}` has a {kind} dependency on unpublished crate `{dep_name}`; \
                 cargo publish cannot resolve it"
            ));
        } else if req != "*" {
            errors.push(format!(
                "`{name}` has a versioned dev-dependency `{dep_name} = \"{req}\"` on an \
                 unpublished crate; drop the version so cargo publish strips the path-only dev-dep"
            ));
        }
    }
}

fn cargo_metadata() -> Result<serde_json::Value, String> {
    let data = command_output_os(
        cargo(),
        &["metadata", "--locked", "--no-deps", "--format-version", "1"],
    )?;
    serde_json::from_str(&data).map_err(|err| format!("failed to parse cargo metadata: {err}"))
}

fn package_name(package: &serde_json::Value) -> Result<&str, String> {
    package
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "cargo metadata package missing name".to_string())
}

fn publish_false(package: &serde_json::Value) -> bool {
    package
        .get("publish")
        .and_then(serde_json::Value::as_array)
        .is_some_and(Vec::is_empty)
}

fn has_lib_target(package: &serde_json::Value) -> bool {
    package
        .get("targets")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|targets| {
            targets.iter().any(|target| {
                target
                    .get("kind")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|kind| {
                        kind.iter()
                            .any(|entry| entry.as_str().is_some_and(|entry| entry == "lib"))
                    })
            })
        })
}

fn has_docs_rs_metadata(package: &serde_json::Value) -> bool {
    let Some(docs_rs) = package
        .get("metadata")
        .and_then(|metadata| metadata.get("docs"))
        .and_then(|docs| docs.get("rs"))
    else {
        return false;
    };

    docs_rs
        .get("all-features")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
        && docs_rs
            .get("targets")
            .and_then(serde_json::Value::as_array)
            .is_some_and(Vec::is_empty)
}

fn validate_publish_workflow(errors: &mut Vec<String>) -> Result<(), String> {
    let workflow_path = Path::new(".github/workflows/publish.yml");
    let workflow = fs::read_to_string(workflow_path)
        .map_err(|err| format!("failed to read {}: {err}", workflow_path.display()))?;
    let workflow: serde_yaml_ng::Value = serde_yaml_ng::from_str(&workflow)
        .map_err(|err| format!("failed to parse {}: {err}", workflow_path.display()))?;
    let mut crates = Vec::new();
    collect_publish_workflow_crates(&workflow, &mut crates);

    let expected = PUBLISHABLE_PACKAGES
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if crates != expected {
        errors.push(format!(
            "{} publish order is {:?}, expected {:?}",
            workflow_path.display(),
            crates,
            expected
        ));
    }

    let seen = crates.iter().map(String::as_str).collect::<BTreeSet<_>>();
    for package in PUBLISHABLE_PACKAGES {
        if !seen.contains(package) {
            errors.push(format!(
                "{} is missing publish job for `{package}`",
                workflow_path.display()
            ));
        }
    }
    for package in crates {
        if !PUBLISHABLE_PACKAGES.contains(&package.as_str()) {
            errors.push(format!(
                "{} publishes unknown workspace crate `{package}`",
                workflow_path.display()
            ));
        }
    }

    Ok(())
}

fn collect_publish_workflow_crates(value: &serde_yaml_ng::Value, crates: &mut Vec<String>) {
    match value {
        serde_yaml_ng::Value::String(text) => {
            for line in text.lines() {
                if let Some(package) = publish_crate_from_run_line(line) {
                    crates.push(package);
                }
            }
        }
        serde_yaml_ng::Value::Sequence(items) => {
            for item in items {
                collect_publish_workflow_crates(item, crates);
            }
        }
        serde_yaml_ng::Value::Mapping(map) => {
            for value in map.values() {
                collect_publish_workflow_crates(value, crates);
            }
        }
        _ => {}
    }
}

fn publish_crate_from_run_line(line: &str) -> Option<String> {
    let marker = "scripts/publish-crate.sh";
    let after = line.split_once(marker)?.1;
    after
        .split_whitespace()
        .next()
        .map(|package| package.trim_matches(['"', '\'']).to_string())
}

fn validate_publish_script(errors: &mut Vec<String>) -> Result<(), String> {
    let script_path = Path::new("scripts/publish-crate.sh");
    let script = fs::read_to_string(script_path)
        .map_err(|err| format!("failed to read {}: {err}", script_path.display()))?;
    let crates = shell_array_values(&script, "publishable_crates").ok_or_else(|| {
        format!(
            "{} does not define the publishable_crates shell array",
            script_path.display()
        )
    })?;
    let expected = PUBLISHABLE_PACKAGES
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if crates != expected {
        errors.push(format!(
            "{} publishable_crates is {:?}, expected {:?}",
            script_path.display(),
            crates,
            expected
        ));
    }
    Ok(())
}

fn shell_array_values(script: &str, name: &str) -> Option<Vec<String>> {
    let marker = format!("{name}=(");
    let mut values = Vec::new();
    let mut in_array = false;
    for raw_line in script.lines() {
        let line = raw_line.trim();
        if !in_array {
            if line == marker {
                in_array = true;
            }
            continue;
        }
        if line == ")" {
            return Some(values);
        }
        let line = line.split('#').next()?.trim();
        if line.is_empty() {
            continue;
        }
        values.extend(
            line.split_whitespace()
                .map(|entry| entry.trim_matches(['"', '\'']).to_string()),
        );
    }
    None
}

fn validate_release_docs(errors: &mut Vec<String>) -> Result<(), String> {
    let release_doc_path = Path::new("docs/release.md");
    let release_doc = fs::read_to_string(release_doc_path)
        .map_err(|err| format!("failed to read {}: {err}", release_doc_path.display()))?;
    for package in PUBLISHABLE_PACKAGES {
        if !release_doc.contains(&format!("`{package}`")) {
            errors.push(format!(
                "{} does not document publishable crate `{package}`",
                release_doc_path.display()
            ));
        }
    }
    for required in [
        "cargo xtask release-integrity",
        "CRATES_IO_ALLOW_PUBLISHED_RERUN",
        "v<workspace.package.version>",
    ] {
        if !release_doc.contains(required) {
            errors.push(format!(
                "{} does not document `{required}`",
                release_doc_path.display()
            ));
        }
    }
    Ok(())
}

fn str_set(values: &[&'static str]) -> BTreeSet<&'static str> {
    values.iter().copied().collect()
}

fn release_cpu() -> Result<(), String> {
    let mut args = vec!["test", "--release"];
    for package in CPU_RELEASE_PACKAGES {
        args.push("-p");
        args.push(package);
    }
    run_cargo(&args)
}

fn package() -> Result<(), String> {
    ensure_clean_worktree()?;
    for package in PUBLISHABLE_PACKAGES {
        run_cargo(&["package", "-p", package, "--list"])?;
    }
    for package in REGISTRY_INDEPENDENT_PACKAGES {
        run_cargo(&["package", "-p", package, "--no-verify"])?;
    }
    for package in STAGED_DEPENDENCY_PACKAGES {
        eprintln!(
            "skipping strict package verification for {package}: unpublished workspace dependencies are staged for publication; `cargo package --list` validated package contents"
        );
    }
    Ok(())
}

fn ensure_clean_worktree() -> Result<(), String> {
    let status = process::command_output_os(OsString::from("git"), &["status", "--porcelain"])?;
    if status.trim().is_empty() {
        Ok(())
    } else {
        Err(format!(
            "working tree must be clean before packaging:\n{status}"
        ))
    }
}

fn run_cargo(args: &[&str]) -> Result<(), String> {
    run_cargo_with_env(args, &[])
}

fn run_cargo_with_env(args: &[&str], envs: &[(&str, &str)]) -> Result<(), String> {
    run_program(cargo(), args, envs)
}

fn run_cargo_test_with_pass_floor(
    args: &[&str],
    envs: &[(&str, &str)],
    min_passed: usize,
    label: &str,
) -> Result<(), String> {
    let mut test_args = args.to_vec();
    test_args.extend_from_slice(&["--", "--nocapture"]);
    let env_display = envs
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(" ");
    let display_prefix = if env_display.is_empty() {
        String::new()
    } else {
        format!("{env_display} ")
    };
    eprintln!(
        "+ {display_prefix}{} {}",
        cargo().to_string_lossy(),
        test_args.join(" ")
    );

    let mut command = Command::new(cargo());
    command.args(&test_args);
    for (key, value) in envs {
        command.env(key, value);
    }
    let output = command
        .output()
        .map_err(|err| format!("failed to start `{}`: {err}", cargo().to_string_lossy()))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    print!("{stdout}");
    eprint!("{stderr}");
    if !output.status.success() {
        return Err(format!(
            "`{}` exited with {} while running {label}",
            cargo().to_string_lossy(),
            output.status
        ));
    }

    let passed = passed_test_count(&format!("{stdout}\n{stderr}"));
    if passed < min_passed {
        return Err(format!(
            "{label} executed {passed} tests, expected at least {min_passed}; \
             check required comparator tools and skip gates"
        ));
    }
    Ok(())
}

fn passed_test_count(output: &str) -> usize {
    output
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("test result: ok.")?;
            if !rest.contains(" passed") {
                return None;
            }
            rest.split_whitespace().next()?.parse::<usize>().ok()
        })
        .sum()
}

fn run_nightly_cargo(args: &[&str]) -> Result<(), String> {
    let mut rustup_args = vec!["run", "nightly", "cargo"];
    rustup_args.extend_from_slice(args);
    run_program(OsString::from("rustup"), &rustup_args, &[])
}

fn run_nightly_cargo_in_dir_owned(dir: &str, args: &[String]) -> Result<(), String> {
    let mut rustup_args = vec![
        "run".to_string(),
        "nightly".to_string(),
        "cargo".to_string(),
    ];
    rustup_args.extend_from_slice(args);
    run_program_in_dir_owned_with_program(OsString::from("rustup"), dir, &rustup_args, &[])
}

fn run_program(program: OsString, args: &[&str], envs: &[(&str, &str)]) -> Result<(), String> {
    process::run_command(program, args, process::CommandContext::new().envs(envs))
}

fn run_program_in_dir_owned_with_program(
    program: OsString,
    dir: &str,
    args: &[String],
    envs: &[(&str, &str)],
) -> Result<(), String> {
    process::run_command_owned(
        program,
        args,
        process::CommandContext::new()
            .current_dir(Path::new(dir))
            .envs(envs),
    )
}

fn command_output(program: &str, args: &[&str]) -> Result<String, String> {
    command_output_os(OsString::from(program), args)
}

fn command_output_allow_failure(program: &str, args: &[&str]) -> Result<String, String> {
    process::command_output_allow_failure(program, args)
}

fn command_output_os(program: OsString, args: &[&str]) -> Result<String, String> {
    process::command_output_os(program, args)
}

fn command_output_os_detailed(program: OsString, args: &[&str]) -> Result<String, String> {
    let display = format!("{} {}", program.to_string_lossy(), args.join(" "));
    let output = process::command_output(program, args, process::CommandContext::new())?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");
    Err(format!(
        "`{display}` exited with {}:\n{}",
        output.status,
        combined.trim()
    ))
}

fn host_description() -> String {
    command_output("uname", &["-a"])
        .unwrap_or_else(|_| format!("{} {}", env::consts::OS, env::consts::ARCH))
}

fn workspace_version() -> Result<String, String> {
    let manifest = fs::read_to_string("Cargo.toml")
        .map_err(|err| format!("failed to read Cargo.toml: {err}"))?;
    manifest
        .lines()
        .find_map(|line| {
            let line = line.trim();
            line.strip_prefix("version")
                .and_then(|rest| rest.split('"').nth(1))
                .map(str::to_string)
        })
        .ok_or_else(|| "failed to find workspace package version".to_string())
}

fn comparator_versions() -> Vec<(String, String)> {
    vec![
        (
            "OpenJPEG".to_string(),
            comparator_command_version("J2K_OPENJPEG_DECOMPRESS_BIN", "opj_decompress", &["-h"]),
        ),
        ("Grok".to_string(), grok_comparator_version()),
        (
            "libjpeg-turbo".to_string(),
            command_output("pkg-config", &["--modversion", "libturbojpeg"])
                .map(|version| format!("pkg-config libturbojpeg {version}"))
                .unwrap_or_else(|err| format!("unavailable: {err}")),
        ),
    ]
}

fn grok_comparator_version() -> String {
    if let Ok(version) = command_output("pkg-config", &["--modversion", "libgrokj2k"]) {
        let lib_dir = command_output("pkg-config", &["--variable", "libdir", "libgrokj2k"])
            .unwrap_or_else(|err| format!("libdir unavailable: {err}"));
        return format!("pkg-config libgrokj2k {version}; libdir: {lib_dir}");
    }
    env::var("J2K_GROK_ROOT")
        .map(|root| format!("configured root: {root}"))
        .unwrap_or_else(|_| {
            "unavailable: pkg-config libgrokj2k and J2K_GROK_ROOT not set".to_string()
        })
}

fn comparator_command_version(env_var: &str, fallback: &str, args: &[&str]) -> String {
    let program = env::var(env_var).unwrap_or_else(|_| fallback.to_string());
    let path = program.clone();
    command_output_allow_failure(&program, args)
        .map(|version| format!("{}; path: {path}", best_version_line(&version)))
        .unwrap_or_else(|err| format!("unavailable: {err}; path: {path}"))
}

fn best_version_line(output: &str) -> &str {
    output
        .lines()
        .find(|line| line.contains("compiled against") || line.contains("version"))
        .or_else(|| output.lines().find(|line| !line.trim().is_empty()))
        .unwrap_or("version unavailable")
}

fn render_benchmark_report(report: &BenchmarkReport) -> String {
    let mut out = String::new();
    writeln!(&mut out, "# Benchmark publication report").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "- command: {}", report.command).unwrap();
    writeln!(&mut out, "- host: {}", report.host).unwrap();
    writeln!(&mut out, "- rustc: {}", one_line(&report.rustc)).unwrap();
    writeln!(&mut out, "- cargo: {}", one_line(&report.cargo)).unwrap();
    writeln!(&mut out, "- crate revision: {}", report.git_revision).unwrap();
    writeln!(
        &mut out,
        "- workspace version: {}",
        report.workspace_version
    )
    .unwrap();
    writeln!(&mut out, "- input source: {}", report.input_source).unwrap();
    writeln!(
        &mut out,
        "- J2K_COMPARE_THREADS: {}",
        report.compare_threads
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## comparator versions").unwrap();
    for (name, version) in &report.comparator_versions {
        writeln!(&mut out, "- {name}: {version}").unwrap();
    }
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## skipped rows").unwrap();
    if report.skipped_rows.is_empty() {
        writeln!(&mut out, "- none recorded").unwrap();
    } else {
        for row in &report.skipped_rows {
            writeln!(&mut out, "- {row}").unwrap();
        }
    }
    out
}

fn one_line(value: &str) -> String {
    value.lines().next().unwrap_or(value).to_string()
}

fn split_semicolon_list(value: &str) -> Vec<String> {
    value
        .split(';')
        .map(str::trim)
        .filter(|row| !row.is_empty())
        .map(str::to_string)
        .collect()
}

fn rust_sources(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    collect_rust_sources(root, &mut out)?;
    Ok(out)
}

fn collect_rust_sources(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in
        fs::read_dir(dir).map_err(|err| format!("failed to read {}: {err}", dir.display()))?
    {
        let entry =
            entry.map_err(|err| format!("failed to read {} entry: {err}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_rust_sources(&path, out)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn print_bench_report_help() {
    println!(
        "usage: cargo xtask bench-report [--command <command>] [--input-source <source>] \
         [--skipped-row <row>]... [--out <path>]"
    );
}

fn print_stable_api_help() {
    println!(
        "usage: cargo xtask stable-api [--write]\n\n\
         Without --write, checks docs/stable-api-1.0.public-api.txt against \
         cargo-public-api output for all 1.0-stable library crates. With \
         --write, refreshes the snapshot. This task must run on macOS so \
         target-gated Metal APIs are included."
    );
}

fn print_codec_math_codegen_help() {
    println!(
        "usage: cargo xtask codec-math-codegen [--write]\n\n\
         Without --write, checks generated Rust and Metal codec-math fragments \
         against the Rust source of truth. With --write, refreshes the fragments."
    );
}

fn print_help() {
    println!(
        "usage: cargo xtask <task>\n\n\
         tasks:\n\
          ci            fmt, clippy, panic-surface, test, docs, and unsafe-audit\n\
           fmt           check rustfmt\n\
           clippy        run clippy with warnings denied\n\
           clippy-strict run stricter J2K clippy gates\n\
           test          run workspace tests\n\
           nextest       run workspace tests with cargo-nextest\n\
           doc           build workspace docs with warnings denied\n\
           typos         run typos\n\
           bench-build   compile benchmark targets\n\
           bench-report  print or write a benchmark publication report\n\
           adoption-benchmark run CPU comparator and optional CUDA/Metal adoption benchmark bundle [--features adoption]\n\
           adoption-curate stage inspectable external J2K fixtures and a pinned manifest [--features adoption]\n\
           adoption-manifest generate decode and encode fixture manifests for adoption benchmarks [--features adoption]\n\
           adoption-materialize stage source images into fixed J2K/HTJ2K fixtures and manifests [--features adoption]\n\
           adoption-report render a marketing-safe report from an adoption benchmark bundle [--features adoption]\n\
          public-support verify the public J2K/HTJ2K support matrix and publication gates [--final]\n\
          j2k-bench-signoff run required OpenJPEG/Grok parity and J2K compare bench compile gates\n\
          j2k-perf-guard compare CPU J2K Criterion medians against a baseline git ref\n\
          codec-math-codegen check generated codec-math Rust and Metal fragments\n\
           fuzz-build    compile fuzz harnesses\n\
           fuzz-run      run scheduled fuzz targets with J2K_FUZZ_RUNS\n\
           stable-api    check the generated 1.0 public API inventory snapshot\n\
           semver        verify computed release types and reviewed API diff [--write-report]\n\
           deny          run cargo-deny\n\
           miri          run selected CPU/no_std crates under Miri\n\
           machete       run cargo-machete unused-dependency scan\n\
           panic-surface run the production-library unwrap/expect ratchet\n\
           no-std        check no_std-compatible codec crates\n\
           unsafe-audit  verify docs/unsafe-audit.md lists unsafe Rust sources\n\
           downstream-smoke run facade and transcode examples used by integration docs\n\
           repo-lint     run repository policy checks owned by xtask [--strict]\n\
           release-integrity validate publish membership, docs.rs metadata, workflow order, and release docs\n\
           release-status verify one frozen SHA's CI aggregate and both GPU jobs [--sha SHA] [--repository owner/name]\n\
           release-cpu   run release-mode CPU codec tests\n\
           metal-compile compile all Metal targets and run default/pure tests on hosted macOS\n\
           release-metal run fail-closed release-mode Metal hardware validation on macOS\n\
           coverage      enforce >=80% changed executable Rust coverage [host|metal|cuda] [--base REV]\n\
           package       preflight publishable package contents from a clean worktree; strict for registry-independent crates and list-only for staged dependencies"
    );
}

#[cfg(test)]
mod tests {
    use super::passed_test_count;
    use super::perf_guard::{
        compare_estimates, discover_estimates, sync_benchmark_sources, BenchEstimate,
        RegressionOutcome,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn passed_test_count_sums_rust_test_summaries() {
        let output = "\
running 8 tests
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
running 1 test
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
";

        assert_eq!(passed_test_count(output), 9);
    }

    #[test]
    fn passed_test_count_ignores_failed_or_unrelated_lines() {
        let output = "\
test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
some other output: 12 passed
";

        assert_eq!(passed_test_count(output), 0);
    }

    #[test]
    fn compare_estimates_flags_median_regression_above_threshold() {
        let baseline = BenchEstimate {
            id: "j2k_public_decode/htj2k_gray8_full_512x512".to_string(),
            median_ns: 1_000.0,
            median_lower_ns: 990.0,
            median_upper_ns: 1_000.0,
        };
        let current = BenchEstimate {
            id: baseline.id.clone(),
            median_ns: 1_120.0,
            median_lower_ns: 1_120.0,
            median_upper_ns: 1_130.0,
        };

        let outcomes = compare_estimates(&[baseline], &[current], 10.0).unwrap();

        assert_eq!(
            outcomes,
            vec![RegressionOutcome {
                id: "j2k_public_decode/htj2k_gray8_full_512x512".to_string(),
                baseline_ns: 1_000.0,
                current_ns: 1_120.0,
                delta_percent: 12.0,
                enforced: true,
                threshold_exceeded: true,
                regressed: true,
            }]
        );
    }

    #[test]
    fn compare_estimates_allows_median_delta_at_threshold() {
        let baseline = BenchEstimate {
            id: "tier1_bitplane_decode/decode_64x64/default".to_string(),
            median_ns: 2_000.0,
            median_lower_ns: 1_990.0,
            median_upper_ns: 2_000.0,
        };
        let current = BenchEstimate {
            id: baseline.id.clone(),
            median_ns: 2_200.0,
            median_lower_ns: 2_200.0,
            median_upper_ns: 2_210.0,
        };

        let outcomes = compare_estimates(&[baseline], &[current], 10.0).unwrap();

        assert!(!outcomes[0].regressed);
        assert_eq!(outcomes[0].delta_percent, 10.0);
    }

    #[test]
    fn compare_estimates_reports_missing_current_result() {
        let baseline = BenchEstimate {
            id: "j2k_public_decode/htj2k_gray8_full_512x512".to_string(),
            median_ns: 500.0,
            median_lower_ns: 490.0,
            median_upper_ns: 510.0,
        };

        let err = compare_estimates(&[baseline], &[], 10.0).unwrap_err();

        assert!(
            err.contains("missing current benchmark result"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn compare_estimates_requires_confident_material_regression() {
        let tiny_baseline = BenchEstimate {
            id: "htj2k_refinement_block_decode/ds0_ht_09_b11_sigprop".to_string(),
            median_ns: 100.0,
            median_lower_ns: 99.0,
            median_upper_ns: 101.0,
        };
        let tiny_current = BenchEstimate {
            id: tiny_baseline.id.clone(),
            median_ns: 130.0,
            median_lower_ns: 129.0,
            median_upper_ns: 131.0,
        };
        let noisy_baseline = BenchEstimate {
            id: "j2k_public_decode_gray/gray8_full_128x128".to_string(),
            median_ns: 610_000.0,
            median_lower_ns: 606_000.0,
            median_upper_ns: 615_000.0,
        };
        let noisy_current = BenchEstimate {
            id: noisy_baseline.id.clone(),
            median_ns: 673_000.0,
            median_lower_ns: 664_000.0,
            median_upper_ns: 681_000.0,
        };

        let outcomes = compare_estimates(
            &[tiny_baseline, noisy_baseline],
            &[tiny_current, noisy_current],
            10.0,
        )
        .unwrap();

        assert!(outcomes.iter().all(|outcome| !outcome.regressed));
    }

    #[test]
    fn discover_estimates_reads_criterion_median_point_estimates() {
        let root = temp_dir("j2k-perf-guard-test");
        let estimate_path = root
            .join("target")
            .join("criterion")
            .join("j2k_public_decode")
            .join("rgb8_full_128x128")
            .join("new");
        fs::create_dir_all(&estimate_path).unwrap();
        fs::write(
            estimate_path.join("estimates.json"),
            r#"{"median":{"point_estimate":321.5,"confidence_interval":{"lower_bound":321.5,"upper_bound":321.5}}}"#,
        )
        .unwrap();

        let estimates = discover_estimates(&root.join("target").join("criterion")).unwrap();

        assert_eq!(
            estimates,
            vec![BenchEstimate {
                id: "j2k_public_decode/rgb8_full_128x128".to_string(),
                median_ns: 321.5,
                median_lower_ns: 321.5,
                median_upper_ns: 321.5,
            }]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn sync_benchmark_sources_overlays_current_bench_files() {
        let root = temp_dir("j2k-perf-guard-sync-test");
        let source = root.join("source");
        let target = root.join("target");
        let jpeg_manifest = "crates/j2k-jpeg/Cargo.toml";
        let cuda_manifest = "crates/j2k-cuda/Cargo.toml";
        let public_bench = "crates/j2k/benches/public_api.rs";
        let jpeg_encode_bench = "crates/j2k-jpeg/benches/encode_cpu.rs";
        let cuda_decode_bench = "crates/j2k-cuda/benches/htj2k_decode.rs";
        let cuda_encode_bench = "crates/j2k-cuda/benches/htj2k_encode.rs";
        let native_bench = "crates/j2k-native/benches/tier1_bitplane.rs";
        let native_sigprop_bench = "crates/j2k-native/benches/htj2k_sigprop_phase.rs";
        let native_fixture = "crates/j2k-native/fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k";
        fs::create_dir_all(source.join("crates/j2k/benches")).unwrap();
        fs::create_dir_all(source.join("crates/j2k-jpeg/benches")).unwrap();
        fs::create_dir_all(source.join("crates/j2k-cuda/benches")).unwrap();
        fs::create_dir_all(source.join("crates/j2k-native/benches")).unwrap();
        fs::create_dir_all(source.join("crates/j2k-native/fixtures/htj2k")).unwrap();
        fs::create_dir_all(target.join("crates/j2k-jpeg")).unwrap();
        fs::create_dir_all(target.join("crates/j2k-cuda")).unwrap();
        fs::create_dir_all(target.join("crates/j2k/benches")).unwrap();
        fs::create_dir_all(target.join("crates/j2k-jpeg/benches")).unwrap();
        fs::create_dir_all(target.join("crates/j2k-cuda/benches")).unwrap();
        fs::create_dir_all(target.join("crates/j2k-native/benches")).unwrap();
        fs::create_dir_all(target.join("crates/j2k-native/fixtures/htj2k")).unwrap();
        fs::write(
            source.join(jpeg_manifest),
            "[package]\nname = \"j2k-jpeg\"\n\n[[bench]]\nname = \"encode_cpu\"\nharness = false\n",
        )
        .unwrap();
        fs::write(
            target.join(jpeg_manifest),
            "[package]\nname = \"j2k-jpeg\"\n",
        )
        .unwrap();
        fs::write(
            target.join(cuda_manifest),
            "[package]\nname = \"j2k-cuda\"\n",
        )
        .unwrap();
        fs::write(source.join(public_bench), "current public bench").unwrap();
        fs::write(source.join(jpeg_encode_bench), "current jpeg encode bench").unwrap();
        fs::write(source.join(cuda_decode_bench), "current cuda decode bench").unwrap();
        fs::write(source.join(cuda_encode_bench), "current cuda encode bench").unwrap();
        fs::write(source.join(native_bench), "current native bench").unwrap();
        fs::write(source.join(native_sigprop_bench), "current sigprop bench").unwrap();
        fs::write(source.join(native_fixture), "current fixture").unwrap();
        fs::write(target.join(public_bench), "old public bench").unwrap();
        fs::write(target.join(jpeg_encode_bench), "old jpeg encode bench").unwrap();
        fs::write(target.join(cuda_decode_bench), "old cuda decode bench").unwrap();
        fs::write(target.join(cuda_encode_bench), "old cuda encode bench").unwrap();
        fs::write(target.join(native_bench), "old native bench").unwrap();
        fs::write(target.join(native_sigprop_bench), "old sigprop bench").unwrap();
        fs::write(target.join(native_fixture), "old fixture").unwrap();

        sync_benchmark_sources(&source, &target).unwrap();

        assert_eq!(
            fs::read_to_string(target.join(public_bench)).unwrap(),
            "current public bench"
        );
        assert_eq!(
            fs::read_to_string(target.join(jpeg_encode_bench)).unwrap(),
            "current jpeg encode bench"
        );
        let target_jpeg_manifest = fs::read_to_string(target.join(jpeg_manifest)).unwrap();
        assert!(
            target_jpeg_manifest.contains("name = \"encode_cpu\"")
                && target_jpeg_manifest.contains("harness = false"),
            "baseline overlay must register encode_cpu as a Criterion bench"
        );
        assert_eq!(
            fs::read_to_string(target.join(cuda_decode_bench)).unwrap(),
            "current cuda decode bench"
        );
        assert_eq!(
            fs::read_to_string(target.join(cuda_encode_bench)).unwrap(),
            "current cuda encode bench"
        );
        let target_cuda_manifest = fs::read_to_string(target.join(cuda_manifest)).unwrap();
        assert!(
            target_cuda_manifest.contains("name = \"htj2k_decode\"")
                && target_cuda_manifest.contains("name = \"htj2k_encode\"")
                && target_cuda_manifest.contains("required-features = [\"cuda-runtime\"]"),
            "baseline overlay must register CUDA HTJ2K Criterion benches"
        );
        assert_eq!(
            fs::read_to_string(target.join(native_bench)).unwrap(),
            "current native bench"
        );
        assert_eq!(
            fs::read_to_string(target.join(native_sigprop_bench)).unwrap(),
            "current sigprop bench"
        );
        assert_eq!(
            fs::read_to_string(target.join(native_fixture)).unwrap(),
            "current fixture"
        );
        let _ = fs::remove_dir_all(root);
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("{name}-{}-{nanos}", std::process::id()));
        dir
    }
}
