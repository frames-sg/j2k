use std::env;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

mod perf_guard;

const PUBLISHABLE_PACKAGES: &[&str] = &[
    "signinum-core",
    "signinum-cuda-runtime",
    "signinum-profile",
    "signinum-j2k-native",
    "signinum-jpeg",
    "signinum-tilecodec",
    "signinum-j2k",
    "signinum-transcode",
    "signinum-jpeg-metal",
    "signinum-j2k-metal",
    "signinum-transcode-metal",
    "signinum-jpeg-cuda",
    "signinum-j2k-cuda",
    "signinum-cli",
    "signinum",
];

const REGISTRY_INDEPENDENT_PACKAGES: &[&str] =
    &["signinum-core", "signinum-cuda-runtime", "signinum-profile"];

const STAGED_DEPENDENCY_PACKAGES: &[&str] = &[
    "signinum-j2k-native",
    "signinum-jpeg",
    "signinum-tilecodec",
    "signinum-j2k",
    "signinum-transcode",
    "signinum-jpeg-metal",
    "signinum-j2k-metal",
    "signinum-transcode-metal",
    "signinum-jpeg-cuda",
    "signinum-j2k-cuda",
    "signinum-cli",
    "signinum",
];

const CPU_RELEASE_PACKAGES: &[&str] = &[
    "signinum-core",
    "signinum-jpeg",
    "signinum-j2k-native",
    "signinum-j2k",
    "signinum-tilecodec",
    "signinum-cli",
    "signinum",
];

const STABLE_SEMVER_PACKAGES: &[&str] = &[
    "signinum",
    "signinum-core",
    "signinum-jpeg",
    "signinum-j2k",
    "signinum-tilecodec",
    "signinum-jpeg-metal",
    "signinum-j2k-metal",
    "signinum-jpeg-cuda",
    "signinum-j2k-cuda",
    "signinum-transcode",
    "signinum-transcode-metal",
    "signinum-j2k-native",
    "signinum-cuda-runtime",
    "signinum-profile",
];

const STABLE_DOC_LIBRARY_PACKAGES: &[&str] = &[
    "signinum",
    "signinum-core",
    "signinum-jpeg",
    "signinum-j2k",
    "signinum-tilecodec",
    "signinum-jpeg-metal",
    "signinum-j2k-metal",
    "signinum-jpeg-cuda",
    "signinum-j2k-cuda",
    "signinum-transcode",
    "signinum-transcode-metal",
    "signinum-j2k-native",
    "signinum-cuda-runtime",
    "signinum-profile",
];

const STABLE_API_SNAPSHOT: &str = "docs/stable-api-1.0.public-api.txt";
const CARGO_PUBLIC_API_VERSION: &str = "0.52.0";

const NO_STD_TARGET: &str = "aarch64-unknown-none";

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
        "j2k-bench-signoff" => j2k_bench_signoff(),
        "j2k-perf-guard" => perf_guard::j2k_perf_guard(env::args().skip(2)),
        "fuzz-build" => fuzz_build(),
        "fuzz-run" => fuzz_run(),
        "stable-api" => stable_api(env::args().skip(2)),
        "semver" => semver(),
        "deny" => deny(),
        "machete" => machete(),
        "no-std" => no_std(),
        "unsafe-audit" => verify_unsafe_audit(),
        "downstream-smoke" => downstream_smoke(),
        "release-cpu" => release_cpu(),
        "release-metal" => release_metal(),
        "coverage" => coverage(),
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
    clippy()?;
    test()?;
    doc()?;
    verify_unsafe_audit()
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
        "signinum-j2k-native",
        "-p",
        "signinum-j2k",
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

    test_workspace_without_benches(&["--exclude", "signinum-j2k-metal"])?;
    if skip_j2k_metal_runtime_on_hosted_github_macos() {
        eprintln!(
            "skipping signinum-j2k-metal runtime tests on GitHub-hosted macOS; \
             self-hosted gpu-validation runs the Metal runtime suite"
        );
        return test_package_without_benches("signinum-j2k-metal", true);
    }
    test_package_without_benches("signinum-j2k-metal", false)
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

    let mut doc_args = vec!["test", "--workspace", "--all-features", "--doc"];
    doc_args.extend_from_slice(extra_args);
    run_cargo(&doc_args)
}

fn test_package_without_benches(package: &str, no_run: bool) -> Result<(), String> {
    let mut test_args = vec![
        "test",
        "-p",
        package,
        "--all-features",
        "--lib",
        "--bins",
        "--tests",
    ];
    if no_run {
        test_args.push("--no-run");
    }

    if no_run {
        return run_cargo(&test_args);
    }

    run_cargo_with_env(&test_args, &[("RUST_TEST_THREADS", "1")])?;
    run_cargo(&["test", "-p", package, "--all-features", "--doc"])
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
        &["doc", "-p", "signinum-cli", "--no-deps"],
        &[("RUSTDOCFLAGS", "-D warnings -D missing_docs")],
    )
}

fn typos() -> Result<(), String> {
    run_program(OsString::from("typos"), &[], &[])
}

fn bench_build() -> Result<(), String> {
    run_cargo(&[
        "bench",
        "-p",
        "signinum-j2k",
        "--bench",
        "public_api",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-j2k-native",
        "--bench",
        "tier1_bitplane",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-j2k-native",
        "--bench",
        "htj2k_sigprop_phase",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-j2k-native",
        "--bench",
        "direct_cpu",
        "--no-run",
    ])?;
    run_cargo(&["bench", "-p", "signinum", "--bench", "facade", "--no-run"])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-jpeg",
        "--bench",
        "encode_cpu",
        "--no-run",
    ])?;
    run_cargo(&["bench", "-p", "signinum-jpeg", "--no-run"])?;
    run_cargo(&["bench", "-p", "signinum-jpeg-metal", "--no-run"])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-jpeg-cuda",
        "--bench",
        "device_decode",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-j2k-cuda",
        "--bench",
        "encode_stages",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-j2k-cuda",
        "--bench",
        "htj2k_decode",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-j2k-cuda",
        "--bench",
        "htj2k_encode",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-tilecodec",
        "--bench",
        "compare",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-transcode",
        "--bench",
        "dct53",
        "--no-run",
    ])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-transcode-metal",
        "--bench",
        "dct97",
        "--no-run",
    ])
}

fn j2k_bench_signoff() -> Result<(), String> {
    run_cargo_with_env(
        &[
            "test",
            "-p",
            "signinum-j2k-compare",
            "--test",
            "in_process_parity",
        ],
        &[
            ("SIGNINUM_REQUIRE_OPENJPEG", "1"),
            ("SIGNINUM_REQUIRE_GROK", "1"),
        ],
    )?;
    run_cargo_with_env(
        &["test", "-p", "signinum-j2k", "--test", "openjpeg_parity"],
        &[("SIGNINUM_REQUIRE_OPENJPEG", "1")],
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
    let mut command = env::var("SIGNINUM_BENCH_COMMAND").unwrap_or_else(|_| "not recorded".into());
    let mut input_source = env::var("SIGNINUM_BENCH_INPUT_SOURCE")
        .or_else(|_| env::var("SIGNINUM_BENCH_INPUTS"))
        .unwrap_or_else(|_| "not recorded".into());
    let mut out_path = None::<PathBuf>;
    let mut skipped_rows = env::var("SIGNINUM_BENCH_SKIPPED_ROWS")
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
        compare_threads: env::var("SIGNINUM_J2K_COMPARE_THREADS")
            .unwrap_or_else(|_| "not set".to_string()),
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
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/signinum-j2k/fuzz/Cargo.toml",
    ])?;
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/signinum-jpeg/fuzz/Cargo.toml",
    ])?;
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/signinum-tilecodec/fuzz/Cargo.toml",
    ])
}

const FUZZ_TARGETS: &[(&str, &str)] = &[
    ("crates/signinum-j2k", "decode_fuzz"),
    ("crates/signinum-j2k", "parse_fuzz"),
    ("crates/signinum-jpeg", "decode_fuzz"),
    ("crates/signinum-jpeg", "parse_fuzz"),
    ("crates/signinum-tilecodec", "decompress_fuzz"),
];

fn fuzz_run() -> Result<(), String> {
    let runs = env::var("SIGNINUM_FUZZ_RUNS").unwrap_or_else(|_| "1000".to_string());
    let max_total_time = env::var("SIGNINUM_FUZZ_MAX_TOTAL_TIME_SECONDS").ok();

    for (crate_dir, target) in FUZZ_TARGETS {
        let mut args = vec![
            "fuzz".to_string(),
            "run".to_string(),
            (*target).to_string(),
            "--".to_string(),
            format!("-runs={runs}"),
        ];
        if let Some(seconds) = &max_total_time {
            args.push(format!("-max_total_time={seconds}"));
        }
        run_cargo_in_dir_owned(crate_dir, &args)?;
    }
    Ok(())
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

fn render_stable_api_snapshot() -> Result<String, String> {
    let tool_version =
        command_output_os(cargo(), &["public-api", "--version"]).map_err(|err| {
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
        "# Signinum 1.0 Public API Snapshot\n\n\
         This file is generated by `cargo xtask stable-api --write` from \
         `cargo public-api -p <package> --all-features -sss --color never`.\n\n\
         Generator: `{tool_version}`.\n\n\
         It is the item-level companion to `docs/stable-api-1.0.md`: every \
         public module, type, trait, function, method, constant, variant, and \
         field reported here is semver-visible unless moved private before 1.0.\n"
    )
    .unwrap();

    for package in STABLE_DOC_LIBRARY_PACKAGES {
        let api = command_output_os(
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
        "## `signinum-cli`\n\n\
         `signinum-cli` is a binary package. Its stable command, stdout/stderr, \
         and exit-code contract is documented in `docs/stable-api-1.0.md`.\n"
    )
    .unwrap();

    Ok(out)
}

fn semver() -> Result<(), String> {
    let toolchain = env::var("SIGNINUM_SEMVER_TOOLCHAIN").unwrap_or_else(|_| "stable".to_string());
    let toolchain_arg = format!("+{toolchain}");
    for package in STABLE_SEMVER_PACKAGES {
        run_program(
            OsString::from("cargo"),
            &[
                toolchain_arg.as_str(),
                "semver-checks",
                "check-release",
                "--package",
                package,
                "--release-type",
                "major",
            ],
            &[],
        )?;
    }
    Ok(())
}

fn deny() -> Result<(), String> {
    run_cargo(&["deny", "check", "licenses", "advisories", "bans", "sources"])
}

fn machete() -> Result<(), String> {
    run_program(OsString::from("cargo-machete"), &[], &[])
}

fn no_std() -> Result<(), String> {
    run_program(
        OsString::from("rustup"),
        &["target", "add", NO_STD_TARGET],
        &[],
    )?;
    run_cargo(&["check", "-p", "signinum-core", "--target", NO_STD_TARGET])?;
    run_cargo(&[
        "check",
        "-p",
        "signinum-j2k-native",
        "--no-default-features",
        "--target",
        NO_STD_TARGET,
    ])
}

fn verify_unsafe_audit() -> Result<(), String> {
    let audit_path = Path::new("docs/unsafe-audit.md");
    let audit = fs::read_to_string(audit_path)
        .map_err(|err| format!("failed to read {}: {err}", audit_path.display()))?;
    let mut missing = Vec::new();
    for path in rust_sources(Path::new("crates"))? {
        let source = fs::read_to_string(&path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        if source.contains("unsafe ") || source.contains("unsafe{") {
            let relative = path.to_string_lossy().replace('\\', "/");
            if !audit.contains(&relative) {
                missing.push(relative);
            }
        }
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
    run_cargo(&["test", "-p", "signinum", "--examples"])?;
    run_cargo(&["test", "-p", "signinum-transcode", "--examples"])
}

fn release_cpu() -> Result<(), String> {
    let mut args = vec!["test", "--release"];
    for package in CPU_RELEASE_PACKAGES {
        args.push("-p");
        args.push(package);
    }
    run_cargo(&args)
}

fn release_metal() -> Result<(), String> {
    if env::consts::OS != "macos" {
        eprintln!("skipping Metal release tests on {}", env::consts::OS);
        return Ok(());
    }
    if skip_j2k_metal_runtime_on_hosted_github_macos() {
        eprintln!(
            "skipping signinum-j2k-metal release runtime tests on GitHub-hosted macOS; \
             self-hosted gpu-validation runs the Metal runtime suite"
        );
        run_cargo_with_env(
            &["test", "--release", "-p", "signinum-jpeg-metal"],
            &[("RUST_TEST_THREADS", "1")],
        )?;
        return run_cargo(&["test", "--release", "-p", "signinum-j2k-metal", "--no-run"]);
    }
    run_cargo_with_env(
        &[
            "test",
            "--release",
            "-p",
            "signinum-jpeg-metal",
            "-p",
            "signinum-j2k-metal",
        ],
        &[("RUST_TEST_THREADS", "1")],
    )
}

fn skip_j2k_metal_runtime_on_hosted_github_macos() -> bool {
    env::consts::OS == "macos"
        && env::var_os("GITHUB_ACTIONS").is_some()
        && env::var_os("SIGNINUM_RUN_HOSTED_J2K_METAL_RUNTIME_TESTS").is_none()
}

fn coverage() -> Result<(), String> {
    run_cargo(&[
        "llvm-cov",
        "--workspace",
        "--all-features",
        "--lcov",
        "--output-path",
        "lcov.info",
    ])
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
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map_err(|err| format!("failed to start `git status --porcelain`: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "`git status --porcelain` exited with {}",
            output.status
        ));
    }

    let status = String::from_utf8_lossy(&output.stdout);
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

fn run_cargo_in_dir_owned(dir: &str, args: &[String]) -> Result<(), String> {
    run_program_in_dir_owned(cargo(), dir, args, &[])
}

fn run_cargo_with_env(args: &[&str], envs: &[(&str, &str)]) -> Result<(), String> {
    run_program(cargo(), args, envs)
}

fn run_program(program: OsString, args: &[&str], envs: &[(&str, &str)]) -> Result<(), String> {
    let display = program.to_string_lossy();
    eprintln!("+ {} {}", display, args.join(" "));
    let mut command = Command::new(&program);
    command.args(args);
    for (key, value) in envs {
        command.env(key, value);
    }
    let status = command
        .status()
        .map_err(|err| format!("failed to start `{display}`: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`{display}` exited with {status}"))
    }
}

fn run_program_in_dir_owned(
    program: OsString,
    dir: &str,
    args: &[String],
    envs: &[(&str, &str)],
) -> Result<(), String> {
    let display = program.to_string_lossy();
    eprintln!("+ cd {dir} && {} {}", display, args.join(" "));
    let mut command = Command::new(&program);
    command.args(args);
    command.current_dir(dir);
    for (key, value) in envs {
        command.env(key, value);
    }
    let status = command
        .status()
        .map_err(|err| format!("failed to start `{display}`: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`{display}` exited with {status}"))
    }
}

fn cargo() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

fn command_output(program: &str, args: &[&str]) -> Result<String, String> {
    command_output_os(OsString::from(program), args)
}

fn command_output_allow_failure(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|err| format!("failed to start `{program}`: {err}"))?;
    let mut text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if text.is_empty() {
        text = stderr;
    } else if !stderr.is_empty() {
        text.push('\n');
        text.push_str(&stderr);
    }
    if text.is_empty() {
        Err(format!(
            "`{program}` exited with {} and no output",
            output.status
        ))
    } else {
        Ok(text)
    }
}

fn command_output_os(program: OsString, args: &[&str]) -> Result<String, String> {
    let display = program.to_string_lossy();
    let output = Command::new(&program)
        .args(args)
        .output()
        .map_err(|err| format!("failed to start `{display}`: {err}"))?;
    if !output.status.success() {
        return Err(format!("`{display}` exited with {}", output.status));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
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
            comparator_command_version(
                "SIGNINUM_OPENJPEG_DECOMPRESS_BIN",
                "opj_decompress",
                &["-h"],
            ),
        ),
        (
            "Grok".to_string(),
            env::var("SIGNINUM_GROK_ROOT")
                .map(|root| format!("configured root: {root}"))
                .unwrap_or_else(|_| "unavailable: SIGNINUM_GROK_ROOT not set".to_string()),
        ),
        (
            "libjpeg-turbo".to_string(),
            command_output("pkg-config", &["--modversion", "libturbojpeg"])
                .map(|version| format!("pkg-config libturbojpeg {version}"))
                .unwrap_or_else(|err| format!("unavailable: {err}")),
        ),
    ]
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
        "- SIGNINUM_J2K_COMPARE_THREADS: {}",
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
         --write, refreshes the snapshot."
    );
}

fn print_help() {
    println!(
        "usage: cargo xtask <task>\n\n\
         tasks:\n\
           ci            fmt, clippy, test, and docs\n\
           fmt           check rustfmt\n\
           clippy        run clippy with warnings denied\n\
           clippy-strict run stricter J2K clippy gates\n\
           test          run workspace tests\n\
           nextest       run workspace tests with cargo-nextest\n\
           doc           build workspace docs with warnings denied\n\
           typos         run typos\n\
           bench-build   compile benchmark targets\n\
           bench-report  print or write a benchmark publication report\n\
           j2k-bench-signoff run required OpenJPEG/Grok parity and J2K compare bench compile gates\n\
           j2k-perf-guard compare CPU J2K Criterion medians against a baseline git ref\n\
           fuzz-build    compile fuzz harnesses\n\
           fuzz-run      run scheduled fuzz targets with SIGNINUM_FUZZ_RUNS\n\
           stable-api    check the generated 1.0 public API inventory snapshot\n\
           semver        check stable library crates across the 1.0 major-release boundary\n\
           deny          run cargo-deny\n\
           machete       run cargo-machete unused-dependency scan\n\
           no-std        check no_std-compatible codec crates\n\
           unsafe-audit  verify docs/unsafe-audit.md lists unsafe Rust sources\n\
           downstream-smoke run facade and transcode examples used by integration docs\n\
           release-cpu   run release-mode CPU codec tests\n\
           release-metal run release-mode Metal tests on macOS\n\
           coverage      generate lcov.info with cargo-llvm-cov\n\
           package       preflight publishable package contents from a clean worktree; strict for registry-independent crates and list-only for staged dependencies"
    );
}

#[cfg(test)]
mod tests {
    use super::perf_guard::{
        compare_estimates, discover_estimates, sync_benchmark_sources, BenchEstimate,
        RegressionOutcome,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

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
        let root = temp_dir("signinum-perf-guard-test");
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
        let root = temp_dir("signinum-perf-guard-sync-test");
        let source = root.join("source");
        let target = root.join("target");
        let jpeg_manifest = "crates/signinum-jpeg/Cargo.toml";
        let cuda_manifest = "crates/signinum-j2k-cuda/Cargo.toml";
        let public_bench = "crates/signinum-j2k/benches/public_api.rs";
        let jpeg_encode_bench = "crates/signinum-jpeg/benches/encode_cpu.rs";
        let cuda_decode_bench = "crates/signinum-j2k-cuda/benches/htj2k_decode.rs";
        let cuda_encode_bench = "crates/signinum-j2k-cuda/benches/htj2k_encode.rs";
        let native_bench = "crates/signinum-j2k-native/benches/tier1_bitplane.rs";
        let native_sigprop_bench = "crates/signinum-j2k-native/benches/htj2k_sigprop_phase.rs";
        let native_fixture =
            "crates/signinum-j2k-native/fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k";
        fs::create_dir_all(source.join("crates/signinum-j2k/benches")).unwrap();
        fs::create_dir_all(source.join("crates/signinum-jpeg/benches")).unwrap();
        fs::create_dir_all(source.join("crates/signinum-j2k-cuda/benches")).unwrap();
        fs::create_dir_all(source.join("crates/signinum-j2k-native/benches")).unwrap();
        fs::create_dir_all(source.join("crates/signinum-j2k-native/fixtures/htj2k")).unwrap();
        fs::create_dir_all(target.join("crates/signinum-jpeg")).unwrap();
        fs::create_dir_all(target.join("crates/signinum-j2k-cuda")).unwrap();
        fs::create_dir_all(target.join("crates/signinum-j2k/benches")).unwrap();
        fs::create_dir_all(target.join("crates/signinum-jpeg/benches")).unwrap();
        fs::create_dir_all(target.join("crates/signinum-j2k-cuda/benches")).unwrap();
        fs::create_dir_all(target.join("crates/signinum-j2k-native/benches")).unwrap();
        fs::create_dir_all(target.join("crates/signinum-j2k-native/fixtures/htj2k")).unwrap();
        fs::write(
            source.join(jpeg_manifest),
            "[package]\nname = \"signinum-jpeg\"\n\n[[bench]]\nname = \"encode_cpu\"\nharness = false\n",
        )
        .unwrap();
        fs::write(
            target.join(jpeg_manifest),
            "[package]\nname = \"signinum-jpeg\"\n",
        )
        .unwrap();
        fs::write(
            target.join(cuda_manifest),
            "[package]\nname = \"signinum-j2k-cuda\"\n",
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
