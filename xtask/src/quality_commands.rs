use std::collections::BTreeSet;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::Path;

use crate::codegen_commands::codec_math_codegen;
use crate::command_support::{
    command_output_os, run_cargo, run_cargo_with_env, run_nightly_cargo,
    run_nightly_cargo_in_dir_owned, run_program, rust_sources,
};
use crate::panic_surface::panic_surface;
use crate::release_commands::STABLE_DOC_LIBRARY_PACKAGES;

const NO_STD_TARGET: &str = "aarch64-unknown-none";
const NO_STD_CORE_PORTABLE_TARGET: &str = "wasm32-unknown-unknown";

pub(super) fn ci() -> Result<(), String> {
    fmt()?;
    codec_math_codegen(std::iter::empty::<String>())?;
    clippy()?;
    panic_surface()?;
    test()?;
    doc()?;
    verify_unsafe_audit()
}

pub(super) fn repo_lint(args: impl Iterator<Item = String>) -> Result<(), String> {
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

pub(super) fn fmt() -> Result<(), String> {
    run_cargo(&["fmt", "--all", "--", "--check"])
}

pub(super) fn clippy() -> Result<(), String> {
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

pub(super) fn clippy_strict() -> Result<(), String> {
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

pub(super) fn test() -> Result<(), String> {
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

pub(super) fn nextest() -> Result<(), String> {
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

pub(super) fn doc() -> Result<(), String> {
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

pub(super) fn typos() -> Result<(), String> {
    run_program(OsString::from("typos"), &[], &[])
}

pub(super) fn fuzz_build() -> Result<(), String> {
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

pub(super) fn fuzz_run() -> Result<(), String> {
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

pub(super) fn deny() -> Result<(), String> {
    run_cargo(&["deny", "check", "licenses", "advisories", "bans", "sources"])
}

pub(super) fn miri() -> Result<(), String> {
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

pub(super) fn machete() -> Result<(), String> {
    run_program(OsString::from("cargo-machete"), &["--with-metadata"], &[])
}

pub(super) fn no_std() -> Result<(), String> {
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

pub(super) fn verify_unsafe_audit() -> Result<(), String> {
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

pub(super) fn downstream_smoke() -> Result<(), String> {
    run_cargo(&["test", "-p", "j2k", "--examples"])?;
    run_cargo(&["test", "-p", "j2k-transcode", "--examples"])
}
