// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(unix)]

#[path = "command_orchestration/support.rs"]
mod support;

use std::fs;

use support::{assert_success, Harness};

#[test]
fn release_status_executes_exact_sha_verification_without_exposing_tokens() {
    let harness = Harness::new();
    let explicit_sha = "A".repeat(40);
    let remote_sha = "B".repeat(40);

    assert_success(
        &harness.run_with_env(
            &[
                "release-status",
                "--sha",
                &explicit_sha,
                "--repository",
                "frames-sg/j2k",
            ],
            &[("GH_TOKEN", "present")],
        ),
        "release-status with explicit repository",
    );
    assert_success(
        &harness.run_with_env(
            &["release-status", "--sha", &remote_sha],
            &[("GITHUB_TOKEN", "present")],
        ),
        "release-status with remote-derived repository",
    );

    let help = harness.run(&["release-status", "--help"]);
    assert!(
        !help.status.success(),
        "help must preserve task error handling"
    );
    assert!(String::from_utf8_lossy(&help.stderr).contains(
        "usage: cargo xtask release-status --sha <40-hex-commit> [--repository owner/name]"
    ));

    let log = harness.log();
    assert_eq!(
        log.lines()
            .filter(|line| line.starts_with("python3 "))
            .count(),
        2
    );
    assert!(log.contains("git config --get remote.origin.url"));
    assert!(log.contains(&format!(
        "--candidate-sha {}",
        explicit_sha.to_ascii_lowercase()
    )));
    assert!(log.contains(&format!(
        "--candidate-sha {}",
        remote_sha.to_ascii_lowercase()
    )));
    assert!(log.contains("--repository frames-sg/j2k"));
    assert!(log.contains("--token-env GH_TOKEN"));
    assert!(log.contains("--token-env GITHUB_TOKEN"));
    assert!(!log.contains("present"), "token values reached command log");
}

#[test]
fn release_critical_orchestrators_run_from_the_workspace_without_real_cargo() {
    let harness = Harness::new();

    assert_success(&harness.run(&["ci"]), "ci");
    assert_success(&harness.run(&["bench-build"]), "bench-build");
    assert_success(&harness.run(&["j2k-bench-signoff"]), "j2k-bench-signoff");
    assert_success(&harness.run(&["release-cpu"]), "release-cpu");
    assert_success(&harness.run(&["release-integrity"]), "release-integrity");
    assert_success(&harness.run(&["package"]), "package");

    let report = harness.path("benchmark-report.md");
    assert_success(
        &harness.run(&[
            "bench-report",
            "--command",
            "cargo bench --workspace",
            "--input-source",
            "pinned fixtures",
            "--skipped-row",
            "missing optional comparator",
            "--out",
            report.to_str().expect("UTF-8 report path"),
        ]),
        "bench-report",
    );
    let report = fs::read_to_string(report).expect("read benchmark report");
    assert!(report.contains("- command: cargo bench --workspace"));
    assert!(report.contains("- input source: pinned fixtures"));
    assert!(report.contains("- missing optional comparator"));

    for args in [
        &["release-integrity", "--unknown"][..],
        &["stable-api", "--unknown"][..],
        &["semver", "--unknown"][..],
    ] {
        let output = harness.run(args);
        assert!(!output.status.success(), "{args:?} must fail closed");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("xtask failed:"),
            "{args:?} must preserve the command error"
        );
    }
    let output = harness.run(&["stable-api"]);
    assert!(
        !output.status.success(),
        "synthetic API must not replace snapshots"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("stable API snapshots are stale"),
        "stable-api must reach snapshot comparison: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = harness.run(&["semver"]);
    assert!(!output.status.success(), "synthetic API must fail semver");
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("committed stable API snapshots are stale"),
        "semver must reach live snapshot comparison: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let log = harness.log();
    assert!(log.contains("fmt --all -- --check|"));
    assert!(log.contains("bench -p j2k --bench public_api --no-run|"));
    assert!(log.contains("test -p j2k-compare --test in_process_parity -- --nocapture|"));
    assert!(log.contains("test --release -p j2k-core"));
    assert!(log.contains("package -p j2k-core --list|"));
    assert!(log.contains("publish -p j2k-core --dry-run|"));
    assert!(log.contains("package -p j2k-cli --no-verify"));
}

#[test]
fn external_quality_commands_preserve_their_complete_fake_tool_plans() {
    let harness = Harness::new();

    for task in ["typos", "miri", "machete", "no-std"] {
        assert_success(&harness.run(&[task]), task);
    }
    assert_success(
        &harness.run_with_env(
            &["fuzz-run"],
            &[
                ("J2K_FUZZ_TARGET", "aarch64-apple-darwin"),
                ("J2K_FUZZ_RUNS", "17"),
                ("J2K_FUZZ_MAX_TOTAL_TIME_SECONDS", "3"),
            ],
        ),
        "fuzz-run",
    );

    let log = harness.log();
    assert!(log.contains("typos \n"));
    assert!(log.contains("cargo-machete --with-metadata\n"));
    assert!(log.contains("rustup run nightly cargo miri test -p j2k-core"));
    assert!(log.contains("rustup target add aarch64-unknown-none"));
    assert!(log.contains("check -p j2k-core --target wasm32-unknown-unknown|"));
    assert!(log.contains(
        "rustup run nightly cargo fuzz run --target aarch64-apple-darwin decode_fuzz -- -runs=17 -max_total_time=3"
    ));
    assert!(
        !log.contains("check -p -p"),
        "duplicate package flag: {log}"
    );
}

#[test]
fn codec_math_codegen_write_is_transactional_across_both_fragments() {
    let harness = Harness::new();
    let workspace = harness.path("codegen-workspace");
    let generated = workspace.join("crates/j2k-codec-math/generated");
    fs::create_dir_all(&generated).expect("create generated fragment directory");
    let metal = generated.join("dwt97_constants.metal");
    let rust = generated.join("dwt97_constants.rs");
    fs::write(&metal, "old metal\n").expect("seed Metal fragment");
    fs::create_dir(&rust).expect("make Rust fragment target unwritable");

    let output = harness.run_in(&workspace, &["codec-math-codegen", "--write"]);

    assert!(!output.status.success(), "unwritable target must fail");
    assert_eq!(
        fs::read_to_string(&metal).expect("read Metal fragment after failure"),
        "old metal\n",
        "failure updating the Rust fragment must not commit the Metal fragment"
    );
    let entries = fs::read_dir(&generated)
        .expect("read generated fragment directory")
        .collect::<Result<Vec<_>, _>>()
        .expect("read generated fragment entries");
    assert_eq!(entries.len(), 2, "failed transaction left sidecar files");
}
