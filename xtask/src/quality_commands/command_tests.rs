// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    clippy, clippy_strict, deny, doc, downstream_smoke, fmt, fuzz_build, nextest, repo_lint, test,
    test_workspace_without_benches,
};
use crate::command_support::use_test_cargo_program;
use crate::test_command::RecordingProgram;

#[test]
fn quality_command_plans_are_complete_and_never_launch_real_tools() {
    let recording = RecordingProgram::new("quality-command-test", "");
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    fmt().expect("format plan");
    clippy().expect("Clippy plan");
    clippy_strict().expect("strict Clippy plan");
    test().expect("test plan");
    test_workspace_without_benches(&["--exclude", "fixture-package"]).expect("custom test plan");
    nextest().expect("nextest plan");
    doc().expect("documentation plan");
    fuzz_build().expect("fuzz build plan");
    deny().expect("dependency policy plan");
    downstream_smoke().expect("downstream smoke plan");
    repo_lint(std::iter::empty()).expect("repo-lint plan");
    repo_lint(["--strict".to_string()].into_iter()).expect("strict repo-lint plan");

    let log = recording.log();
    assert!(log.contains("fmt --all -- --check|"));
    assert!(log.contains("clippy --workspace --all-targets --all-features -- -D warnings|"));
    assert!(log.contains("clippy -p j2k-native -p j2k --all-targets --all-features --no-deps"));
    assert!(log.contains("-W clippy::pedantic -W clippy::nursery -D warnings"));
    assert!(log.contains("test --workspace --all-features --lib --bins --tests"));
    assert!(log.contains("--exclude fixture-package"));
    assert!(log.contains("nextest run --workspace --all-features --lib --bins --tests"));
    assert!(log.contains("doc --workspace --all-features --no-deps|RUSTDOCFLAGS=-D warnings"));
    assert!(
        log.contains("doc -p j2k-core --lib --no-deps|RUSTDOCFLAGS=-D warnings -D missing_docs")
    );
    assert!(log.contains("check --manifest-path crates/j2k/fuzz/Cargo.toml"));
    assert!(log.contains("check --manifest-path crates/j2k-transcode/fuzz/Cargo.toml"));
    assert!(log.contains("deny check licenses advisories bans sources"));
    assert!(log.contains("test -p j2k --examples"));
    assert!(log.contains("test -p j2k-transcode --examples"));
    assert!(log.contains("test -p xtask --test repo_lint -- --nocapture"));
    assert!(log.contains("test -p xtask --test repo_lint -- --nocapture --ignored"));
}

#[test]
fn repo_lint_unknown_argument_fails_before_any_command() {
    let error =
        repo_lint(["--unknown".to_string()].into_iter()).expect_err("unknown repo-lint argument");
    assert!(error.contains("unknown repo-lint argument"));
}
