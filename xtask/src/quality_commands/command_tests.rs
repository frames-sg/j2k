// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    clippy, clippy_strict, doc, fmt, fuzz_build, repo_lint, test, test_workspace_without_benches,
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
    doc().expect("documentation plan");
    fuzz_build().expect("fuzz build plan");
    repo_lint(std::iter::empty()).expect("repo-lint plan");

    let log = recording.log();
    assert!(log.contains("fmt --all -- --check|"));
    assert!(
        log.contains("clippy --workspace --lib --all-features -- -D warnings|"),
        "production library targets must retain allocation-policy Clippy lints: {log}"
    );
    assert!(
        log.contains(
            "clippy --workspace --bins --examples --tests --benches --all-features -- \
             -D warnings -A clippy::disallowed_methods -A clippy::disallowed_macros|"
        ),
        "non-library targets must disable only the codec allocation-policy lints: {log}"
    );
    assert!(
        log.contains("clippy -p xtask --test repo_lint --all-features -- -D warnings|"),
        "the test=false repo-lint target needs an explicit Clippy pass: {log}"
    );
    assert!(log.contains(
        "clippy -p j2k-native -p j2k --lib --all-features --no-deps -- \
         -W clippy::pedantic -W clippy::nursery -D warnings"
    ));
    assert!(log.contains(
        "clippy -p j2k-native -p j2k --bins --examples --tests --benches --all-features \
         --no-deps -- -W clippy::pedantic -W clippy::nursery -D warnings"
    ));
    assert!(log.contains("-A clippy::disallowed_methods -A clippy::disallowed_macros|"));
    assert!(log.contains("test --workspace --all-features --lib --bins --tests"));
    assert!(log.contains("--exclude j2k-alloc-probe"));
    assert!(log.contains("test -p j2k-alloc-probe|"));
    assert!(log.contains("--exclude fixture-package"));
    assert!(log.contains("doc --workspace --all-features --no-deps|RUSTDOCFLAGS=-D warnings"));
    assert!(
        log.contains("doc -p j2k-core --lib --no-deps|RUSTDOCFLAGS=-D warnings -D missing_docs")
    );
    assert!(log.contains("check --manifest-path crates/j2k/fuzz/Cargo.toml"));
    assert!(log.contains("check --manifest-path crates/j2k-transcode/fuzz/Cargo.toml"));
    assert!(log.contains("test -p j2k --examples"));
    assert!(log.contains("test -p j2k-transcode --examples"));
    assert!(log.contains("test -p xtask --test repo_lint -- --nocapture"));
}

#[test]
fn repo_lint_rejects_retired_strict_and_unknown_arguments_before_any_command() {
    let recording = RecordingProgram::new("repo-lint-argument-test", "");
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    for argument in ["--strict", "--unknown"] {
        let error = repo_lint([argument.to_string()].into_iter())
            .expect_err("unsupported repo-lint argument");
        assert!(error.contains("unknown repo-lint argument"));
    }
    assert!(!recording.was_invoked());
}
