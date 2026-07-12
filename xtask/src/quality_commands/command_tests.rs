// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use super::{
    clippy, clippy_strict, deny, doc, downstream_smoke, fmt, fuzz_build, nextest, repo_lint, test,
    test_workspace_without_benches,
};
use crate::command_support::use_test_cargo_program;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "j2k-quality-command-test-{}-{}",
        std::process::id(),
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).expect("create quality command test directory");
    path
}

fn recording_cargo(root: &Path) -> (PathBuf, PathBuf) {
    let program = root.join("cargo.sh");
    let log = root.join("cargo.log");
    fs::write(
        &program,
        format!(
            "#!/bin/sh\nprintf '%s|RUSTDOCFLAGS=%s|RUST_TEST_THREADS=%s\\n' \"$*\" \"${{RUSTDOCFLAGS-unset}}\" \"${{RUST_TEST_THREADS-unset}}\" >> '{}'\n",
            log.display()
        ),
    )
    .expect("write recording Cargo");
    let mut permissions = fs::metadata(&program)
        .expect("recording Cargo metadata")
        .permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&program, permissions).expect("make recording Cargo executable");
    (program, log)
}

#[test]
fn quality_command_plans_are_complete_and_never_launch_real_tools() {
    let root = temp_dir();
    let (program, log) = recording_cargo(&root);
    let _cargo = use_test_cargo_program(program.into_os_string());

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

    let log = fs::read_to_string(log).expect("Cargo command log");
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
