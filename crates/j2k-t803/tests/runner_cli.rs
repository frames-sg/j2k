// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "runner")]

use std::{fs, path::PathBuf, process::Command};

fn empty_cache(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("j2k-t803-cli-{label}-{}", std::process::id()));
    if path.exists() {
        fs::remove_dir_all(&path).expect("remove stale test cache");
    }
    fs::create_dir(&path).expect("create test cache");
    path
}

#[test]
fn run_fails_closed_when_the_pinned_corpus_is_absent() {
    let cache = empty_cache("missing");
    let output = Command::new(env!("CARGO_BIN_EXE_j2k-t803-runner"))
        .args([
            "run",
            "--iut",
            "cpu",
            "--suite",
            "part15",
            "--development",
            "--cache-dir",
        ])
        .arg(&cache)
        .output()
        .expect("run T.803 CLI");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("archive is absent"),
        "unexpected stderr: {stderr}"
    );
    fs::remove_dir_all(cache).expect("remove test cache");
}

#[test]
fn run_rejects_an_unknown_suite_before_touching_the_corpus() {
    let output = Command::new(env!("CARGO_BIN_EXE_j2k-t803-runner"))
        .args(["run", "--iut", "cpu", "--suite", "jpeg"])
        .output()
        .expect("run T.803 CLI");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown T.803 suite \"jpeg\"; expected part1|part15|all"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn verify_requires_an_explicit_independent_evidence_scope() {
    let output = Command::new(env!("CARGO_BIN_EXE_j2k-t803-runner"))
        .args([
            "verify",
            "--candidate-sha",
            "0123456789abcdef0123456789abcdef01234567",
            "--report",
            "missing.json",
        ])
        .output()
        .expect("run T.803 verify CLI");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("requires --scope cpu|cuda|metal|all"),
        "unexpected stderr: {stderr}"
    );
}
