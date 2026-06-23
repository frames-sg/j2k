// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use j2k_test_support::{minimal_gray8_jpeg, minimal_jp2};

fn j2k_bin() -> &'static str {
    env!("CARGO_BIN_EXE_j2k")
}

fn run_j2k(args: &[&str]) -> Output {
    Command::new(j2k_bin())
        .args(args)
        .output()
        .expect("run j2k CLI")
}

fn write_temp_file(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("j2k-cli-tests-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create CLI test temp dir");
    let path = dir.join(name);
    fs::write(&path, bytes).expect("write CLI test input");
    path
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn inspect_cli_reports_jpeg_info() {
    let jpeg = minimal_gray8_jpeg();
    let input = write_temp_file("grayscale_8x8.jpg", &jpeg);

    let output = run_j2k(&["inspect", path_str(&input)]);

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    let stdout = stdout(&output);
    assert!(stdout.contains('8'));
    assert!(stdout.contains("Grayscale"));
    assert!(stdout.contains("bit=8"));
}

#[test]
fn inspect_cli_reports_jp2_info() {
    let input = write_temp_file("minimal.jp2", &minimal_jp2());

    let output = run_j2k(&["inspect", path_str(&input)]);

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    let stdout = stdout(&output);
    assert!(stdout.contains("128"));
    assert!(stdout.contains("64"));
    assert!(stdout.contains("levels=6"));
}

#[test]
fn inspect_cli_rejects_unknown_subcommand() {
    let output = run_j2k(&["unknown"]);

    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("unknown subcommand: unknown"));
}

#[test]
fn inspect_cli_reports_missing_file() {
    let missing = std::env::temp_dir()
        .join(format!("j2k-cli-tests-{}", std::process::id()))
        .join("missing.jpg");

    let output = run_j2k(&["inspect", path_str(&missing)]);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("error reading"));
}

#[test]
fn inspect_cli_reports_invalid_jpeg() {
    let input = write_temp_file("invalid.jpg", b"not a jpeg");

    let output = run_j2k(&["inspect", path_str(&input)]);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("error:"));
}

fn path_str(path: &Path) -> &str {
    path.to_str().expect("test path is UTF-8")
}
