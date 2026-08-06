// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, process::Command};

#[test]
fn t803_delegates_to_the_fail_closed_runner() {
    let cache = std::env::temp_dir().join(format!("j2k-xtask-t803-{}", std::process::id()));
    if cache.exists() {
        fs::remove_dir_all(&cache).expect("remove stale test cache");
    }
    fs::create_dir(&cache).expect("create test cache");

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "t803",
            "run",
            "--iut",
            "cpu",
            "--development",
            "--cache-dir",
        ])
        .arg(&cache)
        .output()
        .expect("run xtask T.803 command");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("archive is absent"),
        "unexpected stderr: {stderr}"
    );
    fs::remove_dir_all(cache).expect("remove test cache");
}
