// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(unix)]

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct Harness {
    root: PathBuf,
    cargo: PathBuf,
    log: PathBuf,
}

impl Harness {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "j2k-xtask-orchestration-{}-{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).expect("create orchestration test directory");
        let cargo = root.join("cargo.sh");
        let log = root.join("cargo.log");
        let real_cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
        fs::write(
            &cargo,
            format!(
                "#!/bin/sh\nprintf '%s|RUSTDOCFLAGS=%s|RUST_TEST_THREADS=%s\\n' \"$*\" \"${{RUSTDOCFLAGS-unset}}\" \"${{RUST_TEST_THREADS-unset}}\" >> '{}'\nif [ \"$1\" = metadata ]; then exec \"{}\" \"$@\"; fi\nif [ \"$1\" = clippy ]; then printf '%s\\n' '{{\"reason\":\"build-finished\",\"success\":true}}'; exit 0; fi\nif [ \"$1\" = test ]; then printf 'test result: ok. 100 passed; 0 failed;\\n'; exit 0; fi\nif [ \"$1\" = -V ]; then printf 'cargo 1.96.0\\n'; fi\n",
                log.display(),
                real_cargo
            ),
        )
        .expect("write fake Cargo");
        let mut permissions = fs::metadata(&cargo)
            .expect("fake Cargo metadata")
            .permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&cargo, permissions).expect("make fake Cargo executable");
        Self { root, cargo, log }
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(args)
            .current_dir(workspace_root())
            .env("CARGO", &self.cargo)
            .output()
            .expect("run xtask child process")
    }

    fn log(&self) -> String {
        fs::read_to_string(&self.log).expect("read fake Cargo log")
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("remove orchestration test directory");
    }
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest has workspace parent")
}

fn assert_success(output: &Output, task: &str) {
    assert!(
        output.status.success(),
        "{task} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn release_critical_orchestrators_run_from_the_workspace_without_real_cargo() {
    let harness = Harness::new();

    assert_success(&harness.run(&["ci"]), "ci");
    assert_success(&harness.run(&["bench-build"]), "bench-build");
    assert_success(&harness.run(&["j2k-bench-signoff"]), "j2k-bench-signoff");
    assert_success(&harness.run(&["release-cpu"]), "release-cpu");
    assert_success(&harness.run(&["release-integrity"]), "release-integrity");

    let report = harness.root.join("benchmark-report.md");
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

    let log = harness.log();
    assert!(log.contains("fmt --all -- --check|"));
    assert!(log.contains("bench -p j2k --bench public_api --no-run|"));
    assert!(log.contains("test -p j2k-compare --test in_process_parity -- --nocapture|"));
    assert!(log.contains("test --release -p j2k-core"));
}
