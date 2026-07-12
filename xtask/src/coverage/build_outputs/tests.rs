// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::{BuildOutputEvidence, CurrentBuildTarget};

static NEXT_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

struct TestTargetDir {
    path: PathBuf,
}

impl TestTargetDir {
    fn new() -> Self {
        let id = NEXT_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "j2k-coverage-build-output-{}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create build-output test directory");
        Self { path }
    }

    fn output(&self, scope: &str, content: &str) {
        write_output(&self.path, scope, content);
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestTargetDir {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            eprintln!(
                "failed to remove build-output test directory {}: {error}",
                self.path.display()
            );
        }
    }
}

fn selected(package: &str) -> BTreeSet<String> {
    BTreeSet::from([package.to_string()])
}

fn write_output(target: &Path, scope: &str, content: &str) {
    let path = target.join(scope).join("output");
    fs::create_dir_all(path.parent().expect("build output parent"))
        .expect("create build output scope");
    fs::write(path, content).expect("write build output");
}

fn current_target(base: &TestTargetDir) -> CurrentBuildTarget {
    CurrentBuildTarget::create_in_base(base.path()).expect("create current build target")
}

#[test]
fn identical_rerun_output_is_current_build_evidence() {
    let base = TestTargetDir::new();
    let scope = "debug/build/j2k-jpeg-0123456789abcdef";
    let content = "cargo::rustc-cfg=simd_fast\n";
    base.output(scope, content);
    let current = current_target(&base);
    let current_path = current.path().unwrap().to_path_buf();
    write_output(current.path().unwrap(), scope, content);
    let evidence = BuildOutputEvidence::capture(current).unwrap();
    let packages = selected("j2k-jpeg");

    assert!(!current_path.exists());
    assert!(evidence.current_cfg_flags(&packages, &packages).unwrap()["j2k-jpeg"]["simd_fast"]);
}

#[test]
fn stale_scope_output_is_outside_current_build_provenance() {
    let base = TestTargetDir::new();
    base.output(
        "debug/build/j2k-jpeg-0123456789abcdef",
        "cargo::rustc-check-cfg=cfg(simd_fast)\n",
    );
    let current = current_target(&base);
    write_output(
        current.path().unwrap(),
        "release/build/j2k-jpeg-fedcba9876543210",
        "cargo::rustc-check-cfg=cfg(simd_fast)\ncargo::rustc-cfg=simd_fast\n",
    );
    let evidence = BuildOutputEvidence::capture(current).unwrap();
    let packages = selected("j2k-jpeg");

    assert!(evidence.current_cfg_flags(&packages, &packages).unwrap()["j2k-jpeg"]["simd_fast"]);
}

#[test]
fn current_output_cfg_values_are_parsed() {
    let base = TestTargetDir::new();
    let current = current_target(&base);
    let scope = "debug/build/j2k-jpeg-0123456789abcdef";
    write_output(
        current.path().unwrap(),
        scope,
        "cargo::rustc-cfg=simd_fast\n",
    );
    let evidence = BuildOutputEvidence::capture(current).unwrap();
    let packages = selected("j2k-jpeg");

    assert!(evidence.current_cfg_flags(&packages, &packages).unwrap()["j2k-jpeg"]["simd_fast"]);
}

#[test]
fn target_triple_current_output_is_cfg_evidence() {
    let base = TestTargetDir::new();
    let current = current_target(&base);
    write_output(
        current.path().unwrap(),
        "aarch64-apple-darwin/debug/build/j2k-jpeg-fedcba9876543210",
        "cargo::rustc-check-cfg=cfg(neon)\ncargo::rustc-cfg=neon\n",
    );
    let evidence = BuildOutputEvidence::capture(current).unwrap();
    let packages = selected("j2k-jpeg");

    assert!(evidence.current_cfg_flags(&packages, &packages).unwrap()["j2k-jpeg"]["neon"]);
}

#[test]
fn conflicting_current_scopes_fail_closed() {
    let base = TestTargetDir::new();
    let current = current_target(&base);
    write_output(
        current.path().unwrap(),
        "debug/build/j2k-jpeg-0123456789abcdef",
        "cargo::rustc-check-cfg=cfg(simd_fast)\ncargo::rustc-cfg=simd_fast\n",
    );
    write_output(
        current.path().unwrap(),
        "release/build/j2k-jpeg-fedcba9876543210",
        "cargo::rustc-check-cfg=cfg(simd_fast)\n",
    );
    let evidence = BuildOutputEvidence::capture(current).unwrap();
    let packages = selected("j2k-jpeg");

    let error = evidence
        .current_cfg_flags(&packages, &packages)
        .unwrap_err();
    assert!(error.contains("conflict"), "{error}");
}

#[test]
fn hyphenated_package_name_matches_its_full_build_scope() {
    let base = TestTargetDir::new();
    let current = current_target(&base);
    write_output(
        current.path().unwrap(),
        "debug/build/j2k-cuda-runtime-0123456789abcdef",
        "cargo::rustc-cfg=cuda_runtime\n",
    );
    let evidence = BuildOutputEvidence::capture(current).unwrap();
    let packages = BTreeSet::from(["j2k".to_string(), "j2k-cuda-runtime".to_string()]);
    let build_scripts = selected("j2k-cuda-runtime");

    let flags = evidence
        .current_cfg_flags(&packages, &build_scripts)
        .unwrap();
    assert!(!flags.contains_key("j2k"));
    assert!(flags["j2k-cuda-runtime"]["cuda_runtime"]);
}

#[test]
fn missing_selected_package_build_output_fails_closed() {
    let base = TestTargetDir::new();
    let evidence = BuildOutputEvidence::capture(current_target(&base)).unwrap();
    let packages = selected("j2k-jpeg");

    let error = evidence
        .current_cfg_flags(&packages, &packages)
        .unwrap_err();
    assert!(error.contains("no build-script output"), "{error}");
}

#[test]
fn selected_package_without_build_script_needs_no_output() {
    let base = TestTargetDir::new();
    let evidence = BuildOutputEvidence::capture(current_target(&base)).unwrap();

    assert!(evidence
        .current_cfg_flags(&selected("j2k-core"), &BTreeSet::new())
        .unwrap()
        .is_empty());
}
