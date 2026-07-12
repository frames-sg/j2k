// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, ffi::OsString, path::Path};

use super::super::{
    baseline_api_snapshot, capture_command, current_api_snapshot, require_macos, semver,
    verify_baseline_tag, workspace_package_versions, SnapshotKind, CARGO_PUBLIC_API_VERSION,
    HIDDEN_API_SNAPSHOT, PUBLIC_API_SNAPSHOT,
};

fn workspace_path(relative: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest has workspace parent")
        .join(relative)
        .to_str()
        .expect("workspace path is UTF-8")
        .to_string()
}

#[test]
fn command_capture_preserves_success_failure_and_non_utf8_diagnostics() {
    assert_eq!(
        capture_command(
            OsString::from("/bin/sh"),
            &["-c", "printf '  success  \\n'"],
            &[],
            "capture success",
        ),
        Ok("success".to_string())
    );

    let error = capture_command(
        OsString::from("/bin/sh"),
        &["-c", "printf 'stdout'; printf 'stderr' >&2; exit 7"],
        &[],
        "capture failure",
    )
    .expect_err("nonzero command");
    assert!(error.contains("exit status: 7"));
    assert!(error.contains("stdout"));
    assert!(error.contains("stderr"));
    assert!(error.contains("capture failure"));

    let error = capture_command(
        OsString::from("/bin/sh"),
        &["-c", "printf '\\377'"],
        &[],
        "capture bytes",
    )
    .expect_err("non-UTF-8 stdout");
    assert!(error.contains("non-UTF-8 stdout"));
    assert!(error.contains("capture bytes"));
}

#[test]
fn committed_semver_inputs_match_the_pinned_workspace_contract() {
    verify_baseline_tag().expect("pinned baseline tag");
    let baseline = baseline_api_snapshot(CARGO_PUBLIC_API_VERSION).expect("baseline API snapshot");
    assert!(baseline.starts_with("# J2K 1.0 Public API Snapshot"));

    let ordinary = current_api_snapshot(
        &workspace_path(PUBLIC_API_SNAPSHOT),
        SnapshotKind::Ordinary,
        CARGO_PUBLIC_API_VERSION,
    )
    .expect("ordinary API snapshot");
    let hidden = current_api_snapshot(
        &workspace_path(HIDDEN_API_SNAPSHOT),
        SnapshotKind::Hidden,
        CARGO_PUBLIC_API_VERSION,
    )
    .expect("hidden API snapshot");
    assert!(ordinary.starts_with("# J2K 1.0 Public API Snapshot"));
    assert!(hidden.starts_with("# J2K 1.0 Rustdoc-Hidden Public API Snapshot"));

    let versions = workspace_package_versions().expect("workspace package versions");
    assert_eq!(versions.get("j2k").map(String::as_str), Some("0.7.0"));
    assert!(versions.keys().collect::<BTreeSet<_>>().len() > 10);
}

#[cfg(target_os = "macos")]
#[test]
fn semver_top_level_rejects_a_mismatched_collector_pin_before_live_inventory() {
    require_macos().expect("macOS semver precondition");
    let error =
        semver(std::iter::empty(), &["j2k"], "0.0.0").expect_err("mismatched cargo-public-api pin");
    assert!(error.contains("requested cargo-public-api 0.0.0"));
    assert!(error.contains(CARGO_PUBLIC_API_VERSION));
}
