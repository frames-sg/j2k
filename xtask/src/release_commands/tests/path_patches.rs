// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    validate_release_metadata, workspace_path_patch_provenance_paths, ReleaseIntegrityMode,
};
use crate::test_command::RecordingProgram;

const PATCH_APPROVAL_CHILD_ENV: &str = "XTASK_TEST_PATCH_APPROVAL_WORKSPACE_CHILD";

#[test]
fn path_patch_provenance_discovery_rejects_repository_escape() {
    let error = workspace_path_patch_provenance_paths(
        "[patch.crates-io]\noutside = { path = \"../outside\" }\n",
    )
    .unwrap_err();
    assert!(error.contains("repository-relative"), "{error}");
}

#[test]
fn release_integrity_publish_mode_requires_approval_for_every_workspace_path_patch() {
    if std::env::var_os(PATCH_APPROVAL_CHILD_ENV).is_some() {
        let mut errors = Vec::new();
        validate_release_metadata("0.7.4", ReleaseIntegrityMode::Publish, &mut errors)
            .expect("read hermetic release metadata");
        assert_eq!(errors.len(), 1, "unexpected publish errors: {errors:?}");
        assert!(
            errors[0].contains("third_party/second-patched/PATCH_PROVENANCE.md"),
            "publish rejection must identify the unapproved patch: {errors:?}"
        );
        return;
    }

    let fixture = RecordingProgram::new("release-path-patch-approval", "");
    let release_root = fixture
        .program()
        .parent()
        .expect("recording program parent")
        .join("release-root");
    for directory in ["third_party/first-patched", "third_party/second-patched"] {
        std::fs::create_dir_all(release_root.join(directory))
            .expect("create hermetic patch directory");
    }
    std::fs::write(
        release_root.join("Cargo.toml"),
        "[workspace.package]\nversion = \"0.7.4\"\n\n[patch.crates-io]\nfirst = { path = \"third_party/first-patched\" }\nsecond = { path = \"third_party/second-patched\" }\n",
    )
    .expect("write workspace path patches");
    std::fs::write(
        release_root.join("CHANGELOG.md"),
        "# Changelog\n\n## [0.7.4] - 2026-07-16\n",
    )
    .expect("write finalized changelog fixture");
    std::fs::write(
        release_root.join("third_party/first-patched/PATCH_PROVENANCE.md"),
        "## Release approval\n\n- Reviewer identity: `maintainer-one`\n- Approval date: `2026-07-20`\n",
    )
    .expect("write approved patch provenance");
    std::fs::write(
        release_root.join("third_party/second-patched/PATCH_PROVENANCE.md"),
        "## Release approval\n\n- Status: pending maintainer review\n",
    )
    .expect("write pending patch provenance");

    let output = std::process::Command::new(std::env::current_exe().expect("current test binary"))
        .arg("release_commands::tests::path_patches::release_integrity_publish_mode_requires_approval_for_every_workspace_path_patch")
        .arg("--exact")
        .arg("--nocapture")
        .current_dir(&release_root)
        .env(PATCH_APPROVAL_CHILD_ENV, "1")
        .output()
        .expect("run path-patch approval child");
    assert!(
        output.status.success(),
        "path-patch child failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
