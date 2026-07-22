// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{release_integrity, PUBLISHABLE_PACKAGES};
use crate::{command_support::use_test_cargo_program, test_command::RecordingProgram};

const WORKSPACE_CHILD_ENV: &str = "XTASK_TEST_RELEASE_INTEGRITY_WORKSPACE_CHILD";

fn run_test_from_workspace(test_name: &str, recording: &RecordingProgram) {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask workspace root");
    let output = std::process::Command::new(std::env::current_exe().expect("current test binary"))
        .arg(test_name)
        .arg("--exact")
        .arg("--nocapture")
        .current_dir(workspace)
        .env(WORKSPACE_CHILD_ENV, "1")
        .env("CARGO", recording.program())
        .output()
        .expect("run release-integrity test from workspace root");
    assert!(
        output.status.success(),
        "workspace child failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

pub(super) fn metadata_program(label: &str, metadata: &serde_json::Value) -> RecordingProgram {
    RecordingProgram::new(label, &format!("printf '%s\\n' '{metadata}'"))
}

pub(super) fn complete_publishable_metadata() -> serde_json::Value {
    let workspace_version = include_str!("../../../../Cargo.toml")
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("version")
                .and_then(|rest| rest.split('"').nth(1))
        })
        .expect("workspace package version");
    let packages = PUBLISHABLE_PACKAGES
        .iter()
        .map(|name| {
            let targets = if *name == "j2k-cli" {
                Vec::new()
            } else {
                vec![serde_json::json!({"kind": ["lib"]})]
            };
            serde_json::json!({
                "id": name,
                "name": name,
                "version": workspace_version,
                "publish": null,
                "readme": "README.md",
                "dependencies": [],
                "targets": targets,
                "metadata": {
                    "docs": {"rs": {"all-features": true, "targets": []}}
                },
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "workspace_members": PUBLISHABLE_PACKAGES,
        "packages": packages,
    })
}

#[test]
fn release_integrity_accepts_complete_hermetic_metadata_in_pre_candidate_mode() {
    let recording = metadata_program(
        "release-integrity-valid-metadata",
        &complete_publishable_metadata(),
    );
    if std::env::var_os(WORKSPACE_CHILD_ENV).is_some() {
        release_integrity(std::iter::empty()).expect("valid pre-candidate release integrity");
        return;
    }

    run_test_from_workspace(
        "release_commands::tests::integrity::release_integrity_accepts_complete_hermetic_metadata_in_pre_candidate_mode",
        &recording,
    );

    assert!(recording
        .log()
        .starts_with("metadata --locked --no-deps --format-version 1|"));
}

#[test]
fn release_integrity_aggregates_invalid_package_metadata_without_publishing() {
    let metadata = serde_json::json!({
        "workspace_members": ["core", "internal", "outsider", "native"],
        "packages": [
            {
                "id": "core",
                "name": "j2k-core",
                "publish": [],
            },
            {
                "id": "internal",
                "name": "internal",
                "publish": [],
            },
            {
                "id": "outsider",
                "name": "outsider",
                "publish": null,
            },
            {
                "id": "native",
                "name": "j2k-native",
                "publish": null,
                "version": "9.9.9",
                "readme": null,
                "dependencies": [],
                "targets": [],
                "metadata": {},
            },
        ],
    });
    let recording = metadata_program("release-integrity-invalid-metadata", &metadata);
    if std::env::var_os(WORKSPACE_CHILD_ENV).is_none() {
        run_test_from_workspace(
            "release_commands::tests::integrity::release_integrity_aggregates_invalid_package_metadata_without_publishing",
            &recording,
        );
        return;
    }

    let error = release_integrity(std::iter::empty())
        .expect_err("invalid package metadata must fail release integrity");

    for expected in [
        "`j2k-core` is listed as publishable but has `publish = false`",
        "`outsider` is neither in PUBLISHABLE_PACKAGES nor explicitly `publish = false`",
        "`j2k-native` version 9.9.9 does not match workspace version",
        "`j2k-native` is publishable but has no package README",
        "`j2k-native` is publishable but missing [package.metadata.docs.rs]",
        "`j2k-native` is publishable but has no library target",
        "`j2k-profile` is listed in PUBLISHABLE_PACKAGES but is not a workspace member",
    ] {
        assert!(error.contains(expected), "missing `{expected}` in {error}");
    }
}

#[test]
fn release_integrity_rejects_non_json_cargo_metadata() {
    let recording = RecordingProgram::new(
        "release-integrity-non-json-metadata",
        "printf '%s\\n' 'not-json'",
    );
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    let error = release_integrity(std::iter::empty())
        .expect_err("non-JSON cargo metadata must fail release integrity");

    assert!(error.contains("failed to parse cargo metadata"));
}

#[test]
fn release_integrity_rejects_publishable_packages_without_versions() {
    let mut metadata = complete_publishable_metadata();
    metadata["packages"][0]
        .as_object_mut()
        .expect("package record")
        .remove("version");
    let recording = metadata_program("release-integrity-missing-version", &metadata);
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    let error = release_integrity(std::iter::empty())
        .expect_err("publishable package without a version must reject");

    assert!(
        error.contains("cargo metadata package `j2k-core` has no string version"),
        "unexpected error: {error}"
    );
}
