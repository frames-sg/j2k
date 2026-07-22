// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use super::{
    has_docs_rs_metadata, has_lib_target, package_name, publish_false, release_cpu,
    release_integrity, validate_package_gate_partition, validate_publish_script_source,
    validate_publish_workflow_source, validate_release_docs_source,
    validate_unpublished_dependencies, workspace_package_records,
};

#[cfg(unix)]
mod file_boundaries;
#[cfg(unix)]
mod integrity;
#[cfg(unix)]
mod orchestration;
#[cfg(unix)]
mod path_patches;
mod validation;

#[cfg(unix)]
use crate::{command_support::use_test_cargo_program, test_command::RecordingProgram};

#[test]
fn release_metadata_shape_helpers_fail_closed() {
    let valid = serde_json::json!({
        "name": "j2k-core",
        "publish": [],
        "targets": [{"kind": ["lib", "rlib"]}],
        "metadata": {"docs": {"rs": {"all-features": true, "targets": []}}},
    });
    assert_eq!(package_name(&valid), Ok("j2k-core"));
    assert!(publish_false(&valid));
    assert!(has_lib_target(&valid));
    assert!(has_docs_rs_metadata(&valid));

    assert!(package_name(&serde_json::json!({})).is_err());
    assert!(!publish_false(&serde_json::json!({"publish": false})));
    assert!(!has_lib_target(
        &serde_json::json!({"targets": [{"kind": "lib"}]})
    ));
    assert!(!has_docs_rs_metadata(&serde_json::json!({
        "metadata": {"docs": {"rs": {"all-features": false, "targets": []}}}
    })));
    assert!(!has_docs_rs_metadata(&serde_json::json!({
        "metadata": {"docs": {"rs": {"all-features": true, "targets": ["x86_64"]}}}
    })));
}

#[test]
fn workspace_package_records_preserve_member_order_and_reject_malformed_graphs() {
    let valid = serde_json::json!({
        "packages": [
            {"id": "a", "name": "alpha"},
            {"id": "b", "name": "beta"},
        ],
        "workspace_members": ["b", "a"],
    });
    let packages = workspace_package_records(&valid).expect("valid workspace metadata");
    assert_eq!(package_name(packages[0]), Ok("beta"));
    assert_eq!(package_name(packages[1]), Ok("alpha"));

    let malformed = [
        (
            serde_json::json!({"workspace_members": []}),
            "packages array",
        ),
        (
            serde_json::json!({"packages": [], "workspace_members": {}}),
            "workspace_members array",
        ),
        (
            serde_json::json!({"packages": [{}], "workspace_members": []}),
            "packages[0] has no string id",
        ),
        (
            serde_json::json!({
                "packages": [{"id": "a"}, {"id": "a"}],
                "workspace_members": ["a"],
            }),
            "duplicate package id `a`",
        ),
        (
            serde_json::json!({"packages": [], "workspace_members": [1]}),
            "workspace_members[0] is not a string",
        ),
        (
            serde_json::json!({"packages": [{"id": "a"}], "workspace_members": ["a", "a"]}),
            "workspace_members contains duplicate id `a`",
        ),
        (
            serde_json::json!({"packages": [], "workspace_members": ["missing"]}),
            "workspace member `missing` has no package record",
        ),
    ];
    for (metadata, expected) in malformed {
        let error = workspace_package_records(&metadata).expect_err("malformed metadata rejects");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn unpublished_dependency_policy_distinguishes_publishable_dev_and_runtime_edges() {
    let unpublished = BTreeSet::from(["internal"]);
    let package = serde_json::json!({
        "dependencies": [
            {"name": "published", "kind": null, "req": "1"},
            {"name": "internal", "kind": null, "req": "*"},
            {"name": "internal", "kind": "build", "req": "*"},
            {"name": "internal", "kind": "dev", "req": "1"},
            {"name": "internal", "kind": "dev", "req": "*"},
        ]
    });
    let mut errors = Vec::new();
    validate_unpublished_dependencies("consumer", &package, &unpublished, &mut errors)
        .expect("well-shaped dependencies");
    assert_eq!(errors.len(), 3);
    assert!(errors[0].contains("normal dependency on unpublished crate `internal`"));
    assert!(errors[1].contains("build dependency on unpublished crate `internal`"));
    assert!(errors[2].contains("versioned dev-dependency `internal = \"1\"`"));

    let malformed = [
        (serde_json::json!({}), "has no dependencies array"),
        (
            serde_json::json!({"dependencies": [{}]}),
            "dependency[0] has no string name",
        ),
        (
            serde_json::json!({"dependencies": [{"name": "internal", "kind": 4, "req": "*"}]}),
            "has invalid kind",
        ),
        (
            serde_json::json!({"dependencies": [{"name": "internal", "kind": "dev"}]}),
            "has no string requirement",
        ),
    ];
    for (package, expected) in malformed {
        let error =
            validate_unpublished_dependencies("consumer", &package, &unpublished, &mut Vec::new())
                .expect_err("malformed dependency metadata rejects");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn checked_in_publish_workflow_script_docs_and_partitions_agree() {
    let mut errors = Vec::new();
    validate_package_gate_partition(&mut errors);
    validate_publish_workflow_source(
        include_str!("../../../.github/workflows/publish.yml"),
        &mut errors,
    )
    .expect("parse publish workflow");
    validate_publish_script_source(
        include_str!("../../../scripts/publish-crate.sh"),
        &mut errors,
    );
    validate_release_docs_source(include_str!("../../../docs/release.md"), &mut errors);

    assert!(errors.is_empty(), "release contract drift: {errors:#?}");
}

#[test]
fn publish_workflow_rejects_checkout_that_can_peel_an_annotated_tag() {
    let workflow = include_str!("../../../.github/workflows/publish.yml").replacen(
        "          ref: ${{ github.ref }}\n",
        "",
        1,
    );
    let mut errors = Vec::new();

    validate_publish_workflow_source(&workflow, &mut errors).expect("parse publish workflow");

    assert!(errors
        .iter()
        .any(|error| error.contains("bind all 2 checkout steps to the triggering ref")));
}

#[cfg(unix)]
#[test]
fn release_cpu_executes_the_complete_fake_cargo_plan() {
    let recording = RecordingProgram::new("release-command-test", "");
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    release_cpu().expect("release CPU plan");

    let log = recording.log();
    assert!(log.starts_with("test --release -p j2k-core -p j2k-codec-math"));
    assert!(log.contains("-p j2k-native -p j2k -p j2k-tilecodec -p j2k-cli|"));
    assert_eq!(log.lines().count(), 1);
}

#[test]
fn release_integrity_rejects_invalid_modes_before_external_work() {
    let error = release_integrity(["--unknown".to_string()].into_iter())
        .expect_err("unknown release-integrity argument");
    assert!(error.contains("unknown release-integrity argument"));
}
