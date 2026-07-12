// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::release_commands::{validate_unpublished_dependencies, workspace_package_records};
use std::collections::BTreeSet;

#[test]
fn workspace_package_records_fail_closed_on_malformed_metadata() {
    let valid = serde_json::json!({
        "packages": [{"id": "outside"}, {"id": "member-b"}, {"id": "member-a"}],
        "workspace_members": ["member-a", "member-b"],
    });
    let records = workspace_package_records(&valid).expect("valid workspace package records");
    assert_eq!(
        records
            .iter()
            .map(|record| record["id"].as_str().expect("record id"))
            .collect::<Vec<_>>(),
        ["member-a", "member-b"]
    );

    for (metadata, expected) in [
        (
            serde_json::json!({"packages": [{"id": "member"}], "workspace_members": [7]}),
            "workspace_members[0] is not a string",
        ),
        (
            serde_json::json!({"packages": [{"id": "member"}], "workspace_members": ["member", "member"]}),
            "duplicate id `member`",
        ),
        (
            serde_json::json!({"packages": [{"id": "other"}], "workspace_members": ["member"]}),
            "workspace member `member` has no package record",
        ),
        (
            serde_json::json!({"packages": [{}], "workspace_members": []}),
            "packages[0] has no string id",
        ),
        (
            serde_json::json!({"packages": [{"id": "member"}, {"id": "member"}], "workspace_members": ["member"]}),
            "duplicate package id `member`",
        ),
    ] {
        let error = workspace_package_records(&metadata).expect_err("malformed metadata rejected");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn unpublished_dependency_metadata_cannot_silently_disappear() {
    let unpublished = BTreeSet::from(["private"]);
    let mut errors = Vec::new();

    for (package, expected) in [
        (serde_json::json!({}), "has no dependencies array"),
        (
            serde_json::json!({"dependencies": [{}]}),
            "dependency[0] has no string name",
        ),
        (
            serde_json::json!({"dependencies": [{"name": "private", "kind": 7, "req": "*"}]}),
            "has invalid kind",
        ),
        (
            serde_json::json!({"dependencies": [{"name": "private", "kind": "dev"}]}),
            "has no string requirement",
        ),
    ] {
        let error =
            validate_unpublished_dependencies("publishable", &package, &unpublished, &mut errors)
                .expect_err("malformed dependency metadata rejected");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
    assert!(errors.is_empty());
}
