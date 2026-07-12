// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeMap, ffi::OsStr};

use super::{
    activated_feature, panic_surface_clippy_args, parse_panic_surface_output,
    parse_panic_surface_selection, PanicSurfaceSelection,
};

fn package() -> serde_json::Value {
    serde_json::json!({
        "id": "codec-id",
        "name": "codec",
        "publish": null,
        "targets": [{"crate_types": ["lib"]}],
        "dependencies": [{"name": "dependency", "rename": "dep_alias"}],
        "features": {"default": [], "accelerated": ["dep_alias?/simd"]}
    })
}

fn metadata(package: &serde_json::Value) -> String {
    serde_json::json!({
        "workspace_members": ["codec-id"],
        "packages": [package],
    })
    .to_string()
}

#[test]
fn selection_metadata_boundaries_fail_closed_with_specific_context() {
    let cases = [
        (
            "not json".to_string(),
            "metadata for panic-surface gate is malformed",
        ),
        (
            serde_json::json!({"workspace_members": ["codec-id"]}).to_string(),
            "no `packages` array",
        ),
        (
            serde_json::json!({"packages": []}).to_string(),
            "no `workspace_members` array",
        ),
        (
            serde_json::json!({"packages": [], "workspace_members": []}).to_string(),
            "no workspace members",
        ),
        (
            serde_json::json!({"packages": [], "workspace_members": [7]}).to_string(),
            "non-string workspace member ID",
        ),
        (
            serde_json::json!({"packages": [], "workspace_members": ["missing"]}).to_string(),
            "omits workspace member package",
        ),
        (
            serde_json::json!({
                "packages": [package()],
                "workspace_members": ["codec-id", "codec-id"]
            })
            .to_string(),
            "duplicate workspace member",
        ),
    ];
    for (metadata, expected) in cases {
        let error = parse_panic_surface_selection(&metadata, &[])
            .expect_err("malformed selection metadata");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn package_publish_target_dependency_and_feature_shapes_fail_closed() {
    let mut cases = Vec::new();
    let mut value = package();
    value.as_object_mut().expect("package").remove("publish");
    cases.push((value, "has no `publish` field"));
    let mut value = package();
    value["publish"] = serde_json::json!(false);
    cases.push((value, "invalid `publish` value"));
    let mut value = package();
    value["publish"] = serde_json::json!([7]);
    cases.push((value, "non-string publish registry"));
    let mut value = package();
    value.as_object_mut().expect("package").remove("targets");
    cases.push((value, "has no `targets` array"));
    let mut value = package();
    value["targets"] = serde_json::json!([{}]);
    cases.push((value, "target without `crate_types`"));
    let mut value = package();
    value["targets"] = serde_json::json!([{"crate_types": [7]}]);
    cases.push((value, "non-string crate type"));
    let mut value = package();
    value
        .as_object_mut()
        .expect("package")
        .remove("dependencies");
    cases.push((value, "has no `dependencies` array"));
    let mut value = package();
    value["dependencies"] = serde_json::json!([{}]);
    cases.push((value, "dependency without a string `name`"));
    let mut value = package();
    value["dependencies"] = serde_json::json!([{"name": "dep", "rename": 7}]);
    cases.push((value, "has an invalid rename"));
    let mut value = package();
    value.as_object_mut().expect("package").remove("features");
    cases.push((value, "has no `features` object"));
    let mut value = package();
    value["features"] = serde_json::json!({"default": "not an array"});
    cases.push((value, "has a non-array definition"));
    let mut value = package();
    value["features"] = serde_json::json!({"default": [7]});
    cases.push((value, "contains a non-string entry"));

    for (package, expected) in cases {
        let error = parse_panic_surface_selection(&metadata(&package), &[])
            .expect_err("malformed package metadata");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn feature_activation_resolves_local_renamed_optional_and_dependency_only_forms() {
    let dependencies = BTreeMap::from([("alias".to_string(), "real-package".to_string())]);
    assert_eq!(
        activated_feature("codec", "default", "local", &dependencies).unwrap(),
        Some(("codec".to_string(), "local".to_string()))
    );
    assert_eq!(
        activated_feature("codec", "default", "alias?/simd", &dependencies).unwrap(),
        Some(("real-package".to_string(), "simd".to_string()))
    );
    assert_eq!(
        activated_feature("codec", "default", "dep:alias", &dependencies).unwrap(),
        None
    );
    let error = activated_feature("codec", "default", "missing/simd", &dependencies)
        .expect_err("unknown dependency activation");
    assert!(error.contains("references unknown dependency `missing`"));
}

#[test]
fn production_feature_cannot_reenable_registered_dev_only_dependency_feature() {
    let error = parse_panic_surface_selection(&metadata(&package()), &[("dependency", "simd")])
        .expect_err("production to dev-only feature edge");
    assert!(error.contains("enables dev-only feature"));
}

#[test]
fn clippy_plan_omits_empty_feature_flag_and_orders_packages() {
    let selection = PanicSurfaceSelection {
        packages: vec!["alpha".to_string(), "beta".to_string()],
        features: Vec::new(),
    };
    let args = panic_surface_clippy_args(&selection);
    assert!(!args.iter().any(|arg| arg == OsStr::new("--features")));
    assert!(args.windows(4).any(|window| {
        window
            == [
                OsStr::new("--package"),
                OsStr::new("alpha"),
                OsStr::new("--package"),
                OsStr::new("beta"),
            ]
    }));
}

#[test]
fn panic_output_parser_rejects_missing_reason_or_success_and_ignores_other_codes() {
    let error =
        parse_panic_surface_output("{}").expect_err("record without reason must be rejected");
    assert!(error.contains("has no string `reason`"));
    let error = parse_panic_surface_output(r#"{"reason":"build-finished"}"#)
        .expect_err("build finish without success must be rejected");
    assert!(error.contains("has no boolean `success`"));
    let output = concat!(
        r#"{"reason":"compiler-message","message":{"code":{"code":"clippy::other"}}}"#,
        "\n",
        r#"{"reason":"build-finished","success":true}"#,
    );
    assert_eq!(parse_panic_surface_output(output), Ok((0, 0)));
}
