// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use super::super::release_manifest::{
    registry_independent_packages, validate_release_manifest_contract,
};
use super::super::{
    parse_release_manifest_source, validate_publish_script_source,
    validate_publish_workflow_source, validate_python_release_manifest_source,
    validate_release_docs_source, validate_unpublished_dependencies,
};

#[test]
fn release_manifest_parser_requires_schema_two_tiered_crate_records() {
    let manifest = parse_release_manifest_source(
        r#"{
            "schema": 2,
            "crates": [
                {"name": "stable-api", "api_contract": "stable"},
                {"name": "experimental-api", "api_contract": "experimental"},
                {"name": "implementation-spi", "api_contract": "implementation"},
                {"name": "cli", "api_contract": "binary"}
            ]
        }"#,
    )
    .expect("valid release manifest");
    assert_eq!(
        manifest.ordered_crates().collect::<Vec<_>>(),
        [
            "stable-api",
            "experimental-api",
            "implementation-spi",
            "cli"
        ]
    );
    assert_eq!(
        manifest
            .api_contracts()
            .map(|(name, contract)| (name, contract.as_str()))
            .collect::<Vec<_>>(),
        [
            ("stable-api", "stable"),
            ("experimental-api", "experimental"),
            ("implementation-spi", "implementation"),
            ("cli", "binary"),
        ]
    );
    assert_eq!(
        manifest.library_packages().collect::<Vec<_>>(),
        ["stable-api", "experimental-api", "implementation-spi"]
    );
    assert_eq!(
        manifest
            .api_contracts()
            .filter_map(|(name, contract)| (contract.as_str() == "stable").then_some(name))
            .collect::<Vec<_>>(),
        ["stable-api"]
    );

    for (source, expected) in [
        (
            r#"{"schema":1,"crates":[{"name":"base","api_contract":"stable"}]}"#,
            "schema must be exactly 2",
        ),
        (
            r#"{"schema":2.0,"crates":[{"name":"base","api_contract":"stable"}]}"#,
            "invalid type",
        ),
        (
            r#"{"schema":true,"crates":[{"name":"base","api_contract":"stable"}]}"#,
            "invalid type",
        ),
        (
            r#"{"schema":2,"schema":2,"crates":[{"name":"base","api_contract":"stable"}]}"#,
            "duplicate field",
        ),
        (
            r#"{"schema":2,"crates":[{"name":"base","api_contract":"stable"}],"extra":true}"#,
            "unknown field",
        ),
        (
            r#"{"schema":2,"crates":[{"name":"base","api_contract":"stable"},{"name":"base","api_contract":"experimental"}]}"#,
            "duplicate",
        ),
        (
            r#"{"schema":2,"crates":[{"name":"base","api_contract":"unsupported"}]}"#,
            "api_contract",
        ),
        (
            r#"{"schema":2,"crates":[{"name":"bad name","api_contract":"stable"}]}"#,
            "malformed crate name",
        ),
        (
            r#"{"schema":2,"crates":[{"name":"base","api_contract":"stable","extra":true}]}"#,
            "unknown field",
        ),
        (r#"{"schema":2,"crates":[]}"#, "must not be empty"),
    ] {
        let error = parse_release_manifest_source(source).expect_err("invalid manifest rejects");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

fn release_package(
    name: &str,
    target_kind: &str,
    dependencies: &serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "id": name,
        "name": name,
        "version": "0.7.6",
        "publish": null,
        "manifest_path": format!("/workspace/{name}/Cargo.toml"),
        "targets": [{"kind": [target_kind]}],
        "dependencies": dependencies,
    })
}

#[test]
fn release_manifest_validation_derives_order_and_registry_independence() {
    let manifest = parse_release_manifest_source(
        r#"{
            "schema": 2,
            "crates": [
                {"name": "base", "api_contract": "stable"},
                {"name": "consumer", "api_contract": "experimental"},
                {"name": "cli", "api_contract": "binary"}
            ]
        }"#,
    )
    .expect("valid release manifest");
    let packages = [
        release_package("base", "lib", &serde_json::json!([])),
        release_package(
            "consumer",
            "lib",
            &serde_json::json!([{
                "name": "base",
                "kind": "build",
                "optional": true,
                "target": "cfg(unix)",
                "path": "/workspace/base",
                "req": "=0.7.6",
                "source": null
            }]),
        ),
        release_package(
            "cli",
            "bin",
            &serde_json::json!([{
                "name": "consumer",
                "kind": null,
                "path": "/workspace/consumer",
                "req": "=0.7.6",
                "source": null
            }]),
        ),
    ];
    let records = packages.iter().collect::<Vec<_>>();
    let mut errors = Vec::new();

    validate_release_manifest_contract(&manifest, &records, &mut errors)
        .expect("well-formed release metadata");

    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    assert_eq!(
        registry_independent_packages(&manifest, &records).expect("derived registry independence"),
        BTreeSet::from(["base".to_string()])
    );
}

#[test]
fn release_manifest_validation_rejects_non_workspace_and_inexact_sibling_edges() {
    let manifest = parse_release_manifest_source(
        r#"{
            "schema": 2,
            "crates": [
                {"name": "base", "api_contract": "stable"},
                {"name": "consumer", "api_contract": "experimental"}
            ]
        }"#,
    )
    .expect("valid release manifest");
    let mut packages = [
        release_package("base", "lib", &serde_json::json!([])),
        release_package(
            "consumer",
            "lib",
            &serde_json::json!([{
                "name": "base",
                "kind": null,
                "path": "/workspace/base",
                "req": "^0.7.6",
                "source": null
            }]),
        ),
    ];
    let records = packages.iter().collect::<Vec<_>>();
    let mut errors = Vec::new();

    validate_release_manifest_contract(&manifest, &records, &mut errors)
        .expect("structurally valid metadata");

    assert!(errors.iter().any(|error| error.contains("exact release")));

    packages[1]["dependencies"][0]["req"] = serde_json::json!("=0.7.6");
    packages[1]["dependencies"][0]["path"] = serde_json::Value::Null;
    packages[1]["dependencies"][0]["source"] =
        serde_json::json!("registry+https://github.com/rust-lang/crates.io-index");
    let records = packages.iter().collect::<Vec<_>>();
    let error = validate_release_manifest_contract(&manifest, &records, &mut Vec::new())
        .expect_err("same-name registry dependency must reject");
    assert!(
        error.contains("workspace/path sourced"),
        "unexpected: {error}"
    );
}

#[test]
fn release_manifest_validation_requires_one_release_version() {
    let manifest = parse_release_manifest_source(
        r#"{
            "schema": 2,
            "crates": [
                {"name": "base", "api_contract": "stable"},
                {"name": "consumer", "api_contract": "experimental"}
            ]
        }"#,
    )
    .expect("valid release manifest");
    let packages = [release_package("base", "lib", &serde_json::json!([])), {
        let mut package = release_package("consumer", "lib", &serde_json::json!([]));
        package["version"] = serde_json::json!("0.8.0");
        package
    }];
    let records = packages.iter().collect::<Vec<_>>();
    let mut errors = Vec::new();

    validate_release_manifest_contract(&manifest, &records, &mut errors)
        .expect("structurally valid metadata");

    assert!(
        errors
            .iter()
            .any(|error| error.contains("must share one release version")),
        "unexpected: {errors:?}"
    );
}

#[test]
fn publish_workflow_validation_reports_parse_and_release_contract_failures() {
    let parse_error = validate_publish_workflow_source("jobs: [", &mut Vec::new())
        .expect_err("malformed workflow YAML must reject");
    assert!(parse_error.contains("failed to parse .github/workflows/publish.yml"));

    let workflow = "jobs:\n  unexpected:\n    runs-on: ubuntu-latest\n";
    let mut errors = Vec::new();
    validate_publish_workflow_source(workflow, &mut errors).expect("valid workflow YAML");

    assert!(errors
        .iter()
        .any(|error| error.contains("exactly preflight and publish jobs")));
    assert!(errors.iter().any(|error| error.contains(
        "does not enforce publication preflight `python3 scripts/publish_release.py publish`"
    )));
}

#[test]
fn publish_script_validation_fails_closed_for_missing_and_drifted_contracts() {
    let mut errors = Vec::new();
    validate_publish_script_source("cargo info\n", &mut errors);

    assert!(errors
        .iter()
        .any(|error| error.contains("--field ordered-crates")));
    assert!(errors.iter().any(|error| error
        .contains("does not enforce publish-script check `scripts/crates_io_version.py state`")));
    assert!(errors
        .iter()
        .any(|error| error.contains("must not treat ambiguous cargo-info failures")));
}

#[test]
fn python_release_manifest_validation_fails_closed_for_missing_graph_contracts() {
    let mut errors = Vec::new();

    validate_python_release_manifest_source("", &mut errors);

    assert!(errors
        .iter()
        .any(|error| error.contains("object_pairs_hook=_reject_duplicate_fields")));
    assert!(errors
        .iter()
        .any(|error| error.contains("must be workspace/path sourced")));
    assert!(errors
        .iter()
        .any(|error| error.contains("expected_requirement = f\"=")));
    assert!(errors
        .iter()
        .any(|error| error.contains("if kind == \"dev\"")));
}

#[test]
fn release_docs_validation_reports_missing_packages_and_operational_guards() {
    let mut errors = Vec::new();

    validate_release_docs_source("", &["j2k-core"], &mut errors);

    assert!(errors
        .iter()
        .any(|error| error.contains("does not document publishable crate `j2k-core`")));
    assert!(
        errors
            .iter()
            .any(|error| error
                .contains("does not document `cargo xtask release-integrity --publish`"))
    );
    assert!(errors
        .iter()
        .any(|error| error.contains("does not document `Only an exact HTTP 404`")));
}

#[test]
fn unpublished_dependency_validation_skips_external_edges_and_accepts_path_only_dev_edges() {
    let unpublished = BTreeSet::from(["internal"]);
    let package = serde_json::json!({
        "dependencies": [
            {"name": "external"},
            {"name": "internal", "kind": "dev", "req": "*"},
        ],
    });
    let mut errors = Vec::new();

    validate_unpublished_dependencies("consumer", &package, &unpublished, &mut errors)
        .expect("valid dependency records");

    assert!(errors.is_empty());
}
