// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use super::super::{
    parse_release_manifest_source, validate_publish_script_source,
    validate_publish_workflow_source, validate_release_docs_source,
    validate_unpublished_dependencies,
};

#[test]
fn release_manifest_parser_requires_unique_ordered_crates_and_prefix_partition() {
    let manifest = parse_release_manifest_source(
        r#"{"schema":1,"ordered_crates":["base","consumer"],"registry_independent":["base"]}"#,
    )
    .expect("valid release manifest");
    assert_eq!(manifest.ordered_crates, ["base", "consumer"]);
    assert_eq!(manifest.registry_independent, ["base"]);

    for (source, expected) in [
        (
            r#"{"schema":1,"ordered_crates":["base","base"],"registry_independent":["base"]}"#,
            "duplicate",
        ),
        (
            r#"{"schema":1,"ordered_crates":["base","consumer"],"registry_independent":["consumer"]}"#,
            "prefix",
        ),
    ] {
        let error = parse_release_manifest_source(source).expect_err("invalid manifest rejects");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
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
fn release_docs_validation_reports_missing_packages_and_operational_guards() {
    let mut errors = Vec::new();

    validate_release_docs_source("", &mut errors);

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
