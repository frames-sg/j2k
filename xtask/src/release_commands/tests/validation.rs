// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use super::super::{
    collect_publish_workflow_crates, publish_crate_from_run_line, validate_publish_script_source,
    validate_publish_workflow_source, validate_release_docs_source,
    validate_unpublished_dependencies,
};

#[test]
fn publish_workflow_validation_reports_parse_and_release_contract_failures() {
    let parse_error = validate_publish_workflow_source("jobs: [", &mut Vec::new())
        .expect_err("malformed workflow YAML must reject");
    assert!(parse_error.contains("failed to parse .github/workflows/publish.yml"));

    let workflow = r#"
checks: |
  --origin-url "${origin_url}"
  --server-url "${GITHUB_SERVER_URL}"
  cargo xtask release-integrity --publish
  scripts/publish-crate.sh --preflight-all
  scripts/publish-crate.sh j2k-core
  scripts/publish-crate.sh unknown-crate
"#;
    let mut errors = Vec::new();
    validate_publish_workflow_source(workflow, &mut errors).expect("valid workflow YAML");

    assert!(errors
        .iter()
        .any(|error| error.contains("publish order is")));
    assert!(errors
        .iter()
        .any(|error| error.contains("missing publish job for `j2k-profile`")));
    assert!(errors
        .iter()
        .any(|error| error.contains("publishes unknown workspace crate `unknown-crate`")));
}

#[test]
fn publish_workflow_collection_ignores_non_commands_and_missing_arguments() {
    let workflow = serde_yaml_ng::Value::Sequence(vec![
        serde_yaml_ng::Value::Bool(true),
        serde_yaml_ng::Value::Null,
        serde_yaml_ng::Value::String("scripts/publish-crate.sh j2k".to_string()),
    ]);
    let mut crates = Vec::new();

    collect_publish_workflow_crates(&workflow, &mut crates);

    assert_eq!(crates, ["j2k"]);
    assert_eq!(
        publish_crate_from_run_line("scripts/publish-crate.sh"),
        None
    );
}

#[test]
fn publish_script_validation_fails_closed_for_missing_and_drifted_contracts() {
    let missing_publishable = validate_publish_script_source("", &mut Vec::new())
        .expect_err("missing publishable array must reject");
    assert!(missing_publishable.contains("does not define the publishable_crates shell array"));

    let missing_independent =
        validate_publish_script_source("publishable_crates=(\n)\n", &mut Vec::new())
            .expect_err("missing independent array must reject");
    assert!(
        missing_independent.contains("does not define the registry_independent_crates shell array")
    );

    let script = r"
publishable_crates=(
  unknown-crate
)
registry_independent_crates=(
  another-unknown-crate
)
cargo info
";
    let mut errors = Vec::new();
    validate_publish_script_source(script, &mut errors).expect("arrays are structurally valid");

    assert!(errors
        .iter()
        .any(|error| error.contains("publishable_crates is")));
    assert!(errors
        .iter()
        .any(|error| error.contains("registry_independent_crates is")));
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
