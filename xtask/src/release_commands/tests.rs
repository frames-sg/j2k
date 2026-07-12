// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    collect_publish_workflow_crates, has_docs_rs_metadata, has_lib_target, package_name,
    publish_crate_from_run_line, publish_false, shell_array_values,
};

#[test]
fn publish_workflow_parser_collects_only_concrete_package_invocations() {
    let workflow: serde_yaml_ng::Value = serde_yaml_ng::from_str(
        "jobs:\n  publish:\n    steps:\n      - run: |\n          scripts/publish-crate.sh --preflight-all\n          scripts/publish-crate.sh 'j2k-core'\n      - run: scripts/publish-crate.sh \"j2k-native\"\n",
    )
    .expect("workflow YAML");
    let mut crates = Vec::new();

    collect_publish_workflow_crates(&workflow, &mut crates);

    assert_eq!(crates, ["j2k-core", "j2k-native"]);
    assert_eq!(
        publish_crate_from_run_line("prefix scripts/publish-crate.sh 'j2k' suffix"),
        Some("j2k".to_string())
    );
    assert_eq!(publish_crate_from_run_line("cargo publish"), None);
    assert_eq!(
        publish_crate_from_run_line("scripts/publish-crate.sh --preflight-all"),
        None
    );
}

#[test]
fn shell_array_parser_handles_comments_quotes_empty_lines_and_unclosed_arrays() {
    let script = r#"
publishable_crates=(
  "j2k-core" 'j2k-native' # staged crates

  j2k
)
"#;
    assert_eq!(
        shell_array_values(script, "publishable_crates"),
        Some(vec![
            "j2k-core".to_string(),
            "j2k-native".to_string(),
            "j2k".to_string(),
        ])
    );
    assert_eq!(shell_array_values(script, "missing"), None);
    assert_eq!(shell_array_values("values=(\none\n", "values"), None);
}

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
