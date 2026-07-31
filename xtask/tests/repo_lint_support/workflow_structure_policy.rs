// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural GitHub Actions policy with workflow YAML as the source of truth.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, PathBuf};
use std::sync::OnceLock;

use serde_yaml_ng::{Mapping, Value};

use super::repo_root;

#[derive(Debug)]
struct Workflow {
    file_name: String,
    document: Value,
}

fn workflows() -> &'static [Workflow] {
    static WORKFLOWS: OnceLock<Vec<Workflow>> = OnceLock::new();
    WORKFLOWS.get_or_init(|| {
        let directory = repo_root().join(".github/workflows");
        let mut paths = fs::read_dir(&directory)
            .expect("read workflow directory")
            .map(|entry| entry.expect("read workflow entry").path())
            .filter(|path| {
                matches!(
                    path.extension().and_then(|extension| extension.to_str()),
                    Some("yml" | "yaml")
                )
            })
            .collect::<Vec<_>>();
        paths.sort();
        assert!(!paths.is_empty(), "no GitHub Actions workflows found");

        paths
            .into_iter()
            .map(|path| {
                let source = fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
                let document = serde_yaml_ng::from_str(&source)
                    .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
                let file_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_else(|| {
                        panic!("workflow path has no UTF-8 name: {}", path.display())
                    })
                    .to_owned();
                Workflow {
                    file_name,
                    document,
                }
            })
            .collect()
    })
}

fn workflow(file_name: &str) -> &'static Workflow {
    workflows()
        .iter()
        .find(|workflow| workflow.file_name == file_name)
        .unwrap_or_else(|| panic!("missing workflow {file_name}"))
}

#[test]
fn every_action_reference_is_pinned_to_a_forty_hex_sha() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        visit_mappings(&workflow.document, &mut |mapping| {
            let Some(reference) = mapping_get(mapping, "uses").and_then(Value::as_str) else {
                return;
            };
            if reference.starts_with("./") || reference.starts_with("docker://") {
                return;
            }
            let pinned = reference.rsplit_once('@').is_some_and(|(_, revision)| {
                revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
            });
            if !pinned {
                violations.push(format!("{}: {reference}", workflow.file_name));
            }
        });
    }
    assert!(
        violations.is_empty(),
        "external actions must use a full forty-hex commit SHA:\n{}",
        violations.join("\n")
    );
}

#[test]
fn every_workflow_declares_default_permissions_including_contents_read() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        let root = value_mapping(&workflow.document, &workflow.file_name);
        let contents = mapping_get(root, "permissions")
            .and_then(Value::as_mapping)
            .and_then(|permissions| mapping_get(permissions, "contents"))
            .and_then(Value::as_str);
        if contents != Some("read") {
            violations.push(workflow.file_name.clone());
        }
    }
    assert!(
        violations.is_empty(),
        "workflows missing top-level contents: read permission: {violations:?}"
    );
}

#[test]
fn no_job_requests_a_write_permission_scope() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        visit_mappings(&workflow.document, &mut |mapping| {
            let Some(permissions) = mapping_get(mapping, "permissions") else {
                return;
            };
            if permissions.as_str() == Some("write-all")
                || permissions.as_mapping().is_some_and(|scopes| {
                    scopes.values().any(|value| value.as_str() == Some("write"))
                })
            {
                violations.push(workflow.file_name.clone());
            }
        });
    }
    violations.sort();
    violations.dedup();
    assert!(
        violations.is_empty(),
        "workflow permission scopes must remain read-only: {violations:?}"
    );
}

#[test]
fn every_downloaded_release_asset_is_checksum_verified() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        visit_mappings(&workflow.document, &mut |mapping| {
            let Some(script) = mapping_get(mapping, "run").and_then(Value::as_str) else {
                return;
            };
            if script.contains("/releases/download/") && !script.contains("sha256sum -c") {
                violations.push(workflow.file_name.clone());
            }
        });
    }
    assert!(
        violations.is_empty(),
        "downloaded release assets need same-step SHA-256 verification: {violations:?}"
    );
}

#[test]
fn the_secret_scan_redacts_findings_and_scans_full_history() {
    let secret_scan = workflow("secret-scan.yml");
    let mut full_checkout = false;
    let mut fail_closed_scan = false;
    visit_mappings(&secret_scan.document, &mut |mapping| {
        if mapping_get(mapping, "uses")
            .and_then(Value::as_str)
            .is_some_and(|reference| reference.starts_with("actions/checkout@"))
        {
            full_checkout = mapping_get(mapping, "with")
                .and_then(Value::as_mapping)
                .and_then(|with| mapping_get(with, "fetch-depth"))
                .is_some_and(is_numeric_zero);
        }
        if let Some(script) = mapping_get(mapping, "run").and_then(Value::as_str) {
            if script.contains("gitleaks detect") {
                fail_closed_scan = script.contains("--source .") && script.contains("--redact");
            }
        }
    });
    assert!(
        full_checkout,
        "secret scan checkout must fetch full history"
    );
    assert!(
        fail_closed_scan,
        "secret scan must redact findings and scan the repository"
    );
}

#[test]
fn manual_publish_dispatch_is_dry_run_only() {
    let publish = workflow("publish.yml");
    let root = value_mapping(&publish.document, &publish.file_name);
    let triggers = mapping_get(root, "on")
        .and_then(Value::as_mapping)
        .expect("publish workflow trigger mapping");
    assert!(
        mapping_get(triggers, "workflow_dispatch").is_some(),
        "publish workflow must keep its manual dry-run entry point"
    );
    let dry_run = mapping_get(root, "env")
        .and_then(Value::as_mapping)
        .and_then(|environment| mapping_get(environment, "DRY_RUN_ONLY"))
        .and_then(Value::as_str)
        .expect("publish DRY_RUN_ONLY expression");
    assert!(
        dry_run.contains("github.event_name == 'workflow_dispatch'"),
        "manual publish dispatch must set DRY_RUN_ONLY"
    );

    let publish_job = jobs(root, &publish.file_name)
        .get(Value::String("publish".to_owned()))
        .and_then(Value::as_mapping)
        .expect("publish job");
    let condition = mapping_get(publish_job, "if")
        .and_then(Value::as_str)
        .expect("publish job event guard");
    assert!(
        condition.contains("github.event_name == 'push'"),
        "the credentialed publish job must not run for workflow_dispatch"
    );
}

#[test]
fn every_job_that_reads_a_secret_declares_a_protected_environment() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        let root = value_mapping(&workflow.document, &workflow.file_name);
        for (job_name, job) in jobs(root, &workflow.file_name) {
            if value_contains_secret_reference(job)
                && job
                    .as_mapping()
                    .and_then(|mapping| mapping_get(mapping, "environment"))
                    .is_none_or(Value::is_null)
            {
                violations.push(format!("{}:{}", workflow.file_name, display_key(job_name)));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "jobs reading secrets need a protected environment:\n{}",
        violations.join("\n")
    );
}

#[test]
fn no_workflow_uses_pull_request_target_or_interpolates_untrusted_event_fields() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        let root = value_mapping(&workflow.document, &workflow.file_name);
        let triggers = mapping_get(root, "on")
            .and_then(Value::as_mapping)
            .unwrap_or_else(|| panic!("{} trigger mapping", workflow.file_name));
        if mapping_get(triggers, "pull_request_target").is_some() {
            violations.push(format!("{}: pull_request_target", workflow.file_name));
        }
        visit_mappings(&workflow.document, &mut |mapping| {
            let Some(script) = mapping_get(mapping, "run").and_then(Value::as_str) else {
                return;
            };
            for token in untrusted_event_run_tokens(script) {
                violations.push(format!("{}: {token}", workflow.file_name));
            }
        });
    }
    assert!(
        violations.is_empty(),
        "untrusted event text must not reach shell bodies:\n{}",
        violations.join("\n")
    );
}

#[test]
fn every_required_pr_gate_is_listed_in_the_aggregate_job_needs() {
    let ci = workflow("ci.yml");
    let root = value_mapping(&ci.document, &ci.file_name);
    let jobs = jobs(root, &ci.file_name);
    let aggregate_name = Value::String("pr-required-checks".to_owned());
    let aggregate = jobs
        .get(&aggregate_name)
        .and_then(Value::as_mapping)
        .expect("PR aggregate job");

    let mut expected = jobs
        .keys()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    expected.remove("pr-required-checks");
    let actual = string_set(
        mapping_get(aggregate, "needs").expect("PR aggregate needs"),
        "PR aggregate needs",
    );
    assert_eq!(
        actual, expected,
        "PR aggregate needs must cover every sibling gate"
    );
}

#[test]
fn local_reusable_workflow_references_resolve_to_files_in_this_repo() {
    let mut violations = Vec::new();
    for source_workflow in workflows() {
        visit_mappings(&source_workflow.document, &mut |mapping| {
            let Some(reference) = mapping_get(mapping, "uses").and_then(Value::as_str) else {
                return;
            };
            if !reference.starts_with("./.github/workflows/") {
                return;
            }
            let relative = reference.trim_start_matches("./");
            let path = PathBuf::from(relative);
            if path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
            {
                violations.push(format!(
                    "{}: path escape {reference}",
                    source_workflow.file_name
                ));
                return;
            }
            let target = repo_root().join(&path);
            if !target.is_file() {
                violations.push(format!(
                    "{}: missing {reference}",
                    source_workflow.file_name
                ));
                return;
            }
            let Some(target_name) = target.file_name().and_then(|name| name.to_str()) else {
                violations.push(format!(
                    "{}: non-UTF-8 {reference}",
                    source_workflow.file_name
                ));
                return;
            };
            let target_workflow = workflow(target_name);
            let target_root = value_mapping(&target_workflow.document, &target_workflow.file_name);
            let callable = mapping_get(target_root, "on")
                .and_then(Value::as_mapping)
                .is_some_and(|triggers| mapping_get(triggers, "workflow_call").is_some());
            if !callable {
                violations.push(format!(
                    "{}: target is not reusable {reference}",
                    source_workflow.file_name
                ));
            }
        });
    }
    assert!(
        violations.is_empty(),
        "local reusable workflow references must resolve in-repo:\n{}",
        violations.join("\n")
    );
}

fn value_mapping<'a>(value: &'a Value, label: &str) -> &'a Mapping {
    value
        .as_mapping()
        .unwrap_or_else(|| panic!("{label} must contain a YAML mapping"))
}

fn jobs<'a>(root: &'a Mapping, label: &str) -> &'a Mapping {
    mapping_get(root, "jobs")
        .and_then(Value::as_mapping)
        .unwrap_or_else(|| panic!("{label} must contain a jobs mapping"))
}

fn mapping_get<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.get(Value::String(key.to_owned())).or_else(|| {
        (key == "on")
            .then(|| mapping.get(Value::Bool(true)))
            .flatten()
    })
}

fn visit_mappings(value: &Value, visitor: &mut impl FnMut(&Mapping)) {
    match value {
        Value::Mapping(mapping) => {
            visitor(mapping);
            for (key, value) in mapping {
                visit_mappings(key, visitor);
                visit_mappings(value, visitor);
            }
        }
        Value::Sequence(values) => {
            for value in values {
                visit_mappings(value, visitor);
            }
        }
        _ => {}
    }
}

fn value_contains_secret_reference(value: &Value) -> bool {
    match value {
        Value::String(text) => github_expression_tokens(text)
            .iter()
            .any(|token| token == "secrets" || token.starts_with("secrets.")),
        Value::Sequence(values) => values.iter().any(value_contains_secret_reference),
        Value::Mapping(mapping) => mapping.iter().any(|(key, value)| {
            value_contains_secret_reference(key) || value_contains_secret_reference(value)
        }),
        _ => false,
    }
}

fn string_set(value: &Value, label: &str) -> BTreeSet<String> {
    match value {
        Value::String(value) => [value.clone()].into_iter().collect(),
        Value::Sequence(values) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .unwrap_or_else(|| panic!("{label} entries must be strings"))
                    .to_owned()
            })
            .collect(),
        _ => panic!("{label} must be a string or sequence"),
    }
}

fn is_numeric_zero(value: &Value) -> bool {
    value.as_u64() == Some(0) || value.as_i64() == Some(0) || value.as_str() == Some("0")
}

fn display_key(value: &Value) -> String {
    value
        .as_str()
        .map_or_else(|| format!("{value:?}"), str::to_owned)
}

fn is_untrusted_event_run_token(token: &str) -> bool {
    token == "github.head_ref"
        || (token.starts_with("github.event.")
            && [".title", ".body", ".head.ref"]
                .iter()
                .any(|suffix| token.ends_with(suffix)))
}

fn untrusted_event_run_tokens(script: &str) -> BTreeSet<String> {
    github_expression_tokens(script)
        .into_iter()
        .filter(|token| is_untrusted_event_run_token(token))
        .collect()
}

fn github_expression_tokens(text: &str) -> BTreeSet<String> {
    let compact = text
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();
    let canonical = compact
        .replace("['", ".")
        .replace("']", "")
        .replace("[\"", ".")
        .replace("\"]", "");

    let mut tokens = BTreeSet::new();
    let mut remaining = canonical.as_str();
    while let Some(start) = remaining.find("${{") {
        let expression = &remaining[start + 3..];
        let (expression, next) = expression.find("}}").map_or((expression, None), |end| {
            (&expression[..end], Some(&expression[end + 2..]))
        });
        tokens.extend(
            expression
                .split(|character: char| {
                    !(character.is_ascii_alphanumeric() || character == '_' || character == '.')
                })
                .filter(|token| !token.is_empty())
                .map(str::to_owned),
        );
        let Some(next) = next else {
            break;
        };
        remaining = next;
    }
    tokens
}

#[test]
fn untrusted_event_run_tokens_cover_event_families_without_rejecting_commit_shas() {
    for token in [
        "github.event.pull_request.title",
        "github.event.issue.body",
        "github.event.discussion.title",
        "github.event.pull_request.head.ref",
        "github.head_ref",
    ] {
        assert!(is_untrusted_event_run_token(token), "{token}");
    }
    for token in [
        "github.event.pull_request.head.sha",
        "github.event.pull_request.base.sha",
        "github.event.before",
        "github.event_name",
    ] {
        assert!(!is_untrusted_event_run_token(token), "{token}");
    }
}

#[test]
fn untrusted_event_run_tokens_normalize_bracket_and_dotted_property_access() {
    for script in [
        "echo '${{ github.event.issue.title }}'",
        "echo '${{ github ['event'] ['discussion'] ['body'] }}'",
        "echo '${{ github[\"event\"].pull_request['head'][\"ref\"] }}'",
        "echo '${{ github ['head_ref'] }}'",
    ] {
        assert!(
            !untrusted_event_run_tokens(script).is_empty(),
            "expected untrusted reference in {script}"
        );
    }

    for script in [
        "echo '${{ github.event.pull_request.head.sha }}'",
        "echo '${{ github ['event'] ['pull_request'] ['base'] ['sha'] }}'",
        "echo '${{ github.event.before }}'",
    ] {
        assert!(
            untrusted_event_run_tokens(script).is_empty(),
            "unexpected untrusted reference in {script}"
        );
    }
}

#[test]
fn secret_reference_detection_normalizes_bracket_and_dotted_property_access() {
    for reference in [
        "${{ secrets.CARGO_TOKEN }}",
        "${{ secrets ['CARGO_TOKEN'] }}",
        "${{ toJson(secrets) }}",
    ] {
        let value = Value::Sequence(vec![Value::String(reference.to_owned())]);
        assert!(
            value_contains_secret_reference(&value),
            "expected secret reference in {reference}"
        );
    }

    for reference in [
        "${{ env.CARGO_TOKEN }}",
        "${{ vars.CARGO_TOKEN }}",
        "documentation mentioning secrets.CARGO_TOKEN outside an expression",
    ] {
        let value = Value::String(reference.to_owned());
        assert!(
            !value_contains_secret_reference(&value),
            "unexpected secret reference in {reference}"
        );
    }
}
