// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use serde_yaml_ng::Value;

pub(super) fn value_contains_secret_reference(value: &Value) -> bool {
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

pub(super) fn untrusted_event_run_tokens(script: &str) -> BTreeSet<String> {
    github_expression_tokens(script)
        .into_iter()
        .filter(|token| is_untrusted_event_run_token(token))
        .collect()
}

fn is_untrusted_event_run_token(token: &str) -> bool {
    token == "github.head_ref"
        || (token.starts_with("github.event.")
            && [".title", ".body", ".head.ref"]
                .iter()
                .any(|suffix| token.ends_with(suffix)))
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
