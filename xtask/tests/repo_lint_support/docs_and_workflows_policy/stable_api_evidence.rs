// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeMap, fs};

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[derive(Debug)]
struct SummaryEvidence {
    removed_fingerprint: String,
    added_fingerprint: String,
    hidden_count: u64,
    hidden_fingerprint: String,
}

#[test]
fn reviewed_api_diff_artifacts_cover_every_ordinary_and_hidden_fingerprint() {
    let root = repo_root();
    let report = fs::read_to_string(root.join("engineering/reviewed-public-api-diff-0.7.2.md"))
        .expect("read reviewed API diff report");
    let config_source = fs::read_to_string(root.join("engineering/public-api-review-0.7.2.yml"))
        .expect("read public API review config");
    let config: serde_yaml_ng::Value =
        serde_yaml_ng::from_str(&config_source).expect("parse public API review config");
    let config = config.as_mapping().expect("review config root mapping");
    assert_eq!(
        config.get("version").and_then(serde_yaml_ng::Value::as_u64),
        Some(2),
        "review config schema must cover hidden inventory evidence"
    );
    assert_eq!(
        config
            .get("candidate_version")
            .and_then(serde_yaml_ng::Value::as_str),
        Some("0.7.2")
    );
    let reviews = config
        .get("reviews")
        .and_then(serde_yaml_ng::Value::as_mapping)
        .expect("review config reviews mapping");

    let summary = parse_report_summary(&report);
    assert_eq!(summary.len(), 17, "API diff must list every stable library");
    assert_eq!(
        reviews.len(),
        summary.len(),
        "every stable package requires one exact review entry"
    );
    assert_review_evidence(reviews, &summary);

    assert_pattern_checks(&[
        PatternCheck::new("reviewed API report hidden evidence", &report).required(&[
            "Rustdoc-hidden items",
            "Hidden inventory fingerprint",
            "Full hidden-inventory fingerprint",
        ]),
        PatternCheck::new("reviewed API diff new-package classification", &report).required(&[
            "## New packages without a 0.6.2 registry baseline",
            "`j2k-codec-math` `0.7.2`",
        ]),
    ]);
}

fn parse_report_summary(report: &str) -> BTreeMap<String, SummaryEvidence> {
    let mut summary = BTreeMap::new();
    for line in report.lines().filter(|line| line.starts_with("| `")) {
        let cells = line.split('|').map(str::trim).collect::<Vec<_>>();
        assert_eq!(cells.len(), 12, "malformed API diff summary row: {line}");
        let package = cells[1].trim_matches('`').to_string();
        let hidden_count = cells[9]
            .parse::<u64>()
            .unwrap_or_else(|error| panic!("invalid hidden count for {package}: {error}"));
        let evidence = SummaryEvidence {
            removed_fingerprint: cells[7].trim_matches('`').to_string(),
            added_fingerprint: cells[8].trim_matches('`').to_string(),
            hidden_count,
            hidden_fingerprint: cells[10].trim_matches('`').to_string(),
        };
        assert!(
            summary.insert(package, evidence).is_none(),
            "duplicate package in API diff summary"
        );
    }
    summary
}

fn assert_review_evidence(
    reviews: &serde_yaml_ng::Mapping,
    summary: &BTreeMap<String, SummaryEvidence>,
) {
    for (package, evidence) in summary {
        let review = reviews
            .get(package.as_str())
            .and_then(serde_yaml_ng::Value::as_mapping)
            .unwrap_or_else(|| panic!("{package} lacks a review entry"));
        for (field, expected) in [
            ("removed_fingerprint", evidence.removed_fingerprint.as_str()),
            ("added_fingerprint", evidence.added_fingerprint.as_str()),
            ("hidden_fingerprint", evidence.hidden_fingerprint.as_str()),
        ] {
            assert_eq!(
                review.get(field).and_then(serde_yaml_ng::Value::as_str),
                Some(expected),
                "{package} {field} is stale"
            );
        }
        assert_eq!(
            review
                .get("hidden_count")
                .and_then(serde_yaml_ng::Value::as_u64),
            Some(evidence.hidden_count),
            "{package} hidden count is stale"
        );
        let rationale = review
            .get("rationale")
            .and_then(serde_yaml_ng::Value::as_str)
            .unwrap_or_else(|| panic!("{package} lacks an ordinary diff rationale"));
        assert!(
            rationale.trim().len() >= 20,
            "{package} rationale is too short"
        );
        if evidence.hidden_count > 0 {
            let hidden_rationale = review
                .get("hidden_rationale")
                .and_then(serde_yaml_ng::Value::as_str)
                .unwrap_or_else(|| panic!("{package} lacks a hidden-inventory rationale"));
            assert!(
                hidden_rationale.trim().len() >= 20,
                "{package} hidden rationale is too short"
            );
        }
    }
}
