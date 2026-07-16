// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use super::review::{parse_review_config, validate_reviews, ReviewConfig, ReviewEntry};
use super::{
    computed_release_type, current_snapshot_uses_generation_contract, fingerprint,
    parse_api_snapshot, parse_options, render_report, semver_cargo_args, semver_check_release_type,
    snapshot_uses_generator, validate_package_partition, PackageApiDiff, ReleaseType, SnapshotKind,
    Version, SEMVER_BASELINE_PACKAGES, SEMVER_NEW_PACKAGES,
};

mod api_planning;
#[cfg(unix)]
mod command_boundaries;
mod parsing;

#[test]
fn release_type_uses_cargo_pre_one_compatibility_rules() {
    assert_eq!(
        computed_release_type(
            Version::parse("0.6.2").unwrap(),
            Version::parse("0.7.0").unwrap()
        )
        .unwrap(),
        ReleaseType::Major
    );
    assert_eq!(
        computed_release_type(
            Version::parse("1.6.2").unwrap(),
            Version::parse("1.7.0").unwrap()
        )
        .unwrap(),
        ReleaseType::Minor
    );
    assert_eq!(
        computed_release_type(
            Version::parse("1.6.2").unwrap(),
            Version::parse("1.6.3").unwrap()
        )
        .unwrap(),
        ReleaseType::Patch
    );
    assert_eq!(
        computed_release_type(
            Version::parse("0.6.2").unwrap(),
            Version::parse("0.6.3").unwrap()
        )
        .unwrap(),
        ReleaseType::Minor
    );
    assert_eq!(
        computed_release_type(
            Version::parse("0.0.2").unwrap(),
            Version::parse("0.0.3").unwrap()
        )
        .unwrap(),
        ReleaseType::Major
    );
    assert!(computed_release_type(
        Version::parse("0.7.0").unwrap(),
        Version::parse("0.6.2").unwrap()
    )
    .is_err());
}

#[test]
fn package_partition_lists_new_packages_explicitly() {
    let mut stable = SEMVER_BASELINE_PACKAGES.to_vec();
    stable.extend_from_slice(SEMVER_NEW_PACKAGES);
    assert!(validate_package_partition(&stable).is_ok());
    stable.pop();
    assert!(validate_package_partition(&stable).is_err());
}

#[test]
fn parses_snapshot_sections_as_sets() {
    let snapshot = "## `alpha`\n\n```text\npub fn alpha::b()\npub fn alpha::a()\n```\n\n## `beta`\n\n```text\npub struct beta::B\n```\n";
    let parsed = parse_api_snapshot(snapshot).unwrap();
    assert_eq!(
        parsed["alpha"].iter().cloned().collect::<Vec<_>>(),
        ["pub fn alpha::a()", "pub fn alpha::b()"]
    );
    assert_eq!(parsed["beta"].len(), 1);

    let empty = parse_api_snapshot("## `alpha`\n\n```text\n```\n").unwrap();
    assert!(empty["alpha"].is_empty());
    assert!(parse_api_snapshot("## `alpha`\n```text\n").is_err());
    assert!(parse_api_snapshot("## `alpha`\n```text\n```\n## `alpha`\n```text\n```\n").is_err());
}

#[test]
fn snapshot_generator_marker_must_match_a_complete_line() {
    let valid = "# J2K 1.0 Public API Snapshot\n\n\
                 Generator: `cargo-public-api 0.52.0`.\nbody\n";
    assert!(snapshot_uses_generator(valid, "0.52.0"));
    assert!(!snapshot_uses_generator(
        "# J2K 1.0 Public API Snapshot\n\n\
         header mentions Generator: `cargo-public-api 0.52.0`. inline\n",
        "0.52.0"
    ));
    assert!(!snapshot_uses_generator(
        "# J2K 1.0 Public API Snapshot\n\n\
         Generator: `cargo-public-api 0.52.0-dev`.\n",
        "0.52.0"
    ));
    assert!(!snapshot_uses_generator(
        "wrong header\nGenerator: `cargo-public-api 0.52.0`.\n",
        "0.52.0"
    ));

    let current = "# J2K 1.0 Rustdoc-Hidden Public API Snapshot\n\n\
                   Generator: `cargo-public-api 0.52.0`.\n\n\
                   Rustdoc toolchain: `nightly-2026-06-28`.\n\
                   Target: `aarch64-apple-darwin`.\n";
    assert!(current_snapshot_uses_generation_contract(
        current,
        SnapshotKind::Hidden,
        "0.52.0"
    ));
    assert!(!current_snapshot_uses_generation_contract(
        &current.replace("aarch64-apple-darwin", "x86_64-apple-darwin"),
        SnapshotKind::Hidden,
        "0.52.0"
    ));
}

#[test]
fn fingerprints_are_order_independent_and_empty_is_none() {
    let first = ["b".to_string(), "a".to_string()]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let second = ["a".to_string(), "b".to_string()]
        .into_iter()
        .collect::<BTreeSet<_>>();
    assert_eq!(fingerprint(&first), fingerprint(&second));
    assert_eq!(fingerprint(&BTreeSet::new()), "none");
}

#[test]
fn report_regeneration_flag_is_explicit() {
    assert!(!parse_options(std::iter::empty()).unwrap().write_report);
    assert!(
        parse_options(["--write-report".to_string()].into_iter())
            .unwrap()
            .write_report
    );
    assert!(parse_options(["--write".to_string()].into_iter()).is_err());
}

#[test]
fn semver_checks_commands_use_the_pinned_rustup_toolchain() {
    assert_eq!(
        semver_cargo_args(["semver-checks", "--version"]),
        ["run", "1.96", "cargo", "semver-checks", "--version"]
    );
}

#[test]
fn source_incompatible_patch_exception_is_scoped_to_0_7_4() {
    let mut diff = PackageApiDiff {
        package: "alpha".to_string(),
        candidate_version: "0.7.4".to_string(),
        release_type: Some(ReleaseType::Minor),
        baseline_count: 1,
        candidate_count: 0,
        added: BTreeSet::new(),
        removed: ["pub fn alpha::removed()".to_string()]
            .into_iter()
            .collect(),
        hidden: BTreeSet::new(),
    };
    assert_eq!(semver_check_release_type(&diff), ReleaseType::Major);
    diff.candidate_version = "0.7.5".to_string();
    assert_eq!(semver_check_release_type(&diff), ReleaseType::Minor);
}

#[test]
fn report_has_one_published_details_section_and_hidden_evidence() {
    let diff = PackageApiDiff {
        package: "alpha".to_string(),
        candidate_version: "0.7.0".to_string(),
        release_type: Some(ReleaseType::Major),
        baseline_count: 1,
        candidate_count: 1,
        added: BTreeSet::new(),
        removed: BTreeSet::new(),
        hidden: ["pub fn alpha::hidden()".to_string()].into_iter().collect(),
    };
    let report = render_report("0.7.0", &[diff], "0.52.0");
    assert_eq!(report.matches("## Published-package details").count(), 1);
    assert!(report.contains("Rustdoc-hidden candidate items: 1"));
    assert!(report.contains("Full hidden-inventory fingerprint: `fnv1a64:"));
}

#[test]
fn parses_review_config_and_rejects_unknown_fields() {
    let source = "\
version: 2
baseline_tag: v0.7.3
baseline_version: 0.7.3
candidate_version: 0.7.4
reviews:
  alpha:
    removed_fingerprint: 'fnv1a64:1234'
    added_fingerprint: 'none'
    hidden_count: 1
    hidden_fingerprint: 'fnv1a64:5678'
    rationale: 'Intentional reviewed compatibility change.'
    hidden_rationale: 'Reviewed reachable implementation adapter inventory.'
";
    let value: serde_yaml_ng::Value = serde_yaml_ng::from_str(source).unwrap();
    let parsed = parse_review_config(&value).unwrap();
    assert_eq!(parsed.candidate_version, "0.7.4");
    assert_eq!(parsed.reviews.len(), 1);

    let invalid = source.replacen("    rationale:", "    unknown: nope\n    rationale:", 1);
    let value: serde_yaml_ng::Value = serde_yaml_ng::from_str(&invalid).unwrap();
    assert!(parse_review_config(&value).is_err());

    let pending = source.replace(
        "Intentional reviewed compatibility change.",
        "PENDING maintainer review of this compatibility change.",
    );
    let value: serde_yaml_ng::Value = serde_yaml_ng::from_str(&pending).unwrap();
    let error = parse_review_config(&value).unwrap_err();
    assert!(error.contains("pending maintainer review"));
}

#[test]
fn unreviewed_and_stale_breaking_fingerprints_fail() {
    let removed = ["pub fn alpha::old()".to_string()]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let added = ["pub fn alpha::new()".to_string()]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let diff = PackageApiDiff {
        package: "alpha".to_string(),
        candidate_version: "0.7.0".to_string(),
        release_type: Some(ReleaseType::Major),
        baseline_count: 1,
        candidate_count: 1,
        added,
        removed,
        hidden: BTreeSet::new(),
    };
    let empty = ReviewConfig {
        candidate_version: "0.7.0".to_string(),
        reviews: BTreeMap::new(),
    };
    assert!(validate_reviews(&empty, "0.7.0", &[diff]).is_err());

    let removed = ["pub fn alpha::old()".to_string()]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let diff = PackageApiDiff {
        package: "alpha".to_string(),
        candidate_version: "0.7.0".to_string(),
        release_type: Some(ReleaseType::Major),
        baseline_count: 1,
        candidate_count: 0,
        added: BTreeSet::new(),
        removed,
        hidden: ["pub fn alpha::hidden()".to_string()].into_iter().collect(),
    };
    let reviews = [(
        "alpha".to_string(),
        ReviewEntry {
            removed_fingerprint: "fnv1a64:stale".to_string(),
            added_fingerprint: "none".to_string(),
            hidden_count: 1,
            hidden_fingerprint: fingerprint(
                &["pub fn alpha::hidden()".to_string()].into_iter().collect(),
            ),
            rationale: "Intentional reviewed compatibility change.".to_string(),
            hidden_rationale: Some(
                "Reviewed reachable implementation adapter inventory.".to_string(),
            ),
        },
    )]
    .into_iter()
    .collect();
    let stale = ReviewConfig {
        candidate_version: "0.7.0".to_string(),
        reviews,
    };
    assert!(validate_reviews(&stale, "0.7.0", &[diff]).is_err());
}
