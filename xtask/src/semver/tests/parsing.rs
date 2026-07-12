// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    current_snapshot_uses_generation_contract, exact_metadata_line, parse_api_snapshot,
    parse_options, require_version_token, SnapshotKind, Version, PUBLIC_API_TARGET,
    PUBLIC_API_TOOLCHAIN,
};

#[test]
fn release_version_parser_rejects_non_release_shapes() {
    for (version, expected) in [
        ("1", "missing its minor component"),
        ("1.2", "missing its patch component"),
        ("1.2.3.4", "exactly three components"),
        ("1.two.3", "invalid minor component"),
        ("1.2.3-alpha", "must not contain prerelease"),
        ("1.2.3+build", "must not contain prerelease"),
        ("18446744073709551616.0.0", "invalid major component"),
    ] {
        let error = Version::parse(version).expect_err("invalid release version");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn pinned_tool_version_requires_exact_tool_and_version_tokens() {
    assert_eq!(
        require_version_token(
            "cargo-semver-checks 0.48.0",
            "cargo-semver-checks",
            "0.48.0"
        ),
        Ok(())
    );
    for output in [
        "cargo-semver-checks 0.48.1",
        "other-tool 0.48.0",
        "cargo-semver-checks",
        "",
    ] {
        let error = require_version_token(output, "cargo-semver-checks", "0.48.0")
            .expect_err("mismatched tool version");
        assert!(error.contains("version must be 0.48.0"));
        assert!(error.contains(output));
    }
}

#[test]
fn semver_options_reject_duplicate_report_regeneration() {
    let error =
        parse_options(["--write-report".to_string(), "--write-report".to_string()].into_iter())
            .expect_err("duplicate regeneration flag");
    assert!(error.contains("duplicate semver argument"));
}

#[test]
fn api_snapshot_parser_fails_closed_on_malformed_fences() {
    for (snapshot, expected) in [
        ("```text\nitem\n```\n", "before a package heading"),
        ("## `alpha`\n```text\n```text\n", "nested text fences"),
        (
            "## `alpha`\n```text\n## `beta`\n",
            "heading inside a text fence",
        ),
        ("## `alpha`\n```\n", "unmatched closing fence"),
        ("header only\n", "did not contain package API sections"),
    ] {
        let error = parse_api_snapshot(snapshot).expect_err("malformed snapshot");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn current_snapshot_contract_rejects_duplicate_or_out_of_window_metadata() {
    let valid = format!(
        "# J2K 1.0 Public API Snapshot\n\nGenerator: `cargo-public-api 0.52.0`.\n\nRustdoc toolchain: `{PUBLIC_API_TOOLCHAIN}`.\nTarget: `{PUBLIC_API_TARGET}`.\n"
    );
    assert!(current_snapshot_uses_generation_contract(
        &valid,
        SnapshotKind::Ordinary,
        "0.52.0"
    ));
    assert!(!current_snapshot_uses_generation_contract(
        &format!("{valid}Generator: `cargo-public-api 0.52.0`.\n"),
        SnapshotKind::Ordinary,
        "0.52.0"
    ));
    let late_metadata = format!(
        "# J2K 1.0 Public API Snapshot\n{}Generator: `cargo-public-api 0.52.0`.\nRustdoc toolchain: `{PUBLIC_API_TOOLCHAIN}`.\nTarget: `{PUBLIC_API_TARGET}`.\n",
        "padding\n".repeat(25)
    );
    assert!(!current_snapshot_uses_generation_contract(
        &late_metadata,
        SnapshotKind::Ordinary,
        "0.52.0"
    ));
}

#[test]
fn exact_metadata_line_requires_one_complete_match() {
    let lines = ["Generator: exact", "other"];
    assert!(exact_metadata_line(
        &lines,
        "Generator:",
        "Generator: exact"
    ));
    assert!(!exact_metadata_line(
        &["Generator: exact suffix"],
        "Generator:",
        "Generator: exact"
    ));
    assert!(!exact_metadata_line(
        &["Generator: exact", "Generator: exact"],
        "Generator:",
        "Generator: exact"
    ));
}
