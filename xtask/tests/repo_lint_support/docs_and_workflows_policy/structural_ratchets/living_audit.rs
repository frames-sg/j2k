// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

const LIVING_AUDIT: &str = "engineering/ai-codebase-audit-remediation-plan.md";
const LIVING_AUDIT_MAX_LINES: usize = 500;

#[test]
fn living_audit_stays_current_and_bounded() {
    let source = fs::read_to_string(repo_root().join(LIVING_AUDIT))
        .unwrap_or_else(|error| panic!("read {LIVING_AUDIT}: {error}"));

    assert!(
        source.lines().count() < LIVING_AUDIT_MAX_LINES,
        "{LIVING_AUDIT} is a current-state register, not a historical task diary; keep it below {LIVING_AUDIT_MAX_LINES} lines and use Git history for completed work"
    );
    for stale_claim in [
        "workspace `0.7.0` remains staged",
        "0.7.0 remains staged",
        "blocked 0.7.0",
        "Final candidate ratio: pending",
    ] {
        assert!(
            !source.contains(stale_claim),
            "{LIVING_AUDIT} contains stale present-tense release claim {stale_claim:?}"
        );
    }
    for required in [
        "v0.7.3 verdict",
        "Active debt",
        "Accepted large-file register",
        "Accepted clone register",
        "Verification matrix",
    ] {
        assert!(
            source.contains(required),
            "{LIVING_AUDIT} must retain the current-state section {required:?}"
        );
    }
}

#[test]
fn current_candidate_matrix_does_not_claim_invalidated_green_evidence() {
    let source = fs::read_to_string(repo_root().join(LIVING_AUDIT))
        .unwrap_or_else(|error| panic!("read {LIVING_AUDIT}: {error}"));
    let matrix = source
        .split("## Verification matrix")
        .nth(1)
        .and_then(|tail| tail.split("## Living-document rule").next())
        .expect("verification matrix section");

    for row in matrix.lines().filter(|line| line.starts_with("| `")) {
        let columns = row.split('|').map(str::trim).collect::<Vec<_>>();
        let candidate = columns
            .get(3)
            .unwrap_or_else(|| panic!("candidate evidence column in {row:?}"));
        assert!(
            !candidate.contains("pass"),
            "current candidate evidence was invalidated by later source edits: {row}"
        );
        assert!(
            candidate.contains("pending") || candidate.contains("required"),
            "current candidate row must state its pending or required final validation: {row}"
        );
    }
}
