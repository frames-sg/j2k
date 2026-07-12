// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    exclusion_matches, validate_evidence_test_source, CoverageExclusion, EvidenceClass,
    ExclusionMatcher,
};

const MARKER_EXCLUSION: CoverageExclusion = CoverageExclusion {
    id: "marker-test",
    reason: "test fixture",
    matcher: ExclusionMatcher::MarkerSpan {
        path: "src/generated.rs",
        start: "// begin generated",
        end: "// end generated",
    },
    evidence: &[],
};

#[test]
fn marker_spans_match_only_the_closed_reviewed_interval() {
    let source = [
        "fn before() {}",
        "// begin generated",
        "0",
        "// end generated",
    ];

    assert!(!exclusion_matches(&MARKER_EXCLUSION, "src/other.rs", 3, &source).unwrap());
    assert!(!exclusion_matches(&MARKER_EXCLUSION, "src/generated.rs", 1, &source).unwrap());
    for line in 2..=4 {
        assert!(exclusion_matches(&MARKER_EXCLUSION, "src/generated.rs", line, &source).unwrap());
    }
}

#[test]
fn missing_ambiguous_and_reversed_markers_fail_closed() {
    for (source, expected) in [
        (vec!["// end generated"], "missing"),
        (
            vec![
                "// begin generated",
                "// begin generated",
                "// end generated",
            ],
            "ambiguous",
        ),
        (vec!["// end generated", "// begin generated"], "order"),
    ] {
        let error =
            exclusion_matches(&MARKER_EXCLUSION, "src/generated.rs", 1, &source).unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn path_patterns_honor_every_boundary() {
    let exclusion = CoverageExclusion {
        id: "pattern-test",
        reason: "test fixture",
        matcher: ExclusionMatcher::PathPattern {
            prefix: "generated/",
            contains: Some("/simt/"),
            excludes: Some("/host/"),
            suffix: ".rs",
        },
        evidence: &[],
    };

    assert!(exclusion_matches(&exclusion, "generated/a/simt/lib.rs", 1, &[]).unwrap());
    for path in [
        "other/a/simt/lib.rs",
        "generated/a/lib.rs",
        "generated/a/simt/host/lib.rs",
        "generated/a/simt/lib.c",
    ] {
        assert!(
            !exclusion_matches(&exclusion, path, 1, &[]).unwrap(),
            "{path}"
        );
    }
}

#[test]
fn duplicate_or_non_runnable_evidence_symbols_fail_closed() {
    let duplicate = "#[test] fn parity() {}\nmod nested { #[test] fn parity() {} }\n";
    let error = validate_evidence_test_source(
        "tests/parity.rs",
        "parity",
        EvidenceClass::Primary,
        duplicate,
    )
    .unwrap_err();
    assert!(error.contains("ambiguous"), "{error}");

    let should_panic = "#[test]\n#[should_panic]\nfn parity() {}\n";
    let error = validate_evidence_test_source(
        "tests/parity.rs",
        "parity",
        EvidenceClass::Primary,
        should_panic,
    )
    .unwrap_err();
    assert!(error.contains("must not be ignored"), "{error}");
}
