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
fn cuda_oxide_matchers_honor_every_boundary() {
    let device = CoverageExclusion {
        id: "device-test",
        reason: "test fixture",
        matcher: ExclusionMatcher::CudaOxideDeviceRust,
        evidence: &[],
    };
    let scaffold = CoverageExclusion {
        id: "scaffold-test",
        reason: "test fixture",
        matcher: ExclusionMatcher::CudaOxideHostScaffold,
        evidence: &[],
    };

    assert!(exclusion_matches(
        &device,
        "crates/j2k-cuda-jpeg-engine/src/cuda_oxide_demo/simt/src/lib.rs",
        1,
        &[]
    )
    .unwrap());
    assert!(exclusion_matches(
        &scaffold,
        "crates/j2k-cuda-transcode-engine/src/cuda_oxide_demo/src/main.rs",
        1,
        &[]
    )
    .unwrap());
    for path in [
        "crates/j2k-cuda-other-engine/src/cuda_oxide_demo/simt/src/lib.rs",
        "crates/j2k-cuda-jpeg-engine/src/not_cuda_oxide/simt/src/lib.rs",
        "crates/j2k-cuda-jpeg-engine/src/cuda_oxide_demo/simt/src/lib.c",
        "crates/j2k-cuda-jpeg-engine/src/cuda_oxide_demo/host/src/main.rs",
    ] {
        assert!(!exclusion_matches(&device, path, 1, &[]).unwrap(), "{path}");
        assert!(
            !exclusion_matches(&scaffold, path, 1, &[]).unwrap(),
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
