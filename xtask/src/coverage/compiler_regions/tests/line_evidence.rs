// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{CompilerRegionEvidence, CompilerRegionReport, SourceSpan};

#[test]
fn line_evidence_uses_the_most_specific_intersecting_region() {
    let path = "crates/demo/src/lib.rs";
    let report = CompilerRegionReport::for_test(
        path,
        &[
            (SourceSpan::new(1, 1, 5, 1).unwrap(), 9),
            (SourceSpan::new(3, 5, 3, 10).unwrap(), 0),
        ],
    );

    assert_eq!(
        report.evidence_for_line(path, 2),
        Some(CompilerRegionEvidence::Covered)
    );
    assert_eq!(
        report.evidence_for_line(path, 3),
        Some(CompilerRegionEvidence::Uncovered),
        "a covered outer function region must not mask a nested zero-count branch"
    );
    assert_eq!(
        report.evidence_for_line(path, 5),
        Some(CompilerRegionEvidence::NonInstrumentable),
        "an exclusive end at column one must not own the next line"
    );
    assert_eq!(report.evidence_for_line("crates/missing.rs", 2), None);
}
