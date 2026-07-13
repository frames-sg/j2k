// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::Path;

use super::{parse_compiler_regions, CompilerRegionEvidence, SourceSpan};

mod line_evidence;

const ROOT: &str = "/workspace/j2k";

#[test]
fn parser_aggregates_code_regions_by_normalized_repository_path() {
    let report = parse_compiler_regions(
        &format!(
            r#"{{
                "type": "llvm.coverage.json.export",
                "version": "3.1.0",
                "data": [{{
                    "files": [],
                    "functions": [{{
                        "filenames": ["{ROOT}/crates/demo/src/lib.rs"],
                        "regions": [
                            [1, 20, 1, 29, 0, 0, 0, 0],
                            [1, 20, 1, 29, 7, 0, 0, 0],
                            [1, 1, 1, 50, 9, 0, 0, 0],
                            [1, 20, 1, 29, 99, 0, 0, 3]
                        ]
                    }}]
                }}]
            }}"#
        ),
        Path::new(ROOT),
    )
    .unwrap();

    assert_eq!(
        report
            .evidence_for(
                "crates/demo/src/lib.rs",
                SourceSpan::new(1, 20, 1, 29).unwrap(),
            )
            .unwrap(),
        CompilerRegionEvidence::Covered
    );
}

#[test]
fn body_without_a_nested_code_region_is_compiler_noninstrumentable() {
    let report = parse_compiler_regions(
        &format!(
            r#"{{
                "type": "llvm.coverage.json.export",
                "version": "3.1.0",
                "data": [{{
                    "files": [],
                    "functions": [{{
                        "filenames": ["{ROOT}/crates/demo/src/lib.rs"],
                        "regions": [[1, 1, 1, 50, 9, 0, 0, 0]]
                    }}]
                }}]
            }}"#
        ),
        Path::new(ROOT),
    )
    .unwrap();

    assert_eq!(
        report
            .evidence_for(
                "crates/demo/src/lib.rs",
                SourceSpan::new(1, 20, 1, 29).unwrap(),
            )
            .unwrap(),
        CompilerRegionEvidence::NonInstrumentable
    );
}

#[test]
fn nested_zero_count_code_region_is_uncovered() {
    let report = parse_compiler_regions(
        &format!(
            r#"{{
                "type": "llvm.coverage.json.export",
                "version": "3.1.0",
                "data": [{{
                    "files": [],
                    "functions": [{{
                        "filenames": ["{ROOT}/crates/demo/src/lib.rs"],
                        "regions": [[1, 22, 1, 28, 0, 0, 0, 0]]
                    }}]
                }}]
            }}"#
        ),
        Path::new(ROOT),
    )
    .unwrap();

    assert_eq!(
        report
            .evidence_for(
                "crates/demo/src/lib.rs",
                SourceSpan::new(1, 20, 1, 29).unwrap(),
            )
            .unwrap(),
        CompilerRegionEvidence::Uncovered
    );
}

#[test]
fn malformed_or_unrelated_reports_fail_closed() {
    for input in [
        "{}",
        r#"{"type":"wrong","version":"3.1.0","data":[]}"#,
        r#"{"type":"llvm.coverage.json.export","version":"3.1.0","data":[{"files":[],"functions":[{"filenames":["/workspace/j2k/src/lib.rs"],"regions":[[1,2,1]]}]}]}"#,
    ] {
        assert!(
            parse_compiler_regions(input, Path::new(ROOT)).is_err(),
            "{input}"
        );
    }

    let report = parse_compiler_regions(
        r#"{"type":"llvm.coverage.json.export","version":"3.1.0","data":[{"files":[],"functions":[]}] }"#,
        Path::new(ROOT),
    )
    .unwrap();
    assert!(report
        .evidence_for(
            "crates/missing/src/lib.rs",
            SourceSpan::new(1, 1, 1, 2).unwrap(),
        )
        .is_err());
}

#[test]
fn dependency_macro_expansion_regions_are_ignored_without_hiding_repository_regions() {
    let report = parse_compiler_regions(
        &format!(
            r#"{{
                "type": "llvm.coverage.json.export",
                "version": "3.1.0",
                "data": [{{
                    "files": [{{"filename": "{ROOT}/crates/demo/src/lib.rs"}}],
                    "functions": [{{
                        "filenames": [
                            "/cargo/registry/dependency/src/macros.rs",
                            "{ROOT}/crates/demo/src/lib.rs"
                        ],
                        "regions": [
                            [1, 1, 1, 10, 3, 0, 0, 0],
                            [2, 5, 2, 12, 4, 1, 0, 0]
                        ]
                    }}]
                }}]
            }}"#
        ),
        Path::new(ROOT),
    )
    .unwrap();

    assert_eq!(
        report
            .evidence_for(
                "crates/demo/src/lib.rs",
                SourceSpan::new(2, 5, 2, 12).unwrap(),
            )
            .unwrap(),
        CompilerRegionEvidence::Covered
    );
}
