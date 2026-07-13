// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::Path;

use super::super::{parse_compiler_regions, CompilerRegionEvidence, SourceSpan};
use super::ROOT;

#[test]
fn parser_collapses_parent_components_in_repository_absolute_paths() {
    let report = parse_compiler_regions(
        &format!(
            r#"{{
                "type": "llvm.coverage.json.export",
                "version": "3.1.0",
                "data": [{{
                    "files": [],
                    "functions": [{{
                        "filenames": ["{ROOT}/crates/demo/src/bin/../../benches/support.rs"],
                        "regions": [[1, 1, 1, 20, 3, 0, 0, 0]]
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
                "crates/demo/benches/support.rs",
                SourceSpan::new(1, 1, 1, 20).unwrap(),
            )
            .unwrap(),
        CompilerRegionEvidence::Covered
    );
}
