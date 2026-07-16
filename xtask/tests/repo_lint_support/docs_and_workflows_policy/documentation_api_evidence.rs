// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::repo_lint_support::{
    assert_file_pattern_checks, assert_pattern_checks, repo_root, xtask_sources, FilePatternCheck,
    PatternCheck,
};

#[test]
fn codec_api_guide_covers_public_surfaces() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("README.md")
            .named("README.md codec API surface")
            .required(&[
                "Codec contracts",
                "decode_region_scaled_into",
                "decode_rows",
                "TileBatchDecode",
                "BackendRequest::Auto",
                "BackendRequest::Metal",
                "BackendRequest::Cuda",
                "DeviceSurface",
                "ScratchPool",
                "J2kContext",
                "j2k_jpeg::DecoderContext",
            ])],
    );
}

#[test]
fn ci_workflow_keeps_docs_and_benchmark_compile_gates() {
    let root = repo_root();
    let xtask = xtask_sources(root);
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new(".github/workflows/full-validation.yml")
                .named("CI workflow docs and benchmark compile gates")
                .required(&["cargo xtask doc", "cargo xtask bench-build --lane host"])
                .forbidden(&["macos-13"]),
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("xtask benchmark compile gate", &xtask).required(&[
            "\"doc\"",
            "\"--workspace\"",
            "\"--all-features\"",
            "\"--no-deps\"",
            "\"j2k-jpeg-metal\"",
            "\"j2k-metal\"",
            "\"--no-run\"",
        ]),
    ]);
}
