// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::PathBuf};

fn read(relative_path: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative_path);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn assert_line_budget(relative_path: &str, source: &str, max_lines: usize) {
    let line_count = source.lines().count();
    assert!(
        line_count < max_lines,
        "{relative_path} has {line_count} lines; expected fewer than {max_lines}"
    );
}

fn assert_contains_all(source_name: &str, source: &str, patterns: &[&str]) {
    for pattern in patterns {
        assert!(
            source.contains(pattern),
            "{source_name} must contain `{pattern}`"
        );
    }
}

fn assert_contains_none(source_name: &str, source: &str, patterns: &[&str]) {
    for pattern in patterns {
        assert!(
            !source.contains(pattern),
            "{source_name} must not contain `{pattern}`"
        );
    }
}

#[test]
fn classic_bitplane_encoder_stays_split_by_responsibility() {
    let root = read("src/j2c/bitplane_encode.rs");
    let tokens = read("src/j2c/bitplane_encode/tokens.rs");
    let passes = read("src/j2c/bitplane_encode/passes.rs");
    let segments = read("src/j2c/bitplane_encode/segments.rs");
    let distortion = read("src/j2c/bitplane_encode/distortion.rs");

    for (path, source) in [
        ("bitplane_encode.rs", root.as_str()),
        ("bitplane_encode/tokens.rs", tokens.as_str()),
        ("bitplane_encode/passes.rs", passes.as_str()),
        ("bitplane_encode/segments.rs", segments.as_str()),
        ("bitplane_encode/distortion.rs", distortion.as_str()),
    ] {
        assert_line_budget(path, source, 700);
        assert_contains_none(path, source, &["use super::*", "include!(", "#[allow("]);
    }

    assert_contains_all(
        "bitplane encoder facade",
        &root,
        &[
            "mod tokens;",
            "mod passes;",
            "mod segments;",
            "mod distortion;",
            "pub(crate) use self::tokens::",
            "segments::encode_segmented_code_block(",
        ],
    );
    assert_contains_none(
        "bitplane encoder facade",
        &root,
        &[
            "struct ClassicTier1TokenReader",
            "struct SegmentedCodeBlockEncoder",
            "fn significance_propagation_pass(",
            "fn segment_distortion_delta(",
            "for coding_pass in 0..total_passes",
        ],
    );
    assert_contains_all(
        "Tier-1 token packing",
        &tokens,
        &[
            "pub(crate) fn pack_classic_selective_bypass_tier1_tokens",
            "struct ClassicTier1TokenReader",
        ],
    );
    assert_contains_all(
        "Tier-1 pass kernels",
        &passes,
        &[
            "pub(super) fn significance_propagation_pass",
            "pub(super) fn magnitude_refinement_pass",
            "pub(super) fn cleanup_pass",
            "fn encode_sign<",
        ],
    );
    assert_contains_all(
        "Tier-1 segment scheduling",
        &segments,
        &[
            "pub(super) fn encode_segmented_code_block",
            "struct SegmentedCodeBlockEncoder",
            "fn begin_segment_for_pass",
            "fn finish_current_segment",
        ],
    );
    assert_contains_all(
        "Tier-1 PCRD accounting",
        &distortion,
        &[
            "pub(super) fn segment_distortion_delta",
            "fn coefficient_distortion_after_passes",
            "fn reconstructed_magnitude_after_passes",
        ],
    );
}
