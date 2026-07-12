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

struct BitplaneSources {
    root: String,
    tokens: String,
    token_reader: String,
    passes: String,
    preparation: String,
    segments: String,
    segment_encoder: String,
    distortion: String,
}

impl BitplaneSources {
    fn read() -> Self {
        Self {
            root: read("src/j2c/bitplane_encode.rs"),
            tokens: read("src/j2c/bitplane_encode/tokens.rs"),
            token_reader: read("src/j2c/bitplane_encode/tokens/reader.rs"),
            passes: read("src/j2c/bitplane_encode/passes.rs"),
            preparation: read("src/j2c/bitplane_encode/preparation.rs"),
            segments: read("src/j2c/bitplane_encode/segments.rs"),
            segment_encoder: read("src/j2c/bitplane_encode/segments/encoder.rs"),
            distortion: read("src/j2c/bitplane_encode/distortion.rs"),
        }
    }

    fn assert_boundaries(&self) {
        for (path, source) in [
            ("bitplane_encode.rs", self.root.as_str()),
            ("bitplane_encode/tokens.rs", self.tokens.as_str()),
            (
                "bitplane_encode/tokens/reader.rs",
                self.token_reader.as_str(),
            ),
            ("bitplane_encode/passes.rs", self.passes.as_str()),
            ("bitplane_encode/preparation.rs", self.preparation.as_str()),
            ("bitplane_encode/segments.rs", self.segments.as_str()),
            (
                "bitplane_encode/segments/encoder.rs",
                self.segment_encoder.as_str(),
            ),
            ("bitplane_encode/distortion.rs", self.distortion.as_str()),
        ] {
            assert_line_budget(path, source, 700);
            assert_contains_none(path, source, &["use super::*", "include!(", "#[allow("]);
        }
    }

    fn assert_facade(&self) {
        assert_contains_all(
            "bitplane encoder facade",
            &self.root,
            &[
                "mod tokens;",
                "mod passes;",
                "mod preparation;",
                "mod segments;",
                "mod distortion;",
                "pub(crate) use self::tokens::",
                "segments::try_encode_segmented_code_block(",
            ],
        );
        assert_contains_none(
            "bitplane encoder facade",
            &self.root,
            &[
                "struct ClassicTier1TokenReader",
                "struct SegmentedCodeBlockEncoder",
                "fn significance_propagation_pass(",
                "fn try_prepare_padded_coefficients_from_view",
                "fn segment_distortion_delta(",
                "for coding_pass in 0..total_passes",
            ],
        );
    }

    fn assert_module_ownership(&self) {
        assert_contains_all(
            "Tier-1 token packing",
            &self.tokens,
            &[
                "pub(crate) fn pack_classic_selective_bypass_tier1_tokens",
                "mod reader;",
                "use reader::ClassicTier1TokenReader;",
            ],
        );
        assert_contains_all(
            "Tier-1 token reader",
            &self.token_reader,
            &["pub(super) struct ClassicTier1TokenReader", "fn read_bits"],
        );
        assert_contains_all(
            "Tier-1 coefficient preparation",
            &self.preparation,
            &[
                "pub(super) fn try_prepare_padded_coefficients_from_view",
                "try_untracked_vec_filled",
                "coefficient.unsigned_magnitude()",
            ],
        );
        assert_contains_all(
            "Tier-1 pass kernels",
            &self.passes,
            &[
                "pub(super) fn significance_propagation_pass",
                "pub(super) fn magnitude_refinement_pass",
                "pub(super) fn cleanup_pass",
                "fn encode_sign<",
            ],
        );
        assert_contains_all(
            "Tier-1 segment scheduling",
            &self.segments,
            &[
                "pub(super) fn try_encode_segmented_code_block",
                "mod encoder;",
                "use encoder::SegmentedCodeBlockEncoder;",
            ],
        );
        assert_contains_all(
            "Tier-1 segment encoder",
            &self.segment_encoder,
            &[
                "struct SegmentedCodeBlockEncoder",
                "fn try_begin_segment_for_pass",
                "fn try_finish_current_segment",
            ],
        );
        assert_contains_all(
            "Tier-1 PCRD accounting",
            &self.distortion,
            &[
                "pub(super) fn segment_distortion_delta",
                "fn coefficient_distortion_after_passes",
                "fn reconstructed_magnitude_after_passes",
            ],
        );
    }
}

#[test]
fn classic_bitplane_encoder_stays_split_by_responsibility() {
    let sources = BitplaneSources::read();
    sources.assert_boundaries();
    sources.assert_facade();
    sources.assert_module_ownership();
}
