// SPDX-License-Identifier: MIT OR Apache-2.0

//! Responsibility and regression ratchets for facade decode orchestration.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn facade_decode_keeps_pixel_layout_conversion_in_explicit_children() {
    let root = read("crates/j2k/src/decode.rs");
    let output = read_source_files(
        repo_root(),
        &[
            "crates/j2k/src/decode/output.rs",
            "crates/j2k/src/decode/output/u8.rs",
            "crates/j2k/src/decode/output/u16.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("facade decode orchestrator", &root)
            .required(&[
                "mod component_handoff;",
                "mod output;",
                "decode_image_into_with_native_context",
                "decode_image_region_into_with_native_context",
                "decode_warnings_for_settings",
            ])
            .forbidden(&[
                "fn write_u8_output(",
                "fn write_u16_output(",
                "fn write_components_u8_output(",
                "fn convert_or_copy_u16(",
                "include!(",
            ]),
        PatternCheck::new("facade decode output modules", &output)
            .required(&[
                "mod u16;",
                "mod u8;",
                "pub(in crate::decode) fn write_u8_output(",
                "pub(in crate::decode) fn write_u16_output(",
                "direct_u8_decode_accepts_exact_rgb_and_gray_layouts",
                "eight_bit_samples_widen_across_the_complete_u16_domain",
                "synthesized_alpha_matches_native_sample_storage",
            ])
            .forbidden(&["use super::*", "include!("]),
    ]);
}

#[test]
fn facade_decode_responsibility_modules_stay_focused() {
    for (relative, max_lines) in [
        ("crates/j2k/src/decode.rs", 220),
        ("crates/j2k/src/decode/output.rs", 25),
        ("crates/j2k/src/decode/output/u8.rs", 375),
        ("crates/j2k/src/decode/output/u16.rs", 260),
    ] {
        let lines = read(relative).lines().count();
        assert!(
            lines < max_lines,
            "{relative} must stay below its {max_lines}-line responsibility ratchet"
        );
    }
}
