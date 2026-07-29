// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free tile-part cursor ratchets.

use super::read;
use crate::repo_lint_support::{assert_pattern_checks, PatternCheck};

#[test]
fn tile_part_decode_uses_a_borrowing_nonallocating_cursor() {
    let cursor = read("crates/j2k-native/src/j2c/tile/cursor.rs");
    let production_cursor = cursor
        .split("#[cfg(test)]")
        .next()
        .expect("cursor production prefix");
    let tile = read("crates/j2k-native/src/j2c/tile.rs");
    let segment = read("crates/j2k-native/src/j2c/segment.rs");
    let coverage = format!(
        "{}\n{}",
        read("crates/j2k-native/src/j2c/tile/cursor/tests.rs"),
        read("crates/j2k-native/src/tests.rs")
    );

    assert!(
        cursor.lines().count() <= 160,
        "tile-part cursor must stay focused"
    );
    assert_pattern_checks(&[
        PatternCheck::new("borrowing tile-part cursor", production_cursor)
            .required(&[
                "pub(crate) enum TilePartCursor<'part, 'data>",
                "retained_headers: &'part [BitReader<'data>]",
                "packet_lengths: PacketLengthCursor<'part>",
                "header: part.headers.first()?.clone()",
                "body: part.body.clone()",
                "lengths: &'part [u32]",
                "while header.at_end()",
                "validate_all_packet_lengths_consumed",
            ])
            .forbidden(&["Vec<", "Vec::", ".to_vec()", ".collect"]),
        PatternCheck::new("immutable retained tile-part state", &tile).forbidden(&[
            "active_header_reader: usize",
            "next: usize,\n}",
            "impl<'a> TilePart<'a> {\n    pub(crate) fn header(&mut self)",
        ]),
        PatternCheck::new("cursor-based packet segmentation", &segment)
            .required(&[
                "tile_part.cursor().and_then(|cursor|",
                "mut tile_part: TilePartCursor<'_, 'a>",
            ])
            .forbidden(&["tile_part.clone()", "mut tile_part: TilePart<'a>"]),
        PatternCheck::new("tile-part cursor behavior coverage", &coverage).required(&[
            "separated_ppm_ppt_cursor_spans_multiple_readers_without_mutating_retained_state",
            "plt_length_cursor_validates_each_packet_and_resets_for_reuse",
            "repeated_decode_and_recode_reuse_immutable_tile_part_metadata",
            "repeated_direct_plan_build_reuses_immutable_tile_part_metadata",
        ]),
    ]);
}
