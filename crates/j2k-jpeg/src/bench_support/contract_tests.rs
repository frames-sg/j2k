// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Bench420DispatchStats, BenchBlockActivityCounts, BenchColorRowScratch, BenchFast420Profile,
    BenchRgb420RowPairScratch, BenchUpsampleH2V2Scratch,
};

#[test]
fn profile_and_dispatch_accessors_preserve_exact_accounting() {
    let stats = Bench420DispatchStats::default();
    assert_eq!(stats.scalar_chunks(), 0);
    assert_eq!(stats.neon_tail_chunks(), 0);

    let mut blocks = BenchBlockActivityCounts::default();
    blocks.record_dc_only();
    blocks.record_bottom_half_zero();
    blocks.record_general();
    assert_eq!(blocks.total_blocks(), 3);
    assert_eq!(blocks.dc_only_blocks(), 1);
    assert_eq!(blocks.bottom_half_zero_blocks(), 1);
    assert_eq!(blocks.general_blocks(), 1);

    let mut profile = BenchFast420Profile::default();
    profile.set_total_ns(101);
    profile.set_tile_count(7);
    profile.add_parse_plan_ns(11);
    profile.add_mcu_decode_ns(13);
    profile.add_rgb_emit_ns(17);
    profile.add_finish_ns(19);
    *profile.block_activity_counts_mut() = blocks;
    assert_eq!(profile.total_ns(), 101);
    assert_eq!(profile.parse_plan_ns(), 11);
    assert_eq!(profile.mcu_decode_ns(), 13);
    assert_eq!(profile.rgb_emit_ns(), 17);
    assert_eq!(profile.finish_ns(), 19);
    assert_eq!(profile.tile_count(), 7);
    assert_eq!(profile.block_activity_counts(), blocks);
}

#[test]
fn deterministic_scratch_helpers_match_their_reference_paths() {
    let mut detected_rows = BenchRgb420RowPairScratch::new(17);
    let mut scalar_rows = BenchRgb420RowPairScratch::new(17);
    detected_rows.run();
    scalar_rows.run_reference();
    assert_eq!(detected_rows.top, scalar_rows.top);
    assert_eq!(detected_rows.bottom, scalar_rows.bottom);

    let mut upsample = BenchUpsampleH2V2Scratch::new(9);
    upsample.run();
    assert_eq!(upsample.top.len(), 18);
    assert_eq!(upsample.bot.len(), 18);
    assert!(upsample.top.iter().any(|&sample| sample != 0));
    assert!(upsample.bot.iter().any(|&sample| sample != 0));

    let mut detected_color = BenchColorRowScratch::new(17);
    let mut scalar_color = BenchColorRowScratch::new(17);
    detected_color.run_backend();
    scalar_color.run_scalar();
    assert_eq!(detected_color.rgb, scalar_color.rgb);
}
