// SPDX-License-Identifier: Apache-2.0

use signinum_jpeg::bench_support::bench_profile_fast420_tile_batch;

#[test]
fn profiles_baseline_420_fast_tile_batch() {
    let bytes = include_bytes!("../fixtures/conformance/baseline_420_16x16.jpg");

    let profile = bench_profile_fast420_tile_batch(bytes, 3)
        .expect("profile should not fail")
        .expect("baseline 4:2:0 fixture should use fast tile path");

    assert_eq!(profile.tile_count(), 3);
    assert!(profile.total_ns() > 0);
    assert!(profile.parse_plan_ns() > 0);
    assert!(profile.mcu_decode_ns() > 0);
    assert!(profile.rgb_emit_ns() > 0);

    let counts = profile.block_activity_counts();
    assert_eq!(counts.total_blocks(), 18);
    assert_eq!(
        counts.total_blocks(),
        counts.dc_only_blocks() + counts.bottom_half_zero_blocks() + counts.general_blocks()
    );
}

#[test]
fn skips_non_fast_tile_inputs_without_error() {
    let bytes = include_bytes!("../fixtures/conformance/grayscale_8x8.jpg");

    let profile = bench_profile_fast420_tile_batch(bytes, 1).expect("profile should not fail");

    assert!(profile.is_none());
}
