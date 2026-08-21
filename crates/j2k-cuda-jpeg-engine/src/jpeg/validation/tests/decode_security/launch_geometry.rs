// SPDX-License-Identifier: MIT OR Apache-2.0

use super::decode_plan_for;
use crate::{
    jpeg::validation::{decode_plan::jpeg_checkpoint_launch_geometry, validate_jpeg_rgb8_plan},
    CudaJpegEntropyCheckpoint, CudaJpegRgb8Sampling,
};

#[test]
fn adaptive_geometry_keeps_small_counts_serial_and_packs_at_128() {
    for checkpoint_count in [1, 16, 64, 127] {
        let geometry = jpeg_checkpoint_launch_geometry(checkpoint_count)
            .expect("adaptive serial checkpoint launch geometry");
        assert_eq!(geometry.grid(), (checkpoint_count, 1, 1));
        assert_eq!(geometry.block(), (1, 1, 1));
    }
    for (checkpoint_count, expected_grid) in [(128, 1), (129, 2)] {
        let geometry = jpeg_checkpoint_launch_geometry(checkpoint_count)
            .expect("adaptive packed checkpoint launch geometry");
        assert_eq!(geometry.grid(), (expected_grid, 1, 1));
        assert_eq!(geometry.block(), (128, 1, 1));
    }
}

#[test]
fn packed_geometry_covers_tail_and_largest_abi_count() {
    let tail = jpeg_checkpoint_launch_geometry(128 * 17 + 1).expect("packed tail launch geometry");
    assert_eq!(tail.grid(), (18, 1, 1));
    assert_eq!(tail.block(), (128, 1, 1));

    let large = jpeg_checkpoint_launch_geometry(u32::MAX)
        .expect("largest ABI checkpoint count remains launchable when packed");
    assert_eq!(large.grid(), (u32::MAX.div_ceil(128), 1, 1));
    assert_eq!(large.block(), (128, 1, 1));
}

#[test]
fn validation_preserves_one_status_slot_per_checkpoint() {
    let checkpoints = (0..129)
        .map(|index| CudaJpegEntropyCheckpoint {
            mcu_index: index,
            entropy_pos: index,
            ..CudaJpegEntropyCheckpoint::default()
        })
        .collect::<Vec<_>>();
    let entropy = [0; 129];
    let mut plan = decode_plan_for(CudaJpegRgb8Sampling::Fast444, (129 * 8, 8), &checkpoints);
    plan.entropy_bytes = &entropy;

    let validated = validate_jpeg_rgb8_plan(&plan).expect("packed checkpoint validation");
    assert_eq!(validated.params.checkpoint_count, 129);
    assert_eq!(validated.geometry.grid(), (2, 1, 1));
    assert_eq!(validated.geometry.block(), (128, 1, 1));
}
