// SPDX-License-Identifier: MIT OR Apache-2.0

use super::fixtures::grayscale_jpeg;
use crate::decoder::Decoder;
use crate::internal::checkpoint::{checkpoint_before_mcu, CpuCheckpointCache};

#[test]
fn lazy_non_restart_checkpoint_extends_to_nearest_cadence_before_target() {
    let bytes = grayscale_jpeg(48, 48);
    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = &decoder.plan;
    let scan_bytes = &decoder.bytes[plan.scan_offset..];
    let mut cache = CpuCheckpointCache::default();

    let checkpoint = checkpoint_before_mcu(plan, scan_bytes, 4, 17, 0, &mut cache)
        .expect("checkpoint lookup")
        .expect("target beyond one cadence should produce a checkpoint");
    assert_eq!(checkpoint.mcu_index, 16);
    assert!(cache
        .checkpoints
        .iter()
        .any(|checkpoint| checkpoint.mcu_index == 16));
}
