// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::adapter::fast_packet::checkpoints::{
    build_fast_entropy_checkpoints, inspect_fast_entropy_checkpoints,
    packet_checkpoints_from_device,
};
use crate::adapter::fast_packet::{FastPacketError, JpegEntropyCheckpointV1};
use crate::decoder::Decoder;
use crate::error::JpegError;
use crate::internal::checkpoint::{
    build_checkpoint_plan_from_validated_with_live_budget, total_mcus, validate_scan_bytes,
    DeviceCheckpoint,
};

#[test]
fn packet_checkpoint_conversion_is_fallible_and_preserves_values() {
    let checkpoint = DeviceCheckpoint {
        mcu_index: 7,
        scan_offset: 0,
        bit_accumulator: 0x8000_0000_0000_0000,
        bits_buffered: 1,
        prev_dc: [11, 22, 33, 44],
        expected_rst: 0,
    };
    let packet_bytes = core::mem::size_of::<JpegEntropyCheckpointV1>();
    let converted = packet_checkpoints_from_device(&[checkpoint], &[], packet_bytes)
        .expect("one checkpoint fits the cap");
    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].mcu_index, checkpoint.mcu_index);
    assert_eq!(converted[0].bit_acc, checkpoint.bit_accumulator);
    assert_eq!(converted[0].bit_count, 1);
    assert_eq!(converted[0].y_prev_dc, 11);
    assert_eq!(converted[0].cb_prev_dc, 22);
    assert_eq!(converted[0].cr_prev_dc, 33);

    let error = packet_checkpoints_from_device(&[checkpoint, checkpoint], &[], packet_bytes)
        .expect_err("two checkpoints exceed a one-checkpoint cap");
    assert!(matches!(
        error,
        FastPacketError::Decode(JpegError::MemoryCapExceeded { requested, cap })
            if requested == packet_bytes * 2 && cap == packet_bytes
    ));

    let invalid_offset_checkpoint = DeviceCheckpoint {
        scan_offset: 1,
        ..checkpoint
    };
    let error = packet_checkpoints_from_device(&[invalid_offset_checkpoint], &[0xff], packet_bytes)
        .expect_err("checkpoint conversion must preserve entropy-offset failures");
    assert_eq!(error, FastPacketError::TruncatedEntropy);
}

#[test]
fn packet_checkpoint_conversion_destuffs_ordered_offsets_in_one_forward_pass() {
    let scan_bytes = [0x11, 0xff, 0x00, 0x22, 0xff, 0xd0, 0x33, 0xff, 0xd9];
    let source_offsets = [0usize, 1, 3, 4, 6, 7, 9];
    let checkpoints: Vec<_> = source_offsets
        .into_iter()
        .enumerate()
        .map(|(mcu_index, scan_offset)| DeviceCheckpoint {
            mcu_index: u32::try_from(mcu_index).expect("test MCU index fits u32"),
            scan_offset,
            bit_accumulator: 0,
            bits_buffered: 0,
            prev_dc: [0; 4],
            expected_rst: 0,
        })
        .collect();
    let allocation_cap = checkpoints.len() * core::mem::size_of::<JpegEntropyCheckpointV1>();

    let converted = packet_checkpoints_from_device(&checkpoints, &scan_bytes, allocation_cap)
        .expect("ordered offsets should be converted monotonically");
    assert_eq!(
        converted
            .iter()
            .map(|checkpoint| checkpoint.entropy_pos)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 3, 4, 4]
    );

    let unordered = [checkpoints[2], checkpoints[1]];
    let error = packet_checkpoints_from_device(&unordered, &scan_bytes, allocation_cap)
        .expect_err("backtracking source offsets must fail closed");
    assert_eq!(error, FastPacketError::TruncatedEntropy);
}

#[test]
fn direct_packet_checkpoints_match_device_checkpoint_conversion() {
    let bytes = j2k_test_support::baseline_420_restart_32x16_jpeg();
    let decoder = Decoder::new(&bytes).expect("fixture decoder");
    let scan = &decoder.bytes[decoder.plan.scan_offset..];
    let validated =
        validate_scan_bytes(scan, true, decoder.plan.scan_offset).expect("fixture scan validation");
    let layout = inspect_fast_entropy_checkpoints(&decoder, total_mcus(&decoder.plan));
    let cap = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

    let mut direct_live_bytes = 0;
    let direct =
        build_fast_entropy_checkpoints(&decoder, validated, layout, &mut direct_live_bytes, cap)
            .expect("direct packet checkpoints");
    let mut device_live_bytes = 0;
    let device = build_checkpoint_plan_from_validated_with_live_budget(
        &decoder.plan,
        validated,
        layout.cadence_mcus,
        &mut device_live_bytes,
        cap,
    )
    .expect("device checkpoints");
    let converted = packet_checkpoints_from_device(&device, validated.payload(), cap)
        .expect("legacy parity conversion");
    assert_eq!(direct, converted);
}
