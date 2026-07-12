// SPDX-License-Identifier: MIT OR Apache-2.0

use super::support::{
    baseline_grayscale_jpeg, insert_restart_interval, restart_coded_grayscale_jpeg, ColorSpace,
    Cow, Decoder, BASELINE_420, BASELINE_422, BASELINE_444,
};

#[test]
fn adapter_device_plan_exposes_scan_metadata() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = j2k_jpeg::adapter::build_device_plan(&decoder, 4).expect("device plan");

    assert_eq!(plan.dimensions, (16, 16));
    assert_eq!(plan.color_space, ColorSpace::YCbCr);
    assert_eq!(plan.components.len(), 3);
    assert_eq!(plan.checkpoints[0].mcu_index, 0);
    assert!(!plan.scan_bytes.is_empty());
}

#[test]
fn adapter_device_plan_borrows_scan_bytes_for_well_formed_streams() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = j2k_jpeg::adapter::build_device_plan(&decoder, 4).expect("device plan");

    assert!(matches!(plan.scan_bytes, Cow::Borrowed(_)));
}

#[test]
fn adapter_device_plan_keeps_fast_420_shape_information() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = j2k_jpeg::adapter::build_device_plan(&decoder, 4).expect("device plan");

    assert!(plan.matches_fast_420);
    assert!(!plan.matches_fast_422);
    assert!(!plan.matches_fast_444);
}

#[test]
fn adapter_device_plan_keeps_fast_422_shape_information() {
    let decoder = Decoder::new(BASELINE_422).expect("decoder");
    let plan = j2k_jpeg::adapter::build_device_plan(&decoder, 4).expect("device plan");

    assert!(!plan.matches_fast_420);
    assert!(plan.matches_fast_422);
    assert!(!plan.matches_fast_444);
}

#[test]
fn adapter_device_plan_keeps_fast_444_shape_information() {
    let decoder = Decoder::new(BASELINE_444).expect("decoder");
    let plan = j2k_jpeg::adapter::build_device_plan(&decoder, 4).expect("device plan");

    assert!(!plan.matches_fast_420);
    assert!(!plan.matches_fast_422);
    assert!(plan.matches_fast_444);
}

#[test]
fn adapter_device_plan_scan_bytes_keep_terminal_eoi() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = j2k_jpeg::adapter::build_device_plan(&decoder, 4).expect("device plan");

    assert!(plan.scan_bytes.ends_with(&[0xff, 0xd9]));
}

#[test]
fn adapter_device_plan_checkpoint_cadence_handles_multi_mcu_inputs() {
    let bytes = baseline_grayscale_jpeg(24, 24);
    let decoder = Decoder::new(&bytes).expect("grayscale decoder");

    let cadence_zero =
        j2k_jpeg::adapter::build_device_plan(&decoder, 0).expect("zero-cadence plan");
    let cadence_two = j2k_jpeg::adapter::build_device_plan(&decoder, 2).expect("cadence-two plan");

    assert_eq!(
        cadence_zero
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.mcu_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8]
    );
    let zero_offsets = cadence_zero
        .checkpoints
        .iter()
        .map(|checkpoint| checkpoint.scan_offset)
        .collect::<Vec<_>>();
    assert_eq!(zero_offsets.first(), Some(&0));
    assert!(zero_offsets.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(
        cadence_two
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.mcu_index)
            .collect::<Vec<_>>(),
        vec![0, 2, 4, 6, 8]
    );
    let cadence_two_offsets = cadence_two
        .checkpoints
        .iter()
        .map(|checkpoint| checkpoint.scan_offset)
        .collect::<Vec<_>>();
    assert_eq!(cadence_two_offsets.first(), Some(&0));
    assert!(cadence_two_offsets
        .windows(2)
        .all(|pair| pair[0] <= pair[1]));
    assert!(cadence_two
        .checkpoints
        .iter()
        .all(|checkpoint| checkpoint.bits_buffered <= 64 && checkpoint.expected_rst == 0));
}

#[test]
fn adapter_device_plan_restart_checkpoints_capture_resume_state() {
    let bytes = restart_coded_grayscale_jpeg(24, 24);
    let decoder = Decoder::new(&bytes).expect("restart-coded decoder");
    let plan = j2k_jpeg::adapter::build_device_plan(&decoder, 2).expect("device plan");

    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.mcu_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.scan_offset)
            .collect::<Vec<_>>(),
        vec![0, 3, 6, 9, 12, 15, 18, 21, 24]
    );
    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.expected_rst)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 0]
    );
    assert!(plan
        .checkpoints
        .iter()
        .all(|checkpoint| checkpoint.bits_buffered == 0 && checkpoint.prev_dc == [0; 4]));
}

#[test]
fn adapter_device_plan_treats_dri_zero_as_non_restart_fast_path() {
    let bytes = insert_restart_interval(BASELINE_420.to_vec(), 0);
    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = j2k_jpeg::adapter::build_device_plan(&decoder, 2).expect("device plan");

    assert_eq!(plan.restart_interval, None);
    assert!(plan.matches_fast_420);
    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.expected_rst)
            .collect::<Vec<_>>(),
        vec![0; plan.checkpoints.len()]
    );
}

#[test]
fn adapter_device_plan_handles_restart_after_partial_entropy_byte() {
    let bytes = restart_coded_grayscale_jpeg(16, 8);
    let decoder = Decoder::new(&bytes).expect("restart-coded decoder");
    let plan = j2k_jpeg::adapter::build_device_plan(&decoder, 2).expect("device plan");

    assert_eq!(plan.checkpoints.len(), 2);
    assert_eq!(plan.checkpoints[1].mcu_index, 1);
    assert_eq!(plan.checkpoints[1].scan_offset, 3);
    assert_eq!(plan.checkpoints[1].expected_rst, 1);
}
