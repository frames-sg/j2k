mod fixtures;

use slidecodec_jpeg::__private::metal_fast420::{
    build_metal_fast420_packet, MetalFast420PacketError,
};

#[test]
fn baseline_420_fixture_builds_fast420_packet() {
    let bytes = fixtures::minimal_baseline_420_jpeg();
    let packet = build_metal_fast420_packet(&bytes).expect("fast420 packet");

    assert_eq!(packet.dimensions, (16, 16));
    assert_eq!(packet.mcus_per_row, 1);
    assert_eq!(packet.mcu_rows, 1);
    assert!(!packet.entropy_bytes.is_empty());
    assert!(packet.y_dc_table.values_len > 0);
    assert!(packet.y_ac_table.values_len > 0);
    assert!(packet.cb_dc_table.values_len > 0);
    assert!(packet.cb_ac_table.values_len > 0);
    assert!(packet.cr_dc_table.values_len > 0);
    assert!(packet.cr_ac_table.values_len > 0);
}

#[test]
fn grayscale_fixture_is_rejected_for_fast420_subset() {
    let bytes = fixtures::grayscale_8x8_jpeg();
    let error = build_metal_fast420_packet(&bytes).expect_err("grayscale must be rejected");

    assert!(matches!(
        error,
        MetalFast420PacketError::UnsupportedColorSpace(_)
            | MetalFast420PacketError::UnsupportedSampling
    ));
}

#[test]
fn progressive_fixture_is_rejected_for_fast420_subset() {
    let bytes = fixtures::progressive_8x8_jpeg();
    let error = build_metal_fast420_packet(&bytes).expect_err("progressive must be rejected");

    assert!(matches!(error, MetalFast420PacketError::UnsupportedSof(_)));
}
