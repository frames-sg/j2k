mod fixtures;

use slidecodec_jpeg::__private::metal_fast420::{
    build_metal_fast420_packet, MetalFast420PacketError,
};

fn rewrite_component_ids_to_zero_based(bytes: &mut [u8]) {
    let mut pos = 2usize;
    while pos + 4 <= bytes.len() {
        if bytes[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = bytes[pos + 1];
        if matches!(marker, 0xD8 | 0xD9) {
            pos += 2;
            continue;
        }
        let len = u16::from_be_bytes([bytes[pos + 2], bytes[pos + 3]]) as usize;
        match marker {
            0xC0 | 0xC1 => {
                let components = bytes[pos + 9] as usize;
                let mut component_pos = pos + 10;
                for next_id in 0..components {
                    bytes[component_pos] = next_id as u8;
                    component_pos += 3;
                }
            }
            0xDA => {
                let components = bytes[pos + 4] as usize;
                let mut component_pos = pos + 5;
                for next_id in 0..components {
                    bytes[component_pos] = next_id as u8;
                    component_pos += 2;
                }
                return;
            }
            _ => {}
        }
        pos += 2 + len;
    }
}

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
fn baseline_420_fixture_with_zero_based_component_ids_builds_fast420_packet() {
    let mut bytes = fixtures::minimal_baseline_420_jpeg();
    rewrite_component_ids_to_zero_based(&mut bytes);

    let packet = build_metal_fast420_packet(&bytes).expect("fast420 packet");
    assert_eq!(packet.dimensions, (16, 16));
    assert_eq!(packet.mcus_per_row, 1);
    assert_eq!(packet.mcu_rows, 1);
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
