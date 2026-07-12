// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::{Decoder, JpegError};

fn push_segment(bytes: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    let length = u16::try_from(payload.len() + 2).expect("test segment length fits u16");
    bytes.extend_from_slice(&[0xff, marker]);
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(payload);
}

fn push_quant_table(bytes: &mut Vec<u8>, slot: u8, value: u8) {
    let mut payload = Vec::with_capacity(65);
    payload.push(slot);
    payload.extend(core::iter::repeat_n(value, 64));
    push_segment(bytes, 0xdb, &payload);
}

fn push_progressive_sof(bytes: &mut Vec<u8>, quant_slots: &[u8]) {
    let mut payload = Vec::with_capacity(6 + quant_slots.len() * 3);
    payload.extend_from_slice(&[8, 0, 8, 0, 8]);
    payload.push(u8::try_from(quant_slots.len()).expect("test component count fits u8"));
    for (index, &quant_slot) in quant_slots.iter().enumerate() {
        payload.push(u8::try_from(index + 1).expect("test component id fits u8"));
        payload.extend_from_slice(&[0x11, quant_slot]);
    }
    push_segment(bytes, 0xc2, &payload);
}

fn push_minimal_huffman_tables(bytes: &mut Vec<u8>) {
    let mut payload = Vec::with_capacity(36);
    for class_and_slot in [0x00, 0x10] {
        payload.push(class_and_slot);
        payload.push(1);
        payload.extend(core::iter::repeat_n(0, 15));
        payload.push(0);
    }
    push_segment(bytes, 0xc4, &payload);
}

fn push_scan(bytes: &mut Vec<u8>, components: &[u8], ss: u8, se: u8, ah: u8, al: u8) -> usize {
    let mut payload = Vec::with_capacity(4 + components.len() * 2);
    payload.push(u8::try_from(components.len()).expect("test scan count fits u8"));
    for &component in components {
        payload.extend_from_slice(&[component, 0]);
    }
    payload.extend_from_slice(&[ss, se, (ah << 4) | al]);
    push_segment(bytes, 0xda, &payload);
    let entropy_offset = bytes.len();
    bytes.push(0);
    entropy_offset
}

fn valid_redefinition_fixture() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    push_quant_table(&mut bytes, 0, 1);
    push_quant_table(&mut bytes, 1, 2);
    push_progressive_sof(&mut bytes, &[0, 1, 1]);
    push_minimal_huffman_tables(&mut bytes);

    push_scan(&mut bytes, &[1], 0, 0, 0, 0);
    // Component 1 has finished using slot 0; its latched value must remain 1.
    push_quant_table(&mut bytes, 0, 5);
    // Components 2 and 3 have not appeared yet and must latch this definition.
    push_quant_table(&mut bytes, 1, 7);
    push_scan(&mut bytes, &[2, 3], 0, 0, 0, 0);
    // A byte-identical version remains the same resolved table for component 2.
    push_quant_table(&mut bytes, 1, 7);
    // Slot 2 is unused by every frame component.
    push_quant_table(&mut bytes, 2, 9);
    push_scan(&mut bytes, &[2], 1, 63, 0, 0);
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

#[test]
fn valid_redefinitions_bind_each_component_at_first_scan() {
    let bytes = valid_redefinition_fixture();
    let decoder = Decoder::new(&bytes).expect("valid progressive DQT lifecycle");
    let progressive = decoder.progressive_plan.as_ref().expect("progressive plan");

    assert_eq!(progressive.components[0].quant, [1; 64]);
    assert_eq!(progressive.components[1].quant, [7; 64]);
    assert_eq!(progressive.components[2].quant, [7; 64]);
    assert_eq!(decoder.plan.components[0].quant, [1; 64]);
    assert_eq!(decoder.plan.components[1].quant, [7; 64]);
}

#[test]
fn quant_redefinition_after_component_latch_is_rejected() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    push_quant_table(&mut bytes, 0, 1);
    push_progressive_sof(&mut bytes, &[0]);
    push_minimal_huffman_tables(&mut bytes);
    push_scan(&mut bytes, &[1], 0, 0, 0, 0);
    push_quant_table(&mut bytes, 0, 5);
    let offending_offset = push_scan(&mut bytes, &[1], 1, 63, 0, 0);
    bytes.extend_from_slice(&[0xff, 0xd9]);

    let error = Decoder::new(&bytes).expect_err("post-latch DQT change must be rejected");
    assert_eq!(
        error,
        JpegError::ProgressiveQuantTableChanged {
            offset: offending_offset,
            component: 1,
            table_id: 0,
        }
    );
    assert_eq!(error.offset(), Some(offending_offset));
}
