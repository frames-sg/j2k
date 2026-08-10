// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2c::build::{Layer, Segment};
use crate::j2c::decode::DecodeAllocationBudget;
use crate::j2c::rect::IntRect;

#[test]
fn collector_uses_the_selected_non_empty_ht_set() {
    let block = CodeBlock {
        rect: IntRect::from_xywh(0, 0, 1, 1),
        x_idx: 0,
        y_idx: 0,
        layers: 0..1,
        has_been_included: true,
        missing_bit_planes: 2,
        number_of_coding_passes: 3,
        ht_total_coding_passes: 6,
        ht_first_cleanup_pass: Some(0),
        ht_selected_set: Some(1),
        coding: Some(crate::j2c::build::CodeBlockCoding::HighThroughput),
        l_block: 3,
        non_empty_layer_count: 1,
    };
    let mut storage = DecompositionStorage::default();
    storage.layers.push(Layer {
        segments: Some(0..4),
    });
    storage.segments.extend([
        Segment {
            idx: 0,
            coding_pases: 1,
            data_length: 1,
            data: &[0x10],
        },
        Segment {
            idx: 1,
            coding_pases: 2,
            data_length: 1,
            data: &[0x20],
        },
        Segment {
            idx: 2,
            coding_pases: 1,
            data_length: 2,
            data: &[0x30, 0x31],
        },
        Segment {
            idx: 3,
            coding_pases: 2,
            data_length: 1,
            data: &[0x40],
        },
    ]);

    let mut selected = Vec::new();
    visit_code_block_segments(&block, &storage, |kind, data| {
        selected.push((kind, data));
        Ok(())
    })
    .unwrap();

    assert_eq!(
        selected,
        [
            (HtCodeBlockSegmentKind::Cleanup, &[0x30, 0x31][..]),
            (HtCodeBlockSegmentKind::Refinement, &[0x40][..]),
        ]
    );
}

#[test]
fn referenced_collector_visits_refinement_fragments_in_packet_order() {
    let block = CodeBlock {
        rect: IntRect::from_xywh(0, 0, 1, 1),
        x_idx: 0,
        y_idx: 0,
        layers: 0..2,
        has_been_included: true,
        missing_bit_planes: 1,
        number_of_coding_passes: 3,
        ht_total_coding_passes: 3,
        ht_first_cleanup_pass: Some(0),
        ht_selected_set: Some(0),
        coding: Some(crate::j2c::build::CodeBlockCoding::HighThroughput),
        l_block: 3,
        non_empty_layer_count: 2,
    };
    let mut storage = DecompositionStorage::default();
    storage.layers.extend([
        Layer {
            segments: Some(0..2),
        },
        Layer {
            segments: Some(2..3),
        },
    ]);
    storage.segments.extend([
        Segment {
            idx: 0,
            coding_pases: 1,
            data_length: 2,
            data: &[0x10, 0x11],
        },
        Segment {
            idx: 1,
            coding_pases: 1,
            data_length: 1,
            data: &[0x20],
        },
        Segment {
            idx: 1,
            coding_pases: 1,
            data_length: 2,
            data: &[0x21, 0x22],
        },
    ]);
    let mut selected = Vec::new();

    visit_code_block_segments(&block, &storage, |kind, data| {
        selected.push((kind, data));
        Ok(())
    })
    .unwrap();

    assert_eq!(
        selected,
        [
            (HtCodeBlockSegmentKind::Cleanup, &[0x10, 0x11][..]),
            (HtCodeBlockSegmentKind::Refinement, &[0x20][..]),
            (HtCodeBlockSegmentKind::Refinement, &[0x21, 0x22][..]),
        ]
    );
}

#[test]
fn owned_collector_concatenates_refinement_fragments_in_packet_order() {
    let block = CodeBlock {
        rect: IntRect::from_xywh(0, 0, 1, 1),
        x_idx: 0,
        y_idx: 0,
        layers: 0..2,
        has_been_included: true,
        missing_bit_planes: 1,
        number_of_coding_passes: 3,
        ht_total_coding_passes: 3,
        ht_first_cleanup_pass: Some(0),
        ht_selected_set: Some(0),
        coding: Some(crate::j2c::build::CodeBlockCoding::HighThroughput),
        l_block: 3,
        non_empty_layer_count: 2,
    };
    let mut storage = DecompositionStorage::default();
    storage.layers.extend([
        Layer {
            segments: Some(0..2),
        },
        Layer {
            segments: Some(2..3),
        },
    ]);
    storage.segments.extend([
        Segment {
            idx: 0,
            coding_pases: 1,
            data_length: 2,
            data: &[0x10, 0x11],
        },
        Segment {
            idx: 1,
            coding_pases: 1,
            data_length: 1,
            data: &[0x20],
        },
        Segment {
            idx: 1,
            coding_pases: 1,
            data_length: 2,
            data: &[0x21, 0x22],
        },
    ]);
    let mut budget = DecodeAllocationBudget::for_storage(&storage).unwrap();

    let selected = collect_code_block_data(&block, &storage, &mut budget).unwrap();

    assert_eq!(selected.cleanup_length, 2);
    assert_eq!(selected.refinement_length, 3);
    assert_eq!(selected.data, [0x10, 0x11, 0x20, 0x21, 0x22]);
}
