// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};
use core::mem::size_of;

use super::cdef::ChannelType;
use super::cmap::{ComponentMappingEntry, ComponentMappingType};
use super::colr::{ColorSpace as NativeColorSpace, ColorSpecificationBox};
use super::container::{implicit_mapping_budget, parse_jp2_header_box};
use super::metadata::{public_metadata_from_boxes, Jp2ColorSpec};
use super::{ComponentDescriptor, ImageBoxes};

#[test]
fn repeated_metadata_boxes_replace_old_owners_without_losing_accounting() {
    let mut header = required_header_prefix(&[1, 0, 0, 0, 0, 0, 17]);
    push_child(&mut header, *b"bpcc", &[7]);
    push_child(&mut header, *b"bpcc", &[0x87]);
    push_child(&mut header, *b"pclr", &palette_payload(7));
    push_child(&mut header, *b"pclr", &palette_payload(9));
    push_child(&mut header, *b"cdef", &cdef_payload(0));
    push_child(&mut header, *b"cdef", &cdef_payload(1));
    push_child(&mut header, *b"cmap", &[0, 0, 0, 0]);
    push_child(&mut header, *b"cmap", &[0, 0, 1, 0]);

    let boxes = parse_jp2_header_box(&header, true, 0).expect("repeated metadata parses");
    assert_eq!(
        boxes.bits_per_component,
        [ComponentDescriptor {
            bit_depth: 8,
            signed: true,
        }]
    );
    assert_eq!(
        boxes.palette.as_ref().and_then(|palette| palette.map(0, 0)),
        Some(9)
    );
    assert!(matches!(
        boxes
            .channel_definition
            .as_ref()
            .and_then(|box_| box_.channel_definitions.first())
            .map(|definition| definition.channel_type),
        Some(ChannelType::Opacity)
    ));
    assert!(matches!(
        boxes
            .component_mapping
            .as_ref()
            .and_then(|box_| box_.entries.first())
            .map(|entry| entry.mapping_type),
        Some(ComponentMappingType::Palette { column: 0 })
    ));

    let expected = shallow_allocated_bytes(&boxes);
    assert_eq!(
        boxes.allocated_bytes().expect("metadata byte count"),
        expected
    );
}

#[test]
fn non_strict_mid_palette_error_keeps_the_previous_complete_owner() {
    let mut header = required_header_prefix(&[1, 0, 0, 0, 0, 0, 17]);
    push_child(&mut header, *b"pclr", &palette_payload(7));
    // Two rows and two columns are declared, but the first row is truncated.
    push_child(&mut header, *b"pclr", &[0, 2, 2, 7, 7, 1]);
    push_child(&mut header, *b"cmap", &[0, 0, 1, 0]);

    let boxes = parse_jp2_header_box(&header, false, 0).expect("non-strict palette recovery");
    assert_eq!(
        boxes.palette.as_ref().and_then(|palette| palette.map(0, 0)),
        Some(7)
    );
    assert_eq!(
        boxes
            .component_mapping
            .as_ref()
            .map_or(0, |mapping| mapping.entries.len()),
        1
    );
    assert_eq!(
        boxes.allocated_bytes().expect("metadata byte count"),
        shallow_allocated_bytes(&boxes)
    );
}

#[test]
fn public_conversion_moves_icc_and_palette_payload_owners() {
    let mut header = required_header_prefix(&[2, 0, 0, 1, 2, 3, 4]);
    push_child(&mut header, *b"pclr", &palette_payload(11));
    let boxes = parse_jp2_header_box(&header, true, 0).expect("metadata parses");
    let source_icc_capacity = match &boxes.color_specifications[0].color_space {
        NativeColorSpace::Icc(profile) => profile.capacity(),
        _ => panic!("expected ICC profile"),
    };
    let palette = boxes.palette.as_ref().expect("palette");
    let source_outer_capacity = palette.entries.capacity();
    let source_entry_capacity = palette.entries[0].capacity();

    let metadata = public_metadata_from_boxes(boxes).expect("move conversion");
    let Jp2ColorSpec::IccProfile { profile } = &metadata.color_specs[0] else {
        panic!("expected public ICC profile");
    };
    let palette = metadata.palette.as_ref().expect("public palette");
    assert_eq!(profile.capacity(), source_icc_capacity);
    assert_eq!(palette.entries.capacity(), source_outer_capacity);
    assert_eq!(palette.entries[0].capacity(), source_entry_capacity);
}

#[test]
fn retained_jp2_parse_baseline_covers_nested_icc_and_palette_owners() {
    let mut header = required_header_prefix(&[2, 0, 0, 1, 2, 3, 4]);
    push_child(&mut header, *b"pclr", &palette_payload(11));
    let probe = parse_jp2_header_box(&header, true, 0).expect("probe JP2 metadata parses");
    let retained = probe.allocated_bytes().expect("probe metadata capacity");
    drop(probe);

    let exact_baseline = crate::DEFAULT_MAX_DECODE_BYTES - retained;
    parse_jp2_header_box(&header, true, exact_baseline)
        .expect("nested JP2 metadata fits exact aggregate cap");
    assert!(
        parse_jp2_header_box(&header, true, exact_baseline + 1).is_err(),
        "one byte beyond nested JP2 metadata cap must fail"
    );
}

#[test]
fn implicit_mapping_allocation_counts_external_baseline_at_exact_boundary() {
    let retained_container_bytes = 17;
    let mapping_bytes = size_of::<ComponentMappingEntry>();
    let exact_baseline = crate::DEFAULT_MAX_DECODE_BYTES - retained_container_bytes - mapping_bytes;
    let mut exact = implicit_mapping_budget(retained_container_bytes, exact_baseline)
        .expect("implicit mapping exact baseline");
    exact
        .try_vec::<ComponentMappingEntry>(1, "implicit JP2 component mappings")
        .expect("one implicit mapping fits exact aggregate cap");
    assert_eq!(exact.live_bytes(), crate::DEFAULT_MAX_DECODE_BYTES);

    let mut one_over = implicit_mapping_budget(retained_container_bytes, exact_baseline + 1)
        .expect("baseline alone remains under the cap");
    assert!(matches!(
        one_over.try_vec::<ComponentMappingEntry>(1, "implicit JP2 component mappings"),
        Err(crate::DecodeError::AllocationTooLarge { .. })
    ));
}

fn required_header_prefix(color_payload: &[u8]) -> Vec<u8> {
    let mut header = Vec::new();
    let mut image_header = Vec::new();
    image_header.extend_from_slice(&1_u32.to_be_bytes());
    image_header.extend_from_slice(&1_u32.to_be_bytes());
    image_header.extend_from_slice(&1_u16.to_be_bytes());
    image_header.extend_from_slice(&[0xff, 7, 0, 0]);
    push_child(&mut header, *b"ihdr", &image_header);
    push_child(&mut header, *b"colr", color_payload);
    header
}

fn palette_payload(value: u8) -> Vec<u8> {
    vec![0, 1, 1, 7, value]
}

fn cdef_payload(channel_type: u16) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&1_u16.to_be_bytes());
    payload.extend_from_slice(&0_u16.to_be_bytes());
    payload.extend_from_slice(&channel_type.to_be_bytes());
    payload.extend_from_slice(&0_u16.to_be_bytes());
    payload
}

fn push_child(output: &mut Vec<u8>, box_type: [u8; 4], payload: &[u8]) {
    let box_len = u32::try_from(payload.len() + 8).expect("small test box");
    output.extend_from_slice(&box_len.to_be_bytes());
    output.extend_from_slice(&box_type);
    output.extend_from_slice(payload);
}

fn shallow_allocated_bytes(boxes: &ImageBoxes) -> usize {
    let mut bytes = boxes.bits_per_component.capacity() * size_of::<ComponentDescriptor>();
    bytes += boxes.color_specifications.capacity() * size_of::<ColorSpecificationBox>();
    for color in &boxes.color_specifications {
        if let NativeColorSpace::Icc(profile) = &color.color_space {
            bytes += profile.capacity();
        }
    }
    if let Some(palette) = &boxes.palette {
        bytes += palette.columns.capacity() * size_of::<crate::jp2::pclr::PaletteColumn>();
        bytes += palette.entries.capacity() * size_of::<Vec<u64>>();
        bytes += palette.entries.iter().map(Vec::capacity).sum::<usize>() * size_of::<u64>();
    }
    if let Some(mapping) = &boxes.component_mapping {
        bytes += mapping.entries.capacity() * size_of::<ComponentMappingEntry>();
    }
    if let Some(definition) = &boxes.channel_definition {
        bytes += definition.channel_definitions.capacity()
            * size_of::<crate::jp2::cdef::ChannelDefinition>();
    }
    bytes
}
