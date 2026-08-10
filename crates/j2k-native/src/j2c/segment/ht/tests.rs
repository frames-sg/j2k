// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2c::build::CodeBlockCoding;
use crate::j2c::codestream::CodeBlockStyle;
use crate::j2c::rect::IntRect;
use crate::writer::BitWriter;
use alloc::vec::Vec;

fn code_block(missing_bit_planes: u8) -> CodeBlock {
    CodeBlock {
        rect: IntRect::from_xywh(0, 0, 1, 1),
        x_idx: 0,
        y_idx: 0,
        layers: 0..1,
        has_been_included: true,
        missing_bit_planes,
        number_of_coding_passes: 0,
        ht_total_coding_passes: 0,
        ht_first_cleanup_pass: None,
        ht_selected_set: None,
        coding: None,
        l_block: 3,
        non_empty_layer_count: 0,
    }
}

fn parse_packet(
    writer: BitWriter,
    num_passes: u8,
    block: &mut CodeBlock,
    segments: &mut Vec<Segment<'static>>,
) -> PacketResult {
    let data = writer.finish();
    let mut reader = BitReader::new(&data);
    let mut allocation_error = None;
    let result = parse_segment_lengths(
        &mut reader,
        num_passes,
        CodeBlockStyle {
            high_throughput_block_coding: true,
            ..Default::default()
        },
        block,
        segments,
        0,
        &mut allocation_error,
    );
    assert!(allocation_error.is_none());
    result
}

#[test]
fn mixed_length_without_ht_discriminator_selects_classic() {
    let mut writer = BitWriter::new();
    writer.write_bit(0);
    writer.write_bits(5, 3);
    let data = writer.finish();
    let mut reader = BitReader::new(&data);
    let mut allocation_error = None;
    let mut block = code_block(1);
    let mut segments = Vec::new();

    parse_segment_lengths(
        &mut reader,
        1,
        CodeBlockStyle {
            high_throughput_block_coding: true,
            mixed_block_coding: true,
            ..Default::default()
        },
        &mut block,
        &mut segments,
        0,
        &mut allocation_error,
    )
    .unwrap();

    assert_eq!(block.coding, Some(CodeBlockCoding::Classic));
    assert_eq!(block.number_of_coding_passes, 1);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].coding_pases, 1);
    assert_eq!(segments[0].data_length, 5);
}

#[test]
fn mixed_length_with_zero_msb_selects_ht_cleanup() {
    let mut writer = BitWriter::new();
    writer.write_bits(0b10, 2);
    writer.write_bits(5, 4);
    let data = writer.finish();
    let mut reader = BitReader::new(&data);
    let mut allocation_error = None;
    let mut block = code_block(1);
    let mut segments = Vec::new();

    parse_segment_lengths(
        &mut reader,
        1,
        CodeBlockStyle {
            high_throughput_block_coding: true,
            mixed_block_coding: true,
            ..Default::default()
        },
        &mut block,
        &mut segments,
        0,
        &mut allocation_error,
    )
    .unwrap();

    assert_eq!(block.coding, Some(CodeBlockCoding::HighThroughput));
    assert_eq!(block.ht_first_cleanup_pass, Some(0));
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].data_length, 5);
}

#[test]
fn mixed_placeholder_consumes_the_full_classic_width_once() {
    let mut writer = BitWriter::new();
    writer.write_bit(0);
    writer.write_bits(0, 4);
    writer.write_bit(1);
    let data = writer.finish();
    let mut reader = BitReader::new(&data);
    let mut allocation_error = None;
    let mut block = code_block(1);
    let mut segments = Vec::new();

    parse_segment_lengths(
        &mut reader,
        3,
        CodeBlockStyle {
            high_throughput_block_coding: true,
            mixed_block_coding: true,
            ..Default::default()
        },
        &mut block,
        &mut segments,
        0,
        &mut allocation_error,
    )
    .unwrap();

    assert_eq!(block.coding, None);
    assert_eq!(block.ht_total_coding_passes, 3);
    assert!(segments.is_empty());
    assert_eq!(reader.read_bits_with_stuffing(1), Some(1));
}

fn encode_num_passes(num_passes: u8) -> Vec<u8> {
    let mut writer = BitWriter::new();

    match num_passes {
        1 => writer.write_bit(0),
        2 => writer.write_bits(0b10, 2),
        3..=5 => {
            writer.write_bits(0b11, 2);
            writer.write_bits(u32::from(num_passes - 3), 2);
        }
        6..=36 => {
            writer.write_bits(0b11, 2);
            writer.write_bits(0b11, 2);
            writer.write_bits(u32::from(num_passes - 6), 5);
        }
        37..=164 => {
            writer.write_bits(0b11, 2);
            writer.write_bits(0b11, 2);
            writer.write_bits(31, 5);
            writer.write_bits(u32::from(num_passes - 37), 7);
        }
        _ => unreachable!(),
    }

    writer.finish()
}

#[test]
fn code_block_style_detects_high_throughput() {
    let style = CodeBlockStyle {
        high_throughput_block_coding: true,
        ..Default::default()
    };
    assert!(style.uses_high_throughput_block_coding());

    let style = CodeBlockStyle::default();
    assert!(!style.uses_high_throughput_block_coding());
}

#[test]
fn coding_pass_count_round_trips() {
    for num_passes in [1u8, 2, 3, 4, 5, 6, 19, 37, 38, 100, 164] {
        let data = encode_num_passes(num_passes);
        let mut reader = BitReader::new(&data);

        assert_eq!(decode_num_coding_passes(&mut reader), Some(num_passes));
    }
}

#[test]
fn first_cleanup_folds_placeholder_passes() {
    let mut writer = BitWriter::new();
    writer.write_bit(0);
    writer.write_bits(5, 5);
    let mut block = code_block(2);
    let mut segments = Vec::new();

    parse_packet(writer, 4, &mut block, &mut segments).unwrap();

    assert_eq!(block.ht_total_coding_passes, 4);
    assert_eq!(block.ht_first_cleanup_pass, Some(3));
    assert_eq!(block.ht_selected_set, Some(0));
    assert_eq!(block.missing_bit_planes, 3);
    assert_eq!(block.number_of_coding_passes, 1);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].idx, 0);
    assert_eq!(segments[0].coding_pases, 4);
    assert_eq!(segments[0].data_length, 5);
    assert_eq!(block.l_block, 3);
}

#[test]
fn placeholder_only_contribution_is_accepted() {
    let mut writer = BitWriter::new();
    writer.write_bit(0);
    writer.write_bits(0, 4);
    let mut block = code_block(2);
    let mut segments = Vec::new();

    parse_packet(writer, 3, &mut block, &mut segments).unwrap();

    assert_eq!(block.ht_total_coding_passes, 3);
    assert_eq!(block.ht_first_cleanup_pass, None);
    assert_eq!(block.ht_selected_set, None);
    assert_eq!(block.missing_bit_planes, 2);
    assert_eq!(block.number_of_coding_passes, 0);
    assert!(segments.is_empty());
}

#[test]
fn first_cleanup_after_an_earlier_placeholder_packet_is_selected() {
    let mut placeholder = BitWriter::new();
    placeholder.write_bit(0);
    placeholder.write_bits(0, 4);
    let mut block = code_block(2);
    let mut segments = Vec::new();
    parse_packet(placeholder, 3, &mut block, &mut segments).unwrap();

    let mut cleanup = BitWriter::new();
    cleanup.write_bit(0);
    cleanup.write_bits(5, 3);
    cleanup.write_bits(7, 4);
    parse_packet(cleanup, 3, &mut block, &mut segments).unwrap();

    assert_eq!(block.ht_total_coding_passes, 6);
    assert_eq!(block.ht_first_cleanup_pass, Some(3));
    assert_eq!(block.ht_selected_set, Some(0));
    assert_eq!(block.missing_bit_planes, 3);
    assert_eq!(block.number_of_coding_passes, 3);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].idx, 0);
    assert_eq!(segments[0].data_length, 5);
    assert_eq!(segments[1].idx, 1);
    assert_eq!(segments[1].coding_pases, 2);
    assert_eq!(segments[1].data_length, 7);
}

#[test]
fn first_cleanup_reads_refinement_segment() {
    let mut writer = BitWriter::new();
    writer.write_bits(0b110, 3);
    writer.write_bits(9, 5);
    writer.write_bits(17, 6);
    let mut block = code_block(1);
    let mut segments = Vec::new();

    parse_packet(writer, 3, &mut block, &mut segments).unwrap();

    assert_eq!(block.number_of_coding_passes, 3);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].data_length, 9);
    assert_eq!(segments[1].coding_pases, 2);
    assert_eq!(segments[1].data_length, 17);
    assert_eq!(block.l_block, 5);
}

#[test]
fn later_non_empty_ht_set_replaces_the_selected_set() {
    let mut first = BitWriter::new();
    first.write_bit(0);
    first.write_bits(5, 3);
    first.write_bits(7, 4);
    let mut block = code_block(1);
    let mut segments = Vec::new();
    parse_packet(first, 3, &mut block, &mut segments).unwrap();

    let mut second = BitWriter::new();
    second.write_bit(0);
    second.write_bits(6, 3);
    second.write_bits(8, 4);
    parse_packet(second, 3, &mut block, &mut segments).unwrap();

    assert_eq!(block.ht_total_coding_passes, 6);
    assert_eq!(block.ht_selected_set, Some(1));
    assert_eq!(block.missing_bit_planes, 2);
    assert_eq!(block.number_of_coding_passes, 3);
    assert_eq!(segments.len(), 4);
    assert_eq!(segments[2].idx, 2);
    assert_eq!(segments[2].data_length, 6);
    assert_eq!(segments[3].idx, 3);
    assert_eq!(segments[3].data_length, 8);
}

#[test]
fn empty_later_ht_set_preserves_the_selected_set() {
    let mut first = BitWriter::new();
    first.write_bit(0);
    first.write_bits(5, 3);
    first.write_bits(7, 4);
    let mut block = code_block(1);
    let mut segments = Vec::new();
    parse_packet(first, 3, &mut block, &mut segments).unwrap();

    let mut empty = BitWriter::new();
    empty.write_bit(0);
    empty.write_bits(0, 3);
    empty.write_bits(0, 4);
    parse_packet(empty, 3, &mut block, &mut segments).unwrap();

    assert_eq!(block.ht_total_coding_passes, 6);
    assert_eq!(block.ht_selected_set, Some(0));
    assert_eq!(block.missing_bit_planes, 1);
    assert_eq!(block.number_of_coding_passes, 3);
    assert_eq!(segments.len(), 2);
}

#[test]
fn refinement_segment_may_be_split_across_packets() {
    let mut cleanup = BitWriter::new();
    cleanup.write_bit(0);
    cleanup.write_bits(5, 3);
    let mut block = code_block(1);
    let mut segments = Vec::new();
    parse_packet(cleanup, 1, &mut block, &mut segments).unwrap();

    let mut sigprop = BitWriter::new();
    sigprop.write_bit(0);
    sigprop.write_bits(3, 3);
    parse_packet(sigprop, 1, &mut block, &mut segments).unwrap();

    let mut magref = BitWriter::new();
    magref.write_bit(0);
    magref.write_bits(4, 3);
    parse_packet(magref, 1, &mut block, &mut segments).unwrap();

    assert_eq!(block.number_of_coding_passes, 3);
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[1].idx, 1);
    assert_eq!(segments[1].coding_pases, 1);
    assert_eq!(segments[1].data_length, 3);
    assert_eq!(segments[2].idx, 1);
    assert_eq!(segments[2].coding_pases, 1);
    assert_eq!(segments[2].data_length, 4);
}
