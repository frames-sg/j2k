// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::*;
use crate::j2c::tag_tree::{TagNode, TagTree};
use crate::reader::BitReader;
use crate::{EncodeError, J2kPacketizationProgressionOrder};

fn decode_num_ht_coding_passes_for_test(data: &[u8]) -> Option<u8> {
    let mut reader = BitReader::new(data);
    decode_num_ht_coding_passes_from_reader_for_test(&mut reader)
}

fn decode_num_ht_coding_passes_from_reader_for_test(reader: &mut BitReader<'_>) -> Option<u8> {
    let mut num_passes = 1u32;

    if reader.read_bits_with_stuffing(1)? == 1 {
        num_passes = 2;

        if reader.read_bits_with_stuffing(1)? == 1 {
            let extension = reader.read_bits_with_stuffing(2)?;
            num_passes = 3 + extension;

            if extension == 3 {
                let extension = reader.read_bits_with_stuffing(5)?;
                num_passes = 6 + extension;

                if extension == 31 {
                    num_passes = 37 + reader.read_bits_with_stuffing(7)?;
                }
            }
        }
    }

    u8::try_from(num_passes).ok()
}

fn decode_num_coding_passes_for_test(data: &[u8]) -> Option<u8> {
    let mut reader = BitReader::new(data);
    decode_num_coding_passes_from_reader_for_test(&mut reader)
}

fn decode_num_coding_passes_from_reader_for_test(reader: &mut BitReader<'_>) -> Option<u8> {
    let passes = if reader.peak_bits_with_stuffing(9) == Some(0x1ff) {
        reader.read_bits_with_stuffing(9)?;
        reader.read_bits_with_stuffing(7)? + 37
    } else if reader.peak_bits_with_stuffing(4) == Some(0x0f) {
        reader.read_bits_with_stuffing(4)?;
        reader.read_bits_with_stuffing(5)? + 6
    } else if reader.peak_bits_with_stuffing(4) == Some(0b1110) {
        reader.read_bits_with_stuffing(4)?;
        5
    } else if reader.peak_bits_with_stuffing(4) == Some(0b1101) {
        reader.read_bits_with_stuffing(4)?;
        4
    } else if reader.peak_bits_with_stuffing(4) == Some(0b1100) {
        reader.read_bits_with_stuffing(4)?;
        3
    } else if reader.peak_bits_with_stuffing(2) == Some(0b10) {
        reader.read_bits_with_stuffing(2)?;
        2
    } else if reader.peak_bits_with_stuffing(1) == Some(0) {
        reader.read_bits_with_stuffing(1)?;
        1
    } else {
        return None;
    };
    u8::try_from(passes).ok()
}

#[test]
fn test_empty_packet() {
    let mut resolution = ResolutionPacket {
        subbands: vec![SubbandPrecinct {
            code_blocks: vec![CodeBlockPacketData {
                data: Vec::new(),
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 0,
                classic_segment_lengths: Vec::new(),
                num_zero_bitplanes: 31,
                previously_included: false,
                l_block: 3,
                block_coding_mode: BlockCodingMode::Classic,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };

    let packet = form_packet(&mut resolution).expect("valid test packet");
    assert!(!packet.is_empty());
}

#[test]
fn malformed_packet_layout_returns_an_error() {
    let mut resolution = ResolutionPacket {
        subbands: vec![SubbandPrecinct {
            code_blocks: vec![CodeBlockPacketData {
                data: Vec::new(),
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 0,
                classic_segment_lengths: Vec::new(),
                num_zero_bitplanes: 0,
                previously_included: false,
                l_block: 3,
                block_coding_mode: BlockCodingMode::Classic,
            }],
            num_cbs_x: 0,
            num_cbs_y: 1,
        }],
    };

    assert_eq!(
        form_packet(&mut resolution),
        Err(EncodeError::InvalidInput {
            what: "invalid packet subband code-block layout",
        })
    );
}

#[test]
fn packet_length_bit_count_overflow_returns_an_error() {
    let mut writer = BitWriter::new();
    let mut l_block = u32::from(u8::MAX) + 1;
    let num_bits = l_block;
    assert_eq!(
        encode_length(0, &mut l_block, num_bits, &mut writer),
        Err(EncodeError::InvalidInput {
            what: "packet length bit count exceeds u8",
        })
    );
}

#[test]
fn test_non_empty_packet() {
    let mut resolution = ResolutionPacket {
        subbands: vec![SubbandPrecinct {
            code_blocks: vec![CodeBlockPacketData {
                data: vec![0x12, 0x34, 0x56],
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                classic_segment_lengths: Vec::new(),
                num_zero_bitplanes: 20,
                previously_included: false,
                l_block: 3,
                block_coding_mode: BlockCodingMode::Classic,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };

    let packet = form_packet(&mut resolution).expect("valid test packet");
    assert!(packet.len() >= 3);
}

#[test]
fn packet_header_round_trips_varied_8x8_codeblock_lengths() {
    let zero_bitplanes = [
        2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1, 2, 3, 2, 2, 1, 1,
        1, 1, 2, 3, 2, 2, 1, 1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2,
        2, 1, 1, 1,
    ];
    let lengths = [
        1901, 2062, 1895, 2329, 2860, 2842, 2852, 2836, 2174, 2121, 1878, 2197, 2877, 2870, 2854,
        2862, 2097, 2143, 1906, 2059, 2724, 2879, 2860, 2847, 1928, 1967, 2105, 2318, 2605, 2911,
        2892, 2860, 1998, 1995, 2073, 2075, 2339, 2935, 2896, 2897, 1877, 1938, 1841, 2000, 2271,
        2877, 2826, 2828, 2098, 1899, 1953, 2061, 2135, 2886, 2869, 2909, 2168, 1921, 1966, 2048,
        2159, 2792, 2853, 2815,
    ];
    let mut resolution = ResolutionPacket {
        subbands: vec![SubbandPrecinct {
            code_blocks: zero_bitplanes
                .iter()
                .copied()
                .zip(lengths.iter().copied())
                .map(|(num_zero_bitplanes, len)| CodeBlockPacketData {
                    data: vec![0; len],
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1 + 3 * (8 - num_zero_bitplanes) - 2,
                    classic_segment_lengths: Vec::new(),
                    num_zero_bitplanes,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::Classic,
                })
                .collect(),
            num_cbs_x: 8,
            num_cbs_y: 8,
        }],
    };

    let packet = form_packet(&mut resolution).expect("valid test packet");
    let body_len: usize = lengths.iter().sum();
    let header_len = packet.len() - body_len;
    let mut reader = BitReader::new(&packet[..header_len]);
    assert_eq!(reader.read_bits_with_stuffing(1), Some(1));

    let mut inclusion_nodes = Vec::<TagNode>::new();
    let mut inclusion_tree = TagTree::new(8, 8, &mut inclusion_nodes);
    let mut zbp_nodes = Vec::<TagNode>::new();
    let mut zbp_tree = TagTree::new(8, 8, &mut zbp_nodes);

    for (idx, (&expected_zbp, &expected_len)) in
        zero_bitplanes.iter().zip(lengths.iter()).enumerate()
    {
        let index = u32::try_from(idx).expect("8x8 test code-block index fits u32");
        let x = index % 8;
        let y = index / 8;
        let included = inclusion_tree
            .read(x, y, &mut reader, 1, &mut inclusion_nodes)
            .expect("inclusion tag")
            == 0;
        assert!(included, "inclusion at index {idx}");

        let actual_zbp = zbp_tree
            .read(x, y, &mut reader, u32::MAX, &mut zbp_nodes)
            .expect("zero bitplane tag");
        assert_eq!(actual_zbp, u32::from(expected_zbp), "zbp at index {idx}");

        let passes = decode_num_coding_passes_from_reader_for_test(&mut reader)
            .expect("number of coding passes");
        let mut l_block = 3u32;
        while reader.read_bits_with_stuffing(1).expect("lblock increment") == 1 {
            l_block += 1;
        }
        let length_bits = l_block + u32::from(passes).ilog2();
        let actual_len = reader
            .read_bits_with_stuffing(
                u8::try_from(length_bits).expect("packet length bit count fits u8"),
            )
            .expect("code-block length");
        assert_eq!(
            actual_len,
            u32::try_from(expected_len).expect("test payload length fits u32"),
            "length at index {idx}"
        );
    }
}

#[test]
fn packet_header_trailing_ff_stuffs_zero_before_body() {
    for len in 1..4096 {
        let mut resolution = ResolutionPacket {
            subbands: vec![SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data: vec![0x80; len],
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    classic_segment_lengths: Vec::new(),
                    num_zero_bitplanes: 0,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };

        let packet = form_packet(&mut resolution).expect("valid test packet");
        let header_len = packet.len() - len;
        let has_boundary_ff = packet[header_len - 1] == 0xff
            || (header_len >= 2
                && packet[header_len - 2] == 0xff
                && packet[header_len - 1] == 0x00);

        if !has_boundary_ff {
            continue;
        }

        let mut reader = BitReader::new(&packet);
        assert_eq!(reader.read_bits_with_stuffing(1), Some(1));

        let mut inclusion_nodes = Vec::<TagNode>::new();
        let mut inclusion_tree = TagTree::new(1, 1, &mut inclusion_nodes);
        let included = inclusion_tree
            .read(0, 0, &mut reader, 1, &mut inclusion_nodes)
            .expect("inclusion tag")
            == 0;
        assert!(included);

        let mut zbp_nodes = Vec::<TagNode>::new();
        let mut zbp_tree = TagTree::new(1, 1, &mut zbp_nodes);
        assert_eq!(
            zbp_tree
                .read(0, 0, &mut reader, u32::MAX, &mut zbp_nodes)
                .expect("zero bitplane tag"),
            0
        );

        let passes = decode_num_coding_passes_from_reader_for_test(&mut reader)
            .expect("number of coding passes");
        assert_eq!(passes, 1);

        let mut l_block = 3u32;
        while reader.read_bits_with_stuffing(1).expect("lblock increment") == 1 {
            l_block += 1;
        }
        let actual_len = reader
            .read_bits_with_stuffing(
                u8::try_from(l_block).expect("packet length bit count fits u8"),
            )
            .expect("code-block length");
        assert_eq!(
            actual_len,
            u32::try_from(len).expect("test payload length fits u32")
        );

        reader.align();
        let expected_body = vec![0x80; len];
        assert_eq!(reader.offset(), header_len);
        assert_eq!(reader.read_bytes(len), Some(expected_body.as_slice()));
        return;
    }

    panic!("did not find a packet header ending in 0xff");
}

#[test]
fn classic_pass_terminated_lengths_share_one_lblock_increment() {
    let lengths = [1u32, 9, 17];
    let mut code_block = CodeBlockPacketData {
        data: vec![
            0;
            usize::try_from(lengths.iter().sum::<u32>())
                .expect("test payload length fits usize")
        ],
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        num_coding_passes: u8::try_from(lengths.len()).expect("pass count fits u8"),
        classic_segment_lengths: lengths.to_vec(),
        num_zero_bitplanes: 0,
        previously_included: false,
        l_block: 3,
        block_coding_mode: BlockCodingMode::Classic,
    };
    let mut writer = BitWriter::new();
    let data_len = u32::try_from(code_block.data.len()).expect("test payload length fits u32");

    encode_num_coding_passes(code_block.num_coding_passes, &mut writer)
        .expect("valid classic pass count");
    encode_classic_segment_lengths(&mut code_block, data_len, &mut writer)
        .expect("classic segment lengths encode");

    let bytes = writer.finish();
    let mut reader = BitReader::new(&bytes);
    let passes = decode_num_coding_passes_from_reader_for_test(&mut reader)
        .expect("number of coding passes");
    assert_eq!(
        passes,
        u8::try_from(lengths.len()).expect("pass count fits u8")
    );

    let mut l_block = 3u32;
    while reader.read_bits_with_stuffing(1).expect("lblock increment") == 1 {
        l_block += 1;
    }

    let decoded_lengths: Vec<_> = lengths
        .iter()
        .map(|_| {
            reader
                .read_bits_with_stuffing(
                    u8::try_from(l_block).expect("packet length bit count fits u8"),
                )
                .expect("terminated pass segment length")
        })
        .collect();
    assert_eq!(decoded_lengths, lengths);
}

#[test]
fn test_multi_subband_packet() {
    let mut resolution = ResolutionPacket {
        subbands: vec![
            SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data: vec![0x10, 0x20],
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    classic_segment_lengths: Vec::new(),
                    num_zero_bitplanes: 20,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            },
            SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data: vec![0x30, 0x40],
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    classic_segment_lengths: Vec::new(),
                    num_zero_bitplanes: 22,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            },
            SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data: vec![0x50],
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    classic_segment_lengths: Vec::new(),
                    num_zero_bitplanes: 24,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            },
        ],
    };

    let packet = form_packet(&mut resolution).expect("valid test packet");
    // Should contain all 5 bytes of code-block data
    assert!(packet.len() >= 5);
}

#[test]
fn test_encode_num_passes() {
    let mut w = BitWriter::new();
    encode_num_coding_passes(1, &mut w).expect("valid classic pass count");
    let d = w.finish();
    assert_eq!(d.len(), 1);
}

#[test]
fn test_encode_num_passes_round_trip() {
    for num_passes in [1u8, 2, 3, 4, 5, 6, 19, 36, 37, 38, 100, 164] {
        let mut w = BitWriter::new();
        encode_num_coding_passes(num_passes, &mut w).expect("valid classic pass count");
        let data = w.finish();
        assert_eq!(decode_num_coding_passes_for_test(&data), Some(num_passes));
    }
}

#[test]
fn classic_pass_count_boundary_headers_are_bit_exact() {
    for (num_passes, expected) in [
        (36u8, vec![0xff, 0x00]),
        (37u8, vec![0xff, 0x40, 0x00]),
        (164u8, vec![0xff, 0x7f, 0x80]),
    ] {
        let mut writer = BitWriter::new();
        encode_num_coding_passes(num_passes, &mut writer)
            .expect("valid classic pass-count boundary");
        assert_eq!(writer.finish(), expected, "pass count {num_passes}");
    }
}

#[test]
fn test_encode_num_ht_passes_round_trip() {
    for num_passes in [1u8, 2, 3, 4, 5, 6, 19, 37, 38, 100, 164] {
        let mut w = BitWriter::new();
        encode_num_ht_coding_passes(num_passes, &mut w).expect("valid HT pass count");
        let data = w.finish();
        assert_eq!(
            decode_num_ht_coding_passes_for_test(&data),
            Some(num_passes)
        );
    }
}

#[test]
fn test_non_empty_ht_packet() {
    let mut resolution = ResolutionPacket {
        subbands: vec![SubbandPrecinct {
            code_blocks: vec![CodeBlockPacketData {
                data: vec![0x12, 0x34, 0x56],
                ht_cleanup_length: 3,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                classic_segment_lengths: Vec::new(),
                num_zero_bitplanes: 20,
                previously_included: false,
                l_block: 3,
                block_coding_mode: BlockCodingMode::HighThroughput,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };

    let packet = form_packet(&mut resolution).expect("valid test packet");
    assert!(packet.len() >= 3);
}

#[test]
fn ht_packet_header_round_trips_refinement_pass_count_and_length() {
    let payload = vec![0x12, 0x34, 0x56, 0x78, 0x9a];
    let mut resolution = ResolutionPacket {
        subbands: vec![SubbandPrecinct {
            code_blocks: vec![CodeBlockPacketData {
                data: payload.clone(),
                ht_cleanup_length: 3,
                ht_refinement_length: 2,
                num_coding_passes: 3,
                classic_segment_lengths: Vec::new(),
                num_zero_bitplanes: 2,
                previously_included: false,
                l_block: 3,
                block_coding_mode: BlockCodingMode::HighThroughput,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };

    let packet = form_packet(&mut resolution).expect("valid test packet");
    let header_len = packet.len() - payload.len();
    let mut reader = BitReader::new(&packet[..header_len]);
    assert_eq!(reader.read_bits_with_stuffing(1), Some(1));

    let mut inclusion_nodes = Vec::<TagNode>::new();
    let mut inclusion_tree = TagTree::new(1, 1, &mut inclusion_nodes);
    assert_eq!(
        inclusion_tree.read(0, 0, &mut reader, 1, &mut inclusion_nodes),
        Some(0)
    );

    let mut zbp_nodes = Vec::<TagNode>::new();
    let mut zbp_tree = TagTree::new(1, 1, &mut zbp_nodes);
    assert_eq!(
        zbp_tree.read(0, 0, &mut reader, u32::MAX, &mut zbp_nodes),
        Some(2)
    );

    let passes = decode_num_ht_coding_passes_from_reader_for_test(&mut reader)
        .expect("HT coding pass count");
    assert_eq!(passes, 3);

    let mut l_block = 3u32;
    let mut length_bits = bits_for_ht_cleanup_length(l_block, passes);
    while reader.read_bits_with_stuffing(1).expect("lblock increment") == 1 {
        l_block += 1;
        length_bits += 1;
    }
    assert_eq!(
        reader.read_bits_with_stuffing(
            u8::try_from(length_bits).expect("cleanup length bit count fits u8")
        ),
        Some(3)
    );
    let refinement_bits = l_block + 1;
    assert_eq!(
        reader.read_bits_with_stuffing(
            u8::try_from(refinement_bits).expect("refinement length bit count fits u8")
        ),
        Some(2)
    );
    assert_eq!(&packet[header_len..], payload.as_slice());
}

#[test]
fn ht_packet_segment_lengths_reject_overflowing_refinement_sum() {
    let code_block = CodeBlockPacketData {
        data: vec![0x12],
        ht_cleanup_length: u32::MAX,
        ht_refinement_length: 1,
        num_coding_passes: 3,
        classic_segment_lengths: Vec::new(),
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: BlockCodingMode::HighThroughput,
    };

    let err = ht_segment_lengths(&code_block).expect_err("overflowing HT lengths rejected");

    assert_eq!(
        err,
        EncodeError::ArithmeticOverflow {
            what: "multi-pass HTJ2K packet contribution length overflow",
        }
    );
}

fn single_block_packet(data: Vec<u8>, previously_included: bool) -> ResolutionPacket {
    ResolutionPacket {
        subbands: vec![SubbandPrecinct {
            code_blocks: vec![CodeBlockPacketData {
                data,
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                classic_segment_lengths: Vec::new(),
                num_zero_bitplanes: 0,
                previously_included,
                l_block: 3,
                block_coding_mode: BlockCodingMode::Classic,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    }
}

#[test]
fn explicit_packet_descriptors_control_packet_order() {
    let first = single_block_packet(vec![0xA0], false);
    let second = single_block_packet(vec![0xB0], false);
    let mut expected_second = single_block_packet(vec![0xB0], false);
    let mut expected_first = single_block_packet(vec![0xA0], false);
    let expected = [
        form_packet(&mut expected_second).expect("valid second test packet"),
        form_packet(&mut expected_first).expect("valid first test packet"),
    ]
    .concat();

    let actual = form_tile_bitstream_with_descriptors(
        &mut [first, second],
        &[
            PacketDescriptor {
                packet_index: 1,
                state_index: 1,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
            PacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 0,
                resolution: 1,
                component: 0,
                precinct: 0,
            },
        ],
    )
    .expect("descriptor packetization");

    assert_eq!(actual, expected);
}

#[test]
fn explicit_packet_descriptors_reuse_packet_state_across_layers() {
    let first = single_block_packet(vec![0x11], false);
    let second = single_block_packet(vec![0x22], false);

    let mut expected_first = single_block_packet(vec![0x11], false);
    let first_bytes = form_packet(&mut expected_first).expect("valid first test packet");
    let l_block_after_first = expected_first.subbands[0].code_blocks[0].l_block;
    let mut expected_second = single_block_packet(vec![0x22], true);
    expected_second.subbands[0].code_blocks[0].l_block = l_block_after_first;
    let expected = [
        first_bytes,
        form_packet(&mut expected_second).expect("valid second test packet"),
    ]
    .concat();

    let actual = form_tile_bitstream_with_descriptors(
        &mut [first, second],
        &[
            PacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
            PacketDescriptor {
                packet_index: 1,
                state_index: 0,
                layer: 1,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
        ],
    )
    .expect("stateful descriptor packetization");

    assert_eq!(actual, expected);
}

#[test]
fn explicit_packet_descriptors_reject_sparse_max_state_before_allocation() {
    let mut packets = [single_block_packet(vec![0x11], false)];
    let descriptors = [PacketDescriptor {
        packet_index: 0,
        state_index: u32::MAX,
        layer: 0,
        resolution: 0,
        component: 0,
        precinct: 0,
    }];

    let error = form_tile_bitstream_with_descriptors(&mut packets, &descriptors)
        .expect_err("sparse state index must be rejected before allocation");

    assert_eq!(
        error,
        EncodeError::InvalidInput {
            what: "packet descriptor state index out of range",
        }
    );
}

#[test]
fn implicit_single_layer_component_progressions_preserve_packet_bytes() {
    let mut default_packets = [single_block_packet(vec![0x11, 0x22], false)];
    let expected =
        form_tile_bitstream(&mut default_packets, 1, 1).expect("valid implicit packetization");

    for progression in [
        J2kPacketizationProgressionOrder::Lrcp,
        J2kPacketizationProgressionOrder::Rlcp,
        J2kPacketizationProgressionOrder::Rpcl,
        J2kPacketizationProgressionOrder::Pcrl,
        J2kPacketizationProgressionOrder::Cprl,
    ] {
        let mut packets = [single_block_packet(vec![0x11, 0x22], false)];
        let actual = form_tile_bitstream_for_progression(&mut packets, 1, 1, progression)
            .expect("one-layer one-component implicit packetization");
        assert_eq!(actual, expected, "progression {progression:?}");
    }
}

#[test]
fn implicit_progression_rejects_multidimensional_packetization() {
    let expected = || {
        EncodeError::InvalidInput {
            what: "implicit packet progression requires exactly one layer and one component; use explicit packet descriptors for multidimensional packetization",
        }
    };
    let mut layered = [single_block_packet(vec![0x11], false)];
    assert_eq!(
        form_tile_bitstream_for_progression(
            &mut layered,
            2,
            1,
            J2kPacketizationProgressionOrder::Lrcp,
        ),
        Err(expected()),
    );
    let mut multicomponent = [single_block_packet(vec![0x11], false)];
    assert_eq!(
        form_tile_bitstream_for_progression(
            &mut multicomponent,
            1,
            2,
            J2kPacketizationProgressionOrder::Lrcp,
        ),
        Err(expected()),
    );
}
