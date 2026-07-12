// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::*;
use crate::j2c::tile::{MergedTilePart, SeparatedTilePart};

#[test]
fn separated_ppm_ppt_cursor_spans_multiple_readers_without_mutating_retained_state() {
    let first = [0x13_u8];
    let empty = [];
    let second = [0x57_u8];
    let body = [0x9B_u8];
    let tile_part = TilePart::Separated(SeparatedTilePart {
        headers: vec![
            BitReader::new(&first),
            BitReader::new(&empty),
            BitReader::new(&second),
        ],
        body: BitReader::new(&body),
        packet_lengths: PacketLengthMetadata::new(false, Vec::new()),
    });

    for _ in 0..2 {
        let mut cursor = tile_part.cursor().expect("separated cursor");
        assert_eq!(cursor.header().read_byte(), Some(0x13));
        assert_eq!(cursor.header().read_byte(), Some(0x57));
        assert_eq!(cursor.body().read_byte(), Some(0x9B));
        assert!(cursor.header().at_end());
        assert!(cursor.body().at_end());
    }

    let TilePart::Separated(retained) = &tile_part else {
        panic!("separated fixture")
    };
    assert!(retained.headers.iter().all(|reader| reader.offset() == 0));
    assert_eq!(retained.body.offset(), 0);
}

#[test]
fn plt_length_cursor_validates_each_packet_and_resets_for_reuse() {
    let data = [0x11_u8, 0x22];
    let tile_part = TilePart::Merged(MergedTilePart {
        data: BitReader::new(&data),
        packet_lengths: PacketLengthMetadata::new(true, vec![1, 1]),
    });

    for _ in 0..2 {
        let mut cursor = tile_part.cursor().expect("merged cursor");
        for expected in data {
            let packet_start = cursor.packet_start_offset();
            assert_eq!(cursor.body().read_byte(), Some(expected));
            cursor
                .validate_packet_length(packet_start)
                .expect("matching PLT length");
        }
        cursor
            .validate_all_packet_lengths_consumed()
            .expect("all PLT entries consumed");
    }

    let mismatched_part = TilePart::Merged(MergedTilePart {
        data: BitReader::new(&data),
        packet_lengths: PacketLengthMetadata::new(true, vec![1]),
    });
    let mut mismatched = mismatched_part.cursor().expect("mismatched cursor");
    let packet_start = mismatched.packet_start_offset();
    assert_eq!(mismatched.body().read_bytes(2), Some(data.as_slice()));
    assert_eq!(mismatched.validate_packet_length(packet_start), None);
}
