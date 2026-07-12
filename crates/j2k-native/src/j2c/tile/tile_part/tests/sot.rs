// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{parse_tile_part, sot_marker};
use super::{header, inherited_tile_state, tile_part_bytes};
use crate::error::{DecodeError, MarkerError, TileError};
use crate::reader::BitReader;

#[test]
fn sot_marker_rejects_noncanonical_segment_length() {
    for length in [9_u16, 11] {
        let mut bytes = [
            0x00, 0x00, // Lsot
            0x00, 0x00, // Isot
            0x00, 0x00, 0x00, 0x0e, // Psot
            0x00, 0x01, // TPsot, TNsot
        ];
        bytes[..2].copy_from_slice(&length.to_be_bytes());
        let mut reader = BitReader::new(&bytes);

        assert!(sot_marker(&mut reader).is_none());
        assert_eq!(reader.offset(), 2, "invalid Lsot must not consume Isot");
    }
}

#[test]
fn sot_marker_accepts_only_a_complete_canonical_header() {
    let bytes = [
        0x00, 0x0a, // Lsot
        0x00, 0x02, // Isot
        0x00, 0x00, 0x00, 0x20, // Psot
        0x03, 0x04, // TPsot, TNsot
    ];
    let mut reader = BitReader::new(&bytes);
    let parsed = sot_marker(&mut reader).expect("canonical SOT header");

    assert_eq!(parsed.tile_index, 2);
    assert_eq!(parsed.tile_part_length, 32);
    assert_eq!(parsed.tile_part_index, 3);
    assert_eq!(parsed.num_tile_parts, 4);
    assert_eq!(reader.offset(), bytes.len());

    for truncated_len in 0..bytes.len() {
        let mut reader = BitReader::new(&bytes[..truncated_len]);
        assert!(sot_marker(&mut reader).is_none());
        assert!(reader.offset() <= truncated_len);
    }
}

#[test]
fn invalid_sot_length_is_typed_and_preserves_tile_owner_state() {
    let header = header();
    let (mut tiles, mut budget, retained_before) = inherited_tile_state(&header);
    let bytes = [
        0xff, 0x90, // SOT
        0x00, 0x09, // invalid Lsot
        0x00, 0x00, // Isot
        0x00, 0x00, 0x00, 0x0e, // Psot
        0x00, 0x01, // TPsot, TNsot
        0xff, 0x93, // SOD
    ];
    let mut reader = BitReader::new(&bytes);
    let mut ppm_packet_idx = 0;

    let error = parse_tile_part(
        &mut reader,
        &header,
        &mut tiles,
        &mut ppm_packet_idx,
        &mut budget,
    )
    .expect_err("invalid Lsot must reject");

    assert_eq!(error, DecodeError::Marker(MarkerError::ParseFailure("SOT")));
    assert_eq!(reader.offset(), 4, "only marker and Lsot are consumed");
    assert_eq!(ppm_packet_idx, 0);
    assert!(tiles[0].tile_parts.is_empty());
    assert_eq!(budget.retained_bytes(), retained_before);
    budget
        .validate_owner_graph(&tiles)
        .expect("failed SOT parsing leaves the owner graph unchanged");
}

#[test]
fn invalid_tile_index_and_short_psot_preserve_tile_owner_state() {
    let header = header();
    for (bytes, expected) in [
        (
            tile_part_bytes(1, 14, true),
            DecodeError::Tile(TileError::InvalidIndex),
        ),
        (
            tile_part_bytes(0, 11, true),
            DecodeError::Tile(TileError::Invalid),
        ),
    ] {
        let (mut tiles, mut budget, retained_before) = inherited_tile_state(&header);
        let mut reader = BitReader::new(&bytes);
        let mut ppm_packet_idx = 0;

        let error = parse_tile_part(
            &mut reader,
            &header,
            &mut tiles,
            &mut ppm_packet_idx,
            &mut budget,
        )
        .expect_err("invalid SOT semantics must reject");

        assert_eq!(error, expected);
        assert_eq!(ppm_packet_idx, 0);
        assert!(tiles[0].tile_parts.is_empty());
        assert_eq!(budget.retained_bytes(), retained_before);
        budget
            .validate_owner_graph(&tiles)
            .expect("invalid SOT semantics leave owners unchanged");
    }
}
