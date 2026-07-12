// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::parse_tile_part;
use super::{header, inherited_tile_state, tile_part_bytes};
use crate::error::{DecodeError, MarkerError};
use crate::j2c::tile::TilePart;
use crate::reader::BitReader;

#[test]
fn strict_and_lenient_missing_sod_preserve_tile_owner_state() {
    for strict in [true, false] {
        let mut header = header();
        header.strict = strict;
        let (mut tiles, mut budget, retained_before) = inherited_tile_state(&header);
        let bytes = tile_part_bytes(0, 0, false);
        let mut reader = BitReader::new(&bytes);
        let mut ppm_packet_idx = 0;

        let result = parse_tile_part(
            &mut reader,
            &header,
            &mut tiles,
            &mut ppm_packet_idx,
            &mut budget,
        );

        if strict {
            assert_eq!(result, Err(DecodeError::Marker(MarkerError::Invalid)));
        } else {
            result.expect("lenient missing SOD remains tolerated");
        }
        assert_eq!(ppm_packet_idx, 0);
        assert!(tiles[0].tile_parts.is_empty());
        assert_eq!(budget.retained_bytes(), retained_before);
        budget
            .validate_owner_graph(&tiles)
            .expect("missing SOD leaves owners unchanged");
    }
}

#[test]
fn minimal_merged_tile_part_commits_one_empty_body_transactionally() {
    let header = header();
    let (mut tiles, mut budget, retained_before) = inherited_tile_state(&header);
    let bytes = tile_part_bytes(0, 14, true);
    let mut reader = BitReader::new(&bytes);
    let mut ppm_packet_idx = 0;

    parse_tile_part(
        &mut reader,
        &header,
        &mut tiles,
        &mut ppm_packet_idx,
        &mut budget,
    )
    .expect("minimal empty merged tile part");

    assert_eq!(reader.offset(), bytes.len());
    assert_eq!(ppm_packet_idx, 0);
    assert_eq!(tiles[0].tile_parts.len(), 1);
    let TilePart::Merged(part) = &tiles[0].tile_parts[0] else {
        panic!("tile part without external headers stays merged");
    };
    assert!(part.data.tail().is_none_or(<[u8]>::is_empty));
    assert!(part.packet_lengths.lengths.is_empty());
    assert!(budget.retained_bytes() >= retained_before);
    budget
        .validate_owner_graph(&tiles)
        .expect("committed merged tile part is fully accounted");
}

#[test]
fn malformed_ppt_rolls_back_temporary_owner_capacity() {
    let header = header();
    let (mut tiles, mut budget, retained_before) = inherited_tile_state(&header);

    let malformed_ppt = [
        0xff, 0x90, // SOT
        0x00, 0x0a, // Lsot
        0x00, 0x00, // Isot
        0x00, 0x00, 0x00, 0x00, // Psot: extends to input end
        0x00, 0x01, // TPsot, TNsot
        0xff, 0x61, // PPT
        0x00, 0x02, // Lppt has no Zppt byte
    ];
    let mut reader = BitReader::new(&malformed_ppt);
    let mut ppm_packet_idx = 0;
    let error = parse_tile_part(
        &mut reader,
        &header,
        &mut tiles,
        &mut ppm_packet_idx,
        &mut budget,
    )
    .expect_err("malformed PPT must reject");

    assert_eq!(error, DecodeError::Marker(MarkerError::ParseFailure("PPT")));
    assert_eq!(budget.retained_bytes(), retained_before);
    budget
        .validate_owner_graph(&tiles)
        .expect("temporary PPT capacity is fully rolled back");
}
