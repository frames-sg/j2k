// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::parse_tile_part;
use super::{header, inherited_tile_state};
use crate::reader::BitReader;
use crate::{DecodeError, DecodingError};

#[test]
fn tile_part_coc_cannot_shorten_the_requested_reduction_ladder() {
    let mut header = header();
    header.skipped_resolution_levels = 1;
    header.size_data.x_resolution_shrink_factor = 2;
    header.size_data.y_resolution_shrink_factor = 2;
    for component in &mut header.component_infos {
        component.coding_style.parameters.num_decomposition_levels = 1;
        component.coding_style.parameters.num_resolution_levels = 2;
        component
            .coding_style
            .parameters
            .precinct_exponents
            .push((15, 15));
    }
    header
        .global_coding_style
        .component_parameters
        .parameters
        .num_decomposition_levels = 1;
    header
        .global_coding_style
        .component_parameters
        .parameters
        .num_resolution_levels = 2;
    header
        .global_coding_style
        .component_parameters
        .parameters
        .precinct_exponents
        .push((15, 15));

    let (mut tiles, mut budget, _) = inherited_tile_state(&header);
    let mut bytes = vec![
        0xff, 0x90, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0x00, 0x01,
    ];
    // COC for component zero with no decomposition levels, followed by SOD.
    bytes.extend_from_slice(&[
        0xff, 0x53, 0x00, 0x09, 0x00, 0x00, 0x00, 0x04, 0x04, 0x00, 0x01, 0xff, 0x93,
    ]);
    let mut ppm_packet_idx = 0;

    let error = parse_tile_part(
        &mut BitReader::new(&bytes),
        &header,
        &mut tiles,
        &mut ppm_packet_idx,
        &mut budget,
    )
    .expect_err("tile COC must not undercut the requested reduction");

    assert!(matches!(
        error,
        DecodeError::Decoding(DecodingError::UnsupportedFeature(
            "tile coding style has fewer levels than the requested reduction"
        ))
    ));
}
