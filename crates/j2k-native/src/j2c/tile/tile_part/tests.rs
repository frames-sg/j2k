// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::*;
use crate::error::DecodeError;
use crate::j2c::codestream::{
    CodeBlockStyle, CodingStyleComponent, CodingStyleDefault, CodingStyleFlags,
    CodingStyleParameters, ComponentInfo, ComponentSizeInfo, ProgressionOrder, QuantizationInfo,
    QuantizationStyle, SizeData, StepSize, WaveletTransform,
};

fn header() -> Header<'static> {
    let size_info = ComponentSizeInfo {
        precision: 8,
        signed: false,
        horizontal_resolution: 1,
        vertical_resolution: 1,
    };
    let parameters = || CodingStyleParameters {
        num_decomposition_levels: 0,
        num_resolution_levels: 1,
        code_block_width: 6,
        code_block_height: 6,
        code_block_style: CodeBlockStyle::default(),
        transformation: WaveletTransform::Reversible53,
        precinct_exponents: vec![(15, 15)],
    };
    let component = ComponentInfo {
        size_info,
        coding_style: CodingStyleComponent {
            flags: CodingStyleFlags::default(),
            parameters: parameters(),
        },
        quantization_info: QuantizationInfo {
            quantization_style: QuantizationStyle::NoQuantization,
            guard_bits: 2,
            step_sizes: vec![StepSize {
                mantissa: 0,
                exponent: 8,
            }],
        },
        roi_shift: 0,
    };

    Header {
        size_data: SizeData {
            reference_grid_width: 1,
            reference_grid_height: 1,
            image_area_x_offset: 0,
            image_area_y_offset: 0,
            tile_width: 1,
            tile_height: 1,
            tile_x_offset: 0,
            tile_y_offset: 0,
            component_sizes: vec![size_info],
            x_shrink_factor: 1,
            y_shrink_factor: 1,
            x_resolution_shrink_factor: 1,
            y_resolution_shrink_factor: 1,
        },
        global_coding_style: CodingStyleDefault {
            progression_order: ProgressionOrder::LayerResolutionComponentPosition,
            num_layers: 1,
            mct: false,
            component_parameters: CodingStyleComponent {
                flags: CodingStyleFlags::default(),
                parameters: parameters(),
            },
        },
        component_infos: vec![component],
        progression_changes: Vec::new(),
        plm_packet_lengths: Vec::new(),
        ppm_packets: Vec::new(),
        skipped_resolution_levels: 0,
        strict: true,
    }
}

#[test]
fn malformed_ppt_rolls_back_temporary_owner_capacity() {
    let header = header();
    let mut budget = TileMetadataBudget::for_image(&header, 0).expect("tile budget");
    let mut tiles = Vec::new();
    budget
        .try_reserve_retained(&mut tiles, 1)
        .expect("outer tile owner");
    tiles.push(Tile::new(0, &header));
    super::super::metadata::inherit_tile_metadata(&mut tiles[0], &header, &mut budget)
        .expect("inherited tile metadata");
    let retained_before = budget.retained_bytes();

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
