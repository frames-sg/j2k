// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::{Header, Tile, TileMetadataBudget};
use crate::j2c::codestream::{
    CodeBlockStyle, CodingStyleComponent, CodingStyleDefault, CodingStyleFlags,
    CodingStyleParameters, ComponentInfo, ComponentSizeInfo, ProgressionOrder, QuantizationInfo,
    QuantizationStyle, SizeData, StepSize, WaveletTransform,
};

pub(super) fn header() -> Header<'static> {
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

pub(super) fn inherited_tile_state<'a>(
    header: &'a Header<'a>,
) -> (Vec<Tile<'a>>, TileMetadataBudget, usize) {
    let mut budget = TileMetadataBudget::for_image(header, 0).expect("tile budget");
    let mut tiles = Vec::new();
    budget
        .try_reserve_retained(&mut tiles, 1)
        .expect("outer tile owner");
    tiles.push(Tile::new(0, header));
    super::super::metadata::inherit_tile_metadata(&mut tiles[0], header, &mut budget)
        .expect("inherited tile metadata");
    let retained_before = budget.retained_bytes();
    (tiles, budget, retained_before)
}

pub(super) fn tile_part_bytes(
    tile_index: u16,
    tile_part_length: u32,
    include_sod: bool,
) -> Vec<u8> {
    let mut bytes = vec![0xff, 0x90, 0x00, 0x0a];
    bytes.extend_from_slice(&tile_index.to_be_bytes());
    bytes.extend_from_slice(&tile_part_length.to_be_bytes());
    bytes.extend_from_slice(&[0x00, 0x01]);
    if include_sod {
        bytes.extend_from_slice(&[0xff, 0x93]);
    }
    bytes
}

mod sot;
mod transaction;
