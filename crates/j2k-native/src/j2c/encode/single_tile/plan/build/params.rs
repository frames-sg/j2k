// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-allocating assembly of marker parameters from already-owned vectors.

use crate::j2c::encode::{BlockCodingMode, EncodeParams};

use super::owners::EncodeParamOwners;
use super::{BuildRequest, PlanGeometry};

pub(super) fn build_encode_params(
    request: &BuildRequest<'_>,
    geometry: PlanGeometry,
    owners: EncodeParamOwners,
) -> EncodeParams {
    EncodeParams {
        width: request.width,
        height: request.height,
        tile_width: request
            .options
            .tile_size
            .map_or(request.width, |(tile_width, _)| tile_width),
        tile_height: request
            .options
            .tile_size
            .map_or(request.height, |(_, tile_height)| tile_height),
        num_components: request.num_components,
        bit_depth: request.bit_depth,
        signed: request.signed,
        component_sample_info: owners.component_sample_info,
        component_quantization_step_sizes: owners.component_quantization_step_sizes,
        num_decomposition_levels: geometry.num_levels,
        reversible: request.options.reversible,
        code_block_width_exp: request.options.code_block_width_exp,
        code_block_height_exp: request.options.code_block_height_exp,
        num_layers: request.options.num_layers,
        use_mct: geometry.use_mct,
        guard_bits: geometry.guard_bits,
        block_coding_mode: request.block_coding_mode,
        progression_order: request.options.progression_order,
        write_tlm: request.options.write_tlm,
        write_plt: request.options.write_plt,
        write_plm: request.options.write_plm,
        write_ppm: request.options.write_ppm,
        write_ppt: request.options.write_ppt,
        write_sop: request.options.write_sop,
        write_eph: request.options.write_eph,
        terminate_coding_passes: request.block_coding_mode == BlockCodingMode::Classic
            && request.options.num_layers > 1,
        component_sampling: owners.component_sampling,
        roi_component_shifts: owners.roi_component_shifts,
        precinct_exponents: owners.precinct_exponents,
    }
}
