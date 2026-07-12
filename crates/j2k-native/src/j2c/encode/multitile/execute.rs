// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grid-level standard multi-tile orchestration.

use super::super::{J2kEncodeStageAccelerator, NativeEncodePipelineResult, Vec};
use super::plan::{build_loop_plan, FinalPlanRequest, LoopPlanRequest};
use super::tile::{encode_tile, TileGrid, TilePosition};
use super::{
    finalize_multitile_codestream, quantization_retained_bytes, reserve_tile_parts,
    MultiTileEncodeRequest,
};

pub(in crate::j2c::encode) fn encode_multitile_impl(
    request: &MultiTileEncodeRequest<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let grid = TileGrid::try_new(request)?;
    let loop_plan = build_loop_plan(&LoopPlanRequest {
        width: request.width,
        height: request.height,
        tile_width: request.tile_width,
        tile_height: request.tile_height,
        num_components: request.num_components,
        bit_depth: request.bit_depth,
        options: request.options,
        roi_regions: request.roi_regions,
        component_sample_info: request.component_sample_info,
        block_coding_mode: request.block_coding_mode,
        session: request.session,
    })?;
    let loop_retained_bytes = loop_plan.retained_bytes();
    let mut tile_bodies =
        reserve_tile_parts(grid.tile_count(), loop_retained_bytes, request.session)?;

    for row in 0..grid.rows() {
        for column in 0..grid.columns() {
            let position = TilePosition::try_new(request, &grid, row, column)?;
            encode_tile(
                request,
                &loop_plan,
                loop_retained_bytes,
                &mut tile_bodies,
                position,
                accelerator,
            )?;
        }
    }

    let final_plan = loop_plan.into_final_plan(&FinalPlanRequest {
        width: request.width,
        height: request.height,
        tile_width: request.tile_width,
        tile_height: request.tile_height,
        num_components: request.num_components,
        bit_depth: request.bit_depth,
        signed: request.signed,
        options: request.options,
        roi_regions: request.roi_regions,
        component_sample_info: request.component_sample_info,
        block_coding_mode: request.block_coding_mode,
        tile_bodies: &tile_bodies,
        session: request.session,
    })?;
    let final_planning_bytes = quantization_retained_bytes(&final_plan.quant_params)?;
    finalize_multitile_codestream(
        &final_plan.params,
        &tile_bodies,
        &final_plan.quant_params,
        final_planning_bytes,
        request.session,
    )
}
