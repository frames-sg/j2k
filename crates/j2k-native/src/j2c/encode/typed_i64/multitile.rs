// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-tile typed high-bit orchestration.

use alloc::vec::Vec;

use super::super::allocation::{checked_add_bytes, checked_element_bytes};
use super::super::multitile::{
    append_encoded_tile_parts, encode_options_retained_bytes, finalize_multitile_codestream,
    quantization_retained_bytes, reserve_tile_parts,
};
use super::super::single_tile::ownership::encode_params_retained_bytes;
use super::super::tile_parts::{
    consume_packetized_tile_into_tile_parts, encoded_tile_parts_retained_bytes,
};
use super::super::{
    packetize_i64_component_resolution_packets, CpuOnlyJ2kEncodeStageAccelerator, EncodeOptions,
    EncodeTypedComponentPlane, I64PacketizeRequest, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession,
};
use super::geometry::min_sampled_tile_component_decomposition_levels;
use super::plan::{
    try_high_bit_options, try_precinct_exponents, TypedI64ExecutionRequest, TypedI64HighBitPlan,
};
use super::prepare::{prepare_typed_component_planes_i64_packets, TypedPlanePacketRequest};

mod input;
use input::{tile_plane_data_retained_bytes, try_extract_tile_planes, try_tile_plane_views};

pub(super) struct TypedI64MultiTileRequest<'a, 'input> {
    pub(super) planes: &'a [EncodeTypedComponentPlane<'a>],
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) options: &'a EncodeOptions,
    pub(super) tile_width: u32,
    pub(super) tile_height: u32,
    pub(super) num_components: u16,
    pub(super) session: &'a NativeEncodeSession<'input>,
}

#[expect(
    clippy::similar_names,
    reason = "paired axis and tile names follow JPEG 2000 notation"
)]
pub(super) fn encode_typed_component_planes_53_i64_multitile(
    request: &TypedI64MultiTileRequest<'_, '_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let num_x_tiles = request.width.div_ceil(request.tile_width);
    let num_y_tiles = request.height.div_ceil(request.tile_height);
    let num_tiles = num_x_tiles.checked_mul(num_y_tiles).ok_or(
        NativeEncodePipelineError::arithmetic_overflow("typed multi-tile tile count"),
    )?;
    if num_tiles > u32::from(u16::MAX) + 1 {
        return Err(NativeEncodePipelineError::unsupported(
            "multi-tile encode supports at most 65536 tiles",
        ));
    }
    let num_levels = min_sampled_tile_component_decomposition_levels(
        request.planes,
        request.width,
        request.height,
        request.tile_width,
        request.tile_height,
    )
    .map_err(NativeEncodePipelineError::arithmetic_overflow)?
    .min(request.options.num_decomposition_levels);
    let plan = TypedI64HighBitPlan::try_new(
        request.planes,
        request.options,
        num_levels,
        0,
        request.session,
    )?;
    let plan_bytes = plan.retained_bytes()?;
    let mut child_options = try_high_bit_options(
        request.options,
        plan.component_sampling(),
        num_levels,
        plan_bytes,
        request.session,
    )?;
    let options_bytes = encode_options_retained_bytes(&child_options)?;
    let precinct_exponents = try_precinct_exponents(
        &child_options,
        num_levels,
        checked_add_bytes(plan_bytes, options_bytes, "typed i64 multi-tile plan")?,
        request.session,
    )?;
    let execution = plan.try_into_execution(TypedI64ExecutionRequest {
        dimensions: (request.width, request.height),
        tile_dimensions: (request.tile_width, request.tile_height),
        num_components: request.num_components,
        options: request.options,
        precinct_exponents,
        retained_base_bytes: options_bytes,
        session: request.session,
    })?;
    child_options.tile_size = None;
    child_options.write_tlm = false;
    child_options.write_plt = false;
    child_options.write_plm = false;
    child_options.write_ppm = false;
    child_options.write_ppt = false;
    let planning_bytes = checked_add_bytes(
        options_bytes,
        execution.retained_bytes()?,
        "typed i64 multi-tile retained plan",
    )?;
    let tile_count = usize::try_from(num_tiles).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("typed multi-tile tile count exceeds usize")
    })?;
    let mut tile_bodies = reserve_tile_parts(tile_count, planning_bytes, request.session)?;

    for tile_y in 0..num_y_tiles {
        for tile_x in 0..num_x_tiles {
            encode_one_tile(
                request,
                tile_x,
                tile_y,
                num_x_tiles,
                num_levels,
                &execution,
                &child_options,
                planning_bytes,
                &mut tile_bodies,
            )?;
        }
    }

    drop(child_options);
    let (params, quant_params) = execution.into_final_parts();
    let final_planning_bytes = checked_add_bytes(
        encode_params_retained_bytes(&params)?,
        quantization_retained_bytes(&quant_params)?,
        "typed i64 multi-tile final plan",
    )?;
    finalize_multitile_codestream(
        &params,
        &tile_bodies,
        &quant_params,
        final_planning_bytes,
        request.session,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "one tile transition keeps validated geometry and every retained parent owner explicit"
)]
fn encode_one_tile(
    request: &TypedI64MultiTileRequest<'_, '_>,
    tile_x: u32,
    tile_y: u32,
    num_x_tiles: u32,
    num_levels: u8,
    execution: &super::plan::TypedI64ExecutionPlan,
    child_options: &EncodeOptions,
    planning_bytes: usize,
    tile_bodies: &mut Vec<super::super::tile_parts::EncodedTilePart>,
) -> NativeEncodePipelineResult<()> {
    let tile_index = tile_y
        .checked_mul(num_x_tiles)
        .and_then(|base| base.checked_add(tile_x))
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("typed tile index"))?;
    let tile_index = u16::try_from(tile_index).map_err(|_| {
        NativeEncodePipelineError::internal_invariant(
            "validated typed tile index exceeds the marker field",
        )
    })?;
    let x0 = tile_x
        .checked_mul(request.tile_width)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("typed tile x offset"))?;
    let y0 = tile_y
        .checked_mul(request.tile_height)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("typed tile y offset"))?;
    let actual_width = (request.width - x0).min(request.tile_width);
    let actual_height = (request.height - y0).min(request.tile_height);
    let accumulated_bytes = encoded_tile_parts_retained_bytes(tile_bodies, tile_bodies.capacity())?;
    let iteration_base = checked_add_bytes(
        planning_bytes,
        accumulated_bytes,
        "typed i64 multi-tile retained owners",
    )?;
    let tile_plane_data =
        try_extract_tile_planes(request, x0, y0, actual_width, actual_height, iteration_base)?;
    let tile_data_bytes = tile_plane_data_retained_bytes(&tile_plane_data)?;
    let tile_views = try_tile_plane_views(
        request.planes,
        &tile_plane_data,
        checked_add_bytes(
            iteration_base,
            tile_data_bytes,
            "typed i64 tile plane views",
        )?,
        request.session,
    )?;
    let view_bytes = checked_add_bytes(
        checked_element_bytes::<EncodeTypedComponentPlane<'_>>(
            tile_views.planes.capacity(),
            "typed i64 tile plane views",
        )?,
        checked_element_bytes::<(u32, u32)>(
            tile_views.component_dimensions.capacity(),
            "typed i64 tile component dimensions",
        )?,
        "typed i64 tile plane views",
    )?;
    let component_resolution_packets = prepare_typed_component_planes_i64_packets(
        &tile_views.planes,
        TypedPlanePacketRequest {
            component_dimensions: &tile_views.component_dimensions,
            component_step_sizes: &execution.component_step_sizes,
            num_levels,
            subband_settings: execution.subband_settings(request.options),
            retained_base_bytes: checked_add_bytes(
                iteration_base,
                checked_add_bytes(
                    tile_data_bytes,
                    view_bytes,
                    "typed i64 tile preparation scratch",
                )?,
                "typed i64 tile preparation",
            )?,
            session: request.session,
        },
    )?;
    drop(tile_views);
    drop(tile_plane_data);

    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    let packetized_tile = packetize_i64_component_resolution_packets(
        component_resolution_packets,
        I64PacketizeRequest {
            width: actual_width,
            height: actual_height,
            num_components: request.num_components,
            num_levels,
            params: &execution.params,
            options: child_options,
            retained_base_bytes: iteration_base,
            session: request.session,
            accelerator: &mut accelerator,
        },
    )?;
    let new_parts = consume_packetized_tile_into_tile_parts(
        tile_index,
        packetized_tile,
        request.options.tile_part_packet_limit,
        iteration_base,
        request.session,
    )?;
    append_encoded_tile_parts(tile_bodies, new_parts, planning_bytes, 0, request.session)
}
