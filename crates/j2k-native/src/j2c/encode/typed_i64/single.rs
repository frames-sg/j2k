// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single-tile typed high-bit orchestration.

use alloc::vec::Vec;

use super::super::allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed};
use super::super::multitile::{encode_options_retained_bytes, quantization_retained_bytes};
use super::super::single_tile::ownership::encode_params_retained_bytes;
use super::super::{
    max_decomposition_levels, packetize_i64_component_resolution_packets,
    write_single_tile_packetized_codestream_for_session, CpuOnlyJ2kEncodeStageAccelerator,
    EncodeOptions, EncodeTypedComponentPlane, I64PacketizeRequest, NativeEncodePipelineResult,
    NativeEncodeSession,
};
use super::plan::{
    try_high_bit_options, try_precinct_exponents, TypedI64ExecutionRequest, TypedI64HighBitPlan,
};
use super::prepare::{prepare_typed_component_planes_i64_packets, TypedPlanePacketRequest};

pub(super) fn encode_typed_component_planes_53_i64_single(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    num_components: u16,
    options: &EncodeOptions,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let num_levels = planes
        .iter()
        .map(|plane| {
            let component_width = width.div_ceil(u32::from(plane.x_rsiz));
            let component_height = height.div_ceil(u32::from(plane.y_rsiz));
            max_decomposition_levels(component_width, component_height)
        })
        .min()
        .unwrap_or(0)
        .min(options.num_decomposition_levels);
    let plan = TypedI64HighBitPlan::try_new(planes, options, num_levels, 0, session)?;
    let plan_bytes = plan.retained_bytes()?;
    let high_bit_options = try_high_bit_options(
        options,
        plan.component_sampling(),
        num_levels,
        plan_bytes,
        session,
    )?;
    let options_bytes = encode_options_retained_bytes(&high_bit_options)?;
    let precinct_exponents = try_precinct_exponents(
        &high_bit_options,
        num_levels,
        checked_add_bytes(plan_bytes, options_bytes, "typed i64 single plan")?,
        session,
    )?;
    let tile_dimensions = options.tile_size.unwrap_or((width, height));
    let execution = plan.try_into_execution(TypedI64ExecutionRequest {
        dimensions: (width, height),
        tile_dimensions,
        num_components,
        options,
        precinct_exponents,
        retained_base_bytes: options_bytes,
        session,
    })?;
    let execution_bytes = execution.retained_bytes()?;
    let planning_bytes = checked_add_bytes(
        options_bytes,
        execution_bytes,
        "typed i64 single retained plan",
    )?;
    let component_dimensions =
        try_component_dimensions(planes, width, height, planning_bytes, session)?;
    let dimension_bytes = checked_element_bytes::<(u32, u32)>(
        component_dimensions.capacity(),
        "typed i64 component dimensions",
    )?;
    let component_resolution_packets = prepare_typed_component_planes_i64_packets(
        planes,
        TypedPlanePacketRequest {
            component_dimensions: &component_dimensions,
            component_step_sizes: &execution.component_step_sizes,
            num_levels,
            subband_settings: execution.subband_settings(options),
            retained_base_bytes: checked_add_bytes(
                planning_bytes,
                dimension_bytes,
                "typed i64 single preparation",
            )?,
            session,
        },
    )?;
    drop(component_dimensions);

    let tile_part_packet_limit = high_bit_options.tile_part_packet_limit;
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    let packetized_tile = packetize_i64_component_resolution_packets(
        component_resolution_packets,
        I64PacketizeRequest {
            width,
            height,
            num_components,
            num_levels,
            params: &execution.params,
            options: &high_bit_options,
            retained_base_bytes: planning_bytes,
            session,
            accelerator: &mut accelerator,
        },
    )?;
    drop(high_bit_options);
    let (params, quant_params) = execution.into_final_parts();
    let final_planning_bytes = checked_add_bytes(
        encode_params_retained_bytes(&params)?,
        quantization_retained_bytes(&quant_params)?,
        "typed i64 single final plan",
    )?;
    write_single_tile_packetized_codestream_for_session(
        &params,
        &packetized_tile,
        &quant_params,
        tile_part_packet_limit,
        final_planning_bytes,
        session,
    )
}

fn try_component_dimensions(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    retained_base_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<(u32, u32)>> {
    let requested =
        checked_element_bytes::<(u32, u32)>(planes.len(), "typed i64 component dimensions")?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            requested,
            "typed i64 component dimensions",
        )?,
        "typed i64 component dimensions",
    )?;
    let mut dimensions = Vec::new();
    dimensions
        .try_reserve_exact(planes.len())
        .map_err(|_| host_allocation_failed("typed i64 component dimensions", requested))?;
    dimensions.extend(planes.iter().map(|plane| {
        (
            width.div_ceil(u32::from(plane.x_rsiz)),
            height.div_ceil(u32::from(plane.y_rsiz)),
        )
    }));
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_element_bytes::<(u32, u32)>(
                dimensions.capacity(),
                "typed i64 component dimensions",
            )?,
            "typed i64 component dimensions",
        )?,
        "typed i64 component dimensions",
    )?;
    Ok(dimensions)
}
