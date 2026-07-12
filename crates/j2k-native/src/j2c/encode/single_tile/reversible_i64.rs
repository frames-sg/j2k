// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    forward_rct_i64, packet_encode, packetize_i64_component_resolution_packets, EncodeOptions,
    I64PacketizeRequest, J2kEncodeStageAccelerator, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession, MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES,
};
use super::ownership::single_tile_plan_retained_bytes;
use super::plan::SingleTilePlan;

mod input;
mod prepare;
use input::{try_deinterleave_to_i64, I64DeinterleaveRequest};
use prepare::prepare_i64_component_packet_tree;

pub(super) struct ReversibleI64SingleTileRequest<'a, 'input, A: J2kEncodeStageAccelerator> {
    pub(super) pixels: &'a [u8],
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) num_components: u16,
    pub(super) bit_depth: u8,
    pub(super) signed: bool,
    pub(super) options: &'a EncodeOptions,
    pub(super) plan: SingleTilePlan,
    pub(super) session: &'a NativeEncodeSession<'input>,
    pub(super) accelerator: &'a mut A,
}

pub(super) fn encode_reversible_i64_single_tile_packets<A: J2kEncodeStageAccelerator>(
    request: ReversibleI64SingleTileRequest<'_, '_, A>,
) -> NativeEncodePipelineResult<(packet_encode::PacketizedTileData, SingleTilePlan)> {
    let ReversibleI64SingleTileRequest {
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        plan,
        session,
        accelerator,
    } = request;
    let max_reversible_gain = if plan.num_levels == 0 { 0 } else { 2 };
    if u16::from(bit_depth) + max_reversible_gain > MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES {
        return Err(NativeEncodePipelineError::unsupported(
            "25-38 bit reversible encode exceeds the current no-quantization guard/exponent signaling limit",
        ));
    }
    let retained_plan_bytes = single_tile_plan_retained_bytes(&plan)?;
    let mut components = try_deinterleave_to_i64(&I64DeinterleaveRequest {
        pixels,
        num_pixels: plan.num_pixels,
        num_components,
        bit_depth,
        signed,
        retained_base_bytes: retained_plan_bytes,
        session,
    })?;
    if plan.use_mct {
        forward_rct_i64(&mut components);
    }
    let component_resolution_packets =
        prepare_i64_component_packet_tree(&mut components, &plan, retained_plan_bytes, session)?;
    drop(components);
    let packetized_tile = packetize_i64_component_resolution_packets(
        component_resolution_packets,
        I64PacketizeRequest {
            width,
            height,
            num_components,
            num_levels: plan.num_levels,
            params: &plan.params,
            options,
            retained_base_bytes: retained_plan_bytes,
            session,
            accelerator,
        },
    )?;
    Ok((packetized_tile, plan))
}
