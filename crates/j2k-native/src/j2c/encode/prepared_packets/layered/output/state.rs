// SPDX-License-Identifier: MIT OR Apache-2.0

//! Construction of one progression state across every quality layer.

use super::super::super::super::allocation::checked_add_bytes;
use super::super::super::super::tier1_allocation::{
    resolution_packet_ownership, subband_precincts_ownership, Tier1PhaseTracker,
};
use super::super::super::super::{
    CodeBlockPacketData, J2kPacketizationPacketDescriptor, LayeredPreparedPacket,
    LayeredPreparedSubband, NativeEncodePipelineError, NativeEncodePipelineResult,
    ResolutionPacket, SubbandPrecinct, Vec,
};
use super::contributions::{build_block_contributions, ContributionOwners};

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerOutputContext {
    pub(super) num_layers: u8,
    pub(super) layer_count: usize,
    pub(super) fixed: usize,
    pub(super) resolution_owner_bytes: usize,
    pub(super) resolution_capacity: usize,
    pub(super) descriptor_bytes: usize,
}

pub(super) fn append_layered_packet(
    layered_packet: LayeredPreparedPacket,
    state_index: usize,
    resolution_packets: &mut Vec<ResolutionPacket>,
    descriptors: &mut Vec<J2kPacketizationPacketDescriptor>,
    context: LayerOutputContext,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    let (mut layer_packets, _) = tracker.try_vec::<ResolutionPacket>(
        context.layer_count,
        [
            context.fixed,
            context.resolution_owner_bytes,
            context.descriptor_bytes,
        ],
        "per-layer resolution packet owners",
    )?;
    for _ in 0..context.layer_count {
        let current_bytes = resolution_packet_ownership(&layer_packets, layer_packets.capacity())?;
        let (subbands, _) = tracker.try_vec::<SubbandPrecinct>(
            layered_packet.subbands.len(),
            [
                context.fixed,
                context.resolution_owner_bytes,
                context.descriptor_bytes,
                current_bytes,
            ],
            "per-layer subband owners",
        )?;
        layer_packets.push(ResolutionPacket { subbands });
    }
    for subband in layered_packet.subbands {
        append_layered_subband(
            subband,
            resolution_packets,
            &mut layer_packets,
            context,
            tracker,
        )?;
    }

    let state_index = u32::try_from(state_index).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("packet descriptor state index exceeds u32")
    })?;
    for (layer_idx, layer_packet) in layer_packets.into_iter().enumerate() {
        let packet_index = u32::try_from(resolution_packets.len()).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("packet descriptor index exceeds u32")
        })?;
        resolution_packets.push(layer_packet);
        descriptors.push(J2kPacketizationPacketDescriptor {
            packet_index,
            state_index,
            layer: u8::try_from(layer_idx).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("quality layer index exceeds u8")
            })?,
            resolution: layered_packet.resolution,
            component: layered_packet.component,
            precinct: layered_packet.precinct,
        });
    }
    Ok(())
}

fn append_layered_subband(
    subband: LayeredPreparedSubband,
    resolution_packets: &[ResolutionPacket],
    layer_packets: &mut Vec<ResolutionPacket>,
    context: LayerOutputContext,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    let mut layer_subbands = Vec::new();
    let current_layer_packet_bytes =
        resolution_packet_ownership(layer_packets, layer_packets.capacity())?;
    let layer_subband_outer = tracker.try_reserve_additional(
        &mut layer_subbands,
        context.layer_count,
        [
            context.fixed,
            context.resolution_owner_bytes,
            context.descriptor_bytes,
            current_layer_packet_bytes,
        ],
        "layer contribution subband owners",
    )?;
    let mut layer_code_block_bytes = 0usize;
    for _ in 0..context.layer_count {
        let (code_blocks, bytes) = tracker.try_vec::<CodeBlockPacketData>(
            subband.blocks.len(),
            [
                context.fixed,
                context.resolution_owner_bytes,
                context.descriptor_bytes,
                current_layer_packet_bytes,
                layer_subband_outer,
                layer_code_block_bytes,
            ],
            "layer contribution code-block owners",
        )?;
        layer_code_block_bytes = checked_add_bytes(
            layer_code_block_bytes,
            bytes,
            "layer contribution code-block graph",
        )?;
        layer_subbands.push(SubbandPrecinct {
            code_blocks,
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
        });
    }

    for block in subband.blocks {
        let current_output_bytes =
            resolution_packet_ownership(resolution_packets, context.resolution_capacity)?;
        let local_subband_bytes =
            subband_precincts_ownership(&layer_subbands, layer_subbands.capacity())?;
        let local_layer_packet_bytes =
            resolution_packet_ownership(layer_packets, layer_packets.capacity())?;
        let contributions = build_block_contributions(
            block,
            context.num_layers,
            ContributionOwners {
                fixed: context.fixed,
                current_output_bytes,
                local_subband_bytes,
                local_layer_packet_bytes,
            },
            tracker,
        )?;
        if contributions.len() != layer_subbands.len() {
            return Err(NativeEncodePipelineError::internal_invariant(
                "layer contribution count mismatch",
            ));
        }
        for (layer_subband, contribution) in layer_subbands.iter_mut().zip(contributions) {
            layer_subband.code_blocks.push(contribution);
        }
    }
    for (layer_packet, layer_subband) in layer_packets.iter_mut().zip(layer_subbands) {
        layer_packet.subbands.push(layer_subband);
    }
    Ok(())
}
