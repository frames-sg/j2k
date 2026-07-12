// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move-only construction of packet contributions and descriptors.

use super::super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::super::{
    EncodeProgressionOrder, J2kPacketizationPacketDescriptor, LayeredPreparedPacket,
    NativeEncodePipelineResult, ResolutionPacket, Vec,
};
use super::ownership::layered_packets_ownership;

mod contributions;
mod state;
use state::{append_layered_packet, LayerOutputContext};

pub(super) fn build_layer_packets(
    layered_packets: Vec<LayeredPreparedPacket>,
    num_layers: u8,
    progression_order: EncodeProgressionOrder,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<(Vec<ResolutionPacket>, Vec<J2kPacketizationPacketDescriptor>)> {
    let layer_count = usize::from(num_layers);
    let output_count = layered_packets.len().checked_mul(layer_count).ok_or(
        crate::EncodeError::ArithmeticOverflow {
            what: "layered resolution packet count",
        },
    )?;
    let layered_bytes = layered_packets_ownership(&layered_packets, layered_packets.capacity())?;
    let fixed = layered_bytes;
    let (mut resolution_packets, resolution_owner_bytes) = tracker.try_vec::<ResolutionPacket>(
        output_count,
        [fixed],
        "layered resolution packet owners",
    )?;
    let (mut descriptors, descriptor_bytes) = tracker.try_vec::<J2kPacketizationPacketDescriptor>(
        output_count,
        [fixed, resolution_owner_bytes],
        "layered packet descriptors",
    )?;

    let context = LayerOutputContext {
        num_layers,
        layer_count,
        fixed,
        resolution_owner_bytes,
        resolution_capacity: resolution_packets.capacity(),
        descriptor_bytes,
    };
    for (state_index, layered_packet) in layered_packets.into_iter().enumerate() {
        append_layered_packet(
            layered_packet,
            state_index,
            &mut resolution_packets,
            &mut descriptors,
            context,
            tracker,
        )?;
    }
    crate::sort_packet_descriptors_for_progression(
        &mut descriptors,
        progression_order.packetization_order(),
    );
    Ok((resolution_packets, descriptors))
}
