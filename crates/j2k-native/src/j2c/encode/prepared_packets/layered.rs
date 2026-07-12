// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible multi-layer Tier-1 and rate-control state machine.

use super::super::tier1_allocation::{prepared_packets_ownership, Tier1PhaseTracker};
use super::super::{
    EncodeProgressionOrder, J2kEncodeStageAccelerator, J2kPacketizationPacketDescriptor,
    LayeredPreparedPacket, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeSession, PreparedResolutionPacket, ResolutionPacket, Vec,
};

mod assignment;
use assignment::apply_budget_assignments;
mod classic;
mod ht;
mod output;
use output::build_layer_packets;
mod ownership;
mod packet;
use packet::{append_layered_prepared_packet, LayeredPacketContext};
mod state;
use state::LayeredRateControlState;
#[cfg(test)]
mod tests;

pub(in crate::j2c::encode) fn encode_prepared_resolution_packets_layered_for_session(
    prepared_packets: Vec<PreparedResolutionPacket>,
    num_layers: u8,
    progression_order: EncodeProgressionOrder,
    quality_layer_byte_targets: &[u64],
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<(Vec<ResolutionPacket>, Vec<J2kPacketizationPacketDescriptor>)> {
    encode_prepared_resolution_packets_layered_accounted(
        prepared_packets,
        num_layers,
        progression_order,
        quality_layer_byte_targets,
        session,
        retained_base_bytes,
        accelerator,
    )
    .map(|outcome| (outcome.packets, outcome.descriptors))
}

struct LayeredEncodeOutcome {
    packets: Vec<ResolutionPacket>,
    descriptors: Vec<J2kPacketizationPacketDescriptor>,
    #[cfg(test)]
    peak_phase_bytes: usize,
}

fn encode_prepared_resolution_packets_layered_accounted(
    prepared_packets: Vec<PreparedResolutionPacket>,
    num_layers: u8,
    progression_order: EncodeProgressionOrder,
    quality_layer_byte_targets: &[u64],
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<LayeredEncodeOutcome> {
    let layer_count = usize::from(num_layers);
    if layer_count == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "quality layer count must be non-zero",
        ));
    }
    let source = prepared_packets_ownership(&prepared_packets, prepared_packets.capacity())?;
    let source_bytes = source.total()?;
    let packet_count = prepared_packets.len();
    let mut tracker = Tier1PhaseTracker::new(session, retained_base_bytes);
    let (mut layered_packets, _) = tracker.try_vec::<LayeredPreparedPacket>(
        packet_count,
        [source_bytes],
        "layered packet owners",
    )?;
    let mut rate_control = LayeredRateControlState::default();

    for prepared_packet in prepared_packets {
        append_layered_prepared_packet(
            prepared_packet,
            &mut layered_packets,
            &mut rate_control,
            LayeredPacketContext {
                num_layers,
                quality_layer_byte_targets,
                source_bytes,
                session,
                retained_base_bytes,
            },
            &mut tracker,
            accelerator,
        )?;
    }

    let layered_packet_capacity = layered_packets.capacity();
    apply_budget_assignments(
        &mut layered_packets,
        layered_packet_capacity,
        &rate_control,
        layer_count,
        quality_layer_byte_targets,
        &mut tracker,
    )?;
    let (packets, descriptors) =
        build_layer_packets(layered_packets, num_layers, progression_order, &mut tracker)?;
    Ok(LayeredEncodeOutcome {
        packets,
        descriptors,
        #[cfg(test)]
        peak_phase_bytes: tracker.peak_phase_bytes(),
    })
}
