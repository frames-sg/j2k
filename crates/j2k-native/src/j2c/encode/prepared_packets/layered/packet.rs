// SPDX-License-Identifier: MIT OR Apache-2.0

//! One prepared packet's move-only layered Tier-1 construction.

use super::super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::super::{
    BlockCodingMode, J2kEncodeStageAccelerator, LayeredPreparedBlock, LayeredPreparedPacket,
    LayeredPreparedSubband, NativeEncodePipelineResult, NativeEncodeSession,
    PreparedResolutionPacket, Vec,
};
use super::classic::{encode_layered_classic_subband, LayeredClassicContext};
use super::ht::{encode_layered_ht_subband, LayeredHtContext};
use super::ownership::{layered_packet_ownership, layered_packets_ownership};
use super::state::LayeredRateControlState;

#[derive(Clone, Copy)]
pub(super) struct LayeredPacketContext<'a, 'session> {
    pub(super) num_layers: u8,
    pub(super) quality_layer_byte_targets: &'a [u64],
    pub(super) source_bytes: usize,
    pub(super) session: &'a NativeEncodeSession<'session>,
    pub(super) retained_base_bytes: usize,
}

pub(super) fn append_layered_prepared_packet(
    prepared_packet: PreparedResolutionPacket,
    layered_packets: &mut Vec<LayeredPreparedPacket>,
    rate_control: &mut LayeredRateControlState,
    context: LayeredPacketContext<'_, '_>,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    let packet_idx = layered_packets.len();
    let current_live = rate_control.live_bytes(
        context.source_bytes,
        layered_packets_ownership(layered_packets, layered_packets.capacity())?,
    )?;
    let (subbands, _) = tracker.try_vec::<LayeredPreparedSubband>(
        prepared_packet.subbands.len(),
        [current_live],
        "layered subband owners",
    )?;
    let mut layered_packet = LayeredPreparedPacket {
        component: prepared_packet.component,
        resolution: prepared_packet.resolution,
        precinct: prepared_packet.precinct,
        subbands,
    };

    for subband in prepared_packet.subbands {
        let subband_idx = layered_packet.subbands.len();
        let current_live = rate_control.live_bytes(
            context.source_bytes,
            layered_packet_ownership(&layered_packet)?,
        )?;
        let (blocks, _) = tracker.try_vec::<LayeredPreparedBlock>(
            subband.code_blocks.len(),
            [current_live],
            "layered code-block owners",
        )?;
        let mut layered_subband = LayeredPreparedSubband {
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
            blocks,
        };
        match subband.block_coding_mode {
            BlockCodingMode::Classic => encode_layered_classic_subband(
                subband,
                &mut layered_subband,
                rate_control,
                LayeredClassicContext {
                    packet_idx,
                    subband_idx,
                    num_layers: context.num_layers,
                    quality_layer_byte_targets: context.quality_layer_byte_targets,
                    source_bytes: context.source_bytes,
                    layered_packets,
                    layered_packet_capacity: layered_packets.capacity(),
                    layered_packet: &layered_packet,
                },
                tracker,
            )?,
            BlockCodingMode::HighThroughput => encode_layered_ht_subband(
                subband,
                &mut layered_subband,
                rate_control,
                LayeredHtContext {
                    packet_idx,
                    subband_idx,
                    num_layers: context.num_layers,
                    quality_layer_byte_targets: context.quality_layer_byte_targets,
                    source_bytes: context.source_bytes,
                    layered_packets,
                    layered_packet_capacity: layered_packets.capacity(),
                    layered_packet: &layered_packet,
                    session: context.session,
                    retained_base_bytes: context.retained_base_bytes,
                },
                tracker,
                accelerator,
            )?,
        }
        layered_packet.subbands.push(layered_subband);
    }
    layered_packets.push(layered_packet);
    Ok(())
}
