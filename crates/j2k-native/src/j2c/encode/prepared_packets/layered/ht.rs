// SPDX-License-Identifier: MIT OR Apache-2.0

//! High-throughput Tier-1 handoff for layered packet construction.

use super::super::super::allocation::checked_add_bytes;
use super::super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::super::{
    bitplane_encode, ht_segment_count, ht_segment_rate, ht_unbudgeted_segment_layers_accounted,
    HtSegmentAssignmentCandidate, HtSegmentLocation, J2kEncodeStageAccelerator,
    LayeredPreparedBlock, LayeredPreparedPacket, LayeredPreparedSubband,
    NativeEncodePipelineResult, NativeEncodeSession, PreparedEncodeSubband, Vec,
};
use super::ownership::{checked_sum, layered_block_build_owner_bytes};
use super::state::LayeredRateControlState;

mod prepare;
use prepare::try_encode_layered_ht_output;

#[derive(Clone, Copy)]
pub(super) struct LayeredHtContext<'a, 'session> {
    pub(super) packet_idx: usize,
    pub(super) subband_idx: usize,
    pub(super) num_layers: u8,
    pub(super) quality_layer_byte_targets: &'a [u64],
    pub(super) source_bytes: usize,
    pub(super) layered_packets: &'a [LayeredPreparedPacket],
    pub(super) layered_packet_capacity: usize,
    pub(super) layered_packet: &'a LayeredPreparedPacket,
    pub(super) session: &'a NativeEncodeSession<'session>,
    pub(super) retained_base_bytes: usize,
}

#[derive(Clone, Copy)]
struct HtSegmentContext {
    packet_idx: usize,
    subband_idx: usize,
    block_idx: usize,
    block_count: usize,
    layered_block_idx: usize,
    num_layers: u8,
    budgeted: bool,
    layered_owners: usize,
    complete_ht_output_bytes: usize,
    layered_live: usize,
}

pub(super) fn encode_layered_ht_subband(
    subband: PreparedEncodeSubband,
    layered_subband: &mut LayeredPreparedSubband,
    rate_control: &mut LayeredRateControlState,
    context: LayeredHtContext<'_, '_>,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    let mut output = try_encode_layered_ht_output(
        subband,
        layered_subband,
        rate_control.owner_bytes()?,
        context,
        tracker,
        accelerator,
    )?;
    let block_count = output.blocks.len();
    for (block_idx, block) in output.blocks.drain(..).enumerate() {
        let encoded = bitplane_encode::EncodedCodeBlock {
            data: block.data,
            num_coding_passes: block.num_coding_passes,
            num_zero_bitplanes: block.num_zero_bitplanes,
            ht_cleanup_length: block.ht_cleanup_length,
            ht_refinement_length: block.ht_refinement_length,
        };
        let current_payload_bytes = encoded.data.capacity();
        let other_ht_payload_bytes = output
            .remaining_payload_bytes
            .checked_sub(current_payload_bytes)
            .ok_or(crate::EncodeError::InternalInvariant {
                what: "layered HT remaining payload underflowed",
            })?;
        let other_ht_output_bytes = checked_add_bytes(
            output.structural_bytes,
            other_ht_payload_bytes,
            "layered HT remaining output owners",
        )?;
        let complete_ht_output_bytes = checked_add_bytes(
            other_ht_output_bytes,
            current_payload_bytes,
            "layered HT output owners",
        )?;
        let layered_owners = layered_block_build_owner_bytes(
            output.other_source_bytes,
            context.layered_packets,
            context.layered_packet_capacity,
            context.layered_packet,
            layered_subband,
        )?;
        let layered_live = checked_sum(
            [
                layered_owners,
                other_ht_output_bytes,
                rate_control.owner_bytes()?,
            ],
            "layered HT block owners",
        )?;
        let segment_layers = ht_segment_layers(
            &encoded,
            rate_control,
            HtSegmentContext {
                packet_idx: context.packet_idx,
                subband_idx: context.subband_idx,
                block_idx,
                block_count,
                layered_block_idx: layered_subband.blocks.len(),
                num_layers: context.num_layers,
                budgeted: !context.quality_layer_byte_targets.is_empty(),
                layered_owners,
                complete_ht_output_bytes,
                layered_live,
            },
            tracker,
        )?;
        layered_subband
            .blocks
            .push(LayeredPreparedBlock::HighThroughput {
                encoded,
                segment_layers,
            });
        output.remaining_payload_bytes = other_ht_payload_bytes;
        rate_control.ht_block_index = rate_control.ht_block_index.checked_add(1).ok_or(
            crate::EncodeError::ArithmeticOverflow {
                what: "HTJ2K segment block index",
            },
        )?;
    }
    Ok(())
}

fn ht_segment_layers(
    encoded: &bitplane_encode::EncodedCodeBlock,
    rate_control: &mut LayeredRateControlState,
    context: HtSegmentContext,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<Vec<usize>> {
    if !context.budgeted {
        return ht_unbudgeted_segment_layers_accounted(
            encoded,
            context.num_layers,
            context.block_idx,
            context.block_count,
            tracker,
            context.layered_live,
        );
    }
    let segment_count = ht_segment_count(encoded);
    rate_control.ht_candidate_bytes = tracker.try_reserve_additional(
        &mut rate_control.ht_candidates,
        segment_count,
        [
            context.layered_owners,
            context.complete_ht_output_bytes,
            rate_control.ht_location_bytes,
            rate_control.classic_candidate_bytes,
            rate_control.classic_location_bytes,
        ],
        "HT PCRD candidates",
    )?;
    rate_control.ht_location_bytes = tracker.try_reserve_additional(
        &mut rate_control.ht_locations,
        segment_count,
        [
            context.layered_owners,
            context.complete_ht_output_bytes,
            rate_control.ht_candidate_bytes,
            rate_control.classic_candidate_bytes,
            rate_control.classic_location_bytes,
        ],
        "HT PCRD locations",
    )?;
    let (mut layers, _) = tracker.try_vec::<usize>(
        segment_count,
        [
            context.layered_owners,
            context.complete_ht_output_bytes,
            rate_control.ht_candidate_bytes,
            rate_control.ht_location_bytes,
            rate_control.classic_candidate_bytes,
            rate_control.classic_location_bytes,
        ],
        "HT segment-layer metadata",
    )?;
    for segment_idx in 0..segment_count {
        rate_control
            .ht_candidates
            .push(HtSegmentAssignmentCandidate {
                block_index: rate_control.ht_block_index,
                segment_index: segment_idx,
                rate: ht_segment_rate(encoded, segment_idx)?,
            });
        rate_control.ht_locations.push(HtSegmentLocation {
            packet_idx: context.packet_idx,
            subband_idx: context.subband_idx,
            block_idx: context.layered_block_idx,
            segment_idx,
        });
        layers.push(usize::from(context.num_layers).saturating_sub(1));
    }
    Ok(layers)
}
