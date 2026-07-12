// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic Tier-1 block encoding for layered packet construction.

use alloc::vec::Vec;

use super::super::super::tier1_allocation::{segmented_block_ownership, Tier1PhaseTracker};
use super::super::super::{
    bitplane_encode, classic_multilayer_code_block_style,
    classic_unbudgeted_segment_layers_accounted, ClassicSegmentAssignmentCandidate,
    ClassicSegmentLocation, LayeredPreparedBlock, LayeredPreparedPacket, LayeredPreparedSubband,
    NativeEncodePipelineError, NativeEncodePipelineResult, PreparedCodeBlockCoefficients,
    PreparedEncodeSubband,
};
use super::ownership::{checked_sum, layered_block_build_owner_bytes};
use super::state::LayeredRateControlState;

#[derive(Clone, Copy)]
pub(super) struct LayeredClassicContext<'a> {
    pub(super) packet_idx: usize,
    pub(super) subband_idx: usize,
    pub(super) num_layers: u8,
    pub(super) quality_layer_byte_targets: &'a [u64],
    pub(super) source_bytes: usize,
    pub(super) layered_packets: &'a [LayeredPreparedPacket],
    pub(super) layered_packet_capacity: usize,
    pub(super) layered_packet: &'a LayeredPreparedPacket,
}

#[derive(Clone, Copy)]
struct ClassicSegmentContext {
    packet_idx: usize,
    subband_idx: usize,
    block_idx: usize,
    layer_count: usize,
    budgeted: bool,
    layered_owners: usize,
    layered_live: usize,
}

pub(super) fn encode_layered_classic_subband(
    subband: PreparedEncodeSubband,
    layered_subband: &mut LayeredPreparedSubband,
    rate_control: &mut LayeredRateControlState,
    context: LayeredClassicContext<'_>,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    let style = classic_multilayer_code_block_style();
    for block in subband.code_blocks {
        let block_idx = layered_subband.blocks.len();
        let layered_owners = layered_block_build_owner_bytes(
            context.source_bytes,
            context.layered_packets,
            context.layered_packet_capacity,
            context.layered_packet,
            layered_subband,
        )?;
        let layered_live = checked_sum(
            [layered_owners, rate_control.owner_bytes()?],
            "layered classic block owners",
        )?;
        let worker = bitplane_encode::classic_worker_allocation(
            block.width as usize,
            block.height as usize,
            subband.total_bitplanes,
        )?;
        tracker.check(
            [layered_live, worker.output_bytes, worker.scratch_bytes],
            "layered classic Tier-1 worker frontier",
        )?;
        let encoded = match block.coefficients {
            PreparedCodeBlockCoefficients::I32(values) => {
                bitplane_encode::try_encode_code_block_segments_with_style(
                    &values,
                    block.width,
                    block.height,
                    subband.sub_band_type,
                    subband.total_bitplanes,
                    &style,
                )?
            }
            PreparedCodeBlockCoefficients::I64(values) => {
                bitplane_encode::try_encode_code_block_segments_with_style_i64(
                    &values,
                    block.width,
                    block.height,
                    subband.sub_band_type,
                    subband.total_bitplanes,
                    &style,
                )?
            }
            PreparedCodeBlockCoefficients::Empty => {
                return Err(NativeEncodePipelineError::internal_invariant(
                    "classic Tier-1 coefficient storage is missing",
                ))
            }
        };
        let segment_layers = classic_segment_layers(
            &encoded,
            rate_control,
            ClassicSegmentContext {
                packet_idx: context.packet_idx,
                subband_idx: context.subband_idx,
                block_idx,
                layer_count: usize::from(context.num_layers),
                budgeted: !context.quality_layer_byte_targets.is_empty(),
                layered_owners,
                layered_live,
            },
            tracker,
        )?;
        layered_subband.blocks.push(LayeredPreparedBlock::Classic {
            encoded,
            segment_layers,
        });
        rate_control.classic_block_index = rate_control.classic_block_index.checked_add(1).ok_or(
            crate::EncodeError::ArithmeticOverflow {
                what: "classic PCRD block index",
            },
        )?;
    }
    Ok(())
}

fn classic_segment_layers(
    encoded: &bitplane_encode::EncodedCodeBlockWithSegments,
    rate_control: &mut LayeredRateControlState,
    context: ClassicSegmentContext,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<Vec<usize>> {
    if !context.budgeted {
        return classic_unbudgeted_segment_layers_accounted(
            encoded,
            u8::try_from(context.layer_count).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("quality layer count exceeds u8")
            })?,
            tracker,
            context.layered_live,
        );
    }
    let segment_count = encoded.segments.len();
    let encoded_bytes = segmented_block_ownership(encoded)?;
    rate_control.classic_candidate_bytes = tracker.try_reserve_additional(
        &mut rate_control.classic_candidates,
        segment_count,
        [
            context.layered_owners,
            encoded_bytes,
            rate_control.classic_location_bytes,
            rate_control.ht_candidate_bytes,
            rate_control.ht_location_bytes,
        ],
        "classic PCRD candidates",
    )?;
    rate_control.classic_location_bytes = tracker.try_reserve_additional(
        &mut rate_control.classic_locations,
        segment_count,
        [
            context.layered_owners,
            encoded_bytes,
            rate_control.classic_candidate_bytes,
            rate_control.ht_candidate_bytes,
            rate_control.ht_location_bytes,
        ],
        "classic PCRD locations",
    )?;
    let (mut layers, _) = tracker.try_vec::<usize>(
        segment_count,
        [
            context.layered_owners,
            encoded_bytes,
            rate_control.classic_candidate_bytes,
            rate_control.classic_location_bytes,
            rate_control.ht_candidate_bytes,
            rate_control.ht_location_bytes,
        ],
        "classic segment-layer metadata",
    )?;
    for (segment_idx, segment) in encoded.segments.iter().enumerate() {
        rate_control
            .classic_candidates
            .push(ClassicSegmentAssignmentCandidate {
                block_index: rate_control.classic_block_index,
                segment_index: segment_idx,
                rate: u64::from(segment.data_length),
                distortion_delta: segment.distortion_delta,
            });
        rate_control.classic_locations.push(ClassicSegmentLocation {
            packet_idx: context.packet_idx,
            subband_idx: context.subband_idx,
            block_idx: context.block_idx,
            segment_idx,
        });
        layers.push(context.layer_count.saturating_sub(1));
    }
    Ok(layers)
}
