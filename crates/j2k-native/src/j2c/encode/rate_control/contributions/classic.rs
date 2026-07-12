// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic Tier-1 segment metadata and per-layer payload construction.

use super::super::super::allocation::{checked_add_bytes, checked_element_bytes};
use super::super::super::tier1_allocation::{segmented_block_ownership, Tier1PhaseTracker};
use super::super::super::{
    bitplane_encode, CodeBlockPacketData, NativeEncodePipelineError, NativeEncodePipelineResult,
    Vec,
};
use super::{layer_pass_count, previous_layer_pass_count};

mod build;
use build::{build_classic_layer_contribution, ClassicContributionOwners};

pub(in crate::j2c::encode) fn classic_unbudgeted_segment_layers_accounted(
    encoded: &bitplane_encode::EncodedCodeBlockWithSegments,
    num_layers: u8,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    retained_live_bytes: usize,
) -> NativeEncodePipelineResult<Vec<usize>> {
    let encoded_bytes = segmented_block_ownership(encoded)?;
    let (mut segment_layers, _) = tracker.try_vec::<usize>(
        encoded.segments.len(),
        [retained_live_bytes, encoded_bytes],
        "classic segment-layer metadata",
    )?;
    for segment in &encoded.segments {
        let mut assigned = None;
        for layer_idx in 0..usize::from(num_layers) {
            let previous_pass =
                previous_layer_pass_count(encoded.num_coding_passes, layer_idx, num_layers)?;
            let cumulative_passes = if layer_idx + 1 == usize::from(num_layers) {
                encoded.num_coding_passes
            } else {
                layer_pass_count(encoded.num_coding_passes, layer_idx + 1, num_layers)?
            };
            if segment.start_coding_pass >= previous_pass
                && segment.end_coding_pass <= cumulative_passes
            {
                assigned = Some(layer_idx);
                break;
            }
        }
        segment_layers.push(assigned.ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "classic quality layer split must align to terminated coding passes",
            )
        })?);
    }
    Ok(segment_layers)
}

pub(in crate::j2c::encode) fn classic_layer_contributions_accounted(
    encoded: &bitplane_encode::EncodedCodeBlockWithSegments,
    num_layers: u8,
    segment_layers: &[usize],
    segment_layer_capacity: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    retained_live_bytes: usize,
) -> NativeEncodePipelineResult<Vec<CodeBlockPacketData>> {
    let layer_count = usize::from(num_layers);
    if segment_layers.len() != encoded.segments.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "classic PCRD segment assignment count mismatch",
        ));
    }
    if segment_layer_capacity < segment_layers.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "classic segment-layer capacity is smaller than its length",
        ));
    }
    if segment_layers.iter().any(|&layer| layer >= layer_count) {
        return Err(NativeEncodePipelineError::internal_invariant(
            "classic PCRD segment layer exceeds layer count",
        ));
    }
    let encoded_bytes = segmented_block_ownership(encoded)?;
    let layer_metadata_bytes =
        checked_element_bytes::<usize>(segment_layer_capacity, "classic segment-layer metadata")?;
    let (mut contributions, contribution_owner_bytes) = tracker.try_vec::<CodeBlockPacketData>(
        layer_count,
        [retained_live_bytes, encoded_bytes, layer_metadata_bytes],
        "classic layer contribution owners",
    )?;
    let owners = ClassicContributionOwners::new(
        retained_live_bytes,
        encoded_bytes,
        layer_metadata_bytes,
        contribution_owner_bytes,
    );
    let mut contribution_nested_bytes = 0usize;
    for layer_idx in 0..layer_count {
        let (contribution, nested_bytes) = build_classic_layer_contribution(
            encoded,
            segment_layers,
            layer_idx,
            owners,
            contribution_nested_bytes,
            tracker,
        )?;
        contribution_nested_bytes = checked_add_bytes(
            contribution_nested_bytes,
            nested_bytes,
            "classic contribution owners",
        )?;
        contributions.push(contribution);
    }
    Ok(contributions)
}
