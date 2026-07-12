// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible construction of one classic quality-layer contribution.

use super::super::super::super::allocation::checked_add_bytes;
use super::super::super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::super::super::{
    bitplane_encode, BlockCodingMode, CodeBlockPacketData, NativeEncodePipelineError,
    NativeEncodePipelineResult,
};

#[derive(Clone, Copy)]
pub(super) struct ClassicContributionOwners {
    retained_live: usize,
    encoded: usize,
    layer_metadata: usize,
    contribution_owners: usize,
}

impl ClassicContributionOwners {
    pub(super) const fn new(
        retained_live: usize,
        encoded: usize,
        layer_metadata: usize,
        contribution_owners: usize,
    ) -> Self {
        Self {
            retained_live,
            encoded,
            layer_metadata,
            contribution_owners,
        }
    }
}

struct ClassicContributionPlan {
    payload_len: usize,
    segment_count: usize,
    coding_passes: u8,
}

pub(super) fn build_classic_layer_contribution(
    encoded: &bitplane_encode::EncodedCodeBlockWithSegments,
    segment_layers: &[usize],
    layer_idx: usize,
    owners: ClassicContributionOwners,
    contribution_nested_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<(CodeBlockPacketData, usize)> {
    let plan = classic_contribution_plan(encoded, segment_layers, layer_idx)?;
    let live = [
        owners.retained_live,
        owners.encoded,
        owners.layer_metadata,
        owners.contribution_owners,
        contribution_nested_bytes,
    ];
    let (mut data, data_bytes) =
        tracker.try_vec::<u8>(plan.payload_len, live, "classic layer contribution payload")?;
    let (mut segment_lengths, segment_bytes) = tracker.try_vec::<u32>(
        plan.segment_count,
        live.into_iter().chain([data_bytes]),
        "classic layer segment lengths",
    )?;
    for (segment_idx, segment) in encoded.segments.iter().enumerate() {
        if segment_layers[segment_idx] != layer_idx {
            continue;
        }
        let start = usize::try_from(segment.data_offset).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow(
                "classic code-block segment offset overflow",
            )
        })?;
        let len = usize::try_from(segment.data_length).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow(
                "classic code-block segment length overflow",
            )
        })?;
        let end = start.checked_add(len).ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow(
                "classic code-block segment range overflow",
            )
        })?;
        data.extend_from_slice(encoded.data.get(start..end).ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "classic code-block segment range invalid",
            )
        })?);
        segment_lengths.push(segment.data_length);
    }
    let nested_bytes = checked_add_bytes(data_bytes, segment_bytes, "classic contribution owners")?;
    Ok((
        CodeBlockPacketData {
            data,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: plan.coding_passes,
            classic_segment_lengths: segment_lengths,
            num_zero_bitplanes: encoded.num_zero_bitplanes,
            previously_included: false,
            l_block: 3,
            block_coding_mode: BlockCodingMode::Classic,
        },
        nested_bytes,
    ))
}

fn classic_contribution_plan(
    encoded: &bitplane_encode::EncodedCodeBlockWithSegments,
    segment_layers: &[usize],
    layer_idx: usize,
) -> NativeEncodePipelineResult<ClassicContributionPlan> {
    let mut payload_len = 0usize;
    let mut segment_count = 0usize;
    let mut coding_passes = 0u8;
    for (segment_idx, segment) in encoded.segments.iter().enumerate() {
        if segment_layers[segment_idx] != layer_idx {
            continue;
        }
        let segment_len = usize::try_from(segment.data_length).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow(
                "classic code-block segment length overflow",
            )
        })?;
        payload_len = payload_len.checked_add(segment_len).ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("classic layer contribution payload")
        })?;
        segment_count = segment_count.checked_add(1).ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("classic layer segment count")
        })?;
        let segment_passes = segment
            .end_coding_pass
            .checked_sub(segment.start_coding_pass)
            .ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant(
                    "classic code-block coding-pass range invalid",
                )
            })?;
        coding_passes = coding_passes.checked_add(segment_passes).ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow(
                "classic code-block contribution pass count overflow",
            )
        })?;
    }
    Ok(ClassicContributionPlan {
        payload_len,
        segment_count,
        coding_passes,
    })
}
