// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move-only HT subband handoff into layered block construction.

use super::super::super::super::allocation::checked_add_bytes;
use super::super::super::super::tier1_allocation::{
    prepared_subbands_ownership, subband_precincts_ownership, Tier1PhaseTracker,
};
use super::super::super::super::{
    encode_prepared_subbands_for_session, ht_block_encode, BlockCodingMode, CodeBlockPacketData,
    J2kEncodeStageAccelerator, LayeredPreparedSubband, NativeEncodePipelineError,
    NativeEncodePipelineResult, PreparedCodeBlockCoefficients, PreparedEncodeSubband, Vec,
};
use super::super::ownership::{checked_sum, layered_block_build_owner_bytes};
use super::super::state::LayeredRateControlState;
use super::LayeredHtContext;

mod tile;
use tile::eligible_subband;
pub(in crate::j2c::encode::prepared_packets::layered) use tile::try_select_tile_ht_candidates;

pub(super) struct LayeredHtOutput {
    pub(super) blocks: Vec<CodeBlockPacketData>,
    pub(super) structural_bytes: usize,
    pub(super) remaining_payload_bytes: usize,
    pub(super) other_source_bytes: usize,
}

#[derive(Clone, Copy)]
pub(super) struct CandidateOwnerBytes {
    layered_live: usize,
    subband: usize,
    structural: usize,
}

pub(super) fn try_encode_layered_ht_output(
    subband: PreparedEncodeSubband,
    layered_subband: &LayeredPreparedSubband,
    rate_control: &mut LayeredRateControlState,
    context: LayeredHtContext<'_, '_>,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<LayeredHtOutput> {
    let rate_control_owner_bytes = rate_control.owner_bytes()?;
    let subband_bytes = prepared_subbands_ownership(core::slice::from_ref(&subband), 0)?.total()?;
    let other_source_bytes = context.source_bytes.checked_sub(subband_bytes).ok_or(
        crate::EncodeError::InternalInvariant {
            what: "layered HT source ownership underflowed",
        },
    )?;
    let layered_owners = layered_block_build_owner_bytes(
        other_source_bytes,
        context.layered_packets,
        context.layered_packet_capacity,
        context.layered_packet,
        layered_subband,
    )?;
    let layered_live = checked_sum(
        [layered_owners, rate_control_owner_bytes],
        "layered HT subband owners",
    )?;
    if !context.quality_layer_byte_targets.is_empty() && eligible_subband(&subband) {
        return try_encode_bounded_ht_output(
            subband,
            subband_bytes,
            other_source_bytes,
            layered_live,
            rate_control,
            tracker,
        );
    }
    let (mut one_subband, _) = tracker.try_vec::<PreparedEncodeSubband>(
        1,
        [layered_live],
        "layered HT subband handoff owner",
    )?;
    one_subband.push(subband);
    let tier1_base = checked_add_bytes(
        context.retained_base_bytes,
        layered_live,
        "layered HT Tier-1 retained owners",
    )?;
    let mut precincts = encode_prepared_subbands_for_session(
        one_subband,
        context.session,
        tier1_base,
        accelerator,
    )?;
    if precincts.len() != 1 {
        return Err(NativeEncodePipelineError::internal_invariant(
            "layered HT subband output count mismatch",
        ));
    }
    let output_bytes = subband_precincts_ownership(&precincts, precincts.capacity())?;
    let payload_bytes = precincts.iter().try_fold(0usize, |total, precinct| {
        precinct.code_blocks.iter().try_fold(total, |total, block| {
            checked_add_bytes(total, block.data.capacity(), "layered HT output payload")
        })
    })?;
    let structural_bytes =
        output_bytes
            .checked_sub(payload_bytes)
            .ok_or(crate::EncodeError::InternalInvariant {
                what: "layered HT output ownership underflowed",
            })?;
    let precinct = precincts.pop().ok_or_else(|| {
        NativeEncodePipelineError::internal_invariant("layered HT subband output is missing")
    })?;
    Ok(LayeredHtOutput {
        blocks: precinct.code_blocks,
        structural_bytes,
        remaining_payload_bytes: payload_bytes,
        other_source_bytes,
    })
}

fn try_encode_bounded_ht_output(
    subband: PreparedEncodeSubband,
    subband_bytes: usize,
    other_source_bytes: usize,
    layered_live: usize,
    rate_control: &mut LayeredRateControlState,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<LayeredHtOutput> {
    let block_count = subband.code_blocks.len();
    let (mut blocks, structural_bytes) = tracker.try_vec::<CodeBlockPacketData>(
        block_count,
        [layered_live, subband_bytes],
        "bounded HT candidate output owners",
    )?;
    let mut remaining_payload_bytes = 0usize;
    for _block in subband.code_blocks {
        let selected = rate_control.take_selected_ht_candidate()?;
        let payload_bytes = selected.data.capacity();
        let packet_block = packet_block_from_ht_candidate(selected);
        remaining_payload_bytes = checked_add_bytes(
            remaining_payload_bytes,
            payload_bytes,
            "bounded HT selected payload",
        )?;
        blocks.push(packet_block);
    }
    Ok(LayeredHtOutput {
        blocks,
        structural_bytes,
        remaining_payload_bytes,
        other_source_bytes,
    })
}

pub(super) fn try_accelerated_candidate_outputs(
    subband: &PreparedEncodeSubband,
    cleanup_bitplanes: &[u8],
    owners: CandidateOwnerBytes,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Option<alloc::vec::IntoIter<crate::EncodedHtJ2kCodeBlockSet>>> {
    let job_count = subband
        .code_blocks
        .len()
        .checked_mul(cleanup_bitplanes.len())
        .ok_or(crate::EncodeError::ArithmeticOverflow {
            what: "bounded HT candidate job count",
        })?;
    let (mut jobs, job_bytes) = tracker.try_vec::<crate::J2kHtCodeBlockSetEncodeJob<'_>>(
        job_count,
        [owners.layered_live, owners.subband, owners.structural],
        "bounded HT candidate job descriptors",
    )?;
    for block in &subband.code_blocks {
        let PreparedCodeBlockCoefficients::I32(coefficients) = &block.coefficients else {
            return Err(NativeEncodePipelineError::internal_invariant(
                "bounded HT candidates require i32 coefficients",
            ));
        };
        for &cleanup_bitplane in cleanup_bitplanes {
            jobs.push(crate::J2kHtCodeBlockSetEncodeJob {
                coefficients,
                width: block.width,
                height: block.height,
                total_bitplanes: subband.total_bitplanes,
                cleanup_bitplane,
                target_coding_passes: if cleanup_bitplane == 0 { 1 } else { 3 },
            });
        }
    }
    tracker.check(
        [
            owners.layered_live,
            owners.subband,
            owners.structural,
            job_bytes,
        ],
        "bounded HT accelerator candidate jobs",
    )?;
    let outputs = accelerator
        .encode_ht_code_block_sets(&jobs)
        .map_err(|source| crate::EncodeError::Accelerator {
            operation: "HT Tier-1 candidate-set batch encode",
            source,
        })?;
    if outputs
        .as_ref()
        .is_some_and(|values| values.len() != job_count)
    {
        return Err(candidate_accelerator_error(
            "accelerated HT candidate-set batch length mismatch",
        ));
    }
    Ok(outputs.map(Vec::into_iter))
}

fn packet_block_from_ht_candidate(
    selected: super::super::super::super::bitplane_encode::EncodedCodeBlock,
) -> CodeBlockPacketData {
    CodeBlockPacketData {
        data: selected.data,
        ht_cleanup_length: selected.ht_cleanup_length,
        ht_refinement_length: selected.ht_refinement_length,
        ht_sigprop_length: selected.ht_sigprop_length,
        ht_magref_length: selected.ht_magref_length,
        ht_distortion_deltas: selected.ht_distortion_deltas,
        num_coding_passes: selected.num_coding_passes,
        classic_segment_lengths: Vec::new(),
        num_zero_bitplanes: selected.num_zero_bitplanes,
        previously_included: false,
        l_block: 3,
        block_coding_mode: BlockCodingMode::HighThroughput,
    }
}

pub(super) fn accelerated_candidates_for_block(
    outputs: &mut alloc::vec::IntoIter<crate::EncodedHtJ2kCodeBlockSet>,
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    cleanup_bitplanes: &[u8],
) -> NativeEncodePipelineResult<Vec<super::super::super::super::bitplane_encode::EncodedCodeBlock>>
{
    let mut candidates = Vec::new();
    candidates
        .try_reserve_exact(cleanup_bitplanes.len())
        .map_err(|_| crate::EncodeError::HostAllocationFailed {
            what: "accelerated HT candidate owners",
            bytes: cleanup_bitplanes.len().saturating_mul(core::mem::size_of::<
                super::super::super::super::bitplane_encode::EncodedCodeBlock,
            >()),
        })?;
    for &cleanup_bitplane in cleanup_bitplanes {
        let output = outputs.next().ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("accelerated HT candidate is missing")
        })?;
        candidates.push(validated_accelerated_candidate(
            output,
            coefficients,
            width,
            height,
            total_bitplanes,
            cleanup_bitplane,
        )?);
    }
    Ok(candidates)
}

fn validated_accelerated_candidate(
    output: crate::EncodedHtJ2kCodeBlockSet,
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    cleanup_bitplane: u8,
) -> NativeEncodePipelineResult<super::super::super::super::bitplane_encode::EncodedCodeBlock> {
    let expected_passes = if cleanup_bitplane == 0 { 1 } else { 3 };
    let refinement_length = output
        .sigprop_length
        .checked_add(output.magref_length)
        .ok_or_else(|| candidate_accelerator_error("HT candidate refinement length overflow"))?;
    let expected_length = output
        .cleanup_length
        .checked_add(refinement_length)
        .and_then(|length| usize::try_from(length).ok())
        .ok_or_else(|| candidate_accelerator_error("HT candidate payload length overflow"))?;
    let input_is_zero = coefficients.iter().all(|coefficient| *coefficient == 0);
    let valid_empty = output.num_coding_passes == 0
        && input_is_zero
        && output.data.is_empty()
        && output.cleanup_length == 0
        && refinement_length == 0
        && output.num_zero_bitplanes == total_bitplanes;
    let expected_missing = total_bitplanes - cleanup_bitplane - 1;
    let valid_nonempty = output.num_coding_passes == expected_passes
        && !input_is_zero
        && output.data.len() == expected_length
        && output.cleanup_length > 0
        && output.num_zero_bitplanes == expected_missing
        && (expected_passes == 1 || refinement_length > 0);
    if !valid_empty && !valid_nonempty {
        return Err(candidate_accelerator_error(
            "accelerated HT candidate metadata mismatch",
        ));
    }
    let distortion = if valid_empty {
        [0.0; 3]
    } else {
        ht_block_encode::code_block_set_distortion_deltas(
            coefficients,
            width,
            height,
            cleanup_bitplane,
            expected_passes,
        )?
    };
    Ok(
        super::super::super::super::bitplane_encode::EncodedCodeBlock {
            data: output.data,
            num_coding_passes: output.num_coding_passes,
            num_zero_bitplanes: output.num_zero_bitplanes,
            ht_cleanup_length: output.cleanup_length,
            ht_refinement_length: refinement_length,
            ht_sigprop_length: output.sigprop_length,
            ht_magref_length: output.magref_length,
            ht_distortion_deltas: distortion,
        },
    )
}

fn candidate_accelerator_error(detail: &'static str) -> NativeEncodePipelineError {
    crate::EncodeError::Accelerator {
        operation: "HT Tier-1 candidate-set batch encode",
        source: crate::J2kEncodeStageError::internal_invariant(detail),
    }
    .into()
}
