// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move-only HT subband handoff into layered block construction.

use super::super::super::super::allocation::checked_add_bytes;
use super::super::super::super::tier1_allocation::{
    prepared_subbands_ownership, subband_precincts_ownership, Tier1PhaseTracker,
};
use super::super::super::super::{
    encode_prepared_subbands_for_session, CodeBlockPacketData, J2kEncodeStageAccelerator,
    LayeredPreparedSubband, NativeEncodePipelineError, NativeEncodePipelineResult,
    PreparedEncodeSubband, Vec,
};
use super::super::ownership::{checked_sum, layered_block_build_owner_bytes};
use super::LayeredHtContext;

pub(super) struct LayeredHtOutput {
    pub(super) blocks: Vec<CodeBlockPacketData>,
    pub(super) structural_bytes: usize,
    pub(super) remaining_payload_bytes: usize,
    pub(super) other_source_bytes: usize,
}

pub(super) fn try_encode_layered_ht_output(
    subband: PreparedEncodeSubband,
    layered_subband: &LayeredPreparedSubband,
    rate_control_owner_bytes: usize,
    context: LayeredHtContext<'_, '_>,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<LayeredHtOutput> {
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
