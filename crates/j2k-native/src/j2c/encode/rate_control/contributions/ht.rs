// SPDX-License-Identifier: MIT OR Apache-2.0

//! HTJ2K segment metadata and fallible per-layer payload construction.

use super::super::super::allocation::{checked_add_bytes, checked_element_bytes};
use super::super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::super::{
    bitplane_encode, BlockCodingMode, CodeBlockPacketData, NativeEncodePipelineError,
    NativeEncodePipelineResult, Vec,
};
use super::{ht_segment_count, ht_target_layer, layer_pass_count};

mod layout;
use layout::{ht_contribution_layout, HtContributionLayout};

#[cfg(test)]
pub(in crate::j2c::encode) fn ht_layer_contributions(
    encoded: &bitplane_encode::EncodedCodeBlock,
    num_layers: u8,
    segment_layers: &[usize],
) -> NativeEncodePipelineResult<Vec<CodeBlockPacketData>> {
    let layout = ht_contribution_layout(encoded, num_layers, segment_layers)?;
    let HtContributionLayout {
        layer_count,
        cleanup_len,
        refinement_len,
        refinement_end,
    } = layout;

    let mut contributions = Vec::new();
    contributions.try_reserve_exact(layer_count).map_err(|_| {
        crate::EncodeError::HostAllocationFailed {
            what: "HTJ2K layer contribution owners",
            bytes: layer_count.saturating_mul(core::mem::size_of::<CodeBlockPacketData>()),
        }
    })?;
    for layer_idx in 0..layer_count {
        let has_cleanup = segment_layers.first() == Some(&layer_idx);
        let has_refinement =
            encoded.num_coding_passes > 1 && segment_layers.get(1) == Some(&layer_idx);
        let payload_len = usize::from(has_cleanup)
            .checked_mul(cleanup_len)
            .and_then(|bytes| bytes.checked_add(usize::from(has_refinement) * refinement_len))
            .ok_or_else(|| {
                NativeEncodePipelineError::arithmetic_overflow(
                    "HTJ2K layer contribution payload overflow",
                )
            })?;
        let mut data = Vec::new();
        data.try_reserve_exact(payload_len).map_err(|_| {
            crate::EncodeError::HostAllocationFailed {
                what: "HTJ2K layer contribution payload",
                bytes: payload_len,
            }
        })?;
        let mut num_coding_passes = 0u8;
        if has_cleanup {
            data.extend_from_slice(encoded.data.get(..cleanup_len).ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant("HTJ2K cleanup segment range invalid")
            })?);
            num_coding_passes = 1;
        }
        if has_refinement {
            data.extend_from_slice(encoded.data.get(cleanup_len..refinement_end).ok_or_else(
                || {
                    NativeEncodePipelineError::internal_invariant(
                        "HTJ2K refinement segment range invalid",
                    )
                },
            )?);
            num_coding_passes = num_coding_passes
                .checked_add(encoded.num_coding_passes - 1)
                .ok_or_else(|| {
                    NativeEncodePipelineError::arithmetic_overflow(
                        "HTJ2K packet contribution pass count overflow",
                    )
                })?;
        }
        contributions.push(CodeBlockPacketData {
            data,
            ht_cleanup_length: if has_cleanup {
                encoded.ht_cleanup_length
            } else {
                0
            },
            ht_refinement_length: if has_refinement {
                encoded.ht_refinement_length
            } else {
                0
            },
            num_coding_passes,
            classic_segment_lengths: Vec::new(),
            num_zero_bitplanes: encoded.num_zero_bitplanes,
            previously_included: false,
            l_block: 3,
            block_coding_mode: BlockCodingMode::HighThroughput,
        });
    }
    Ok(contributions)
}

pub(in crate::j2c::encode) fn ht_unbudgeted_segment_layers_accounted(
    encoded: &bitplane_encode::EncodedCodeBlock,
    num_layers: u8,
    block_idx: usize,
    block_count: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    retained_live_bytes: usize,
) -> NativeEncodePipelineResult<Vec<usize>> {
    let segment_count = ht_segment_count(encoded);
    let encoded_bytes = encoded.data.capacity();
    let (mut segment_layers, _) = tracker.try_vec::<usize>(
        segment_count,
        [retained_live_bytes, encoded_bytes],
        "HT segment-layer metadata",
    )?;
    if segment_count == 0 {
        return Ok(segment_layers);
    }
    let layer_count = usize::from(num_layers);
    if layer_count == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "HTJ2K layer allocation requires at least one quality layer",
        ));
    }
    if encoded.num_coding_passes == 1 {
        segment_layers.push(ht_target_layer(block_idx, block_count, layer_count)?);
        return Ok(segment_layers);
    }
    let mut min_layer = 0usize;
    for end_pass in [1, encoded.num_coding_passes] {
        let mut assigned = None;
        for layer_idx in min_layer..layer_count {
            let cumulative_passes = if layer_idx + 1 == layer_count {
                encoded.num_coding_passes
            } else {
                layer_pass_count(encoded.num_coding_passes, layer_idx + 1, num_layers)?
            };
            if end_pass <= cumulative_passes {
                assigned = Some(layer_idx);
                break;
            }
        }
        let assigned = assigned.ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "HTJ2K quality layer split must align to segment boundaries",
            )
        })?;
        segment_layers.push(assigned);
        min_layer = assigned;
    }
    Ok(segment_layers)
}

pub(in crate::j2c::encode) fn ht_layer_contributions_accounted(
    encoded: &bitplane_encode::EncodedCodeBlock,
    num_layers: u8,
    segment_layers: &[usize],
    segment_layer_capacity: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    retained_live_bytes: usize,
) -> NativeEncodePipelineResult<Vec<CodeBlockPacketData>> {
    if segment_layer_capacity < segment_layers.len() {
        return Err(crate::EncodeError::InternalInvariant {
            what: "HT segment-layer capacity is smaller than its length",
        }
        .into());
    }
    let layout = ht_contribution_layout(encoded, num_layers, segment_layers)?;
    let HtContributionLayout {
        layer_count,
        cleanup_len,
        refinement_len,
        refinement_end,
    } = layout;

    let encoded_bytes = encoded.data.capacity();
    let layer_metadata_bytes =
        checked_element_bytes::<usize>(segment_layer_capacity, "HT segment-layer metadata")?;
    let (mut contributions, contribution_owner_bytes) = tracker.try_vec::<CodeBlockPacketData>(
        layer_count,
        [retained_live_bytes, encoded_bytes, layer_metadata_bytes],
        "HT layer contribution owners",
    )?;
    let mut contribution_payload_bytes = 0usize;
    for layer_idx in 0..layer_count {
        let has_cleanup = segment_layers.first() == Some(&layer_idx);
        let has_refinement =
            encoded.num_coding_passes > 1 && segment_layers.get(1) == Some(&layer_idx);
        let payload_len = usize::from(has_cleanup)
            .checked_mul(cleanup_len)
            .and_then(|bytes| bytes.checked_add(usize::from(has_refinement) * refinement_len))
            .ok_or(crate::EncodeError::ArithmeticOverflow {
                what: "HT layer contribution payload",
            })?;
        let (mut data, data_bytes) = tracker.try_vec::<u8>(
            payload_len,
            [
                retained_live_bytes,
                encoded_bytes,
                layer_metadata_bytes,
                contribution_owner_bytes,
                contribution_payload_bytes,
            ],
            "HT layer contribution payload",
        )?;
        let mut num_coding_passes = 0u8;
        if has_cleanup {
            data.extend_from_slice(encoded.data.get(..cleanup_len).ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant("HTJ2K cleanup segment range invalid")
            })?);
            num_coding_passes = 1;
        }
        if has_refinement {
            data.extend_from_slice(encoded.data.get(cleanup_len..refinement_end).ok_or_else(
                || {
                    NativeEncodePipelineError::internal_invariant(
                        "HTJ2K refinement segment range invalid",
                    )
                },
            )?);
            num_coding_passes = num_coding_passes
                .checked_add(encoded.num_coding_passes - 1)
                .ok_or_else(|| {
                    NativeEncodePipelineError::arithmetic_overflow(
                        "HTJ2K packet contribution pass count overflow",
                    )
                })?;
        }
        contribution_payload_bytes = checked_add_bytes(
            contribution_payload_bytes,
            data_bytes,
            "HT layer contribution payload graph",
        )?;
        contributions.push(CodeBlockPacketData {
            data,
            ht_cleanup_length: if has_cleanup {
                encoded.ht_cleanup_length
            } else {
                0
            },
            ht_refinement_length: if has_refinement {
                encoded.ht_refinement_length
            } else {
                0
            },
            num_coding_passes,
            classic_segment_lengths: Vec::new(),
            num_zero_bitplanes: encoded.num_zero_bitplanes,
            previously_included: false,
            l_block: 3,
            block_coding_mode: BlockCodingMode::HighThroughput,
        });
    }
    Ok(contributions)
}
