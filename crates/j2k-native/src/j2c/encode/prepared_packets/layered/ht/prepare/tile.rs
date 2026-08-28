// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whole-tile HT candidate generation and final-set selection.

use super::{
    accelerated_candidates_for_block, try_accelerated_candidate_outputs, CandidateOwnerBytes,
};
use crate::j2c::bitplane_encode::EncodedCodeBlock;
use crate::j2c::encode::allocation::{checked_add_bytes, checked_element_bytes};
use crate::j2c::encode::tier1_allocation::Tier1PhaseTracker;
use crate::j2c::encode::{
    ht_block_encode, BlockCodingMode, J2kEncodeStageAccelerator, NativeEncodePipelineError,
    NativeEncodePipelineResult, PreparedCodeBlockCoefficients, PreparedEncodeSubband,
    PreparedResolutionPacket, Vec,
};

struct TileCandidates {
    blocks: Vec<EncodedCodeBlock>,
    ranges: Vec<ht_block_encode::HtCandidateRange>,
    block_structural_bytes: usize,
    range_bytes: usize,
    payload_bytes: usize,
}

pub(in crate::j2c::encode::prepared_packets::layered) fn try_select_tile_ht_candidates(
    packets: &[PreparedResolutionPacket],
    final_target: u64,
    source_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<EncodedCodeBlock>> {
    let (block_count, candidate_count) = tile_candidate_counts(packets)?;
    if block_count == 0 {
        return Ok(Vec::new());
    }
    let inventory = try_generate_tile_candidates(
        packets,
        block_count,
        candidate_count,
        source_bytes,
        tracker,
        accelerator,
    )?;
    let selector_workspace =
        ht_block_encode::tile_candidate_selection_workspace_bytes(candidate_count, block_count)?;
    tracker.check(
        [
            source_bytes,
            inventory.block_structural_bytes,
            inventory.range_bytes,
            inventory.payload_bytes,
            selector_workspace,
            ht_block_encode::HtEncodeWorkspace::ALLOCATION_BYTES,
        ],
        "whole-tile HT candidate selection",
    )?;
    let selections = ht_block_encode::select_tile_code_block_candidates(
        &inventory.blocks,
        &inventory.ranges,
        final_target,
    )?;
    if selections.len() != block_count {
        return Err(NativeEncodePipelineError::internal_invariant(
            "whole-tile HT selection count mismatch",
        ));
    }
    try_materialize_tile_selection(inventory, selections, block_count, source_bytes, tracker)
}

fn try_generate_tile_candidates(
    packets: &[PreparedResolutionPacket],
    block_count: usize,
    candidate_count: usize,
    source_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<TileCandidates> {
    let (mut candidates, candidate_structural_bytes) = tracker.try_vec::<EncodedCodeBlock>(
        candidate_count,
        [source_bytes],
        "whole-tile HT candidate owners",
    )?;
    let (mut ranges, range_bytes) = tracker.try_vec::<ht_block_encode::HtCandidateRange>(
        block_count,
        [source_bytes, candidate_structural_bytes],
        "whole-tile HT candidate ranges",
    )?;
    let mut workspace = ht_block_encode::HtEncodeWorkspace::try_new()?;
    let mut candidate_payload_bytes = 0usize;

    for packet in packets {
        for subband in &packet.subbands {
            if !eligible_subband(subband) {
                continue;
            }
            let cleanup_bitplanes =
                ht_block_encode::candidate_cleanup_bitplanes(subband.total_bitplanes)?;
            let mut accelerated = try_accelerated_candidate_outputs(
                subband,
                cleanup_bitplanes,
                CandidateOwnerBytes {
                    layered_live: source_bytes,
                    subband: candidate_structural_bytes,
                    structural: range_bytes,
                },
                tracker,
                accelerator,
            )?;
            for block in &subband.code_blocks {
                let PreparedCodeBlockCoefficients::I32(coefficients) = &block.coefficients else {
                    return Err(NativeEncodePipelineError::internal_invariant(
                        "bounded HT candidates require i32 coefficients",
                    ));
                };
                let block_candidates = if let Some(outputs) = accelerated.as_mut() {
                    accelerated_candidates_for_block(
                        outputs,
                        coefficients,
                        block.width,
                        block.height,
                        subband.total_bitplanes,
                        cleanup_bitplanes,
                    )?
                } else {
                    ht_block_encode::try_encode_code_block_candidate_sets_with_workspace(
                        coefficients,
                        block.width,
                        block.height,
                        subband.total_bitplanes,
                        &mut workspace,
                    )?
                };
                let start = candidates.len();
                for candidate in block_candidates {
                    candidate_payload_bytes = checked_add_bytes(
                        candidate_payload_bytes,
                        candidate.data.capacity(),
                        "whole-tile HT candidate payload",
                    )?;
                    candidates.push(candidate);
                }
                ranges.push(ht_block_encode::HtCandidateRange {
                    start,
                    len: candidates.len() - start,
                });
            }
            if accelerated
                .as_mut()
                .is_some_and(|outputs| outputs.next().is_some())
            {
                return Err(NativeEncodePipelineError::internal_invariant(
                    "accelerated HT candidate batch has trailing outputs",
                ));
            }
        }
    }
    if candidates.len() != candidate_count || ranges.len() != block_count {
        return Err(NativeEncodePipelineError::internal_invariant(
            "whole-tile HT candidate count mismatch",
        ));
    }
    Ok(TileCandidates {
        blocks: candidates,
        ranges,
        block_structural_bytes: candidate_structural_bytes,
        range_bytes,
        payload_bytes: candidate_payload_bytes,
    })
}

fn try_materialize_tile_selection(
    inventory: TileCandidates,
    selections: Vec<ht_block_encode::HtCandidateSelection>,
    block_count: usize,
    source_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<Vec<EncodedCodeBlock>> {
    let selection_bytes = checked_element_bytes::<ht_block_encode::HtCandidateSelection>(
        selections.capacity(),
        "whole-tile HT selections",
    )?;
    let (mut selected, selected_structural_bytes) = tracker.try_vec::<EncodedCodeBlock>(
        block_count,
        [
            source_bytes,
            inventory.block_structural_bytes,
            inventory.range_bytes,
            inventory.payload_bytes,
            selection_bytes,
        ],
        "whole-tile selected HT blocks",
    )?;
    let mut selections = selections.into_iter().peekable();
    for (candidate_index, mut candidate) in inventory.blocks.into_iter().enumerate() {
        let Some(selection) = selections.peek().copied() else {
            break;
        };
        if selection.candidate_index != candidate_index {
            continue;
        }
        selections.next();
        if selection.num_coding_passes != 0 {
            candidate = ht_block_encode::truncate_code_block_candidate(
                candidate,
                selection.num_coding_passes,
            )?;
        }
        selected.push(candidate);
    }
    if selections.next().is_some() || selected.len() != block_count {
        return Err(NativeEncodePipelineError::internal_invariant(
            "whole-tile HT selected candidate is missing",
        ));
    }
    tracker.check(
        [source_bytes, selected_structural_bytes]
            .into_iter()
            .chain(selected.iter().map(|candidate| candidate.data.capacity())),
        "whole-tile selected HT candidate owners",
    )?;
    selected.reverse();
    Ok(selected)
}

pub(super) fn eligible_subband(subband: &PreparedEncodeSubband) -> bool {
    subband.block_coding_mode == BlockCodingMode::HighThroughput
        && subband.preencoded_ht_code_blocks.is_none()
        && subband
            .code_blocks
            .iter()
            .all(|block| matches!(block.coefficients, PreparedCodeBlockCoefficients::I32(_)))
}

fn tile_candidate_counts(
    packets: &[PreparedResolutionPacket],
) -> NativeEncodePipelineResult<(usize, usize)> {
    let mut block_count = 0usize;
    let mut candidate_count = 0usize;
    for packet in packets {
        for subband in &packet.subbands {
            if !eligible_subband(subband) {
                continue;
            }
            let candidates_per_block =
                ht_block_encode::candidate_cleanup_bitplanes(subband.total_bitplanes)?.len();
            block_count = block_count.checked_add(subband.code_blocks.len()).ok_or(
                crate::EncodeError::ArithmeticOverflow {
                    what: "whole-tile HT candidate block count",
                },
            )?;
            candidate_count = candidate_count
                .checked_add(
                    subband
                        .code_blocks
                        .len()
                        .checked_mul(candidates_per_block)
                        .ok_or(crate::EncodeError::ArithmeticOverflow {
                            what: "whole-tile HT candidate count",
                        })?,
                )
                .ok_or(crate::EncodeError::ArithmeticOverflow {
                    what: "whole-tile HT candidate count",
                })?;
        }
    }
    Ok((block_count, candidate_count))
}
