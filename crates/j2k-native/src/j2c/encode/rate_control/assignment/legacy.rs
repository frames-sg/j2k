// SPDX-License-Identifier: MIT OR Apache-2.0

//! Test-only compatibility solvers used by byte-parity regressions.

use alloc::vec;

use super::super::super::Vec;
use super::super::{
    ClassicLayerBudgetAllocator, ClassicSegmentAssignmentCandidate, HtSegmentAssignmentCandidate,
};
use super::{compare_classic_segment_candidates, enforce_classic_assignment_monotonicity};

pub(in crate::j2c::encode) fn assign_classic_segment_layers_by_slope(
    candidates: &[ClassicSegmentAssignmentCandidate],
    layer_count: usize,
    cumulative_targets: &[u64],
) -> Result<Vec<usize>, &'static str> {
    let mut allocator = ClassicLayerBudgetAllocator::new(cumulative_targets, layer_count)?;
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let block_count = candidates
        .iter()
        .map(|candidate| candidate.block_index)
        .max()
        .and_then(|max| max.checked_add(1))
        .ok_or("classic PCRD block count overflow")?;
    let mut block_candidates = vec![Vec::new(); block_count];
    for (candidate_idx, candidate) in candidates.iter().enumerate() {
        block_candidates
            .get_mut(candidate.block_index)
            .ok_or("classic PCRD block index mismatch")?
            .push(candidate_idx);
    }
    for block in &mut block_candidates {
        block.sort_by_key(|&idx| candidates[idx].segment_index);
    }

    let mut block_min_layers = vec![0usize; block_count];
    let mut assignments = vec![layer_count.saturating_sub(1); candidates.len()];
    let mut next_block_segment = vec![0usize; block_count];
    let mut remaining = candidates.len();
    while remaining > 0 {
        let candidate_idx = block_candidates
            .iter()
            .enumerate()
            .filter_map(|(block_idx, block)| block.get(next_block_segment[block_idx]).copied())
            .min_by(|&left, &right| compare_classic_segment_candidates(candidates, left, right))
            .ok_or("classic PCRD candidate queue underflow")?;
        let candidate = candidates[candidate_idx];
        let min_layer = *block_min_layers
            .get(candidate.block_index)
            .ok_or("classic PCRD block index mismatch")?;
        let layer = allocator.assign_segment(min_layer, candidate.rate)?;
        assignments[candidate_idx] = layer;
        if let Some(block_layer) = block_min_layers.get_mut(candidate.block_index) {
            *block_layer = layer;
        }
        if let Some(next) = next_block_segment.get_mut(candidate.block_index) {
            *next = next
                .checked_add(1)
                .ok_or("classic PCRD segment index overflow")?;
        }
        remaining -= 1;
    }

    enforce_classic_assignment_monotonicity(candidates, &mut assignments);
    Ok(assignments)
}

pub(in crate::j2c::encode) fn assign_ht_segment_layers_by_budget(
    candidates: &[HtSegmentAssignmentCandidate],
    layer_count: usize,
    cumulative_targets: &[u64],
) -> Result<Vec<usize>, &'static str> {
    let mut allocator = ClassicLayerBudgetAllocator::new(cumulative_targets, layer_count)?;
    let mut assignments = vec![layer_count.saturating_sub(1); candidates.len()];
    let mut candidate_order = Vec::new();
    candidate_order
        .try_reserve_exact(candidates.len())
        .map_err(|_| "HTJ2K candidate-order allocation failed")?;
    candidate_order.extend(0..candidates.len());
    candidate_order
        .sort_by_key(|&idx| (candidates[idx].block_index, candidates[idx].segment_index));
    let mut block_min_layers = vec![
        0usize;
        candidates
            .iter()
            .map(|candidate| candidate.block_index)
            .max()
            .map_or(0, |idx| idx + 1)
    ];

    for candidate_idx in candidate_order {
        let candidate = candidates
            .get(candidate_idx)
            .ok_or("HTJ2K segment candidate index mismatch")?;
        let min_layer = *block_min_layers
            .get(candidate.block_index)
            .ok_or("HTJ2K segment candidate block index mismatch")?;
        let layer = allocator.assign_segment(min_layer, candidate.rate)?;
        assignments[candidate_idx] = layer;
        if let Some(block_layer) = block_min_layers.get_mut(candidate.block_index) {
            *block_layer = layer;
        }
    }

    Ok(assignments)
}
