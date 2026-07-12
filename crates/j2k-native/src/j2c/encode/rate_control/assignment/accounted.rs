// SPDX-License-Identifier: MIT OR Apache-2.0

//! Session-accounted candidate workspaces and segment assignments.

use super::super::super::allocation::{checked_add_bytes, checked_element_bytes};
use super::super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::super::{NativeEncodePipelineError, NativeEncodePipelineResult, Vec};
use super::super::{
    classic_rate_target_tolerance, ClassicLayerBudgetAllocator, HtSegmentAssignmentCandidate,
};

mod classic;
pub(in crate::j2c::encode) use classic::assign_classic_segment_layers_by_slope_accounted;

struct HtAssignmentWorkspace {
    allocator: ClassicLayerBudgetAllocator,
    assignments: Vec<usize>,
    candidate_order: Vec<usize>,
    block_min_layers: Vec<usize>,
}

pub(in crate::j2c::encode) fn assign_ht_segment_layers_by_budget_accounted(
    candidates: &[HtSegmentAssignmentCandidate],
    candidate_capacity: usize,
    layer_count: usize,
    cumulative_targets: &[u64],
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    retained_live_bytes: usize,
) -> NativeEncodePipelineResult<Vec<usize>> {
    validate_ht_assignment_inputs(
        candidates,
        candidate_capacity,
        layer_count,
        cumulative_targets,
    )?;
    let mut workspace = try_ht_assignment_workspace(
        candidates,
        candidate_capacity,
        layer_count,
        cumulative_targets,
        tracker,
        retained_live_bytes,
    )?;
    for candidate_idx in workspace.candidate_order {
        let candidate = candidates.get(candidate_idx).ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("HTJ2K segment candidate index mismatch")
        })?;
        let min_layer = *workspace
            .block_min_layers
            .get(candidate.block_index)
            .ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant(
                    "HTJ2K segment candidate block index mismatch",
                )
            })?;
        let layer = workspace
            .allocator
            .assign_segment(min_layer, candidate.rate)
            .map_err(NativeEncodePipelineError::arithmetic_overflow)?;
        workspace.assignments[candidate_idx] = layer;
        workspace.block_min_layers[candidate.block_index] = layer;
    }
    Ok(workspace.assignments)
}

fn validate_ht_assignment_inputs(
    candidates: &[HtSegmentAssignmentCandidate],
    candidate_capacity: usize,
    layer_count: usize,
    cumulative_targets: &[u64],
) -> NativeEncodePipelineResult<()> {
    if !cumulative_targets.is_empty() && cumulative_targets.len() != layer_count {
        return Err(NativeEncodePipelineError::invalid_input(
            "quality layer byte target count must match quality layer count",
        ));
    }
    if cumulative_targets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(NativeEncodePipelineError::invalid_input(
            "quality layer byte targets must be cumulative and monotonic",
        ));
    }
    if candidate_capacity < candidates.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HT PCRD candidate capacity is smaller than its length",
        ));
    }
    Ok(())
}

fn try_ht_assignment_workspace(
    candidates: &[HtSegmentAssignmentCandidate],
    candidate_capacity: usize,
    layer_count: usize,
    cumulative_targets: &[u64],
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    retained_live_bytes: usize,
) -> NativeEncodePipelineResult<HtAssignmentWorkspace> {
    let candidate_bytes = checked_element_bytes::<HtSegmentAssignmentCandidate>(
        candidate_capacity,
        "HT PCRD candidates",
    )?;
    let fixed = checked_add_bytes(
        retained_live_bytes,
        candidate_bytes,
        "HT PCRD retained owners",
    )?;
    let (mut targets, target_bytes) = tracker.try_vec::<u64>(
        cumulative_targets.len(),
        [fixed],
        "HT PCRD cumulative targets",
    )?;
    for &target in cumulative_targets {
        targets.push(target.saturating_add(classic_rate_target_tolerance(target)));
    }
    let (mut used, used_bytes) = tracker.try_vec::<u64>(
        cumulative_targets.len(),
        [fixed, target_bytes],
        "HT PCRD cumulative usage",
    )?;
    used.resize(cumulative_targets.len(), 0);
    let allocator = ClassicLayerBudgetAllocator {
        cumulative_targets: targets,
        cumulative_used: used,
    };
    let block_count = ht_assignment_block_count(candidates)?;
    let (mut assignments, assignment_bytes) = tracker.try_vec::<usize>(
        candidates.len(),
        [fixed, target_bytes, used_bytes],
        "HT PCRD segment assignments",
    )?;
    assignments.resize(candidates.len(), layer_count.saturating_sub(1));
    let (mut candidate_order, order_bytes) = tracker.try_vec::<usize>(
        candidates.len(),
        [fixed, target_bytes, used_bytes, assignment_bytes],
        "HT PCRD candidate order",
    )?;
    candidate_order.extend(0..candidates.len());
    candidate_order
        .sort_by_key(|&idx| (candidates[idx].block_index, candidates[idx].segment_index));
    let (mut block_min_layers, block_min_bytes) = tracker.try_vec::<usize>(
        block_count,
        [
            fixed,
            target_bytes,
            used_bytes,
            assignment_bytes,
            order_bytes,
        ],
        "HT PCRD block minimum layers",
    )?;
    block_min_layers.resize(block_count, 0);
    tracker.check(
        [
            fixed,
            target_bytes,
            used_bytes,
            assignment_bytes,
            order_bytes,
            block_min_bytes,
        ],
        "HT PCRD workspace",
    )?;
    Ok(HtAssignmentWorkspace {
        allocator,
        assignments,
        candidate_order,
        block_min_layers,
    })
}

fn ht_assignment_block_count(
    candidates: &[HtSegmentAssignmentCandidate],
) -> NativeEncodePipelineResult<usize> {
    candidates
        .iter()
        .map(|candidate| candidate.block_index)
        .max()
        .map_or(Ok(0usize), |index| {
            index.checked_add(1).ok_or_else(|| {
                NativeEncodePipelineError::arithmetic_overflow("HT PCRD block count")
            })
        })
}
