// SPDX-License-Identifier: MIT OR Apache-2.0

//! Session-accounted candidate workspaces and segment assignments.

use super::super::super::allocation::{checked_add_bytes, checked_element_bytes};
use super::super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::super::{NativeEncodePipelineError, NativeEncodePipelineResult, Vec};
use super::super::{
    classic_rate_target_tolerance, ClassicLayerBudgetAllocator, HtSegmentAssignmentCandidate,
};
use super::{
    compare_ht_segment_candidates, enforce_ht_assignment_monotonicity,
    ordered_candidate_block_count,
};

mod classic;
pub(in crate::j2c::encode) use classic::assign_classic_segment_layers_by_slope_accounted;

struct HtAssignmentWorkspace {
    allocator: ClassicLayerBudgetAllocator,
    assignments: Vec<usize>,
    block_frontiers: Vec<Option<usize>>,
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
    let block_count = validate_ht_assignment_inputs(
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
        block_count,
        tracker,
        retained_live_bytes,
    )?;
    for _ in 0..candidates.len() {
        let candidate_idx = workspace
            .block_frontiers
            .iter()
            .filter_map(|candidate| *candidate)
            .min_by(|&left, &right| compare_ht_segment_candidates(candidates, left, right))
            .ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant(
                    "HTJ2K PCRD candidate queue underflow",
                )
            })?;
        let candidate = candidates[candidate_idx];
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
        let block_index = candidate.block_index;
        workspace.block_min_layers[block_index] = layer;
        workspace.block_frontiers[block_index] = candidate_idx.checked_add(1).filter(|&next| {
            candidates
                .get(next)
                .is_some_and(|candidate| candidate.block_index == block_index)
        });
    }
    enforce_ht_assignment_monotonicity(candidates, &mut workspace.assignments);
    Ok(workspace.assignments)
}

fn validate_ht_assignment_inputs(
    candidates: &[HtSegmentAssignmentCandidate],
    candidate_capacity: usize,
    layer_count: usize,
    cumulative_targets: &[u64],
) -> NativeEncodePipelineResult<usize> {
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
    ordered_candidate_block_count(
        candidates
            .iter()
            .map(|candidate| (candidate.block_index, candidate.segment_index)),
        "HTJ2K PCRD candidates are not grouped in segment order",
    )
}

fn try_ht_assignment_workspace(
    candidates: &[HtSegmentAssignmentCandidate],
    candidate_capacity: usize,
    layer_count: usize,
    cumulative_targets: &[u64],
    block_count: usize,
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
    let (mut block_min_layers, block_min_bytes) = tracker.try_vec::<usize>(
        block_count,
        [fixed, target_bytes, used_bytes],
        "HT PCRD block minimum layers",
    )?;
    block_min_layers.resize(block_count, 0);
    let (mut block_frontiers, frontier_bytes) = tracker.try_vec::<Option<usize>>(
        block_count,
        [fixed, target_bytes, used_bytes, block_min_bytes],
        "HT PCRD block frontiers",
    )?;
    block_frontiers.resize(block_count, None);
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        if candidate.segment_index == 0 {
            block_frontiers[candidate.block_index] = Some(candidate_index);
        }
    }
    let live = [
        fixed,
        target_bytes,
        used_bytes,
        block_min_bytes,
        frontier_bytes,
    ];
    let (mut assignments, assignment_bytes) =
        tracker.try_vec::<usize>(candidates.len(), live, "HT PCRD segment assignments")?;
    assignments.resize(candidates.len(), layer_count.saturating_sub(1));
    tracker.check(
        live.into_iter().chain([assignment_bytes]),
        "HT PCRD workspace",
    )?;
    Ok(HtAssignmentWorkspace {
        allocator,
        assignments,
        block_frontiers,
        block_min_layers,
    })
}
