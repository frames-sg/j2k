// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared ordering and monotonicity rules for layer assignment.

use super::super::Ordering;
use super::{ClassicSegmentAssignmentCandidate, LayeredPreparedBlock, LayeredPreparedPacket};

mod accounted;
pub(in crate::j2c::encode) use accounted::{
    assign_classic_segment_layers_by_slope_accounted, assign_ht_segment_layers_by_budget_accounted,
};
#[cfg(test)]
mod legacy;
#[cfg(test)]
pub(in crate::j2c::encode) use legacy::{
    assign_classic_segment_layers_by_slope, assign_ht_segment_layers_by_budget,
};

fn compare_classic_segment_candidates(
    candidates: &[ClassicSegmentAssignmentCandidate],
    left: usize,
    right: usize,
) -> Ordering {
    let left_candidate = candidates[left];
    let right_candidate = candidates[right];
    pcrd_slope(right_candidate)
        .partial_cmp(&pcrd_slope(left_candidate))
        .unwrap_or(Ordering::Equal)
        .then_with(|| left_candidate.block_index.cmp(&right_candidate.block_index))
        .then_with(|| {
            left_candidate
                .segment_index
                .cmp(&right_candidate.segment_index)
        })
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the codec float domain intentionally receives bounded integer samples or metadata at this rounding boundary"
)]
fn pcrd_slope(candidate: ClassicSegmentAssignmentCandidate) -> f64 {
    if candidate.rate == 0 {
        return f64::INFINITY;
    }
    candidate.distortion_delta / candidate.rate as f64
}

fn enforce_classic_assignment_monotonicity(
    candidates: &[ClassicSegmentAssignmentCandidate],
    assignments: &mut [usize],
) {
    for candidate_idx in 0..candidates.len() {
        let candidate = candidates[candidate_idx];
        let min_layer = candidates
            .iter()
            .enumerate()
            .filter(|(_, prior)| {
                prior.block_index == candidate.block_index
                    && prior.segment_index <= candidate.segment_index
            })
            .map(|(prior_idx, _)| assignments[prior_idx])
            .max()
            .unwrap_or(0);
        assignments[candidate_idx] = assignments[candidate_idx].max(min_layer);
    }
}

pub(in crate::j2c::encode) fn enforce_classic_segment_layer_monotonicity(
    layered_packets: &mut [LayeredPreparedPacket],
) {
    for packet in layered_packets {
        for subband in &mut packet.subbands {
            for block in &mut subband.blocks {
                if let LayeredPreparedBlock::Classic { segment_layers, .. } = block {
                    let mut min_layer = 0usize;
                    for layer in segment_layers {
                        if *layer < min_layer {
                            *layer = min_layer;
                        }
                        min_layer = *layer;
                    }
                }
            }
        }
    }
}

pub(in crate::j2c::encode) fn enforce_ht_segment_layer_monotonicity(
    layered_packets: &mut [LayeredPreparedPacket],
) {
    for packet in layered_packets {
        for subband in &mut packet.subbands {
            for block in &mut subband.blocks {
                if let LayeredPreparedBlock::HighThroughput { segment_layers, .. } = block {
                    let mut min_layer = 0usize;
                    for layer in segment_layers {
                        if *layer < min_layer {
                            *layer = min_layer;
                        }
                        min_layer = *layer;
                    }
                }
            }
        }
    }
}
