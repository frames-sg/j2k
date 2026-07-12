// SPDX-License-Identifier: MIT OR Apache-2.0

//! Application of budget-solver results to layered packet state.

use super::super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::super::{
    assign_classic_segment_layers_by_slope_accounted, assign_ht_segment_layers_by_budget_accounted,
    enforce_classic_segment_layer_monotonicity, enforce_ht_segment_layer_monotonicity,
    LayeredPreparedPacket, NativeEncodePipelineError, NativeEncodePipelineResult,
};
use super::ownership::{checked_sum, layered_packets_ownership};
use super::state::LayeredRateControlState;

mod location;
use location::{classic_segment_layer_mut, ht_segment_layer_mut};

pub(super) fn apply_budget_assignments(
    layered_packets: &mut [LayeredPreparedPacket],
    layered_packet_capacity: usize,
    rate_control: &LayeredRateControlState,
    layer_count: usize,
    quality_layer_byte_targets: &[u64],
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    if quality_layer_byte_targets.is_empty() {
        return Ok(());
    }
    let layered_bytes = layered_packets_ownership(layered_packets, layered_packet_capacity)?;
    apply_classic_assignments(
        layered_packets,
        rate_control,
        layer_count,
        quality_layer_byte_targets,
        layered_bytes,
        tracker,
    )?;
    apply_ht_assignments(
        layered_packets,
        rate_control,
        layer_count,
        quality_layer_byte_targets,
        layered_bytes,
        tracker,
    )?;
    Ok(())
}

fn apply_classic_assignments(
    layered_packets: &mut [LayeredPreparedPacket],
    rate_control: &LayeredRateControlState,
    layer_count: usize,
    quality_layer_byte_targets: &[u64],
    layered_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    let classic_retained = checked_sum(
        [
            layered_bytes,
            rate_control.classic_location_bytes,
            rate_control.ht_candidate_bytes,
            rate_control.ht_location_bytes,
        ],
        "classic PCRD retained owners",
    )?;
    let assignments = assign_classic_segment_layers_by_slope_accounted(
        &rate_control.classic_candidates,
        rate_control.classic_candidates.capacity(),
        layer_count,
        quality_layer_byte_targets,
        tracker,
        classic_retained,
    )?;
    if assignments.len() != rate_control.classic_locations.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "classic PCRD assignment location mismatch",
        ));
    }
    for (location, layer) in rate_control.classic_locations.iter().zip(assignments) {
        *classic_segment_layer_mut(layered_packets, location)? = layer;
    }
    enforce_classic_segment_layer_monotonicity(layered_packets);
    Ok(())
}

fn apply_ht_assignments(
    layered_packets: &mut [LayeredPreparedPacket],
    rate_control: &LayeredRateControlState,
    layer_count: usize,
    quality_layer_byte_targets: &[u64],
    layered_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    let ht_retained = checked_sum(
        [
            layered_bytes,
            rate_control.classic_candidate_bytes,
            rate_control.classic_location_bytes,
            rate_control.ht_location_bytes,
        ],
        "HT PCRD retained owners",
    )?;
    let assignments = assign_ht_segment_layers_by_budget_accounted(
        &rate_control.ht_candidates,
        rate_control.ht_candidates.capacity(),
        layer_count,
        quality_layer_byte_targets,
        tracker,
        ht_retained,
    )?;
    if assignments.len() != rate_control.ht_locations.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HTJ2K segment assignment location mismatch",
        ));
    }
    for (location, layer) in rate_control.ht_locations.iter().zip(assignments) {
        *ht_segment_layer_mut(layered_packets, location)? = layer;
    }
    enforce_ht_segment_layer_monotonicity(layered_packets);
    Ok(())
}
