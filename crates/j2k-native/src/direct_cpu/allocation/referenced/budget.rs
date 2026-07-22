// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    bitplane, ht_block_decode, DirectAllocationBudget, DirectComponentBandScratch,
    DirectComponentPlane, DirectCpuBand, DirectWorkspaceBudget, J2kDirectCpuScratch, Result,
};
use super::view::{
    referenced_band_target, referenced_component_band_count, referenced_component_plane_len,
    referenced_temporary_workspace_bytes, validate_referenced_shape, ReferencedPlanView,
};

pub(super) fn validate_referenced_aggregate_plan(
    plan: ReferencedPlanView<'_>,
    retained_plan_bytes: usize,
    compressed_payload_bytes: usize,
    retained_classic_workspace_dimensions: Option<(u32, u32)>,
    retained_ht_workspace_dimensions: Option<(u32, u32)>,
    scratch: &J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    validate_referenced_shape(plan)?;
    let mut budget = DirectAllocationBudget::default();
    budget.include_bytes(retained_plan_bytes)?;
    let temporary_workspace_bytes = referenced_temporary_workspace_bytes(plan)?;
    include_referenced_scratch_allocations(
        &mut budget,
        plan,
        compressed_payload_bytes,
        retained_classic_workspace_dimensions,
        retained_ht_workspace_dimensions,
        scratch,
    )?;
    let base_bytes = budget.bytes;
    budget.include_bytes(temporary_workspace_bytes)?;
    Ok(DirectWorkspaceBudget {
        base_bytes,
        peak_bytes: budget.bytes,
    })
}

fn include_referenced_scratch_allocations(
    budget: &mut DirectAllocationBudget,
    plan: ReferencedPlanView<'_>,
    compressed_payload_bytes: usize,
    retained_classic_workspace_dimensions: Option<(u32, u32)>,
    retained_ht_workspace_dimensions: Option<(u32, u32)>,
    scratch: &J2kDirectCpuScratch,
) -> Result<()> {
    let component_count = plan.component_count();
    budget.include_capacity::<DirectComponentBandScratch>(
        scratch.component_band_sets.capacity().max(component_count),
    )?;
    budget.include_capacity::<DirectComponentPlane>(
        scratch.component_planes.capacity().max(component_count),
    )?;
    for component_index in 0..component_count {
        let band_count = referenced_component_band_count(plan, component_index)?;
        let component = scratch.component_band_sets.get(component_index);
        budget.include_capacity::<DirectCpuBand>(
            component
                .map_or(0, |component| component.bands.capacity())
                .max(band_count),
        )?;
        for band_index in 0..band_count {
            let target_len = referenced_band_target(plan, component_index, band_index)?;
            let retained_capacity = component
                .and_then(|component| component.bands.get(band_index))
                .map_or(0, |band| band.coefficients.capacity());
            budget.include_capacity::<f32>(retained_capacity.max(target_len))?;
        }
        let plane_len = referenced_component_plane_len(plan, component_index)?;
        let retained_capacity = scratch
            .component_planes
            .get(component_index)
            .map_or(0, |plane| plane.samples.capacity());
        budget.include_capacity::<f32>(retained_capacity.max(plane_len))?;
    }
    budget.include_capacity::<u8>(
        scratch
            .compressed_payload
            .capacity()
            .max(compressed_payload_bytes),
    )?;
    let retained_classic_workspace_bytes = scratch.classic_workspace.allocated_bytes()?;
    let target_classic_workspace_bytes = retained_classic_workspace_dimensions
        .map_or(Ok(0), |(width, height)| {
            bitplane::classic_decode_workspace_bytes(width, height)
        })?;
    budget.include_bytes(retained_classic_workspace_bytes.max(target_classic_workspace_bytes))?;
    let retained_ht_workspace_bytes = scratch.ht_workspace.allocated_bytes()?;
    let target_ht_workspace_bytes = retained_ht_workspace_dimensions
        .map_or(Ok(0), |(width, height)| {
            ht_block_decode::ht_decode_workspace_bytes(width, height)
        })?;
    budget.include_bytes(retained_ht_workspace_bytes.max(target_ht_workspace_bytes))
}
