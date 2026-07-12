// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate direct-plan input, retained scratch, and scalar workspace accounting.

use super::{
    checked_area, DirectComponentBandScratch, DirectComponentPlane, DirectCpuBand,
    J2kDirectCpuScratch,
};
use crate::error::{DecodeError, Result, ValidationError};
use crate::j2c::{bitplane, ht_block_decode};
use crate::{
    try_reserve_decode_elements, J2kDirectColorPlan, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, DEFAULT_MAX_DECODE_BYTES,
};
use alloc::vec::Vec;
use core::mem::size_of;

pub(super) fn prepare_direct_scratch(
    plan: &J2kDirectColorPlan,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    normalize_retained_scratch(plan, scratch)?;
    if let Err(error) = validate_aggregate_plan(plan, scratch) {
        if !matches!(
            error,
            DecodeError::Validation(ValidationError::ImageTooLarge)
        ) {
            return Err(error);
        }
        // Retention is optional. Retry from an empty owner so prior larger
        // plans cannot make the current logical request fail.
        scratch.clear();
        validate_aggregate_plan(plan, scratch)?;
    }

    if let Err(error) = reserve_scratch(plan, scratch) {
        scratch.clear();
        return Err(error);
    }
    match validate_aggregate_plan(plan, scratch) {
        Ok(workspace_budget) => Ok(workspace_budget),
        Err(error) => {
            scratch.clear();
            Err(error)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DirectWorkspaceBudget {
    base_bytes: usize,
    peak_bytes: usize,
}

impl DirectWorkspaceBudget {
    pub(super) fn validate_workspace(self, workspace_bytes: usize) -> Result<()> {
        let total = self
            .base_bytes
            .checked_add(workspace_bytes)
            .ok_or(ValidationError::ImageTooLarge)?;
        if total > DEFAULT_MAX_DECODE_BYTES {
            return Err(ValidationError::ImageTooLarge.into());
        }
        Ok(())
    }

    pub(super) const fn peak_bytes(self) -> usize {
        self.peak_bytes
    }
}

fn normalize_retained_scratch(
    plan: &J2kDirectColorPlan,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<()> {
    let component_count = plan.component_plans.len();
    normalize_outer_owner(&mut scratch.component_band_sets, component_count);
    normalize_outer_owner(&mut scratch.component_planes, component_count);

    fill_without_allocation(
        &mut scratch.component_band_sets,
        component_count,
        DirectComponentBandScratch::default,
    );
    fill_without_allocation(
        &mut scratch.component_planes,
        component_count,
        DirectComponentPlane::default,
    );

    for (component_idx, component_plan) in plan.component_plans.iter().enumerate() {
        let band_count = component_band_count(component_plan)?;
        if let Some(component) = scratch.component_band_sets.get_mut(component_idx) {
            normalize_outer_owner(&mut component.bands, band_count);
            fill_without_allocation(&mut component.bands, band_count, DirectCpuBand::empty);
            for_each_band_target(component_plan, |band_idx, target_len| {
                if let Some(band) = component.bands.get_mut(band_idx) {
                    band.coefficients.clear();
                    if band.coefficients.capacity() < target_len {
                        band.coefficients = Vec::new();
                    }
                }
                Ok(())
            })?;
            component.active_len = 0;
        }

        let plane_len = component_plane_len(component_plan)?;
        if let Some(plane) = scratch.component_planes.get_mut(component_idx) {
            plane.samples.clear();
            if plane.samples.capacity() < plane_len {
                plane.samples = Vec::new();
            }
        }
    }
    Ok(())
}

fn normalize_outer_owner<T>(values: &mut Vec<T>, target_len: usize) {
    if values.capacity() < target_len {
        *values = Vec::new();
    } else {
        values.truncate(target_len);
    }
}

fn fill_without_allocation<T>(
    values: &mut Vec<T>,
    target_len: usize,
    mut new_value: impl FnMut() -> T,
) {
    while values.len() < target_len && values.len() < values.capacity() {
        values.push(new_value());
    }
}

fn validate_aggregate_plan(
    plan: &J2kDirectColorPlan,
    scratch: &J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    let mut budget = DirectAllocationBudget::default();
    let temporary_workspace_bytes = include_plan_allocations(&mut budget, plan)?;
    include_scratch_allocations(&mut budget, plan, scratch)?;
    let base_bytes = budget.bytes;
    budget.include_bytes(temporary_workspace_bytes)?;
    Ok(DirectWorkspaceBudget {
        base_bytes,
        peak_bytes: budget.bytes,
    })
}

#[derive(Default)]
struct DirectAllocationBudget {
    bytes: usize,
}

impl DirectAllocationBudget {
    fn include_capacity<T>(&mut self, capacity: usize) -> Result<()> {
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .ok_or(ValidationError::ImageTooLarge)?;
        self.include_bytes(bytes)
    }

    fn include_bytes(&mut self, bytes: usize) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or(ValidationError::ImageTooLarge)?;
        if self.bytes > DEFAULT_MAX_DECODE_BYTES {
            return Err(ValidationError::ImageTooLarge.into());
        }
        Ok(())
    }
}

fn include_plan_allocations(
    budget: &mut DirectAllocationBudget,
    plan: &J2kDirectColorPlan,
) -> Result<usize> {
    budget.include_bytes(plan.retained_allocation_bytes()?)?;
    let mut classic_dimensions = None;
    let mut ht_dimensions = None;
    for component in &plan.component_plans {
        for step in &component.steps {
            match step {
                J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    for job in &sub_band.jobs {
                        observe_max_dimensions(&mut classic_dimensions, job.width, job.height);
                    }
                }
                J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                    for job in &sub_band.jobs {
                        observe_max_dimensions(&mut ht_dimensions, job.width, job.height);
                    }
                }
                J2kDirectGrayscaleStep::Idwt(_) | J2kDirectGrayscaleStep::Store(_) => {}
            }
        }
    }
    let classic_bytes = classic_dimensions.map_or(Ok(0), |(width, height)| {
        bitplane::classic_decode_workspace_bytes(width, height)
    })?;
    let ht_bytes = ht_dimensions.map_or(Ok(0), |(width, height)| {
        ht_block_decode::ht_decode_workspace_bytes(width, height)
    })?;
    Ok(classic_bytes.max(ht_bytes))
}

fn observe_max_dimensions(target: &mut Option<(u32, u32)>, width: u32, height: u32) {
    *target = Some(
        target.map_or((width, height), |(current_width, current_height)| {
            (current_width.max(width), current_height.max(height))
        }),
    );
}

fn include_scratch_allocations(
    budget: &mut DirectAllocationBudget,
    plan: &J2kDirectColorPlan,
    scratch: &J2kDirectCpuScratch,
) -> Result<()> {
    let component_count = plan.component_plans.len();
    budget.include_capacity::<DirectComponentBandScratch>(
        scratch.component_band_sets.capacity().max(component_count),
    )?;
    budget.include_capacity::<DirectComponentPlane>(
        scratch.component_planes.capacity().max(component_count),
    )?;

    for (component_idx, component_plan) in plan.component_plans.iter().enumerate() {
        let band_count = component_band_count(component_plan)?;
        let component = scratch.component_band_sets.get(component_idx);
        budget.include_capacity::<DirectCpuBand>(
            component
                .map_or(0, |component| component.bands.capacity())
                .max(band_count),
        )?;
        for_each_band_target(component_plan, |band_idx, target_len| {
            let retained_capacity = component
                .and_then(|component| component.bands.get(band_idx))
                .map_or(0, |band| band.coefficients.capacity());
            budget.include_capacity::<f32>(retained_capacity.max(target_len))
        })?;

        let plane_len = component_plane_len(component_plan)?;
        let retained_capacity = scratch
            .component_planes
            .get(component_idx)
            .map_or(0, |plane| plane.samples.capacity());
        budget.include_capacity::<f32>(retained_capacity.max(plane_len))?;
    }
    Ok(())
}

fn reserve_scratch(plan: &J2kDirectColorPlan, scratch: &mut J2kDirectCpuScratch) -> Result<()> {
    let component_count = plan.component_plans.len();
    try_reserve_decode_elements(&mut scratch.component_band_sets, component_count)?;
    try_reserve_decode_elements(&mut scratch.component_planes, component_count)?;
    while scratch.component_band_sets.len() < component_count {
        scratch
            .component_band_sets
            .push(DirectComponentBandScratch::default());
    }
    while scratch.component_planes.len() < component_count {
        scratch
            .component_planes
            .push(DirectComponentPlane::default());
    }

    for (component_idx, component_plan) in plan.component_plans.iter().enumerate() {
        let component = &mut scratch.component_band_sets[component_idx];
        let band_count = component_band_count(component_plan)?;
        try_reserve_decode_elements(&mut component.bands, band_count)?;
        while component.bands.len() < band_count {
            component.bands.push(DirectCpuBand::empty());
        }
        for_each_band_target(component_plan, |band_idx, target_len| {
            try_reserve_decode_elements(&mut component.bands[band_idx].coefficients, target_len)
        })?;

        let plane_len = component_plane_len(component_plan)?;
        try_reserve_decode_elements(
            &mut scratch.component_planes[component_idx].samples,
            plane_len,
        )?;
    }
    Ok(())
}

fn component_band_count(plan: &J2kDirectGrayscalePlan) -> Result<usize> {
    plan.steps.iter().try_fold(0usize, |count, step| {
        if matches!(
            step,
            J2kDirectGrayscaleStep::ClassicSubBand(_)
                | J2kDirectGrayscaleStep::HtSubBand(_)
                | J2kDirectGrayscaleStep::Idwt(_)
        ) {
            count
                .checked_add(1)
                .ok_or(ValidationError::ImageTooLarge.into())
        } else {
            Ok(count)
        }
    })
}

fn component_plane_len(plan: &J2kDirectGrayscalePlan) -> Result<usize> {
    let mut dimensions = None;
    for step in &plan.steps {
        let J2kDirectGrayscaleStep::Store(store) = step else {
            continue;
        };
        let current = (store.output_width, store.output_height);
        if dimensions.is_some_and(|dimensions| dimensions != current) {
            return Err(crate::DecodingError::CodeBlockDecodeFailure.into());
        }
        dimensions = Some(current);
    }
    dimensions.map_or(Ok(0), |(width, height)| checked_area(width, height))
}

fn for_each_band_target(
    plan: &J2kDirectGrayscalePlan,
    mut visit: impl FnMut(usize, usize) -> Result<()>,
) -> Result<()> {
    let mut band_idx = 0usize;
    for step in &plan.steps {
        let target_len = match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                Some(checked_area(sub_band.width, sub_band.height)?)
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                Some(checked_area(sub_band.width, sub_band.height)?)
            }
            J2kDirectGrayscaleStep::Idwt(step) => {
                Some(checked_area(step.rect.width(), step.rect.height())?)
            }
            J2kDirectGrayscaleStep::Store(_) => None,
        };
        if let Some(target_len) = target_len {
            visit(band_idx, target_len)?;
            band_idx = band_idx
                .checked_add(1)
                .ok_or(ValidationError::ImageTooLarge)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DirectAllocationBudget, DirectWorkspaceBudget};
    use crate::error::{DecodeError, ValidationError};
    use crate::DEFAULT_MAX_DECODE_BYTES;

    #[test]
    fn aggregate_budget_has_an_exact_shared_cap_boundary() {
        let mut budget = DirectAllocationBudget {
            bytes: DEFAULT_MAX_DECODE_BYTES - 1,
        };
        budget.include_bytes(1).expect("exact boundary fits");
        assert_eq!(
            budget.include_bytes(1),
            Err(DecodeError::Validation(ValidationError::ImageTooLarge))
        );
    }

    #[test]
    fn actual_scalar_workspace_uses_the_remaining_direct_budget() {
        let budget = DirectWorkspaceBudget {
            base_bytes: DEFAULT_MAX_DECODE_BYTES - 1,
            peak_bytes: DEFAULT_MAX_DECODE_BYTES - 1,
        };
        budget.validate_workspace(1).expect("exact boundary fits");
        assert_eq!(
            budget.validate_workspace(2),
            Err(DecodeError::Validation(ValidationError::ImageTooLarge))
        );
    }
}
