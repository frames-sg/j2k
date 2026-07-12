// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible construction of nested ROI plan owners.

use alloc::vec::Vec;

use crate::j2c::encode::{
    validate_roi_encode_options_nonallocating, BlockCodingMode, ComponentRoiEncodePlan,
    ComponentRoiEncodeRegion, EncodeOptions, EncodeRoiRegion, NativeEncodePipelineError,
    NativeEncodePipelineResult,
};

use super::PlanConstruction;

#[expect(
    clippy::too_many_arguments,
    reason = "ROI construction keeps validated image, sampling, bitplane, and coding policy explicit"
)]
pub(in crate::j2c::encode::single_tile::plan) fn try_roi_plans(
    construction: &mut PlanConstruction<'_, '_>,
    options: &EncodeOptions,
    roi_regions: &[EncodeRoiRegion],
    num_components: u16,
    width: u32,
    height: u32,
    component_sampling: &[(u8, u8)],
    base_bitplanes: u8,
    block_coding_mode: BlockCodingMode,
) -> NativeEncodePipelineResult<Vec<ComponentRoiEncodePlan>> {
    validate_roi_encode_options_nonallocating(
        options,
        roi_regions,
        num_components,
        width,
        height,
        base_bitplanes,
        block_coding_mode,
    )?;

    let component_count = usize::from(num_components);
    let mut plans = construction.try_vec(component_count, "single-tile ROI plan owners")?;
    for component in 0..component_count {
        let shift = options
            .roi_component_shifts
            .get(component)
            .copied()
            .unwrap_or(0);
        plans.push(ComponentRoiEncodePlan {
            shift,
            planned_region_count: 0,
            regions: Vec::new(),
        });
    }
    for region in roi_regions {
        let plan = plans
            .get_mut(usize::from(region.component))
            .ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant(
                    "validated ROI region component index is out of range",
                )
            })?;
        plan.planned_region_count = plan.planned_region_count.checked_add(1).ok_or(
            crate::EncodeError::ArithmeticOverflow {
                what: "single-tile ROI region count",
            },
        )?;
    }
    for plan in &mut plans {
        let count = usize::try_from(plan.planned_region_count).map_err(|_| {
            crate::EncodeError::ArithmeticOverflow {
                what: "single-tile ROI region count",
            }
        })?;
        plan.regions = construction.try_vec(count, "single-tile ROI region owners")?;
    }

    for region in roi_regions {
        append_roi_region(&mut plans, region, width, height, component_sampling)?;
    }
    Ok(plans)
}

fn append_roi_region(
    plans: &mut [ComponentRoiEncodePlan],
    region: &EncodeRoiRegion,
    width: u32,
    height: u32,
    component_sampling: &[(u8, u8)],
) -> NativeEncodePipelineResult<()> {
    let horizontal_end = region.x.checked_add(region.width).ok_or_else(|| {
        NativeEncodePipelineError::internal_invariant(
            "validated ROI horizontal bounds overflowed during construction",
        )
    })?;
    let vertical_end = region.y.checked_add(region.height).ok_or_else(|| {
        NativeEncodePipelineError::internal_invariant(
            "validated ROI vertical bounds overflowed during construction",
        )
    })?;
    let component = usize::from(region.component);
    let plan = plans.get_mut(component).ok_or_else(|| {
        NativeEncodePipelineError::internal_invariant(
            "validated ROI region component index is out of range",
        )
    })?;
    if plan.shift == 0 {
        plan.shift = region.shift;
    }

    let &(horizontal_sampling, vertical_sampling) =
        component_sampling.get(component).ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "component sampling count does not match component count",
            )
        })?;
    let component_width = width.div_ceil(u32::from(horizontal_sampling));
    let component_height = height.div_ceil(u32::from(vertical_sampling));
    let component_horizontal_start = region.x / u32::from(horizontal_sampling);
    let component_vertical_start = region.y / u32::from(vertical_sampling);
    let component_horizontal_end = horizontal_end
        .div_ceil(u32::from(horizontal_sampling))
        .min(component_width);
    let component_vertical_end = vertical_end
        .div_ceil(u32::from(vertical_sampling))
        .min(component_height);
    if component_horizontal_start >= component_horizontal_end
        || component_vertical_start >= component_vertical_end
    {
        return Err(NativeEncodePipelineError::internal_invariant(
            "validated ROI region does not intersect component grid",
        ));
    }
    plan.regions.push(ComponentRoiEncodeRegion {
        x: component_horizontal_start,
        y: component_vertical_start,
        width: component_horizontal_end - component_horizontal_start,
        height: component_vertical_end - component_vertical_start,
    });
    Ok(())
}
