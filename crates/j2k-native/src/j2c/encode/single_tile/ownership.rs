// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact allocator-capacity accounting for live single-tile plans.

use alloc::vec::Vec;

use super::super::allocation::{checked_add_bytes, checked_element_bytes};
use super::super::{ComponentRoiEncodePlan, QuantStepSize};
use super::plan::{CodestreamFinalPlan, SingleTilePlan};
use crate::EncodeResult;

mod params;
mod prepared;
mod transform;

pub(in crate::j2c::encode) use params::encode_params_retained_bytes;
pub(super) use prepared::{prepared_packet_tree_retained_bytes, prepared_packets_retained_bytes};
pub(super) use transform::{
    component_planes_retained_bytes, dwt_component_sources_retained_bytes,
    prepared_transforms_retained_bytes,
};
pub(in crate::j2c::encode) use transform::{
    cpu_dwt_transient_bytes, dwt_decompositions_retained_bytes,
};

pub(super) fn single_tile_plan_retained_bytes(plan: &SingleTilePlan) -> EncodeResult<usize> {
    let mut bytes = 0;
    bytes = add_capacity::<QuantStepSize>(bytes, plan.step_sizes.capacity(), "tile step sizes")?;
    bytes = add_capacity::<(u16, u16)>(bytes, plan.quant_params.capacity(), "tile quantization")?;
    bytes = add_capacity::<Vec<QuantStepSize>>(
        bytes,
        plan.component_step_sizes.capacity(),
        "tile component step vectors",
    )?;
    for steps in &plan.component_step_sizes {
        bytes = add_capacity::<QuantStepSize>(bytes, steps.capacity(), "tile component steps")?;
    }
    bytes =
        add_capacity::<ComponentRoiEncodePlan>(bytes, plan.roi_plans.capacity(), "tile ROI plans")?;
    for roi in &plan.roi_plans {
        if usize::try_from(roi.planned_region_count).ok() != Some(roi.regions.len()) {
            return Err(crate::EncodeError::InternalInvariant {
                what: "tile ROI plan region count does not match its retained owner",
            });
        }
        bytes = add_capacity::<super::super::ComponentRoiEncodeRegion>(
            bytes,
            roi.regions.capacity(),
            "tile ROI regions",
        )?;
    }
    bytes = add_capacity::<u8>(
        bytes,
        plan.roi_component_shifts.capacity(),
        "tile ROI shifts",
    )?;

    checked_add_bytes(
        bytes,
        encode_params_retained_bytes(&plan.params)?,
        "tile encode parameters",
    )
}

pub(super) fn codestream_final_plan_retained_bytes(
    plan: &CodestreamFinalPlan,
) -> EncodeResult<usize> {
    checked_add_bytes(
        encode_params_retained_bytes(&plan.params)?,
        checked_element_bytes::<(u16, u16)>(
            plan.quant_params.capacity(),
            "final tile quantization",
        )?,
        "final tile marker plan",
    )
}

fn add_capacity<T>(bytes: usize, capacity: usize, what: &'static str) -> EncodeResult<usize> {
    checked_add_bytes(bytes, checked_element_bytes::<T>(capacity, what)?, what)
}
