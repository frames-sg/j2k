// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use j2k_core::PixelFormat;
use j2k_core::Rect;
use j2k_native::J2kDirectColorPlan;

use crate::Error;

#[cfg(test)]
use super::direct_plan_types::PreparedDirectGrayscaleStep;
use super::{
    direct_plan_types::{
        PreparedClassicSubBandGroup, PreparedDirectColorPlan, PreparedDirectGrayscalePlan,
        PreparedHtSubBandGroup,
    },
    direct_prepare::{prepare_direct_grayscale_plan, prepare_direct_grayscale_plan_for_cpu_upload},
    direct_roi::crop_prepared_direct_grayscale_plan_to_output_region,
    direct_tier1::DirectTier1Mode,
};

pub(crate) fn prepare_direct_color_plan(
    plan: &J2kDirectColorPlan,
) -> Result<PreparedDirectColorPlan, Error> {
    prepare_direct_color_plan_with_tier1_mode(plan, DirectTier1Mode::Metal)
}

pub(crate) fn prepare_direct_color_plan_for_cpu_upload(
    plan: &J2kDirectColorPlan,
) -> Result<PreparedDirectColorPlan, Error> {
    prepare_direct_color_plan_with_tier1_mode(plan, DirectTier1Mode::CpuUpload)
}

fn prepare_direct_color_plan_with_tier1_mode(
    plan: &J2kDirectColorPlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedDirectColorPlan, Error> {
    if plan.component_plans.len() != 3 {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect color plan expected 3 component plans, got {}",
                plan.component_plans.len()
            ),
        });
    }
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect prepared color component plans",
    );
    let mut component_plans = budget.try_vec(
        plan.component_plans.len(),
        "J2K MetalDirect prepared color component plans",
    )?;
    for component in &plan.component_plans {
        component_plans.push(match tier1_prepare_mode {
            DirectTier1Mode::Metal => prepare_direct_grayscale_plan(component),
            DirectTier1Mode::CpuUpload => prepare_direct_grayscale_plan_for_cpu_upload(component),
        }?);
    }
    Ok(PreparedDirectColorPlan {
        dimensions: plan.dimensions,
        bit_depths: plan.bit_depths,
        alpha_bit_depth: None,
        signed: false,
        mct: plan.mct,
        transform: plan.transform,
        component_plans,
    })
}

pub(crate) fn crop_prepared_direct_color_plan_to_output_region(
    plan: &mut PreparedDirectColorPlan,
    region: Rect,
) -> Result<(), Error> {
    if region.w == 0 || region.h == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect region-scaled color plan has an empty output region"
                .to_string(),
        });
    }

    for component_plan in &mut plan.component_plans {
        crop_prepared_direct_grayscale_plan_to_output_region(component_plan, region)?;
        if component_plan.dimensions != (region.w, region.h) {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K MetalDirect color component crop produced {:?}, expected {:?}",
                    component_plan.dimensions,
                    (region.w, region.h)
                ),
            });
        }
    }

    plan.dimensions = (region.w, region.h);
    Ok(())
}

impl PreparedDirectGrayscalePlan {
    pub(in crate::compute) fn classic_group_starting_at(
        &self,
        step_idx: usize,
    ) -> Option<&PreparedClassicSubBandGroup> {
        self.classic_groups
            .iter()
            .find(|group| group.start_step == step_idx)
    }

    pub(in crate::compute) fn ht_group_starting_at(
        &self,
        step_idx: usize,
    ) -> Option<&PreparedHtSubBandGroup> {
        self.ht_groups
            .iter()
            .find(|group| group.start_step == step_idx)
    }
}

#[cfg(test)]
pub(in crate::compute) fn prepared_direct_grayscale_plan_compute_encoder_count(
    plan: &PreparedDirectGrayscalePlan,
    _fmt: PixelFormat,
) -> usize {
    usize::from(!plan.steps.is_empty())
}

#[cfg(test)]
pub(in crate::compute) fn prepared_repeated_direct_ht_cleanup_dispatch_count(
    plan: &PreparedDirectGrayscalePlan,
) -> usize {
    let mut dispatches = 0;
    let mut step_idx = 0;
    while step_idx < plan.steps.len() {
        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            dispatches += 1;
            step_idx = group.end_step;
            continue;
        }
        if matches!(
            plan.steps[step_idx],
            PreparedDirectGrayscaleStep::HtSubBand(_)
        ) {
            dispatches += 1;
        }
        step_idx += 1;
    }
    dispatches
}
