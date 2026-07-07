// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
mod direct_plan_types;
#[cfg(target_os = "macos")]
pub(crate) use self::direct_plan_types::{PreparedDirectColorPlan, PreparedDirectGrayscalePlan};
#[cfg(target_os = "macos")]
use self::direct_plan_types::{
    HtCodedArena, PreparedClassicSubBand, PreparedClassicSubBandGroup,
    PreparedClassicSubBandGroupMember, PreparedDirectGrayscaleStep, PreparedDirectIdwt,
    PreparedHtSubBand, PreparedHtSubBandGroup, PreparedHtSubBandGroupMember,
};
#[cfg(target_os = "macos")]
mod direct_plane_pack;
#[cfg(target_os = "macos")]
use self::direct_plane_pack::{PlaneStage, encode_mct_rgb8_to_surface_in_command_buffer, encode_plane_stage_to_surface_in_command_buffer, repeated_shared_direct_color_plan_count, encode_repeated_mct_rgb8_to_surfaces_in_command_buffer, encode_batched_mct_rgb8_to_surfaces_in_command_buffer};
#[cfg(target_os = "macos")]
mod direct_prepare;
#[cfg(target_os = "macos")]
pub(crate) use self::direct_prepare::*;
#[cfg(target_os = "macos")]
mod direct_roi;
#[cfg(target_os = "macos")]
pub(crate) use self::direct_roi::*;
#[cfg(target_os = "macos")]
mod direct_grayscale_execute;
#[cfg(target_os = "macos")]
pub(crate) use self::direct_grayscale_execute::*;
#[cfg(target_os = "macos")]
mod direct_stacked_batch;
#[cfg(target_os = "macos")]
use self::direct_stacked_batch::{signed_sample_bias, DirectBandSlice, lookup_direct_band_slice_entry, lookup_direct_band_slice, encode_repeated_direct_grayscale_plan_in_command_buffer, RepeatedDirectGrayscalePlanRequest, supports_stacked_direct_component_plane_batch, encode_stacked_direct_component_plane_batch, StackedDirectComponentPlaneBatchRequest, try_encode_stacked_mct_rgb8_direct_color_batch, StackedDirectColorBatchRequest, encode_prepared_direct_color_plan_in_command_buffer, DirectColorPlanRequest};
#[cfg(target_os = "macos")]
mod direct_surface_pack;
#[cfg(target_os = "macos")]
use self::direct_surface_pack::{
    copy_plane_samples, encode_gray_plane_to_surface_in_command_buffer_with_offset,
    encode_gray_plane_to_surface_in_encoder, encode_repeated_gray_plane_to_surfaces_in_command_buffer,
    output_shape_for,
};
#[cfg(all(target_os = "macos", test))]
use self::direct_surface_pack::j2k_pack_kernel_name_for;

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_color_plan(
    plan: &J2kDirectColorPlan,
) -> Result<PreparedDirectColorPlan, Error> {
    prepare_direct_color_plan_with_tier1_mode(plan, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_color_plan_for_cpu_upload(
    plan: &J2kDirectColorPlan,
) -> Result<PreparedDirectColorPlan, Error> {
    prepare_direct_color_plan_with_tier1_mode(plan, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
fn prepare_direct_color_plan_with_tier1_mode(
    plan: &J2kDirectColorPlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedDirectColorPlan, Error> {
    let component_plans = plan
        .component_plans
        .iter()
        .map(|component| match tier1_prepare_mode {
            DirectTier1Mode::Metal => prepare_direct_grayscale_plan(component),
            DirectTier1Mode::CpuUpload => prepare_direct_grayscale_plan_for_cpu_upload(component),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if component_plans.len() != 3 {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect color plan expected 3 component plans, got {}",
                component_plans.len()
            ),
        });
    }
    Ok(PreparedDirectColorPlan {
        dimensions: plan.dimensions,
        bit_depths: plan.bit_depths,
        mct: plan.mct,
        transform: plan.transform,
        component_plans,
    })
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
impl PreparedDirectGrayscalePlan {
    fn classic_group_starting_at(&self, step_idx: usize) -> Option<&PreparedClassicSubBandGroup> {
        self.classic_groups
            .iter()
            .find(|group| group.start_step == step_idx)
    }

    fn ht_group_starting_at(&self, step_idx: usize) -> Option<&PreparedHtSubBandGroup> {
        self.ht_groups
            .iter()
            .find(|group| group.start_step == step_idx)
    }
}

#[cfg(all(test, target_os = "macos"))]
fn prepared_direct_grayscale_plan_compute_encoder_count(
    plan: &PreparedDirectGrayscalePlan,
    _fmt: PixelFormat,
) -> usize {
    usize::from(!plan.steps.is_empty())
}

#[cfg(all(test, target_os = "macos"))]
fn prepared_repeated_direct_ht_cleanup_dispatch_count(plan: &PreparedDirectGrayscalePlan) -> usize {
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
