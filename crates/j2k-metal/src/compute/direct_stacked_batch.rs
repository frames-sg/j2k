// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use metal::{Buffer, CommandBufferRef};

use super::{
    build_flattened_cpu_tier1_cache, dispatch_inverse_mct_buffers_in_command_buffer, elapsed_us,
    encode_batched_mct_rgb8_to_surfaces_in_command_buffer,
    encode_mct_rgb8_to_surface_in_command_buffer, encode_plane_stage_to_surface_in_command_buffer,
    encode_prepared_direct_component_plane_in_command_buffer,
    encode_repeated_mct_rgb8_to_surfaces_in_command_buffer, flattened_hybrid_cpu_tier1_enabled,
    metal_profile_stages_enabled, repeated_shared_direct_color_plan_count,
    should_flatten_hybrid_cpu_tier1_color_batch, DirectColorBatchCommandBuffers,
    DirectComponentPlaneRequest, DirectHybridStageTimings, DirectScratchBuffer, DirectStatusCheck,
    DirectTier1Mode, Error, FlattenedCpuTier1Cache, MetalRuntime, NativeColorSpace, PixelFormat,
    PlaneStage, PreparedDirectColorPlan, PreparedDirectGrayscalePlan, Surface,
};

mod command_submission;
mod repeated_grayscale;
mod resources;
mod result;
mod validation;

use self::command_submission::submit_stacked_component_commands;
pub(super) use self::repeated_grayscale::{
    encode_repeated_direct_grayscale_plan_in_command_buffer, RepeatedDirectGrayscalePlanRequest,
};
use self::resources::prepare_stacked_component_resources;
pub(super) use self::resources::{
    lookup_direct_band_slice, lookup_direct_band_slice_entry,
    lookup_repeated_direct_band_layout_entry, DirectBandSlice,
};
use self::result::assemble_stacked_component_result;
use self::validation::plan_stacked_component_batch;
pub(super) use self::validation::supports_stacked_direct_component_plane_batch;

#[cfg(target_os = "macos")]
pub(super) fn signed_sample_bias(bit_depth: u8) -> f32 {
    2.0_f32.powi(i32::from(bit_depth) - 1)
}

#[cfg(target_os = "macos")]
pub(super) struct DirectColorPlanRequest<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a CommandBufferRef,
    pub(super) plan: &'a PreparedDirectColorPlan,
    pub(super) fmt: PixelFormat,
    pub(super) tier1_mode: DirectTier1Mode,
    pub(super) stage_timings: &'a mut DirectHybridStageTimings,
    pub(super) retained_buffers: &'a mut Vec<Buffer>,
    pub(super) retained_cpu_coefficients: &'a mut Vec<Vec<f32>>,
    pub(super) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(super) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(super) fn encode_prepared_direct_color_plan_in_command_buffer(
    request: DirectColorPlanRequest<'_>,
) -> Result<Surface, Error> {
    let DirectColorPlanRequest {
        runtime,
        command_buffer,
        plan,
        fmt,
        tier1_mode,
        stage_timings,
        retained_buffers,
        retained_cpu_coefficients,
        status_checks,
        scratch_buffers,
    } = request;
    if plan.component_plans.len() != 3 {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect color execution expected 3 component plans, got {}",
                plan.component_plans.len()
            ),
        });
    }

    let mut planes = Vec::with_capacity(3);
    for component_plan in &plan.component_plans {
        planes.push(encode_prepared_direct_component_plane_in_command_buffer(
            DirectComponentPlaneRequest {
                runtime,
                command_buffer,
                plan: component_plan,
                tier1_mode,
                stage_timings,
                retained_buffers,
                retained_cpu_coefficients,
                status_checks,
                scratch_buffers,
            },
        )?);
    }

    if plan.mct && fmt == PixelFormat::Rgb8 {
        let encode_started = metal_profile_stages_enabled().then(Instant::now);
        let surface = encode_mct_rgb8_to_surface_in_command_buffer(
            runtime,
            command_buffer,
            [&planes[0], &planes[1], &planes[2]],
            plan.dimensions,
            plan.bit_depths,
            plan.transform,
        )?;
        if let Some(started) = encode_started {
            stage_timings.metal_mct_pack_encode += elapsed_us(started);
        }
        return Ok(surface);
    }

    if plan.mct {
        let len = plan.dimensions.0 as usize * plan.dimensions.1 as usize;
        let encode_started = metal_profile_stages_enabled().then(Instant::now);
        status_checks.push(dispatch_inverse_mct_buffers_in_command_buffer(
            runtime,
            command_buffer,
            [&planes[0], &planes[1], &planes[2]],
            len,
            plan.transform,
            [
                signed_sample_bias(plan.bit_depths[0]),
                signed_sample_bias(plan.bit_depths[1]),
                signed_sample_bias(plan.bit_depths[2]),
            ],
        )?);
        if let Some(started) = encode_started {
            stage_timings.metal_mct_pack_encode += elapsed_us(started);
        }
    }

    let stage = PlaneStage {
        dims: plan.dimensions,
        plane_count: 3,
        color_space: NativeColorSpace::RGB,
        has_alpha: false,
        bit_depths: [
            u32::from(plan.bit_depths[0]),
            u32::from(plan.bit_depths[1]),
            u32::from(plan.bit_depths[2]),
            0,
        ],
        planes: [
            Some(planes[0].clone()),
            Some(planes[1].clone()),
            Some(planes[2].clone()),
            None,
        ],
    };
    let encode_started = metal_profile_stages_enabled().then(Instant::now);
    let surface =
        encode_plane_stage_to_surface_in_command_buffer(runtime, command_buffer, &stage, fmt);
    if let Some(started) = encode_started {
        stage_timings.metal_mct_pack_encode += elapsed_us(started);
    }
    surface
}

#[cfg(target_os = "macos")]
pub(super) struct StackedDirectComponentPlane {
    pub(super) buffer: Buffer,
    pub(super) dimensions: (u32, u32),
    pub(super) count: usize,
}

#[cfg(target_os = "macos")]
pub(super) struct StackedDirectColorBatchRequest<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffers: DirectColorBatchCommandBuffers<'a>,
    pub(super) plans: &'a [Arc<PreparedDirectColorPlan>],
    pub(super) tier1_mode: DirectTier1Mode,
    pub(super) force_flattened_cpu_tier1: bool,
    pub(super) stage_timings: &'a mut DirectHybridStageTimings,
    pub(super) retained_buffers: &'a mut Vec<Buffer>,
    pub(super) retained_cpu_coefficients: &'a mut Vec<Vec<f32>>,
    pub(super) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(super) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(super) fn try_encode_stacked_mct_rgb8_direct_color_batch(
    request: StackedDirectColorBatchRequest<'_>,
) -> Result<Option<Vec<Surface>>, Error> {
    let StackedDirectColorBatchRequest {
        runtime,
        command_buffers,
        plans,
        tier1_mode,
        force_flattened_cpu_tier1,
        stage_timings,
        retained_buffers,
        retained_cpu_coefficients,
        status_checks,
        scratch_buffers,
    } = request;
    let Some(first) = plans.first() else {
        return Ok(Some(Vec::new()));
    };
    let repeated_count = repeated_shared_direct_color_plan_count(plans);
    if plans.len() <= 1
        || !first.mct
        || first.component_plans.len() != 3
        || !plans.iter().all(|plan| {
            plan.mct
                && plan.dimensions == first.dimensions
                && plan.bit_depths == first.bit_depths
                && plan.transform == first.transform
                && plan.component_plans.len() == 3
        })
    {
        return Ok(None);
    }
    let execution_plans = if repeated_count.is_some() {
        &plans[..1]
    } else {
        plans
    };

    let flattened_cpu_tier1_cache = if tier1_mode == DirectTier1Mode::CpuUpload
        && (force_flattened_cpu_tier1
            || flattened_hybrid_cpu_tier1_enabled()
            || should_flatten_hybrid_cpu_tier1_color_batch(execution_plans))
    {
        Some(build_flattened_cpu_tier1_cache(
            runtime,
            execution_plans,
            stage_timings,
            retained_buffers,
            retained_cpu_coefficients,
        )?)
    } else {
        None
    };

    let mut stacked_planes = Vec::with_capacity(3);
    for component_idx in 0..3 {
        let component_plan_refs = execution_plans
            .iter()
            .map(|plan| &plan.component_plans[component_idx])
            .collect::<Vec<_>>();
        if !supports_stacked_direct_component_plane_batch(&component_plan_refs) {
            return Ok(None);
        }
        stacked_planes.push(encode_stacked_direct_component_plane_batch(
            StackedDirectComponentPlaneBatchRequest {
                runtime,
                command_buffers,
                plans: &component_plan_refs,
                component_idx,
                flattened_cpu_tier1_cache: flattened_cpu_tier1_cache.as_ref(),
                tier1_mode,
                stage_timings,
                retained_buffers,
                retained_cpu_coefficients,
                status_checks,
                scratch_buffers,
            },
        )?);
    }

    if !stacked_planes
        .iter()
        .all(|plane| plane.dimensions == first.dimensions && plane.count == execution_plans.len())
    {
        return Ok(None);
    }

    let encode_started = metal_profile_stages_enabled().then(Instant::now);
    let mct_plane_buffers = [
        &stacked_planes[0].buffer,
        &stacked_planes[1].buffer,
        &stacked_planes[2].buffer,
    ];
    let surfaces = if let Some(count) = repeated_count {
        encode_repeated_mct_rgb8_to_surfaces_in_command_buffer(
            runtime,
            command_buffers.mct_pack,
            mct_plane_buffers,
            first.dimensions,
            count,
            first.bit_depths,
            first.transform,
        )?
    } else {
        encode_batched_mct_rgb8_to_surfaces_in_command_buffer(
            runtime,
            command_buffers.mct_pack,
            mct_plane_buffers,
            first.dimensions,
            execution_plans.len(),
            first.bit_depths,
            first.transform,
        )?
    };
    if let Some(started) = encode_started {
        stage_timings.metal_mct_pack_encode += elapsed_us(started);
    }
    Ok(Some(surfaces))
}

#[cfg(target_os = "macos")]
pub(super) struct StackedDirectComponentPlaneBatchRequest<'a, 'p> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffers: DirectColorBatchCommandBuffers<'a>,
    pub(super) plans: &'a [&'p PreparedDirectGrayscalePlan],
    pub(super) component_idx: usize,
    pub(super) flattened_cpu_tier1_cache: Option<&'a FlattenedCpuTier1Cache>,
    pub(super) tier1_mode: DirectTier1Mode,
    pub(super) stage_timings: &'a mut DirectHybridStageTimings,
    pub(super) retained_buffers: &'a mut Vec<Buffer>,
    pub(super) retained_cpu_coefficients: &'a mut Vec<Vec<f32>>,
    pub(super) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(super) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(super) fn encode_stacked_direct_component_plane_batch(
    request: StackedDirectComponentPlaneBatchRequest<'_, '_>,
) -> Result<StackedDirectComponentPlane, Error> {
    let tier1_mode = request.tier1_mode;
    let plan = plan_stacked_component_batch(request.plans, tier1_mode)?;
    let mut resources = prepare_stacked_component_resources(plan.count);
    submit_stacked_component_commands(request, &plan, &mut resources)?;
    assemble_stacked_component_result(resources, &plan, tier1_mode)
}
