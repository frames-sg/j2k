// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use metal::{Buffer, CommandBufferRef, ComputeCommandEncoderRef};

use super::decode_dispatch::mct::dispatch_inverse_mct_buffers_in_command_buffer;
use super::direct_grayscale_execute::{
    checked_coefficient_len, encode_prepared_direct_component_plane_in_command_buffer,
    DirectComponentPlaneRequest,
};
use super::{
    build_flattened_cpu_tier1_cache, elapsed_us,
    encode_batched_mct_rgb8_to_surfaces_in_command_buffer,
    encode_mct_rgb8_to_surface_in_command_buffer, encode_plane_stage_to_surface_in_command_buffer,
    encode_repeated_mct_rgb8_to_surfaces_in_command_buffer, flattened_hybrid_cpu_tier1_enabled,
    metal_profile_stages_enabled, record_hybrid_stacked_component_batch,
    record_stacked_component_batch, should_flatten_hybrid_cpu_tier1_color_batch,
    DirectColorBatchCommandBuffers, DirectHybridStageTimings, DirectScratchBuffer,
    DirectStatusCheck, DirectTier1Mode, Error, FlattenedCpuTier1Cache, MetalRuntime,
    NativeColorSpace, PixelFormat, PlaneStage, PreparedDirectColorPlan,
    PreparedDirectGrayscalePlan, Surface,
};

mod command_submission;
mod repeated_grayscale;
mod resources;
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
pub(super) use self::validation::supports_stacked_direct_component_plane_batch;
use self::validation::{
    plan_stacked_component_batch, preflight_stacked_mct_rgb8_color_batch,
    StackedColorBatchPreflight,
};

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
        let len = checked_coefficient_len(
            plan.dimensions.0,
            plan.dimensions.1,
            "J2K MetalDirect color MCT plane span overflow",
        )?;
        let encode_started = metal_profile_stages_enabled().then(Instant::now);
        dispatch_inverse_mct_buffers_in_command_buffer(
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
        )?;
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
        status_checks,
        scratch_buffers,
    } = request;
    if plans.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let Some(preflight) = preflight_stacked_mct_rgb8_color_batch(plans)? else {
        return Ok(None);
    };

    let flattened_cpu_tier1_cache = if tier1_mode == DirectTier1Mode::CpuUpload
        && (force_flattened_cpu_tier1
            || flattened_hybrid_cpu_tier1_enabled()
            || should_flatten_hybrid_cpu_tier1_color_batch(preflight.execution_plans))
    {
        Some(build_flattened_cpu_tier1_cache(
            runtime,
            preflight.execution_plans,
            stage_timings,
            retained_buffers,
        )?)
    } else {
        None
    };

    let mut stacked_planes = Vec::with_capacity(3);
    for (component_idx, component_plan_refs) in preflight.component_plan_refs.iter().enumerate() {
        let plane =
            encode_stacked_direct_component_plane_batch(StackedDirectComponentPlaneBatchRequest {
                runtime,
                command_buffers,
                compute_encoder: None,
                plans: component_plan_refs,
                component_idx,
                flattened_cpu_tier1_cache: flattened_cpu_tier1_cache.as_ref(),
                tier1_mode,
                stage_timings,
                retained_buffers,
                status_checks,
                scratch_buffers,
            })?;
        if plane.dimensions != preflight.first.dimensions
            || plane.count != preflight.execution_plans.len()
        {
            return Err(Error::MetalStateInvariant {
                state: "J2K Metal stacked color execution",
                reason: "encoded component plane diverged from the completed preflight",
            });
        }
        stacked_planes.push(plane);
    }

    encode_stacked_mct_rgb8_surfaces(
        runtime,
        command_buffers,
        &preflight,
        &stacked_planes,
        stage_timings,
    )
    .map(Some)
}

fn encode_stacked_mct_rgb8_surfaces(
    runtime: &MetalRuntime,
    command_buffers: DirectColorBatchCommandBuffers<'_>,
    preflight: &StackedColorBatchPreflight<'_>,
    stacked_planes: &[StackedDirectComponentPlane],
    stage_timings: &mut DirectHybridStageTimings,
) -> Result<Vec<Surface>, Error> {
    let [plane0, plane1, plane2] = stacked_planes else {
        return Err(Error::MetalStateInvariant {
            state: "J2K Metal stacked color execution",
            reason: "completed preflight did not produce exactly three component planes",
        });
    };
    let encode_started = metal_profile_stages_enabled().then(Instant::now);
    let mct_plane_buffers = [&plane0.buffer, &plane1.buffer, &plane2.buffer];
    let surfaces = if let Some(count) = preflight.repeated_count {
        encode_repeated_mct_rgb8_to_surfaces_in_command_buffer(
            runtime,
            command_buffers.mct_pack,
            mct_plane_buffers,
            preflight.first.dimensions,
            count,
            preflight.first.bit_depths,
            preflight.first.transform,
        )?
    } else {
        encode_batched_mct_rgb8_to_surfaces_in_command_buffer(
            runtime,
            command_buffers.mct_pack,
            mct_plane_buffers,
            preflight.first.dimensions,
            preflight.execution_plans.len(),
            preflight.first.bit_depths,
            preflight.first.transform,
        )?
    };
    if let Some(started) = encode_started {
        stage_timings.metal_mct_pack_encode += elapsed_us(started);
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
pub(super) struct StackedDirectComponentPlaneBatchRequest<'a, 'p> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffers: DirectColorBatchCommandBuffers<'a>,
    pub(super) compute_encoder: Option<&'a ComputeCommandEncoderRef>,
    pub(super) plans: &'a [&'p PreparedDirectGrayscalePlan],
    pub(super) component_idx: usize,
    pub(super) flattened_cpu_tier1_cache: Option<&'a FlattenedCpuTier1Cache>,
    pub(super) tier1_mode: DirectTier1Mode,
    pub(super) stage_timings: &'a mut DirectHybridStageTimings,
    pub(super) retained_buffers: &'a mut Vec<Buffer>,
    pub(super) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(super) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(super) fn encode_stacked_direct_component_plane_batch(
    request: StackedDirectComponentPlaneBatchRequest<'_, '_>,
) -> Result<StackedDirectComponentPlane, Error> {
    let tier1_mode = request.tier1_mode;
    let plan = plan_stacked_component_batch(request.plans, tier1_mode)?;
    let mut resources = prepare_stacked_component_resources(plan.count, plan.first.steps.len())?;
    submit_stacked_component_commands(request, &plan, &mut resources)?;
    let final_plane = resources.final_plane.ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color component batch did not produce a final plane".to_string(),
    })?;
    record_stacked_component_batch();
    record_hybrid_stacked_component_batch(tier1_mode);
    Ok(StackedDirectComponentPlane {
        buffer: final_plane.buffer,
        dimensions: plan.first.dimensions,
        count: plan.count,
    })
}
