// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    borrow_mut_slice_buffer, commit_and_wait_metal, completed_command_buffer_gpu_duration,
    decode_prepared_classic_sub_band_group_on_cpu_profile,
    decode_prepared_classic_sub_band_on_cpu_profile,
    decode_prepared_ht_sub_band_group_on_cpu_profile, decode_prepared_ht_sub_band_on_cpu_profile,
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_store_component_buffer_in_encoder_with_offsets, elapsed_us,
    emit_direct_hybrid_stage_timings, encode_gray_plane_to_surface_in_encoder,
    encode_gray_store_to_surface_in_encoder,
    encode_prepared_classic_sub_band_group_to_buffer_in_encoder,
    encode_prepared_classic_sub_band_to_buffer_in_encoder,
    encode_prepared_direct_color_plan_in_command_buffer,
    encode_prepared_ht_sub_band_group_to_buffer_in_encoder,
    encode_prepared_ht_sub_band_to_buffer_in_encoder,
    encode_repeated_direct_grayscale_plan_in_command_buffer,
    encode_repeated_gray_plane_to_surfaces_in_command_buffer,
    encode_stacked_direct_component_plane_batch, hybrid_stage_signpost,
    idwt_input_windows_from_slices, j2k_scalar_pack_params, label_command_buffer,
    lookup_direct_band_slice, lookup_direct_band_slice_entry,
    metal_profile_decode_split_commands_enabled, metal_profile_stages_enabled,
    prepared_direct_color_plan_supports_runtime, prepared_idwt_output_len, prepared_idwt_params,
    record_completed_decode_split_gpu_stages, recycle_scratch_buffers, size_of,
    supports_stacked_direct_component_plane_batch, take_f32_scratch_buffer,
    try_encode_stacked_mct_rgb8_direct_color_batch, validate_direct_status,
    wait_for_completion_metal, with_runtime, with_runtime_for_device, Arc, BandRequiredRegion,
    Buffer, CommandBufferRef, CpuTier1DecodeSubstageCounters, DecodeHybridSplitCommandBuffers,
    Device, DirectBandSlice, DirectColorBatchCommandBuffers, DirectColorPlanRequest,
    DirectHybridStageTimings, DirectScratchBuffer, DirectStatusCheck, DirectTier1Mode, Error,
    IdwtSubBandBuffers, Instant, J2kGrayStoreParams, J2kStoreParams, J2kWaveletTransform,
    MetalRuntime, PixelFormat, PreparedDirectColorPlan, PreparedDirectGrayscalePlan,
    PreparedDirectGrayscaleStep, RepeatedDirectGrayscalePlanRequest, SingleIdwtDispatch,
    StackedDirectColorBatchRequest, StackedDirectComponentPlaneBatchRequest, Surface,
    SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD, SIGNPOST_DECODE_HYBRID_COMMAND_WAIT,
};

mod component_plane;
mod single;

pub(in crate::compute) use self::component_plane::*;
pub(in crate::compute) use self::single::*;

#[cfg(target_os = "macos")]
pub(crate) fn execute_repeated_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surfaces = encode_repeated_direct_grayscale_plan_in_command_buffer(
            RepeatedDirectGrayscalePlanRequest {
                runtime,
                command_buffer,
                plan,
                fmt,
                count,
                retained_buffers: &mut retained_buffers,
                status_checks: &mut status_checks,
                scratch_buffers: &mut scratch_buffers,
            },
        )?;
        commit_and_wait_metal(command_buffer)?;
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surface = encode_prepared_direct_grayscale_plan_in_command_buffer(
            runtime,
            command_buffer,
            plan,
            fmt,
            &mut retained_buffers,
            &mut status_checks,
            &mut scratch_buffers,
        )?;
        commit_and_wait_metal(command_buffer)?;
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surface)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_with_device(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_prepared_direct_grayscale_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_batch(
    plans: &[Arc<PreparedDirectGrayscalePlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }

    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut retained_cpu_coefficients = Vec::<Vec<f32>>::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let mut stage_timings = DirectHybridStageTimings::default();
        let mut surfaces = Vec::with_capacity(plans.len());

        let component_plan_refs = plans.iter().map(Arc::as_ref).collect::<Vec<_>>();
        if plans.len() > 1 && supports_stacked_direct_component_plane_batch(&component_plan_refs) {
            let stacked_plane = encode_stacked_direct_component_plane_batch(
                StackedDirectComponentPlaneBatchRequest {
                    runtime,
                    command_buffers: DirectColorBatchCommandBuffers::single(command_buffer),
                    plans: &component_plan_refs,
                    component_idx: 0,
                    flattened_cpu_tier1_cache: None,
                    tier1_mode: DirectTier1Mode::Metal,
                    stage_timings: &mut stage_timings,
                    retained_buffers: &mut retained_buffers,
                    retained_cpu_coefficients: &mut retained_cpu_coefficients,
                    status_checks: &mut status_checks,
                    scratch_buffers: &mut scratch_buffers,
                },
            )?;
            let first = plans.first().expect("plans is not empty");
            if stacked_plane.dimensions == first.dimensions && stacked_plane.count == plans.len() {
                surfaces = encode_repeated_gray_plane_to_surfaces_in_command_buffer(
                    runtime,
                    command_buffer,
                    &stacked_plane.buffer,
                    first.dimensions,
                    first.bit_depth,
                    fmt,
                    plans.len(),
                )?;
            }
        }

        for plan in plans {
            if !surfaces.is_empty() {
                break;
            }
            surfaces.push(encode_prepared_direct_grayscale_plan_in_command_buffer(
                runtime,
                command_buffer,
                plan,
                fmt,
                &mut retained_buffers,
                &mut status_checks,
                &mut scratch_buffers,
            )?);
        }

        commit_and_wait_metal(command_buffer)?;
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        drop(retained_cpu_coefficients);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [Arc::new(plan.clone())];
    let mut surfaces = execute_prepared_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_with_device(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| execute_prepared_direct_color_plan(plan, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1(plans, fmt, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [Arc::new(plan.clone())];
    let mut surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect hybrid color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan_with_device(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_hybrid_cpu_tier1_direct_color_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1(plans, fmt, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
pub(super) fn execute_direct_color_plan_batch_with_tier1(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1_options(plans, fmt, tier1_mode, false)
}

#[cfg(all(target_os = "macos", test))]
pub(super) fn execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1_options(plans, fmt, DirectTier1Mode::CpuUpload, true)
}

#[cfg(target_os = "macos")]
pub(super) fn execute_direct_color_plan_batch_with_tier1_options(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
    force_flattened_cpu_tier1: bool,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }
    if tier1_mode == DirectTier1Mode::Metal
        && plans
            .iter()
            .any(|plan| !prepared_direct_color_plan_supports_runtime(plan, fmt))
    {
        return Err(Error::MetalDirectFallback {
            message: "unsupported classic kernel input in direct component plan".to_string(),
            reason: crate::MetalDirectFallbackReason::UnsupportedRuntimeInput,
        });
    }

    with_runtime(|runtime| {
        let mut retained_buffers = Vec::new();
        let mut retained_cpu_coefficients = Vec::<Vec<f32>>::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let mut stage_timings = DirectHybridStageTimings::default();
        let profile_hybrid_stages =
            tier1_mode == DirectTier1Mode::CpuUpload && metal_profile_stages_enabled();

        if fmt == PixelFormat::Rgb8
            && profile_hybrid_stages
            && metal_profile_decode_split_commands_enabled()
        {
            let split_command_buffers = DecodeHybridSplitCommandBuffers::new(runtime);
            if let Some(surfaces) =
                try_encode_stacked_mct_rgb8_direct_color_batch(StackedDirectColorBatchRequest {
                    runtime,
                    command_buffers: split_command_buffers.refs(),
                    plans,
                    tier1_mode,
                    force_flattened_cpu_tier1,
                    stage_timings: &mut stage_timings,
                    retained_buffers: &mut retained_buffers,
                    retained_cpu_coefficients: &mut retained_cpu_coefficients,
                    status_checks: &mut status_checks,
                    scratch_buffers: &mut scratch_buffers,
                })?
            {
                split_command_buffers.commit_in_order();
                let wait_started = Instant::now();
                let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
                wait_for_completion_metal(&split_command_buffers.mct_pack)?;
                stage_timings.command_wait += elapsed_us(wait_started);
                record_completed_decode_split_gpu_stages(
                    &mut stage_timings,
                    &split_command_buffers,
                );
                for status_check in status_checks {
                    validate_direct_status(status_check)?;
                }
                emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
                drop(retained_buffers);
                drop(retained_cpu_coefficients);
                recycle_scratch_buffers(runtime, scratch_buffers)?;
                return Ok(surfaces);
            }

            drop(split_command_buffers);
            retained_buffers.clear();
            retained_cpu_coefficients.clear();
            status_checks.clear();
            scratch_buffers.clear();
            stage_timings = DirectHybridStageTimings::default();
        }

        let command_buffer = runtime.queue.new_command_buffer();
        if profile_hybrid_stages {
            label_command_buffer(command_buffer, "j2k decode hybrid direct color batch");
        }

        if fmt == PixelFormat::Rgb8 {
            if let Some(surfaces) =
                try_encode_stacked_mct_rgb8_direct_color_batch(StackedDirectColorBatchRequest {
                    runtime,
                    command_buffers: DirectColorBatchCommandBuffers::single(command_buffer),
                    plans,
                    tier1_mode,
                    force_flattened_cpu_tier1,
                    stage_timings: &mut stage_timings,
                    retained_buffers: &mut retained_buffers,
                    retained_cpu_coefficients: &mut retained_cpu_coefficients,
                    status_checks: &mut status_checks,
                    scratch_buffers: &mut scratch_buffers,
                })?
            {
                command_buffer.commit();
                let wait_started = profile_hybrid_stages.then(Instant::now);
                let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
                wait_for_completion_metal(command_buffer)?;
                if let Some(started) = wait_started {
                    stage_timings.command_wait += elapsed_us(started);
                }
                if profile_hybrid_stages {
                    if let Some(duration) = completed_command_buffer_gpu_duration(command_buffer) {
                        stage_timings.gpu_command += duration.as_micros();
                    }
                }
                for status_check in status_checks {
                    validate_direct_status(status_check)?;
                }
                if tier1_mode == DirectTier1Mode::CpuUpload {
                    emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
                }
                drop(retained_buffers);
                drop(retained_cpu_coefficients);
                recycle_scratch_buffers(runtime, scratch_buffers)?;
                return Ok(surfaces);
            }
        }

        let mut surfaces = Vec::with_capacity(plans.len());

        for plan in plans {
            let surface =
                encode_prepared_direct_color_plan_in_command_buffer(DirectColorPlanRequest {
                    runtime,
                    command_buffer,
                    plan,
                    fmt,
                    tier1_mode,
                    stage_timings: &mut stage_timings,
                    retained_buffers: &mut retained_buffers,
                    retained_cpu_coefficients: &mut retained_cpu_coefficients,
                    status_checks: &mut status_checks,
                    scratch_buffers: &mut scratch_buffers,
                })?;
            surfaces.push(surface);
        }

        command_buffer.commit();
        let wait_started = profile_hybrid_stages.then(Instant::now);
        let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
        wait_for_completion_metal(command_buffer)?;
        if let Some(started) = wait_started {
            stage_timings.command_wait += elapsed_us(started);
        }
        if profile_hybrid_stages {
            if let Some(duration) = completed_command_buffer_gpu_duration(command_buffer) {
                stage_timings.gpu_command += duration.as_micros();
            }
        }
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        if tier1_mode == DirectTier1Mode::CpuUpload {
            emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
        }
        drop(retained_buffers);
        drop(retained_cpu_coefficients);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surfaces)
    })
}
