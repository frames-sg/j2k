// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;
#[cfg(target_os = "macos")]
use std::sync::Arc;
use std::time::Instant;

use crate::metal_types::{Buffer, CommandBufferRef, ComputeCommandEncoderRef};

use super::abi::J2kBandCells;
use super::decode_dispatch::mct::dispatch_inverse_mct_buffers_in_command_buffer;
use super::decode_dispatch::{
    fused_idwt_mct_rgb8_supported, FusedIdwtStoreSource, IdwtSubBandBuffers,
};
use super::direct_commands::new_compute_command_encoder;
use super::direct_grayscale_execute::{
    checked_coefficient_len, encode_prepared_direct_component_plane_before_final_level,
    encode_prepared_direct_component_plane_in_command_buffer, final_identity_stored_level,
    DirectComponentPlaneRequest, FinalLevelInputs,
};
use super::{
    build_flattened_cpu_tier1_cache, elapsed_us,
    encode_batched_mct_rgb8_to_surfaces_in_command_buffer,
    encode_fused_idwt_mct_rgb8_to_surface_in_command_buffer,
    encode_mct_rgb8_to_surface_in_command_buffer, encode_plane_stage_to_surface_in_command_buffer,
    encode_repeated_mct_rgb8_to_surfaces_in_command_buffer, flattened_hybrid_cpu_tier1_enabled,
    metal_profile_stages_enabled, record_hybrid_stacked_component_batch,
    record_stacked_component_batch, should_flatten_hybrid_cpu_tier1_color_batch,
    DirectColorBatchCommandBuffers, DirectHybridStageTimings, DirectScratchBuffer,
    DirectStatusCheck, DirectTier1Mode, Error, FlattenedCpuTier1Cache, MetalRuntime,
    NativeColorSpace, PixelFormat, PlaneStage, PreparedDirectColorPlan,
    PreparedDirectGrayscalePlan, SharedClassicTier1Group, Surface,
};
use j2k_native::J2kWaveletTransform;

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
    mut request: DirectColorPlanRequest<'_>,
) -> Result<Surface, Error> {
    if let Some(surface) = try_encode_fused_mct_rgb8_color_plan(&mut request)? {
        return Ok(surface);
    }
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

    let mut plane_budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K MetalDirect component planes");
    let mut planes = plane_budget.try_vec(3, "J2K MetalDirect component plane handles")?;
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

/// Encodes an MCT RGB8 `request` whose components each end in a last wavelet
/// level and an identity store of the whole plane: the steps before each last
/// level, then one kernel that synthesizes the three last levels and writes
/// RGB8, so the float planes are never written. `None` encodes nothing; a
/// plan without three components is an error on either route.
#[cfg(target_os = "macos")]
fn try_encode_fused_mct_rgb8_color_plan(
    request: &mut DirectColorPlanRequest<'_>,
) -> Result<Option<Surface>, Error> {
    let (runtime, command_buffer, plan) = (request.runtime, request.command_buffer, request.plan);
    if plan.component_plans.len() != 3 {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect color execution expected 3 component plans, got {}",
                plan.component_plans.len()
            ),
        });
    }
    if !plan.mct || request.fmt != PixelFormat::Rgb8 {
        return Ok(None);
    }
    let Some(finals) = plan
        .component_plans
        .iter()
        .map(|component| final_identity_stored_level(component, plan.dimensions))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let transform = finals[0].step.transform;
    if !finals.iter().all(|idwt| idwt.step.transform == transform)
        || !fused_idwt_mct_rgb8_supported(runtime.decode()?, transform, plan.dimensions)
    {
        return Ok(None);
    }
    let mut inputs =
        crate::batch_allocation::BatchMetadataBudget::new("J2K MetalDirect fused colour levels")
            .try_vec(3, "J2K MetalDirect fused colour level inputs")?;
    for (component_plan, idwt) in plan.component_plans.iter().zip(&finals) {
        let encoder = new_compute_command_encoder(command_buffer)?;
        let level = encode_prepared_direct_component_plane_before_final_level(
            DirectComponentPlaneRequest {
                runtime,
                command_buffer,
                plan: component_plan,
                tier1_mode: request.tier1_mode,
                stage_timings: &mut *request.stage_timings,
                retained_buffers: &mut *request.retained_buffers,
                status_checks: &mut *request.status_checks,
                scratch_buffers: &mut *request.scratch_buffers,
            },
            &encoder,
            idwt,
        );
        encoder.endEncoding();
        inputs.push(level?);
    }
    let cells = fused_band_cells(runtime, &inputs)?;
    if let Some((buffer, _)) = &cells {
        request.retained_buffers.push(buffer.clone());
    }
    encode_fused_idwt_mct_rgb8_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &[
            fused_level_source(&inputs[0], transform),
            fused_level_source(&inputs[1], transform),
            fused_level_source(&inputs[2], transform),
        ],
        cells.as_ref().map(|(buffer, bands)| (buffer, bands)),
        plan.dimensions,
        plan.bit_depths,
        plan.transform,
    )
    .map(Some)
}

/// One buffer holding the cell maps of the three components' high-pass
/// sub-bands, HL, LH, HH per component, after a leading covered cell that a
/// sub-band zeroed as usual reads for every sample; `None` when every
/// sub-band was zeroed as usual.
#[cfg(target_os = "macos")]
fn fused_band_cells(
    runtime: &MetalRuntime,
    inputs: &[FinalLevelInputs],
) -> Result<Option<(Buffer, [J2kBandCells; 9])>, Error> {
    if inputs
        .iter()
        .all(|(_, _, cells)| cells.iter().all(Option::is_none))
    {
        return Ok(None);
    }
    let dense = J2kBandCells {
        offset: 0,
        row_cells: 1,
        shift: 31,
        reserved: 0,
    };
    let mut bands = [dense; 9];
    let len = 1 + inputs
        .iter()
        .flat_map(|(_, _, cells)| cells.iter().flatten())
        .map(|cells| cells.covered.len())
        .sum::<usize>();
    let mut bytes = crate::batch_allocation::BatchMetadataBudget::new("J2K MetalDirect cell maps")
        .try_vec(len, "J2K MetalDirect sub-band cell maps")?;
    bytes.push(1u8);
    for (band, cells) in bands
        .iter_mut()
        .zip(inputs.iter().flat_map(|(_, _, cells)| cells))
    {
        if let Some(cells) = cells {
            *band = J2kBandCells {
                offset: u32::try_from(bytes.len()).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal sub-band cell map exceeds u32".to_string(),
                })?,
                row_cells: cells.row_cells,
                shift: cells.shift,
                reserved: 0,
            };
            bytes.extend_from_slice(&cells.covered);
        }
    }
    let buffer = super::HtInputAllocator::new(&runtime.device, &runtime.ht_input_heap)
        .copied_slice(&bytes)?;
    Ok(Some((buffer, bands)))
}

#[cfg(target_os = "macos")]
fn fused_level_source(
    (bands, params, _): &FinalLevelInputs,
    transform: J2kWaveletTransform,
) -> FusedIdwtStoreSource<'_> {
    let [ll, hl, lh, hh] = bands;
    FusedIdwtStoreSource {
        sub_bands: IdwtSubBandBuffers {
            ll: &ll.buffer,
            ll_offset: ll.offset_bytes,
            hl: &hl.buffer,
            hl_offset: hl.offset_bytes,
            lh: &lh.buffer,
            lh_offset: lh.offset_bytes,
            hh: &hh.buffer,
            hh_offset: hh.offset_bytes,
        },
        params: *params,
        transform,
    }
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
            None,
        )?)
    } else {
        None
    };

    let mut plane_budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect stacked component planes",
    );
    let mut stacked_planes =
        plane_budget.try_vec(3, "J2K MetalDirect stacked component plane handles")?;
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

/// A dense grayscale group that a stacked batch's last wavelet level may
/// write directly, with the group pack's sample conversion.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct StackedGrayOutput<'a> {
    pub(super) buffer: &'a crate::metal_types::BufferRef,
    pub(super) offset_bytes: usize,
    pub(super) kind: super::decode_dispatch::store::GrayStoreKind,
    pub(super) max_value: f32,
    pub(super) u8_scale: f32,
    pub(super) u16_scale: f32,
}

/// The result of [`encode_stacked_direct_gray_batch`].
#[cfg(target_os = "macos")]
pub(super) enum StackedGrayBatch {
    /// The batch produced its final plane for the caller to pack.
    Plane(StackedDirectComponentPlane),
    /// The last wavelet level already wrote the requested output.
    Stored,
}

#[cfg(target_os = "macos")]
pub(super) fn encode_stacked_direct_component_plane_batch(
    request: StackedDirectComponentPlaneBatchRequest<'_, '_>,
) -> Result<StackedDirectComponentPlane, Error> {
    encode_stacked_component_plane(request, None)
}

/// Like [`encode_stacked_direct_component_plane_batch`], but lets the last
/// wavelet level store a grayscale group straight into `output` when the
/// plan allows it.
#[cfg(target_os = "macos")]
pub(super) fn encode_stacked_direct_gray_batch(
    request: StackedDirectComponentPlaneBatchRequest<'_, '_>,
    output: StackedGrayOutput<'_>,
) -> Result<StackedGrayBatch, Error> {
    encode_stacked_component(request, None, Some(output))
}

/// Like [`encode_stacked_direct_component_plane_batch`], but reads classic
/// coefficients that a dispatch shared with other groups already decoded.
#[cfg(target_os = "macos")]
pub(super) fn encode_stacked_direct_component_plane_batch_with_shared_tier1(
    request: StackedDirectComponentPlaneBatchRequest<'_, '_>,
    shared_tier1: &mut SharedClassicTier1Group,
) -> Result<StackedDirectComponentPlane, Error> {
    if request.tier1_mode != DirectTier1Mode::Metal || request.flattened_cpu_tier1_cache.is_some() {
        return Err(Error::MetalStateInvariant {
            state: "J2K Metal shared classic Tier-1",
            reason: "shared coefficients replace a Metal Tier-1 dispatch only",
        });
    }
    encode_stacked_component_plane(request, Some(shared_tier1))
}

#[cfg(target_os = "macos")]
fn encode_stacked_component_plane(
    request: StackedDirectComponentPlaneBatchRequest<'_, '_>,
    shared_tier1: Option<&mut SharedClassicTier1Group>,
) -> Result<StackedDirectComponentPlane, Error> {
    match encode_stacked_component(request, shared_tier1, None)? {
        StackedGrayBatch::Plane(plane) => Ok(plane),
        StackedGrayBatch::Stored => Err(Error::MetalStateInvariant {
            state: "J2K MetalDirect stacked component batch",
            reason: "a batch without a grayscale output stored one",
        }),
    }
}

#[cfg(target_os = "macos")]
fn encode_stacked_component(
    request: StackedDirectComponentPlaneBatchRequest<'_, '_>,
    shared_tier1: Option<&mut SharedClassicTier1Group>,
    gray_output: Option<StackedGrayOutput<'_>>,
) -> Result<StackedGrayBatch, Error> {
    let tier1_mode = request.tier1_mode;
    let plan = plan_stacked_component_batch(request.plans, tier1_mode)?;
    let mut resources = prepare_stacked_component_resources(plan.count, plan.first.steps.len())?;
    submit_stacked_component_commands(request, shared_tier1, gray_output, &plan, &mut resources)?;
    record_stacked_component_batch();
    record_hybrid_stacked_component_batch(tier1_mode);
    if resources.gray_output_written {
        return Ok(StackedGrayBatch::Stored);
    }
    let final_plane = resources.final_plane.ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color component batch did not produce a final plane".to_string(),
    })?;
    Ok(StackedGrayBatch::Plane(StackedDirectComponentPlane {
        buffer: final_plane.buffer,
        dimensions: plan.first.dimensions,
        count: plan.count,
    }))
}
