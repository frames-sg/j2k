// SPDX-License-Identifier: MIT OR Apache-2.0
// j2k-coverage: shared-accelerator-host

use super::super::allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed};
use super::super::{
    encode_forward_dwt, forward_mct, profile, public_packetization_progression_order,
    try_deinterleave_to_f32, try_encode_forward_ict, try_encode_forward_rct,
    validate_deinterleaved_components, BlockCodingMode, EncodeComponentSampleInfo, EncodeOptions,
    EncodeRoiRegion, ForwardDwtRequest, J2kDeinterleaveToF32Job, J2kEncodeStageAccelerator,
    J2kHtj2kTileEncodeJob, J2kResidentEncodeInput, J2kResidentHtj2kTileEncodeJob,
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
    ResidentHtj2kEncodeError, Vec,
};
use super::coefficient_source::{validate_component_sampling_dwt_geometry, OwnedDwtComponent};
use super::ownership::{
    component_planes_retained_bytes, dwt_component_sources_retained_bytes,
    single_tile_plan_retained_bytes,
};
use super::plan::SingleTilePlan;
use super::resident::resident_error_from_encode_error;

pub(super) struct PreparedComponentTransforms {
    pub(super) decompositions: Vec<OwnedDwtComponent>,
    pub(super) deinterleave_us: u128,
    pub(super) mct_us: u128,
    pub(super) dwt_us: u128,
}

pub(super) struct AcceleratedComponentRequest<'a, 'input> {
    pub(super) pixels: &'a [u8],
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) num_components: u16,
    pub(super) bit_depth: u8,
    pub(super) signed: bool,
    pub(super) options: &'a EncodeOptions,
    pub(super) plan: &'a SingleTilePlan,
    pub(super) profile_enabled: bool,
    pub(super) session: &'a NativeEncodeSession<'input>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
pub(super) fn try_encode_complete_ht_tile(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    component_sample_info: &[EncodeComponentSampleInfo],
    roi_regions: &[EncodeRoiRegion],
    plan: &SingleTilePlan,
    profile_enabled: bool,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Option<(Vec<u8>, u128)>> {
    let stage_start = profile::profile_now(profile_enabled);
    if complete_ht_tile_supported(options, plan)
        && component_sample_info.is_empty()
        && plan.roi_component_shifts.iter().all(|shift| *shift == 0)
        && roi_regions.is_empty()
    {
        let input =
            J2kResidentEncodeInput::new(width, height, num_components, bit_depth, signed)
                .map_err(|error| NativeEncodePipelineError::internal_invariant(error.reason()))?;
        let resident_job = resident_htj2k_tile_job(input, options, plan);
        let phase = session.checked_phase(
            single_tile_plan_retained_bytes(plan)?,
            "retained single-tile accelerator plan",
        )?;
        if let Some(tile_data) = accelerator
            .encode_htj2k_tile(J2kHtj2kTileEncodeJob {
                pixels,
                width: resident_job.input.width(),
                height: resident_job.input.height(),
                num_components: resident_job.input.num_components(),
                bit_depth: resident_job.input.bit_depth(),
                signed: resident_job.input.signed(),
                num_decomposition_levels: resident_job.num_decomposition_levels,
                reversible: resident_job.reversible,
                use_mct: resident_job.use_mct,
                guard_bits: resident_job.guard_bits,
                code_block_width: resident_job.code_block_width,
                code_block_height: resident_job.code_block_height,
                progression_order: resident_job.progression_order,
                component_sampling: resident_job.component_sampling,
                quantization_steps: resident_job.quantization_steps,
            })
            .map_err(|source| crate::EncodeError::Accelerator {
                operation: "whole-tile HTJ2K encode",
                source,
            })?
        {
            phase.reconcile_accelerator_vec(&tile_data, "accelerator whole-tile HTJ2K output")?;
            return Ok(Some((tile_data, profile::elapsed_us(stage_start))));
        }
    }

    Ok(None)
}

pub(super) fn encode_complete_resident_ht_tile(
    input: J2kResidentEncodeInput,
    options: &EncodeOptions,
    plan: &SingleTilePlan,
    profile_enabled: bool,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<(Vec<u8>, u128), ResidentHtj2kEncodeError> {
    if !complete_ht_tile_supported(options, plan)
        || plan.roi_component_shifts.iter().any(|shift| *shift != 0)
    {
        return Err(ResidentHtj2kEncodeError::Unsupported(
            "resident HTJ2K encode options require the staged host pipeline",
        ));
    }

    let stage_start = profile::profile_now(profile_enabled);
    let job = resident_htj2k_tile_job(input, options, plan);
    let phase = session
        .checked_phase(
            single_tile_plan_retained_bytes(plan).map_err(resident_error_from_encode_error)?,
            "retained resident single-tile accelerator plan",
        )
        .map_err(resident_error_from_encode_error)?;
    match accelerator
        .encode_resident_htj2k_tile(job)
        .map_err(ResidentHtj2kEncodeError::Accelerator)?
    {
        Some(tile_data) => {
            phase
                .reconcile_accelerator_vec(
                    &tile_data,
                    "resident accelerator whole-tile HTJ2K output",
                )
                .map_err(resident_error_from_encode_error)?;
            Ok((tile_data, profile::elapsed_us(stage_start)))
        }
        None => Err(ResidentHtj2kEncodeError::Declined),
    }
}

fn complete_ht_tile_supported(options: &EncodeOptions, plan: &SingleTilePlan) -> bool {
    plan.params.block_coding_mode == BlockCodingMode::HighThroughput
        && plan.params.num_layers == 1
        && !(plan.params.write_plt
            || plan.params.write_plm
            || plan.params.write_ppm
            || plan.params.write_ppt
            || plan.params.write_sop
            || plan.params.write_eph
            || options.tile_part_packet_limit.is_some())
}

fn resident_htj2k_tile_job<'a>(
    input: J2kResidentEncodeInput,
    options: &EncodeOptions,
    plan: &'a SingleTilePlan,
) -> J2kResidentHtj2kTileEncodeJob<'a> {
    J2kResidentHtj2kTileEncodeJob {
        input,
        num_decomposition_levels: plan.num_levels,
        reversible: options.reversible,
        use_mct: plan.use_mct,
        guard_bits: plan.guard_bits,
        code_block_width: plan.cb_width,
        code_block_height: plan.cb_height,
        progression_order: public_packetization_progression_order(options.progression_order),
        component_sampling: &plan.params.component_sampling,
        quantization_steps: &plan.quant_params,
    }
}

pub(super) fn prepare_accelerated_components(
    request: &AcceleratedComponentRequest<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<PreparedComponentTransforms> {
    let plan_bytes = single_tile_plan_retained_bytes(request.plan)?;
    let (mut components, component_bytes, deinterleave_us) =
        prepare_deinterleaved_components(request, plan_bytes, accelerator)?;
    let mct_us = apply_forward_mct(request, &mut components, accelerator)?;
    let (decompositions, dwt_us) = prepare_dwt_components(
        request,
        components,
        component_bytes,
        plan_bytes,
        accelerator,
    )?;
    Ok(PreparedComponentTransforms {
        decompositions,
        deinterleave_us,
        mct_us,
        dwt_us,
    })
}

fn prepare_deinterleaved_components(
    request: &AcceleratedComponentRequest<'_, '_>,
    plan_bytes: usize,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<(Vec<Vec<f32>>, usize, u128)> {
    let requested_sample_count = request
        .plan
        .num_pixels
        .checked_mul(usize::from(request.num_components))
        .ok_or(crate::EncodeError::ArithmeticOverflow {
            what: "transform component sample count",
        })?;
    let requested_bytes = checked_add_bytes(
        checked_element_bytes::<Vec<f32>>(
            usize::from(request.num_components),
            "transform component plane owners",
        )?,
        checked_element_bytes::<f32>(requested_sample_count, "transform component plane samples")?,
        "transform component plane owners",
    )?;
    request.session.checked_phase(
        checked_add_bytes(plan_bytes, requested_bytes, "deinterleave phase")?,
        "native encode deinterleave phase",
    )?;

    let stage_start = profile::profile_now(request.profile_enabled);
    let components = accelerator
        .encode_deinterleave(J2kDeinterleaveToF32Job {
            pixels: request.pixels,
            num_pixels: request.plan.num_pixels,
            num_components: request.num_components,
            bit_depth: request.bit_depth,
            signed: request.signed,
        })
        .map_err(|source| crate::EncodeError::Accelerator {
            operation: "pixel deinterleave",
            source,
        })?;
    let components = match components {
        Some(components) => validate_deinterleaved_components(
            components,
            request.num_components,
            request.plan.num_pixels,
        )
        .map_err(|detail| crate::EncodeError::Accelerator {
            operation: "pixel deinterleave",
            source: crate::J2kEncodeStageError::internal_invariant(detail),
        })?,
        None => try_deinterleave_to_f32(
            request.pixels,
            request.plan.num_pixels,
            request.num_components,
            request.bit_depth,
            request.signed,
        )?,
    };
    let actual_bytes = component_planes_retained_bytes(&components, components.capacity())?;
    request.session.checked_phase(
        checked_add_bytes(plan_bytes, actual_bytes, "deinterleave output phase")?,
        "native encode deinterleave output",
    )?;
    Ok((components, actual_bytes, profile::elapsed_us(stage_start)))
}

fn apply_forward_mct(
    request: &AcceleratedComponentRequest<'_, '_>,
    components: &mut [Vec<f32>],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<u128> {
    let stage_start = profile::profile_now(request.profile_enabled);
    if request.plan.use_mct {
        if request.options.reversible {
            if !try_encode_forward_rct(components, accelerator).map_err(|source| {
                crate::EncodeError::Accelerator {
                    operation: "forward RCT",
                    source,
                }
            })? {
                forward_mct::forward_rct(components);
            }
        } else if !try_encode_forward_ict(components, accelerator).map_err(|source| {
            crate::EncodeError::Accelerator {
                operation: "forward ICT",
                source,
            }
        })? {
            forward_mct::forward_ict(components);
        }
    }
    Ok(profile::elapsed_us(stage_start))
}

fn prepare_dwt_components(
    request: &AcceleratedComponentRequest<'_, '_>,
    mut components: Vec<Vec<f32>>,
    component_bytes: usize,
    plan_bytes: usize,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<(Vec<OwnedDwtComponent>, u128)> {
    let stage_start = profile::profile_now(request.profile_enabled);
    let mut workspace = try_dwt_workspace(request, components.len(), component_bytes, plan_bytes)?;
    for component_index in 0..components.len() {
        let component = core::mem::take(&mut components[component_index]);
        let remaining_component_bytes =
            component_planes_retained_bytes(&components, components.capacity())?;
        let current_component_bytes = checked_element_bytes::<f32>(
            component.capacity(),
            "transform current component samples",
        )?;
        let prior_dwt_bytes = dwt_component_sources_retained_bytes(
            &workspace.decompositions,
            workspace.decompositions.capacity(),
        )?;
        let live_before_dwt = checked_add_bytes(
            checked_add_bytes(
                plan_bytes,
                checked_add_bytes(
                    remaining_component_bytes,
                    current_component_bytes,
                    "DWT input phase",
                )?,
                "DWT input phase",
            )?,
            checked_add_bytes(prior_dwt_bytes, workspace.scratch_bytes, "DWT input phase")?,
            "DWT input phase",
        )?;
        request
            .session
            .checked_phase(live_before_dwt, "native encode packed DWT phase")?;
        workspace.decompositions.push(encode_forward_dwt(
            ForwardDwtRequest {
                component,
                width: request.width,
                height: request.height,
                num_levels: request.plan.num_levels,
                reversible: request.options.reversible,
                session: request.session,
                retained_base_bytes: live_before_dwt,
                line_scratch: &mut workspace.line_scratch,
            },
            accelerator,
        )?);
        let actual_dwt_bytes = dwt_component_sources_retained_bytes(
            &workspace.decompositions,
            workspace.decompositions.capacity(),
        )?;
        request.session.checked_phase(
            checked_add_bytes(
                checked_add_bytes(plan_bytes, remaining_component_bytes, "DWT output phase")?,
                checked_add_bytes(
                    actual_dwt_bytes,
                    workspace.scratch_bytes,
                    "DWT output phase",
                )?,
                "DWT output phase",
            )?,
            "native encode DWT output phase",
        )?;
    }
    drop(components);
    drop(workspace.line_scratch);
    validate_component_sampling_dwt_geometry(
        &workspace.decompositions,
        request.width,
        request.height,
        &request.plan.params.component_sampling,
    )?;
    Ok((workspace.decompositions, profile::elapsed_us(stage_start)))
}

struct DwtWorkspace {
    decompositions: Vec<OwnedDwtComponent>,
    line_scratch: Vec<f32>,
    scratch_bytes: usize,
}

fn try_dwt_workspace(
    request: &AcceleratedComponentRequest<'_, '_>,
    component_count: usize,
    component_bytes: usize,
    plan_bytes: usize,
) -> NativeEncodePipelineResult<DwtWorkspace> {
    let requested_owner_bytes = checked_element_bytes::<OwnedDwtComponent>(
        component_count,
        "transform DWT component owners",
    )?;
    let scratch_count = usize::try_from(request.width)
        .map_err(|_| crate::EncodeError::ArithmeticOverflow {
            what: "packed DWT scratch width",
        })?
        .max(usize::try_from(request.height).map_err(|_| {
            crate::EncodeError::ArithmeticOverflow {
                what: "packed DWT scratch height",
            }
        })?);
    let requested_scratch_bytes =
        checked_element_bytes::<f32>(scratch_count, "packed DWT line scratch")?;
    request.session.checked_phase(
        checked_add_bytes(
            checked_add_bytes(
                checked_add_bytes(plan_bytes, component_bytes, "DWT owner phase")?,
                requested_owner_bytes,
                "DWT owner phase",
            )?,
            requested_scratch_bytes,
            "DWT owner phase",
        )?,
        "native encode DWT owner phase",
    )?;
    let mut decompositions = Vec::new();
    decompositions
        .try_reserve_exact(component_count)
        .map_err(|_| {
            host_allocation_failed("transform DWT component owners", requested_owner_bytes)
        })?;
    let mut line_scratch = Vec::new();
    line_scratch
        .try_reserve_exact(scratch_count)
        .map_err(|_| host_allocation_failed("packed DWT line scratch", requested_scratch_bytes))?;
    line_scratch.resize(scratch_count, 0.0);
    let scratch_bytes =
        checked_element_bytes::<f32>(line_scratch.capacity(), "packed DWT line scratch")?;
    request.session.checked_phase(
        checked_add_bytes(
            checked_add_bytes(plan_bytes, component_bytes, "DWT scratch phase")?,
            checked_add_bytes(
                checked_element_bytes::<OwnedDwtComponent>(
                    decompositions.capacity(),
                    "transform DWT component owners",
                )?,
                scratch_bytes,
                "DWT scratch phase",
            )?,
            "DWT scratch phase",
        )?,
        "native encode DWT scratch phase",
    )?;
    Ok(DwtWorkspace {
        decompositions,
        line_scratch,
        scratch_bytes,
    })
}

#[cfg(test)]
#[path = "accelerator/tests.rs"]
mod tests;
