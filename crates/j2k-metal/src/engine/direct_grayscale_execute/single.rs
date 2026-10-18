// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use j2k_metal_support::MetalImageDestination;

use super::{
    new_compute_command_encoder, Buffer, CommandBufferRef, DirectScratchBuffer, DirectStatusCheck,
    Error, MetalRuntime, PixelFormat, PreparedDirectGrayscalePlan, Surface,
};

mod execution;

use self::execution::SingleGrayscaleExecution;

#[cfg(target_os = "macos")]
struct GrayscalePlanExecutionRequest<'a> {
    runtime: &'a MetalRuntime,
    command_buffer: &'a CommandBufferRef,
    plan: &'a PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    destination: Option<(&'a MetalImageDestination, usize)>,
    native_destination: bool,
    retained_buffers: &'a mut Vec<Buffer>,
    status_checks: &'a mut Vec<DirectStatusCheck>,
    scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(in crate::engine) struct DirectGrayscaleDestinationExecutionRequest<'a> {
    pub(in crate::engine) runtime: &'a MetalRuntime,
    pub(in crate::engine) command_buffer: &'a CommandBufferRef,
    pub(in crate::engine) plan: &'a PreparedDirectGrayscalePlan,
    pub(in crate::engine) fmt: PixelFormat,
    pub(in crate::engine) destination: &'a MetalImageDestination,
    pub(in crate::engine) destination_item_index: usize,
    pub(in crate::engine) retained_buffers: &'a mut Vec<Buffer>,
    pub(in crate::engine) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(in crate::engine) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn encode_prepared_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Surface, Error> {
    encode_prepared_direct_grayscale_plan_in_command_buffer_inner(
        GrayscalePlanExecutionRequest {
            runtime,
            command_buffer,
            plan,
            fmt,
            destination: None,
            native_destination: false,
            retained_buffers,
            status_checks,
            scratch_buffers,
        },
        None,
    )?
    .ok_or_else(|| Error::MetalStateInvariant {
        state: "J2K Metal direct grayscale execution",
        reason: "surface execution completed without a surface",
    })
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn encode_prepared_direct_grayscale_plan_into_in_encoder(
    request: DirectGrayscaleDestinationExecutionRequest<'_>,
    encoder: &crate::metal_types::ComputeCommandEncoderRef,
) -> Result<(), Error> {
    let DirectGrayscaleDestinationExecutionRequest {
        runtime,
        command_buffer,
        plan,
        fmt,
        destination,
        destination_item_index,
        retained_buffers,
        status_checks,
        scratch_buffers,
    } = request;
    let surface = encode_prepared_direct_grayscale_plan_in_command_buffer_inner(
        GrayscalePlanExecutionRequest {
            runtime,
            command_buffer,
            plan,
            fmt,
            destination: Some((destination, destination_item_index)),
            native_destination: true,
            retained_buffers,
            status_checks,
            scratch_buffers,
        },
        Some(encoder),
    )?;
    if surface.is_some() {
        return Err(Error::MetalStateInvariant {
            state: "J2K Metal direct grayscale destination execution",
            reason: "destination execution unexpectedly allocated a surface",
        });
    }
    Ok(())
}

/// Encode one plan whose final store writes the request's destination with
/// the same format conversion as a surface decode, in its own compute encoder.
#[cfg(target_os = "macos")]
pub(in crate::engine) fn encode_prepared_direct_grayscale_plan_as_surface_into_command_buffer(
    request: DirectGrayscaleDestinationExecutionRequest<'_>,
) -> Result<(), Error> {
    let DirectGrayscaleDestinationExecutionRequest {
        runtime,
        command_buffer,
        plan,
        fmt,
        destination,
        destination_item_index,
        retained_buffers,
        status_checks,
        scratch_buffers,
    } = request;
    let surface = encode_prepared_direct_grayscale_plan_in_command_buffer_inner(
        GrayscalePlanExecutionRequest {
            runtime,
            command_buffer,
            plan,
            fmt,
            destination: Some((destination, destination_item_index)),
            native_destination: false,
            retained_buffers,
            status_checks,
            scratch_buffers,
        },
        None,
    )?;
    if surface.is_some() {
        return Err(Error::MetalStateInvariant {
            state: "J2K Metal direct grayscale destination execution",
            reason: "destination execution unexpectedly allocated a surface",
        });
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_prepared_direct_grayscale_plan_in_command_buffer_inner(
    request: GrayscalePlanExecutionRequest<'_>,
    existing_encoder: Option<&crate::metal_types::ComputeCommandEncoderRef>,
) -> Result<Option<Surface>, Error> {
    let GrayscalePlanExecutionRequest {
        runtime,
        command_buffer,
        plan,
        fmt,
        destination,
        native_destination,
        retained_buffers,
        status_checks,
        scratch_buffers,
    } = request;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect grayscale band metadata",
    );
    let bands = budget.try_vec(plan.steps.len(), "J2K MetalDirect grayscale band metadata")?;
    let owned_encoder = if existing_encoder.is_none() {
        Some(new_compute_command_encoder(command_buffer)?)
    } else {
        None
    };
    let Some(encoder) = existing_encoder.or(owned_encoder.as_deref()) else {
        return Err(Error::MetalStateInvariant {
            state: "J2K Metal direct grayscale execution",
            reason: "compute encoder ownership is missing",
        });
    };
    let (destination, destination_item_index) =
        destination.map_or((None, 0), |(destination, index)| (Some(destination), index));
    let mut execution = SingleGrayscaleExecution {
        runtime,
        encoder,
        fmt,
        dimensions: plan.dimensions,
        bit_depth: plan.bit_depth,
        retained_buffers,
        status_checks,
        scratch_buffers,
        bands,
        final_surface: None,
        destination,
        native_destination,
        destination_item_index,
        destination_written: false,
        round_centered_store: plan.round_centered_store,
    };
    let result = (|| {
        let mut step_idx = 0;
        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                for _ in 0..step_repeats(&plan.steps[step_idx]) {
                    execution.encode_classic_group(group)?;
                }
                step_idx = group.end_step;
                continue;
            }
            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                for _ in 0..step_repeats(&plan.steps[step_idx]) {
                    execution.encode_ht_group(group)?;
                }
                step_idx = group.end_step;
                continue;
            }
            if let (
                super::PreparedDirectGrayscaleStep::Idwt(idwt),
                Some(super::PreparedDirectGrayscaleStep::Store(store)),
            ) = (&plan.steps[step_idx], plan.steps.get(step_idx + 1))
            {
                if fuse_last_level_into_store()
                    && execution.encode_idwt_into_store(idwt, store, &plan.steps[step_idx + 2..])?
                {
                    step_idx += 2;
                    continue;
                }
            }
            for _ in 0..step_repeats(&plan.steps[step_idx]) {
                execution.encode_step(&plan.steps[step_idx])?;
            }
            step_idx += 1;
        }
        execution.finish()
    })();
    if let Some(encoder) = owned_encoder {
        encoder.endEncoding();
    }
    result
}

/// Whether a last wavelet level may write the final store directly: always,
/// except when `single_image_step_profile` repeats or disables steps.
#[cfg(target_os = "macos")]
fn fuse_last_level_into_store() -> bool {
    #[cfg(test)]
    {
        use crate::engine::test_counters::{
            fused_2d_idwt_for_test, single_step_repeats_for_test, SingleStepKind,
        };
        fused_2d_idwt_for_test()
            && single_step_repeats_for_test(SingleStepKind::Idwt) == 1
            && single_step_repeats_for_test(SingleStepKind::Store) == 1
    }
    #[cfg(not(test))]
    {
        true
    }
}

/// How many times to encode `step`: once, except under
/// `single_image_step_profile`.
#[cfg(target_os = "macos")]
fn step_repeats(step: &super::PreparedDirectGrayscaleStep) -> u8 {
    #[cfg(test)]
    {
        use crate::engine::test_counters::{single_step_repeats_for_test, SingleStepKind};
        let kind = match step {
            super::PreparedDirectGrayscaleStep::Idwt(_) => SingleStepKind::Idwt,
            super::PreparedDirectGrayscaleStep::Store(_) => SingleStepKind::Store,
            _ => SingleStepKind::Tier1,
        };
        single_step_repeats_for_test(kind)
    }
    #[cfg(not(test))]
    {
        let _ = step;
        1
    }
}
