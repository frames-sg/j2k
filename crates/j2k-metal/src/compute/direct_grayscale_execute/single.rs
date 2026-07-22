// SPDX-License-Identifier: MIT OR Apache-2.0

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
    retained_buffers: &'a mut Vec<Buffer>,
    status_checks: &'a mut Vec<DirectStatusCheck>,
    scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct DirectGrayscaleDestinationExecutionRequest<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) plan: &'a PreparedDirectGrayscalePlan,
    pub(in crate::compute) fmt: PixelFormat,
    pub(in crate::compute) destination: &'a MetalImageDestination,
    pub(in crate::compute) destination_item_index: usize,
    pub(in crate::compute) retained_buffers: &'a mut Vec<Buffer>,
    pub(in crate::compute) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(in crate::compute) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_prepared_direct_grayscale_plan_in_command_buffer(
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
pub(in crate::compute) fn encode_prepared_direct_grayscale_plan_into_in_encoder(
    request: DirectGrayscaleDestinationExecutionRequest<'_>,
    encoder: &metal::ComputeCommandEncoderRef,
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

#[cfg(target_os = "macos")]
fn encode_prepared_direct_grayscale_plan_in_command_buffer_inner(
    request: GrayscalePlanExecutionRequest<'_>,
    existing_encoder: Option<&metal::ComputeCommandEncoderRef>,
) -> Result<Option<Surface>, Error> {
    let GrayscalePlanExecutionRequest {
        runtime,
        command_buffer,
        plan,
        fmt,
        destination,
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
    let encoder = existing_encoder.unwrap_or_else(|| {
        owned_encoder
            .as_ref()
            .expect("missing owned grayscale compute encoder")
    });
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
        destination_item_index,
        destination_written: false,
    };
    let result = (|| {
        let mut step_idx = 0;
        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                execution.encode_classic_group(group)?;
                step_idx = group.end_step;
                continue;
            }
            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                execution.encode_ht_group(group)?;
                step_idx = group.end_step;
                continue;
            }
            execution.encode_step(&plan.steps[step_idx])?;
            step_idx += 1;
        }
        execution.finish()
    })();
    if let Some(encoder) = owned_encoder {
        encoder.end_encoding();
    }
    result
}
