// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked Metal command creation and completion boundaries.

use crate::error::metal_kernel_support_error;
use crate::metal_types::{
    BlitCommandEncoder, CommandBuffer, CommandBufferRef, CommandQueueRef, ComputeCommandEncoder,
};
use crate::Error;
use j2k_metal_support::{
    checked_blit_command_encoder, checked_command_buffer, checked_compute_command_encoder,
    commit_and_wait, wait_for_completion,
};

pub(in crate::compute) fn new_command_buffer(
    queue: &CommandQueueRef,
) -> Result<CommandBuffer, Error> {
    checked_command_buffer(queue).map_err(|source| {
        metal_kernel_support_error("JPEG Metal command buffer creation failed", source)
    })
}

pub(in crate::compute) fn new_compute_command_encoder(
    command_buffer: &CommandBufferRef,
) -> Result<ComputeCommandEncoder, Error> {
    checked_compute_command_encoder(command_buffer).map_err(|source| {
        metal_kernel_support_error("JPEG Metal compute encoder creation failed", source)
    })
}

pub(in crate::compute) fn new_blit_command_encoder(
    command_buffer: &CommandBufferRef,
) -> Result<BlitCommandEncoder, Error> {
    checked_blit_command_encoder(command_buffer).map_err(|source| {
        metal_kernel_support_error("JPEG Metal blit encoder creation failed", source)
    })
}

pub(in crate::compute) fn commit_and_wait_jpeg(
    command_buffer: &CommandBufferRef,
) -> Result<(), Error> {
    let result = commit_and_wait(command_buffer)
        .map_err(|error| metal_kernel_support_error(error.to_string(), error));
    #[cfg(test)]
    gpu_time::record(command_buffer);
    result
}

pub(in crate::compute) fn wait_for_completion_jpeg(
    command_buffer: &CommandBufferRef,
) -> Result<(), Error> {
    let result = wait_for_completion(command_buffer)
        .map_err(|error| metal_kernel_support_error(error.to_string(), error));
    #[cfg(test)]
    gpu_time::record(command_buffer);
    result
}

/// Test-only GPU execution time of the command buffers this thread waited on.
#[cfg(test)]
pub(in crate::compute) mod gpu_time {
    use std::cell::Cell;

    use objc2_metal::MTLCommandBuffer as _;

    use crate::metal_types::CommandBufferRef;

    thread_local! {
        static GPU_SECONDS: Cell<f64> = const { Cell::new(0.0) };
        static SUBMIT_SECONDS: Cell<(f64, f64)> = const { Cell::new((0.0, 0.0)) };
    }

    pub(super) fn record(command_buffer: &CommandBufferRef) {
        let seconds = command_buffer.GPUEndTime() - command_buffer.GPUStartTime();
        GPU_SECONDS.with(|total| total.set(total.get() + seconds));
        let scheduling = command_buffer.kernelEndTime() - command_buffer.kernelStartTime();
        let queued = command_buffer.GPUStartTime() - command_buffer.kernelEndTime();
        SUBMIT_SECONDS.with(|total| {
            let (sum_scheduling, sum_queued) = total.get();
            total.set((sum_scheduling + scheduling, sum_queued + queued));
        });
    }

    /// Returns the accumulated GPU seconds and resets the counter.
    pub(in crate::compute) fn take_seconds() -> f64 {
        GPU_SECONDS.with(|total| total.replace(0.0))
    }

    /// Returns the accumulated CPU-side scheduling seconds (kernel start to
    /// kernel end) and queue seconds (kernel end to GPU start), then resets.
    pub(in crate::compute) fn take_submit_seconds() -> (f64, f64) {
        SUBMIT_SECONDS.with(|total| total.replace((0.0, 0.0)))
    }
}
