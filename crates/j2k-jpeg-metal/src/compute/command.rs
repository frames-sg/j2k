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
    commit_and_wait(command_buffer)
        .map_err(|error| metal_kernel_support_error(error.to_string(), error))
}

pub(in crate::compute) fn wait_for_completion_jpeg(
    command_buffer: &CommandBufferRef,
) -> Result<(), Error> {
    wait_for_completion(command_buffer)
        .map_err(|error| metal_kernel_support_error(error.to_string(), error))
}
