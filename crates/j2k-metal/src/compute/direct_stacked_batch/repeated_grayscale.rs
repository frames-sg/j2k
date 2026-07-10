// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::{Buffer, CommandBufferRef};

use super::super::{
    DirectScratchBuffer, DirectStatusCheck, Error, MetalRuntime, PixelFormat,
    PreparedDirectGrayscalePlan, Surface,
};

mod execution;

pub(in super::super) use self::execution::encode_repeated_direct_grayscale_plan_in_command_buffer;

pub(in super::super) struct RepeatedDirectGrayscalePlanRequest<'a> {
    pub(in super::super) runtime: &'a MetalRuntime,
    pub(in super::super) command_buffer: &'a CommandBufferRef,
    pub(in super::super) plan: &'a PreparedDirectGrayscalePlan,
    pub(in super::super) fmt: PixelFormat,
    pub(in super::super) count: usize,
    pub(in super::super) retained_buffers: &'a mut Vec<Buffer>,
    pub(in super::super) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(in super::super) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}
