// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_metal_support::{
    checked_blit_command_encoder, checked_command_buffer, checked_compute_command_encoder,
};
use metal::{
    BlitCommandEncoder, CommandBuffer, CommandBufferRef, CommandQueueRef, ComputeCommandEncoder,
};

use crate::error::metal_kernel_support_error;
use crate::profile_env::label_command_buffer;

use super::{Error, MetalRuntime};

pub(in crate::compute) fn new_command_buffer(
    queue: &CommandQueueRef,
) -> Result<CommandBuffer, Error> {
    #[cfg(test)]
    super::test_counters::record_metal_command_buffer();
    checked_command_buffer(queue).map_err(|source| {
        metal_kernel_support_error("J2K Metal command buffer creation failed", source)
    })
}

pub(in crate::compute) fn new_compute_command_encoder(
    command_buffer: &CommandBufferRef,
) -> Result<ComputeCommandEncoder, Error> {
    #[cfg(test)]
    super::test_counters::record_metal_compute_encoder();
    checked_compute_command_encoder(command_buffer).map_err(|source| {
        metal_kernel_support_error("J2K Metal compute encoder creation failed", source)
    })
}

pub(in crate::compute) fn new_blit_command_encoder(
    command_buffer: &CommandBufferRef,
) -> Result<BlitCommandEncoder, Error> {
    checked_blit_command_encoder(command_buffer).map_err(|source| {
        metal_kernel_support_error("J2K Metal blit encoder creation failed", source)
    })
}

#[derive(Clone, Copy)]
pub(super) struct DirectIdwtCommandBuffers<'a> {
    pub(super) interleave: &'a CommandBufferRef,
    pub(super) horizontal: &'a CommandBufferRef,
    pub(super) vertical: &'a CommandBufferRef,
}

impl<'a> DirectIdwtCommandBuffers<'a> {
    pub(super) fn single(command_buffer: &'a CommandBufferRef) -> Self {
        Self {
            interleave: command_buffer,
            horizontal: command_buffer,
            vertical: command_buffer,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct DirectColorBatchCommandBuffers<'a> {
    pub(super) default: &'a CommandBufferRef,
    pub(super) idwt: DirectIdwtCommandBuffers<'a>,
    pub(super) store: &'a CommandBufferRef,
    pub(super) mct_pack: &'a CommandBufferRef,
}

impl<'a> DirectColorBatchCommandBuffers<'a> {
    pub(super) fn single(command_buffer: &'a CommandBufferRef) -> Self {
        Self {
            default: command_buffer,
            idwt: DirectIdwtCommandBuffers::single(command_buffer),
            store: command_buffer,
            mct_pack: command_buffer,
        }
    }
}

pub(super) struct DecodeHybridSplitCommandBuffers {
    pub(super) idwt_interleave: CommandBuffer,
    pub(super) idwt_horizontal: CommandBuffer,
    pub(super) idwt_vertical: CommandBuffer,
    pub(super) store: CommandBuffer,
    pub(super) mct_pack: CommandBuffer,
}

impl DecodeHybridSplitCommandBuffers {
    pub(super) fn new(runtime: &MetalRuntime) -> Result<Self, Error> {
        let idwt_interleave = new_command_buffer(&runtime.queue)?;
        label_command_buffer(&idwt_interleave, "j2k decode hybrid IDWT interleave stage");
        let idwt_horizontal = new_command_buffer(&runtime.queue)?;
        label_command_buffer(&idwt_horizontal, "j2k decode hybrid IDWT horizontal stage");
        let idwt_vertical = new_command_buffer(&runtime.queue)?;
        label_command_buffer(&idwt_vertical, "j2k decode hybrid IDWT vertical stage");
        let store = new_command_buffer(&runtime.queue)?;
        label_command_buffer(&store, "j2k decode hybrid store stage");
        let mct_pack = new_command_buffer(&runtime.queue)?;
        label_command_buffer(&mct_pack, "j2k decode hybrid MCT pack stage");
        Ok(Self {
            idwt_interleave,
            idwt_horizontal,
            idwt_vertical,
            store,
            mct_pack,
        })
    }

    pub(super) fn refs(&self) -> DirectColorBatchCommandBuffers<'_> {
        DirectColorBatchCommandBuffers {
            default: &self.idwt_interleave,
            idwt: DirectIdwtCommandBuffers {
                interleave: &self.idwt_interleave,
                horizontal: &self.idwt_horizontal,
                vertical: &self.idwt_vertical,
            },
            store: &self.store,
            mct_pack: &self.mct_pack,
        }
    }

    pub(super) fn commit_in_order(&self) {
        self.idwt_interleave.commit();
        self.idwt_horizontal.commit();
        self.idwt_vertical.commit();
        self.store.commit();
        self.mct_pack.commit();
    }
}
