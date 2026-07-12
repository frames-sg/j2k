// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::{CommandBuffer, CommandBufferRef};

use crate::profile_env::label_command_buffer;

use super::{new_command_buffer, Error, MetalRuntime};

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
