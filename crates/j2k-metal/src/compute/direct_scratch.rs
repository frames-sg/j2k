// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use metal::Buffer;

use super::MetalRuntime;
use crate::{buffer_pool::PooledBuffer, Error};

pub(super) struct DirectScratchBuffer {
    pub(super) buffer: PooledBuffer,
}

pub(super) fn take_f32_scratch_buffer(
    runtime: &MetalRuntime,
    len: usize,
) -> Result<DirectScratchBuffer, Error> {
    let bytes = len.max(1).saturating_mul(size_of::<f32>());
    Ok(DirectScratchBuffer {
        buffer: runtime.take_private_buffer(bytes)?,
    })
}

pub(super) fn recycle_scratch_buffers(
    runtime: &MetalRuntime,
    scratch_buffers: Vec<DirectScratchBuffer>,
) -> Result<(), Error> {
    for scratch in scratch_buffers {
        runtime.recycle_private_buffer(scratch.buffer)?;
    }
    Ok(())
}

pub(super) fn take_recyclable_private_buffer(
    runtime: &MetalRuntime,
    bytes: usize,
    recyclable_private_buffers: &mut Vec<PooledBuffer>,
) -> Result<Buffer, Error> {
    let bytes = bytes.max(1);
    let owner = runtime.take_private_buffer(bytes)?;
    let buffer = owner.buffer().clone();
    recyclable_private_buffers.push(owner);
    Ok(buffer)
}

pub(super) fn recycle_private_buffers(
    runtime: &MetalRuntime,
    recyclable_private_buffers: Vec<PooledBuffer>,
) -> Result<(), Error> {
    for buffer in recyclable_private_buffers {
        runtime.recycle_private_buffer(buffer)?;
    }
    Ok(())
}

pub(super) fn take_recyclable_shared_buffer(
    runtime: &MetalRuntime,
    bytes: usize,
    recyclable_shared_buffers: &mut Vec<PooledBuffer>,
) -> Result<Buffer, Error> {
    let bytes = bytes.max(1);
    let owner = runtime.take_shared_buffer(bytes)?;
    let buffer = owner.buffer().clone();
    recyclable_shared_buffers.push(owner);
    Ok(buffer)
}

pub(super) fn recycle_shared_buffers(
    runtime: &MetalRuntime,
    recyclable_shared_buffers: Vec<PooledBuffer>,
) -> Result<(), Error> {
    for buffer in recyclable_shared_buffers {
        runtime.recycle_shared_buffer(buffer)?;
    }
    Ok(())
}
