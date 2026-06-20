// SPDX-License-Identifier: Apache-2.0

use std::mem::size_of;

use metal::Buffer;

use super::MetalRuntime;
use crate::Error;

pub(super) struct DirectScratchBuffer {
    pub(super) bytes: usize,
    pub(super) buffer: Buffer,
}

pub(super) fn take_f32_scratch_buffer(
    runtime: &MetalRuntime,
    len: usize,
) -> Result<DirectScratchBuffer, Error> {
    let bytes = len.max(1).saturating_mul(size_of::<f32>());
    Ok(DirectScratchBuffer {
        bytes,
        buffer: runtime.take_private_buffer(bytes)?,
    })
}

pub(super) fn recycle_scratch_buffers(
    runtime: &MetalRuntime,
    scratch_buffers: Vec<DirectScratchBuffer>,
) -> Result<(), Error> {
    for scratch in scratch_buffers {
        runtime.recycle_private_buffer(scratch.bytes, scratch.buffer)?;
    }
    Ok(())
}

pub(super) fn take_recyclable_private_buffer(
    runtime: &MetalRuntime,
    bytes: usize,
    recyclable_private_buffers: &mut Vec<(usize, Buffer)>,
) -> Result<Buffer, Error> {
    let bytes = bytes.max(1);
    let buffer = runtime.take_private_buffer(bytes)?;
    recyclable_private_buffers.push((bytes, buffer.clone()));
    Ok(buffer)
}

pub(super) fn recycle_private_buffers(
    runtime: &MetalRuntime,
    recyclable_private_buffers: Vec<(usize, Buffer)>,
) -> Result<(), Error> {
    for (bytes, buffer) in recyclable_private_buffers {
        runtime.recycle_private_buffer(bytes, buffer)?;
    }
    Ok(())
}

pub(super) fn take_recyclable_shared_buffer(
    runtime: &MetalRuntime,
    bytes: usize,
    recyclable_shared_buffers: &mut Vec<(usize, Buffer)>,
) -> Result<Buffer, Error> {
    let bytes = bytes.max(1);
    let buffer = runtime.take_shared_buffer(bytes)?;
    recyclable_shared_buffers.push((bytes, buffer.clone()));
    Ok(buffer)
}

pub(super) fn recycle_shared_buffers(
    runtime: &MetalRuntime,
    recyclable_shared_buffers: Vec<(usize, Buffer)>,
) -> Result<(), Error> {
    for (bytes, buffer) in recyclable_shared_buffers {
        runtime.recycle_shared_buffer(bytes, buffer)?;
    }
    Ok(())
}
