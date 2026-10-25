// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::mem::size_of;

use crate::metal_types::Buffer;

use super::{copied_recyclable_shared_slice_buffer, with_runtime_for_session, MetalRuntime};
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
    let mut first_error = None;
    for scratch in scratch_buffers {
        if let Err(error) = runtime.recycle_private_buffer(scratch.buffer) {
            if first_error.is_none() {
                first_error = Some(error);
            }
        }
    }
    first_error.map_or(Ok(()), Err)
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

/// Copies `bytes` into a pooled shared buffer of `session`'s runtime and adds
/// its owner to `recyclable_shared_buffers`, for
/// [`recycle_session_shared_buffers`] once no GPU work reads it.
#[cfg(target_os = "macos")]
pub(crate) fn copied_session_shared_buffer(
    session: &crate::MetalBackendSession,
    bytes: &[u8],
    recyclable_shared_buffers: &mut Vec<PooledBuffer>,
) -> Result<Buffer, Error> {
    with_runtime_for_session(session, |runtime| {
        copied_recyclable_shared_slice_buffer(runtime, bytes, recyclable_shared_buffers)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn recycle_session_shared_buffers(
    session: &crate::MetalBackendSession,
    recyclable_shared_buffers: Vec<PooledBuffer>,
) -> Result<(), Error> {
    with_runtime_for_session(session, |runtime| {
        recycle_shared_buffers(runtime, recyclable_shared_buffers)
    })
}
