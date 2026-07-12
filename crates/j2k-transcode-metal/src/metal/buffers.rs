// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_buffer_read_vec, checked_buffer_write, size_of, try_transcode_vec_with_capacity,
    Buffer, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob, Device, MetalSupportError,
    MetalTranscodeError, DWT97_BLOCK_COEFFICIENTS, METAL_DCT97_UNSUPPORTED_GRID,
    METAL_READBACK_CHUNK_BYTES,
};

mod basis;
mod storage;
pub(super) use storage::{
    buffer_with_slice, dwt97_block_value_count, dwt97_blocks_buffer, dwt97_jobs_value_count,
    output_buffer, output_i32_buffer, private_f32_buffer,
};

const READBACK_CHUNK_VALUES: usize = METAL_READBACK_CHUNK_BYTES / size_of::<u32>();
const DWT97_UPLOAD_CHUNK_BLOCKS: usize = 64;
const DWT97_UPLOAD_CHUNK_VALUES: usize = DWT97_UPLOAD_CHUNK_BLOCKS * DWT97_BLOCK_COEFFICIENTS;

pub(super) fn dwt97_batch_blocks_buffer(
    device: &Device,
    jobs: &[DctGridToDwt97Job<'_>],
) -> Result<Buffer, MetalTranscodeError> {
    let value_count = dwt97_jobs_value_count(jobs.iter().map(|job| job.blocks.len()))?;
    let mut buffer = output_buffer(device, value_count)?;
    let mut offset = 0;
    for job in jobs {
        offset += write_dwt97_blocks_to_buffer_at(&mut buffer, offset, job.blocks)?;
    }
    debug_assert_eq!(offset, value_count);
    Ok(buffer)
}

pub(super) fn dwt97_codeblock_batch_blocks_buffer(
    device: &Device,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
) -> Result<Buffer, MetalTranscodeError> {
    let value_count = dwt97_jobs_value_count(jobs.iter().map(|job| job.blocks.len()))?;
    let mut buffer = output_buffer(device, value_count)?;
    let mut offset = 0;
    for job in jobs {
        offset += write_dwt97_blocks_to_buffer_at(&mut buffer, offset, job.blocks)?;
    }
    debug_assert_eq!(offset, value_count);
    Ok(buffer)
}

pub(super) fn write_dwt97_blocks_to_buffer(
    buffer: &mut Buffer,
    blocks: &[[[f64; 8]; 8]],
) -> Result<(), MetalTranscodeError> {
    let written = write_dwt97_blocks_to_buffer_at(buffer, 0, blocks)?;
    debug_assert_eq!(written, dwt97_block_value_count(blocks.len())?);
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the Metal kernel ABI intentionally consumes f32 DCT coefficients"
)]
#[expect(
    unsafe_code,
    reason = "the checked Metal buffer helper requires one audited host write boundary"
)]
pub(super) fn write_dwt97_blocks_to_buffer_at(
    buffer: &mut Buffer,
    start: usize,
    blocks: &[[[f64; 8]; 8]],
) -> Result<usize, MetalTranscodeError> {
    let value_count = dwt97_block_value_count(blocks.len())?;
    let end = start
        .checked_add(value_count)
        .ok_or(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ))?;
    if end > buffer_f32_capacity(buffer) {
        return Err(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ));
    }

    let mut written = start;
    for block_chunk in blocks.chunks(DWT97_UPLOAD_CHUNK_BLOCKS) {
        let mut values = [0.0_f32; DWT97_UPLOAD_CHUNK_VALUES];
        let mut value_idx = 0usize;
        for block in block_chunk {
            for row in block {
                for &coefficient in row {
                    values[value_idx] = coefficient as f32;
                    value_idx += 1;
                }
            }
        }
        let byte_offset =
            written
                .checked_mul(size_of::<f32>())
                .ok_or(MetalTranscodeError::UnsupportedJob(
                    METAL_DCT97_UNSUPPORTED_GRID,
                ))?;
        // SAFETY: DWT input buffers are populated before they are submitted to
        // a Metal command buffer, and this function has exclusive staging access.
        unsafe { checked_buffer_write::<f32>(buffer, byte_offset, &values[..value_idx]) }
            .map_err(|error| MetalTranscodeError::support("Metal DCT block upload", error))?;
        written = written
            .checked_add(value_idx)
            .ok_or(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID,
            ))?;
    }
    if written != end {
        return Err(MetalTranscodeError::Kernel(
            "Metal DCT upload length disagrees with validated workspace",
        ));
    }
    Ok(value_count)
}

pub(super) fn buffer_f32_capacity(buffer: &Buffer) -> usize {
    let element_size = size_of::<f32>() as u64;
    usize::try_from(buffer.length() / element_size).unwrap_or(usize::MAX)
}

pub(super) fn read_f32_buffer(
    buffer: &Buffer,
    value_count: usize,
) -> Result<Vec<f64>, MetalTranscodeError> {
    read_f32_buffer_at(buffer, 0, value_count)
}

pub(super) fn read_f32_buffer_at(
    buffer: &Buffer,
    start: usize,
    value_count: usize,
) -> Result<Vec<f64>, MetalTranscodeError> {
    let mut output =
        try_transcode_vec_with_capacity(value_count, "Metal f32-to-f64 output materialization")?;
    read_buffer_chunks::<f32>(buffer, start, value_count, |values| {
        output.extend(values.iter().copied().map(f64::from));
    })?;
    Ok(output)
}

pub(super) fn read_i32_buffer_at(
    buffer: &Buffer,
    start: usize,
    value_count: usize,
) -> Result<Vec<i32>, MetalTranscodeError> {
    let mut output =
        try_transcode_vec_with_capacity(value_count, "Metal i32 output materialization")?;
    read_buffer_chunks::<i32>(buffer, start, value_count, |values| {
        output.extend_from_slice(values);
    })?;
    Ok(output)
}

#[expect(
    unsafe_code,
    reason = "the checked Metal buffer helper requires one audited post-completion read boundary"
)]
fn read_buffer_chunks<T: j2k_core::accelerator::GpuAbi>(
    buffer: &Buffer,
    start: usize,
    value_count: usize,
    mut consume: impl FnMut(&[T]),
) -> Result<(), MetalTranscodeError> {
    let end =
        start
            .checked_add(value_count)
            .ok_or(MetalTranscodeError::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "Metal output readback range",
            })?;
    let mut position = start;
    while position < end {
        let chunk_len = (end - position).min(READBACK_CHUNK_VALUES);
        let byte_offset = position.checked_mul(size_of::<T>()).ok_or(
            MetalTranscodeError::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "Metal output readback range",
            },
        )?;
        // SAFETY: Every caller waits for the producing command buffer before
        // materializing these owned values. The checked helper validates the
        // byte range and creates an owned, bounded chunk.
        let values = unsafe { checked_buffer_read_vec(buffer, byte_offset, chunk_len) }
            .map_err(|error| map_readback_error::<T>(error, chunk_len))?;
        consume(&values);
        position += chunk_len;
    }
    Ok(())
}

fn map_readback_error<T>(error: MetalSupportError, element_count: usize) -> MetalTranscodeError {
    if matches!(error, MetalSupportError::BufferReadbackAllocation { .. }) {
        MetalTranscodeError::HostAllocationFailed {
            requested: element_count.saturating_mul(size_of::<T>()),
            what: "Metal output readback chunk",
        }
    } else {
        MetalTranscodeError::support("Metal output readback", error)
    }
}

pub(super) fn idct8_basis_table() -> [f32; 64] {
    basis::idct8_basis_table()
}
