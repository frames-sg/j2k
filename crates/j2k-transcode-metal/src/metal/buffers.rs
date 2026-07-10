// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_buffer_read_vec, checked_buffer_write, private_buffer, shared_buffer_for_len,
    shared_buffer_with_slice, size_of, Buffer, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob,
    Device, MetalTranscodeError, DWT97_BLOCK_COEFFICIENTS, METAL_DCT97_UNSUPPORTED_GRID,
    METAL_DCT_KERNEL_FAILED, PI,
};

pub(super) fn buffer_with_slice<T: j2k_core::accelerator::GpuAbi>(
    device: &Device,
    values: &[T],
) -> Buffer {
    shared_buffer_with_slice(device, values)
}

pub(super) fn dwt97_blocks_buffer(
    device: &Device,
    blocks: &[[[f64; 8]; 8]],
) -> Result<Buffer, MetalTranscodeError> {
    let value_count = dwt97_block_value_count(blocks.len())?;
    let mut buffer = output_buffer(device, value_count);
    write_dwt97_blocks_to_buffer(&mut buffer, blocks)?;
    Ok(buffer)
}

pub(super) fn dwt97_batch_blocks_buffer(
    device: &Device,
    jobs: &[DctGridToDwt97Job<'_>],
) -> Result<Buffer, MetalTranscodeError> {
    let value_count = dwt97_jobs_value_count(jobs.iter().map(|job| job.blocks.len()))?;
    let mut buffer = output_buffer(device, value_count);
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
    let mut buffer = output_buffer(device, value_count);
    let mut offset = 0;
    for job in jobs {
        offset += write_dwt97_blocks_to_buffer_at(&mut buffer, offset, job.blocks)?;
    }
    debug_assert_eq!(offset, value_count);
    Ok(buffer)
}

pub(super) fn dwt97_jobs_value_count(
    mut block_counts: impl Iterator<Item = usize>,
) -> Result<usize, MetalTranscodeError> {
    block_counts.try_fold(0_usize, |total, block_count| {
        let block_values = dwt97_block_value_count(block_count)?;
        total
            .checked_add(block_values)
            .ok_or(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID,
            ))
    })
}

pub(super) fn dwt97_block_value_count(block_count: usize) -> Result<usize, MetalTranscodeError> {
    let value_count = block_count.checked_mul(DWT97_BLOCK_COEFFICIENTS).ok_or(
        MetalTranscodeError::UnsupportedJob(METAL_DCT97_UNSUPPORTED_GRID),
    )?;
    value_count
        .checked_mul(size_of::<f32>())
        .ok_or(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ))?;
    Ok(value_count)
}

pub(super) fn write_dwt97_blocks_to_buffer(
    buffer: &mut Buffer,
    blocks: &[[[f64; 8]; 8]],
) -> Result<(), MetalTranscodeError> {
    let written = write_dwt97_blocks_to_buffer_at(buffer, 0, blocks)?;
    debug_assert_eq!(written, dwt97_block_value_count(blocks.len())?);
    Ok(())
}

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

    let byte_offset =
        start
            .checked_mul(size_of::<f32>())
            .ok_or(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID,
            ))?;
    let mut values = Vec::with_capacity(value_count);
    for block in blocks {
        for row in block {
            for &coefficient in row {
                values.push(coefficient as f32);
            }
        }
    }
    // SAFETY: DWT input buffers are populated before they are submitted to a
    // Metal command buffer, and this function has exclusive staging access.
    unsafe { checked_buffer_write::<f32>(buffer, byte_offset, &values) }
        .map_err(|_| MetalTranscodeError::UnsupportedJob(METAL_DCT97_UNSUPPORTED_GRID))?;
    Ok(values.len())
}

pub(super) fn buffer_f32_capacity(buffer: &Buffer) -> usize {
    let element_size = size_of::<f32>() as u64;
    usize::try_from(buffer.length() / element_size).unwrap_or(usize::MAX)
}

pub(super) fn output_buffer(device: &Device, value_count: usize) -> Buffer {
    shared_buffer_for_len::<f32>(device, value_count)
}

pub(super) fn private_f32_buffer(device: &Device, value_count: usize) -> Buffer {
    private_buffer(device, value_count.saturating_mul(size_of::<f32>()))
}

pub(super) fn output_i32_buffer(device: &Device, value_count: usize) -> Buffer {
    shared_buffer_for_len::<i32>(device, value_count)
}

pub(super) fn read_f32_buffer(
    buffer: &Buffer,
    value_count: usize,
) -> Result<Vec<f64>, MetalTranscodeError> {
    shared_f32_slice(buffer, value_count).map(|values| f32_slice_to_f64(&values))
}

pub(super) fn read_i32_buffer(
    buffer: &Buffer,
    value_count: usize,
) -> Result<Vec<i32>, MetalTranscodeError> {
    shared_i32_slice(buffer, value_count)
}

pub(super) fn shared_f32_slice(
    buffer: &Buffer,
    value_count: usize,
) -> Result<Vec<f32>, MetalTranscodeError> {
    // SAFETY: Every caller waits for the producing command buffer before
    // materializing these owned values.
    unsafe { checked_buffer_read_vec(buffer, 0, value_count) }
        .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))
}

pub(super) fn shared_i32_slice(
    buffer: &Buffer,
    value_count: usize,
) -> Result<Vec<i32>, MetalTranscodeError> {
    // SAFETY: Every caller waits for the producing command buffer before
    // materializing these owned values.
    unsafe { checked_buffer_read_vec(buffer, 0, value_count) }
        .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))
}

pub(super) fn f32_slice_to_f64(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&value| f64::from(value)).collect()
}

pub(super) fn idct8_basis_table() -> [f32; 64] {
    let mut table = [0.0; 64];
    for sample_idx in 0..8 {
        for freq in 0..8 {
            table[sample_idx * 8 + freq] = idct8_basis(sample_idx, freq);
        }
    }
    table
}

pub(super) fn idct8_basis(sample_idx: usize, freq: usize) -> f32 {
    let scale = if freq == 0 {
        (1.0_f32 / 8.0).sqrt()
    } else {
        (2.0_f32 / 8.0).sqrt()
    };
    scale * (((sample_idx as f32 + 0.5) * freq as f32 * PI) / 8.0).cos()
}
