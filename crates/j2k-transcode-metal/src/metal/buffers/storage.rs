// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    private_buffer_for_len, shared_buffer_for_len, shared_buffer_with_slice, size_of, Buffer,
    Device, MetalTranscodeError, DWT97_BLOCK_COEFFICIENTS, METAL_DCT97_UNSUPPORTED_GRID,
};

pub(in crate::metal) fn buffer_with_slice<T: j2k_core::accelerator::GpuAbi>(
    device: &Device,
    values: &[T],
) -> Result<Buffer, MetalTranscodeError> {
    shared_buffer_with_slice(device, values, "Metal transcode input upload")
}

pub(in crate::metal) fn dwt97_blocks_buffer(
    device: &Device,
    blocks: &[[[f64; 8]; 8]],
) -> Result<Buffer, MetalTranscodeError> {
    let value_count = dwt97_block_value_count(blocks.len())?;
    let mut buffer = output_buffer(device, value_count)?;
    super::write_dwt97_blocks_to_buffer(&mut buffer, blocks)?;
    Ok(buffer)
}

pub(in crate::metal) fn dwt97_jobs_value_count(
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

pub(in crate::metal) fn dwt97_block_value_count(
    block_count: usize,
) -> Result<usize, MetalTranscodeError> {
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

pub(in crate::metal) fn output_buffer(
    device: &Device,
    value_count: usize,
) -> Result<Buffer, MetalTranscodeError> {
    shared_buffer_for_len::<f32>(device, value_count, "Metal f32 output")
}

pub(in crate::metal) fn private_f32_buffer(
    device: &Device,
    value_count: usize,
) -> Result<Buffer, MetalTranscodeError> {
    private_buffer_for_len::<f32>(device, value_count, "Metal private f32 workspace")
}

pub(in crate::metal) fn output_i32_buffer(
    device: &Device,
    value_count: usize,
) -> Result<Buffer, MetalTranscodeError> {
    shared_buffer_for_len::<i32>(device, value_count, "Metal i32 output")
}
