// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    checked_device_element_count, checked_device_workspace_bytes, checked_host_element_count,
    checked_host_workspace_bytes, size_of, validate_codeblock_output_host_workspace,
    Dwt97TwoDimensional, Htj2k97CodeBlockOptions, MetalTranscodeError, ProjectedBands,
    DWT97_BLOCK_COEFFICIENTS, METAL_READBACK_CHUNK_BYTES,
};

pub(in crate::metal) fn validate_float_projection_allocations(
    total_blocks: usize,
    width: usize,
    height: usize,
    batch_count: usize,
    weight_host_bytes: usize,
    weight_device_bytes: usize,
    device_sample_buffers: usize,
) -> Result<(), MetalTranscodeError> {
    let output_count = checked_host_element_count::<f64>(
        &[width, height, batch_count],
        "projected wavelet output bands",
    )?;
    let input_count = checked_device_element_count::<f32>(
        &[total_blocks, DWT97_BLOCK_COEFFICIENTS],
        "DCT coefficient upload",
    )?;
    let device_sample_count = checked_device_element_count::<f32>(
        &[width, height, batch_count],
        "projected wavelet device output",
    )?;
    let source_metadata_count = checked_host_element_count::<ProjectedBands>(
        &[batch_count],
        "projected wavelet batch metadata",
    )?;
    let destination_metadata_count = checked_host_element_count::<Dwt97TwoDimensional<f64>>(
        &[batch_count],
        "projected wavelet result metadata",
    )?;
    let output_bytes = output_count.saturating_mul(size_of::<f64>());
    let source_metadata_bytes = source_metadata_count.saturating_mul(size_of::<ProjectedBands>());
    let destination_metadata_bytes =
        destination_metadata_count.saturating_mul(size_of::<Dwt97TwoDimensional<f64>>());
    let readback_peak = output_bytes
        .saturating_add(source_metadata_bytes)
        .saturating_add(METAL_READBACK_CHUNK_BYTES);
    let conversion_peak = output_bytes
        .saturating_add(source_metadata_bytes)
        .saturating_add(destination_metadata_bytes);
    checked_host_workspace_bytes(
        &[
            weight_host_bytes,
            readback_peak.max(conversion_peak).max(weight_device_bytes),
        ],
        "projected wavelet host workspace",
    )?;
    checked_device_workspace_bytes(
        &[
            input_count.saturating_mul(size_of::<f32>()),
            device_sample_count
                .saturating_mul(size_of::<f32>())
                .saturating_mul(device_sample_buffers),
            weight_device_bytes,
        ],
        "projected wavelet device workspace",
    )?;
    Ok(())
}

pub(in crate::metal) fn validate_codeblock_projection_allocations(
    total_blocks: usize,
    width: usize,
    height: usize,
    batch_count: usize,
    options: Htj2k97CodeBlockOptions,
) -> Result<(), MetalTranscodeError> {
    validate_codeblock_output_host_workspace(width, height, batch_count, options)?;
    let input_count = checked_device_element_count::<f32>(
        &[total_blocks, DWT97_BLOCK_COEFFICIENTS],
        "HTJ2K DCT coefficient upload",
    )?;
    let device_sample_count = checked_device_element_count::<f32>(
        &[width, height, batch_count],
        "HTJ2K wavelet device workspace",
    )?;
    checked_device_element_count::<i32>(
        &[width, height, batch_count],
        "HTJ2K prequantized device output",
    )?;
    let input_bytes = input_count.saturating_mul(size_of::<f32>());
    let sample_bytes = device_sample_count.saturating_mul(size_of::<f32>());
    checked_device_workspace_bytes(
        &[input_bytes, sample_bytes, sample_bytes, sample_bytes],
        "prequantized HTJ2K device workspace",
    )?;
    Ok(())
}
