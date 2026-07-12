// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_batch_len, checked_host_element_count, checked_host_workspace_bytes,
    code_block_len_from_exp, dwt97_total_bitplanes, read_i32_buffer_at, size_of,
    try_transcode_vec_from_slice, try_transcode_vec_with_capacity, u32_param,
    DctGridToHtj2k97CodeBlockJob, Dwt97CodeBlockOutputBuffers, Htj2k97CodeBlockOptions,
    J2kSubBandType, MetalTranscodeError, PrequantizedHtj2k97CodeBlock,
    PrequantizedHtj2k97Component, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
    ProjectionBatchShape, METAL_DCT97_UNSUPPORTED_GRID, METAL_READBACK_CHUNK_BYTES,
};

pub(super) fn validate_codeblock_output_host_workspace(
    width: usize,
    height: usize,
    batch_count: usize,
    options: Htj2k97CodeBlockOptions,
) -> Result<(), MetalTranscodeError> {
    let output_count = checked_host_element_count::<i32>(
        &[width, height, batch_count],
        "prequantized HTJ2K coefficients",
    )?;
    let item_count =
        checked_host_element_count::<i32>(&[width, height], "prequantized HTJ2K item readback")?;
    let codeblocks_per_item = codeblocks_per_item(width, height, options)?;
    let codeblock_count = checked_host_element_count::<PrequantizedHtj2k97CodeBlock>(
        &[codeblocks_per_item, batch_count],
        "prequantized HTJ2K code-block metadata",
    )?;
    let component_count = checked_host_element_count::<PrequantizedHtj2k97Component>(
        &[batch_count],
        "prequantized HTJ2K component metadata",
    )?;
    let resolution_count = checked_host_element_count::<PrequantizedHtj2k97Resolution>(
        &[2, batch_count],
        "prequantized HTJ2K resolution metadata",
    )?;
    let subband_count = checked_host_element_count::<PrequantizedHtj2k97Subband>(
        &[4, batch_count],
        "prequantized HTJ2K subband metadata",
    )?;
    checked_host_workspace_bytes(
        &[
            output_count.saturating_mul(size_of::<i32>()),
            item_count.saturating_mul(size_of::<i32>()),
            codeblock_count.saturating_mul(size_of::<PrequantizedHtj2k97CodeBlock>()),
            component_count.saturating_mul(size_of::<PrequantizedHtj2k97Component>()),
            resolution_count.saturating_mul(size_of::<PrequantizedHtj2k97Resolution>()),
            subband_count.saturating_mul(size_of::<PrequantizedHtj2k97Subband>()),
            METAL_READBACK_CHUNK_BYTES,
        ],
        "prequantized HTJ2K host workspace",
    )?;
    Ok(())
}

pub(super) fn read_prequantized_97_codeblock_outputs(
    buffers: &Dwt97CodeBlockOutputBuffers,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    shape: ProjectionBatchShape,
    options: Htj2k97CodeBlockOptions,
    unsupported_grid: &'static str,
) -> Result<Vec<PrequantizedHtj2k97Component>, MetalTranscodeError> {
    if jobs.len() != shape.batch_count {
        return Err(MetalTranscodeError::UnsupportedJob(unsupported_grid));
    }
    validate_codeblock_output_host_workspace(
        shape.width as usize,
        shape.height as usize,
        shape.batch_count,
        options,
    )?;
    let mut components = try_transcode_vec_with_capacity(
        shape.batch_count,
        "prequantized HTJ2K component metadata",
    )?;
    for (idx, job) in jobs.iter().enumerate() {
        components.push(read_component(
            buffers,
            shape,
            idx,
            job.x_rsiz,
            job.y_rsiz,
            options,
            unsupported_grid,
        )?);
    }
    Ok(components)
}

fn read_component(
    buffers: &Dwt97CodeBlockOutputBuffers,
    shape: ProjectionBatchShape,
    item_index: usize,
    x_rsiz: u8,
    y_rsiz: u8,
    options: Htj2k97CodeBlockOptions,
    unsupported_grid: &'static str,
) -> Result<PrequantizedHtj2k97Component, MetalTranscodeError> {
    let ll = read_band(&buffers.ll, shape.ll_len, item_index, unsupported_grid)?;
    let hl = read_band(&buffers.hl, shape.hl_len, item_index, unsupported_grid)?;
    let lh = read_band(&buffers.lh, shape.lh_len, item_index, unsupported_grid)?;
    let hh = read_band(&buffers.hh, shape.hh_len, item_index, unsupported_grid)?;
    let base = resolution_from_subbands([subband(
        &ll,
        shape.low_width,
        shape.low_height,
        J2kSubBandType::LowLow,
        options,
    )?])?;
    let detail = resolution_from_subbands([
        subband(
            &hl,
            shape.high_width,
            shape.low_height,
            J2kSubBandType::HighLow,
            options,
        )?,
        subband(
            &lh,
            shape.low_width,
            shape.high_height,
            J2kSubBandType::LowHigh,
            options,
        )?,
        subband(
            &hh,
            shape.high_width,
            shape.high_height,
            J2kSubBandType::HighHigh,
            options,
        )?,
    ])?;
    Ok(PrequantizedHtj2k97Component {
        x_rsiz,
        y_rsiz,
        resolutions: try_vec_from_array([base, detail], "prequantized HTJ2K resolution metadata")?,
    })
}

fn read_band(
    buffer: &super::Buffer,
    band_len: usize,
    item_index: usize,
    unsupported_grid: &'static str,
) -> Result<Vec<i32>, MetalTranscodeError> {
    read_i32_buffer_at(
        buffer,
        checked_batch_len(band_len, item_index, unsupported_grid)?,
        band_len,
    )
}

fn resolution_from_subbands<const N: usize>(
    subbands: [PrequantizedHtj2k97Subband; N],
) -> Result<PrequantizedHtj2k97Resolution, MetalTranscodeError> {
    Ok(PrequantizedHtj2k97Resolution {
        subbands: try_vec_from_array(subbands, "prequantized HTJ2K subband metadata")?,
    })
}

fn subband(
    values: &[i32],
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    options: Htj2k97CodeBlockOptions,
) -> Result<PrequantizedHtj2k97Subband, MetalTranscodeError> {
    prequantized_subband_from_codeblock_buffer(
        values,
        width,
        height,
        sub_band_type,
        dwt97_total_bitplanes(options, sub_band_type),
        options,
    )
}

fn prequantized_subband_from_codeblock_buffer(
    values: &[i32],
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    total_bitplanes: u8,
    options: Htj2k97CodeBlockOptions,
) -> Result<PrequantizedHtj2k97Subband, MetalTranscodeError> {
    if width == 0 || height == 0 {
        return Ok(PrequantizedHtj2k97Subband {
            sub_band_type,
            num_cbs_x: 0,
            num_cbs_y: 0,
            total_bitplanes: 0,
            code_blocks: Vec::new(),
        });
    }

    let cb_width = code_block_len_from_exp(options.code_block_width_exp)?;
    let cb_height = code_block_len_from_exp(options.code_block_height_exp)?;
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let code_block_count = checked_codeblock_count(num_cbs_x, num_cbs_y)?;
    let mut code_blocks = try_transcode_vec_with_capacity(
        code_block_count,
        "prequantized HTJ2K code-block metadata",
    )?;
    let mut offset = 0usize;
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let block_width = (width - cbx * cb_width).min(cb_width);
            let block_height = (height - cby * cb_height).min(cb_height);
            let len = block_width.checked_mul(block_height).ok_or(
                MetalTranscodeError::UnsupportedJob(METAL_DCT97_UNSUPPORTED_GRID),
            )?;
            let end = offset
                .checked_add(len)
                .ok_or(MetalTranscodeError::UnsupportedJob(
                    METAL_DCT97_UNSUPPORTED_GRID,
                ))?;
            let coefficients = try_transcode_vec_from_slice(
                values
                    .get(offset..end)
                    .ok_or(MetalTranscodeError::UnsupportedJob(
                        METAL_DCT97_UNSUPPORTED_GRID,
                    ))?,
                "prequantized HTJ2K code-block coefficients",
            )?;
            code_blocks.push(PrequantizedHtj2k97CodeBlock {
                coefficients,
                width: u32_param(block_width, METAL_DCT97_UNSUPPORTED_GRID)?,
                height: u32_param(block_height, METAL_DCT97_UNSUPPORTED_GRID)?,
            });
            offset = end;
        }
    }
    if offset != values.len() {
        return Err(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ));
    }

    Ok(PrequantizedHtj2k97Subband {
        sub_band_type,
        num_cbs_x: u32_param(num_cbs_x, METAL_DCT97_UNSUPPORTED_GRID)?,
        num_cbs_y: u32_param(num_cbs_y, METAL_DCT97_UNSUPPORTED_GRID)?,
        total_bitplanes,
        code_blocks,
    })
}

fn codeblocks_per_item(
    width: usize,
    height: usize,
    options: Htj2k97CodeBlockOptions,
) -> Result<usize, MetalTranscodeError> {
    let cb_width = code_block_len_from_exp(options.code_block_width_exp)?;
    let cb_height = code_block_len_from_exp(options.code_block_height_exp)?;
    let low_width = width.div_ceil(2);
    let high_width = width / 2;
    let low_height = height.div_ceil(2);
    let high_height = height / 2;
    [
        (low_width, low_height),
        (high_width, low_height),
        (low_width, high_height),
        (high_width, high_height),
    ]
    .into_iter()
    .try_fold(0usize, |total, (band_width, band_height)| {
        let count = checked_codeblock_count(
            band_width.div_ceil(cb_width),
            band_height.div_ceil(cb_height),
        )?;
        total
            .checked_add(count)
            .ok_or_else(codeblock_metadata_overflow)
    })
}

fn checked_codeblock_count(columns: usize, rows: usize) -> Result<usize, MetalTranscodeError> {
    columns
        .checked_mul(rows)
        .ok_or_else(codeblock_metadata_overflow)
}

fn codeblock_metadata_overflow() -> MetalTranscodeError {
    MetalTranscodeError::HostAllocationTooLarge {
        requested: usize::MAX,
        cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        what: "prequantized HTJ2K code-block metadata",
    }
}

fn try_vec_from_array<T, const N: usize>(
    values: [T; N],
    what: &'static str,
) -> Result<Vec<T>, MetalTranscodeError> {
    let mut output = try_transcode_vec_with_capacity(N, what)?;
    output.extend(values);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::validate_codeblock_output_host_workspace;
    use crate::MetalTranscodeError;
    use core::mem::size_of;
    use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
    use j2k_transcode::{
        Htj2k97CodeBlockOptions, IrreversibleQuantizationSubbandScales,
        PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component, PrequantizedHtj2k97Resolution,
        PrequantizedHtj2k97Subband,
    };

    fn options() -> Htj2k97CodeBlockOptions {
        Htj2k97CodeBlockOptions {
            bit_depth: 8,
            guard_bits: 1,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            irreversible_quantization_scale: 1.0,
            irreversible_quantization_subband_scales:
                IrreversibleQuantizationSubbandScales::default(),
        }
    }

    #[test]
    fn aggregate_output_metadata_rejects_combined_cap_excess() {
        let per_component = size_of::<i32>()
            + size_of::<PrequantizedHtj2k97CodeBlock>()
            + size_of::<PrequantizedHtj2k97Component>()
            + 2 * size_of::<PrequantizedHtj2k97Resolution>()
            + 4 * size_of::<PrequantizedHtj2k97Subband>();
        let batch_count = DEFAULT_MAX_HOST_ALLOCATION_BYTES / per_component + 1;
        assert!(matches!(
            validate_codeblock_output_host_workspace(1, 1, batch_count, options()),
            Err(MetalTranscodeError::HostAllocationTooLarge {
                what: "prequantized HTJ2K host workspace",
                ..
            })
        ));
    }
}
