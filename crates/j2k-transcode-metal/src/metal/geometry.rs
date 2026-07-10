// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    bind_projection_band_buffers, dispatch_projection_threads, htj2k97_subband_delta,
    htj2k97_subband_total_bitplanes, size_of, Buffer, ComputeCommandEncoderRef,
    DctBatchProjectionParams, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob,
    DctGridToReversibleDwt53Job, DctProjectionParams, Htj2k97CodeBlockOptions, J2kSubBandType,
    MetalSparseRow, MetalSparseRows, MetalTranscodeError, MetalWeightTap,
    Reversible53ProjectionParams, SparseWeightRow, METAL_DCT97_UNSUPPORTED_GRID,
    METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
};

#[derive(Clone, Copy)]
pub(super) struct BandGeometry {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) block_cols: u32,
    pub(super) band_width: u32,
    pub(super) band_height: u32,
}

#[derive(Clone, Copy)]
pub(super) struct BatchBandGeometry {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) block_cols: u32,
    pub(super) blocks_per_item: u32,
    pub(super) band_width: u32,
    pub(super) band_height: u32,
    pub(super) output_stride: u32,
    pub(super) batch_count: u32,
}

#[derive(Clone, Copy)]
pub(super) struct ReversibleBandGeometry {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) block_cols: u32,
    pub(super) blocks_per_item: u32,
    pub(super) band_width: u32,
    pub(super) band_height: u32,
    pub(super) output_stride: u32,
    pub(super) batch_count: u32,
    pub(super) vertical_low: bool,
    pub(super) horizontal_low: bool,
}

#[derive(Clone, Copy)]
pub(super) struct ReversibleBatchKernelGeometry {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) block_cols: u32,
    pub(super) blocks_per_item: u32,
    pub(super) batch_count: u32,
}

pub(super) fn reversible_band_geometry(
    base: ReversibleBatchKernelGeometry,
    band_width: usize,
    band_height: usize,
    output_stride: usize,
    vertical_low: bool,
    horizontal_low: bool,
) -> Result<ReversibleBandGeometry, MetalTranscodeError> {
    Ok(ReversibleBandGeometry {
        width: base.width,
        height: base.height,
        block_cols: base.block_cols,
        blocks_per_item: base.blocks_per_item,
        band_width: u32_param(band_width, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
        band_height: u32_param(band_height, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
        output_stride: u32_param(output_stride, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
        batch_count: base.batch_count,
        vertical_low,
        horizontal_low,
    })
}

pub(super) fn dispatch_reversible_band(
    encoder: &ComputeCommandEncoderRef,
    output: &Buffer,
    geometry: ReversibleBandGeometry,
) {
    if geometry.band_width == 0 || geometry.band_height == 0 {
        return;
    }

    let params = Reversible53ProjectionParams {
        width: geometry.width,
        height: geometry.height,
        block_cols: geometry.block_cols,
        blocks_per_item: geometry.blocks_per_item,
        band_width: geometry.band_width,
        band_height: geometry.band_height,
        output_stride: geometry.output_stride,
        vertical_low: u32::from(geometry.vertical_low),
        horizontal_low: u32::from(geometry.horizontal_low),
    };
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes(
        2,
        size_of::<Reversible53ProjectionParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_projection_threads(
        encoder,
        u64::from(geometry.band_width),
        u64::from(geometry.band_height),
        u64::from(geometry.batch_count),
    );
}

pub(super) fn dispatch_band(
    encoder: &ComputeCommandEncoderRef,
    x_weights: (&Buffer, &Buffer),
    y_weights: (&Buffer, &Buffer),
    output: &Buffer,
    geometry: BandGeometry,
) {
    if geometry.band_width == 0 || geometry.band_height == 0 {
        return;
    }

    let params = DctProjectionParams {
        width: geometry.width,
        height: geometry.height,
        block_cols: geometry.block_cols,
        band_width: geometry.band_width,
        band_height: geometry.band_height,
    };
    bind_projection_band_buffers(encoder, x_weights, y_weights, output);
    encoder.set_bytes(
        7,
        size_of::<DctProjectionParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_projection_threads(
        encoder,
        u64::from(geometry.band_width),
        u64::from(geometry.band_height),
        1,
    );
}

pub(super) fn dispatch_band_batch(
    encoder: &ComputeCommandEncoderRef,
    x_weights: (&Buffer, &Buffer),
    y_weights: (&Buffer, &Buffer),
    output: &Buffer,
    geometry: BatchBandGeometry,
) {
    if geometry.band_width == 0 || geometry.band_height == 0 {
        return;
    }

    let params = DctBatchProjectionParams {
        width: geometry.width,
        height: geometry.height,
        block_cols: geometry.block_cols,
        blocks_per_item: geometry.blocks_per_item,
        band_width: geometry.band_width,
        band_height: geometry.band_height,
        output_stride: geometry.output_stride,
    };
    bind_projection_band_buffers(encoder, x_weights, y_weights, output);
    encoder.set_bytes(
        7,
        size_of::<DctBatchProjectionParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_projection_threads(
        encoder,
        u64::from(geometry.band_width),
        u64::from(geometry.band_height),
        u64::from(geometry.batch_count),
    );
}

pub(super) fn validate_grid(
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    unsupported_grid: &'static str,
) -> Result<(), MetalTranscodeError> {
    let expected_blocks = block_cols
        .checked_mul(block_rows)
        .ok_or(MetalTranscodeError::UnsupportedJob(unsupported_grid))?;
    let covered_width = block_cols
        .checked_mul(8)
        .ok_or(MetalTranscodeError::UnsupportedJob(unsupported_grid))?;
    let covered_height = block_rows
        .checked_mul(8)
        .ok_or(MetalTranscodeError::UnsupportedJob(unsupported_grid))?;

    if block_count != expected_blocks
        || width == 0
        || height == 0
        || width > covered_width
        || height > covered_height
    {
        return Err(MetalTranscodeError::UnsupportedJob(unsupported_grid));
    }
    Ok(())
}

pub(super) fn validate_reversible_batch_geometry(
    jobs: &[DctGridToReversibleDwt53Job<'_>],
) -> Result<(), MetalTranscodeError> {
    let Some(first) = jobs.first() else {
        return Ok(());
    };

    for job in jobs {
        validate_grid(
            job.dequantized_blocks.len(),
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
            METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
        )?;

        if job.block_cols != first.block_cols
            || job.block_rows != first.block_rows
            || job.width != first.width
            || job.height != first.height
        {
            return Err(MetalTranscodeError::UnsupportedJob(
                METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
            ));
        }
    }

    Ok(())
}

pub(super) fn validate_dwt97_batch_geometry(
    jobs: &[DctGridToDwt97Job<'_>],
) -> Result<(), MetalTranscodeError> {
    let Some(first) = jobs.first() else {
        return Ok(());
    };

    for job in jobs {
        validate_grid(
            job.blocks.len(),
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
            METAL_DCT97_UNSUPPORTED_GRID,
        )?;

        if job.block_cols != first.block_cols
            || job.block_rows != first.block_rows
            || job.width != first.width
            || job.height != first.height
        {
            return Err(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID,
            ));
        }
    }

    Ok(())
}

pub(super) fn validate_dwt97_codeblock_batch_geometry(
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
) -> Result<(), MetalTranscodeError> {
    let Some(first) = jobs.first() else {
        return Ok(());
    };

    for job in jobs {
        validate_grid(
            job.blocks.len(),
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
            METAL_DCT97_UNSUPPORTED_GRID,
        )?;

        if job.block_cols != first.block_cols
            || job.block_rows != first.block_rows
            || job.width != first.width
            || job.height != first.height
        {
            return Err(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID,
            ));
        }
    }

    Ok(())
}

pub(super) fn validate_htj2k97_codeblock_options(
    options: Htj2k97CodeBlockOptions,
) -> Result<(), MetalTranscodeError> {
    // Shared with CUDA so the two backends accept/reject identical options.
    // Option failures keep their own message instead of the grid-geometry one.
    j2k_transcode::validate_htj2k97_codeblock_options(options)
        .map(|_| ())
        .map_err(MetalTranscodeError::UnsupportedJob)
}

pub(super) fn code_block_len_from_exp(exp: u8) -> Result<usize, MetalTranscodeError> {
    1usize
        .checked_shl(u32::from(exp) + 2)
        .filter(|&value| value > 0)
        .ok_or(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ))
}

pub(super) fn dwt97_total_bitplanes(
    options: Htj2k97CodeBlockOptions,
    sub_band_type: J2kSubBandType,
) -> u8 {
    htj2k97_subband_total_bitplanes(options, sub_band_type)
}

pub(super) fn dwt97_quantize_inv_delta(
    options: Htj2k97CodeBlockOptions,
    sub_band_type: J2kSubBandType,
) -> f32 {
    (1.0 / htj2k97_subband_delta(options, sub_band_type)) as f32
}

pub(super) fn checked_batch_len(
    value_len: usize,
    batch_count: usize,
    unsupported_grid: &'static str,
) -> Result<usize, MetalTranscodeError> {
    value_len
        .checked_mul(batch_count)
        .ok_or(MetalTranscodeError::UnsupportedJob(unsupported_grid))
}

pub(super) fn u32_param(
    value: usize,
    unsupported_grid: &'static str,
) -> Result<u32, MetalTranscodeError> {
    u32::try_from(value).map_err(|_| MetalTranscodeError::UnsupportedJob(unsupported_grid))
}

pub(super) fn metal_sparse_rows(
    rows: &[SparseWeightRow],
    unsupported_grid: &'static str,
) -> Result<MetalSparseRows, MetalTranscodeError> {
    let mut metal_rows = Vec::with_capacity(rows.len());
    let mut taps = Vec::new();
    for row in rows {
        let offset = u32_param(taps.len(), unsupported_grid)?;
        let count = u32_param(row.taps.len(), unsupported_grid)?;
        metal_rows.push(MetalSparseRow { offset, count });
        for tap in &row.taps {
            taps.push(MetalWeightTap {
                sample_idx: u32_param(tap.sample_idx, unsupported_grid)?,
                weight: tap.weight,
            });
        }
    }
    Ok(MetalSparseRows {
        rows: metal_rows,
        taps,
    })
}
