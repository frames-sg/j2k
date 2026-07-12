// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    checked_batch_len, private_f32_buffer, u32_param, Buffer, DctGridToDwt97Job,
    DctGridToHtj2k97CodeBlockJob, MetalRuntime, MetalTranscodeError, ProjectionBatchShape,
    METAL_DCT97_UNSUPPORTED_GRID,
};

pub(super) fn dwt97_staged_batch_shape(
    jobs: &[DctGridToDwt97Job<'_>],
    first: &DctGridToDwt97Job<'_>,
) -> Result<ProjectionBatchShape, MetalTranscodeError> {
    staged_batch_shape(
        jobs.len(),
        first.block_cols,
        first.block_rows,
        first.width,
        first.height,
    )
}

pub(super) fn dwt97_codeblock_batch_shape(
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    first: &DctGridToHtj2k97CodeBlockJob<'_>,
) -> Result<ProjectionBatchShape, MetalTranscodeError> {
    staged_batch_shape(
        jobs.len(),
        first.block_cols,
        first.block_rows,
        first.width,
        first.height,
    )
}

fn staged_batch_shape(
    batch_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<ProjectionBatchShape, MetalTranscodeError> {
    let low_width = width.div_ceil(2);
    let high_width = width / 2;
    let low_height = height.div_ceil(2);
    let high_height = height / 2;
    let blocks_per_item =
        block_cols
            .checked_mul(block_rows)
            .ok_or(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID,
            ))?;
    let band_len = |band_width: usize, band_height: usize| {
        band_width
            .checked_mul(band_height)
            .ok_or(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID,
            ))
    };

    Ok(ProjectionBatchShape {
        batch_count,
        batch_count_u32: u32_param(batch_count, METAL_DCT97_UNSUPPORTED_GRID)?,
        width: u32_param(width, METAL_DCT97_UNSUPPORTED_GRID)?,
        height: u32_param(height, METAL_DCT97_UNSUPPORTED_GRID)?,
        block_cols: u32_param(block_cols, METAL_DCT97_UNSUPPORTED_GRID)?,
        blocks_per_item: u32_param(blocks_per_item, METAL_DCT97_UNSUPPORTED_GRID)?,
        low_width,
        low_height,
        high_width,
        high_height,
        ll_len: band_len(low_width, low_height)?,
        hl_len: band_len(high_width, low_height)?,
        lh_len: band_len(low_width, high_height)?,
        hh_len: band_len(high_width, high_height)?,
    })
}

pub(super) struct Dwt97StagedRowBuffers {
    pub(super) low: Buffer,
    pub(super) high: Buffer,
}

pub(super) fn dwt97_staged_row_buffers(
    runtime: &MetalRuntime,
    shape: ProjectionBatchShape,
) -> Result<Dwt97StagedRowBuffers, MetalTranscodeError> {
    let height = shape.height as usize;
    Ok(Dwt97StagedRowBuffers {
        low: row_buffer(runtime, height, shape.low_width, shape.batch_count)?,
        high: row_buffer(runtime, height, shape.high_width, shape.batch_count)?,
    })
}

fn row_buffer(
    runtime: &MetalRuntime,
    height: usize,
    width: usize,
    batch_count: usize,
) -> Result<Buffer, MetalTranscodeError> {
    let item_len = height
        .checked_mul(width)
        .ok_or(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ))?;
    private_f32_buffer(
        &runtime.device,
        checked_batch_len(item_len, batch_count, METAL_DCT97_UNSUPPORTED_GRID)?,
    )
}
