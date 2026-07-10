// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    buffer_with_slice, checked_batch_len, commit_and_wait, dispatch_dct_grid_to_dwt53_with_runtime,
    dispatch_reversible_band, idct_blocks_to_signed_samples_rayon, output_i32_buffer,
    read_i32_buffer, reversible_band_geometry, u32_param, validate_grid,
    validate_reversible_batch_geometry, Buffer, DctGridToDwt53Job, DctGridToReversibleDwt53Job,
    Dwt53TwoDimensional, MetalRuntime, MetalTranscodeError, MetalTranscodeSession,
    ReversibleBatchKernelGeometry, ReversibleDwt53FirstLevel, METAL_DCT53_UNSUPPORTED_GRID,
    METAL_DCT_KERNEL_FAILED, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
};

pub(crate) fn dispatch_dct_grid_to_reversible_dwt53(
    session: &mut MetalTranscodeSession,
    job: DctGridToReversibleDwt53Job<'_>,
) -> Result<ReversibleDwt53FirstLevel, MetalTranscodeError> {
    let mut outputs =
        dispatch_dct_grid_to_reversible_dwt53_batch(session, core::slice::from_ref(&job))?;
    outputs
        .pop()
        .ok_or(MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))
}

pub(crate) fn dispatch_dct_grid_to_reversible_dwt53_batch(
    session: &mut MetalTranscodeSession,
    jobs: &[DctGridToReversibleDwt53Job<'_>],
) -> Result<Vec<ReversibleDwt53FirstLevel>, MetalTranscodeError> {
    let Some(first) = jobs.first() else {
        return Ok(Vec::new());
    };
    validate_reversible_batch_geometry(jobs)?;

    let blocks_per_item = first.block_cols.checked_mul(first.block_rows).ok_or(
        MetalTranscodeError::UnsupportedJob(METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID),
    )?;
    let mut block_samples = Vec::with_capacity(blocks_per_item.saturating_mul(jobs.len()));
    for job in jobs {
        block_samples.extend(idct_blocks_to_signed_samples_rayon(job.dequantized_blocks));
    }

    session.with_runtime(|runtime| {
        dispatch_reversible_dwt53_batch_with_runtime(
            runtime,
            &block_samples,
            jobs.len(),
            first.block_cols,
            first.width,
            first.height,
        )
    })
}

pub(crate) fn dispatch_dct_grid_to_dwt53(
    session: &mut MetalTranscodeSession,
    job: DctGridToDwt53Job<'_>,
) -> Result<Dwt53TwoDimensional<f64>, MetalTranscodeError> {
    validate_grid(
        job.blocks.len(),
        job.block_cols,
        job.block_rows,
        job.width,
        job.height,
        METAL_DCT53_UNSUPPORTED_GRID,
    )?;
    session.with_runtime(|runtime| dispatch_dct_grid_to_dwt53_with_runtime(runtime, job))
}

#[allow(clippy::similar_names)]
pub(super) fn dispatch_reversible_dwt53_batch_with_runtime(
    runtime: &MetalRuntime,
    block_samples: &[[i32; 64]],
    batch_count: usize,
    block_cols: usize,
    width: usize,
    height: usize,
) -> Result<Vec<ReversibleDwt53FirstLevel>, MetalTranscodeError> {
    if batch_count == 0 {
        return Ok(Vec::new());
    }
    if !block_samples.len().is_multiple_of(batch_count) {
        return Err(MetalTranscodeError::UnsupportedJob(
            METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
        ));
    }

    let blocks_per_item = block_samples.len() / batch_count;
    let blocks_per_item_u32 = u32_param(blocks_per_item, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?;
    let batch_count_u32 = u32_param(batch_count, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?;
    let width_u32 = u32_param(width, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?;
    let height_u32 = u32_param(height, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?;
    let block_cols_u32 = u32_param(block_cols, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?;
    let kernel_geometry = ReversibleBatchKernelGeometry {
        width: width_u32,
        height: height_u32,
        block_cols: block_cols_u32,
        blocks_per_item: blocks_per_item_u32,
        batch_count: batch_count_u32,
    };
    let low_width = width.div_ceil(2);
    let high_width = width / 2;
    let low_height = height.div_ceil(2);
    let high_height = height / 2;
    let ll_len = low_width * low_height;
    let hl_len = high_width * low_height;
    let lh_len = low_width * high_height;
    let hh_len = high_width * high_height;
    let output_shape = ReversibleBatchOutputShape {
        low_width,
        low_height,
        high_width,
        high_height,
        ll_len,
        hl_len,
        lh_len,
        hh_len,
        batch_count,
    };
    let blocks = buffer_with_slice(&runtime.device, block_samples);

    let ll_buffer = output_i32_buffer(
        &runtime.device,
        checked_batch_len(ll_len, batch_count, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
    );
    let hl_buffer = output_i32_buffer(
        &runtime.device,
        checked_batch_len(hl_len, batch_count, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
    );
    let lh_buffer = output_i32_buffer(
        &runtime.device,
        checked_batch_len(lh_len, batch_count, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
    );
    let hh_buffer = output_i32_buffer(
        &runtime.device,
        checked_batch_len(hh_len, batch_count, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
    );
    let output_buffers = ReversibleOutputBuffers {
        ll: &ll_buffer,
        hl: &hl_buffer,
        lh: &lh_buffer,
        hh: &hh_buffer,
    };

    let command_buffer = runtime.queue.new_command_buffer();
    command_buffer.set_label("j2k-transcode-metal reversible dct53 projection");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.reversible53_project_band);
    encoder.set_buffer(0, Some(&blocks), 0);

    dispatch_reversible_band(
        encoder,
        &ll_buffer,
        reversible_band_geometry(kernel_geometry, low_width, low_height, ll_len, true, true)?,
    );
    dispatch_reversible_band(
        encoder,
        &hl_buffer,
        reversible_band_geometry(kernel_geometry, high_width, low_height, hl_len, true, false)?,
    );
    dispatch_reversible_band(
        encoder,
        &lh_buffer,
        reversible_band_geometry(kernel_geometry, low_width, high_height, lh_len, false, true)?,
    );
    dispatch_reversible_band(
        encoder,
        &hh_buffer,
        reversible_band_geometry(
            kernel_geometry,
            high_width,
            high_height,
            hh_len,
            false,
            false,
        )?,
    );

    encoder.end_encoding();
    commit_and_wait(command_buffer)
        .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;

    read_reversible_batch_outputs(output_buffers, output_shape)
}

#[derive(Clone, Copy)]
pub(super) struct ReversibleBatchOutputShape {
    pub(super) low_width: usize,
    pub(super) low_height: usize,
    pub(super) high_width: usize,
    pub(super) high_height: usize,
    pub(super) ll_len: usize,
    pub(super) hl_len: usize,
    pub(super) lh_len: usize,
    pub(super) hh_len: usize,
    pub(super) batch_count: usize,
}

#[derive(Clone, Copy)]
pub(super) struct ReversibleOutputBuffers<'a> {
    pub(super) ll: &'a Buffer,
    pub(super) hl: &'a Buffer,
    pub(super) lh: &'a Buffer,
    pub(super) hh: &'a Buffer,
}

pub(super) fn read_reversible_batch_outputs(
    buffers: ReversibleOutputBuffers<'_>,
    shape: ReversibleBatchOutputShape,
) -> Result<Vec<ReversibleDwt53FirstLevel>, MetalTranscodeError> {
    let ll = read_i32_buffer(
        buffers.ll,
        checked_batch_len(
            shape.ll_len,
            shape.batch_count,
            METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
        )?,
    )?;
    let hl = read_i32_buffer(
        buffers.hl,
        checked_batch_len(
            shape.hl_len,
            shape.batch_count,
            METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
        )?,
    )?;
    let lh = read_i32_buffer(
        buffers.lh,
        checked_batch_len(
            shape.lh_len,
            shape.batch_count,
            METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
        )?,
    )?;
    let hh = read_i32_buffer(
        buffers.hh,
        checked_batch_len(
            shape.hh_len,
            shape.batch_count,
            METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
        )?,
    )?;

    let mut outputs = Vec::with_capacity(shape.batch_count);
    for idx in 0..shape.batch_count {
        outputs.push(ReversibleDwt53FirstLevel {
            ll: ll[idx * shape.ll_len..idx * shape.ll_len + shape.ll_len].to_vec(),
            hl: hl[idx * shape.hl_len..idx * shape.hl_len + shape.hl_len].to_vec(),
            lh: lh[idx * shape.lh_len..idx * shape.lh_len + shape.lh_len].to_vec(),
            hh: hh[idx * shape.hh_len..idx * shape.hh_len + shape.hh_len].to_vec(),
            low_width: shape.low_width,
            low_height: shape.low_height,
            high_width: shape.high_width,
            high_height: shape.high_height,
        });
    }

    Ok(outputs)
}
