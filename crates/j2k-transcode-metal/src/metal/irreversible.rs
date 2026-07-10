// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_batch_len, code_block_len_from_exp, commit_and_wait,
    dispatch_projected_bands_batch_with_runtime, dispatch_projected_bands_with_runtime,
    dispatch_projection_threads, dwt97_batch_blocks_buffer, dwt97_codeblock_batch_blocks_buffer,
    dwt97_codeblock_output_buffers, dwt97_codeblock_output_transfer_bytes,
    dwt97_codeblock_output_transfer_count, dwt97_quantize_inv_delta, private_f32_buffer,
    projection_batch_output_buffers, projection_batch_output_transfer_bytes,
    projection_batch_output_transfer_count, projection_batch_private_output_buffers,
    read_prequantized_97_codeblock_outputs, read_projected_batch_outputs, size_of,
    staged_threads_per_group, u32_param, validate_dwt97_batch_geometry,
    validate_dwt97_codeblock_batch_geometry, validate_grid, validate_htj2k97_codeblock_options,
    validate_resident_dct_handoffs_for_dwt97_jobs, validate_resident_dct_handoffs_for_htj2k_jobs,
    validate_resident_dwt_handoffs_for_dwt97_jobs, validate_resident_dwt_handoffs_for_htj2k_jobs,
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, Dct97ColumnLiftParams,
    Dct97IdctRowLiftParams, Dct97QuantizeCodeblocksParams, DctGridToDwt53Job, DctGridToDwt97Job,
    DctGridToHtj2k97CodeBlockJob, Dwt53TwoDimensional, Dwt97BatchStageTimings,
    Dwt97CodeBlockOutputBuffers, Dwt97TwoDimensional, Htj2k97CodeBlockOptions, Instant,
    J2kSubBandType, MTLSize, MetalRuntime, MetalTranscodeError, MetalTranscodeSession,
    PrequantizedHtj2k97Component, ProjectionBatchJob, ProjectionBatchOutputBuffers,
    ProjectionBatchShape, ProjectionJob, SparseDwt53WeightRows, SparseDwt97WeightRows,
    DWT97_STAGED_COLUMNS_PER_GROUP, DWT97_STAGED_MAX_AXIS, DWT97_STAGED_ROWS_PER_GROUP,
    METAL_DCT53_UNSUPPORTED_GRID, METAL_DCT97_UNSUPPORTED_GRID, METAL_DCT_KERNEL_FAILED,
};

pub(crate) fn dispatch_dct_grid_to_dwt97(
    session: &mut MetalTranscodeSession,
    job: DctGridToDwt97Job<'_>,
) -> Result<Dwt97TwoDimensional<f64>, MetalTranscodeError> {
    validate_grid(
        job.blocks.len(),
        job.block_cols,
        job.block_rows,
        job.width,
        job.height,
        METAL_DCT97_UNSUPPORTED_GRID,
    )?;
    session.with_runtime(|runtime| dispatch_dct_grid_to_dwt97_with_runtime(runtime, job))
}

pub(crate) fn dispatch_dct_grid_to_dwt97_batch(
    session: &mut MetalTranscodeSession,
    jobs: &[DctGridToDwt97Job<'_>],
) -> Result<(Vec<Dwt97TwoDimensional<f64>>, Dwt97BatchStageTimings), MetalTranscodeError> {
    let Some(first) = jobs.first() else {
        return Ok((Vec::new(), Dwt97BatchStageTimings::default()));
    };
    validate_dwt97_batch_geometry(jobs)?;
    session
        .with_runtime(|runtime| dispatch_dct_grid_to_dwt97_batch_with_runtime(runtime, jobs, first))
}

pub(crate) fn dispatch_dct_grid_to_htj2k97_codeblock_batch(
    session: &mut MetalTranscodeSession,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<PrequantizedHtj2k97Component>, Dwt97BatchStageTimings), MetalTranscodeError> {
    let Some(first) = jobs.first() else {
        return Ok((Vec::new(), Dwt97BatchStageTimings::default()));
    };
    validate_dwt97_codeblock_batch_geometry(jobs)?;
    validate_htj2k97_codeblock_options(options)?;
    session.with_runtime(|runtime| {
        dispatch_dct_grid_to_htj2k97_codeblock_batch_with_runtime(runtime, jobs, first, options)
    })
}

#[allow(clippy::similar_names)]
pub(super) fn dispatch_dct_grid_to_dwt53_with_runtime(
    runtime: &MetalRuntime,
    job: DctGridToDwt53Job<'_>,
) -> Result<Dwt53TwoDimensional<f64>, MetalTranscodeError> {
    let x_weights = SparseDwt53WeightRows::for_len(job.width);
    let y_weights = SparseDwt53WeightRows::for_len(job.height);
    let bands = dispatch_projected_bands_with_runtime(
        runtime,
        ProjectionJob {
            blocks: job.blocks,
            block_cols: job.block_cols,
            width: job.width,
            height: job.height,
            x_low: &x_weights.low,
            x_high: &x_weights.high,
            y_low: &y_weights.low,
            y_high: &y_weights.high,
            unsupported_grid: METAL_DCT53_UNSUPPORTED_GRID,
            label: "j2k-transcode-metal dct53 projection",
        },
    )?;

    Ok(Dwt53TwoDimensional {
        ll: bands.ll,
        hl: bands.hl,
        lh: bands.lh,
        hh: bands.hh,
        low_width: bands.low_width,
        low_height: bands.low_height,
        high_width: bands.high_width,
        high_height: bands.high_height,
    })
}

#[allow(clippy::similar_names)]
pub(super) fn dispatch_dct_grid_to_dwt97_with_runtime(
    runtime: &MetalRuntime,
    job: DctGridToDwt97Job<'_>,
) -> Result<Dwt97TwoDimensional<f64>, MetalTranscodeError> {
    let x_weights = SparseDwt97WeightRows::for_len(job.width);
    let y_weights = SparseDwt97WeightRows::for_len(job.height);
    let bands = dispatch_projected_bands_with_runtime(
        runtime,
        ProjectionJob {
            blocks: job.blocks,
            block_cols: job.block_cols,
            width: job.width,
            height: job.height,
            x_low: &x_weights.low,
            x_high: &x_weights.high,
            y_low: &y_weights.low,
            y_high: &y_weights.high,
            unsupported_grid: METAL_DCT97_UNSUPPORTED_GRID,
            label: "j2k-transcode-metal dct97 projection",
        },
    )?;

    Ok(Dwt97TwoDimensional {
        ll: bands.ll,
        hl: bands.hl,
        lh: bands.lh,
        hh: bands.hh,
        low_width: bands.low_width,
        low_height: bands.low_height,
        high_width: bands.high_width,
        high_height: bands.high_height,
    })
}

#[allow(clippy::similar_names)]
pub(super) fn dispatch_dct_grid_to_dwt97_batch_with_runtime(
    runtime: &MetalRuntime,
    jobs: &[DctGridToDwt97Job<'_>],
    first: &DctGridToDwt97Job<'_>,
) -> Result<(Vec<Dwt97TwoDimensional<f64>>, Dwt97BatchStageTimings), MetalTranscodeError> {
    if staged_dwt97_batch_supported(first) {
        return dispatch_dct_grid_to_dwt97_batch_staged_with_runtime(runtime, jobs, first);
    }

    let x_weights = SparseDwt97WeightRows::for_len(first.width);
    let y_weights = SparseDwt97WeightRows::for_len(first.height);
    let bands = dispatch_projected_bands_batch_with_runtime(
        runtime,
        ProjectionBatchJob {
            jobs,
            block_cols: first.block_cols,
            block_rows: first.block_rows,
            width: first.width,
            height: first.height,
            x_low: &x_weights.low,
            x_high: &x_weights.high,
            y_low: &y_weights.low,
            y_high: &y_weights.high,
            unsupported_grid: METAL_DCT97_UNSUPPORTED_GRID,
            label: "j2k-transcode-metal batched dct97 projection",
        },
    )?;

    Ok((
        bands
            .into_iter()
            .map(|bands| Dwt97TwoDimensional {
                ll: bands.ll,
                hl: bands.hl,
                lh: bands.lh,
                hh: bands.hh,
                low_width: bands.low_width,
                low_height: bands.low_height,
                high_width: bands.high_width,
                high_height: bands.high_height,
            })
            .collect(),
        Dwt97BatchStageTimings::default(),
    ))
}

pub(super) fn staged_dwt97_batch_supported(first: &DctGridToDwt97Job<'_>) -> bool {
    first.width <= DWT97_STAGED_MAX_AXIS && first.height <= DWT97_STAGED_MAX_AXIS
}

pub(super) fn staged_dwt97_codeblock_batch_supported(
    first: &DctGridToHtj2k97CodeBlockJob<'_>,
) -> bool {
    first.width <= DWT97_STAGED_MAX_AXIS && first.height <= DWT97_STAGED_MAX_AXIS
}

pub(super) fn dispatch_dct_grid_to_dwt97_batch_staged_with_runtime(
    runtime: &MetalRuntime,
    jobs: &[DctGridToDwt97Job<'_>],
    first: &DctGridToDwt97Job<'_>,
) -> Result<(Vec<Dwt97TwoDimensional<f64>>, Dwt97BatchStageTimings), MetalTranscodeError> {
    let shape = dwt97_staged_batch_shape(jobs, first)?;
    let mut timings = Dwt97BatchStageTimings::default();

    let pack_upload_start = Instant::now();
    let blocks = dwt97_batch_blocks_buffer(&runtime.device, jobs)?;
    let row_buffers = dwt97_staged_row_buffers(runtime, shape)?;
    let output_buffers =
        projection_batch_output_buffers(runtime, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    timings.pack_upload_us = pack_upload_start.elapsed().as_micros();
    timings.pack_upload_transfers = usize::from(blocks.length() > 0);
    timings.pack_upload_bytes = blocks.length();
    timings.resident_dct_handoff_count =
        validate_resident_dct_handoffs_for_dwt97_jobs(&blocks, jobs)?;
    timings.resident_dwt_handoff_count =
        validate_resident_dwt_handoffs_for_dwt97_jobs(&output_buffers, jobs, shape)?;

    let command_buffer = runtime.queue.new_command_buffer();
    command_buffer.set_label("j2k-transcode-metal dct97 staged lift batch");
    let row_start = Instant::now();
    encode_dwt97_staged_row_lift(
        command_buffer,
        runtime,
        first.height,
        shape,
        &blocks,
        &row_buffers,
    )?;
    timings.idct_row_lift_us = row_start.elapsed().as_micros();

    let column_start = Instant::now();
    encode_dwt97_staged_column_lift(
        command_buffer,
        runtime,
        shape,
        &row_buffers,
        &output_buffers,
    )?;
    commit_and_wait(command_buffer)
        .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
    timings.column_lift_us = column_start.elapsed().as_micros();

    let readback_start = Instant::now();
    let bands = read_projected_batch_outputs(&output_buffers, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    timings.readback_us = readback_start.elapsed().as_micros();
    timings.readback_transfers = projection_batch_output_transfer_count(&output_buffers);
    timings.readback_bytes = projection_batch_output_transfer_bytes(&output_buffers);

    Ok((
        bands
            .into_iter()
            .map(|bands| Dwt97TwoDimensional {
                ll: bands.ll,
                hl: bands.hl,
                lh: bands.lh,
                hh: bands.hh,
                low_width: bands.low_width,
                low_height: bands.low_height,
                high_width: bands.high_width,
                high_height: bands.high_height,
            })
            .collect(),
        timings,
    ))
}

pub(super) fn dispatch_dct_grid_to_htj2k97_codeblock_batch_with_runtime(
    runtime: &MetalRuntime,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    first: &DctGridToHtj2k97CodeBlockJob<'_>,
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<PrequantizedHtj2k97Component>, Dwt97BatchStageTimings), MetalTranscodeError> {
    if !staged_dwt97_codeblock_batch_supported(first) {
        return Err(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ));
    }

    let shape = dwt97_codeblock_batch_shape(jobs, first)?;
    let mut timings = Dwt97BatchStageTimings::default();

    let pack_upload_start = Instant::now();
    let blocks = dwt97_codeblock_batch_blocks_buffer(&runtime.device, jobs)?;
    let row_buffers = dwt97_staged_row_buffers(runtime, shape)?;
    let band_buffers =
        projection_batch_private_output_buffers(runtime, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    let codeblock_buffers =
        dwt97_codeblock_output_buffers(runtime, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    timings.pack_upload_us = pack_upload_start.elapsed().as_micros();
    timings.pack_upload_transfers = usize::from(blocks.length() > 0);
    timings.pack_upload_bytes = blocks.length();
    timings.resident_dct_handoff_count =
        validate_resident_dct_handoffs_for_htj2k_jobs(&blocks, jobs)?;
    timings.resident_dwt_handoff_count =
        validate_resident_dwt_handoffs_for_htj2k_jobs(&band_buffers, jobs, shape)?;

    let command_buffer = runtime.queue.new_command_buffer();
    command_buffer.set_label("j2k-transcode-metal dct97 codeblock pipeline batch");
    let row_start = Instant::now();
    encode_dwt97_staged_row_lift(
        command_buffer,
        runtime,
        first.height,
        shape,
        &blocks,
        &row_buffers,
    )?;
    timings.idct_row_lift_us = row_start.elapsed().as_micros();

    let column_start = Instant::now();
    encode_dwt97_staged_column_lift(command_buffer, runtime, shape, &row_buffers, &band_buffers)?;
    timings.column_lift_us = column_start.elapsed().as_micros();

    let quantize_start = Instant::now();
    encode_dwt97_quantize_codeblocks(
        command_buffer,
        runtime,
        shape,
        options,
        &band_buffers,
        &codeblock_buffers,
    )?;
    commit_and_wait(command_buffer)
        .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
    timings.quantize_codeblock_us = quantize_start.elapsed().as_micros();

    let readback_start = Instant::now();
    let components = read_prequantized_97_codeblock_outputs(
        &codeblock_buffers,
        jobs,
        shape,
        options,
        METAL_DCT97_UNSUPPORTED_GRID,
    )?;
    timings.readback_us = readback_start.elapsed().as_micros();
    timings.readback_transfers = dwt97_codeblock_output_transfer_count(&codeblock_buffers);
    timings.readback_bytes = dwt97_codeblock_output_transfer_bytes(&codeblock_buffers);

    Ok((components, timings))
}

pub(super) fn dwt97_staged_batch_shape(
    jobs: &[DctGridToDwt97Job<'_>],
    first: &DctGridToDwt97Job<'_>,
) -> Result<ProjectionBatchShape, MetalTranscodeError> {
    let low_width = first.width.div_ceil(2);
    let high_width = first.width / 2;
    let low_height = first.height.div_ceil(2);
    let high_height = first.height / 2;
    let blocks_per_item = first.block_cols.checked_mul(first.block_rows).ok_or(
        MetalTranscodeError::UnsupportedJob(METAL_DCT97_UNSUPPORTED_GRID),
    )?;

    Ok(ProjectionBatchShape {
        batch_count: jobs.len(),
        batch_count_u32: u32_param(jobs.len(), METAL_DCT97_UNSUPPORTED_GRID)?,
        width: u32_param(first.width, METAL_DCT97_UNSUPPORTED_GRID)?,
        height: u32_param(first.height, METAL_DCT97_UNSUPPORTED_GRID)?,
        block_cols: u32_param(first.block_cols, METAL_DCT97_UNSUPPORTED_GRID)?,
        blocks_per_item: u32_param(blocks_per_item, METAL_DCT97_UNSUPPORTED_GRID)?,
        low_width,
        low_height,
        high_width,
        high_height,
        ll_len: low_width * low_height,
        hl_len: high_width * low_height,
        lh_len: low_width * high_height,
        hh_len: high_width * high_height,
    })
}

pub(super) fn dwt97_codeblock_batch_shape(
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    first: &DctGridToHtj2k97CodeBlockJob<'_>,
) -> Result<ProjectionBatchShape, MetalTranscodeError> {
    let low_width = first.width.div_ceil(2);
    let high_width = first.width / 2;
    let low_height = first.height.div_ceil(2);
    let high_height = first.height / 2;
    let blocks_per_item = first.block_cols.checked_mul(first.block_rows).ok_or(
        MetalTranscodeError::UnsupportedJob(METAL_DCT97_UNSUPPORTED_GRID),
    )?;

    Ok(ProjectionBatchShape {
        batch_count: jobs.len(),
        batch_count_u32: u32_param(jobs.len(), METAL_DCT97_UNSUPPORTED_GRID)?,
        width: u32_param(first.width, METAL_DCT97_UNSUPPORTED_GRID)?,
        height: u32_param(first.height, METAL_DCT97_UNSUPPORTED_GRID)?,
        block_cols: u32_param(first.block_cols, METAL_DCT97_UNSUPPORTED_GRID)?,
        blocks_per_item: u32_param(blocks_per_item, METAL_DCT97_UNSUPPORTED_GRID)?,
        low_width,
        low_height,
        high_width,
        high_height,
        ll_len: low_width * low_height,
        hl_len: high_width * low_height,
        lh_len: low_width * high_height,
        hh_len: high_width * high_height,
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
        low: private_f32_buffer(
            &runtime.device,
            checked_batch_len(
                height * shape.low_width,
                shape.batch_count,
                METAL_DCT97_UNSUPPORTED_GRID,
            )?,
        ),
        high: private_f32_buffer(
            &runtime.device,
            checked_batch_len(
                height * shape.high_width,
                shape.batch_count,
                METAL_DCT97_UNSUPPORTED_GRID,
            )?,
        ),
    })
}

pub(super) fn encode_dwt97_staged_row_lift(
    command_buffer: &CommandBufferRef,
    runtime: &MetalRuntime,
    height: usize,
    shape: ProjectionBatchShape,
    blocks: &Buffer,
    row_buffers: &Dwt97StagedRowBuffers,
) -> Result<(), MetalTranscodeError> {
    let params = Dct97IdctRowLiftParams {
        width: shape.width,
        height: shape.height,
        block_cols: shape.block_cols,
        blocks_per_item: shape.blocks_per_item,
        low_width: u32_param(shape.low_width, METAL_DCT97_UNSUPPORTED_GRID)?,
        high_width: u32_param(shape.high_width, METAL_DCT97_UNSUPPORTED_GRID)?,
    };
    let row_groups = height.div_ceil(DWT97_STAGED_ROWS_PER_GROUP);

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.dct97_idct_row_lift_batch);
    encoder.set_buffer(0, Some(blocks), 0);
    encoder.set_buffer(1, Some(&runtime.idct_basis), 0);
    encoder.set_buffer(2, Some(&row_buffers.low), 0);
    encoder.set_buffer(3, Some(&row_buffers.high), 0);
    encoder.set_bytes(
        4,
        size_of::<Dct97IdctRowLiftParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: row_groups as u64,
            height: u64::from(shape.batch_count_u32),
            depth: 1,
        },
        staged_threads_per_group(),
    );
    encoder.end_encoding();
    Ok(())
}

pub(super) fn encode_dwt97_staged_column_lift(
    command_buffer: &CommandBufferRef,
    runtime: &MetalRuntime,
    shape: ProjectionBatchShape,
    row_buffers: &Dwt97StagedRowBuffers,
    output_buffers: &ProjectionBatchOutputBuffers,
) -> Result<(), MetalTranscodeError> {
    let row_low_stride = (shape.height as usize).checked_mul(shape.low_width).ok_or(
        MetalTranscodeError::UnsupportedJob(METAL_DCT97_UNSUPPORTED_GRID),
    )?;
    let row_high_stride = (shape.height as usize)
        .checked_mul(shape.high_width)
        .ok_or(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ))?;
    let params = Dct97ColumnLiftParams {
        height: shape.height,
        low_width: u32_param(shape.low_width, METAL_DCT97_UNSUPPORTED_GRID)?,
        high_width: u32_param(shape.high_width, METAL_DCT97_UNSUPPORTED_GRID)?,
        low_height: u32_param(shape.low_height, METAL_DCT97_UNSUPPORTED_GRID)?,
        high_height: u32_param(shape.high_height, METAL_DCT97_UNSUPPORTED_GRID)?,
        row_low_stride: u32_param(row_low_stride, METAL_DCT97_UNSUPPORTED_GRID)?,
        row_high_stride: u32_param(row_high_stride, METAL_DCT97_UNSUPPORTED_GRID)?,
        ll_stride: u32_param(shape.ll_len, METAL_DCT97_UNSUPPORTED_GRID)?,
        hl_stride: u32_param(shape.hl_len, METAL_DCT97_UNSUPPORTED_GRID)?,
        lh_stride: u32_param(shape.lh_len, METAL_DCT97_UNSUPPORTED_GRID)?,
        hh_stride: u32_param(shape.hh_len, METAL_DCT97_UNSUPPORTED_GRID)?,
    };
    let column_groups = shape
        .low_width
        .max(shape.high_width)
        .div_ceil(DWT97_STAGED_COLUMNS_PER_GROUP);

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.dct97_column_lift_batch);
    encoder.set_buffer(0, Some(&row_buffers.low), 0);
    encoder.set_buffer(1, Some(&row_buffers.high), 0);
    encoder.set_buffer(2, Some(&output_buffers.ll), 0);
    encoder.set_buffer(3, Some(&output_buffers.hl), 0);
    encoder.set_buffer(4, Some(&output_buffers.lh), 0);
    encoder.set_buffer(5, Some(&output_buffers.hh), 0);
    encoder.set_bytes(
        6,
        size_of::<Dct97ColumnLiftParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: column_groups as u64,
            height: u64::from(shape.batch_count_u32),
            depth: 2,
        },
        staged_threads_per_group(),
    );
    encoder.end_encoding();
    Ok(())
}

pub(super) fn encode_dwt97_quantize_codeblocks(
    command_buffer: &CommandBufferRef,
    runtime: &MetalRuntime,
    shape: ProjectionBatchShape,
    options: Htj2k97CodeBlockOptions,
    band_buffers: &ProjectionBatchOutputBuffers,
    codeblock_buffers: &Dwt97CodeBlockOutputBuffers,
) -> Result<(), MetalTranscodeError> {
    let cb_width = code_block_len_from_exp(options.code_block_width_exp)?;
    let cb_height = code_block_len_from_exp(options.code_block_height_exp)?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.dct97_quantize_codeblocks_batch);
    dispatch_dwt97_quantize_codeblock_band(
        encoder,
        &band_buffers.ll,
        &codeblock_buffers.ll,
        Dwt97QuantizeBand {
            width: shape.low_width,
            height: shape.low_height,
            stride: shape.ll_len,
            cb_width,
            cb_height,
            inv_delta: dwt97_quantize_inv_delta(options, J2kSubBandType::LowLow),
            batch_count: shape.batch_count_u32,
        },
    )?;
    dispatch_dwt97_quantize_codeblock_band(
        encoder,
        &band_buffers.hl,
        &codeblock_buffers.hl,
        Dwt97QuantizeBand {
            width: shape.high_width,
            height: shape.low_height,
            stride: shape.hl_len,
            cb_width,
            cb_height,
            inv_delta: dwt97_quantize_inv_delta(options, J2kSubBandType::HighLow),
            batch_count: shape.batch_count_u32,
        },
    )?;
    dispatch_dwt97_quantize_codeblock_band(
        encoder,
        &band_buffers.lh,
        &codeblock_buffers.lh,
        Dwt97QuantizeBand {
            width: shape.low_width,
            height: shape.high_height,
            stride: shape.lh_len,
            cb_width,
            cb_height,
            inv_delta: dwt97_quantize_inv_delta(options, J2kSubBandType::LowHigh),
            batch_count: shape.batch_count_u32,
        },
    )?;
    dispatch_dwt97_quantize_codeblock_band(
        encoder,
        &band_buffers.hh,
        &codeblock_buffers.hh,
        Dwt97QuantizeBand {
            width: shape.high_width,
            height: shape.high_height,
            stride: shape.hh_len,
            cb_width,
            cb_height,
            inv_delta: dwt97_quantize_inv_delta(options, J2kSubBandType::HighHigh),
            batch_count: shape.batch_count_u32,
        },
    )?;
    encoder.end_encoding();
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct Dwt97QuantizeBand {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) stride: usize,
    pub(super) cb_width: usize,
    pub(super) cb_height: usize,
    pub(super) inv_delta: f32,
    pub(super) batch_count: u32,
}

pub(super) fn dispatch_dwt97_quantize_codeblock_band(
    encoder: &ComputeCommandEncoderRef,
    band_buffer: &Buffer,
    codeblock_buffer: &Buffer,
    band: Dwt97QuantizeBand,
) -> Result<(), MetalTranscodeError> {
    if band.width == 0 || band.height == 0 {
        return Ok(());
    }
    let params = Dct97QuantizeCodeblocksParams {
        band_width: u32_param(band.width, METAL_DCT97_UNSUPPORTED_GRID)?,
        band_height: u32_param(band.height, METAL_DCT97_UNSUPPORTED_GRID)?,
        output_stride: u32_param(band.stride, METAL_DCT97_UNSUPPORTED_GRID)?,
        code_block_width: u32_param(band.cb_width, METAL_DCT97_UNSUPPORTED_GRID)?,
        code_block_height: u32_param(band.cb_height, METAL_DCT97_UNSUPPORTED_GRID)?,
        inv_delta: band.inv_delta,
    };
    encoder.set_buffer(0, Some(band_buffer), 0);
    encoder.set_buffer(1, Some(codeblock_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<Dct97QuantizeCodeblocksParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_projection_threads(
        encoder,
        band.width as u64,
        band.height as u64,
        u64::from(band.batch_count),
    );
    Ok(())
}
