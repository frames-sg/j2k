// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_batch_len, checked_command_buffer, checked_compute_command_encoder,
    code_block_len_from_exp, commit_and_wait, dispatch_projected_bands_with_runtime,
    dispatch_projection_threads, dwt97_batch_blocks_buffer, dwt97_codeblock_batch_blocks_buffer,
    dwt97_codeblock_output_buffers, dwt97_codeblock_output_transfer_bytes,
    dwt97_codeblock_output_transfer_count, dwt97_quantize_inv_delta, private_f32_buffer,
    projection_batch_output_buffers, projection_batch_output_transfer_bytes,
    projection_batch_output_transfer_count, projection_batch_private_output_buffers,
    read_prequantized_97_codeblock_outputs, read_projected_batch_outputs, staged_threads_per_group,
    try_transcode_vec_with_capacity, u32_param, validate_codeblock_projection_allocations,
    validate_dwt97_batch_geometry, validate_dwt97_codeblock_batch_geometry,
    validate_float_projection_allocations, validate_grid, validate_htj2k97_codeblock_options,
    validate_resident_dct_handoffs_for_dwt97_jobs, validate_resident_dct_handoffs_for_htj2k_jobs,
    validate_resident_dwt_handoffs_for_dwt97_jobs, validate_resident_dwt_handoffs_for_htj2k_jobs,
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState,
    Dct97ColumnLiftParams, Dct97IdctRowLiftParams, Dct97PlaneParams, Dct97QuantizeCodeblocksParams,
    DctGridToDwt53Job, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob, Dwt53TwoDimensional,
    Dwt97BatchStageTimings, Dwt97CodeBlockOutputBuffers, Dwt97TwoDimensional,
    Htj2k97CodeBlockOptions, Instant, J2kSubBandType, MetalRuntime, MetalTranscodeError,
    MetalTranscodeSession, PrequantizedHtj2k97Component, ProjectionBatchOutputBuffers,
    ProjectionBatchShape, ProjectionJob, SparseDwt53WeightRows, TranscodeComputeEncoderExt,
    DWT97_STAGED_COLUMNS_PER_GROUP, DWT97_STAGED_MAX_AXIS, DWT97_STAGED_ROWS_PER_GROUP,
    METAL_DCT53_UNSUPPORTED_GRID, METAL_DCT97_UNSUPPORTED_GRID,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLComputeCommandEncoder as _,
};

mod staging;
use staging::{
    dwt97_codeblock_batch_shape, dwt97_staged_batch_shape, dwt97_staged_row_buffers,
    Dwt97StagedRowBuffers,
};

/// A single 9/7 job runs as a batch of one through the canonical kernels.
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
    let (outputs, _timings) = dispatch_dct_grid_to_dwt97_batch(session, &[job])?;
    outputs
        .into_iter()
        .next()
        .ok_or(MetalTranscodeError::Kernel(
            "Metal 9/7 batch of one returned no output",
        ))
}

pub(crate) fn dispatch_dct_grid_to_dwt97_batch(
    session: &mut MetalTranscodeSession,
    jobs: &[DctGridToDwt97Job<'_>],
) -> Result<(Vec<Dwt97TwoDimensional<f64>>, Dwt97BatchStageTimings), MetalTranscodeError> {
    let Some(first) = jobs.first() else {
        return Ok((Vec::new(), Dwt97BatchStageTimings::default()));
    };
    validate_dwt97_batch_geometry(jobs)?;
    let total_blocks = first.blocks.len().checked_mul(jobs.len()).ok_or(
        MetalTranscodeError::HostAllocationTooLarge {
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what: "batched DCT coefficient upload",
        },
    )?;
    // Both routes hold one intermediate plane (row bands or the interleaved
    // plane) beside the output bands.
    validate_float_projection_allocations(
        total_blocks,
        first.width,
        first.height,
        jobs.len(),
        0,
        0,
        2,
    )?;
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
    let total_blocks = first.blocks.len().checked_mul(jobs.len()).ok_or(
        MetalTranscodeError::HostAllocationTooLarge {
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what: "batched HTJ2K DCT coefficient upload",
        },
    )?;
    validate_codeblock_projection_allocations(
        total_blocks,
        first.width,
        first.height,
        jobs.len(),
        options,
    )?;
    session.with_runtime(|runtime| {
        dispatch_dct_grid_to_htj2k97_codeblock_batch_with_runtime(runtime, jobs, first, options)
    })
}

pub(super) fn dispatch_dct_grid_to_dwt53_with_runtime(
    runtime: &MetalRuntime,
    job: DctGridToDwt53Job<'_>,
    x_weights: &SparseDwt53WeightRows,
    y_weights: &SparseDwt53WeightRows,
) -> Result<Dwt53TwoDimensional<f64>, MetalTranscodeError> {
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

/// Whether both axes fit the threadgroup-staged kernels; larger components
/// use the global-memory plane kernels, which compute the same bits.
fn staged_dwt97_supported(width: usize, height: usize) -> bool {
    width <= DWT97_STAGED_MAX_AXIS && height <= DWT97_STAGED_MAX_AXIS
}

/// Intermediate storage of one canonical 9/7 batch transform.
enum Dwt97Workspace {
    /// Deinterleaved row bands for the threadgroup-staged kernels.
    Staged(Dwt97StagedRowBuffers),
    /// One interleaved plane per item for the global-memory kernels.
    Plane(Buffer),
}

impl Dwt97Workspace {
    fn new(
        runtime: &MetalRuntime,
        shape: ProjectionBatchShape,
    ) -> Result<Self, MetalTranscodeError> {
        if staged_dwt97_supported(shape.width as usize, shape.height as usize) {
            return dwt97_staged_row_buffers(runtime, shape).map(Self::Staged);
        }
        let plane_len = (shape.width as usize)
            .checked_mul(shape.height as usize)
            .ok_or(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID,
            ))?;
        let total = checked_batch_len(plane_len, shape.batch_count, METAL_DCT97_UNSUPPORTED_GRID)?;
        // The plane kernels index the whole batch with 32-bit offsets.
        u32_param(total, METAL_DCT97_UNSUPPORTED_GRID)?;
        private_f32_buffer(&runtime.device, total).map(Self::Plane)
    }

    /// Encode the canonical IDCT and both lifting passes into `bands`.
    /// Returns the encode times of the IDCT-plus-row stage and the column
    /// stage.
    fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        runtime: &MetalRuntime,
        shape: ProjectionBatchShape,
        blocks: &Buffer,
        bands: &ProjectionBatchOutputBuffers,
    ) -> Result<(u128, u128), MetalTranscodeError> {
        let row_start = Instant::now();
        match self {
            Self::Staged(rows) => {
                encode_dwt97_staged_row_lift(command_buffer, runtime, shape, blocks, rows)?;
                let row_us = row_start.elapsed().as_micros();
                let column_start = Instant::now();
                encode_dwt97_staged_column_lift(command_buffer, runtime, shape, rows, bands)?;
                Ok((row_us, column_start.elapsed().as_micros()))
            }
            Self::Plane(plane) => {
                let params = dwt97_plane_params(shape)?;
                encode_dwt97_plane_pass(
                    command_buffer,
                    &runtime.dct97_idct_plane_batch,
                    &[blocks, plane],
                    params,
                    (u64::from(params.width), u64::from(params.height)),
                    u64::from(shape.batch_count_u32),
                )?;
                encode_dwt97_plane_lift(command_buffer, runtime, plane, params, false, shape)?;
                encode_dwt97_plane_pass(
                    command_buffer,
                    &runtime.dct97_scale_plane_rows_batch,
                    &[plane],
                    params,
                    (u64::from(params.width), u64::from(params.height)),
                    u64::from(shape.batch_count_u32),
                )?;
                let row_us = row_start.elapsed().as_micros();
                let column_start = Instant::now();
                encode_dwt97_plane_lift(command_buffer, runtime, plane, params, true, shape)?;
                encode_dwt97_plane_pass(
                    command_buffer,
                    &runtime.dct97_split_plane_batch,
                    &[plane, &bands.ll, &bands.hl, &bands.lh, &bands.hh],
                    params,
                    (u64::from(params.width), u64::from(params.height)),
                    u64::from(shape.batch_count_u32),
                )?;
                Ok((row_us, column_start.elapsed().as_micros()))
            }
        }
    }
}

fn dwt97_plane_params(
    shape: ProjectionBatchShape,
) -> Result<Dct97PlaneParams, MetalTranscodeError> {
    let param = |value: usize| u32_param(value, METAL_DCT97_UNSUPPORTED_GRID);
    Ok(Dct97PlaneParams {
        width: shape.width,
        height: shape.height,
        block_cols: shape.block_cols,
        blocks_per_item: shape.blocks_per_item,
        low_width: param(shape.low_width)?,
        high_width: param(shape.high_width)?,
        low_height: param(shape.low_height)?,
        high_height: param(shape.high_height)?,
        ll_stride: param(shape.ll_len)?,
        hl_stride: param(shape.hl_len)?,
        lh_stride: param(shape.lh_len)?,
        hh_stride: param(shape.hh_len)?,
        vertical: 0,
        step: 0,
    })
}

/// Encode the four in-place lifting steps along rows or columns.
fn encode_dwt97_plane_lift(
    command_buffer: &CommandBufferRef,
    runtime: &MetalRuntime,
    plane: &Buffer,
    params: Dct97PlaneParams,
    vertical: bool,
    shape: ProjectionBatchShape,
) -> Result<(), MetalTranscodeError> {
    let (width, height) = (u64::from(params.width), u64::from(params.height));
    let grid = if vertical {
        (width, height.div_ceil(2))
    } else {
        (width.div_ceil(2), height)
    };
    for step in 0..4 {
        encode_dwt97_plane_pass(
            command_buffer,
            &runtime.dct97_lift_plane_step_batch,
            &[plane],
            Dct97PlaneParams {
                vertical: u32::from(vertical),
                step,
                ..params
            },
            grid,
            u64::from(shape.batch_count_u32),
        )?;
    }
    Ok(())
}

/// Encode one plane kernel in its own compute encoder, so every pass sees
/// the completed writes of the previous one.
fn encode_dwt97_plane_pass(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    params: Dct97PlaneParams,
    (width, height): (u64, u64),
    batch_count: u64,
) -> Result<(), MetalTranscodeError> {
    let encoder = checked_compute_command_encoder(command_buffer).map_err(|error| {
        MetalTranscodeError::support("Metal 9/7 plane compute encoder creation", error)
    })?;
    encoder.setComputePipelineState(pipeline);
    let mut index = 0;
    for buffer in buffers {
        encoder.set_buffer(index, Some(buffer), 0)?;
        index += 1;
    }
    encoder.set_bytes::<Dct97PlaneParams>(index, &params)?;
    dispatch_projection_threads(&encoder, width, height, batch_count);
    encoder.endEncoding();
    Ok(())
}

pub(super) fn dispatch_dct_grid_to_dwt97_batch_with_runtime(
    runtime: &MetalRuntime,
    jobs: &[DctGridToDwt97Job<'_>],
    first: &DctGridToDwt97Job<'_>,
) -> Result<(Vec<Dwt97TwoDimensional<f64>>, Dwt97BatchStageTimings), MetalTranscodeError> {
    let shape = dwt97_staged_batch_shape(jobs, first)?;
    let mut timings = Dwt97BatchStageTimings::default();

    let pack_upload_start = Instant::now();
    let blocks = dwt97_batch_blocks_buffer(&runtime.device, jobs)?;
    let workspace = Dwt97Workspace::new(runtime, shape)?;
    let output_buffers =
        projection_batch_output_buffers(runtime, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    timings.pack_upload_us = pack_upload_start.elapsed().as_micros();
    timings.pack_upload_transfers = usize::from(blocks.length() > 0);
    timings.pack_upload_bytes = u64::try_from(blocks.length())
        .map_err(|_| MetalTranscodeError::Kernel("Metal 9/7 upload byte count does not fit u64"))?;
    timings.resident_dct_handoff_count =
        validate_resident_dct_handoffs_for_dwt97_jobs(&blocks, jobs)?;
    timings.resident_dwt_handoff_count =
        validate_resident_dwt_handoffs_for_dwt97_jobs(&output_buffers, jobs, shape)?;

    let command_buffer = checked_command_buffer(&runtime.queue).map_err(|error| {
        MetalTranscodeError::support("Metal 9/7 command buffer creation", error)
    })?;
    command_buffer.setLabel(Some(&NSString::from_str(
        "j2k-transcode-metal dct97 canonical lift batch",
    )));
    let (row_us, column_encode_us) =
        workspace.encode(&command_buffer, runtime, shape, &blocks, &output_buffers)?;
    timings.idct_row_lift_us = row_us;

    let wait_start = Instant::now();
    commit_and_wait(&command_buffer)
        .map_err(|error| MetalTranscodeError::support("Metal 9/7 command buffer", error))?;
    timings.column_lift_us = column_encode_us + wait_start.elapsed().as_micros();

    let readback_start = Instant::now();
    let bands = read_projected_batch_outputs(&output_buffers, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    timings.readback_us = readback_start.elapsed().as_micros();
    timings.readback_transfers = projection_batch_output_transfer_count(&output_buffers);
    timings.readback_bytes = projection_batch_output_transfer_bytes(&output_buffers);

    Ok((dwt97_outputs_from_projected_bands(bands)?, timings))
}

pub(super) fn dispatch_dct_grid_to_htj2k97_codeblock_batch_with_runtime(
    runtime: &MetalRuntime,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    first: &DctGridToHtj2k97CodeBlockJob<'_>,
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<PrequantizedHtj2k97Component>, Dwt97BatchStageTimings), MetalTranscodeError> {
    let shape = dwt97_codeblock_batch_shape(jobs, first)?;
    let mut timings = Dwt97BatchStageTimings::default();

    let pack_upload_start = Instant::now();
    let blocks = dwt97_codeblock_batch_blocks_buffer(&runtime.device, jobs)?;
    let workspace = Dwt97Workspace::new(runtime, shape)?;
    let band_buffers =
        projection_batch_private_output_buffers(runtime, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    let codeblock_buffers =
        dwt97_codeblock_output_buffers(runtime, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    timings.pack_upload_us = pack_upload_start.elapsed().as_micros();
    timings.pack_upload_transfers = usize::from(blocks.length() > 0);
    timings.pack_upload_bytes = u64::try_from(blocks.length()).map_err(|_| {
        MetalTranscodeError::Kernel("Metal staged HTJ2K upload byte count does not fit u64")
    })?;
    timings.resident_dct_handoff_count =
        validate_resident_dct_handoffs_for_htj2k_jobs(&blocks, jobs)?;
    timings.resident_dwt_handoff_count =
        validate_resident_dwt_handoffs_for_htj2k_jobs(&band_buffers, jobs, shape)?;

    let command_buffer = checked_command_buffer(&runtime.queue).map_err(|error| {
        MetalTranscodeError::support("Metal staged HTJ2K command buffer creation", error)
    })?;
    command_buffer.setLabel(Some(&NSString::from_str(
        "j2k-transcode-metal dct97 codeblock pipeline batch",
    )));
    let (row_us, column_us) =
        workspace.encode(&command_buffer, runtime, shape, &blocks, &band_buffers)?;
    timings.idct_row_lift_us = row_us;
    timings.column_lift_us = column_us;

    let quantize_start = Instant::now();
    encode_dwt97_quantize_codeblocks(
        &command_buffer,
        runtime,
        shape,
        options,
        &band_buffers,
        &codeblock_buffers,
    )?;
    commit_and_wait(&command_buffer).map_err(|error| {
        MetalTranscodeError::support("Metal staged HTJ2K command buffer", error)
    })?;
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

fn dwt97_outputs_from_projected_bands(
    bands: Vec<super::ProjectedBands>,
) -> Result<Vec<Dwt97TwoDimensional<f64>>, MetalTranscodeError> {
    let mut outputs = try_transcode_vec_with_capacity(bands.len(), "9/7 batch output metadata")?;
    for bands in bands {
        outputs.push(Dwt97TwoDimensional {
            ll: bands.ll,
            hl: bands.hl,
            lh: bands.lh,
            hh: bands.hh,
            low_width: bands.low_width,
            low_height: bands.low_height,
            high_width: bands.high_width,
            high_height: bands.high_height,
        });
    }
    Ok(outputs)
}

fn encode_dwt97_staged_row_lift(
    command_buffer: &CommandBufferRef,
    runtime: &MetalRuntime,
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
    let row_groups = (shape.height as usize).div_ceil(DWT97_STAGED_ROWS_PER_GROUP);

    let encoder = checked_compute_command_encoder(command_buffer).map_err(|error| {
        MetalTranscodeError::support("Metal 9/7 row-lift compute encoder creation", error)
    })?;
    encoder.setComputePipelineState(&runtime.dct97_idct_row_lift_batch);
    encoder.set_buffer(0, Some(blocks), 0)?;
    encoder.set_buffer(1, Some(&row_buffers.low), 0)?;
    encoder.set_buffer(2, Some(&row_buffers.high), 0)?;
    encoder.set_bytes::<Dct97IdctRowLiftParams>(3, &params)?;
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        super::mtl_size(row_groups as u64, u64::from(shape.batch_count_u32), 1),
        staged_threads_per_group(),
    );
    encoder.endEncoding();
    Ok(())
}

fn encode_dwt97_staged_column_lift(
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

    let encoder = checked_compute_command_encoder(command_buffer).map_err(|error| {
        MetalTranscodeError::support("Metal 9/7 column-lift compute encoder creation", error)
    })?;
    encoder.setComputePipelineState(&runtime.dct97_column_lift_batch);
    encoder.set_buffer(0, Some(&row_buffers.low), 0)?;
    encoder.set_buffer(1, Some(&row_buffers.high), 0)?;
    encoder.set_buffer(2, Some(&output_buffers.ll), 0)?;
    encoder.set_buffer(3, Some(&output_buffers.hl), 0)?;
    encoder.set_buffer(4, Some(&output_buffers.lh), 0)?;
    encoder.set_buffer(5, Some(&output_buffers.hh), 0)?;
    encoder.set_bytes::<Dct97ColumnLiftParams>(6, &params)?;
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        super::mtl_size(column_groups as u64, u64::from(shape.batch_count_u32), 2),
        staged_threads_per_group(),
    );
    encoder.endEncoding();
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
    let encoder = checked_compute_command_encoder(command_buffer).map_err(|error| {
        MetalTranscodeError::support("Metal HTJ2K quantize compute encoder creation", error)
    })?;
    encoder.setComputePipelineState(&runtime.dct97_quantize_codeblocks_batch);
    dispatch_dwt97_quantize_codeblock_band(
        &encoder,
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
        &encoder,
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
        &encoder,
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
        &encoder,
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
    encoder.endEncoding();
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
    encoder.set_buffer(0, Some(band_buffer), 0)?;
    encoder.set_buffer(1, Some(codeblock_buffer), 0)?;
    encoder.set_bytes::<Dct97QuantizeCodeblocksParams>(2, &params)?;
    dispatch_projection_threads(
        encoder,
        band.width as u64,
        band.height as u64,
        u64::from(band.batch_count),
    );
    Ok(())
}
