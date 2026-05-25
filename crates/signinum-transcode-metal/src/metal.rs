// SPDX-License-Identifier: Apache-2.0

//! Metal runtime for direct DCT-grid to one-level wavelet projection.

use std::sync::{Arc, OnceLock};
use std::time::Instant;

use core::f32::consts::PI;
use core::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};
use signinum_transcode::accelerator::{
    idct_blocks_to_signed_samples_rayon, DctGridToDwt53Job, DctGridToDwt97Job,
    DctGridToHtj2k97CodeBlockJob, DctGridToReversibleDwt53Job, Dwt97BatchStageTimings,
    Htj2k97CodeBlockOptions, J2kSubBandType, PrequantizedHtj2k97CodeBlock,
    PrequantizedHtj2k97Component, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
    ReversibleDwt53FirstLevel,
};
use signinum_transcode::dct53_2d::Dwt53TwoDimensional;
use signinum_transcode::dct97_2d::Dwt97TwoDimensional;

use crate::weights::{SparseDwt53WeightRows, SparseDwt97WeightRows, SparseWeightRow};
use crate::MetalTranscodeError;

const SHADER_SOURCE: &str = include_str!("dct97.metal");
const METAL_DCT_KERNEL_FAILED: &str = "Metal DCT wavelet projection failed";
const METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID: &str =
    "Metal reversible DCT 5/3 job has unsupported grid geometry";
const METAL_DCT53_UNSUPPORTED_GRID: &str = "Metal DCT 5/3 job has unsupported grid geometry";
const METAL_DCT97_UNSUPPORTED_GRID: &str = "Metal DCT 9/7 job has unsupported grid geometry";
const DWT97_STAGED_MAX_AXIS: usize = 1024;
const DWT97_STAGED_ROWS_PER_GROUP: usize = 2;
const DWT97_STAGED_COLUMNS_PER_GROUP: usize = 4;
const DWT97_STAGED_THREADS_PER_GROUP: u64 = 256;

static METAL_RUNTIME: OnceLock<Result<Arc<MetalRuntime>, MetalTranscodeError>> = OnceLock::new();

struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    dct_project_band: ComputePipelineState,
    dct_project_band_batch: ComputePipelineState,
    dct97_idct_row_lift_batch: ComputePipelineState,
    dct97_column_lift_batch: ComputePipelineState,
    dct97_quantize_codeblocks_batch: ComputePipelineState,
    reversible53_project_band: ComputePipelineState,
    idct_basis: Buffer,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DctProjectionParams {
    width: u32,
    height: u32,
    block_cols: u32,
    band_width: u32,
    band_height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DctBatchProjectionParams {
    width: u32,
    height: u32,
    block_cols: u32,
    blocks_per_item: u32,
    band_width: u32,
    band_height: u32,
    output_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Dct97IdctRowLiftParams {
    width: u32,
    height: u32,
    block_cols: u32,
    blocks_per_item: u32,
    low_width: u32,
    high_width: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Dct97ColumnLiftParams {
    height: u32,
    low_width: u32,
    high_width: u32,
    low_height: u32,
    high_height: u32,
    row_low_stride: u32,
    row_high_stride: u32,
    ll_stride: u32,
    hl_stride: u32,
    lh_stride: u32,
    hh_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Dct97QuantizeCodeblocksParams {
    band_width: u32,
    band_height: u32,
    output_stride: u32,
    code_block_width: u32,
    code_block_height: u32,
    inv_delta: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Reversible53ProjectionParams {
    width: u32,
    height: u32,
    block_cols: u32,
    blocks_per_item: u32,
    band_width: u32,
    band_height: u32,
    output_stride: u32,
    vertical_low: u32,
    horizontal_low: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalSparseRow {
    offset: u32,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalWeightTap {
    sample_idx: u32,
    weight: f32,
}

struct MetalSparseRows {
    rows: Vec<MetalSparseRow>,
    taps: Vec<MetalWeightTap>,
}

impl MetalRuntime {
    fn new() -> Result<Self, MetalTranscodeError> {
        let device = Device::system_default().ok_or(MetalTranscodeError::MetalUnavailable)?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let function = library
            .get_function("dct97_project_band", None)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let dct_project_band = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let batch_function = library
            .get_function("dct97_project_band_batch", None)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let dct_project_band_batch = device
            .new_compute_pipeline_state_with_function(&batch_function)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let row_lift_function = library
            .get_function("dct97_idct_row_lift_batch", None)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let dct97_idct_row_lift_batch = device
            .new_compute_pipeline_state_with_function(&row_lift_function)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let column_lift_function = library
            .get_function("dct97_column_lift_batch", None)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let dct97_column_lift_batch = device
            .new_compute_pipeline_state_with_function(&column_lift_function)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let quantize_codeblocks_function = library
            .get_function("dct97_quantize_codeblocks_batch", None)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let dct97_quantize_codeblocks_batch = device
            .new_compute_pipeline_state_with_function(&quantize_codeblocks_function)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let reversible_function = library
            .get_function("reversible53_project_band", None)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let reversible53_project_band = device
            .new_compute_pipeline_state_with_function(&reversible_function)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
        let queue = device.new_command_queue();
        let idct_basis_data = idct8_basis_table();
        let idct_basis = device.new_buffer_with_data(
            idct_basis_data.as_ptr().cast(),
            size_of_val(&idct_basis_data) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device,
            queue,
            dct_project_band,
            dct_project_band_batch,
            dct97_idct_row_lift_batch,
            dct97_column_lift_batch,
            dct97_quantize_codeblocks_batch,
            reversible53_project_band,
            idct_basis,
        })
    }
}

pub(crate) fn dispatch_dct_grid_to_reversible_dwt53(
    job: DctGridToReversibleDwt53Job<'_>,
) -> Result<ReversibleDwt53FirstLevel, MetalTranscodeError> {
    let mut outputs = dispatch_dct_grid_to_reversible_dwt53_batch(core::slice::from_ref(&job))?;
    outputs
        .pop()
        .ok_or(MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))
}

pub(crate) fn dispatch_dct_grid_to_reversible_dwt53_batch(
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

    with_runtime(|runtime| {
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
    with_runtime(|runtime| dispatch_dct_grid_to_dwt53_with_runtime(runtime, job))
}

#[allow(clippy::similar_names)]
fn dispatch_reversible_dwt53_batch_with_runtime(
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
    command_buffer.set_label("signinum-transcode-metal reversible dct53 projection");
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
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_reversible_batch_outputs(output_buffers, output_shape)
}

pub(crate) fn dispatch_dct_grid_to_dwt97(
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
    with_runtime(|runtime| dispatch_dct_grid_to_dwt97_with_runtime(runtime, job))
}

pub(crate) fn dispatch_dct_grid_to_dwt97_batch(
    jobs: &[DctGridToDwt97Job<'_>],
) -> Result<(Vec<Dwt97TwoDimensional<f64>>, Dwt97BatchStageTimings), MetalTranscodeError> {
    let Some(first) = jobs.first() else {
        return Ok((Vec::new(), Dwt97BatchStageTimings::default()));
    };
    validate_dwt97_batch_geometry(jobs)?;
    with_runtime(|runtime| dispatch_dct_grid_to_dwt97_batch_with_runtime(runtime, jobs, first))
}

pub(crate) fn dispatch_dct_grid_to_htj2k97_codeblock_batch(
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<PrequantizedHtj2k97Component>, Dwt97BatchStageTimings), MetalTranscodeError> {
    let Some(first) = jobs.first() else {
        return Ok((Vec::new(), Dwt97BatchStageTimings::default()));
    };
    validate_dwt97_codeblock_batch_geometry(jobs)?;
    validate_htj2k97_codeblock_options(options)?;
    with_runtime(|runtime| {
        dispatch_dct_grid_to_htj2k97_codeblock_batch_with_runtime(runtime, jobs, first, options)
    })
}

#[allow(clippy::similar_names)]
fn dispatch_dct_grid_to_dwt53_with_runtime(
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
            label: "signinum-transcode-metal dct53 projection",
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
fn dispatch_dct_grid_to_dwt97_with_runtime(
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
            label: "signinum-transcode-metal dct97 projection",
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
fn dispatch_dct_grid_to_dwt97_batch_with_runtime(
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
            label: "signinum-transcode-metal batched dct97 projection",
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

fn staged_dwt97_batch_supported(first: &DctGridToDwt97Job<'_>) -> bool {
    first.width <= DWT97_STAGED_MAX_AXIS && first.height <= DWT97_STAGED_MAX_AXIS
}

fn staged_dwt97_codeblock_batch_supported(first: &DctGridToHtj2k97CodeBlockJob<'_>) -> bool {
    first.width <= DWT97_STAGED_MAX_AXIS && first.height <= DWT97_STAGED_MAX_AXIS
}

fn dispatch_dct_grid_to_dwt97_batch_staged_with_runtime(
    runtime: &MetalRuntime,
    jobs: &[DctGridToDwt97Job<'_>],
    first: &DctGridToDwt97Job<'_>,
) -> Result<(Vec<Dwt97TwoDimensional<f64>>, Dwt97BatchStageTimings), MetalTranscodeError> {
    let shape = dwt97_staged_batch_shape(jobs, first)?;
    let mut timings = Dwt97BatchStageTimings::default();

    let pack_upload_start = Instant::now();
    let blocks = dwt97_batch_blocks_buffer(&runtime.device, jobs);
    let row_buffers = dwt97_staged_row_buffers(runtime, shape)?;
    let output_buffers =
        projection_batch_output_buffers(runtime, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    timings.pack_upload_us = pack_upload_start.elapsed().as_micros();

    let row_start = Instant::now();
    dispatch_dwt97_staged_row_lift(runtime, first.height, shape, &blocks, &row_buffers)?;
    timings.idct_row_lift_us = row_start.elapsed().as_micros();

    let column_start = Instant::now();
    dispatch_dwt97_staged_column_lift(runtime, shape, &row_buffers, &output_buffers)?;
    timings.column_lift_us = column_start.elapsed().as_micros();

    let readback_start = Instant::now();
    let bands = read_projected_batch_outputs(&output_buffers, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    timings.readback_us = readback_start.elapsed().as_micros();

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

fn dispatch_dct_grid_to_htj2k97_codeblock_batch_with_runtime(
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
    let blocks = dwt97_codeblock_batch_blocks_buffer(&runtime.device, jobs);
    let row_buffers = dwt97_staged_row_buffers(runtime, shape)?;
    let band_buffers =
        projection_batch_output_buffers(runtime, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    let codeblock_buffers =
        dwt97_codeblock_output_buffers(runtime, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    timings.pack_upload_us = pack_upload_start.elapsed().as_micros();

    let row_start = Instant::now();
    dispatch_dwt97_staged_row_lift(runtime, first.height, shape, &blocks, &row_buffers)?;
    timings.idct_row_lift_us = row_start.elapsed().as_micros();

    let column_start = Instant::now();
    dispatch_dwt97_staged_column_lift(runtime, shape, &row_buffers, &band_buffers)?;
    timings.column_lift_us = column_start.elapsed().as_micros();

    let quantize_start = Instant::now();
    dispatch_dwt97_quantize_codeblocks(runtime, shape, options, &band_buffers, &codeblock_buffers)?;
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

    Ok((components, timings))
}

fn dwt97_staged_batch_shape(
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

fn dwt97_codeblock_batch_shape(
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

struct Dwt97StagedRowBuffers {
    low: Buffer,
    high: Buffer,
}

fn dwt97_staged_row_buffers(
    runtime: &MetalRuntime,
    shape: ProjectionBatchShape,
) -> Result<Dwt97StagedRowBuffers, MetalTranscodeError> {
    let height = shape.height as usize;
    Ok(Dwt97StagedRowBuffers {
        low: output_buffer(
            &runtime.device,
            checked_batch_len(
                height * shape.low_width,
                shape.batch_count,
                METAL_DCT97_UNSUPPORTED_GRID,
            )?,
        ),
        high: output_buffer(
            &runtime.device,
            checked_batch_len(
                height * shape.high_width,
                shape.batch_count,
                METAL_DCT97_UNSUPPORTED_GRID,
            )?,
        ),
    })
}

fn dispatch_dwt97_staged_row_lift(
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

    let command_buffer = runtime.queue.new_command_buffer();
    command_buffer.set_label("signinum-transcode-metal dct97 idct row lift batch");
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
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

fn dispatch_dwt97_staged_column_lift(
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

    let command_buffer = runtime.queue.new_command_buffer();
    command_buffer.set_label("signinum-transcode-metal dct97 column lift batch");
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
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

fn dispatch_dwt97_quantize_codeblocks(
    runtime: &MetalRuntime,
    shape: ProjectionBatchShape,
    options: Htj2k97CodeBlockOptions,
    band_buffers: &ProjectionBatchOutputBuffers,
    codeblock_buffers: &Dwt97CodeBlockOutputBuffers,
) -> Result<(), MetalTranscodeError> {
    let cb_width = code_block_len_from_exp(options.code_block_width_exp)?;
    let cb_height = code_block_len_from_exp(options.code_block_height_exp)?;
    let command_buffer = runtime.queue.new_command_buffer();
    command_buffer.set_label("signinum-transcode-metal dct97 quantize codeblocks batch");
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
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

#[derive(Clone, Copy)]
struct Dwt97QuantizeBand {
    width: usize,
    height: usize,
    stride: usize,
    cb_width: usize,
    cb_height: usize,
    inv_delta: f32,
    batch_count: u32,
}

fn dispatch_dwt97_quantize_codeblock_band(
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
    encoder.dispatch_threads(
        MTLSize {
            width: band.width as u64,
            height: band.height as u64,
            depth: u64::from(band.batch_count),
        },
        MTLSize {
            width: 16,
            height: 8,
            depth: 1,
        },
    );
    Ok(())
}

fn staged_threads_per_group() -> MTLSize {
    MTLSize {
        width: DWT97_STAGED_THREADS_PER_GROUP,
        height: 1,
        depth: 1,
    }
}

#[derive(Clone, Copy)]
struct ProjectionJob<'a> {
    blocks: &'a [[[f64; 8]; 8]],
    block_cols: usize,
    width: usize,
    height: usize,
    x_low: &'a [SparseWeightRow],
    x_high: &'a [SparseWeightRow],
    y_low: &'a [SparseWeightRow],
    y_high: &'a [SparseWeightRow],
    unsupported_grid: &'static str,
    label: &'static str,
}

#[derive(Clone, Copy)]
struct ProjectionBatchJob<'a, 'b> {
    jobs: &'a [DctGridToDwt97Job<'b>],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    x_low: &'a [SparseWeightRow],
    x_high: &'a [SparseWeightRow],
    y_low: &'a [SparseWeightRow],
    y_high: &'a [SparseWeightRow],
    unsupported_grid: &'static str,
    label: &'static str,
}

struct ProjectedBands {
    ll: Vec<f64>,
    hl: Vec<f64>,
    lh: Vec<f64>,
    hh: Vec<f64>,
    low_width: usize,
    low_height: usize,
    high_width: usize,
    high_height: usize,
}

#[derive(Clone, Copy)]
struct ReversibleBatchOutputShape {
    low_width: usize,
    low_height: usize,
    high_width: usize,
    high_height: usize,
    ll_len: usize,
    hl_len: usize,
    lh_len: usize,
    hh_len: usize,
    batch_count: usize,
}

#[derive(Clone, Copy)]
struct ReversibleOutputBuffers<'a> {
    ll: &'a Buffer,
    hl: &'a Buffer,
    lh: &'a Buffer,
    hh: &'a Buffer,
}

fn read_reversible_batch_outputs(
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
    );
    let hl = read_i32_buffer(
        buffers.hl,
        checked_batch_len(
            shape.hl_len,
            shape.batch_count,
            METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
        )?,
    );
    let lh = read_i32_buffer(
        buffers.lh,
        checked_batch_len(
            shape.lh_len,
            shape.batch_count,
            METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
        )?,
    );
    let hh = read_i32_buffer(
        buffers.hh,
        checked_batch_len(
            shape.hh_len,
            shape.batch_count,
            METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
        )?,
    );

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

#[allow(clippy::similar_names)]
fn dispatch_projected_bands_with_runtime(
    runtime: &MetalRuntime,
    job: ProjectionJob<'_>,
) -> Result<ProjectedBands, MetalTranscodeError> {
    let width = u32_param(job.width, job.unsupported_grid)?;
    let height = u32_param(job.height, job.unsupported_grid)?;
    let block_cols = u32_param(job.block_cols, job.unsupported_grid)?;
    let low_width = job.width.div_ceil(2);
    let high_width = job.width / 2;
    let low_height = job.height.div_ceil(2);
    let high_height = job.height / 2;

    let x_low = metal_sparse_rows(job.x_low, job.unsupported_grid)?;
    let x_high = metal_sparse_rows(job.x_high, job.unsupported_grid)?;
    let y_low = metal_sparse_rows(job.y_low, job.unsupported_grid)?;
    let y_high = metal_sparse_rows(job.y_high, job.unsupported_grid)?;
    let x_low_rows = buffer_with_slice(&runtime.device, &x_low.rows);
    let x_low_taps = buffer_with_slice(&runtime.device, &x_low.taps);
    let x_high_rows = buffer_with_slice(&runtime.device, &x_high.rows);
    let x_high_taps = buffer_with_slice(&runtime.device, &x_high.taps);
    let y_low_rows = buffer_with_slice(&runtime.device, &y_low.rows);
    let y_low_taps = buffer_with_slice(&runtime.device, &y_low.taps);
    let y_high_rows = buffer_with_slice(&runtime.device, &y_high.rows);
    let y_high_taps = buffer_with_slice(&runtime.device, &y_high.taps);
    let blocks = dwt97_blocks_buffer(&runtime.device, job.blocks);

    let ll_buffer = output_buffer(&runtime.device, low_width * low_height);
    let hl_buffer = output_buffer(&runtime.device, high_width * low_height);
    let lh_buffer = output_buffer(&runtime.device, low_width * high_height);
    let hh_buffer = output_buffer(&runtime.device, high_width * high_height);

    let command_buffer = runtime.queue.new_command_buffer();
    command_buffer.set_label(job.label);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.dct_project_band);
    encoder.set_buffer(0, Some(&blocks), 0);
    encoder.set_buffer(5, Some(&runtime.idct_basis), 0);

    dispatch_band(
        encoder,
        (&x_low_rows, &x_low_taps),
        (&y_low_rows, &y_low_taps),
        &ll_buffer,
        BandGeometry {
            width,
            height,
            block_cols,
            band_width: u32_param(low_width, job.unsupported_grid)?,
            band_height: u32_param(low_height, job.unsupported_grid)?,
        },
    );
    dispatch_band(
        encoder,
        (&x_high_rows, &x_high_taps),
        (&y_low_rows, &y_low_taps),
        &hl_buffer,
        BandGeometry {
            width,
            height,
            block_cols,
            band_width: u32_param(high_width, job.unsupported_grid)?,
            band_height: u32_param(low_height, job.unsupported_grid)?,
        },
    );
    dispatch_band(
        encoder,
        (&x_low_rows, &x_low_taps),
        (&y_high_rows, &y_high_taps),
        &lh_buffer,
        BandGeometry {
            width,
            height,
            block_cols,
            band_width: u32_param(low_width, job.unsupported_grid)?,
            band_height: u32_param(high_height, job.unsupported_grid)?,
        },
    );
    dispatch_band(
        encoder,
        (&x_high_rows, &x_high_taps),
        (&y_high_rows, &y_high_taps),
        &hh_buffer,
        BandGeometry {
            width,
            height,
            block_cols,
            band_width: u32_param(high_width, job.unsupported_grid)?,
            band_height: u32_param(high_height, job.unsupported_grid)?,
        },
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(ProjectedBands {
        ll: read_f32_buffer(&ll_buffer, low_width * low_height),
        hl: read_f32_buffer(&hl_buffer, high_width * low_height),
        lh: read_f32_buffer(&lh_buffer, low_width * high_height),
        hh: read_f32_buffer(&hh_buffer, high_width * high_height),
        low_width,
        low_height,
        high_width,
        high_height,
    })
}

#[allow(clippy::similar_names)]
fn dispatch_projected_bands_batch_with_runtime(
    runtime: &MetalRuntime,
    job: ProjectionBatchJob<'_, '_>,
) -> Result<Vec<ProjectedBands>, MetalTranscodeError> {
    let Some(shape) = projection_batch_shape(job)? else {
        return Ok(Vec::new());
    };

    let weights = projection_batch_weight_buffers(runtime, job)?;
    let blocks = dwt97_batch_blocks_buffer(&runtime.device, job.jobs);
    let outputs = projection_batch_output_buffers(runtime, shape, job.unsupported_grid)?;

    dispatch_projection_batch_bands(runtime, job, shape, &weights, &blocks, &outputs)?;
    read_projected_batch_outputs(&outputs, shape, job.unsupported_grid)
}

#[derive(Clone, Copy)]
struct ProjectionBatchShape {
    batch_count: usize,
    batch_count_u32: u32,
    width: u32,
    height: u32,
    block_cols: u32,
    blocks_per_item: u32,
    low_width: usize,
    low_height: usize,
    high_width: usize,
    high_height: usize,
    ll_len: usize,
    hl_len: usize,
    lh_len: usize,
    hh_len: usize,
}

fn projection_batch_shape(
    job: ProjectionBatchJob<'_, '_>,
) -> Result<Option<ProjectionBatchShape>, MetalTranscodeError> {
    let batch_count = job.jobs.len();
    if batch_count == 0 {
        return Ok(None);
    }

    let low_width = job.width.div_ceil(2);
    let high_width = job.width / 2;
    let low_height = job.height.div_ceil(2);
    let high_height = job.height / 2;
    let blocks_per_item = job
        .block_cols
        .checked_mul(job.block_rows)
        .ok_or(MetalTranscodeError::UnsupportedJob(job.unsupported_grid))?;

    Ok(Some(ProjectionBatchShape {
        batch_count,
        batch_count_u32: u32_param(batch_count, job.unsupported_grid)?,
        width: u32_param(job.width, job.unsupported_grid)?,
        height: u32_param(job.height, job.unsupported_grid)?,
        block_cols: u32_param(job.block_cols, job.unsupported_grid)?,
        blocks_per_item: u32_param(blocks_per_item, job.unsupported_grid)?,
        low_width,
        low_height,
        high_width,
        high_height,
        ll_len: low_width * low_height,
        hl_len: high_width * low_height,
        lh_len: low_width * high_height,
        hh_len: high_width * high_height,
    }))
}

struct ProjectionBatchWeightBuffers {
    x_low_rows: Buffer,
    x_low_taps: Buffer,
    x_high_rows: Buffer,
    x_high_taps: Buffer,
    y_low_rows: Buffer,
    y_low_taps: Buffer,
    y_high_rows: Buffer,
    y_high_taps: Buffer,
}

fn projection_batch_weight_buffers(
    runtime: &MetalRuntime,
    job: ProjectionBatchJob<'_, '_>,
) -> Result<ProjectionBatchWeightBuffers, MetalTranscodeError> {
    let x_low = metal_sparse_rows(job.x_low, job.unsupported_grid)?;
    let x_high = metal_sparse_rows(job.x_high, job.unsupported_grid)?;
    let y_low = metal_sparse_rows(job.y_low, job.unsupported_grid)?;
    let y_high = metal_sparse_rows(job.y_high, job.unsupported_grid)?;

    Ok(ProjectionBatchWeightBuffers {
        x_low_rows: buffer_with_slice(&runtime.device, &x_low.rows),
        x_low_taps: buffer_with_slice(&runtime.device, &x_low.taps),
        x_high_rows: buffer_with_slice(&runtime.device, &x_high.rows),
        x_high_taps: buffer_with_slice(&runtime.device, &x_high.taps),
        y_low_rows: buffer_with_slice(&runtime.device, &y_low.rows),
        y_low_taps: buffer_with_slice(&runtime.device, &y_low.taps),
        y_high_rows: buffer_with_slice(&runtime.device, &y_high.rows),
        y_high_taps: buffer_with_slice(&runtime.device, &y_high.taps),
    })
}

struct ProjectionBatchOutputBuffers {
    ll: Buffer,
    hl: Buffer,
    lh: Buffer,
    hh: Buffer,
}

struct Dwt97CodeBlockOutputBuffers {
    ll: Buffer,
    hl: Buffer,
    lh: Buffer,
    hh: Buffer,
}

fn projection_batch_output_buffers(
    runtime: &MetalRuntime,
    shape: ProjectionBatchShape,
    unsupported_grid: &'static str,
) -> Result<ProjectionBatchOutputBuffers, MetalTranscodeError> {
    Ok(ProjectionBatchOutputBuffers {
        ll: output_buffer(
            &runtime.device,
            checked_batch_len(shape.ll_len, shape.batch_count, unsupported_grid)?,
        ),
        hl: output_buffer(
            &runtime.device,
            checked_batch_len(shape.hl_len, shape.batch_count, unsupported_grid)?,
        ),
        lh: output_buffer(
            &runtime.device,
            checked_batch_len(shape.lh_len, shape.batch_count, unsupported_grid)?,
        ),
        hh: output_buffer(
            &runtime.device,
            checked_batch_len(shape.hh_len, shape.batch_count, unsupported_grid)?,
        ),
    })
}

fn dwt97_codeblock_output_buffers(
    runtime: &MetalRuntime,
    shape: ProjectionBatchShape,
    unsupported_grid: &'static str,
) -> Result<Dwt97CodeBlockOutputBuffers, MetalTranscodeError> {
    Ok(Dwt97CodeBlockOutputBuffers {
        ll: output_i32_buffer(
            &runtime.device,
            checked_batch_len(shape.ll_len, shape.batch_count, unsupported_grid)?,
        ),
        hl: output_i32_buffer(
            &runtime.device,
            checked_batch_len(shape.hl_len, shape.batch_count, unsupported_grid)?,
        ),
        lh: output_i32_buffer(
            &runtime.device,
            checked_batch_len(shape.lh_len, shape.batch_count, unsupported_grid)?,
        ),
        hh: output_i32_buffer(
            &runtime.device,
            checked_batch_len(shape.hh_len, shape.batch_count, unsupported_grid)?,
        ),
    })
}

fn dispatch_projection_batch_bands(
    runtime: &MetalRuntime,
    job: ProjectionBatchJob<'_, '_>,
    shape: ProjectionBatchShape,
    weights: &ProjectionBatchWeightBuffers,
    blocks: &Buffer,
    outputs: &ProjectionBatchOutputBuffers,
) -> Result<(), MetalTranscodeError> {
    let command_buffer = runtime.queue.new_command_buffer();
    command_buffer.set_label(job.label);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.dct_project_band_batch);
    encoder.set_buffer(0, Some(blocks), 0);
    encoder.set_buffer(5, Some(&runtime.idct_basis), 0);

    dispatch_band_batch(
        encoder,
        (&weights.x_low_rows, &weights.x_low_taps),
        (&weights.y_low_rows, &weights.y_low_taps),
        &outputs.ll,
        BatchBandGeometry {
            width: shape.width,
            height: shape.height,
            block_cols: shape.block_cols,
            blocks_per_item: shape.blocks_per_item,
            band_width: u32_param(shape.low_width, job.unsupported_grid)?,
            band_height: u32_param(shape.low_height, job.unsupported_grid)?,
            output_stride: u32_param(shape.ll_len, job.unsupported_grid)?,
            batch_count: shape.batch_count_u32,
        },
    );
    dispatch_band_batch(
        encoder,
        (&weights.x_high_rows, &weights.x_high_taps),
        (&weights.y_low_rows, &weights.y_low_taps),
        &outputs.hl,
        BatchBandGeometry {
            width: shape.width,
            height: shape.height,
            block_cols: shape.block_cols,
            blocks_per_item: shape.blocks_per_item,
            band_width: u32_param(shape.high_width, job.unsupported_grid)?,
            band_height: u32_param(shape.low_height, job.unsupported_grid)?,
            output_stride: u32_param(shape.hl_len, job.unsupported_grid)?,
            batch_count: shape.batch_count_u32,
        },
    );
    dispatch_band_batch(
        encoder,
        (&weights.x_low_rows, &weights.x_low_taps),
        (&weights.y_high_rows, &weights.y_high_taps),
        &outputs.lh,
        BatchBandGeometry {
            width: shape.width,
            height: shape.height,
            block_cols: shape.block_cols,
            blocks_per_item: shape.blocks_per_item,
            band_width: u32_param(shape.low_width, job.unsupported_grid)?,
            band_height: u32_param(shape.high_height, job.unsupported_grid)?,
            output_stride: u32_param(shape.lh_len, job.unsupported_grid)?,
            batch_count: shape.batch_count_u32,
        },
    );
    dispatch_band_batch(
        encoder,
        (&weights.x_high_rows, &weights.x_high_taps),
        (&weights.y_high_rows, &weights.y_high_taps),
        &outputs.hh,
        BatchBandGeometry {
            width: shape.width,
            height: shape.height,
            block_cols: shape.block_cols,
            blocks_per_item: shape.blocks_per_item,
            band_width: u32_param(shape.high_width, job.unsupported_grid)?,
            band_height: u32_param(shape.high_height, job.unsupported_grid)?,
            output_stride: u32_param(shape.hh_len, job.unsupported_grid)?,
            batch_count: shape.batch_count_u32,
        },
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

fn read_projected_batch_outputs(
    buffers: &ProjectionBatchOutputBuffers,
    shape: ProjectionBatchShape,
    unsupported_grid: &'static str,
) -> Result<Vec<ProjectedBands>, MetalTranscodeError> {
    let ll = shared_f32_slice(
        &buffers.ll,
        checked_batch_len(shape.ll_len, shape.batch_count, unsupported_grid)?,
    );
    let hl = shared_f32_slice(
        &buffers.hl,
        checked_batch_len(shape.hl_len, shape.batch_count, unsupported_grid)?,
    );
    let lh = shared_f32_slice(
        &buffers.lh,
        checked_batch_len(shape.lh_len, shape.batch_count, unsupported_grid)?,
    );
    let hh = shared_f32_slice(
        &buffers.hh,
        checked_batch_len(shape.hh_len, shape.batch_count, unsupported_grid)?,
    );

    let mut outputs = Vec::with_capacity(shape.batch_count);
    for idx in 0..shape.batch_count {
        outputs.push(ProjectedBands {
            ll: f32_slice_to_f64(&ll[idx * shape.ll_len..idx * shape.ll_len + shape.ll_len]),
            hl: f32_slice_to_f64(&hl[idx * shape.hl_len..idx * shape.hl_len + shape.hl_len]),
            lh: f32_slice_to_f64(&lh[idx * shape.lh_len..idx * shape.lh_len + shape.lh_len]),
            hh: f32_slice_to_f64(&hh[idx * shape.hh_len..idx * shape.hh_len + shape.hh_len]),
            low_width: shape.low_width,
            low_height: shape.low_height,
            high_width: shape.high_width,
            high_height: shape.high_height,
        });
    }

    Ok(outputs)
}

fn read_prequantized_97_codeblock_outputs(
    buffers: &Dwt97CodeBlockOutputBuffers,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    shape: ProjectionBatchShape,
    options: Htj2k97CodeBlockOptions,
    unsupported_grid: &'static str,
) -> Result<Vec<PrequantizedHtj2k97Component>, MetalTranscodeError> {
    let ll = shared_i32_slice(
        &buffers.ll,
        checked_batch_len(shape.ll_len, shape.batch_count, unsupported_grid)?,
    );
    let hl = shared_i32_slice(
        &buffers.hl,
        checked_batch_len(shape.hl_len, shape.batch_count, unsupported_grid)?,
    );
    let lh = shared_i32_slice(
        &buffers.lh,
        checked_batch_len(shape.lh_len, shape.batch_count, unsupported_grid)?,
    );
    let hh = shared_i32_slice(
        &buffers.hh,
        checked_batch_len(shape.hh_len, shape.batch_count, unsupported_grid)?,
    );

    let mut components = Vec::with_capacity(shape.batch_count);
    for (idx, job) in jobs.iter().enumerate() {
        components.push(PrequantizedHtj2k97Component {
            x_rsiz: job.x_rsiz,
            y_rsiz: job.y_rsiz,
            resolutions: vec![
                PrequantizedHtj2k97Resolution {
                    subbands: vec![prequantized_subband_from_codeblock_buffer(
                        codeblock_item_slice(ll, idx, shape.ll_len, unsupported_grid)?,
                        shape.low_width,
                        shape.low_height,
                        J2kSubBandType::LowLow,
                        dwt97_total_bitplanes(options, J2kSubBandType::LowLow),
                        options,
                    )?],
                },
                PrequantizedHtj2k97Resolution {
                    subbands: vec![
                        prequantized_subband_from_codeblock_buffer(
                            codeblock_item_slice(hl, idx, shape.hl_len, unsupported_grid)?,
                            shape.high_width,
                            shape.low_height,
                            J2kSubBandType::HighLow,
                            dwt97_total_bitplanes(options, J2kSubBandType::HighLow),
                            options,
                        )?,
                        prequantized_subband_from_codeblock_buffer(
                            codeblock_item_slice(lh, idx, shape.lh_len, unsupported_grid)?,
                            shape.low_width,
                            shape.high_height,
                            J2kSubBandType::LowHigh,
                            dwt97_total_bitplanes(options, J2kSubBandType::LowHigh),
                            options,
                        )?,
                        prequantized_subband_from_codeblock_buffer(
                            codeblock_item_slice(hh, idx, shape.hh_len, unsupported_grid)?,
                            shape.high_width,
                            shape.high_height,
                            J2kSubBandType::HighHigh,
                            dwt97_total_bitplanes(options, J2kSubBandType::HighHigh),
                            options,
                        )?,
                    ],
                },
            ],
        });
    }

    Ok(components)
}

fn codeblock_item_slice<'a>(
    values: &'a [i32],
    item_idx: usize,
    stride: usize,
    unsupported_grid: &'static str,
) -> Result<&'a [i32], MetalTranscodeError> {
    let start = item_idx
        .checked_mul(stride)
        .ok_or(MetalTranscodeError::UnsupportedJob(unsupported_grid))?;
    let end = start
        .checked_add(stride)
        .ok_or(MetalTranscodeError::UnsupportedJob(unsupported_grid))?;
    values
        .get(start..end)
        .ok_or(MetalTranscodeError::UnsupportedJob(unsupported_grid))
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
    let mut offset = 0usize;
    let mut code_blocks = Vec::with_capacity(num_cbs_x.saturating_mul(num_cbs_y));
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let block_width = (width - x0).min(cb_width);
            let block_height = (height - y0).min(cb_height);
            let len = block_width.checked_mul(block_height).ok_or(
                MetalTranscodeError::UnsupportedJob(METAL_DCT97_UNSUPPORTED_GRID),
            )?;
            let end = offset
                .checked_add(len)
                .ok_or(MetalTranscodeError::UnsupportedJob(
                    METAL_DCT97_UNSUPPORTED_GRID,
                ))?;
            let coefficients = values
                .get(offset..end)
                .ok_or(MetalTranscodeError::UnsupportedJob(
                    METAL_DCT97_UNSUPPORTED_GRID,
                ))?
                .to_vec();
            code_blocks.push(PrequantizedHtj2k97CodeBlock {
                coefficients,
                width: u32_param(block_width, METAL_DCT97_UNSUPPORTED_GRID)?,
                height: u32_param(block_height, METAL_DCT97_UNSUPPORTED_GRID)?,
            });
            offset = end;
        }
    }

    Ok(PrequantizedHtj2k97Subband {
        sub_band_type,
        num_cbs_x: u32_param(num_cbs_x, METAL_DCT97_UNSUPPORTED_GRID)?,
        num_cbs_y: u32_param(num_cbs_y, METAL_DCT97_UNSUPPORTED_GRID)?,
        total_bitplanes,
        code_blocks,
    })
}

fn with_runtime<R>(
    f: impl FnOnce(&MetalRuntime) -> Result<R, MetalTranscodeError>,
) -> Result<R, MetalTranscodeError> {
    match METAL_RUNTIME.get_or_init(|| MetalRuntime::new().map(Arc::new)) {
        Ok(runtime) => f(runtime),
        Err(error) => Err(*error),
    }
}

#[derive(Clone, Copy)]
struct BandGeometry {
    width: u32,
    height: u32,
    block_cols: u32,
    band_width: u32,
    band_height: u32,
}

#[derive(Clone, Copy)]
struct BatchBandGeometry {
    width: u32,
    height: u32,
    block_cols: u32,
    blocks_per_item: u32,
    band_width: u32,
    band_height: u32,
    output_stride: u32,
    batch_count: u32,
}

#[derive(Clone, Copy)]
struct ReversibleBandGeometry {
    width: u32,
    height: u32,
    block_cols: u32,
    blocks_per_item: u32,
    band_width: u32,
    band_height: u32,
    output_stride: u32,
    batch_count: u32,
    vertical_low: bool,
    horizontal_low: bool,
}

#[derive(Clone, Copy)]
struct ReversibleBatchKernelGeometry {
    width: u32,
    height: u32,
    block_cols: u32,
    blocks_per_item: u32,
    batch_count: u32,
}

fn reversible_band_geometry(
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

fn dispatch_reversible_band(
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
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(geometry.band_width),
            height: u64::from(geometry.band_height),
            depth: u64::from(geometry.batch_count),
        },
        MTLSize {
            width: 16,
            height: 8,
            depth: 1,
        },
    );
}

fn dispatch_band(
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
    encoder.set_buffer(1, Some(x_weights.0), 0);
    encoder.set_buffer(2, Some(x_weights.1), 0);
    encoder.set_buffer(3, Some(y_weights.0), 0);
    encoder.set_buffer(4, Some(y_weights.1), 0);
    encoder.set_buffer(6, Some(output), 0);
    encoder.set_bytes(
        7,
        size_of::<DctProjectionParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(geometry.band_width),
            height: u64::from(geometry.band_height),
            depth: 1,
        },
        MTLSize {
            width: 16,
            height: 8,
            depth: 1,
        },
    );
}

fn dispatch_band_batch(
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
    encoder.set_buffer(1, Some(x_weights.0), 0);
    encoder.set_buffer(2, Some(x_weights.1), 0);
    encoder.set_buffer(3, Some(y_weights.0), 0);
    encoder.set_buffer(4, Some(y_weights.1), 0);
    encoder.set_buffer(6, Some(output), 0);
    encoder.set_bytes(
        7,
        size_of::<DctBatchProjectionParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(geometry.band_width),
            height: u64::from(geometry.band_height),
            depth: u64::from(geometry.batch_count),
        },
        MTLSize {
            width: 16,
            height: 8,
            depth: 1,
        },
    );
}

fn validate_grid(
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

fn validate_reversible_batch_geometry(
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

fn validate_dwt97_batch_geometry(
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

fn validate_dwt97_codeblock_batch_geometry(
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

fn validate_htj2k97_codeblock_options(
    options: Htj2k97CodeBlockOptions,
) -> Result<(), MetalTranscodeError> {
    if options.bit_depth == 0 || options.guard_bits == 0 {
        return Err(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ));
    }
    if !options.irreversible_quantization_scale.is_finite()
        || options.irreversible_quantization_scale <= 0.0
    {
        return Err(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ));
    }
    let _ = code_block_len_from_exp(options.code_block_width_exp)?;
    let _ = code_block_len_from_exp(options.code_block_height_exp)?;
    Ok(())
}

fn code_block_len_from_exp(exp: u8) -> Result<usize, MetalTranscodeError> {
    1usize
        .checked_shl(u32::from(exp) + 2)
        .filter(|&value| value > 0)
        .ok_or(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ))
}

fn dwt97_total_bitplanes(options: Htj2k97CodeBlockOptions, sub_band_type: J2kSubBandType) -> u8 {
    let step = dwt97_quant_step(options, sub_band_type);
    options
        .guard_bits
        .saturating_add(step.exponent)
        .saturating_sub(1)
}

fn dwt97_quantize_inv_delta(
    options: Htj2k97CodeBlockOptions,
    sub_band_type: J2kSubBandType,
) -> f32 {
    1.0 / dwt97_quant_delta(options, sub_band_type)
}

#[derive(Clone, Copy)]
struct Dwt97QuantStep {
    exponent: u8,
    mantissa: u16,
}

fn dwt97_quant_step(
    options: Htj2k97CodeBlockOptions,
    _sub_band_type: J2kSubBandType,
) -> Dwt97QuantStep {
    let base_delta =
        dwt97_pow2i(-i32::from(options.guard_bits)) * options.irreversible_quantization_scale;
    dwt97_quant_step_from_delta(options.bit_depth, base_delta)
}

fn dwt97_quant_step_from_delta(range_bits: u8, delta: f32) -> Dwt97QuantStep {
    let floor_log2 = delta.log2().floor() as i32;
    let mut exponent = i32::from(range_bits) - floor_log2;
    let normalized = delta / dwt97_pow2i(floor_log2);
    let mut mantissa = ((normalized - 1.0) * 2048.0).round() as i32;

    if mantissa >= 2048 {
        exponent -= 1;
        mantissa = 0;
    }

    Dwt97QuantStep {
        exponent: u8::try_from(exponent.clamp(0, 31)).expect("clamped exponent fits u8"),
        mantissa: u16::try_from(mantissa.clamp(0, 2047)).expect("clamped mantissa fits u16"),
    }
}

fn dwt97_quant_delta(options: Htj2k97CodeBlockOptions, sub_band_type: J2kSubBandType) -> f32 {
    let log_gain = match sub_band_type {
        J2kSubBandType::LowLow => 0,
        J2kSubBandType::HighLow | J2kSubBandType::LowHigh => 1,
        J2kSubBandType::HighHigh => 2,
    };
    let range_bits = i32::from(options.bit_depth) + log_gain;
    let step = dwt97_quant_step(options, sub_band_type);
    dwt97_pow2i(range_bits - i32::from(step.exponent)) * (1.0 + f32::from(step.mantissa) / 2048.0)
}

fn dwt97_pow2i(exp: i32) -> f32 {
    if exp >= 0 {
        (1u32 << exp.cast_unsigned()) as f32
    } else {
        1.0 / (1u32 << (-exp).cast_unsigned()) as f32
    }
}

fn checked_batch_len(
    value_len: usize,
    batch_count: usize,
    unsupported_grid: &'static str,
) -> Result<usize, MetalTranscodeError> {
    value_len
        .checked_mul(batch_count)
        .ok_or(MetalTranscodeError::UnsupportedJob(unsupported_grid))
}

fn u32_param(value: usize, unsupported_grid: &'static str) -> Result<u32, MetalTranscodeError> {
    u32::try_from(value).map_err(|_| MetalTranscodeError::UnsupportedJob(unsupported_grid))
}

fn metal_sparse_rows(
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

fn buffer_with_slice<T>(device: &Device, values: &[T]) -> Buffer {
    if values.is_empty() {
        return device.new_buffer(1, MTLResourceOptions::StorageModeShared);
    }
    device.new_buffer_with_data(
        values.as_ptr().cast(),
        size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn dwt97_blocks_buffer(device: &Device, blocks: &[[[f64; 8]; 8]]) -> Buffer {
    let value_count = blocks.len().saturating_mul(64);
    let buffer = output_buffer(device, value_count);
    write_dwt97_blocks_to_buffer(&buffer, blocks);
    buffer
}

fn dwt97_batch_blocks_buffer(device: &Device, jobs: &[DctGridToDwt97Job<'_>]) -> Buffer {
    let value_count = jobs
        .iter()
        .map(|job| job.blocks.len().saturating_mul(64))
        .sum();
    let buffer = output_buffer(device, value_count);
    let mut offset = 0;
    for job in jobs {
        offset += write_dwt97_blocks_to_buffer_at(&buffer, offset, job.blocks);
    }
    debug_assert_eq!(offset, value_count);
    buffer
}

fn dwt97_codeblock_batch_blocks_buffer(
    device: &Device,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
) -> Buffer {
    let value_count = jobs
        .iter()
        .map(|job| job.blocks.len().saturating_mul(64))
        .sum();
    let buffer = output_buffer(device, value_count);
    let mut offset = 0;
    for job in jobs {
        offset += write_dwt97_blocks_to_buffer_at(&buffer, offset, job.blocks);
    }
    debug_assert_eq!(offset, value_count);
    buffer
}

fn write_dwt97_blocks_to_buffer(buffer: &Buffer, blocks: &[[[f64; 8]; 8]]) {
    let written = write_dwt97_blocks_to_buffer_at(buffer, 0, blocks);
    debug_assert_eq!(written, blocks.len().saturating_mul(64));
}

fn write_dwt97_blocks_to_buffer_at(
    buffer: &Buffer,
    start: usize,
    blocks: &[[[f64; 8]; 8]],
) -> usize {
    let mut offset = start;
    unsafe {
        let values = buffer.contents().cast::<f32>();
        for block in blocks {
            for row in block {
                for &coefficient in row {
                    values.add(offset).write(coefficient as f32);
                    offset += 1;
                }
            }
        }
    }
    offset - start
}

fn output_buffer(device: &Device, value_count: usize) -> Buffer {
    device.new_buffer(
        (value_count * size_of::<f32>()).max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn output_i32_buffer(device: &Device, value_count: usize) -> Buffer {
    device.new_buffer(
        (value_count * size_of::<i32>()).max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn read_f32_buffer(buffer: &Buffer, value_count: usize) -> Vec<f64> {
    f32_slice_to_f64(shared_f32_slice(buffer, value_count))
}

fn read_i32_buffer(buffer: &Buffer, value_count: usize) -> Vec<i32> {
    shared_i32_slice(buffer, value_count).to_vec()
}

fn shared_f32_slice(buffer: &Buffer, value_count: usize) -> &[f32] {
    if value_count == 0 {
        return &[];
    }
    unsafe { core::slice::from_raw_parts(buffer.contents().cast::<f32>(), value_count) }
}

fn shared_i32_slice(buffer: &Buffer, value_count: usize) -> &[i32] {
    if value_count == 0 {
        return &[];
    }
    unsafe { core::slice::from_raw_parts(buffer.contents().cast::<i32>(), value_count) }
}

fn f32_slice_to_f64(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&value| f64::from(value)).collect()
}

fn idct8_basis_table() -> [f32; 64] {
    let mut table = [0.0; 64];
    for sample_idx in 0..8 {
        for freq in 0..8 {
            table[sample_idx * 8 + freq] = idct8_basis(sample_idx, freq);
        }
    }
    table
}

fn idct8_basis(sample_idx: usize, freq: usize) -> f32 {
    let scale = if freq == 0 {
        (1.0_f32 / 8.0).sqrt()
    } else {
        (2.0_f32 / 8.0).sqrt()
    };
    scale * (((sample_idx as f32 + 0.5) * freq as f32 * PI) / 8.0).cos()
}
