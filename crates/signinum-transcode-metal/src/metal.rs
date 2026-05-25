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
    DctGridToReversibleDwt53Job, Dwt97BatchStageTimings, ReversibleDwt53FirstLevel,
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

fn dispatch_dct_grid_to_dwt97_batch_staged_with_runtime(
    runtime: &MetalRuntime,
    jobs: &[DctGridToDwt97Job<'_>],
    first: &DctGridToDwt97Job<'_>,
) -> Result<(Vec<Dwt97TwoDimensional<f64>>, Dwt97BatchStageTimings), MetalTranscodeError> {
    let shape = dwt97_staged_batch_shape(jobs, first)?;
    let mut timings = Dwt97BatchStageTimings::default();

    let pack_upload_start = Instant::now();
    let flat_blocks = flatten_batch_blocks(jobs);
    let blocks = buffer_with_slice(&runtime.device, &flat_blocks);
    let row_buffers = dwt97_staged_row_buffers(runtime, shape)?;
    let output_buffers =
        projection_batch_output_buffers(runtime, shape, METAL_DCT97_UNSUPPORTED_GRID)?;
    timings.pack_upload_us = pack_upload_start.elapsed().as_micros();

    let row_start = Instant::now();
    dispatch_dwt97_staged_row_lift(runtime, first, shape, &blocks, &row_buffers)?;
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
    first: &DctGridToDwt97Job<'_>,
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
    let row_groups = first.height.div_ceil(DWT97_STAGED_ROWS_PER_GROUP);

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
    let blocks = buffer_with_slice(&runtime.device, &flatten_blocks(job.blocks));

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
    let blocks = buffer_with_slice(&runtime.device, &flatten_batch_blocks(job.jobs));
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
    let ll = read_f32_buffer(
        &buffers.ll,
        checked_batch_len(shape.ll_len, shape.batch_count, unsupported_grid)?,
    );
    let hl = read_f32_buffer(
        &buffers.hl,
        checked_batch_len(shape.hl_len, shape.batch_count, unsupported_grid)?,
    );
    let lh = read_f32_buffer(
        &buffers.lh,
        checked_batch_len(shape.lh_len, shape.batch_count, unsupported_grid)?,
    );
    let hh = read_f32_buffer(
        &buffers.hh,
        checked_batch_len(shape.hh_len, shape.batch_count, unsupported_grid)?,
    );

    let mut outputs = Vec::with_capacity(shape.batch_count);
    for idx in 0..shape.batch_count {
        outputs.push(ProjectedBands {
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

fn flatten_blocks(blocks: &[[[f64; 8]; 8]]) -> Vec<f32> {
    blocks
        .iter()
        .flat_map(|block| {
            block
                .iter()
                .flat_map(|row| row.iter().map(|&coefficient| coefficient as f32))
        })
        .collect()
}

fn flatten_batch_blocks(jobs: &[DctGridToDwt97Job<'_>]) -> Vec<f32> {
    let value_count = jobs
        .iter()
        .map(|job| job.blocks.len().saturating_mul(64))
        .sum();
    let mut output = Vec::with_capacity(value_count);
    for job in jobs {
        for block in job.blocks {
            for row in block {
                output.extend(row.iter().map(|&coefficient| coefficient as f32));
            }
        }
    }
    output
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
    if value_count == 0 {
        return Vec::new();
    }
    let values =
        unsafe { core::slice::from_raw_parts(buffer.contents().cast::<f32>(), value_count) };
    values.iter().map(|&value| f64::from(value)).collect()
}

fn read_i32_buffer(buffer: &Buffer, value_count: usize) -> Vec<i32> {
    if value_count == 0 {
        return Vec::new();
    }
    let values =
        unsafe { core::slice::from_raw_parts(buffer.contents().cast::<i32>(), value_count) };
    values.to_vec()
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
