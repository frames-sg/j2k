// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal runtime for direct DCT-grid to one-level wavelet projection.

use std::sync::Arc;
use std::time::Instant;

use core::f32::consts::PI;
use core::mem::{size_of, size_of_val};

use j2k_core::{BackendKind, DeviceMemoryRange};
use j2k_metal_support::{
    checked_buffer_contents_slice, checked_buffer_contents_slice_mut, checked_command_queue,
    commit_and_wait, private_buffer, shared_buffer_for_len, shared_buffer_with_slice,
    system_default_device, MetalPipelineLoader,
};
use j2k_transcode::{
    htj2k97_subband_delta, htj2k97_subband_total_bitplanes, idct_blocks_to_signed_samples_rayon,
    DctGridToDwt53Job, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob,
    DctGridToReversibleDwt53Job, Dwt53TwoDimensional, Dwt97BatchStageTimings, Dwt97TwoDimensional,
    Htj2k97CodeBlockOptions, J2kSubBandType, PrequantizedHtj2k97CodeBlock,
    PrequantizedHtj2k97Component, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
    ResidentBufferRef, ResidentColorModel, ResidentComponentGeometry, ResidentDctCoefficientOrder,
    ResidentDctGridLayout, ResidentDwtSubband, ResidentDwtSubbandKind, ResidentDwtSubbandLayout,
    ResidentHandoffError, ResidentJpegDctGrid, ResidentSampleInfo, ResidentSampling,
    ReversibleDwt53FirstLevel,
};
use metal::{
    foreign_types::ForeignType, Buffer, CommandBufferRef, CommandQueue, ComputeCommandEncoderRef,
    ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

use crate::weights::{SparseDwt53WeightRows, SparseDwt97WeightRows, SparseWeightRow};
use crate::MetalTranscodeError;

fn shader_source() -> String {
    [
        r"
#include <metal_stdlib>
using namespace metal;
",
        j2k_codec_math::generated::DWT97_CONSTANTS_METAL,
        "\n",
        include_str!("dct97.metal"),
    ]
    .concat()
}
const METAL_DCT_KERNEL_FAILED: &str = "Metal DCT wavelet projection failed";
const METAL_DCT_RUNTIME_FAILED: &str = "Metal DCT wavelet runtime setup failed";
const METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID: &str =
    "Metal reversible DCT 5/3 job has unsupported grid geometry";
const METAL_DCT53_UNSUPPORTED_GRID: &str = "Metal DCT 5/3 job has unsupported grid geometry";
const METAL_DCT97_UNSUPPORTED_GRID: &str = "Metal DCT 9/7 job has unsupported grid geometry";
const METAL_RESIDENT_HANDOFF_VALIDATION_FAILED: &str =
    "Metal resident transcode handoff descriptor validation failed";
const DWT97_STAGED_MAX_AXIS: usize = 1024;
const DWT97_STAGED_ROWS_PER_GROUP: usize = 2;
const DWT97_STAGED_COLUMNS_PER_GROUP: usize = 4;
const DWT97_STAGED_THREADS_PER_GROUP: u64 = 256;
const DWT97_BLOCK_COEFFICIENTS: usize = 64;

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

#[derive(Clone, Default)]
/// Reusable Metal session for transcode-stage accelerator dispatch.
pub struct MetalTranscodeSession {
    device: Option<Device>,
    runtime: Option<Arc<MetalRuntime>>,
}

impl MetalTranscodeSession {
    /// Create a transcode session bound to an existing Metal device.
    pub fn new(device: Device) -> Self {
        Self {
            device: Some(device),
            runtime: None,
        }
    }

    /// Create a transcode session bound to the system default Metal device.
    pub fn system_default() -> Result<Self, MetalTranscodeError> {
        system_default_device()
            .map(Self::new)
            .map_err(|_| MetalTranscodeError::MetalUnavailable)
    }

    fn runtime(&mut self) -> Result<Arc<MetalRuntime>, MetalTranscodeError> {
        if let Some(runtime) = &self.runtime {
            return Ok(Arc::clone(runtime));
        }
        let runtime = Arc::new(match &self.device {
            Some(device) => MetalRuntime::new_with_device(device.clone())?,
            None => MetalRuntime::new()?,
        });
        self.runtime = Some(Arc::clone(&runtime));
        Ok(runtime)
    }

    fn with_runtime<R>(
        &mut self,
        f: impl FnOnce(&MetalRuntime) -> Result<R, MetalTranscodeError>,
    ) -> Result<R, MetalTranscodeError> {
        let runtime = self.runtime()?;
        f(&runtime)
    }
}

impl core::fmt::Debug for MetalTranscodeSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalTranscodeSession")
            .field("device", &self.device.as_ref().map(|device| device.name()))
            .field("runtime_initialized", &self.runtime.is_some())
            .finish()
    }
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

// SAFETY: Metal ABI structs are repr(C) plain data matching shader layouts.
unsafe impl j2k_core::accelerator::GpuAbi for MetalSparseRow {
    const NAME: &'static str = "MetalSparseRow";
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalWeightTap {
    sample_idx: u32,
    weight: f32,
}

// SAFETY: Metal ABI structs are repr(C) plain data matching shader layouts.
unsafe impl j2k_core::accelerator::GpuAbi for MetalWeightTap {
    const NAME: &'static str = "MetalWeightTap";
}

struct MetalSparseRows {
    rows: Vec<MetalSparseRow>,
    taps: Vec<MetalWeightTap>,
}

impl MetalRuntime {
    fn new() -> Result<Self, MetalTranscodeError> {
        let device = system_default_device().map_err(|_| MetalTranscodeError::MetalUnavailable)?;
        Self::new_with_device(device)
    }

    fn new_with_device(device: Device) -> Result<Self, MetalTranscodeError> {
        let shader_source = shader_source();
        let loader = MetalPipelineLoader::new(&device, &shader_source)
            .map_err(|_| MetalTranscodeError::Runtime(METAL_DCT_RUNTIME_FAILED))?;
        let pipeline = |name| {
            loader
                .pipeline(name)
                .map_err(|_| MetalTranscodeError::Runtime(METAL_DCT_RUNTIME_FAILED))
        };
        let dct_project_band = pipeline("dct97_project_band")?;
        let dct_project_band_batch = pipeline("dct97_project_band_batch")?;
        let dct97_idct_row_lift_batch = pipeline("dct97_idct_row_lift_batch")?;
        let dct97_column_lift_batch = pipeline("dct97_column_lift_batch")?;
        let dct97_quantize_codeblocks_batch = pipeline("dct97_quantize_codeblocks_batch")?;
        let reversible53_project_band = pipeline("reversible53_project_band")?;
        let queue = checked_command_queue(&device)
            .map_err(|_| MetalTranscodeError::Runtime(METAL_DCT_RUNTIME_FAILED))?;
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

fn encode_dwt97_staged_row_lift(
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

fn encode_dwt97_quantize_codeblocks(
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
    dispatch_projection_threads(
        encoder,
        band.width as u64,
        band.height as u64,
        u64::from(band.batch_count),
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

#[inline]
fn projection_thread_grid(width: u64, height: u64, depth: u64) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

#[inline]
fn projection_threads_per_group() -> MTLSize {
    projection_thread_grid(16, 8, 1)
}

#[inline]
fn projection_dispatch_sizes(width: u64, height: u64, depth: u64) -> (MTLSize, MTLSize) {
    (
        projection_thread_grid(width, height, depth),
        projection_threads_per_group(),
    )
}

#[inline]
fn dispatch_projection_threads(
    encoder: &ComputeCommandEncoderRef,
    width: u64,
    height: u64,
    depth: u64,
) {
    let (threads, threads_per_group) = projection_dispatch_sizes(width, height, depth);
    encoder.dispatch_threads(threads, threads_per_group);
}

#[inline]
fn bind_projection_input_buffers(
    encoder: &ComputeCommandEncoderRef,
    blocks: &Buffer,
    idct_basis: &Buffer,
) {
    encoder.set_buffer(0, Some(blocks), 0);
    encoder.set_buffer(5, Some(idct_basis), 0);
}

#[inline]
fn bind_projection_band_buffers(
    encoder: &ComputeCommandEncoderRef,
    x_weights: (&Buffer, &Buffer),
    y_weights: (&Buffer, &Buffer),
    output: &Buffer,
) {
    encoder.set_buffer(1, Some(x_weights.0), 0);
    encoder.set_buffer(2, Some(x_weights.1), 0);
    encoder.set_buffer(3, Some(y_weights.0), 0);
    encoder.set_buffer(4, Some(y_weights.1), 0);
    encoder.set_buffer(6, Some(output), 0);
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
    let blocks = dwt97_blocks_buffer(&runtime.device, job.blocks)?;

    let ll_buffer = output_buffer(&runtime.device, low_width * low_height);
    let hl_buffer = output_buffer(&runtime.device, high_width * low_height);
    let lh_buffer = output_buffer(&runtime.device, low_width * high_height);
    let hh_buffer = output_buffer(&runtime.device, high_width * high_height);

    let command_buffer = runtime.queue.new_command_buffer();
    command_buffer.set_label(job.label);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.dct_project_band);
    bind_projection_input_buffers(encoder, &blocks, &runtime.idct_basis);

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
    commit_and_wait(command_buffer)
        .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;

    Ok(ProjectedBands {
        ll: read_f32_buffer(&ll_buffer, low_width * low_height)?,
        hl: read_f32_buffer(&hl_buffer, high_width * low_height)?,
        lh: read_f32_buffer(&lh_buffer, low_width * high_height)?,
        hh: read_f32_buffer(&hh_buffer, high_width * high_height)?,
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
    let blocks = dwt97_batch_blocks_buffer(&runtime.device, job.jobs)?;
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

#[derive(Clone, Copy)]
struct ResidentDwtBand<'a> {
    buffer: &'a Buffer,
    kind: ResidentDwtSubbandKind,
    width: usize,
    height: usize,
    values_per_item: usize,
}

struct Dwt97CodeBlockOutputBuffers {
    ll: Buffer,
    hl: Buffer,
    lh: Buffer,
    hh: Buffer,
}

fn validate_resident_dct_handoffs_for_dwt97_jobs(
    blocks: &Buffer,
    jobs: &[DctGridToDwt97Job<'_>],
) -> Result<usize, MetalTranscodeError> {
    validate_resident_dct_handoffs(
        blocks,
        jobs.iter().enumerate().map(|(index, job)| {
            (
                index,
                job.blocks.len(),
                job.block_cols,
                job.block_rows,
                job.width,
                job.height,
                1,
                1,
            )
        }),
    )
}

fn validate_resident_dct_handoffs_for_htj2k_jobs(
    blocks: &Buffer,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
) -> Result<usize, MetalTranscodeError> {
    validate_resident_dct_handoffs(
        blocks,
        jobs.iter().enumerate().map(|(index, job)| {
            (
                index,
                job.blocks.len(),
                job.block_cols,
                job.block_rows,
                job.width,
                job.height,
                job.x_rsiz,
                job.y_rsiz,
            )
        }),
    )
}

fn validate_resident_dct_handoffs(
    blocks: &Buffer,
    jobs: impl Iterator<Item = (usize, usize, usize, usize, usize, usize, u8, u8)>,
) -> Result<usize, MetalTranscodeError> {
    let mut count = 0usize;
    let mut byte_offset = 0usize;
    for (component_index, block_count, block_cols, block_rows, width, height, x_rsiz, y_rsiz) in
        jobs
    {
        let value_count = dwt97_block_value_count(block_count)?;
        let byte_len = checked_byte_count(value_count, size_of::<f32>())?;
        let buffer = resident_buffer_ref(blocks, byte_offset, byte_len)?;
        let component =
            resident_component_geometry(component_index, width, height, x_rsiz, y_rsiz)?;
        let sample = resident_result(ResidentSampleInfo::new(32, true))?;
        let row_coefficients = block_cols.checked_mul(DWT97_BLOCK_COEFFICIENTS).ok_or(
            MetalTranscodeError::UnsupportedJob(METAL_DCT97_UNSUPPORTED_GRID),
        )?;
        let row_pitch_bytes = checked_byte_count(row_coefficients, size_of::<f32>())?;
        resident_result(ResidentJpegDctGrid::new(
            buffer,
            component,
            sample,
            ResidentColorModel::Unknown,
            ResidentDctGridLayout {
                block_cols: u32_param(block_cols, METAL_DCT97_UNSUPPORTED_GRID)?,
                block_rows: u32_param(block_rows, METAL_DCT97_UNSUPPORTED_GRID)?,
                row_pitch_bytes,
                bytes_per_coefficient: size_of::<f32>(),
                coefficient_order: ResidentDctCoefficientOrder::Natural,
            },
        ))?
        .require_backend(BackendKind::Metal)
        .map_err(resident_handoff_error)?;
        count = count.saturating_add(1);
        byte_offset =
            byte_offset
                .checked_add(byte_len)
                .ok_or(MetalTranscodeError::UnsupportedJob(
                    METAL_DCT97_UNSUPPORTED_GRID,
                ))?;
    }
    Ok(count)
}

fn validate_resident_dwt_handoffs_for_dwt97_jobs(
    buffers: &ProjectionBatchOutputBuffers,
    jobs: &[DctGridToDwt97Job<'_>],
    shape: ProjectionBatchShape,
) -> Result<usize, MetalTranscodeError> {
    validate_resident_dwt_handoffs(
        buffers,
        shape,
        jobs.iter()
            .enumerate()
            .map(|(index, job)| (index, job.width, job.height, 1, 1)),
    )
}

fn validate_resident_dwt_handoffs_for_htj2k_jobs(
    buffers: &ProjectionBatchOutputBuffers,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    shape: ProjectionBatchShape,
) -> Result<usize, MetalTranscodeError> {
    validate_resident_dwt_handoffs(
        buffers,
        shape,
        jobs.iter()
            .enumerate()
            .map(|(index, job)| (index, job.width, job.height, job.x_rsiz, job.y_rsiz)),
    )
}

fn validate_resident_dwt_handoffs(
    buffers: &ProjectionBatchOutputBuffers,
    shape: ProjectionBatchShape,
    jobs: impl Iterator<Item = (usize, usize, usize, u8, u8)>,
) -> Result<usize, MetalTranscodeError> {
    let bands = [
        ResidentDwtBand {
            buffer: &buffers.ll,
            kind: ResidentDwtSubbandKind::LowLow,
            width: shape.low_width,
            height: shape.low_height,
            values_per_item: shape.ll_len,
        },
        ResidentDwtBand {
            buffer: &buffers.hl,
            kind: ResidentDwtSubbandKind::HighLow,
            width: shape.high_width,
            height: shape.low_height,
            values_per_item: shape.hl_len,
        },
        ResidentDwtBand {
            buffer: &buffers.lh,
            kind: ResidentDwtSubbandKind::LowHigh,
            width: shape.low_width,
            height: shape.high_height,
            values_per_item: shape.lh_len,
        },
        ResidentDwtBand {
            buffer: &buffers.hh,
            kind: ResidentDwtSubbandKind::HighHigh,
            width: shape.high_width,
            height: shape.high_height,
            values_per_item: shape.hh_len,
        },
    ];
    let mut count = 0usize;
    for (batch_index, width, height, x_rsiz, y_rsiz) in jobs {
        let component = resident_component_geometry(batch_index, width, height, x_rsiz, y_rsiz)?;
        for band in bands {
            if band.width == 0 || band.height == 0 {
                continue;
            }
            let item_offset = checked_byte_count(
                batch_index.checked_mul(band.values_per_item).ok_or(
                    MetalTranscodeError::UnsupportedJob(METAL_DCT97_UNSUPPORTED_GRID),
                )?,
                size_of::<f32>(),
            )?;
            let byte_len = checked_byte_count(band.values_per_item, size_of::<f32>())?;
            let row_pitch_bytes = checked_byte_count(band.width, size_of::<f32>())?;
            let buffer = resident_buffer_ref(band.buffer, item_offset, byte_len)?;
            let sample = resident_result(ResidentSampleInfo::new(32, true))?;
            resident_result(ResidentDwtSubband::new(
                buffer,
                component,
                sample,
                ResidentColorModel::Unknown,
                ResidentDwtSubbandLayout {
                    level: 1,
                    subband: band.kind,
                    width: u32_param(band.width, METAL_DCT97_UNSUPPORTED_GRID)?,
                    height: u32_param(band.height, METAL_DCT97_UNSUPPORTED_GRID)?,
                    row_pitch_bytes,
                    bytes_per_coefficient: size_of::<f32>(),
                },
            ))?
            .require_backend(BackendKind::Metal)
            .map_err(resident_handoff_error)?;
            count = count.saturating_add(1);
        }
    }
    Ok(count)
}

fn resident_component_geometry(
    component_index: usize,
    width: usize,
    height: usize,
    x_rsiz: u8,
    y_rsiz: u8,
) -> Result<ResidentComponentGeometry, MetalTranscodeError> {
    let sampling = resident_result(ResidentSampling::new(x_rsiz, y_rsiz))?;
    resident_result(ResidentComponentGeometry::new(
        component_index,
        u32_param(width, METAL_DCT97_UNSUPPORTED_GRID)?,
        u32_param(height, METAL_DCT97_UNSUPPORTED_GRID)?,
        sampling,
    ))
}

fn resident_buffer_ref(
    buffer: &Buffer,
    offset: usize,
    len: usize,
) -> Result<ResidentBufferRef<'_>, MetalTranscodeError> {
    let allocation = u64::try_from(buffer.as_ptr() as usize)
        .map_err(|_| MetalTranscodeError::Kernel(METAL_RESIDENT_HANDOFF_VALIDATION_FAILED))?;
    let allocation_len = usize::try_from(buffer.length())
        .map_err(|_| MetalTranscodeError::Kernel(METAL_RESIDENT_HANDOFF_VALIDATION_FAILED))?;
    resident_result(ResidentBufferRef::with_allocation_len(
        DeviceMemoryRange::new(BackendKind::Metal, allocation, offset, len),
        allocation_len,
    ))
}

fn checked_byte_count(
    value_count: usize,
    bytes_per_value: usize,
) -> Result<usize, MetalTranscodeError> {
    value_count
        .checked_mul(bytes_per_value)
        .ok_or(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ))
}

fn resident_result<T>(result: Result<T, ResidentHandoffError>) -> Result<T, MetalTranscodeError> {
    result.map_err(resident_handoff_error)
}

fn resident_handoff_error(_error: ResidentHandoffError) -> MetalTranscodeError {
    MetalTranscodeError::Kernel(METAL_RESIDENT_HANDOFF_VALIDATION_FAILED)
}

fn projection_batch_output_transfer_count(buffers: &ProjectionBatchOutputBuffers) -> usize {
    [
        buffers.ll.length(),
        buffers.hl.length(),
        buffers.lh.length(),
        buffers.hh.length(),
    ]
    .into_iter()
    .filter(|bytes| *bytes > 0)
    .count()
}

fn projection_batch_output_transfer_bytes(buffers: &ProjectionBatchOutputBuffers) -> u64 {
    [
        buffers.ll.length(),
        buffers.hl.length(),
        buffers.lh.length(),
        buffers.hh.length(),
    ]
    .into_iter()
    .fold(0_u64, u64::saturating_add)
}

fn dwt97_codeblock_output_transfer_count(buffers: &Dwt97CodeBlockOutputBuffers) -> usize {
    [
        buffers.ll.length(),
        buffers.hl.length(),
        buffers.lh.length(),
        buffers.hh.length(),
    ]
    .into_iter()
    .filter(|bytes| *bytes > 0)
    .count()
}

fn dwt97_codeblock_output_transfer_bytes(buffers: &Dwt97CodeBlockOutputBuffers) -> u64 {
    [
        buffers.ll.length(),
        buffers.hl.length(),
        buffers.lh.length(),
        buffers.hh.length(),
    ]
    .into_iter()
    .fold(0_u64, u64::saturating_add)
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

fn projection_batch_private_output_buffers(
    runtime: &MetalRuntime,
    shape: ProjectionBatchShape,
    unsupported_grid: &'static str,
) -> Result<ProjectionBatchOutputBuffers, MetalTranscodeError> {
    Ok(ProjectionBatchOutputBuffers {
        ll: private_f32_buffer(
            &runtime.device,
            checked_batch_len(shape.ll_len, shape.batch_count, unsupported_grid)?,
        ),
        hl: private_f32_buffer(
            &runtime.device,
            checked_batch_len(shape.hl_len, shape.batch_count, unsupported_grid)?,
        ),
        lh: private_f32_buffer(
            &runtime.device,
            checked_batch_len(shape.lh_len, shape.batch_count, unsupported_grid)?,
        ),
        hh: private_f32_buffer(
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
    bind_projection_input_buffers(encoder, blocks, &runtime.idct_basis);

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
    commit_and_wait(command_buffer)
        .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))?;
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
    )?;
    let hl = shared_f32_slice(
        &buffers.hl,
        checked_batch_len(shape.hl_len, shape.batch_count, unsupported_grid)?,
    )?;
    let lh = shared_f32_slice(
        &buffers.lh,
        checked_batch_len(shape.lh_len, shape.batch_count, unsupported_grid)?,
    )?;
    let hh = shared_f32_slice(
        &buffers.hh,
        checked_batch_len(shape.hh_len, shape.batch_count, unsupported_grid)?,
    )?;

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
    )?;
    let hl = shared_i32_slice(
        &buffers.hl,
        checked_batch_len(shape.hl_len, shape.batch_count, unsupported_grid)?,
    )?;
    let lh = shared_i32_slice(
        &buffers.lh,
        checked_batch_len(shape.lh_len, shape.batch_count, unsupported_grid)?,
    )?;
    let hh = shared_i32_slice(
        &buffers.hh,
        checked_batch_len(shape.hh_len, shape.batch_count, unsupported_grid)?,
    )?;

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
    dispatch_projection_threads(
        encoder,
        u64::from(geometry.band_width),
        u64::from(geometry.band_height),
        u64::from(geometry.batch_count),
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
    // Shared with CUDA so the two backends accept/reject identical options.
    // Option failures keep their own message instead of the grid-geometry one.
    j2k_transcode::validate_htj2k97_codeblock_options(options)
        .map(|_| ())
        .map_err(MetalTranscodeError::UnsupportedJob)
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
    htj2k97_subband_total_bitplanes(options, sub_band_type)
}

fn dwt97_quantize_inv_delta(
    options: Htj2k97CodeBlockOptions,
    sub_band_type: J2kSubBandType,
) -> f32 {
    (1.0 / htj2k97_subband_delta(options, sub_band_type)) as f32
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

fn buffer_with_slice<T: j2k_core::accelerator::GpuAbi>(device: &Device, values: &[T]) -> Buffer {
    shared_buffer_with_slice(device, values)
}

fn dwt97_blocks_buffer(
    device: &Device,
    blocks: &[[[f64; 8]; 8]],
) -> Result<Buffer, MetalTranscodeError> {
    let value_count = dwt97_block_value_count(blocks.len())?;
    let mut buffer = output_buffer(device, value_count);
    write_dwt97_blocks_to_buffer(&mut buffer, blocks)?;
    Ok(buffer)
}

fn dwt97_batch_blocks_buffer(
    device: &Device,
    jobs: &[DctGridToDwt97Job<'_>],
) -> Result<Buffer, MetalTranscodeError> {
    let value_count = dwt97_jobs_value_count(jobs.iter().map(|job| job.blocks.len()))?;
    let mut buffer = output_buffer(device, value_count);
    let mut offset = 0;
    for job in jobs {
        offset += write_dwt97_blocks_to_buffer_at(&mut buffer, offset, job.blocks)?;
    }
    debug_assert_eq!(offset, value_count);
    Ok(buffer)
}

fn dwt97_codeblock_batch_blocks_buffer(
    device: &Device,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
) -> Result<Buffer, MetalTranscodeError> {
    let value_count = dwt97_jobs_value_count(jobs.iter().map(|job| job.blocks.len()))?;
    let mut buffer = output_buffer(device, value_count);
    let mut offset = 0;
    for job in jobs {
        offset += write_dwt97_blocks_to_buffer_at(&mut buffer, offset, job.blocks)?;
    }
    debug_assert_eq!(offset, value_count);
    Ok(buffer)
}

fn dwt97_jobs_value_count(
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

fn dwt97_block_value_count(block_count: usize) -> Result<usize, MetalTranscodeError> {
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

fn write_dwt97_blocks_to_buffer(
    buffer: &mut Buffer,
    blocks: &[[[f64; 8]; 8]],
) -> Result<(), MetalTranscodeError> {
    let written = write_dwt97_blocks_to_buffer_at(buffer, 0, blocks)?;
    debug_assert_eq!(written, dwt97_block_value_count(blocks.len())?);
    Ok(())
}

fn write_dwt97_blocks_to_buffer_at(
    buffer: &mut Buffer,
    start: usize,
    blocks: &[[[f64; 8]; 8]],
) -> Result<usize, MetalTranscodeError> {
    let value_count = dwt97_block_value_count(blocks.len())?;
    let end = start
        .checked_add(value_count)
        .ok_or(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ))?;
    if end > buffer_f32_capacity(buffer) {
        return Err(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ));
    }

    let byte_offset =
        start
            .checked_mul(size_of::<f32>())
            .ok_or(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID,
            ))?;
    let values = checked_buffer_contents_slice_mut::<f32>(buffer, byte_offset, value_count)
        .map_err(|_| MetalTranscodeError::UnsupportedJob(METAL_DCT97_UNSUPPORTED_GRID))?;
    let mut offset = 0usize;
    for block in blocks {
        for row in block {
            for &coefficient in row {
                values[offset] = coefficient as f32;
                offset += 1;
            }
        }
    }
    Ok(offset)
}

fn buffer_f32_capacity(buffer: &Buffer) -> usize {
    let element_size = size_of::<f32>() as u64;
    usize::try_from(buffer.length() / element_size).unwrap_or(usize::MAX)
}

fn output_buffer(device: &Device, value_count: usize) -> Buffer {
    shared_buffer_for_len::<f32>(device, value_count)
}

fn private_f32_buffer(device: &Device, value_count: usize) -> Buffer {
    private_buffer(device, value_count.saturating_mul(size_of::<f32>()))
}

fn output_i32_buffer(device: &Device, value_count: usize) -> Buffer {
    shared_buffer_for_len::<i32>(device, value_count)
}

fn read_f32_buffer(buffer: &Buffer, value_count: usize) -> Result<Vec<f64>, MetalTranscodeError> {
    shared_f32_slice(buffer, value_count).map(f32_slice_to_f64)
}

fn read_i32_buffer(buffer: &Buffer, value_count: usize) -> Result<Vec<i32>, MetalTranscodeError> {
    shared_i32_slice(buffer, value_count).map(<[i32]>::to_vec)
}

fn shared_f32_slice(buffer: &Buffer, value_count: usize) -> Result<&[f32], MetalTranscodeError> {
    if value_count == 0 {
        return Ok(&[]);
    }
    checked_buffer_contents_slice(buffer, 0, value_count)
        .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))
}

fn shared_i32_slice(buffer: &Buffer, value_count: usize) -> Result<&[i32], MetalTranscodeError> {
    if value_count == 0 {
        return Ok(&[]);
    }
    checked_buffer_contents_slice(buffer, 0, value_count)
        .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT_KERNEL_FAILED))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_dispatch_sizes_use_16_by_8_threadgroups() {
        let (threads, threadgroup) = projection_dispatch_sizes(5, 6, 7);

        assert_eq!((threads.width, threads.height, threads.depth), (5, 6, 7));
        assert_eq!(
            (threadgroup.width, threadgroup.height, threadgroup.depth),
            (16, 8, 1)
        );
    }

    #[test]
    fn dwt97_block_value_count_rejects_overflow() {
        assert_eq!(dwt97_block_value_count(2), Ok(128));
        assert_eq!(
            dwt97_block_value_count(usize::MAX),
            Err(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID
            ))
        );
    }
}
