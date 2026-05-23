// SPDX-License-Identifier: Apache-2.0

//! Metal runtime for direct DCT-grid to one-level wavelet projection.

use std::sync::{Arc, OnceLock};

use core::f32::consts::PI;
use core::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};
use signinum_transcode::accelerator::{
    idct_blocks_to_signed_samples_rayon, DctGridToDwt53Job, DctGridToDwt97Job,
    DctGridToReversibleDwt53Job, ReversibleDwt53FirstLevel,
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

static METAL_RUNTIME: OnceLock<Result<Arc<MetalRuntime>, MetalTranscodeError>> = OnceLock::new();

struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    dct_project_band: ComputePipelineState,
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
struct Reversible53ProjectionParams {
    width: u32,
    height: u32,
    block_cols: u32,
    band_width: u32,
    band_height: u32,
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
            reversible53_project_band,
            idct_basis,
        })
    }
}

pub(crate) fn dispatch_dct_grid_to_reversible_dwt53(
    job: DctGridToReversibleDwt53Job<'_>,
) -> Result<ReversibleDwt53FirstLevel, MetalTranscodeError> {
    validate_grid(
        job.dequantized_blocks.len(),
        job.block_cols,
        job.block_rows,
        job.width,
        job.height,
        METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
    )?;
    let block_samples = idct_blocks_to_signed_samples_rayon(job.dequantized_blocks);
    with_runtime(|runtime| {
        dispatch_reversible_dwt53_with_runtime(
            runtime,
            &block_samples,
            job.block_cols,
            job.width,
            job.height,
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
fn dispatch_reversible_dwt53_with_runtime(
    runtime: &MetalRuntime,
    block_samples: &[[i32; 64]],
    block_cols: usize,
    width: usize,
    height: usize,
) -> Result<ReversibleDwt53FirstLevel, MetalTranscodeError> {
    let width_u32 = u32_param(width, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?;
    let height_u32 = u32_param(height, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?;
    let block_cols_u32 = u32_param(block_cols, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?;
    let low_width = width.div_ceil(2);
    let high_width = width / 2;
    let low_height = height.div_ceil(2);
    let high_height = height / 2;
    let blocks = buffer_with_slice(&runtime.device, block_samples);

    let ll_buffer = output_i32_buffer(&runtime.device, low_width * low_height);
    let hl_buffer = output_i32_buffer(&runtime.device, high_width * low_height);
    let lh_buffer = output_i32_buffer(&runtime.device, low_width * high_height);
    let hh_buffer = output_i32_buffer(&runtime.device, high_width * high_height);

    let command_buffer = runtime.queue.new_command_buffer();
    command_buffer.set_label("signinum-transcode-metal reversible dct53 projection");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.reversible53_project_band);
    encoder.set_buffer(0, Some(&blocks), 0);

    dispatch_reversible_band(
        encoder,
        &ll_buffer,
        ReversibleBandGeometry {
            width: width_u32,
            height: height_u32,
            block_cols: block_cols_u32,
            band_width: u32_param(low_width, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
            band_height: u32_param(low_height, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
            vertical_low: true,
            horizontal_low: true,
        },
    );
    dispatch_reversible_band(
        encoder,
        &hl_buffer,
        ReversibleBandGeometry {
            width: width_u32,
            height: height_u32,
            block_cols: block_cols_u32,
            band_width: u32_param(high_width, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
            band_height: u32_param(low_height, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
            vertical_low: true,
            horizontal_low: false,
        },
    );
    dispatch_reversible_band(
        encoder,
        &lh_buffer,
        ReversibleBandGeometry {
            width: width_u32,
            height: height_u32,
            block_cols: block_cols_u32,
            band_width: u32_param(low_width, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
            band_height: u32_param(high_height, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
            vertical_low: false,
            horizontal_low: true,
        },
    );
    dispatch_reversible_band(
        encoder,
        &hh_buffer,
        ReversibleBandGeometry {
            width: width_u32,
            height: height_u32,
            block_cols: block_cols_u32,
            band_width: u32_param(high_width, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
            band_height: u32_param(high_height, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
            vertical_low: false,
            horizontal_low: false,
        },
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(ReversibleDwt53FirstLevel {
        ll: read_i32_buffer(&ll_buffer, low_width * low_height),
        hl: read_i32_buffer(&hl_buffer, high_width * low_height),
        lh: read_i32_buffer(&lh_buffer, low_width * high_height),
        hh: read_i32_buffer(&hh_buffer, high_width * high_height),
        low_width,
        low_height,
        high_width,
        high_height,
    })
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
struct ReversibleBandGeometry {
    width: u32,
    height: u32,
    block_cols: u32,
    band_width: u32,
    band_height: u32,
    vertical_low: bool,
    horizontal_low: bool,
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
        band_width: geometry.band_width,
        band_height: geometry.band_height,
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
            depth: 1,
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
