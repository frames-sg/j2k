// SPDX-License-Identifier: Apache-2.0

//! Metal runtime for direct DCT-grid to one-level 9/7 projection.

use std::sync::{Arc, OnceLock};

use core::f32::consts::PI;
use core::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};
use signinum_transcode::accelerator::DctGridToDwt97Job;
use signinum_transcode::dct97_2d::Dwt97TwoDimensional;

use crate::weights::Dwt97WeightRows;
use crate::MetalTranscodeError;

const SHADER_SOURCE: &str = include_str!("dct97.metal");
const METAL_DCT97_KERNEL_FAILED: &str = "Metal DCT 9/7 projection failed";
const METAL_DCT97_UNSUPPORTED_GRID: &str = "Metal DCT 9/7 job has unsupported grid geometry";

static METAL_RUNTIME: OnceLock<Result<Arc<MetalRuntime>, MetalTranscodeError>> = OnceLock::new();

struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    dct97_project_band: ComputePipelineState,
    idct_basis: Buffer,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Dct97ProjectionParams {
    width: u32,
    height: u32,
    block_cols: u32,
    band_width: u32,
    band_height: u32,
}

impl MetalRuntime {
    fn new() -> Result<Self, MetalTranscodeError> {
        let device = Device::system_default().ok_or(MetalTranscodeError::MetalUnavailable)?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT97_KERNEL_FAILED))?;
        let function = library
            .get_function("dct97_project_band", None)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT97_KERNEL_FAILED))?;
        let dct97_project_band = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| MetalTranscodeError::Kernel(METAL_DCT97_KERNEL_FAILED))?;
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
            dct97_project_band,
            idct_basis,
        })
    }
}

pub(crate) fn dispatch_dct_grid_to_dwt97(
    job: DctGridToDwt97Job<'_>,
) -> Result<Dwt97TwoDimensional<f64>, MetalTranscodeError> {
    validate_job(job)?;
    with_runtime(|runtime| dispatch_dct_grid_to_dwt97_with_runtime(runtime, job))
}

#[allow(clippy::similar_names)]
fn dispatch_dct_grid_to_dwt97_with_runtime(
    runtime: &MetalRuntime,
    job: DctGridToDwt97Job<'_>,
) -> Result<Dwt97TwoDimensional<f64>, MetalTranscodeError> {
    let width = u32_param(job.width)?;
    let height = u32_param(job.height)?;
    let block_cols = u32_param(job.block_cols)?;
    let low_width = job.width.div_ceil(2);
    let high_width = job.width / 2;
    let low_height = job.height.div_ceil(2);
    let high_height = job.height / 2;

    let x_weights = Dwt97WeightRows::for_len(job.width);
    let y_weights = Dwt97WeightRows::for_len(job.height);
    let x_low = buffer_with_slice(&runtime.device, &flatten_rows(&x_weights.low));
    let x_high = buffer_with_slice(&runtime.device, &flatten_rows(&x_weights.high));
    let y_low = buffer_with_slice(&runtime.device, &flatten_rows(&y_weights.low));
    let y_high = buffer_with_slice(&runtime.device, &flatten_rows(&y_weights.high));
    let blocks = buffer_with_slice(&runtime.device, &flatten_blocks(job.blocks));

    let ll_buffer = output_buffer(&runtime.device, low_width * low_height);
    let hl_buffer = output_buffer(&runtime.device, high_width * low_height);
    let lh_buffer = output_buffer(&runtime.device, low_width * high_height);
    let hh_buffer = output_buffer(&runtime.device, high_width * high_height);

    let command_buffer = runtime.queue.new_command_buffer();
    command_buffer.set_label("signinum-transcode-metal dct97 projection");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.dct97_project_band);
    encoder.set_buffer(0, Some(&blocks), 0);
    encoder.set_buffer(3, Some(&runtime.idct_basis), 0);

    dispatch_band(
        encoder,
        &x_low,
        &y_low,
        &ll_buffer,
        BandGeometry {
            width,
            height,
            block_cols,
            band_width: u32_param(low_width)?,
            band_height: u32_param(low_height)?,
        },
    );
    dispatch_band(
        encoder,
        &x_high,
        &y_low,
        &hl_buffer,
        BandGeometry {
            width,
            height,
            block_cols,
            band_width: u32_param(high_width)?,
            band_height: u32_param(low_height)?,
        },
    );
    dispatch_band(
        encoder,
        &x_low,
        &y_high,
        &lh_buffer,
        BandGeometry {
            width,
            height,
            block_cols,
            band_width: u32_param(low_width)?,
            band_height: u32_param(high_height)?,
        },
    );
    dispatch_band(
        encoder,
        &x_high,
        &y_high,
        &hh_buffer,
        BandGeometry {
            width,
            height,
            block_cols,
            band_width: u32_param(high_width)?,
            band_height: u32_param(high_height)?,
        },
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(Dwt97TwoDimensional {
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

fn dispatch_band(
    encoder: &ComputeCommandEncoderRef,
    x_weights: &Buffer,
    y_weights: &Buffer,
    output: &Buffer,
    geometry: BandGeometry,
) {
    if geometry.band_width == 0 || geometry.band_height == 0 {
        return;
    }

    let params = Dct97ProjectionParams {
        width: geometry.width,
        height: geometry.height,
        block_cols: geometry.block_cols,
        band_width: geometry.band_width,
        band_height: geometry.band_height,
    };
    encoder.set_buffer(1, Some(x_weights), 0);
    encoder.set_buffer(2, Some(y_weights), 0);
    encoder.set_buffer(4, Some(output), 0);
    encoder.set_bytes(
        5,
        size_of::<Dct97ProjectionParams>() as u64,
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

fn validate_job(job: DctGridToDwt97Job<'_>) -> Result<(), MetalTranscodeError> {
    let expected_blocks =
        job.block_cols
            .checked_mul(job.block_rows)
            .ok_or(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID,
            ))?;
    let covered_width =
        job.block_cols
            .checked_mul(8)
            .ok_or(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID,
            ))?;
    let covered_height =
        job.block_rows
            .checked_mul(8)
            .ok_or(MetalTranscodeError::UnsupportedJob(
                METAL_DCT97_UNSUPPORTED_GRID,
            ))?;

    if job.blocks.len() != expected_blocks
        || job.width == 0
        || job.height == 0
        || job.width > covered_width
        || job.height > covered_height
    {
        return Err(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ));
    }
    Ok(())
}

fn u32_param(value: usize) -> Result<u32, MetalTranscodeError> {
    u32::try_from(value)
        .map_err(|_| MetalTranscodeError::UnsupportedJob(METAL_DCT97_UNSUPPORTED_GRID))
}

fn flatten_rows(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flat_map(|row| row.iter().copied()).collect()
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

fn read_f32_buffer(buffer: &Buffer, value_count: usize) -> Vec<f64> {
    if value_count == 0 {
        return Vec::new();
    }
    let values =
        unsafe { core::slice::from_raw_parts(buffer.contents().cast::<f32>(), value_count) };
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
