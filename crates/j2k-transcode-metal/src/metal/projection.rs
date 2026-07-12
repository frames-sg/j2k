// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_command_buffer, checked_compute_command_encoder, commit_and_wait, dispatch_band,
    dispatch_projection_batch_bands, dwt97_batch_blocks_buffer, dwt97_blocks_buffer, output_buffer,
    projection_batch_output_buffers, projection_batch_shape, projection_batch_weight_buffers,
    read_f32_buffer, read_projected_batch_outputs, u32_param, upload_sparse_rows, BandGeometry,
    Buffer, ComputeCommandEncoderRef, DctGridToDwt97Job, MTLSize, MetalRuntime,
    MetalTranscodeError, SparseWeightRow, DWT97_STAGED_THREADS_PER_GROUP,
};

pub(super) fn staged_threads_per_group() -> MTLSize {
    MTLSize {
        width: DWT97_STAGED_THREADS_PER_GROUP,
        height: 1,
        depth: 1,
    }
}

#[inline]
pub(super) fn projection_thread_grid(width: u64, height: u64, depth: u64) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

#[inline]
pub(super) fn projection_threads_per_group() -> MTLSize {
    projection_thread_grid(16, 8, 1)
}

#[inline]
pub(super) fn projection_dispatch_sizes(width: u64, height: u64, depth: u64) -> (MTLSize, MTLSize) {
    (
        projection_thread_grid(width, height, depth),
        projection_threads_per_group(),
    )
}

#[inline]
pub(super) fn dispatch_projection_threads(
    encoder: &ComputeCommandEncoderRef,
    width: u64,
    height: u64,
    depth: u64,
) {
    let (threads, threads_per_group) = projection_dispatch_sizes(width, height, depth);
    encoder.dispatch_threads(threads, threads_per_group);
}

#[inline]
pub(super) fn bind_projection_input_buffers(
    encoder: &ComputeCommandEncoderRef,
    blocks: &Buffer,
    idct_basis: &Buffer,
) {
    encoder.set_buffer(0, Some(blocks), 0);
    encoder.set_buffer(5, Some(idct_basis), 0);
}

#[inline]
pub(super) fn bind_projection_band_buffers(
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
pub(super) struct ProjectionJob<'a> {
    pub(super) blocks: &'a [[[f64; 8]; 8]],
    pub(super) block_cols: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) x_low: &'a [SparseWeightRow],
    pub(super) x_high: &'a [SparseWeightRow],
    pub(super) y_low: &'a [SparseWeightRow],
    pub(super) y_high: &'a [SparseWeightRow],
    pub(super) unsupported_grid: &'static str,
    pub(super) label: &'static str,
}

#[derive(Clone, Copy)]
pub(super) struct ProjectionBatchJob<'a, 'b> {
    pub(super) jobs: &'a [DctGridToDwt97Job<'b>],
    pub(super) block_cols: usize,
    pub(super) block_rows: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) x_low: &'a [SparseWeightRow],
    pub(super) x_high: &'a [SparseWeightRow],
    pub(super) y_low: &'a [SparseWeightRow],
    pub(super) y_high: &'a [SparseWeightRow],
    pub(super) unsupported_grid: &'static str,
    pub(super) label: &'static str,
}

pub(super) struct ProjectedBands {
    pub(super) ll: Vec<f64>,
    pub(super) hl: Vec<f64>,
    pub(super) lh: Vec<f64>,
    pub(super) hh: Vec<f64>,
    pub(super) low_width: usize,
    pub(super) low_height: usize,
    pub(super) high_width: usize,
    pub(super) high_height: usize,
}

#[expect(
    clippy::similar_names,
    reason = "LL, HL, LH, and HH are standard wavelet subband names"
)]
pub(super) fn dispatch_projected_bands_with_runtime(
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
    let ll_len = checked_band_len(low_width, low_height, job.unsupported_grid)?;
    let hl_len = checked_band_len(high_width, low_height, job.unsupported_grid)?;
    let lh_len = checked_band_len(low_width, high_height, job.unsupported_grid)?;
    let hh_len = checked_band_len(high_width, high_height, job.unsupported_grid)?;

    let (x_low_rows, x_low_taps) =
        upload_sparse_rows(&runtime.device, job.x_low, job.unsupported_grid)?;
    let (x_high_rows, x_high_taps) =
        upload_sparse_rows(&runtime.device, job.x_high, job.unsupported_grid)?;
    let (y_low_rows, y_low_taps) =
        upload_sparse_rows(&runtime.device, job.y_low, job.unsupported_grid)?;
    let (y_high_rows, y_high_taps) =
        upload_sparse_rows(&runtime.device, job.y_high, job.unsupported_grid)?;
    let blocks = dwt97_blocks_buffer(&runtime.device, job.blocks)?;

    let ll_buffer = output_buffer(&runtime.device, ll_len)?;
    let hl_buffer = output_buffer(&runtime.device, hl_len)?;
    let lh_buffer = output_buffer(&runtime.device, lh_len)?;
    let hh_buffer = output_buffer(&runtime.device, hh_len)?;

    let command_buffer = checked_command_buffer(&runtime.queue).map_err(|error| {
        MetalTranscodeError::support("Metal projected-band command buffer creation", error)
    })?;
    command_buffer.set_label(job.label);
    let encoder = checked_compute_command_encoder(&command_buffer).map_err(|error| {
        MetalTranscodeError::support("Metal projected-band compute encoder creation", error)
    })?;
    encoder.set_compute_pipeline_state(&runtime.dct_project_band);
    bind_projection_input_buffers(&encoder, &blocks, &runtime.idct_basis);

    dispatch_band(
        &encoder,
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
        &encoder,
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
        &encoder,
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
        &encoder,
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
    commit_and_wait(&command_buffer).map_err(|error| {
        MetalTranscodeError::support("Metal projected-band command buffer", error)
    })?;

    Ok(ProjectedBands {
        ll: read_f32_buffer(&ll_buffer, ll_len)?,
        hl: read_f32_buffer(&hl_buffer, hl_len)?,
        lh: read_f32_buffer(&lh_buffer, lh_len)?,
        hh: read_f32_buffer(&hh_buffer, hh_len)?,
        low_width,
        low_height,
        high_width,
        high_height,
    })
}

fn checked_band_len(
    width: usize,
    height: usize,
    unsupported_grid: &'static str,
) -> Result<usize, MetalTranscodeError> {
    width
        .checked_mul(height)
        .ok_or(MetalTranscodeError::UnsupportedJob(unsupported_grid))
}

pub(super) fn dispatch_projected_bands_batch_with_runtime(
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
