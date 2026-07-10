// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    bind_projection_input_buffers, buffer_with_slice, checked_batch_len, code_block_len_from_exp,
    commit_and_wait, dispatch_band_batch, dwt97_block_value_count, dwt97_total_bitplanes,
    f32_slice_to_f64, metal_sparse_rows, output_buffer, output_i32_buffer, private_f32_buffer,
    shared_f32_slice, shared_i32_slice, size_of, u32_param, BackendKind, BatchBandGeometry, Buffer,
    DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob, DeviceMemoryRange, ForeignType,
    Htj2k97CodeBlockOptions, J2kSubBandType, MetalRuntime, MetalTranscodeError,
    PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component, PrequantizedHtj2k97Resolution,
    PrequantizedHtj2k97Subband, ProjectedBands, ProjectionBatchJob, ResidentBufferRef,
    ResidentColorModel, ResidentComponentGeometry, ResidentDctCoefficientOrder,
    ResidentDctGridLayout, ResidentDwtSubband, ResidentDwtSubbandKind, ResidentDwtSubbandLayout,
    ResidentHandoffError, ResidentJpegDctGrid, ResidentSampleInfo, ResidentSampling,
    DWT97_BLOCK_COEFFICIENTS, METAL_DCT97_UNSUPPORTED_GRID, METAL_DCT_KERNEL_FAILED,
    METAL_RESIDENT_HANDOFF_VALIDATION_FAILED,
};

#[derive(Clone, Copy)]
pub(super) struct ProjectionBatchShape {
    pub(super) batch_count: usize,
    pub(super) batch_count_u32: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) block_cols: u32,
    pub(super) blocks_per_item: u32,
    pub(super) low_width: usize,
    pub(super) low_height: usize,
    pub(super) high_width: usize,
    pub(super) high_height: usize,
    pub(super) ll_len: usize,
    pub(super) hl_len: usize,
    pub(super) lh_len: usize,
    pub(super) hh_len: usize,
}

pub(super) fn projection_batch_shape(
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

pub(super) struct ProjectionBatchWeightBuffers {
    pub(super) x_low_rows: Buffer,
    pub(super) x_low_taps: Buffer,
    pub(super) x_high_rows: Buffer,
    pub(super) x_high_taps: Buffer,
    pub(super) y_low_rows: Buffer,
    pub(super) y_low_taps: Buffer,
    pub(super) y_high_rows: Buffer,
    pub(super) y_high_taps: Buffer,
}

pub(super) fn projection_batch_weight_buffers(
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

pub(super) struct ProjectionBatchOutputBuffers {
    pub(super) ll: Buffer,
    pub(super) hl: Buffer,
    pub(super) lh: Buffer,
    pub(super) hh: Buffer,
}

#[derive(Clone, Copy)]
pub(super) struct ResidentDwtBand<'a> {
    pub(super) buffer: &'a Buffer,
    pub(super) kind: ResidentDwtSubbandKind,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) values_per_item: usize,
}

pub(super) struct Dwt97CodeBlockOutputBuffers {
    pub(super) ll: Buffer,
    pub(super) hl: Buffer,
    pub(super) lh: Buffer,
    pub(super) hh: Buffer,
}

pub(super) fn validate_resident_dct_handoffs_for_dwt97_jobs(
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

pub(super) fn validate_resident_dct_handoffs_for_htj2k_jobs(
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

pub(super) fn validate_resident_dct_handoffs(
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

pub(super) fn validate_resident_dwt_handoffs_for_dwt97_jobs(
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

pub(super) fn validate_resident_dwt_handoffs_for_htj2k_jobs(
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

pub(super) fn validate_resident_dwt_handoffs(
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

pub(super) fn resident_component_geometry(
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

pub(super) fn resident_buffer_ref(
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

pub(super) fn checked_byte_count(
    value_count: usize,
    bytes_per_value: usize,
) -> Result<usize, MetalTranscodeError> {
    value_count
        .checked_mul(bytes_per_value)
        .ok_or(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID,
        ))
}

pub(super) fn resident_result<T>(
    result: Result<T, ResidentHandoffError>,
) -> Result<T, MetalTranscodeError> {
    result.map_err(resident_handoff_error)
}

pub(super) fn resident_handoff_error(_error: ResidentHandoffError) -> MetalTranscodeError {
    MetalTranscodeError::Kernel(METAL_RESIDENT_HANDOFF_VALIDATION_FAILED)
}

pub(super) fn projection_batch_output_transfer_count(
    buffers: &ProjectionBatchOutputBuffers,
) -> usize {
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

pub(super) fn projection_batch_output_transfer_bytes(
    buffers: &ProjectionBatchOutputBuffers,
) -> u64 {
    [
        buffers.ll.length(),
        buffers.hl.length(),
        buffers.lh.length(),
        buffers.hh.length(),
    ]
    .into_iter()
    .fold(0_u64, u64::saturating_add)
}

pub(super) fn dwt97_codeblock_output_transfer_count(
    buffers: &Dwt97CodeBlockOutputBuffers,
) -> usize {
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

pub(super) fn dwt97_codeblock_output_transfer_bytes(buffers: &Dwt97CodeBlockOutputBuffers) -> u64 {
    [
        buffers.ll.length(),
        buffers.hl.length(),
        buffers.lh.length(),
        buffers.hh.length(),
    ]
    .into_iter()
    .fold(0_u64, u64::saturating_add)
}

pub(super) fn projection_batch_output_buffers(
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

pub(super) fn projection_batch_private_output_buffers(
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

pub(super) fn dwt97_codeblock_output_buffers(
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

pub(super) fn dispatch_projection_batch_bands(
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

pub(super) fn read_projected_batch_outputs(
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

pub(super) fn read_prequantized_97_codeblock_outputs(
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
                        codeblock_item_slice(&ll, idx, shape.ll_len, unsupported_grid)?,
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
                            codeblock_item_slice(&hl, idx, shape.hl_len, unsupported_grid)?,
                            shape.high_width,
                            shape.low_height,
                            J2kSubBandType::HighLow,
                            dwt97_total_bitplanes(options, J2kSubBandType::HighLow),
                            options,
                        )?,
                        prequantized_subband_from_codeblock_buffer(
                            codeblock_item_slice(&lh, idx, shape.lh_len, unsupported_grid)?,
                            shape.low_width,
                            shape.high_height,
                            J2kSubBandType::LowHigh,
                            dwt97_total_bitplanes(options, J2kSubBandType::LowHigh),
                            options,
                        )?,
                        prequantized_subband_from_codeblock_buffer(
                            codeblock_item_slice(&hh, idx, shape.hh_len, unsupported_grid)?,
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

pub(super) fn codeblock_item_slice<'a>(
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

pub(super) fn prequantized_subband_from_codeblock_buffer(
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
