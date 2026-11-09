// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_batch_len, dwt97_block_value_count, output_buffer, output_i32_buffer,
    private_f32_buffer, read_f32_buffer_at, size_of, try_transcode_vec_with_capacity, u32_param,
    BackendKind, Buffer, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob, DeviceMemoryRange,
    MetalRuntime, MetalTranscodeError, ProjectedBands, ResidentBufferRef, ResidentColorModel,
    ResidentComponentGeometry, ResidentDctCoefficientOrder, ResidentDctGridLayout,
    ResidentDwtSubband, ResidentDwtSubbandKind, ResidentDwtSubbandLayout, ResidentHandoffError,
    ResidentJpegDctGrid, ResidentSampleInfo, ResidentSampling, DWT97_BLOCK_COEFFICIENTS,
    METAL_DCT97_UNSUPPORTED_GRID,
};
use objc2::rc::Retained;
use objc2_metal::MTLBuffer as _;

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
        count = count.checked_add(1).ok_or(MetalTranscodeError::Kernel(
            "Metal resident DCT handoff count overflowed",
        ))?;
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
            count = count.checked_add(1).ok_or(MetalTranscodeError::Kernel(
                "Metal resident DWT handoff count overflowed",
            ))?;
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
    let allocation =
        u64::try_from(Retained::as_ptr(buffer).cast::<()>() as usize).map_err(|error| {
            MetalTranscodeError::runtime("Metal resident buffer address conversion", error)
        })?;
    let allocation_len = buffer.length();
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

pub(super) fn resident_handoff_error(error: ResidentHandoffError) -> MetalTranscodeError {
    MetalTranscodeError::runtime("Metal resident handoff validation", error)
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
    .map(|bytes| u64::try_from(bytes).unwrap_or(u64::MAX))
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
    .map(|bytes| u64::try_from(bytes).unwrap_or(u64::MAX))
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
        )?,
        hl: output_buffer(
            &runtime.device,
            checked_batch_len(shape.hl_len, shape.batch_count, unsupported_grid)?,
        )?,
        lh: output_buffer(
            &runtime.device,
            checked_batch_len(shape.lh_len, shape.batch_count, unsupported_grid)?,
        )?,
        hh: output_buffer(
            &runtime.device,
            checked_batch_len(shape.hh_len, shape.batch_count, unsupported_grid)?,
        )?,
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
        )?,
        hl: private_f32_buffer(
            &runtime.device,
            checked_batch_len(shape.hl_len, shape.batch_count, unsupported_grid)?,
        )?,
        lh: private_f32_buffer(
            &runtime.device,
            checked_batch_len(shape.lh_len, shape.batch_count, unsupported_grid)?,
        )?,
        hh: private_f32_buffer(
            &runtime.device,
            checked_batch_len(shape.hh_len, shape.batch_count, unsupported_grid)?,
        )?,
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
        )?,
        hl: output_i32_buffer(
            &runtime.device,
            checked_batch_len(shape.hl_len, shape.batch_count, unsupported_grid)?,
        )?,
        lh: output_i32_buffer(
            &runtime.device,
            checked_batch_len(shape.lh_len, shape.batch_count, unsupported_grid)?,
        )?,
        hh: output_i32_buffer(
            &runtime.device,
            checked_batch_len(shape.hh_len, shape.batch_count, unsupported_grid)?,
        )?,
    })
}

pub(super) fn read_projected_batch_outputs(
    buffers: &ProjectionBatchOutputBuffers,
    shape: ProjectionBatchShape,
    unsupported_grid: &'static str,
) -> Result<Vec<ProjectedBands>, MetalTranscodeError> {
    let mut outputs =
        try_transcode_vec_with_capacity(shape.batch_count, "projected wavelet batch metadata")?;
    for idx in 0..shape.batch_count {
        outputs.push(ProjectedBands {
            ll: read_projected_band(&buffers.ll, shape.ll_len, idx, unsupported_grid)?,
            hl: read_projected_band(&buffers.hl, shape.hl_len, idx, unsupported_grid)?,
            lh: read_projected_band(&buffers.lh, shape.lh_len, idx, unsupported_grid)?,
            hh: read_projected_band(&buffers.hh, shape.hh_len, idx, unsupported_grid)?,
            low_width: shape.low_width,
            low_height: shape.low_height,
            high_width: shape.high_width,
            high_height: shape.high_height,
        });
    }

    Ok(outputs)
}

fn read_projected_band(
    buffer: &Buffer,
    stride: usize,
    item_idx: usize,
    unsupported_grid: &'static str,
) -> Result<Vec<f64>, MetalTranscodeError> {
    read_f32_buffer_at(
        buffer,
        checked_batch_len(stride, item_idx, unsupported_grid)?,
        stride,
    )
}
