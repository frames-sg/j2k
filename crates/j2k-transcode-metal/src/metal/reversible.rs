// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_device_element_count, checked_device_workspace_bytes, checked_host_element_count,
    checked_host_workspace_bytes, dispatch_dct_grid_to_dwt53_with_runtime,
    idct_blocks_to_signed_samples_rayon, size_of, try_transcode_vec_with_capacity,
    validate_float_projection_allocations, validate_grid, validate_reversible_batch_geometry,
    DctGridToDwt53Job, DctGridToReversibleDwt53Job, Dwt53TwoDimensional, MetalTranscodeError,
    MetalTranscodeSession, ReversibleDwt53FirstLevel, SparseDwt53WeightRows, TranscodeStageError,
    METAL_DCT53_UNSUPPORTED_GRID, METAL_DCT_KERNEL_FAILED, METAL_READBACK_CHUNK_BYTES,
    METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
};

mod batch;

const IDCT_MATERIALIZATION_CHUNK_BLOCKS: usize = 1024;

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
    let block_count = checked_host_element_count::<[i32; 64]>(
        &[blocks_per_item, jobs.len()],
        "reversible IDCT block samples",
    )?;
    let output_count = checked_host_element_count::<i32>(
        &[first.width, first.height, jobs.len()],
        "reversible 5/3 output bands",
    )?;
    let metadata_count = checked_host_element_count::<ReversibleDwt53FirstLevel>(
        &[jobs.len()],
        "reversible 5/3 output metadata",
    )?;
    checked_device_element_count::<[i32; 64]>(&[block_count], "reversible IDCT block upload")?;
    checked_device_element_count::<i32>(
        &[first.width, first.height, jobs.len()],
        "reversible 5/3 device output",
    )?;
    let input_bytes = block_count.saturating_mul(size_of::<[i32; 64]>());
    let output_bytes = output_count.saturating_mul(size_of::<i32>());
    let metadata_bytes = metadata_count.saturating_mul(size_of::<ReversibleDwt53FirstLevel>());
    let chunk_bytes = IDCT_MATERIALIZATION_CHUNK_BLOCKS.saturating_mul(size_of::<[i32; 64]>());
    checked_host_workspace_bytes(
        &[
            input_bytes,
            output_bytes
                .saturating_add(metadata_bytes)
                .saturating_add(METAL_READBACK_CHUNK_BYTES)
                .max(chunk_bytes),
        ],
        "reversible 5/3 host workspace",
    )?;
    checked_device_workspace_bytes(
        &[input_bytes, output_bytes],
        "reversible 5/3 device workspace",
    )?;
    let mut block_samples =
        try_transcode_vec_with_capacity(block_count, "reversible IDCT block samples")?;
    for job in jobs {
        for chunk in job
            .dequantized_blocks
            .chunks(IDCT_MATERIALIZATION_CHUNK_BLOCKS)
        {
            block_samples.extend(
                idct_blocks_to_signed_samples_rayon(chunk).map_err(idct_materialization_error)?,
            );
        }
    }
    if block_samples.len() != block_count {
        return Err(MetalTranscodeError::Kernel(
            "reversible IDCT materialization count mismatch",
        ));
    }

    session.with_runtime(|runtime| {
        batch::dispatch_with_runtime(
            runtime,
            &block_samples,
            jobs.len(),
            first.block_cols,
            first.width,
            first.height,
        )
    })
}

fn idct_materialization_error(error: TranscodeStageError) -> MetalTranscodeError {
    match error {
        TranscodeStageError::MemoryCapExceeded { requested, cap } => {
            MetalTranscodeError::HostAllocationTooLarge {
                requested,
                cap,
                what: "reversible IDCT chunk",
            }
        }
        TranscodeStageError::HostAllocationFailed { bytes } => {
            MetalTranscodeError::HostAllocationFailed {
                requested: bytes,
                what: "reversible IDCT chunk",
            }
        }
        other => MetalTranscodeError::runtime("reversible IDCT materialization", other),
    }
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
    let weight_host_bytes = SparseDwt53WeightRows::allocation_bytes_for_len(job.width)?
        .saturating_add(SparseDwt53WeightRows::allocation_bytes_for_len(job.height)?);
    let weight_device_bytes = SparseDwt53WeightRows::metal_bytes_for_len(job.width)?
        .saturating_add(SparseDwt53WeightRows::metal_bytes_for_len(job.height)?);
    validate_float_projection_allocations(
        job.blocks.len(),
        job.width,
        job.height,
        1,
        weight_host_bytes,
        weight_device_bytes,
        1,
    )?;
    let x_weights = SparseDwt53WeightRows::for_len(job.width)?;
    let y_weights = SparseDwt53WeightRows::for_len(job.height)?;
    session.with_runtime(|runtime| {
        dispatch_dct_grid_to_dwt53_with_runtime(runtime, job, &x_weights, &y_weights)
    })
}
