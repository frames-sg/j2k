// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    device_bands_to_preencoded_components, htj2k97_quantize_params, htj2k97_subband_delta,
    transcode_kernels_built, try_transcode_vec_for_product, validate_htj2k97_codeblock_options,
    CudaContext, CudaDwt97BatchGeometry, CudaDwt97BatchStageTimings, CudaDwt97BatchWithPoolRequest,
    CudaHtj2k97CodeblockBatchWithPoolRequest, CudaHtj2k97QuantizeParams,
    CudaHtj2kEncodeStageTimings, CudaTranscodeError, CudaTranscodeSession, DctGridToDwt53Job,
    DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob, DctGridToReversibleDwt53Job,
    Dwt53TwoDimensional, Dwt97BatchStageTimings, Dwt97TwoDimensional, HostPhaseBudget,
    Htj2k97CodeBlockOptions, J2kSubBandType, PreencodedHtj2k97Component,
    PrequantizedHtj2k97Component, ReversibleDwt53FirstLevel, NOT_WIRED,
};
mod components;
use self::components::codeblock_bands_to_components;
mod staging;
pub(super) use self::staging::append_i16_blocks;
use self::staging::{
    account_dwt97_output, account_reversible_output, append_f64_blocks_to_f32,
    bands_to_first_level, dwt97_bands_to_f64_with_live_host_bytes, dwt97_batch_bands_to_f64,
    flatten_blocks, flatten_f64_blocks_to_f32, validate_block_grid,
    validate_staging_and_readback_workspace,
};

pub(super) fn run_reversible(
    context: &CudaContext,
    job: DctGridToReversibleDwt53Job<'_>,
    live_host_bytes: usize,
) -> Result<ReversibleDwt53FirstLevel, CudaTranscodeError> {
    let bands = context
        .j2k_transcode_reversible_dwt53_and_live_host_bytes(
            flatten_blocks(job.dequantized_blocks),
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
            live_host_bytes,
        )
        .map_err(|error| {
            CudaTranscodeError::runtime("CUDA reversible 5/3 transcode dispatch", error)
        })?;
    Ok(bands_to_first_level(bands))
}

pub(crate) fn dispatch_reversible_dwt53(
    session: &mut CudaTranscodeSession,
    job: DctGridToReversibleDwt53Job<'_>,
) -> Result<ReversibleDwt53FirstLevel, CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let context = session.context()?;
    run_reversible(&context, job, 0)
}

pub(crate) fn dispatch_reversible_dwt53_batch(
    session: &mut CudaTranscodeSession,
    jobs: &[DctGridToReversibleDwt53Job<'_>],
) -> Result<Vec<ReversibleDwt53FirstLevel>, CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let context = session.context()?;
    let mut budget = HostPhaseBudget::new("CUDA reversible 5/3 batch outputs");
    let mut outputs =
        budget.try_vec_with_capacity(jobs.len(), "CUDA reversible 5/3 batch outputs")?;
    for job in jobs {
        let output = run_reversible(&context, *job, budget.live_bytes())?;
        account_reversible_output(&mut budget, &output)?;
        outputs.push(output);
    }
    Ok(outputs)
}

pub(crate) fn dispatch_dwt53(
    _job: DctGridToDwt53Job<'_>,
) -> Result<Dwt53TwoDimensional<f64>, CudaTranscodeError> {
    Err(NOT_WIRED)
}

/// Map the runtime's local batch timings onto the transcode accelerator type.
pub(super) fn map_batch_timings(timings: CudaDwt97BatchStageTimings) -> Dwt97BatchStageTimings {
    Dwt97BatchStageTimings {
        pack_upload_us: timings.pack_upload_us,
        pack_upload_transfers: usize::from(timings.pack_upload_us > 0),
        pack_upload_bytes: 0,
        resident_dct_handoff_count: 0,
        idct_row_lift_us: timings.idct_row_lift_us,
        column_lift_us: timings.column_lift_us,
        resident_dwt_handoff_count: 0,
        quantize_codeblock_us: timings.quantize_codeblock_us,
        ht_encode_us: timings.ht_encode_us,
        ht_kernel_us: 0,
        ht_status_readback_us: 0,
        ht_status_readback_transfers: 0,
        ht_status_readback_bytes: 0,
        ht_compact_us: 0,
        ht_output_readback_us: 0,
        ht_output_readback_transfers: 0,
        ht_output_readback_bytes: 0,
        ht_codeblock_dispatches: timings.ht_codeblock_dispatches,
        readback_us: timings.readback_us,
        readback_transfers: usize::from(timings.readback_us > 0),
        readback_bytes: 0,
    }
}

pub(super) fn set_ht_encode_timings(
    timings: &mut Dwt97BatchStageTimings,
    ht_timings: CudaHtj2kEncodeStageTimings,
) {
    timings.ht_encode_us = ht_timings.ht_encode_us;
    timings.ht_kernel_us = ht_timings.ht_kernel_us;
    timings.ht_status_readback_us = ht_timings.ht_status_readback_us;
    timings.ht_status_readback_transfers = usize::from(ht_timings.ht_status_readback_us > 0);
    timings.ht_status_readback_bytes = 0;
    timings.ht_compact_us = ht_timings.ht_compact_us;
    timings.ht_output_readback_us = ht_timings.ht_output_readback_us;
    timings.ht_output_readback_transfers = usize::from(ht_timings.ht_output_readback_us > 0);
    timings.ht_output_readback_bytes = 0;
}

pub(super) fn add_ht_encode_timings(
    timings: &mut Dwt97BatchStageTimings,
    ht_timings: CudaHtj2kEncodeStageTimings,
) {
    timings.ht_encode_us = timings.ht_encode_us.saturating_add(ht_timings.ht_encode_us);
    timings.ht_kernel_us = timings.ht_kernel_us.saturating_add(ht_timings.ht_kernel_us);
    timings.ht_status_readback_us = timings
        .ht_status_readback_us
        .saturating_add(ht_timings.ht_status_readback_us);
    timings.ht_status_readback_transfers = timings
        .ht_status_readback_transfers
        .saturating_add(usize::from(ht_timings.ht_status_readback_us > 0));
    timings.ht_compact_us = timings
        .ht_compact_us
        .saturating_add(ht_timings.ht_compact_us);
    timings.ht_output_readback_us = timings
        .ht_output_readback_us
        .saturating_add(ht_timings.ht_output_readback_us);
    timings.ht_output_readback_transfers = timings
        .ht_output_readback_transfers
        .saturating_add(usize::from(ht_timings.ht_output_readback_us > 0));
}

pub(super) fn accumulate_batch_timings(
    total: &mut Dwt97BatchStageTimings,
    next: Dwt97BatchStageTimings,
) {
    total.pack_upload_us = total.pack_upload_us.saturating_add(next.pack_upload_us);
    total.idct_row_lift_us = total.idct_row_lift_us.saturating_add(next.idct_row_lift_us);
    total.column_lift_us = total.column_lift_us.saturating_add(next.column_lift_us);
    total.resident_dct_handoff_count = total
        .resident_dct_handoff_count
        .saturating_add(next.resident_dct_handoff_count);
    total.resident_dwt_handoff_count = total
        .resident_dwt_handoff_count
        .saturating_add(next.resident_dwt_handoff_count);
    total.quantize_codeblock_us = total
        .quantize_codeblock_us
        .saturating_add(next.quantize_codeblock_us);
    total.ht_encode_us = total.ht_encode_us.saturating_add(next.ht_encode_us);
    total.ht_kernel_us = total.ht_kernel_us.saturating_add(next.ht_kernel_us);
    total.ht_status_readback_us = total
        .ht_status_readback_us
        .saturating_add(next.ht_status_readback_us);
    total.ht_status_readback_transfers = total
        .ht_status_readback_transfers
        .saturating_add(next.ht_status_readback_transfers);
    total.ht_status_readback_bytes = total
        .ht_status_readback_bytes
        .saturating_add(next.ht_status_readback_bytes);
    total.ht_compact_us = total.ht_compact_us.saturating_add(next.ht_compact_us);
    total.ht_output_readback_us = total
        .ht_output_readback_us
        .saturating_add(next.ht_output_readback_us);
    total.ht_output_readback_transfers = total
        .ht_output_readback_transfers
        .saturating_add(next.ht_output_readback_transfers);
    total.ht_output_readback_bytes = total
        .ht_output_readback_bytes
        .saturating_add(next.ht_output_readback_bytes);
    total.ht_codeblock_dispatches = total
        .ht_codeblock_dispatches
        .saturating_add(next.ht_codeblock_dispatches);
    total.readback_us = total.readback_us.saturating_add(next.readback_us);
    total.pack_upload_transfers = total
        .pack_upload_transfers
        .saturating_add(next.pack_upload_transfers);
    total.pack_upload_bytes = total
        .pack_upload_bytes
        .saturating_add(next.pack_upload_bytes);
    total.readback_transfers = total
        .readback_transfers
        .saturating_add(next.readback_transfers);
    total.readback_bytes = total.readback_bytes.saturating_add(next.readback_bytes);
}

pub(super) fn run_dwt97(
    context: &CudaContext,
    job: DctGridToDwt97Job<'_>,
    live_host_bytes: usize,
) -> Result<Dwt97TwoDimensional<f64>, CudaTranscodeError> {
    validate_block_grid(
        job.blocks.len(),
        job.block_cols,
        job.block_rows,
        "CUDA 9/7 DCT block slice does not match its grid",
    )?;
    validate_staging_and_readback_workspace(
        1,
        job.block_cols,
        job.block_rows,
        job.width,
        job.height,
        "CUDA 9/7 single-dispatch host workspace",
    )?;
    let mut staging_budget =
        HostPhaseBudget::with_live_bytes("CUDA 9/7 single staging", live_host_bytes)?;
    let coeffs = flatten_f64_blocks_to_f32(job.blocks, &mut staging_budget)?;
    let bands = context
        .j2k_transcode_dwt97_and_live_host_bytes(
            &coeffs,
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
            staging_budget.live_bytes(),
        )
        .map_err(|error| CudaTranscodeError::runtime("CUDA 9/7 transcode dispatch", error))?;
    drop(coeffs);
    dwt97_bands_to_f64_with_live_host_bytes(bands, live_host_bytes)
}

pub(crate) fn dispatch_dwt97(
    session: &mut CudaTranscodeSession,
    job: DctGridToDwt97Job<'_>,
) -> Result<Dwt97TwoDimensional<f64>, CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let context = session.context()?;
    run_dwt97(&context, job, 0)
}

pub(crate) fn dispatch_dwt97_batch(
    session: &mut CudaTranscodeSession,
    jobs: &[DctGridToDwt97Job<'_>],
) -> Result<(Vec<Dwt97TwoDimensional<f64>>, Dwt97BatchStageTimings), CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let context = session.context()?;

    let Some(first) = jobs.first() else {
        return Ok((Vec::new(), Dwt97BatchStageTimings::default()));
    };

    // Non-uniform geometry falls back to the per-job path (still correct, but no
    // staged batch timings), matching Metal's same-geometry batch gating.
    let uniform = jobs.iter().all(|job| {
        job.block_cols == first.block_cols
            && job.block_rows == first.block_rows
            && job.width == first.width
            && job.height == first.height
    });
    if !uniform {
        let mut budget = HostPhaseBudget::new("nonuniform CUDA 9/7 batch outputs");
        let mut outputs =
            budget.try_vec_with_capacity(jobs.len(), "nonuniform CUDA 9/7 batch outputs")?;
        for job in jobs {
            let output = run_dwt97(&context, *job, budget.live_bytes())?;
            account_dwt97_output(&mut budget, &output)?;
            outputs.push(output);
        }
        return Ok((outputs, Dwt97BatchStageTimings::default()));
    }

    for job in jobs {
        validate_block_grid(
            job.blocks.len(),
            job.block_cols,
            job.block_rows,
            "CUDA 9/7 batch DCT block slice does not match its grid",
        )?;
    }
    validate_staging_and_readback_workspace(
        jobs.len(),
        first.block_cols,
        first.block_rows,
        first.width,
        first.height,
        "CUDA 9/7 batch-dispatch host workspace",
    )?;
    let mut staging_budget = HostPhaseBudget::new("CUDA 9/7 batch staging");
    let mut blocks = staging_budget.try_vec_for_product::<f32>(
        &[jobs.len(), first.block_cols, first.block_rows, 64],
        "CUDA 9/7 batch f32 DCT staging",
    )?;
    for job in jobs {
        append_f64_blocks_to_f32(job.blocks, &mut blocks);
    }
    let pool = session.buffer_pool(&context);
    let (bands, timings) = context
        .j2k_transcode_dwt97_batch_with_pool_and_live_host_bytes(
            CudaDwt97BatchWithPoolRequest {
                blocks: &blocks,
                geometry: CudaDwt97BatchGeometry {
                    item_count: jobs.len(),
                    block_cols: first.block_cols,
                    block_rows: first.block_rows,
                    width: first.width,
                    height: first.height,
                },
                pool: &pool,
            },
            staging_budget.live_bytes(),
        )
        .map_err(|error| CudaTranscodeError::runtime("CUDA 9/7 batch transcode dispatch", error))?;
    drop(blocks);
    let outputs = dwt97_batch_bands_to_f64(bands)?;
    Ok((outputs, map_batch_timings(timings)))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the CUDA quantization ABI intentionally consumes f32 inverse deltas"
)]
pub(crate) fn dispatch_htj2k97_codeblock_batch(
    session: &mut CudaTranscodeSession,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<PrequantizedHtj2k97Component>, Dwt97BatchStageTimings), CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let (cb_width, cb_height) = validate_htj2k97_codeblock_options(options)?;
    let context = session.context()?;

    let Some(first) = jobs.first() else {
        return Ok((Vec::new(), Dwt97BatchStageTimings::default()));
    };

    // The fused staged kernels require uniform block geometry across the batch.
    let uniform = jobs.iter().all(|job| {
        job.block_cols == first.block_cols
            && job.block_rows == first.block_rows
            && job.width == first.width
            && job.height == first.height
    });
    if !uniform {
        return Err(CudaTranscodeError::UnsupportedJob(
            "CUDA 9/7 code-block batch requires uniform job geometry",
        ));
    }

    // Per-subband inverse step sizes from the shared oracle (same numbers the
    // CPU oracle and Metal use), plus code-block geometry from the options.
    let inv_delta =
        |sub: J2kSubBandType| -> f32 { (1.0 / htj2k97_subband_delta(options, sub)) as f32 };
    let params = CudaHtj2k97QuantizeParams {
        inv_delta_ll: inv_delta(J2kSubBandType::LowLow),
        inv_delta_hl: inv_delta(J2kSubBandType::HighLow),
        inv_delta_lh: inv_delta(J2kSubBandType::LowHigh),
        inv_delta_hh: inv_delta(J2kSubBandType::HighHigh),
        cb_width,
        cb_height,
    };

    for job in jobs {
        validate_block_grid(
            job.blocks.len(),
            job.block_cols,
            job.block_rows,
            "CUDA 9/7 code-block DCT slice does not match its grid",
        )?;
    }
    validate_staging_and_readback_workspace(
        jobs.len(),
        first.block_cols,
        first.block_rows,
        first.width,
        first.height,
        "CUDA 9/7 code-block host workspace",
    )?;
    let mut staging_budget = HostPhaseBudget::new("CUDA 9/7 code-block batch staging");
    let mut blocks = staging_budget.try_vec_for_product::<f32>(
        &[jobs.len(), first.block_cols, first.block_rows, 64],
        "CUDA 9/7 code-block batch f32 DCT staging",
    )?;
    for job in jobs {
        append_f64_blocks_to_f32(job.blocks, &mut blocks);
    }
    let pool = session.buffer_pool(&context);
    let (codeblock_bands, timings) = context
        .j2k_transcode_htj2k97_codeblock_batch_with_pool_and_live_host_bytes(
            CudaHtj2k97CodeblockBatchWithPoolRequest {
                blocks: &blocks,
                geometry: CudaDwt97BatchGeometry {
                    item_count: jobs.len(),
                    block_cols: first.block_cols,
                    block_rows: first.block_rows,
                    width: first.width,
                    height: first.height,
                },
                params,
                pool: &pool,
            },
            staging_budget.live_bytes(),
        )
        .map_err(|error| {
            CudaTranscodeError::runtime("CUDA 9/7 code-block batch dispatch", error)
        })?;
    drop(blocks);

    let components = codeblock_bands_to_components(&codeblock_bands, jobs, options)?;
    Ok((components, map_batch_timings(timings)))
}

pub(crate) fn dispatch_htj2k97_preencoded_batch(
    session: &mut CudaTranscodeSession,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<PreencodedHtj2k97Component>, Dwt97BatchStageTimings), CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    validate_htj2k97_codeblock_options(options)?;
    let context = session.context()?;

    let Some(first) = jobs.first() else {
        return Ok((Vec::new(), Dwt97BatchStageTimings::default()));
    };

    let uniform = jobs.iter().all(|job| {
        job.block_cols == first.block_cols
            && job.block_rows == first.block_rows
            && job.width == first.width
            && job.height == first.height
    });
    if !uniform {
        return Err(CudaTranscodeError::UnsupportedJob(
            "CUDA 9/7 resident HT batch requires uniform job geometry",
        ));
    }

    let params = htj2k97_quantize_params(options)?;
    for job in jobs {
        validate_block_grid(
            job.blocks.len(),
            job.block_cols,
            job.block_rows,
            "CUDA 9/7 resident DCT slice does not match its grid",
        )?;
    }
    let mut blocks = try_transcode_vec_for_product::<f32>(
        &[jobs.len(), first.block_cols, first.block_rows, 64],
        "CUDA 9/7 resident batch f32 DCT staging",
    )?;
    for job in jobs {
        append_f64_blocks_to_f32(job.blocks, &mut blocks);
    }
    let pool = session.buffer_pool(&context);
    let (device_bands, cuda_timings) = context
        .j2k_transcode_htj2k97_codeblock_batch_resident_with_pool(
            CudaHtj2k97CodeblockBatchWithPoolRequest {
                blocks: &blocks,
                geometry: CudaDwt97BatchGeometry {
                    item_count: jobs.len(),
                    block_cols: first.block_cols,
                    block_rows: first.block_rows,
                    width: first.width,
                    height: first.height,
                },
                params,
                pool: &pool,
            },
        )
        .map_err(|error| CudaTranscodeError::runtime("CUDA 9/7 resident batch dispatch", error))?;
    drop(blocks);
    let mut timings = map_batch_timings(cuda_timings);

    let resources = session.encode_resources(&context)?;
    let (components, ht_timings, ht_dispatches) = device_bands_to_preencoded_components(
        &context,
        resources.as_ref(),
        &pool,
        &device_bands,
        jobs,
        options,
    )?;
    set_ht_encode_timings(&mut timings, ht_timings);
    timings.ht_codeblock_dispatches = ht_dispatches;
    Ok((components, timings))
}
