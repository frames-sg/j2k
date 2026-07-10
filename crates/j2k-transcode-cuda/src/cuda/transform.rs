// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    device_bands_to_preencoded_components, htj2k97_code_block_dim, htj2k97_quantize_params,
    htj2k97_subband_delta, htj2k97_subband_total_bitplanes, transcode_kernels_built,
    validate_band_len, validate_htj2k97_codeblock_options, CudaContext, CudaDwt97BatchGeometry,
    CudaDwt97BatchStageTimings, CudaDwt97BatchWithPoolRequest, CudaHtj2k97CodeblockBands,
    CudaHtj2k97CodeblockBatchWithPoolRequest, CudaHtj2k97QuantizeParams,
    CudaHtj2kEncodeStageTimings, CudaTranscodeDwt97Bands, CudaTranscodeError,
    CudaTranscodeReversible53Bands, CudaTranscodeSession, DctGridToDwt53Job, DctGridToDwt97Job,
    DctGridToHtj2k97CodeBlockJob, DctGridToReversibleDwt53Job, Dwt53TwoDimensional,
    Dwt97BatchStageTimings, Dwt97TwoDimensional, Htj2k97CodeBlockOptions, J2kSubBandType,
    PreencodedHtj2k97Component, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband, ReversibleDwt53FirstLevel,
    NOT_WIRED,
};

/// Flatten `&[[i16; 64]]` into the contiguous `&[i16]` the runtime job expects.
pub(super) fn flatten_blocks(blocks: &[[i16; 64]]) -> &[i16] {
    blocks.as_flattened()
}

pub(super) fn bands_to_first_level(
    bands: CudaTranscodeReversible53Bands,
) -> ReversibleDwt53FirstLevel {
    ReversibleDwt53FirstLevel {
        ll: bands.ll,
        hl: bands.hl,
        lh: bands.lh,
        hh: bands.hh,
        low_width: bands.low_width,
        low_height: bands.low_height,
        high_width: bands.high_width,
        high_height: bands.high_height,
    }
}

pub(super) fn run_reversible(
    context: &CudaContext,
    job: DctGridToReversibleDwt53Job<'_>,
) -> Result<ReversibleDwt53FirstLevel, CudaTranscodeError> {
    let bands = context
        .j2k_transcode_reversible_dwt53(
            flatten_blocks(job.dequantized_blocks),
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
        )
        .map_err(|_| CudaTranscodeError::Kernel("CUDA reversible 5/3 transcode dispatch failed"))?;
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
    run_reversible(&context, job)
}

pub(crate) fn dispatch_reversible_dwt53_batch(
    session: &mut CudaTranscodeSession,
    jobs: &[DctGridToReversibleDwt53Job<'_>],
) -> Result<Vec<ReversibleDwt53FirstLevel>, CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let context = session.context()?;
    let mut outputs = Vec::with_capacity(jobs.len());
    for job in jobs {
        outputs.push(run_reversible(&context, *job)?);
    }
    Ok(outputs)
}

pub(crate) fn dispatch_dwt53(
    _job: DctGridToDwt53Job<'_>,
) -> Result<Dwt53TwoDimensional<f64>, CudaTranscodeError> {
    Err(NOT_WIRED)
}

/// Append the job's `[[f64; 8]; 8]` natural-order DCT blocks to a contiguous
/// `f32` coefficient buffer (row-major within block) the runtime kernels consume.
pub(super) fn append_f64_blocks_to_f32(blocks: &[[[f64; 8]; 8]], out: &mut Vec<f32>) {
    for block in blocks {
        for row in block {
            for &coeff in row {
                out.push(coeff as f32);
            }
        }
    }
}

/// Append natural-order dequantized i16 DCT blocks directly to the contiguous
/// i16 coefficient buffer the runtime kernels consume.
pub(super) fn append_i16_blocks(blocks: &[[i16; 64]], out: &mut Vec<i16>) {
    out.extend_from_slice(flatten_blocks(blocks));
}

/// Flatten one job's DCT blocks into a fresh contiguous `f32` buffer.
pub(super) fn flatten_f64_blocks_to_f32(blocks: &[[[f64; 8]; 8]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blocks.len() * 64);
    append_f64_blocks_to_f32(blocks, &mut out);
    out
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

pub(super) fn dwt97_bands_to_f64(bands: CudaTranscodeDwt97Bands) -> Dwt97TwoDimensional<f64> {
    let widen = |band: Vec<f32>| -> Vec<f64> { band.into_iter().map(f64::from).collect() };
    Dwt97TwoDimensional {
        ll: widen(bands.ll),
        hl: widen(bands.hl),
        lh: widen(bands.lh),
        hh: widen(bands.hh),
        low_width: bands.low_width,
        low_height: bands.low_height,
        high_width: bands.high_width,
        high_height: bands.high_height,
    }
}

pub(super) fn run_dwt97(
    context: &CudaContext,
    job: DctGridToDwt97Job<'_>,
) -> Result<Dwt97TwoDimensional<f64>, CudaTranscodeError> {
    let coeffs = flatten_f64_blocks_to_f32(job.blocks);
    let bands = context
        .j2k_transcode_dwt97(
            &coeffs,
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
        )
        .map_err(|_| CudaTranscodeError::Kernel("CUDA 9/7 transcode dispatch failed"))?;
    Ok(dwt97_bands_to_f64(bands))
}

pub(crate) fn dispatch_dwt97(
    session: &mut CudaTranscodeSession,
    job: DctGridToDwt97Job<'_>,
) -> Result<Dwt97TwoDimensional<f64>, CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let context = session.context()?;
    run_dwt97(&context, job)
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
        let mut outputs = Vec::with_capacity(jobs.len());
        for job in jobs {
            outputs.push(run_dwt97(&context, *job)?);
        }
        return Ok((outputs, Dwt97BatchStageTimings::default()));
    }

    let mut blocks = Vec::with_capacity(jobs.len() * first.block_cols * first.block_rows * 64);
    for job in jobs {
        append_f64_blocks_to_f32(job.blocks, &mut blocks);
    }
    let pool = session.buffer_pool(&context);
    let (bands, timings) = context
        .j2k_transcode_dwt97_batch_with_pool(CudaDwt97BatchWithPoolRequest {
            blocks: &blocks,
            geometry: CudaDwt97BatchGeometry {
                item_count: jobs.len(),
                block_cols: first.block_cols,
                block_rows: first.block_rows,
                width: first.width,
                height: first.height,
            },
            pool: &pool,
        })
        .map_err(|_| CudaTranscodeError::Kernel("CUDA 9/7 batch transcode dispatch failed"))?;
    let outputs = bands.into_iter().map(dwt97_bands_to_f64).collect();
    Ok((outputs, map_batch_timings(timings)))
}

/// Reslice one subband's code-block-major `i32` buffer (one item) into a
/// prequantized HTJ2K subband, mirroring the shared code-block oracle layout
/// (outer code-block row, inner code-block column, each block row-major).
pub(super) fn subband_from_codeblock_slice(
    data: &[i32],
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    options: Htj2k97CodeBlockOptions,
) -> Result<PrequantizedHtj2k97Subband, CudaTranscodeError> {
    let cb_width = htj2k97_code_block_dim(options.code_block_width_exp)?;
    let cb_height = htj2k97_code_block_dim(options.code_block_height_exp)?;
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let mut code_blocks = Vec::with_capacity(num_cbs_x * num_cbs_y);
    let mut offset = 0usize;
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let block_width = (width - cbx * cb_width).min(cb_width);
            let block_height = (height - cby * cb_height).min(cb_height);
            let len = block_width * block_height;
            let end = offset.checked_add(len).ok_or(CudaTranscodeError::Kernel(
                "CUDA 9/7 code-block band length overflow",
            ))?;
            if end > data.len() {
                return Err(CudaTranscodeError::Kernel(
                    "CUDA 9/7 code-block band output is shorter than expected",
                ));
            }
            code_blocks.push(PrequantizedHtj2k97CodeBlock {
                coefficients: data[offset..end].to_vec(),
                width: block_width as u32,
                height: block_height as u32,
            });
            offset = end;
        }
    }
    if offset != data.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA 9/7 code-block band output has trailing data",
        ));
    }
    Ok(PrequantizedHtj2k97Subband {
        sub_band_type,
        num_cbs_x: num_cbs_x as u32,
        num_cbs_y: num_cbs_y as u32,
        total_bitplanes: htj2k97_subband_total_bitplanes(options, sub_band_type),
        code_blocks,
    })
}

/// Reslice the per-item code-block bands into prequantized HTJ2K components,
/// one per job (resolution nesting `[[LL], [HL, LH, HH]]`).
#[allow(clippy::similar_names)]
pub(super) fn codeblock_bands_to_components(
    bands: &CudaHtj2k97CodeblockBands,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<Vec<PrequantizedHtj2k97Component>, CudaTranscodeError> {
    if bands.item_count != jobs.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA 9/7 code-block band item count mismatch",
        ));
    }
    let ll_size = bands.low_width * bands.low_height;
    let hl_size = bands.high_width * bands.low_height;
    let lh_size = bands.low_width * bands.high_height;
    let hh_size = bands.high_width * bands.high_height;
    validate_band_len(&bands.ll, bands.item_count, ll_size)?;
    validate_band_len(&bands.hl, bands.item_count, hl_size)?;
    validate_band_len(&bands.lh, bands.item_count, lh_size)?;
    validate_band_len(&bands.hh, bands.item_count, hh_size)?;
    jobs.iter()
        .enumerate()
        .map(|(item, job)| {
            let ll = &bands.ll[item * ll_size..(item + 1) * ll_size];
            let hl = &bands.hl[item * hl_size..(item + 1) * hl_size];
            let lh = &bands.lh[item * lh_size..(item + 1) * lh_size];
            let hh = &bands.hh[item * hh_size..(item + 1) * hh_size];
            Ok(PrequantizedHtj2k97Component {
                x_rsiz: job.x_rsiz,
                y_rsiz: job.y_rsiz,
                resolutions: vec![
                    PrequantizedHtj2k97Resolution {
                        subbands: vec![subband_from_codeblock_slice(
                            ll,
                            bands.low_width,
                            bands.low_height,
                            J2kSubBandType::LowLow,
                            options,
                        )?],
                    },
                    PrequantizedHtj2k97Resolution {
                        subbands: vec![
                            subband_from_codeblock_slice(
                                hl,
                                bands.high_width,
                                bands.low_height,
                                J2kSubBandType::HighLow,
                                options,
                            )?,
                            subband_from_codeblock_slice(
                                lh,
                                bands.low_width,
                                bands.high_height,
                                J2kSubBandType::LowHigh,
                                options,
                            )?,
                            subband_from_codeblock_slice(
                                hh,
                                bands.high_width,
                                bands.high_height,
                                J2kSubBandType::HighHigh,
                                options,
                            )?,
                        ],
                    },
                ],
            })
        })
        .collect()
}

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

    let mut blocks = Vec::with_capacity(jobs.len() * first.block_cols * first.block_rows * 64);
    for job in jobs {
        append_f64_blocks_to_f32(job.blocks, &mut blocks);
    }
    let pool = session.buffer_pool(&context);
    let (codeblock_bands, timings) = context
        .j2k_transcode_htj2k97_codeblock_batch_with_pool(CudaHtj2k97CodeblockBatchWithPoolRequest {
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
        })
        .map_err(|_| CudaTranscodeError::Kernel("CUDA 9/7 code-block batch dispatch failed"))?;

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
    let mut blocks = Vec::with_capacity(jobs.len() * first.block_cols * first.block_rows * 64);
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
        .map_err(|_| CudaTranscodeError::Kernel("CUDA 9/7 resident batch dispatch failed"))?;
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
