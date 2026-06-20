// SPDX-License-Identifier: Apache-2.0

//! CUDA dispatch boundary for the transcode accelerator.
//!
//! Each function uploads a DCT-grid job to the device, runs the ported kernel
//! in `j2k-cuda-runtime`, and returns wavelet bands / prequantized
//! components matching the `j2k-transcode` scalar oracle. Kernels are
//! wired incrementally; until a path is wired its dispatch returns a typed
//! [`CudaTranscodeError::UnsupportedJob`], which Auto mode treats as a scalar
//! fallback and Explicit mode surfaces as an error.

use j2k_transcode::accelerator::{
    DctGridI16ToHtj2k97CodeBlockBatch, DctGridI16ToHtj2k97CodeBlockJob, DctGridToDwt53Job,
    DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob, DctGridToReversibleDwt53Job,
    Dwt97BatchStageTimings, EncodedHtJ2kCodeBlock, Htj2k97CodeBlockOptions, J2kSubBandType,
    PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactBatch, PreencodedHtj2k97CompactBatchGroups,
    PreencodedHtj2k97CompactCodeBlock, PreencodedHtj2k97CompactComponent,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Component, PreencodedHtj2k97Resolution, PreencodedHtj2k97Subband,
    PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component, PrequantizedHtj2k97Resolution,
    PrequantizedHtj2k97Subband, ReversibleDwt53FirstLevel,
};
use j2k_transcode::dct53_2d::Dwt53TwoDimensional;
use j2k_transcode::dct97_2d::Dwt97TwoDimensional;
use j2k_transcode::htj2k97_codeblock_oracle::{
    htj2k97_subband_delta, htj2k97_subband_total_bitplanes,
};

use std::sync::Arc;

use j2k_cuda_runtime::{
    transcode_kernels_built, CudaBufferPool, CudaContext, CudaDwt97BatchStageTimings,
    CudaHtj2k97CodeblockBands, CudaHtj2k97DeviceCodeblockBands, CudaHtj2k97QuantizeParams,
    CudaHtj2kCompactEncodedCodeBlock, CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeResidentTarget,
    CudaHtj2kEncodeResources, CudaHtj2kEncodeStageTimings, CudaHtj2kEncodeTables,
    CudaHtj2kEncodedCodeBlock, CudaPooledDeviceBuffer, CudaTranscodeDwt97Bands,
    CudaTranscodeReversible53Bands,
};

use crate::CudaTranscodeError;

/// Returned until a given kernel path is wired to `j2k-cuda-runtime`.
const NOT_WIRED: CudaTranscodeError =
    CudaTranscodeError::UnsupportedJob("j2k-transcode-cuda kernel not yet wired");

type GroupedPreencodedComponents = Vec<(usize, Vec<PreencodedHtj2k97Component>)>;
type GroupedCompactPreencodedComponents = Vec<(usize, Vec<PreencodedHtj2k97CompactComponent>)>;
type ResidentPreencodedGroups = (
    GroupedPreencodedComponents,
    CudaHtj2kEncodeStageTimings,
    usize,
);
type ResidentCompactPreencodedGroups = (
    Vec<u8>,
    GroupedCompactPreencodedComponents,
    CudaHtj2kEncodeStageTimings,
    usize,
);

/// Caller-owned CUDA runtime state reused across transcode dispatches.
#[derive(Clone, Debug, Default)]
pub(crate) struct CudaTranscodeSession {
    context: Option<CudaContext>,
    buffer_pool: Option<CudaBufferPool>,
    encode_resources: Option<Arc<CudaHtj2kEncodeResources>>,
}

impl CudaTranscodeSession {
    fn context(&mut self) -> Result<CudaContext, CudaTranscodeError> {
        if self.context.is_none() {
            self.context = Some(
                CudaContext::system_default().map_err(|_| CudaTranscodeError::CudaUnavailable)?,
            );
        }
        self.context
            .clone()
            .ok_or(CudaTranscodeError::CudaUnavailable)
    }

    fn buffer_pool(&mut self, context: &CudaContext) -> CudaBufferPool {
        if let Some(pool) = &self.buffer_pool {
            return pool.clone();
        }
        let pool = context.buffer_pool();
        self.buffer_pool = Some(pool.clone());
        pool
    }

    fn encode_resources(
        &mut self,
        context: &CudaContext,
    ) -> Result<Arc<CudaHtj2kEncodeResources>, CudaTranscodeError> {
        if let Some(resources) = &self.encode_resources {
            return Ok(Arc::clone(resources));
        }
        let resources = Arc::new(
            context
                .upload_htj2k_encode_resources(cuda_htj2k_encode_tables())
                .map_err(|_| {
                    CudaTranscodeError::Kernel("CUDA HTJ2K encode resource upload failed")
                })?,
        );
        self.encode_resources = Some(Arc::clone(&resources));
        Ok(resources)
    }
}

/// Flatten `&[[i16; 64]]` into the contiguous `&[i16]` the runtime job expects.
fn flatten_blocks(blocks: &[[i16; 64]]) -> &[i16] {
    blocks.as_flattened()
}

fn bands_to_first_level(bands: CudaTranscodeReversible53Bands) -> ReversibleDwt53FirstLevel {
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

fn run_reversible(
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
fn append_f64_blocks_to_f32(blocks: &[[[f64; 8]; 8]], out: &mut Vec<f32>) {
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
fn append_i16_blocks(blocks: &[[i16; 64]], out: &mut Vec<i16>) {
    out.extend_from_slice(flatten_blocks(blocks));
}

/// Flatten one job's DCT blocks into a fresh contiguous `f32` buffer.
fn flatten_f64_blocks_to_f32(blocks: &[[[f64; 8]; 8]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blocks.len() * 64);
    append_f64_blocks_to_f32(blocks, &mut out);
    out
}

/// Map the runtime's local batch timings onto the transcode accelerator type.
fn map_batch_timings(timings: CudaDwt97BatchStageTimings) -> Dwt97BatchStageTimings {
    Dwt97BatchStageTimings {
        pack_upload_us: timings.pack_upload_us,
        idct_row_lift_us: timings.idct_row_lift_us,
        column_lift_us: timings.column_lift_us,
        quantize_codeblock_us: timings.quantize_codeblock_us,
        ht_encode_us: timings.ht_encode_us,
        ht_kernel_us: 0,
        ht_status_readback_us: 0,
        ht_compact_us: 0,
        ht_output_readback_us: 0,
        ht_codeblock_dispatches: timings.ht_codeblock_dispatches,
        readback_us: timings.readback_us,
    }
}

fn set_ht_encode_timings(
    timings: &mut Dwt97BatchStageTimings,
    ht_timings: CudaHtj2kEncodeStageTimings,
) {
    timings.ht_encode_us = ht_timings.ht_encode_us;
    timings.ht_kernel_us = ht_timings.ht_kernel_us;
    timings.ht_status_readback_us = ht_timings.ht_status_readback_us;
    timings.ht_compact_us = ht_timings.ht_compact_us;
    timings.ht_output_readback_us = ht_timings.ht_output_readback_us;
}

fn add_ht_encode_timings(
    timings: &mut Dwt97BatchStageTimings,
    ht_timings: CudaHtj2kEncodeStageTimings,
) {
    timings.ht_encode_us = timings.ht_encode_us.saturating_add(ht_timings.ht_encode_us);
    timings.ht_kernel_us = timings.ht_kernel_us.saturating_add(ht_timings.ht_kernel_us);
    timings.ht_status_readback_us = timings
        .ht_status_readback_us
        .saturating_add(ht_timings.ht_status_readback_us);
    timings.ht_compact_us = timings
        .ht_compact_us
        .saturating_add(ht_timings.ht_compact_us);
    timings.ht_output_readback_us = timings
        .ht_output_readback_us
        .saturating_add(ht_timings.ht_output_readback_us);
}

fn accumulate_batch_timings(total: &mut Dwt97BatchStageTimings, next: Dwt97BatchStageTimings) {
    total.pack_upload_us = total.pack_upload_us.saturating_add(next.pack_upload_us);
    total.idct_row_lift_us = total.idct_row_lift_us.saturating_add(next.idct_row_lift_us);
    total.column_lift_us = total.column_lift_us.saturating_add(next.column_lift_us);
    total.quantize_codeblock_us = total
        .quantize_codeblock_us
        .saturating_add(next.quantize_codeblock_us);
    total.ht_encode_us = total.ht_encode_us.saturating_add(next.ht_encode_us);
    total.ht_kernel_us = total.ht_kernel_us.saturating_add(next.ht_kernel_us);
    total.ht_status_readback_us = total
        .ht_status_readback_us
        .saturating_add(next.ht_status_readback_us);
    total.ht_compact_us = total.ht_compact_us.saturating_add(next.ht_compact_us);
    total.ht_output_readback_us = total
        .ht_output_readback_us
        .saturating_add(next.ht_output_readback_us);
    total.ht_codeblock_dispatches = total
        .ht_codeblock_dispatches
        .saturating_add(next.ht_codeblock_dispatches);
    total.readback_us = total.readback_us.saturating_add(next.readback_us);
}

fn dwt97_bands_to_f64(bands: CudaTranscodeDwt97Bands) -> Dwt97TwoDimensional<f64> {
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

fn run_dwt97(
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
    let (bands, timings) = context
        .j2k_transcode_dwt97_batch_with_pool(
            &blocks,
            jobs.len(),
            first.block_cols,
            first.block_rows,
            first.width,
            first.height,
            &session.buffer_pool(&context),
        )
        .map_err(|_| CudaTranscodeError::Kernel("CUDA 9/7 batch transcode dispatch failed"))?;
    let outputs = bands.into_iter().map(dwt97_bands_to_f64).collect();
    Ok((outputs, map_batch_timings(timings)))
}

/// Reslice one subband's code-block-major `i32` buffer (one item) into a
/// prequantized HTJ2K subband, mirroring the shared code-block oracle layout
/// (outer code-block row, inner code-block column, each block row-major).
fn subband_from_codeblock_slice(
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
fn codeblock_bands_to_components(
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
    let (codeblock_bands, timings) = context
        .j2k_transcode_htj2k97_codeblock_batch_with_pool(
            &blocks,
            jobs.len(),
            first.block_cols,
            first.block_rows,
            first.width,
            first.height,
            params,
            &session.buffer_pool(&context),
        )
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
            &blocks,
            jobs.len(),
            first.block_cols,
            first.block_rows,
            first.width,
            first.height,
            params,
            &pool,
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

fn dispatch_htj2k97_preencoded_i16_batch_with_sink<'a, 'j, R>(
    session: &mut CudaTranscodeSession,
    jobs: &'j [DctGridI16ToHtj2k97CodeBlockJob<'a>],
    options: Htj2k97CodeBlockOptions,
    empty: impl FnOnce() -> R,
    sink: impl FnOnce(
        &CudaContext,
        &CudaHtj2kEncodeResources,
        &CudaBufferPool,
        &CudaHtj2k97DeviceCodeblockBands,
        &'j [DctGridI16ToHtj2k97CodeBlockJob<'a>],
        Htj2k97CodeBlockOptions,
    ) -> Result<(R, CudaHtj2kEncodeStageTimings, usize), CudaTranscodeError>,
) -> Result<(R, Dwt97BatchStageTimings), CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    validate_htj2k97_codeblock_options(options)?;
    let context = session.context()?;

    let Some(first) = jobs.first() else {
        return Ok((empty(), Dwt97BatchStageTimings::default()));
    };

    let uniform = jobs.iter().all(|job| {
        job.block_cols == first.block_cols
            && job.block_rows == first.block_rows
            && job.width == first.width
            && job.height == first.height
    });
    if !uniform {
        return Err(CudaTranscodeError::UnsupportedJob(
            "CUDA 9/7 resident HT i16 batch requires uniform job geometry",
        ));
    }

    let params = htj2k97_quantize_params(options)?;
    let mut blocks = Vec::with_capacity(jobs.len() * first.block_cols * first.block_rows * 64);
    for job in jobs {
        append_i16_blocks(job.dequantized_blocks, &mut blocks);
    }
    let pool = session.buffer_pool(&context);
    let (device_bands, cuda_timings) = context
        .j2k_transcode_htj2k97_codeblock_i16_batch_resident_with_pool(
            &blocks,
            jobs.len(),
            first.block_cols,
            first.block_rows,
            first.width,
            first.height,
            params,
            &pool,
        )
        .map_err(|_| CudaTranscodeError::Kernel("CUDA 9/7 resident i16 batch dispatch failed"))?;
    let mut timings = map_batch_timings(cuda_timings);

    let resources = session.encode_resources(&context)?;
    let (output, ht_timings, ht_dispatches) = sink(
        &context,
        resources.as_ref(),
        &pool,
        &device_bands,
        jobs,
        options,
    )?;
    set_ht_encode_timings(&mut timings, ht_timings);
    timings.ht_codeblock_dispatches = ht_dispatches;
    Ok((output, timings))
}

pub(crate) fn dispatch_htj2k97_preencoded_i16_batch(
    session: &mut CudaTranscodeSession,
    jobs: &[DctGridI16ToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<PreencodedHtj2k97Component>, Dwt97BatchStageTimings), CudaTranscodeError> {
    dispatch_htj2k97_preencoded_i16_batch_with_sink(
        session,
        jobs,
        options,
        Vec::new,
        |context, resources, pool, bands, jobs, options| {
            device_bands_to_preencoded_components(context, resources, pool, bands, jobs, options)
        },
    )
}

pub(crate) fn dispatch_htj2k97_compact_preencoded_i16_batch(
    session: &mut CudaTranscodeSession,
    jobs: &[DctGridI16ToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(PreencodedHtj2k97CompactBatch, Dwt97BatchStageTimings), CudaTranscodeError> {
    dispatch_htj2k97_preencoded_i16_batch_with_sink(
        session,
        jobs,
        options,
        || PreencodedHtj2k97CompactBatch {
            payload: Vec::new(),
            components: Vec::new(),
        },
        |context, resources, pool, bands, jobs, options| {
            device_bands_to_compact_preencoded_batch(context, resources, pool, bands, jobs, options)
        },
    )
}

#[allow(clippy::type_complexity)]
fn dispatch_htj2k97_preencoded_i16_batch_groups_with_sink<'a, 'g, 'j, C, X: Default>(
    session: &mut CudaTranscodeSession,
    groups: &'g [DctGridI16ToHtj2k97CodeBlockBatch<'a, 'j>],
    options: Htj2k97CodeBlockOptions,
    missing_group_error: &'static str,
    sink: impl FnOnce(
        &CudaContext,
        &CudaHtj2kEncodeResources,
        &CudaBufferPool,
        &[ResidentDeviceGroup<'j, DctGridI16ToHtj2k97CodeBlockJob<'a>>],
        Htj2k97CodeBlockOptions,
    ) -> Result<
        (X, Vec<(usize, Vec<C>)>, CudaHtj2kEncodeStageTimings, usize),
        CudaTranscodeError,
    >,
) -> Result<(X, Vec<Vec<C>>, Dwt97BatchStageTimings), CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    validate_htj2k97_codeblock_options(options)?;
    let context = session.context()?;
    let params = htj2k97_quantize_params(options)?;
    let pool = session.buffer_pool(&context);
    let mut timings = Dwt97BatchStageTimings::default();
    let mut outputs = std::iter::repeat_with(|| None)
        .take(groups.len())
        .collect::<Vec<Option<Vec<C>>>>();
    let mut device_groups = Vec::new();

    for (group_index, group) in groups.iter().enumerate() {
        let Some(first) = group.jobs.first() else {
            outputs[group_index] = Some(Vec::new());
            continue;
        };
        let uniform = group.jobs.iter().all(|job| {
            job.block_cols == first.block_cols
                && job.block_rows == first.block_rows
                && job.width == first.width
                && job.height == first.height
        });
        if !uniform {
            return Err(CudaTranscodeError::UnsupportedJob(
                "CUDA grouped 9/7 resident HT i16 batches require uniform geometry inside each group",
            ));
        }

        let mut blocks =
            Vec::with_capacity(group.jobs.len() * first.block_cols * first.block_rows * 64);
        for job in group.jobs {
            append_i16_blocks(job.dequantized_blocks, &mut blocks);
        }
        let (bands, group_timings) = context
            .j2k_transcode_htj2k97_codeblock_i16_batch_resident_with_pool(
                &blocks,
                group.jobs.len(),
                first.block_cols,
                first.block_rows,
                first.width,
                first.height,
                params,
                &pool,
            )
            .map_err(|_| {
                CudaTranscodeError::Kernel("CUDA grouped 9/7 resident i16 batch dispatch failed")
            })?;
        accumulate_batch_timings(&mut timings, map_batch_timings(group_timings));
        device_groups.push(ResidentDeviceGroup {
            group_index,
            bands,
            jobs: group.jobs,
        });
    }

    let mut extra = X::default();
    if !device_groups.is_empty() {
        let resources = session.encode_resources(&context)?;
        let (sink_extra, encoded_groups, ht_timings, ht_dispatches) =
            sink(&context, resources.as_ref(), &pool, &device_groups, options)?;
        extra = sink_extra;
        add_ht_encode_timings(&mut timings, ht_timings);
        timings.ht_codeblock_dispatches = timings
            .ht_codeblock_dispatches
            .saturating_add(ht_dispatches);
        for (group_index, components) in encoded_groups {
            outputs[group_index] = Some(components);
        }
    }

    let outputs = outputs
        .into_iter()
        .map(|components| components.ok_or(CudaTranscodeError::Kernel(missing_group_error)))
        .collect::<Result<Vec<_>, _>>()?;

    Ok((extra, outputs, timings))
}

pub(crate) fn dispatch_htj2k97_preencoded_i16_batch_groups(
    session: &mut CudaTranscodeSession,
    groups: &[DctGridI16ToHtj2k97CodeBlockBatch<'_, '_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<Vec<PreencodedHtj2k97Component>>, Dwt97BatchStageTimings), CudaTranscodeError> {
    let ((), outputs, timings) = dispatch_htj2k97_preencoded_i16_batch_groups_with_sink(
        session,
        groups,
        options,
        "CUDA grouped 9/7 resident HT output group missing",
        |context, resources, pool, device_groups, options| {
            let (encoded_groups, ht_timings, ht_dispatches) =
                device_band_groups_to_preencoded_components(
                    context,
                    resources,
                    pool,
                    device_groups,
                    options,
                )?;
            Ok(((), encoded_groups, ht_timings, ht_dispatches))
        },
    )?;
    Ok((outputs, timings))
}

pub(crate) fn dispatch_htj2k97_compact_preencoded_i16_batch_groups(
    session: &mut CudaTranscodeSession,
    groups: &[DctGridI16ToHtj2k97CodeBlockBatch<'_, '_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(PreencodedHtj2k97CompactBatchGroups, Dwt97BatchStageTimings), CudaTranscodeError> {
    let (payload, groups, timings) = dispatch_htj2k97_preencoded_i16_batch_groups_with_sink(
        session,
        groups,
        options,
        "CUDA grouped 9/7 resident compact HT output group missing",
        |context, resources, pool, device_groups, options| {
            device_band_groups_to_compact_preencoded_components(
                context,
                resources,
                pool,
                device_groups,
                options,
            )
        },
    )?;
    Ok((
        PreencodedHtj2k97CompactBatchGroups { payload, groups },
        timings,
    ))
}

fn htj2k97_quantize_params(
    options: Htj2k97CodeBlockOptions,
) -> Result<CudaHtj2k97QuantizeParams, CudaTranscodeError> {
    let (cb_width, cb_height) = validate_htj2k97_codeblock_options(options)?;
    let inv_delta =
        |sub: J2kSubBandType| -> f32 { (1.0 / htj2k97_subband_delta(options, sub)) as f32 };
    Ok(CudaHtj2k97QuantizeParams {
        inv_delta_ll: inv_delta(J2kSubBandType::LowLow),
        inv_delta_hl: inv_delta(J2kSubBandType::HighLow),
        inv_delta_lh: inv_delta(J2kSubBandType::LowHigh),
        inv_delta_hh: inv_delta(J2kSubBandType::HighHigh),
        cb_width,
        cb_height,
    })
}

#[allow(clippy::similar_names, clippy::too_many_lines)]
fn device_bands_to_preencoded_components<J: Htj2k97ComponentJob>(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    bands: &CudaHtj2k97DeviceCodeblockBands,
    jobs: &[J],
    options: Htj2k97CodeBlockOptions,
) -> Result<
    (
        Vec<PreencodedHtj2k97Component>,
        CudaHtj2kEncodeStageTimings,
        usize,
    ),
    CudaTranscodeError,
> {
    if bands.item_count != jobs.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident 9/7 band item count mismatch",
        ));
    }

    let (ll_subbands, hl_subbands, lh_subbands, hh_subbands, ht_timings, dispatches) =
        encode_resident_subbands(context, resources, pool, bands, bands.item_count, options)?;

    let components =
        assemble_preencoded_components(jobs, ll_subbands, hl_subbands, lh_subbands, hh_subbands)?;

    Ok((components, ht_timings, dispatches))
}

#[allow(clippy::similar_names, clippy::too_many_lines)]
fn device_bands_to_compact_preencoded_batch<J: Htj2k97ComponentJob>(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    bands: &CudaHtj2k97DeviceCodeblockBands,
    jobs: &[J],
    options: Htj2k97CodeBlockOptions,
) -> Result<
    (
        PreencodedHtj2k97CompactBatch,
        CudaHtj2kEncodeStageTimings,
        usize,
    ),
    CudaTranscodeError,
> {
    if bands.item_count != jobs.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident 9/7 band item count mismatch",
        ));
    }

    let (payload, ll_subbands, hl_subbands, lh_subbands, hh_subbands, ht_timings, dispatches) =
        encode_resident_compact_subbands(
            context,
            resources,
            pool,
            bands,
            bands.item_count,
            options,
        )?;

    let components = assemble_compact_preencoded_components(
        jobs,
        ll_subbands,
        hl_subbands,
        lh_subbands,
        hh_subbands,
    )?;

    Ok((
        PreencodedHtj2k97CompactBatch {
            payload,
            components,
        },
        ht_timings,
        dispatches,
    ))
}

type ResidentSubbands = (
    Vec<PreencodedHtj2k97Subband>,
    Vec<PreencodedHtj2k97Subband>,
    Vec<PreencodedHtj2k97Subband>,
    Vec<PreencodedHtj2k97Subband>,
    CudaHtj2kEncodeStageTimings,
    usize,
);

type CompactResidentSubbands = (
    Vec<u8>,
    Vec<PreencodedHtj2k97CompactSubband>,
    Vec<PreencodedHtj2k97CompactSubband>,
    Vec<PreencodedHtj2k97CompactSubband>,
    Vec<PreencodedHtj2k97CompactSubband>,
    CudaHtj2kEncodeStageTimings,
    usize,
);

struct ResidentDeviceGroup<'a, J> {
    group_index: usize,
    bands: CudaHtj2k97DeviceCodeblockBands,
    jobs: &'a [J],
}

struct ResidentSubbandEncodePlan<'a> {
    coefficients: &'a j2k_cuda_runtime::CudaDeviceBuffer,
    coefficient_count: usize,
    jobs: Vec<CudaHtj2kEncodeCodeBlockJob>,
    shapes: Vec<(u32, u32)>,
    sub_band_type: J2kSubBandType,
    num_cbs_x: usize,
    num_cbs_y: usize,
    total_bitplanes: u8,
}

struct ResidentSubbandGroupPlans<'a, J> {
    group_index: usize,
    jobs: &'a [J],
    ll: ResidentSubbandEncodePlan<'a>,
    hl: ResidentSubbandEncodePlan<'a>,
    lh: ResidentSubbandEncodePlan<'a>,
    hh: ResidentSubbandEncodePlan<'a>,
}

impl<'a, J> ResidentSubbandGroupPlans<'a, J> {
    fn plans(&self) -> [&ResidentSubbandEncodePlan<'a>; 4] {
        [&self.ll, &self.hl, &self.lh, &self.hh]
    }
}

trait Htj2k97ComponentJob {
    fn x_rsiz(&self) -> u8;
    fn y_rsiz(&self) -> u8;
}

impl Htj2k97ComponentJob for DctGridToHtj2k97CodeBlockJob<'_> {
    fn x_rsiz(&self) -> u8 {
        self.x_rsiz
    }

    fn y_rsiz(&self) -> u8 {
        self.y_rsiz
    }
}

impl Htj2k97ComponentJob for DctGridI16ToHtj2k97CodeBlockJob<'_> {
    fn x_rsiz(&self) -> u8 {
        self.x_rsiz
    }

    fn y_rsiz(&self) -> u8 {
        self.y_rsiz
    }
}

#[allow(clippy::similar_names)]
fn assemble_preencoded_components<J: Htj2k97ComponentJob>(
    jobs: &[J],
    ll_subbands: Vec<PreencodedHtj2k97Subband>,
    hl_subbands: Vec<PreencodedHtj2k97Subband>,
    lh_subbands: Vec<PreencodedHtj2k97Subband>,
    hh_subbands: Vec<PreencodedHtj2k97Subband>,
) -> Result<Vec<PreencodedHtj2k97Component>, CudaTranscodeError> {
    if ll_subbands.len() != jobs.len()
        || hl_subbands.len() != jobs.len()
        || lh_subbands.len() != jobs.len()
        || hh_subbands.len() != jobs.len()
    {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K component assembly count mismatch",
        ));
    }

    let components = jobs
        .iter()
        .zip(ll_subbands)
        .zip(hl_subbands)
        .zip(lh_subbands)
        .zip(hh_subbands)
        .map(|((((job, ll), hl), lh), hh)| PreencodedHtj2k97Component {
            x_rsiz: job.x_rsiz(),
            y_rsiz: job.y_rsiz(),
            resolutions: vec![
                PreencodedHtj2k97Resolution { subbands: vec![ll] },
                PreencodedHtj2k97Resolution {
                    subbands: vec![hl, lh, hh],
                },
            ],
        })
        .collect();

    Ok(components)
}

#[allow(clippy::similar_names)]
fn assemble_compact_preencoded_components<J: Htj2k97ComponentJob>(
    jobs: &[J],
    ll_subbands: Vec<PreencodedHtj2k97CompactSubband>,
    hl_subbands: Vec<PreencodedHtj2k97CompactSubband>,
    lh_subbands: Vec<PreencodedHtj2k97CompactSubband>,
    hh_subbands: Vec<PreencodedHtj2k97CompactSubband>,
) -> Result<Vec<PreencodedHtj2k97CompactComponent>, CudaTranscodeError> {
    if ll_subbands.len() != jobs.len()
        || hl_subbands.len() != jobs.len()
        || lh_subbands.len() != jobs.len()
        || hh_subbands.len() != jobs.len()
    {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K compact component assembly count mismatch",
        ));
    }

    let components = jobs
        .iter()
        .zip(ll_subbands)
        .zip(hl_subbands)
        .zip(lh_subbands)
        .zip(hh_subbands)
        .map(
            |((((job, ll), hl), lh), hh)| PreencodedHtj2k97CompactComponent {
                x_rsiz: job.x_rsiz(),
                y_rsiz: job.y_rsiz(),
                resolutions: vec![
                    PreencodedHtj2k97CompactResolution { subbands: vec![ll] },
                    PreencodedHtj2k97CompactResolution {
                        subbands: vec![hl, lh, hh],
                    },
                ],
            },
        )
        .collect();

    Ok(components)
}

fn encode_resident_subbands(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    bands: &CudaHtj2k97DeviceCodeblockBands,
    item_count: usize,
    options: Htj2k97CodeBlockOptions,
) -> Result<ResidentSubbands, CudaTranscodeError> {
    let plans = [
        resident_subband_encode_plan(
            &bands.ll,
            item_count,
            bands.low_width,
            bands.low_height,
            J2kSubBandType::LowLow,
            options,
        )?,
        resident_subband_encode_plan(
            &bands.hl,
            item_count,
            bands.high_width,
            bands.low_height,
            J2kSubBandType::HighLow,
            options,
        )?,
        resident_subband_encode_plan(
            &bands.lh,
            item_count,
            bands.low_width,
            bands.high_height,
            J2kSubBandType::LowHigh,
            options,
        )?,
        resident_subband_encode_plan(
            &bands.hh,
            item_count,
            bands.high_width,
            bands.high_height,
            J2kSubBandType::HighHigh,
            options,
        )?,
    ];
    let targets: Vec<_> = plans
        .iter()
        .filter(|plan| !plan.jobs.is_empty())
        .map(|plan| CudaHtj2kEncodeResidentTarget {
            coefficients: plan.coefficients,
            coefficient_count: plan.coefficient_count,
            jobs: &plan.jobs,
        })
        .collect();
    let encoded = context
        .encode_htj2k_codeblocks_multi_resident_with_resources_and_pool(&targets, resources, pool)
        .map_err(|_| CudaTranscodeError::Kernel("CUDA resident multi-input HTJ2K encode failed"))?;
    let expected_blocks = plans.iter().map(|plan| plan.jobs.len()).sum::<usize>();
    if encoded.code_blocks().len() != expected_blocks {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident multi-input HTJ2K encode returned wrong block count",
        ));
    }
    let ht_timings = encoded.stage_timings();
    let dispatches = encoded.execution().kernel_dispatches();
    let mut encoded_blocks = encoded.into_code_blocks().into_iter();

    let ll = split_resident_subband_blocks(&plans[0], item_count, &mut encoded_blocks)?;
    let hl = split_resident_subband_blocks(&plans[1], item_count, &mut encoded_blocks)?;
    let lh = split_resident_subband_blocks(&plans[2], item_count, &mut encoded_blocks)?;
    let hh = split_resident_subband_blocks(&plans[3], item_count, &mut encoded_blocks)?;
    if encoded_blocks.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident multi-input HTJ2K output count mismatch",
        ));
    }

    Ok((ll, hl, lh, hh, ht_timings, dispatches))
}

fn encode_resident_compact_subbands(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    bands: &CudaHtj2k97DeviceCodeblockBands,
    item_count: usize,
    options: Htj2k97CodeBlockOptions,
) -> Result<CompactResidentSubbands, CudaTranscodeError> {
    let plans = [
        resident_subband_encode_plan(
            &bands.ll,
            item_count,
            bands.low_width,
            bands.low_height,
            J2kSubBandType::LowLow,
            options,
        )?,
        resident_subband_encode_plan(
            &bands.hl,
            item_count,
            bands.high_width,
            bands.low_height,
            J2kSubBandType::HighLow,
            options,
        )?,
        resident_subband_encode_plan(
            &bands.lh,
            item_count,
            bands.low_width,
            bands.high_height,
            J2kSubBandType::LowHigh,
            options,
        )?,
        resident_subband_encode_plan(
            &bands.hh,
            item_count,
            bands.high_width,
            bands.high_height,
            J2kSubBandType::HighHigh,
            options,
        )?,
    ];
    let targets: Vec<_> = plans
        .iter()
        .filter(|plan| !plan.jobs.is_empty())
        .map(|plan| CudaHtj2kEncodeResidentTarget {
            coefficients: plan.coefficients,
            coefficient_count: plan.coefficient_count,
            jobs: &plan.jobs,
        })
        .collect();
    let encoded = context
        .encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool(
            &targets, resources, pool,
        )
        .map_err(|_| {
            CudaTranscodeError::Kernel("CUDA resident compact multi-input HTJ2K encode failed")
        })?;
    let expected_blocks = plans.iter().map(|plan| plan.jobs.len()).sum::<usize>();
    if encoded.code_blocks().len() != expected_blocks {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident compact multi-input HTJ2K encode returned wrong block count",
        ));
    }
    let ht_timings = encoded.stage_timings();
    let dispatches = encoded.execution().kernel_dispatches();
    let (payload, encoded_blocks) = encoded.into_payload_and_code_blocks();
    let mut encoded_blocks = encoded_blocks.into_iter();

    let ll = split_resident_compact_subband_blocks(&plans[0], item_count, &mut encoded_blocks)?;
    let hl = split_resident_compact_subband_blocks(&plans[1], item_count, &mut encoded_blocks)?;
    let lh = split_resident_compact_subband_blocks(&plans[2], item_count, &mut encoded_blocks)?;
    let hh = split_resident_compact_subband_blocks(&plans[3], item_count, &mut encoded_blocks)?;
    if encoded_blocks.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident compact multi-input HTJ2K output count mismatch",
        ));
    }

    Ok((payload, ll, hl, lh, hh, ht_timings, dispatches))
}

#[allow(clippy::similar_names)]
fn device_band_groups_to_preencoded_components<J: Htj2k97ComponentJob>(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    groups: &[ResidentDeviceGroup<'_, J>],
    options: Htj2k97CodeBlockOptions,
) -> Result<ResidentPreencodedGroups, CudaTranscodeError> {
    let group_plans = groups
        .iter()
        .map(|group| {
            if group.bands.item_count != group.jobs.len() {
                return Err(CudaTranscodeError::Kernel(
                    "CUDA grouped resident 9/7 band item count mismatch",
                ));
            }
            Ok(ResidentSubbandGroupPlans {
                group_index: group.group_index,
                jobs: group.jobs,
                ll: resident_subband_encode_plan(
                    &group.bands.ll,
                    group.bands.item_count,
                    group.bands.low_width,
                    group.bands.low_height,
                    J2kSubBandType::LowLow,
                    options,
                )?,
                hl: resident_subband_encode_plan(
                    &group.bands.hl,
                    group.bands.item_count,
                    group.bands.high_width,
                    group.bands.low_height,
                    J2kSubBandType::HighLow,
                    options,
                )?,
                lh: resident_subband_encode_plan(
                    &group.bands.lh,
                    group.bands.item_count,
                    group.bands.low_width,
                    group.bands.high_height,
                    J2kSubBandType::LowHigh,
                    options,
                )?,
                hh: resident_subband_encode_plan(
                    &group.bands.hh,
                    group.bands.item_count,
                    group.bands.high_width,
                    group.bands.high_height,
                    J2kSubBandType::HighHigh,
                    options,
                )?,
            })
        })
        .collect::<Result<Vec<_>, CudaTranscodeError>>()?;

    let targets = group_plans
        .iter()
        .flat_map(ResidentSubbandGroupPlans::plans)
        .filter(|plan| !plan.jobs.is_empty())
        .map(|plan| CudaHtj2kEncodeResidentTarget {
            coefficients: plan.coefficients,
            coefficient_count: plan.coefficient_count,
            jobs: &plan.jobs,
        })
        .collect::<Vec<_>>();
    let encoded = context
        .encode_htj2k_codeblocks_multi_resident_with_resources_and_pool(&targets, resources, pool)
        .map_err(|_| {
            CudaTranscodeError::Kernel("CUDA grouped resident multi-input HTJ2K encode failed")
        })?;
    let expected_blocks = group_plans
        .iter()
        .flat_map(ResidentSubbandGroupPlans::plans)
        .map(|plan| plan.jobs.len())
        .sum::<usize>();
    if encoded.code_blocks().len() != expected_blocks {
        return Err(CudaTranscodeError::Kernel(
            "CUDA grouped resident multi-input HTJ2K encode returned wrong block count",
        ));
    }
    let ht_timings = encoded.stage_timings();
    let dispatches = encoded.execution().kernel_dispatches();
    let mut encoded_blocks = encoded.into_code_blocks().into_iter();
    let mut outputs = Vec::with_capacity(group_plans.len());

    for group in &group_plans {
        let item_count = group.jobs.len();
        let ll = split_resident_subband_blocks(&group.ll, item_count, &mut encoded_blocks)?;
        let hl = split_resident_subband_blocks(&group.hl, item_count, &mut encoded_blocks)?;
        let lh = split_resident_subband_blocks(&group.lh, item_count, &mut encoded_blocks)?;
        let hh = split_resident_subband_blocks(&group.hh, item_count, &mut encoded_blocks)?;
        let components = assemble_preencoded_components(group.jobs, ll, hl, lh, hh)?;
        outputs.push((group.group_index, components));
    }
    if encoded_blocks.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA grouped resident multi-input HTJ2K output count mismatch",
        ));
    }

    Ok((outputs, ht_timings, dispatches))
}

#[allow(clippy::similar_names)]
fn device_band_groups_to_compact_preencoded_components<J: Htj2k97ComponentJob>(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    groups: &[ResidentDeviceGroup<'_, J>],
    options: Htj2k97CodeBlockOptions,
) -> Result<ResidentCompactPreencodedGroups, CudaTranscodeError> {
    let group_plans = groups
        .iter()
        .map(|group| {
            if group.bands.item_count != group.jobs.len() {
                return Err(CudaTranscodeError::Kernel(
                    "CUDA grouped resident 9/7 band item count mismatch",
                ));
            }
            Ok(ResidentSubbandGroupPlans {
                group_index: group.group_index,
                jobs: group.jobs,
                ll: resident_subband_encode_plan(
                    &group.bands.ll,
                    group.bands.item_count,
                    group.bands.low_width,
                    group.bands.low_height,
                    J2kSubBandType::LowLow,
                    options,
                )?,
                hl: resident_subband_encode_plan(
                    &group.bands.hl,
                    group.bands.item_count,
                    group.bands.high_width,
                    group.bands.low_height,
                    J2kSubBandType::HighLow,
                    options,
                )?,
                lh: resident_subband_encode_plan(
                    &group.bands.lh,
                    group.bands.item_count,
                    group.bands.low_width,
                    group.bands.high_height,
                    J2kSubBandType::LowHigh,
                    options,
                )?,
                hh: resident_subband_encode_plan(
                    &group.bands.hh,
                    group.bands.item_count,
                    group.bands.high_width,
                    group.bands.high_height,
                    J2kSubBandType::HighHigh,
                    options,
                )?,
            })
        })
        .collect::<Result<Vec<_>, CudaTranscodeError>>()?;

    let targets = group_plans
        .iter()
        .flat_map(ResidentSubbandGroupPlans::plans)
        .filter(|plan| !plan.jobs.is_empty())
        .map(|plan| CudaHtj2kEncodeResidentTarget {
            coefficients: plan.coefficients,
            coefficient_count: plan.coefficient_count,
            jobs: &plan.jobs,
        })
        .collect::<Vec<_>>();
    let encoded = context
        .encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool(
            &targets, resources, pool,
        )
        .map_err(|_| {
            CudaTranscodeError::Kernel(
                "CUDA grouped resident compact multi-input HTJ2K encode failed",
            )
        })?;
    let expected_blocks = group_plans
        .iter()
        .flat_map(ResidentSubbandGroupPlans::plans)
        .map(|plan| plan.jobs.len())
        .sum::<usize>();
    if encoded.code_blocks().len() != expected_blocks {
        return Err(CudaTranscodeError::Kernel(
            "CUDA grouped resident compact multi-input HTJ2K encode returned wrong block count",
        ));
    }
    let ht_timings = encoded.stage_timings();
    let dispatches = encoded.execution().kernel_dispatches();
    let (payload, encoded_blocks) = encoded.into_payload_and_code_blocks();
    let mut encoded_blocks = encoded_blocks.into_iter();
    let mut outputs = Vec::with_capacity(group_plans.len());

    for group in &group_plans {
        let item_count = group.jobs.len();
        let ll = split_resident_compact_subband_blocks(&group.ll, item_count, &mut encoded_blocks)?;
        let hl = split_resident_compact_subband_blocks(&group.hl, item_count, &mut encoded_blocks)?;
        let lh = split_resident_compact_subband_blocks(&group.lh, item_count, &mut encoded_blocks)?;
        let hh = split_resident_compact_subband_blocks(&group.hh, item_count, &mut encoded_blocks)?;
        let components = assemble_compact_preencoded_components(group.jobs, ll, hl, lh, hh)?;
        outputs.push((group.group_index, components));
    }
    if encoded_blocks.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA grouped resident compact multi-input HTJ2K output count mismatch",
        ));
    }

    Ok((payload, outputs, ht_timings, dispatches))
}

#[allow(clippy::too_many_arguments)]
fn resident_subband_encode_plan(
    coefficients: &CudaPooledDeviceBuffer,
    item_count: usize,
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    options: Htj2k97CodeBlockOptions,
) -> Result<ResidentSubbandEncodePlan<'_>, CudaTranscodeError> {
    let coefficient_buffer = coefficients
        .as_device_buffer()
        .ok_or(CudaTranscodeError::Kernel(
            "CUDA resident 9/7 pooled band checkout missing",
        ))?;
    let cb_width = htj2k97_code_block_dim(options.code_block_width_exp)?;
    let cb_height = htj2k97_code_block_dim(options.code_block_height_exp)?;
    let num_cbs_x = if width == 0 {
        0
    } else {
        width.div_ceil(cb_width)
    };
    let num_cbs_y = if height == 0 {
        0
    } else {
        height.div_ceil(cb_height)
    };
    let total_bitplanes = htj2k97_subband_total_bitplanes(options, sub_band_type);
    if width == 0 || height == 0 {
        return Ok(ResidentSubbandEncodePlan {
            coefficients: coefficient_buffer,
            coefficient_count: 0,
            jobs: Vec::new(),
            shapes: Vec::new(),
            sub_band_type,
            num_cbs_x: 0,
            num_cbs_y: 0,
            total_bitplanes,
        });
    }

    let item_stride = width.checked_mul(height).ok_or(CudaTranscodeError::Kernel(
        "CUDA resident 9/7 band dimensions overflow",
    ))?;
    let coefficient_count =
        item_stride
            .checked_mul(item_count)
            .ok_or(CudaTranscodeError::Kernel(
                "CUDA resident 9/7 band item count overflow",
            ))?;
    let mut encode_jobs = Vec::with_capacity(item_count * num_cbs_x * num_cbs_y);
    let mut shapes = Vec::with_capacity(item_count * num_cbs_x * num_cbs_y);
    for item in 0..item_count {
        let item_offset = item
            .checked_mul(item_stride)
            .ok_or(CudaTranscodeError::Kernel(
                "CUDA resident 9/7 band item offset overflow",
            ))?;
        for cby in 0..num_cbs_y {
            for cbx in 0..num_cbs_x {
                let block_width = (width - cbx * cb_width).min(cb_width);
                let block_height = (height - cby * cb_height).min(cb_height);
                let block_offset = cby
                    .checked_mul(cb_height)
                    .and_then(|value| value.checked_mul(width))
                    .and_then(|value| {
                        value.checked_add(cbx.checked_mul(cb_width)?.checked_mul(block_height)?)
                    })
                    .and_then(|value| value.checked_add(item_offset))
                    .ok_or(CudaTranscodeError::Kernel(
                        "CUDA resident 9/7 code-block offset overflow",
                    ))?;
                encode_jobs.push(CudaHtj2kEncodeCodeBlockJob {
                    coefficient_offset: to_u32(block_offset)?,
                    width: to_u32(block_width)?,
                    height: to_u32(block_height)?,
                    total_bitplanes,
                    target_coding_passes: 1,
                });
                shapes.push((to_u32(block_width)?, to_u32(block_height)?));
            }
        }
    }

    Ok(ResidentSubbandEncodePlan {
        coefficients: coefficient_buffer,
        coefficient_count,
        jobs: encode_jobs,
        shapes,
        sub_band_type,
        num_cbs_x,
        num_cbs_y,
        total_bitplanes,
    })
}

fn split_resident_subband_blocks(
    plan: &ResidentSubbandEncodePlan<'_>,
    item_count: usize,
    encoded_blocks: &mut impl Iterator<Item = CudaHtj2kEncodedCodeBlock>,
) -> Result<Vec<PreencodedHtj2k97Subband>, CudaTranscodeError> {
    let blocks_per_item =
        plan.num_cbs_x
            .checked_mul(plan.num_cbs_y)
            .ok_or(CudaTranscodeError::Kernel(
                "CUDA resident HTJ2K code-block count overflow",
            ))?;
    let mut shape_index = 0usize;
    let mut subbands = Vec::with_capacity(item_count);
    for _ in 0..item_count {
        let mut code_blocks = Vec::with_capacity(blocks_per_item);
        for _ in 0..blocks_per_item {
            let (width, height) =
                *plan
                    .shapes
                    .get(shape_index)
                    .ok_or(CudaTranscodeError::Kernel(
                        "CUDA resident HTJ2K shape count mismatch",
                    ))?;
            shape_index = shape_index.saturating_add(1);
            let encoded = encoded_blocks.next().ok_or(CudaTranscodeError::Kernel(
                "CUDA resident HTJ2K output count mismatch",
            ))?;
            let (data, cleanup_length, refinement_length, num_coding_passes, num_zero_bitplanes) =
                encoded.into_parts();
            code_blocks.push(PreencodedHtj2k97CodeBlock {
                width,
                height,
                encoded: EncodedHtJ2kCodeBlock {
                    data,
                    cleanup_length,
                    refinement_length,
                    num_coding_passes,
                    num_zero_bitplanes,
                },
            });
        }
        subbands.push(PreencodedHtj2k97Subband {
            sub_band_type: plan.sub_band_type,
            num_cbs_x: to_u32(plan.num_cbs_x)?,
            num_cbs_y: to_u32(plan.num_cbs_y)?,
            total_bitplanes: plan.total_bitplanes,
            code_blocks,
        });
    }
    if shape_index != plan.shapes.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K shape count mismatch",
        ));
    }
    Ok(subbands)
}

fn split_resident_compact_subband_blocks(
    plan: &ResidentSubbandEncodePlan<'_>,
    item_count: usize,
    encoded_blocks: &mut impl Iterator<Item = CudaHtj2kCompactEncodedCodeBlock>,
) -> Result<Vec<PreencodedHtj2k97CompactSubband>, CudaTranscodeError> {
    let blocks_per_item =
        plan.num_cbs_x
            .checked_mul(plan.num_cbs_y)
            .ok_or(CudaTranscodeError::Kernel(
                "CUDA resident HTJ2K compact code-block count overflow",
            ))?;
    let mut shape_index = 0usize;
    let mut subbands = Vec::with_capacity(item_count);
    for _ in 0..item_count {
        let mut code_blocks = Vec::with_capacity(blocks_per_item);
        for _ in 0..blocks_per_item {
            let (width, height) =
                *plan
                    .shapes
                    .get(shape_index)
                    .ok_or(CudaTranscodeError::Kernel(
                        "CUDA resident HTJ2K compact shape count mismatch",
                    ))?;
            shape_index = shape_index.saturating_add(1);
            let encoded = encoded_blocks.next().ok_or(CudaTranscodeError::Kernel(
                "CUDA resident HTJ2K compact output count mismatch",
            ))?;
            let (
                payload_range,
                cleanup_length,
                refinement_length,
                num_coding_passes,
                num_zero_bitplanes,
            ) = encoded.into_parts();
            code_blocks.push(PreencodedHtj2k97CompactCodeBlock {
                width,
                height,
                payload_range,
                cleanup_length,
                refinement_length,
                num_coding_passes,
                num_zero_bitplanes,
            });
        }
        subbands.push(PreencodedHtj2k97CompactSubband {
            sub_band_type: plan.sub_band_type,
            num_cbs_x: to_u32(plan.num_cbs_x)?,
            num_cbs_y: to_u32(plan.num_cbs_y)?,
            total_bitplanes: plan.total_bitplanes,
            code_blocks,
        });
    }
    if shape_index != plan.shapes.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K compact shape count mismatch",
        ));
    }
    Ok(subbands)
}

fn to_u32(value: usize) -> Result<u32, CudaTranscodeError> {
    u32::try_from(value).map_err(|_| CudaTranscodeError::Kernel("CUDA value exceeds u32"))
}

fn cuda_htj2k_encode_tables() -> CudaHtj2kEncodeTables<'static> {
    CudaHtj2kEncodeTables {
        vlc_table0: j2k_native::ht_vlc_encode_table0(),
        vlc_table1: j2k_native::ht_vlc_encode_table1(),
        uvlc_table: j2k_native::ht_uvlc_encode_table_bytes(),
    }
}

fn validate_band_len(
    band: &[i32],
    item_count: usize,
    item_size: usize,
) -> Result<(), CudaTranscodeError> {
    let expected = item_count
        .checked_mul(item_size)
        .ok_or(CudaTranscodeError::Kernel(
            "CUDA 9/7 code-block band length overflow",
        ))?;
    if band.len() != expected {
        return Err(CudaTranscodeError::Kernel(
            "CUDA 9/7 code-block band output length mismatch",
        ));
    }
    Ok(())
}

fn validate_htj2k97_codeblock_options(
    options: Htj2k97CodeBlockOptions,
) -> Result<(usize, usize), CudaTranscodeError> {
    j2k_transcode::htj2k97_codeblock_oracle::validate_htj2k97_codeblock_options(options)
        .map_err(CudaTranscodeError::UnsupportedJob)
}

fn htj2k97_code_block_dim(exp_minus_two: u8) -> Result<usize, CudaTranscodeError> {
    1usize
        .checked_shl(u32::from(exp_minus_two) + 2)
        .ok_or(CudaTranscodeError::UnsupportedJob(
            "CUDA 9/7 code-block exponent is too large",
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestComponentJob {
        x_rsiz: u8,
        y_rsiz: u8,
    }

    impl Htj2k97ComponentJob for TestComponentJob {
        fn x_rsiz(&self) -> u8 {
            self.x_rsiz
        }

        fn y_rsiz(&self) -> u8 {
            self.y_rsiz
        }
    }

    fn test_subband(sub_band_type: J2kSubBandType, marker: u8) -> PreencodedHtj2k97Subband {
        PreencodedHtj2k97Subband {
            sub_band_type,
            num_cbs_x: 1,
            num_cbs_y: 1,
            total_bitplanes: 8,
            code_blocks: vec![PreencodedHtj2k97CodeBlock {
                width: 1,
                height: 1,
                encoded: EncodedHtJ2kCodeBlock {
                    data: vec![marker; 8],
                    cleanup_length: 8,
                    refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 0,
                },
            }],
        }
    }

    fn payload_ptr(subband: &PreencodedHtj2k97Subband) -> usize {
        subband.code_blocks[0].encoded.data.as_ptr() as usize
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn assemble_preencoded_components_moves_subband_payloads_without_clone() {
        let jobs = [TestComponentJob {
            x_rsiz: 1,
            y_rsiz: 2,
        }];
        let ll = vec![test_subband(J2kSubBandType::LowLow, 1)];
        let hl = vec![test_subband(J2kSubBandType::HighLow, 2)];
        let lh = vec![test_subband(J2kSubBandType::LowHigh, 3)];
        let hh = vec![test_subband(J2kSubBandType::HighHigh, 4)];
        let ll_ptr = payload_ptr(&ll[0]);
        let hl_ptr = payload_ptr(&hl[0]);
        let lh_ptr = payload_ptr(&lh[0]);
        let hh_ptr = payload_ptr(&hh[0]);

        let components = assemble_preencoded_components(&jobs, ll, hl, lh, hh).expect("components");

        assert_eq!(components.len(), 1);
        assert_eq!(components[0].x_rsiz, 1);
        assert_eq!(components[0].y_rsiz, 2);
        assert_eq!(
            payload_ptr(&components[0].resolutions[0].subbands[0]),
            ll_ptr
        );
        assert_eq!(
            payload_ptr(&components[0].resolutions[1].subbands[0]),
            hl_ptr
        );
        assert_eq!(
            payload_ptr(&components[0].resolutions[1].subbands[1]),
            lh_ptr
        );
        assert_eq!(
            payload_ptr(&components[0].resolutions[1].subbands[2]),
            hh_ptr
        );
    }

    #[test]
    fn append_i16_blocks_preserves_prefix_and_flattens_blocks() {
        let mut first = [0i16; 64];
        first[0] = -7;
        first[63] = 42;
        let mut second = [0i16; 64];
        second[1] = 9;
        second[62] = -11;
        let mut out = vec![123];

        append_i16_blocks(&[first, second], &mut out);

        assert_eq!(out[0], 123);
        assert_eq!(out.len(), 1 + 128);
        assert_eq!(out[1], -7);
        assert_eq!(out[64], 42);
        assert_eq!(out[66], 9);
        assert_eq!(out[127], -11);
    }
}
