// SPDX-License-Identifier: Apache-2.0

//! CUDA dispatch boundary for the transcode accelerator.
//!
//! Each function uploads a DCT-grid job to the device, runs the ported kernel
//! in `signinum-cuda-runtime`, and returns wavelet bands / prequantized
//! components matching the `signinum-transcode` scalar oracle. Kernels are
//! wired incrementally; until a path is wired its dispatch returns a typed
//! [`CudaTranscodeError::UnsupportedJob`], which Auto mode treats as a scalar
//! fallback and Explicit mode surfaces as an error.

use signinum_j2k_native::EncodedHtJ2kCodeBlock;
use signinum_transcode::accelerator::{
    DctGridToDwt53Job, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob,
    DctGridToReversibleDwt53Job, Dwt97BatchStageTimings, Htj2k97CodeBlockOptions, J2kSubBandType,
    PreencodedHtj2k97CodeBlock, PreencodedHtj2k97Component, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband, ReversibleDwt53FirstLevel,
};
use signinum_transcode::dct53_2d::Dwt53TwoDimensional;
use signinum_transcode::dct97_2d::Dwt97TwoDimensional;
use signinum_transcode::htj2k97_codeblock_oracle::{
    htj2k97_subband_delta, htj2k97_subband_total_bitplanes,
};

use std::sync::OnceLock;

use signinum_cuda_runtime::{
    transcode_kernels_built, CudaContext, CudaDwt97BatchStageTimings, CudaHtj2k97CodeblockBands,
    CudaHtj2k97DeviceCodeblockBands, CudaHtj2k97QuantizeParams, CudaHtj2kEncodeCodeBlockJob,
    CudaHtj2kEncodeResources, CudaHtj2kEncodeTables, CudaTranscodeDwt97Bands,
    CudaTranscodeReversible53Bands,
};

use crate::CudaTranscodeError;

/// Returned until a given kernel path is wired to `signinum-cuda-runtime`.
const NOT_WIRED: CudaTranscodeError =
    CudaTranscodeError::UnsupportedJob("signinum-transcode-cuda kernel not yet wired");

/// A process-wide CUDA context, created on first use and reused thereafter.
///
/// `CudaContext::system_default()` creates a fresh context every call, and the
/// first kernel launch in a context JIT-compiles the transcode PTX. Creating a
/// context (and re-JITing the kernels) per dispatch dominated wall-clock for
/// real workloads; sharing one context pays that cost once. The context is an
/// `Arc` handle (`Send + Sync`), so cloning it out is cheap.
fn shared_context() -> Result<CudaContext, CudaTranscodeError> {
    static SHARED: OnceLock<CudaContext> = OnceLock::new();
    if let Some(context) = SHARED.get() {
        return Ok(context.clone());
    }

    let context = CudaContext::system_default().map_err(|_| CudaTranscodeError::CudaUnavailable)?;
    let _ = SHARED.set(context);
    SHARED
        .get()
        .cloned()
        .ok_or(CudaTranscodeError::CudaUnavailable)
}

/// Flatten `&[[i16; 64]]` into the contiguous `&[i16]` the runtime job expects.
fn flatten_blocks(blocks: &[[i16; 64]]) -> &[i16] {
    // SAFETY: `[[i16; 64]]` is laid out contiguously, so reinterpreting it as a
    // flat `&[i16]` of `len * 64` elements is a read-only view with identical
    // layout, alignment, and lifetime.
    unsafe { std::slice::from_raw_parts(blocks.as_ptr().cast::<i16>(), blocks.len() * 64) }
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
    job: DctGridToReversibleDwt53Job<'_>,
) -> Result<ReversibleDwt53FirstLevel, CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let context = shared_context()?;
    run_reversible(&context, job)
}

pub(crate) fn dispatch_reversible_dwt53_batch(
    jobs: &[DctGridToReversibleDwt53Job<'_>],
) -> Result<Vec<ReversibleDwt53FirstLevel>, CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let context = shared_context()?;
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
        ht_codeblock_dispatches: timings.ht_codeblock_dispatches,
        readback_us: timings.readback_us,
    }
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
    job: DctGridToDwt97Job<'_>,
) -> Result<Dwt97TwoDimensional<f64>, CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let context = shared_context()?;
    run_dwt97(&context, job)
}

pub(crate) fn dispatch_dwt97_batch(
    jobs: &[DctGridToDwt97Job<'_>],
) -> Result<(Vec<Dwt97TwoDimensional<f64>>, Dwt97BatchStageTimings), CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let context = shared_context()?;

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
        .j2k_transcode_dwt97_batch(
            &blocks,
            jobs.len(),
            first.block_cols,
            first.block_rows,
            first.width,
            first.height,
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
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<PrequantizedHtj2k97Component>, Dwt97BatchStageTimings), CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let (cb_width, cb_height) = validate_htj2k97_codeblock_options(options)?;
    let context = shared_context()?;

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
        .j2k_transcode_htj2k97_codeblock_batch(
            &blocks,
            jobs.len(),
            first.block_cols,
            first.block_rows,
            first.width,
            first.height,
            params,
        )
        .map_err(|_| CudaTranscodeError::Kernel("CUDA 9/7 code-block batch dispatch failed"))?;

    let components = codeblock_bands_to_components(&codeblock_bands, jobs, options)?;
    Ok((components, map_batch_timings(timings)))
}

pub(crate) fn dispatch_htj2k97_preencoded_batch(
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<PreencodedHtj2k97Component>, Dwt97BatchStageTimings), CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    validate_htj2k97_codeblock_options(options)?;
    let context = shared_context()?;

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
    let (device_bands, mut timings) = context
        .j2k_transcode_htj2k97_codeblock_batch_resident(
            &blocks,
            jobs.len(),
            first.block_cols,
            first.block_rows,
            first.width,
            first.height,
            params,
        )
        .map_err(|_| CudaTranscodeError::Kernel("CUDA 9/7 resident batch dispatch failed"))?;

    let resources = shared_encode_resources(&context)?;
    let (components, ht_encode_us, ht_dispatches) =
        device_bands_to_preencoded_components(&context, resources, &device_bands, jobs, options)?;
    timings.ht_encode_us = ht_encode_us;
    timings.ht_codeblock_dispatches = ht_dispatches;
    Ok((components, map_batch_timings(timings)))
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
fn device_bands_to_preencoded_components(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    bands: &CudaHtj2k97DeviceCodeblockBands,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<PreencodedHtj2k97Component>, u128, usize), CudaTranscodeError> {
    if bands.item_count != jobs.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident 9/7 band item count mismatch",
        ));
    }

    let (ll_subbands, mut ht_encode_us, mut dispatches) = encode_resident_subband(
        context,
        resources,
        &bands.ll,
        bands.item_count,
        bands.low_width,
        bands.low_height,
        J2kSubBandType::LowLow,
        options,
    )?;
    let (hl_subbands, hl_us, hl_dispatches) = encode_resident_subband(
        context,
        resources,
        &bands.hl,
        bands.item_count,
        bands.high_width,
        bands.low_height,
        J2kSubBandType::HighLow,
        options,
    )?;
    ht_encode_us = ht_encode_us.saturating_add(hl_us);
    dispatches = dispatches.saturating_add(hl_dispatches);
    let (lh_subbands, lh_us, lh_dispatches) = encode_resident_subband(
        context,
        resources,
        &bands.lh,
        bands.item_count,
        bands.low_width,
        bands.high_height,
        J2kSubBandType::LowHigh,
        options,
    )?;
    ht_encode_us = ht_encode_us.saturating_add(lh_us);
    dispatches = dispatches.saturating_add(lh_dispatches);
    let (hh_subbands, hh_us, hh_dispatches) = encode_resident_subband(
        context,
        resources,
        &bands.hh,
        bands.item_count,
        bands.high_width,
        bands.high_height,
        J2kSubBandType::HighHigh,
        options,
    )?;
    ht_encode_us = ht_encode_us.saturating_add(hh_us);
    dispatches = dispatches.saturating_add(hh_dispatches);

    let components = jobs
        .iter()
        .enumerate()
        .map(|(idx, job)| PreencodedHtj2k97Component {
            x_rsiz: job.x_rsiz,
            y_rsiz: job.y_rsiz,
            resolutions: vec![
                PreencodedHtj2k97Resolution {
                    subbands: vec![ll_subbands[idx].clone()],
                },
                PreencodedHtj2k97Resolution {
                    subbands: vec![
                        hl_subbands[idx].clone(),
                        lh_subbands[idx].clone(),
                        hh_subbands[idx].clone(),
                    ],
                },
            ],
        })
        .collect();

    Ok((components, ht_encode_us, dispatches))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn encode_resident_subband(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    coefficients: &signinum_cuda_runtime::CudaDeviceBuffer,
    item_count: usize,
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<PreencodedHtj2k97Subband>, u128, usize), CudaTranscodeError> {
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
    if width == 0 || height == 0 {
        return Ok((
            (0..item_count)
                .map(|_| PreencodedHtj2k97Subband {
                    sub_band_type,
                    num_cbs_x: 0,
                    num_cbs_y: 0,
                    total_bitplanes: 0,
                    code_blocks: Vec::new(),
                })
                .collect(),
            0,
            0,
        ));
    }

    let total_bitplanes = htj2k97_subband_total_bitplanes(options, sub_band_type);
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

    let encoded = context
        .encode_htj2k_codeblocks_resident_with_resources(
            coefficients,
            coefficient_count,
            &encode_jobs,
            resources,
        )
        .map_err(|_| CudaTranscodeError::Kernel("CUDA resident HTJ2K encode failed"))?;
    if encoded.code_blocks().len() != encode_jobs.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K encode returned wrong block count",
        ));
    }

    let blocks_per_item = num_cbs_x
        .checked_mul(num_cbs_y)
        .ok_or(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K code-block count overflow",
        ))?;
    let mut encoded_iter = encoded.code_blocks().iter();
    let mut shape_iter = shapes.into_iter();
    let mut subbands = Vec::with_capacity(item_count);
    for _ in 0..item_count {
        let mut code_blocks = Vec::with_capacity(blocks_per_item);
        for _ in 0..blocks_per_item {
            let (width, height) = shape_iter.next().ok_or(CudaTranscodeError::Kernel(
                "CUDA resident HTJ2K shape count mismatch",
            ))?;
            let encoded = encoded_iter.next().ok_or(CudaTranscodeError::Kernel(
                "CUDA resident HTJ2K output count mismatch",
            ))?;
            code_blocks.push(PreencodedHtj2k97CodeBlock {
                width,
                height,
                encoded: EncodedHtJ2kCodeBlock {
                    data: encoded.data().to_vec(),
                    cleanup_length: encoded.cleanup_length(),
                    refinement_length: encoded.refinement_length(),
                    num_coding_passes: encoded.num_coding_passes(),
                    num_zero_bitplanes: encoded.num_zero_bitplanes(),
                },
            });
        }
        subbands.push(PreencodedHtj2k97Subband {
            sub_band_type,
            num_cbs_x: to_u32(num_cbs_x)?,
            num_cbs_y: to_u32(num_cbs_y)?,
            total_bitplanes,
            code_blocks,
        });
    }
    if encoded_iter.next().is_some() || shape_iter.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K output count mismatch",
        ));
    }

    Ok((
        subbands,
        encoded.stage_timings().ht_encode_us,
        encoded.execution().kernel_dispatches(),
    ))
}

fn to_u32(value: usize) -> Result<u32, CudaTranscodeError> {
    u32::try_from(value).map_err(|_| CudaTranscodeError::Kernel("CUDA value exceeds u32"))
}

fn cuda_htj2k_encode_tables() -> CudaHtj2kEncodeTables<'static> {
    CudaHtj2kEncodeTables {
        vlc_table0: signinum_j2k_native::ht_vlc_encode_table0(),
        vlc_table1: signinum_j2k_native::ht_vlc_encode_table1(),
        uvlc_table: ht_uvlc_encode_table_bytes(),
    }
}

fn shared_encode_resources(
    context: &CudaContext,
) -> Result<&'static CudaHtj2kEncodeResources, CudaTranscodeError> {
    static RESOURCES: OnceLock<CudaHtj2kEncodeResources> = OnceLock::new();
    if let Some(resources) = RESOURCES.get() {
        return Ok(resources);
    }
    let resources = context
        .upload_htj2k_encode_resources(cuda_htj2k_encode_tables())
        .map_err(|_| CudaTranscodeError::Kernel("CUDA HTJ2K encode resource upload failed"))?;
    let _ = RESOURCES.set(resources);
    RESOURCES.get().ok_or(CudaTranscodeError::Kernel(
        "CUDA HTJ2K encode resources unavailable",
    ))
}

fn ht_uvlc_encode_table_bytes() -> &'static [u8] {
    static TABLE: OnceLock<Vec<u8>> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            signinum_j2k_native::ht_uvlc_encode_table()
                .iter()
                .flat_map(|entry| {
                    [
                        entry.pre,
                        entry.pre_len,
                        entry.suf,
                        entry.suf_len,
                        entry.ext,
                        entry.ext_len,
                    ]
                })
                .collect()
        })
        .as_slice()
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
    if options.bit_depth == 0
        || options.bit_depth > 30
        || options.guard_bits > 30
        || !options.irreversible_quantization_scale.is_finite()
        || options.irreversible_quantization_scale <= 0.0
    {
        return Err(CudaTranscodeError::UnsupportedJob(
            "CUDA 9/7 code-block options are outside supported numeric range",
        ));
    }
    let subband_scales = options.irreversible_quantization_subband_scales;
    if [
        subband_scales.low_low,
        subband_scales.high_low,
        subband_scales.low_high,
        subband_scales.high_high,
    ]
    .iter()
    .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(CudaTranscodeError::UnsupportedJob(
            "CUDA 9/7 code-block quantization options are outside supported range",
        ));
    }

    let cb_width = htj2k97_code_block_dim(options.code_block_width_exp)?;
    let cb_height = htj2k97_code_block_dim(options.code_block_height_exp)?;
    if cb_width > 1024
        || cb_height > 1024
        || cb_width
            .checked_mul(cb_height)
            .is_none_or(|area| area > 4096)
    {
        return Err(CudaTranscodeError::UnsupportedJob(
            "CUDA 9/7 code-block dimensions exceed HTJ2K limits",
        ));
    }

    for subband in [
        J2kSubBandType::LowLow,
        J2kSubBandType::HighLow,
        J2kSubBandType::LowHigh,
        J2kSubBandType::HighHigh,
    ] {
        let delta = htj2k97_subband_delta(options, subband);
        if !delta.is_finite()
            || delta <= 0.0
            || htj2k97_subband_total_bitplanes(options, subband) > 30
        {
            return Err(CudaTranscodeError::UnsupportedJob(
                "CUDA 9/7 code-block quantization options are outside supported range",
            ));
        }
    }

    Ok((cb_width, cb_height))
}

fn htj2k97_code_block_dim(exp_minus_two: u8) -> Result<usize, CudaTranscodeError> {
    1usize
        .checked_shl(u32::from(exp_minus_two) + 2)
        .ok_or(CudaTranscodeError::UnsupportedJob(
            "CUDA 9/7 code-block exponent is too large",
        ))
}
