// SPDX-License-Identifier: Apache-2.0

//! CUDA dispatch boundary for the transcode accelerator.
//!
//! Each function uploads a DCT-grid job to the device, runs the ported kernel
//! in `signinum-cuda-runtime`, and returns wavelet bands / prequantized
//! components matching the `signinum-transcode` scalar oracle. Kernels are
//! wired incrementally; until a path is wired its dispatch returns a typed
//! [`CudaTranscodeError::UnsupportedJob`], which Auto mode treats as a scalar
//! fallback and Explicit mode surfaces as an error.

use signinum_transcode::accelerator::{
    DctGridToDwt53Job, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob,
    DctGridToReversibleDwt53Job, Dwt97BatchStageTimings, Htj2k97CodeBlockOptions, J2kSubBandType,
    PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component, PrequantizedHtj2k97Resolution,
    PrequantizedHtj2k97Subband, ReversibleDwt53FirstLevel,
};
use signinum_transcode::dct53_2d::Dwt53TwoDimensional;
use signinum_transcode::dct97_2d::Dwt97TwoDimensional;
use signinum_transcode::htj2k97_codeblock_oracle::{
    htj2k97_subband_delta, htj2k97_subband_total_bitplanes,
};

use std::sync::OnceLock;

use signinum_cuda_runtime::{
    transcode_kernels_built, CudaContext, CudaDwt97BatchStageTimings, CudaHtj2k97CodeblockBands,
    CudaHtj2k97QuantizeParams, CudaTranscodeDwt97Bands, CudaTranscodeReversible53Bands,
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
) -> PrequantizedHtj2k97Subband {
    let cb_width = 1usize << (options.code_block_width_exp + 2);
    let cb_height = 1usize << (options.code_block_height_exp + 2);
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let mut code_blocks = Vec::with_capacity(num_cbs_x * num_cbs_y);
    let mut offset = 0usize;
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let block_width = (width - cbx * cb_width).min(cb_width);
            let block_height = (height - cby * cb_height).min(cb_height);
            let len = block_width * block_height;
            code_blocks.push(PrequantizedHtj2k97CodeBlock {
                coefficients: data[offset..offset + len].to_vec(),
                width: block_width as u32,
                height: block_height as u32,
            });
            offset += len;
        }
    }
    PrequantizedHtj2k97Subband {
        sub_band_type,
        num_cbs_x: num_cbs_x as u32,
        num_cbs_y: num_cbs_y as u32,
        total_bitplanes: htj2k97_subband_total_bitplanes(options, sub_band_type),
        code_blocks,
    }
}

/// Reslice the per-item code-block bands into prequantized HTJ2K components,
/// one per job (resolution nesting `[[LL], [HL, LH, HH]]`).
#[allow(clippy::similar_names)]
fn codeblock_bands_to_components(
    bands: &CudaHtj2k97CodeblockBands,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Vec<PrequantizedHtj2k97Component> {
    let ll_size = bands.low_width * bands.low_height;
    let hl_size = bands.high_width * bands.low_height;
    let lh_size = bands.low_width * bands.high_height;
    let hh_size = bands.high_width * bands.high_height;
    jobs.iter()
        .enumerate()
        .map(|(item, job)| {
            let ll = &bands.ll[item * ll_size..(item + 1) * ll_size];
            let hl = &bands.hl[item * hl_size..(item + 1) * hl_size];
            let lh = &bands.lh[item * lh_size..(item + 1) * lh_size];
            let hh = &bands.hh[item * hh_size..(item + 1) * hh_size];
            PrequantizedHtj2k97Component {
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
                        )],
                    },
                    PrequantizedHtj2k97Resolution {
                        subbands: vec![
                            subband_from_codeblock_slice(
                                hl,
                                bands.high_width,
                                bands.low_height,
                                J2kSubBandType::HighLow,
                                options,
                            ),
                            subband_from_codeblock_slice(
                                lh,
                                bands.low_width,
                                bands.high_height,
                                J2kSubBandType::LowHigh,
                                options,
                            ),
                            subband_from_codeblock_slice(
                                hh,
                                bands.high_width,
                                bands.high_height,
                                J2kSubBandType::HighHigh,
                                options,
                            ),
                        ],
                    },
                ],
            }
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
        cb_width: 1usize << (options.code_block_width_exp + 2),
        cb_height: 1usize << (options.code_block_height_exp + 2),
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

    let components = codeblock_bands_to_components(&codeblock_bands, jobs, options);
    Ok((components, map_batch_timings(timings)))
}
