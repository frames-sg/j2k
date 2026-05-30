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
    DctGridToReversibleDwt53Job, Dwt97BatchStageTimings, Htj2k97CodeBlockOptions,
    PrequantizedHtj2k97Component, ReversibleDwt53FirstLevel,
};
use signinum_transcode::dct53_2d::Dwt53TwoDimensional;
use signinum_transcode::dct97_2d::Dwt97TwoDimensional;

use signinum_cuda_runtime::{transcode_kernels_built, CudaContext, CudaTranscodeReversible53Bands};

use crate::CudaTranscodeError;

/// Returned until a given kernel path is wired to `signinum-cuda-runtime`.
const NOT_WIRED: CudaTranscodeError =
    CudaTranscodeError::UnsupportedJob("signinum-transcode-cuda kernel not yet wired");

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
    let context = CudaContext::system_default().map_err(|_| CudaTranscodeError::CudaUnavailable)?;
    run_reversible(&context, job)
}

pub(crate) fn dispatch_reversible_dwt53_batch(
    jobs: &[DctGridToReversibleDwt53Job<'_>],
) -> Result<Vec<ReversibleDwt53FirstLevel>, CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    let context = CudaContext::system_default().map_err(|_| CudaTranscodeError::CudaUnavailable)?;
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

pub(crate) fn dispatch_dwt97(
    _job: DctGridToDwt97Job<'_>,
) -> Result<Dwt97TwoDimensional<f64>, CudaTranscodeError> {
    Err(NOT_WIRED)
}

pub(crate) fn dispatch_dwt97_batch(
    _jobs: &[DctGridToDwt97Job<'_>],
) -> Result<(Vec<Dwt97TwoDimensional<f64>>, Dwt97BatchStageTimings), CudaTranscodeError> {
    Err(NOT_WIRED)
}

pub(crate) fn dispatch_htj2k97_codeblock_batch(
    _jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    _options: Htj2k97CodeBlockOptions,
) -> Result<Vec<PrequantizedHtj2k97Component>, CudaTranscodeError> {
    Err(NOT_WIRED)
}
