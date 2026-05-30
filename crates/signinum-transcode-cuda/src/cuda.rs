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

use crate::CudaTranscodeError;

/// Returned until a given kernel path is wired to `signinum-cuda-runtime`.
const NOT_WIRED: CudaTranscodeError =
    CudaTranscodeError::UnsupportedJob("signinum-transcode-cuda kernel not yet wired");

pub(crate) fn dispatch_reversible_dwt53(
    _job: DctGridToReversibleDwt53Job<'_>,
) -> Result<ReversibleDwt53FirstLevel, CudaTranscodeError> {
    Err(NOT_WIRED)
}

pub(crate) fn dispatch_reversible_dwt53_batch(
    _jobs: &[DctGridToReversibleDwt53Job<'_>],
) -> Result<Vec<ReversibleDwt53FirstLevel>, CudaTranscodeError> {
    Err(NOT_WIRED)
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
