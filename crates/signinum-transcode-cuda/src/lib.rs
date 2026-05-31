// SPDX-License-Identifier: Apache-2.0

//! CUDA acceleration for coefficient-domain JPEG to HTJ2K transcode stages.
//!
//! Mirrors `signinum-transcode-metal`: it implements
//! [`DctToWaveletStageAccelerator`] for direct DCT-grid to one-level 5/3 and 9/7
//! wavelet projections (and the fused 9/7 HTJ2K code-block path), so JPEG can be
//! transcoded to HTJ2K without an IDCT->pixels->DWT spatial round-trip. The CPU
//! scalar code in `signinum-transcode` remains the oracle and fallback; this
//! crate never reimplements it.
//!
//! The actual GPU kernels live in `signinum-cuda-runtime` (the repo keeps all
//! `.cu` + `build.rs` PTX there). The GPU path is gated behind the
//! `cuda-runtime` feature; without it this accelerator behaves like Metal's
//! non-macOS path (Explicit -> typed `Err`, Auto -> `Ok(None)` scalar fallback).

#[cfg(feature = "cuda-runtime")]
mod cuda;

use core::fmt;

use signinum_transcode::accelerator::{
    DctGridToDwt53Job, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob,
    DctGridToReversibleDwt53Job, DctToWaveletStageAccelerator, Dwt97BatchStageTimings,
    Htj2k97CodeBlockOptions, PreencodedHtj2k97Component, PrequantizedHtj2k97Component,
    ReversibleDwt53FirstLevel,
};
use signinum_transcode::dct53_2d::Dwt53TwoDimensional;
use signinum_transcode::dct97_2d::Dwt97TwoDimensional;

/// Stable message returned when the CUDA runtime is unavailable (feature not
/// compiled, no device, or the transcode kernels were not built).
pub const CUDA_UNAVAILABLE: &str = "CUDA is unavailable on this host";

/// Default minimum component sample count before Auto mode offers a job to CUDA.
const DEFAULT_AUTO_MIN_SAMPLES: usize = 224 * 224;

/// Error returned by the CUDA transcode accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaTranscodeError {
    /// CUDA is unavailable on this host or the kernels were not built.
    CudaUnavailable,
    /// The request is outside the current CUDA implementation.
    UnsupportedJob(&'static str),
    /// CUDA runtime or kernel execution failed.
    Kernel(&'static str),
}

impl CudaTranscodeError {
    /// Convert into the static message required by the accelerator trait.
    #[must_use]
    pub const fn as_static_str(self) -> &'static str {
        match self {
            Self::CudaUnavailable => CUDA_UNAVAILABLE,
            Self::UnsupportedJob(reason) | Self::Kernel(reason) => reason,
        }
    }

    /// Whether Auto mode may recover from this error by using the scalar
    /// fallback (`Ok(None)`). Hard kernel failures propagate as `Err`.
    #[cfg(feature = "cuda-runtime")]
    const fn is_recoverable(self) -> bool {
        matches!(self, Self::CudaUnavailable | Self::UnsupportedJob(_))
    }
}

impl fmt::Display for CudaTranscodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_static_str())
    }
}

impl std::error::Error for CudaTranscodeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaDispatchMode {
    /// Treat an unavailable/unsupported CUDA dispatch as an error.
    Explicit,
    /// Fall back to the scalar oracle (`Ok(None)`) for small or unsupported
    /// jobs.
    Auto,
}

/// Optional CUDA accelerator for `signinum-transcode` transform stages.
#[derive(Debug, Clone)]
pub struct CudaDctToWaveletStageAccelerator {
    mode: CudaDispatchMode,
    min_auto_samples: usize,
    reversible_dwt53_attempts: usize,
    reversible_dwt53_dispatches: usize,
    reversible_dwt53_batch_attempts: usize,
    reversible_dwt53_batch_dispatches: usize,
    dwt53_attempts: usize,
    dwt53_dispatches: usize,
    dwt97_attempts: usize,
    dwt97_dispatches: usize,
    dwt97_batch_attempts: usize,
    dwt97_batch_dispatches: usize,
    htj2k97_codeblock_batch_attempts: usize,
    htj2k97_codeblock_batch_dispatches: usize,
    last_dwt97_batch_stage_timings: Option<Dwt97BatchStageTimings>,
    resident_ht_encode: bool,
}

impl CudaDctToWaveletStageAccelerator {
    /// Create an accelerator that treats unavailable/unsupported CUDA dispatch
    /// as an error (no silent scalar fallback).
    #[must_use]
    pub const fn new_explicit() -> Self {
        Self::with_mode(CudaDispatchMode::Explicit, 0)
    }

    /// Create an explicit accelerator that keeps 9/7 code-block coefficients
    /// resident and HT-encodes them on the same CUDA context before CPU
    /// packetization.
    #[must_use]
    pub const fn new_explicit_resident_ht_encode() -> Self {
        Self {
            resident_ht_encode: true,
            ..Self::with_mode(CudaDispatchMode::Explicit, 0)
        }
    }

    /// Create an accelerator that falls back to the scalar oracle for small or
    /// unsupported jobs.
    #[must_use]
    pub const fn for_auto() -> Self {
        Self::with_mode(CudaDispatchMode::Auto, DEFAULT_AUTO_MIN_SAMPLES)
    }

    const fn with_mode(mode: CudaDispatchMode, min_auto_samples: usize) -> Self {
        Self {
            mode,
            min_auto_samples,
            reversible_dwt53_attempts: 0,
            reversible_dwt53_dispatches: 0,
            reversible_dwt53_batch_attempts: 0,
            reversible_dwt53_batch_dispatches: 0,
            dwt53_attempts: 0,
            dwt53_dispatches: 0,
            dwt97_attempts: 0,
            dwt97_dispatches: 0,
            dwt97_batch_attempts: 0,
            dwt97_batch_dispatches: 0,
            htj2k97_codeblock_batch_attempts: 0,
            htj2k97_codeblock_batch_dispatches: 0,
            last_dwt97_batch_stage_timings: None,
            resident_ht_encode: false,
        }
    }

    /// Number of reversible 5/3 jobs offered to this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_attempts(&self) -> usize {
        self.reversible_dwt53_attempts
    }

    /// Number of reversible 5/3 jobs handled on the GPU.
    #[must_use]
    pub const fn reversible_dwt53_dispatches(&self) -> usize {
        self.reversible_dwt53_dispatches
    }

    /// Number of reversible 5/3 batches offered to this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_batch_attempts(&self) -> usize {
        self.reversible_dwt53_batch_attempts
    }

    /// Number of reversible 5/3 batches handled on the GPU.
    #[must_use]
    pub const fn reversible_dwt53_batch_dispatches(&self) -> usize {
        self.reversible_dwt53_batch_dispatches
    }

    /// Number of float 5/3 jobs offered to this accelerator.
    #[must_use]
    pub const fn dwt53_attempts(&self) -> usize {
        self.dwt53_attempts
    }

    /// Number of float 5/3 jobs handled on the GPU.
    #[must_use]
    pub const fn dwt53_dispatches(&self) -> usize {
        self.dwt53_dispatches
    }

    /// Number of 9/7 jobs offered to this accelerator.
    #[must_use]
    pub const fn dwt97_attempts(&self) -> usize {
        self.dwt97_attempts
    }

    /// Number of 9/7 jobs handled on the GPU.
    #[must_use]
    pub const fn dwt97_dispatches(&self) -> usize {
        self.dwt97_dispatches
    }

    /// Number of 9/7 batches offered to this accelerator.
    #[must_use]
    pub const fn dwt97_batch_attempts(&self) -> usize {
        self.dwt97_batch_attempts
    }

    /// Number of 9/7 batches handled on the GPU.
    #[must_use]
    pub const fn dwt97_batch_dispatches(&self) -> usize {
        self.dwt97_batch_dispatches
    }

    /// Number of prequantized 9/7 HTJ2K code-block batches offered.
    #[must_use]
    pub const fn htj2k97_codeblock_batch_attempts(&self) -> usize {
        self.htj2k97_codeblock_batch_attempts
    }

    /// Number of prequantized 9/7 HTJ2K code-block batches handled on the GPU.
    #[must_use]
    pub const fn htj2k97_codeblock_batch_dispatches(&self) -> usize {
        self.htj2k97_codeblock_batch_dispatches
    }

    /// Outcome for a job that CUDA cannot serve, resolved by dispatch mode.
    #[cfg(not(feature = "cuda-runtime"))]
    fn unavailable<T>(&self) -> Result<Option<T>, &'static str> {
        match self.mode {
            CudaDispatchMode::Explicit => Err(CUDA_UNAVAILABLE),
            CudaDispatchMode::Auto => Ok(None),
        }
    }

    /// Map a CUDA dispatch error to the trait outcome for the current mode:
    /// Auto recovers from recoverable errors with `Ok(None)`; Explicit and hard
    /// kernel failures propagate as `Err`.
    #[cfg(feature = "cuda-runtime")]
    fn recover<T>(&self, error: CudaTranscodeError) -> Result<Option<T>, &'static str> {
        if self.mode == CudaDispatchMode::Auto && error.is_recoverable() {
            Ok(None)
        } else {
            Err(error.as_static_str())
        }
    }
}

impl Default for CudaDctToWaveletStageAccelerator {
    fn default() -> Self {
        Self::for_auto()
    }
}

impl DctToWaveletStageAccelerator for CudaDctToWaveletStageAccelerator {
    fn supports_dwt97_batch(&self) -> bool {
        true
    }

    // The fused DCT->9/7->prequantized-codeblock path runs the staged 9/7
    // kernels followed by per-subband deadzone quantization into code-block-major
    // layout, mirroring the local Metal backend.
    fn supports_htj2k97_codeblock_batch(&self) -> bool {
        true
    }

    fn dct_grid_to_reversible_dwt53(
        &mut self,
        job: DctGridToReversibleDwt53Job<'_>,
    ) -> Result<Option<ReversibleDwt53FirstLevel>, &'static str> {
        self.reversible_dwt53_attempts = self.reversible_dwt53_attempts.saturating_add(1);

        if self.mode == CudaDispatchMode::Auto
            && job.width.saturating_mul(job.height) < self.min_auto_samples
        {
            return Ok(None);
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            let _ = job;
            self.unavailable()
        }

        #[cfg(feature = "cuda-runtime")]
        {
            match cuda::dispatch_reversible_dwt53(job) {
                Ok(output) => {
                    self.reversible_dwt53_dispatches =
                        self.reversible_dwt53_dispatches.saturating_add(1);
                    Ok(Some(output))
                }
                Err(error) => self.recover(error),
            }
        }
    }

    fn dct_grid_to_reversible_dwt53_batch(
        &mut self,
        jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, &'static str> {
        self.reversible_dwt53_batch_attempts =
            self.reversible_dwt53_batch_attempts.saturating_add(1);

        if jobs.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if self.mode == CudaDispatchMode::Auto
            && jobs
                .iter()
                .all(|job| job.width.saturating_mul(job.height) < self.min_auto_samples)
        {
            return Ok(None);
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            let _ = jobs;
            self.unavailable()
        }

        #[cfg(feature = "cuda-runtime")]
        {
            match cuda::dispatch_reversible_dwt53_batch(jobs) {
                Ok(output) => {
                    self.reversible_dwt53_batch_dispatches =
                        self.reversible_dwt53_batch_dispatches.saturating_add(1);
                    Ok(Some(output))
                }
                Err(error) => self.recover(error),
            }
        }
    }

    fn dct_grid_to_dwt53(
        &mut self,
        job: DctGridToDwt53Job<'_>,
    ) -> Result<Option<Dwt53TwoDimensional<f64>>, &'static str> {
        self.dwt53_attempts = self.dwt53_attempts.saturating_add(1);

        if self.mode == CudaDispatchMode::Auto
            && job.width.saturating_mul(job.height) < self.min_auto_samples
        {
            return Ok(None);
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            let _ = job;
            self.unavailable()
        }

        #[cfg(feature = "cuda-runtime")]
        {
            match cuda::dispatch_dwt53(job) {
                Ok(output) => {
                    self.dwt53_dispatches = self.dwt53_dispatches.saturating_add(1);
                    Ok(Some(output))
                }
                Err(error) => self.recover(error),
            }
        }
    }

    fn dct_grid_to_dwt97(
        &mut self,
        job: DctGridToDwt97Job<'_>,
    ) -> Result<Option<Dwt97TwoDimensional<f64>>, &'static str> {
        self.dwt97_attempts = self.dwt97_attempts.saturating_add(1);

        if self.mode == CudaDispatchMode::Auto
            && job.width.saturating_mul(job.height) < self.min_auto_samples
        {
            return Ok(None);
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            let _ = job;
            self.unavailable()
        }

        #[cfg(feature = "cuda-runtime")]
        {
            match cuda::dispatch_dwt97(job) {
                Ok(output) => {
                    self.dwt97_dispatches = self.dwt97_dispatches.saturating_add(1);
                    Ok(Some(output))
                }
                Err(error) => self.recover(error),
            }
        }
    }

    fn dct_grid_to_dwt97_batch(
        &mut self,
        jobs: &[DctGridToDwt97Job<'_>],
    ) -> Result<Option<Vec<Dwt97TwoDimensional<f64>>>, &'static str> {
        self.dwt97_batch_attempts = self.dwt97_batch_attempts.saturating_add(1);
        self.last_dwt97_batch_stage_timings = None;

        if jobs.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if self.mode == CudaDispatchMode::Auto
            && jobs
                .iter()
                .all(|job| job.width.saturating_mul(job.height) < self.min_auto_samples)
        {
            return Ok(None);
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            let _ = jobs;
            self.unavailable()
        }

        #[cfg(feature = "cuda-runtime")]
        {
            match cuda::dispatch_dwt97_batch(jobs) {
                Ok((output, timings)) => {
                    self.dwt97_batch_dispatches = self.dwt97_batch_dispatches.saturating_add(1);
                    self.last_dwt97_batch_stage_timings = Some(timings);
                    Ok(Some(output))
                }
                Err(error) => self.recover(error),
            }
        }
    }

    fn dct_grid_to_htj2k97_codeblock_batch(
        &mut self,
        jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
        options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<Vec<PrequantizedHtj2k97Component>>, &'static str> {
        // The code-block path is a staged 9/7 batch plus quantization, so it
        // counts as both a 9/7 batch and a code-block batch (matching Metal).
        self.dwt97_batch_attempts = self.dwt97_batch_attempts.saturating_add(1);
        self.htj2k97_codeblock_batch_attempts =
            self.htj2k97_codeblock_batch_attempts.saturating_add(1);
        self.last_dwt97_batch_stage_timings = None;

        if jobs.is_empty() {
            return Ok(Some(Vec::new()));
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            let _ = (jobs, options);
            self.unavailable()
        }

        #[cfg(feature = "cuda-runtime")]
        {
            match cuda::dispatch_htj2k97_codeblock_batch(jobs, options) {
                Ok((output, timings)) => {
                    self.dwt97_batch_dispatches = self.dwt97_batch_dispatches.saturating_add(1);
                    self.htj2k97_codeblock_batch_dispatches =
                        self.htj2k97_codeblock_batch_dispatches.saturating_add(1);
                    self.last_dwt97_batch_stage_timings = Some(timings);
                    Ok(Some(output))
                }
                Err(error) => self.recover(error),
            }
        }
    }

    fn dct_grid_to_htj2k97_preencoded_batch(
        &mut self,
        jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
        options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<Vec<PreencodedHtj2k97Component>>, &'static str> {
        if !self.resident_ht_encode {
            return Ok(None);
        }

        self.dwt97_batch_attempts = self.dwt97_batch_attempts.saturating_add(1);
        self.htj2k97_codeblock_batch_attempts =
            self.htj2k97_codeblock_batch_attempts.saturating_add(1);
        self.last_dwt97_batch_stage_timings = None;

        if jobs.is_empty() {
            return Ok(Some(Vec::new()));
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            let _ = (jobs, options);
            self.unavailable()
        }

        #[cfg(feature = "cuda-runtime")]
        {
            match cuda::dispatch_htj2k97_preencoded_batch(jobs, options) {
                Ok((output, timings)) => {
                    self.dwt97_batch_dispatches = self.dwt97_batch_dispatches.saturating_add(1);
                    self.htj2k97_codeblock_batch_dispatches =
                        self.htj2k97_codeblock_batch_dispatches.saturating_add(1);
                    self.last_dwt97_batch_stage_timings = Some(timings);
                    Ok(Some(output))
                }
                Err(error) => self.recover(error),
            }
        }
    }

    fn last_dwt97_batch_stage_timings(&self) -> Option<Dwt97BatchStageTimings> {
        self.last_dwt97_batch_stage_timings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_mode_without_cuda_runtime_errors_on_reversible_job() {
        // Without the cuda-runtime feature, Explicit mode must surface a typed
        // error rather than silently using the scalar fallback.
        let mut accelerator = CudaDctToWaveletStageAccelerator::new_explicit();
        let blocks: Vec<[i16; 64]> = vec![[0i16; 64]];
        let job = DctGridToReversibleDwt53Job {
            dequantized_blocks: &blocks,
            block_cols: 1,
            block_rows: 1,
            width: 8,
            height: 8,
        };
        let result = accelerator.dct_grid_to_reversible_dwt53(job);
        #[cfg(not(feature = "cuda-runtime"))]
        assert_eq!(result, Err(CUDA_UNAVAILABLE));
        let _ = result;
        assert_eq!(accelerator.reversible_dwt53_attempts(), 1);
    }

    #[test]
    fn auto_mode_falls_back_to_scalar_for_small_jobs() {
        // Auto mode returns Ok(None) for sub-threshold jobs so the transcode
        // pipeline uses its scalar oracle.
        let mut accelerator = CudaDctToWaveletStageAccelerator::for_auto();
        let blocks: Vec<[i16; 64]> = vec![[0i16; 64]];
        let job = DctGridToReversibleDwt53Job {
            dequantized_blocks: &blocks,
            block_cols: 1,
            block_rows: 1,
            width: 8,
            height: 8,
        };
        assert_eq!(accelerator.dct_grid_to_reversible_dwt53(job), Ok(None));
    }

    #[test]
    fn empty_batches_return_empty_without_dispatch() {
        let mut accelerator = CudaDctToWaveletStageAccelerator::new_explicit();
        assert_eq!(
            accelerator.dct_grid_to_reversible_dwt53_batch(&[]),
            Ok(Some(Vec::new()))
        );
        assert_eq!(
            accelerator.dct_grid_to_dwt97_batch(&[]),
            Ok(Some(Vec::new()))
        );
    }
}
