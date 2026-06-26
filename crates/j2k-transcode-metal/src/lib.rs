// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal acceleration for coefficient-domain JPEG to HTJ2K transcode stages.
//!
//! The supported targets are direct DCT-grid to one-level 5/3 and 9/7 wavelet
//! projections used by `j2k-transcode`'s HTJ2K paths. CPU scalar code
//! remains the oracle and fallback.

#[cfg(target_os = "macos")]
mod metal;

#[doc(hidden)]
pub mod weights;

#[cfg(target_os = "macos")]
pub use metal::MetalTranscodeSession;

use core::fmt;

use j2k_transcode::accelerator::{
    DctGridToDwt53Job, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob,
    DctGridToReversibleDwt53Job, DctToWaveletStageAccelerator, Dwt97BatchStageTimings,
    Htj2k97CodeBlockOptions, PrequantizedHtj2k97Component, ReversibleDwt53FirstLevel,
    TranscodeStageError,
};
use j2k_transcode::dct53_2d::Dwt53TwoDimensional;
use j2k_transcode::dct97_2d::Dwt97TwoDimensional;

/// Stable message returned when Metal is unavailable.
pub const METAL_UNAVAILABLE: &str = "Metal is unavailable on this host";

const DEFAULT_AUTO_MIN_SAMPLES: usize = 224 * 224;
const DEFAULT_AUTO_DWT97_MIN_SAMPLES: usize = usize::MAX;
const DEFAULT_AUTO_REVERSIBLE_MIN_SAMPLES: usize = usize::MAX;
const DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_JOBS: usize = 32;
const DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_SAMPLES: usize = 224 * 224 * 32;
const DEFAULT_AUTO_DWT97_BATCH_MIN_JOBS: usize = 32;
const DEFAULT_AUTO_DWT97_BATCH_MIN_SAMPLES: usize = 224 * 224 * 32;
const MAX_AUTO_DWT97_STAGED_BATCH_AXIS: usize = 1024;

/// Error returned by the Metal transcode accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalTranscodeError {
    /// Metal is unavailable on this host or target.
    MetalUnavailable,
    /// The request is outside the current Metal implementation.
    UnsupportedJob(&'static str),
    /// Metal runtime creation or device setup failed.
    Runtime(&'static str),
    /// Metal runtime or kernel execution failed.
    Kernel(&'static str),
}

impl fmt::Display for MetalTranscodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetalUnavailable => f.write_str(METAL_UNAVAILABLE),
            Self::UnsupportedJob(reason) | Self::Runtime(reason) | Self::Kernel(reason) => {
                f.write_str(reason)
            }
        }
    }
}

impl From<MetalTranscodeError> for TranscodeStageError {
    fn from(error: MetalTranscodeError) -> Self {
        match error {
            MetalTranscodeError::MetalUnavailable => Self::DeviceUnavailable,
            MetalTranscodeError::UnsupportedJob(reason) => Self::Unsupported(reason),
            MetalTranscodeError::Runtime(reason) | MetalTranscodeError::Kernel(reason) => {
                Self::Backend(reason.to_string())
            }
        }
    }
}

impl std::error::Error for MetalTranscodeError {}

#[cfg(not(target_os = "macos"))]
#[derive(Clone, Copy, Debug, Default)]
/// Placeholder Metal transcode session for hosts without Metal support.
pub struct MetalTranscodeSession {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
impl MetalTranscodeSession {
    /// Return `MetalUnavailable` on hosts without Metal support.
    pub const fn system_default() -> Result<Self, MetalTranscodeError> {
        Err(MetalTranscodeError::MetalUnavailable)
    }
}

/// Optional Metal accelerator for `j2k-transcode` transform stages.
#[derive(Debug, Clone)]
pub struct MetalDctToWaveletStageAccelerator {
    mode: MetalDispatchMode,
    min_auto_samples: usize,
    min_auto_dwt97_samples: usize,
    min_auto_reversible_samples: usize,
    min_auto_reversible_batch_jobs: usize,
    min_auto_reversible_batch_samples: usize,
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
    min_auto_dwt97_batch_jobs: usize,
    min_auto_dwt97_batch_samples: usize,
    #[cfg(target_os = "macos")]
    session: Option<MetalTranscodeSession>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalDispatchMode {
    Explicit,
    Auto,
}

impl MetalDctToWaveletStageAccelerator {
    /// Create an accelerator that treats unsupported Metal dispatch as an
    /// error.
    #[must_use]
    pub const fn new_explicit() -> Self {
        Self {
            mode: MetalDispatchMode::Explicit,
            min_auto_samples: 0,
            min_auto_dwt97_samples: 0,
            min_auto_reversible_samples: 0,
            min_auto_reversible_batch_jobs: 0,
            min_auto_reversible_batch_samples: 0,
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
            min_auto_dwt97_batch_jobs: 0,
            min_auto_dwt97_batch_samples: 0,
            #[cfg(target_os = "macos")]
            session: None,
        }
    }

    /// Create an accelerator that falls back to scalar CPU for small or
    /// unsupported jobs.
    #[must_use]
    pub const fn for_auto() -> Self {
        Self {
            mode: MetalDispatchMode::Auto,
            min_auto_samples: DEFAULT_AUTO_MIN_SAMPLES,
            min_auto_dwt97_samples: DEFAULT_AUTO_DWT97_MIN_SAMPLES,
            min_auto_reversible_samples: DEFAULT_AUTO_REVERSIBLE_MIN_SAMPLES,
            min_auto_reversible_batch_jobs: DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_JOBS,
            min_auto_reversible_batch_samples: DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_SAMPLES,
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
            min_auto_dwt97_batch_jobs: DEFAULT_AUTO_DWT97_BATCH_MIN_JOBS,
            min_auto_dwt97_batch_samples: DEFAULT_AUTO_DWT97_BATCH_MIN_SAMPLES,
            #[cfg(target_os = "macos")]
            session: None,
        }
    }

    /// Create an explicit-dispatch accelerator bound to a caller-owned Metal session.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn new_explicit_with_session(session: MetalTranscodeSession) -> Self {
        Self::new_explicit().with_session(session)
    }

    /// Create an Auto-mode accelerator bound to a caller-owned Metal session.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn for_auto_with_session(session: MetalTranscodeSession) -> Self {
        Self::for_auto().with_session(session)
    }

    /// Create an explicit-dispatch accelerator bound to an existing Metal device.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn new_explicit_with_device(device: ::metal::Device) -> Self {
        Self::new_explicit_with_session(MetalTranscodeSession::new(device))
    }

    /// Create an Auto-mode accelerator bound to an existing Metal device.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn for_auto_with_device(device: ::metal::Device) -> Self {
        Self::for_auto_with_session(MetalTranscodeSession::new(device))
    }

    /// Bind this accelerator to a caller-owned Metal session.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn with_session(mut self, session: MetalTranscodeSession) -> Self {
        self.session = Some(session);
        self
    }

    /// Bind this accelerator to an existing Metal device.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn with_device(self, device: ::metal::Device) -> Self {
        self.with_session(MetalTranscodeSession::new(device))
    }

    #[cfg(target_os = "macos")]
    fn metal_session(&mut self) -> &mut MetalTranscodeSession {
        self.session
            .get_or_insert_with(MetalTranscodeSession::default)
    }

    /// Override the minimum component sample count used before Auto mode
    /// dispatches non-reversible projection jobs to Metal.
    #[must_use]
    pub const fn with_auto_min_samples(mut self, min_samples: usize) -> Self {
        self.min_auto_samples = min_samples;
        self.min_auto_dwt97_samples = min_samples;
        self
    }

    /// Override the minimum component sample count used before Auto mode
    /// dispatches 9/7 transform jobs to Metal.
    #[must_use]
    pub const fn with_auto_dwt97_min_samples(mut self, min_samples: usize) -> Self {
        self.min_auto_dwt97_samples = min_samples;
        self
    }

    /// Override the 9/7 batch thresholds used before Auto mode dispatches a
    /// same-geometry batch to Metal.
    #[must_use]
    pub const fn with_auto_dwt97_batch_thresholds(
        mut self,
        min_jobs: usize,
        min_samples: usize,
    ) -> Self {
        self.min_auto_dwt97_batch_jobs = min_jobs;
        self.min_auto_dwt97_batch_samples = min_samples;
        self
    }

    /// Override the minimum component sample count used before Auto mode
    /// dispatches single reversible 5/3 jobs to Metal.
    #[must_use]
    pub const fn with_auto_reversible_min_samples(mut self, min_samples: usize) -> Self {
        self.min_auto_reversible_samples = min_samples;
        self
    }

    /// Override the reversible 5/3 batch thresholds used before Auto mode
    /// dispatches a same-geometry batch to Metal.
    #[must_use]
    pub const fn with_auto_reversible_batch_thresholds(
        mut self,
        min_jobs: usize,
        min_samples: usize,
    ) -> Self {
        self.min_auto_reversible_batch_jobs = min_jobs;
        self.min_auto_reversible_batch_samples = min_samples;
        self
    }

    /// Number of reversible integer 5/3 jobs offered to this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_attempts(&self) -> usize {
        self.reversible_dwt53_attempts
    }

    /// Number of reversible integer 5/3 jobs handled by Metal.
    #[must_use]
    pub const fn reversible_dwt53_dispatches(&self) -> usize {
        self.reversible_dwt53_dispatches
    }

    /// Number of reversible integer 5/3 batches offered to this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_batch_attempts(&self) -> usize {
        self.reversible_dwt53_batch_attempts
    }

    /// Number of reversible integer 5/3 batches handled by Metal.
    #[must_use]
    pub const fn reversible_dwt53_batch_dispatches(&self) -> usize {
        self.reversible_dwt53_batch_dispatches
    }

    /// Number of 5/3 projection jobs offered to this accelerator.
    #[must_use]
    pub const fn dwt53_attempts(&self) -> usize {
        self.dwt53_attempts
    }

    /// Number of 5/3 projection jobs handled by Metal.
    #[must_use]
    pub const fn dwt53_dispatches(&self) -> usize {
        self.dwt53_dispatches
    }

    /// Number of 9/7 transform jobs offered to this accelerator.
    #[must_use]
    pub const fn dwt97_attempts(&self) -> usize {
        self.dwt97_attempts
    }

    /// Number of 9/7 transform jobs handled by Metal.
    #[must_use]
    pub const fn dwt97_dispatches(&self) -> usize {
        self.dwt97_dispatches
    }

    /// Number of 9/7 transform batches offered to this accelerator.
    #[must_use]
    pub const fn dwt97_batch_attempts(&self) -> usize {
        self.dwt97_batch_attempts
    }

    /// Number of 9/7 transform batches handled by Metal.
    #[must_use]
    pub const fn dwt97_batch_dispatches(&self) -> usize {
        self.dwt97_batch_dispatches
    }

    /// Number of 9/7 code-block-ready batches offered to this accelerator.
    #[must_use]
    pub const fn htj2k97_codeblock_batch_attempts(&self) -> usize {
        self.htj2k97_codeblock_batch_attempts
    }

    /// Number of 9/7 code-block-ready batches handled by Metal.
    #[must_use]
    pub const fn htj2k97_codeblock_batch_dispatches(&self) -> usize {
        self.htj2k97_codeblock_batch_dispatches
    }

    /// Backend stage timings for the most recent 9/7 batch dispatch.
    #[must_use]
    pub const fn last_dwt97_batch_stage_timings(&self) -> Option<Dwt97BatchStageTimings> {
        self.last_dwt97_batch_stage_timings
    }

    /// Dispatch a same-geometry batch of reversible integer 5/3 DCT-grid
    /// projection jobs. This is an experimental Metal-specific extension used
    /// for WSI tile-component queues; scalar/Rayon remains the portable oracle.
    pub fn dct_grid_to_reversible_dwt53_batch(
        &mut self,
        jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, TranscodeStageError> {
        self.dispatch_reversible_dwt53_batch(jobs)
    }

    fn dispatch_reversible_dwt53_batch(
        &mut self,
        jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, TranscodeStageError> {
        self.reversible_dwt53_batch_attempts =
            self.reversible_dwt53_batch_attempts.saturating_add(1);

        if jobs.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let total_samples = jobs.iter().fold(0usize, |total, job| {
            total.saturating_add(job.width.saturating_mul(job.height))
        });
        // Auto declines with `Ok(None)` (small batches, unavailable Metal,
        // unsupported jobs) so the caller runs its scalar fallback — the same
        // contract as the float 5/3 path and the CUDA accelerator, instead of
        // silently computing a Rayon fallback inside the backend.
        if self.mode == MetalDispatchMode::Auto
            && (jobs.len() < self.min_auto_reversible_batch_jobs
                || total_samples < self.min_auto_reversible_batch_samples)
        {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            match self.mode {
                MetalDispatchMode::Explicit => Err(TranscodeStageError::DeviceUnavailable),
                MetalDispatchMode::Auto => Ok(None),
            }
        }

        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_reversible_dwt53_batch(self.metal_session(), jobs) {
                Ok(output) => {
                    self.reversible_dwt53_batch_dispatches =
                        self.reversible_dwt53_batch_dispatches.saturating_add(1);
                    Ok(Some(output))
                }
                Err(
                    MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_),
                ) if self.mode == MetalDispatchMode::Auto => Ok(None),
                Err(error) => Err(error.into()),
            }
        }
    }
}

impl Default for MetalDctToWaveletStageAccelerator {
    fn default() -> Self {
        Self::for_auto()
    }
}

impl DctToWaveletStageAccelerator for MetalDctToWaveletStageAccelerator {
    fn supports_dwt97_batch(&self) -> bool {
        true
    }

    fn supports_htj2k97_codeblock_batch(&self) -> bool {
        true
    }

    fn dct_grid_to_reversible_dwt53(
        &mut self,
        job: DctGridToReversibleDwt53Job<'_>,
    ) -> Result<Option<ReversibleDwt53FirstLevel>, TranscodeStageError> {
        self.reversible_dwt53_attempts = self.reversible_dwt53_attempts.saturating_add(1);

        // Auto declines with `Ok(None)`; see dispatch_reversible_dwt53_batch.
        if self.mode == MetalDispatchMode::Auto
            && job.width.saturating_mul(job.height) < self.min_auto_reversible_samples
        {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            match self.mode {
                MetalDispatchMode::Explicit => Err(TranscodeStageError::DeviceUnavailable),
                MetalDispatchMode::Auto => Ok(None),
            }
        }

        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_reversible_dwt53(self.metal_session(), job) {
                Ok(output) => {
                    self.reversible_dwt53_dispatches =
                        self.reversible_dwt53_dispatches.saturating_add(1);
                    Ok(Some(output))
                }
                Err(
                    MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_),
                ) if self.mode == MetalDispatchMode::Auto => Ok(None),
                Err(error) => Err(error.into()),
            }
        }
    }

    fn dct_grid_to_reversible_dwt53_batch(
        &mut self,
        jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, TranscodeStageError> {
        self.dispatch_reversible_dwt53_batch(jobs)
    }

    fn dct_grid_to_dwt53(
        &mut self,
        job: DctGridToDwt53Job<'_>,
    ) -> Result<Option<Dwt53TwoDimensional<f64>>, TranscodeStageError> {
        self.dwt53_attempts = self.dwt53_attempts.saturating_add(1);

        if self.mode == MetalDispatchMode::Auto
            && job.width.saturating_mul(job.height) < self.min_auto_samples
        {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            match self.mode {
                MetalDispatchMode::Explicit => Err(TranscodeStageError::DeviceUnavailable),
                MetalDispatchMode::Auto => Ok(None),
            }
        }

        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_dwt53(self.metal_session(), job) {
                Ok(output) => {
                    self.dwt53_dispatches = self.dwt53_dispatches.saturating_add(1);
                    Ok(Some(output))
                }
                Err(
                    MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_),
                ) if self.mode == MetalDispatchMode::Auto => Ok(None),
                Err(error) => Err(error.into()),
            }
        }
    }

    fn dct_grid_to_dwt97(
        &mut self,
        job: DctGridToDwt97Job<'_>,
    ) -> Result<Option<Dwt97TwoDimensional<f64>>, TranscodeStageError> {
        self.dwt97_attempts = self.dwt97_attempts.saturating_add(1);

        if self.mode == MetalDispatchMode::Auto
            && job.width.saturating_mul(job.height) < self.min_auto_dwt97_samples
        {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            match self.mode {
                MetalDispatchMode::Explicit => Err(TranscodeStageError::DeviceUnavailable),
                MetalDispatchMode::Auto => Ok(None),
            }
        }

        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_dwt97(self.metal_session(), job) {
                Ok(output) => {
                    self.dwt97_dispatches = self.dwt97_dispatches.saturating_add(1);
                    Ok(Some(output))
                }
                Err(
                    MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_),
                ) if self.mode == MetalDispatchMode::Auto => Ok(None),
                Err(error) => Err(error.into()),
            }
        }
    }

    fn dct_grid_to_dwt97_batch(
        &mut self,
        jobs: &[DctGridToDwt97Job<'_>],
    ) -> Result<Option<Vec<Dwt97TwoDimensional<f64>>>, TranscodeStageError> {
        self.dwt97_batch_attempts = self.dwt97_batch_attempts.saturating_add(1);
        self.last_dwt97_batch_stage_timings = None;

        if jobs.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let total_samples = jobs.iter().fold(0usize, |total, job| {
            total.saturating_add(job.width.saturating_mul(job.height))
        });
        if self.mode == MetalDispatchMode::Auto
            && (jobs.len() < self.min_auto_dwt97_batch_jobs
                || total_samples < self.min_auto_dwt97_batch_samples)
        {
            return Ok(None);
        }
        if self.mode == MetalDispatchMode::Auto
            && jobs.iter().any(|job| {
                job.width > MAX_AUTO_DWT97_STAGED_BATCH_AXIS
                    || job.height > MAX_AUTO_DWT97_STAGED_BATCH_AXIS
            })
        {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = jobs;
            match self.mode {
                MetalDispatchMode::Explicit => Err(TranscodeStageError::DeviceUnavailable),
                MetalDispatchMode::Auto => Ok(None),
            }
        }

        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_dwt97_batch(self.metal_session(), jobs) {
                Ok((output, timings)) => {
                    self.dwt97_batch_dispatches = self.dwt97_batch_dispatches.saturating_add(1);
                    self.last_dwt97_batch_stage_timings = Some(timings);
                    Ok(Some(output))
                }
                Err(
                    MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_),
                ) if self.mode == MetalDispatchMode::Auto => Ok(None),
                Err(error) => Err(error.into()),
            }
        }
    }

    fn dct_grid_to_htj2k97_codeblock_batch(
        &mut self,
        jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
        options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<Vec<PrequantizedHtj2k97Component>>, TranscodeStageError> {
        self.dwt97_batch_attempts = self.dwt97_batch_attempts.saturating_add(1);
        self.htj2k97_codeblock_batch_attempts =
            self.htj2k97_codeblock_batch_attempts.saturating_add(1);
        self.last_dwt97_batch_stage_timings = None;

        if jobs.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let total_samples = jobs.iter().fold(0usize, |total, job| {
            total.saturating_add(job.width.saturating_mul(job.height))
        });
        if self.mode == MetalDispatchMode::Auto
            && (jobs.len() < self.min_auto_dwt97_batch_jobs
                || total_samples < self.min_auto_dwt97_batch_samples)
        {
            return Ok(None);
        }
        if self.mode == MetalDispatchMode::Auto
            && jobs.iter().any(|job| {
                job.width > MAX_AUTO_DWT97_STAGED_BATCH_AXIS
                    || job.height > MAX_AUTO_DWT97_STAGED_BATCH_AXIS
            })
        {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (jobs, options);
            match self.mode {
                MetalDispatchMode::Explicit => Err(TranscodeStageError::DeviceUnavailable),
                MetalDispatchMode::Auto => Ok(None),
            }
        }

        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_htj2k97_codeblock_batch(
                self.metal_session(),
                jobs,
                options,
            ) {
                Ok((output, timings)) => {
                    self.dwt97_batch_dispatches = self.dwt97_batch_dispatches.saturating_add(1);
                    self.htj2k97_codeblock_batch_dispatches =
                        self.htj2k97_codeblock_batch_dispatches.saturating_add(1);
                    self.last_dwt97_batch_stage_timings = Some(timings);
                    Ok(Some(output))
                }
                Err(
                    MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_),
                ) if self.mode == MetalDispatchMode::Auto => Ok(None),
                Err(error) => Err(error.into()),
            }
        }
    }

    fn last_dwt97_batch_stage_timings(&self) -> Option<Dwt97BatchStageTimings> {
        self.last_dwt97_batch_stage_timings
    }
}
