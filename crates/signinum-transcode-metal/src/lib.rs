// SPDX-License-Identifier: Apache-2.0

//! Metal acceleration for coefficient-domain JPEG to HTJ2K transcode stages.
//!
//! The supported targets are direct DCT-grid to one-level 5/3 and 9/7 wavelet
//! projections used by `signinum-transcode`'s HTJ2K paths. CPU scalar code
//! remains the oracle and fallback.

#[cfg(target_os = "macos")]
mod metal;

#[doc(hidden)]
pub mod weights;

use core::fmt;

use signinum_transcode::accelerator::{
    idct_blocks_to_signed_samples_rayon, reversible_dwt53_first_level_from_block_samples,
    DctGridToDwt53Job, DctGridToDwt97Job, DctGridToReversibleDwt53Job,
    DctToWaveletStageAccelerator, ReversibleDwt53FirstLevel,
};
use signinum_transcode::dct53_2d::Dwt53TwoDimensional;
use signinum_transcode::dct97_2d::Dwt97TwoDimensional;

/// Stable message returned when Metal is unavailable.
pub const METAL_UNAVAILABLE: &str = "Metal is unavailable on this host";

const DEFAULT_AUTO_MIN_SAMPLES: usize = 224 * 224;
const DEFAULT_AUTO_REVERSIBLE_MIN_SAMPLES: usize = usize::MAX;
const DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_JOBS: usize = 32;
const DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_SAMPLES: usize = 224 * 224 * 32;

/// Error returned by the Metal transcode accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalTranscodeError {
    /// Metal is unavailable on this host or target.
    MetalUnavailable,
    /// The request is outside the current Metal implementation.
    UnsupportedJob(&'static str),
    /// Metal runtime or kernel execution failed.
    Kernel(&'static str),
}

impl MetalTranscodeError {
    /// Convert the error into the static message required by the accelerator
    /// trait.
    pub const fn as_static_str(self) -> &'static str {
        match self {
            Self::MetalUnavailable => METAL_UNAVAILABLE,
            Self::UnsupportedJob(reason) | Self::Kernel(reason) => reason,
        }
    }
}

impl fmt::Display for MetalTranscodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_static_str())
    }
}

impl std::error::Error for MetalTranscodeError {}

/// Optional Metal accelerator for `signinum-transcode` transform stages.
#[derive(Debug, Clone)]
pub struct MetalDctToWaveletStageAccelerator {
    mode: MetalDispatchMode,
    min_auto_samples: usize,
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
        }
    }

    /// Create an accelerator that falls back to scalar CPU for small or
    /// unsupported jobs.
    #[must_use]
    pub const fn for_auto() -> Self {
        Self {
            mode: MetalDispatchMode::Auto,
            min_auto_samples: DEFAULT_AUTO_MIN_SAMPLES,
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
        }
    }

    /// Override the minimum component sample count used before Auto mode
    /// dispatches non-reversible 5/3 and 9/7 projection jobs to Metal.
    #[must_use]
    pub const fn with_auto_min_samples(mut self, min_samples: usize) -> Self {
        self.min_auto_samples = min_samples;
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

    /// Number of 9/7 projection jobs offered to this accelerator.
    #[must_use]
    pub const fn dwt97_attempts(&self) -> usize {
        self.dwt97_attempts
    }

    /// Number of 9/7 projection jobs handled by Metal.
    #[must_use]
    pub const fn dwt97_dispatches(&self) -> usize {
        self.dwt97_dispatches
    }

    /// Dispatch a same-geometry batch of reversible integer 5/3 DCT-grid
    /// projection jobs. This is an experimental Metal-specific extension used
    /// for WSI tile-component queues; scalar/Rayon remains the portable oracle.
    pub fn dct_grid_to_reversible_dwt53_batch(
        &mut self,
        jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, &'static str> {
        self.dispatch_reversible_dwt53_batch(jobs)
    }

    fn dispatch_reversible_dwt53_batch(
        &mut self,
        jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, &'static str> {
        self.reversible_dwt53_batch_attempts =
            self.reversible_dwt53_batch_attempts.saturating_add(1);

        if jobs.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let total_samples = jobs.iter().fold(0usize, |total, job| {
            total.saturating_add(job.width.saturating_mul(job.height))
        });
        if self.mode == MetalDispatchMode::Auto
            && (jobs.len() < self.min_auto_reversible_batch_jobs
                || total_samples < self.min_auto_reversible_batch_samples)
        {
            return rayon_reversible_dwt53_batch(jobs).map(Some);
        }

        #[cfg(not(target_os = "macos"))]
        {
            match self.mode {
                MetalDispatchMode::Explicit => {
                    Err(MetalTranscodeError::MetalUnavailable.as_static_str())
                }
                MetalDispatchMode::Auto => rayon_reversible_dwt53_batch(jobs).map(Some),
            }
        }

        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_reversible_dwt53_batch(jobs) {
                Ok(output) => {
                    self.reversible_dwt53_batch_dispatches =
                        self.reversible_dwt53_batch_dispatches.saturating_add(1);
                    Ok(Some(output))
                }
                Err(
                    MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_),
                ) if self.mode == MetalDispatchMode::Auto => {
                    rayon_reversible_dwt53_batch(jobs).map(Some)
                }
                Err(error) => Err(error.as_static_str()),
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
    fn dct_grid_to_reversible_dwt53(
        &mut self,
        job: DctGridToReversibleDwt53Job<'_>,
    ) -> Result<Option<ReversibleDwt53FirstLevel>, &'static str> {
        self.reversible_dwt53_attempts = self.reversible_dwt53_attempts.saturating_add(1);

        if self.mode == MetalDispatchMode::Auto
            && job.width.saturating_mul(job.height) < self.min_auto_reversible_samples
        {
            return rayon_reversible_dwt53(job).map(Some);
        }

        #[cfg(not(target_os = "macos"))]
        {
            match self.mode {
                MetalDispatchMode::Explicit => {
                    Err(MetalTranscodeError::MetalUnavailable.as_static_str())
                }
                MetalDispatchMode::Auto => rayon_reversible_dwt53(job).map(Some),
            }
        }

        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_reversible_dwt53(job) {
                Ok(output) => {
                    self.reversible_dwt53_dispatches =
                        self.reversible_dwt53_dispatches.saturating_add(1);
                    Ok(Some(output))
                }
                Err(
                    MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_),
                ) if self.mode == MetalDispatchMode::Auto => rayon_reversible_dwt53(job).map(Some),
                Err(error) => Err(error.as_static_str()),
            }
        }
    }

    fn dct_grid_to_reversible_dwt53_batch(
        &mut self,
        jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, &'static str> {
        self.dispatch_reversible_dwt53_batch(jobs)
    }

    fn dct_grid_to_dwt53(
        &mut self,
        job: DctGridToDwt53Job<'_>,
    ) -> Result<Option<Dwt53TwoDimensional<f64>>, &'static str> {
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
                MetalDispatchMode::Explicit => {
                    Err(MetalTranscodeError::MetalUnavailable.as_static_str())
                }
                MetalDispatchMode::Auto => Ok(None),
            }
        }

        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_dwt53(job) {
                Ok(output) => {
                    self.dwt53_dispatches = self.dwt53_dispatches.saturating_add(1);
                    Ok(Some(output))
                }
                Err(
                    MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_),
                ) if self.mode == MetalDispatchMode::Auto => Ok(None),
                Err(error) => Err(error.as_static_str()),
            }
        }
    }

    fn dct_grid_to_dwt97(
        &mut self,
        job: DctGridToDwt97Job<'_>,
    ) -> Result<Option<Dwt97TwoDimensional<f64>>, &'static str> {
        self.dwt97_attempts = self.dwt97_attempts.saturating_add(1);

        if self.mode == MetalDispatchMode::Auto
            && job.width.saturating_mul(job.height) < self.min_auto_samples
        {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            match self.mode {
                MetalDispatchMode::Explicit => {
                    Err(MetalTranscodeError::MetalUnavailable.as_static_str())
                }
                MetalDispatchMode::Auto => Ok(None),
            }
        }

        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_dwt97(job) {
                Ok(output) => {
                    self.dwt97_dispatches = self.dwt97_dispatches.saturating_add(1);
                    Ok(Some(output))
                }
                Err(
                    MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_),
                ) if self.mode == MetalDispatchMode::Auto => Ok(None),
                Err(error) => Err(error.as_static_str()),
            }
        }
    }
}

fn rayon_reversible_dwt53(
    job: DctGridToReversibleDwt53Job<'_>,
) -> Result<ReversibleDwt53FirstLevel, &'static str> {
    let block_samples = idct_blocks_to_signed_samples_rayon(job.dequantized_blocks);
    reversible_dwt53_first_level_from_block_samples(
        &block_samples,
        job.block_cols,
        job.block_rows,
        job.width,
        job.height,
    )
}

fn rayon_reversible_dwt53_batch(
    jobs: &[DctGridToReversibleDwt53Job<'_>],
) -> Result<Vec<ReversibleDwt53FirstLevel>, &'static str> {
    jobs.iter()
        .map(|job| rayon_reversible_dwt53(*job))
        .collect()
}
