// SPDX-License-Identifier: Apache-2.0

//! Metal acceleration for coefficient-domain JPEG to HTJ2K transcode stages.
//!
//! The first supported target is the direct DCT-grid to one-level 9/7 wavelet
//! projection used by `signinum-transcode`'s lossy HTJ2K path. CPU scalar code
//! remains the oracle and fallback.

#[doc(hidden)]
pub mod weights;

use core::fmt;

use signinum_transcode::accelerator::{DctGridToDwt97Job, DctToWaveletStageAccelerator};
use signinum_transcode::dct97_2d::Dwt97TwoDimensional;

/// Stable message returned when Metal is unavailable.
pub const METAL_UNAVAILABLE: &str = "Metal is unavailable on this host";

const METAL_DCT97_NOT_IMPLEMENTED: &str = "Metal DCT 9/7 projection is not implemented";
const DEFAULT_AUTO_MIN_SAMPLES: usize = 65_536;

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
            dwt97_attempts: 0,
            dwt97_dispatches: 0,
        }
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
}

impl Default for MetalDctToWaveletStageAccelerator {
    fn default() -> Self {
        Self::for_auto()
    }
}

impl DctToWaveletStageAccelerator for MetalDctToWaveletStageAccelerator {
    fn dct_grid_to_dwt97(
        &mut self,
        job: DctGridToDwt97Job<'_>,
    ) -> Result<Option<Dwt97TwoDimensional<f64>>, &'static str> {
        self.dwt97_attempts = self.dwt97_attempts.saturating_add(1);

        if self.mode == MetalDispatchMode::Auto && job.width * job.height < self.min_auto_samples {
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
            let _ = job;
            match self.mode {
                MetalDispatchMode::Explicit => {
                    Err(MetalTranscodeError::Kernel(METAL_DCT97_NOT_IMPLEMENTED).as_static_str())
                }
                MetalDispatchMode::Auto => Ok(None),
            }
        }
    }
}
