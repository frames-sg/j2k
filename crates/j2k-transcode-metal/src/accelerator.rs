// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_transcode::{
    DctGridToReversibleDwt53Job, DctToWaveletStageCounterEvent as CounterEvent,
    DctToWaveletStageCounters, Dwt97BatchStageTimings, ReversibleDwt53FirstLevel,
    TranscodeStageDispatchMode, TranscodeStageError,
};

#[cfg(target_os = "macos")]
use crate::metal;
#[cfg(target_os = "macos")]
use crate::MetalTranscodeError;
#[cfg(target_os = "macos")]
use crate::MetalTranscodeSession;

mod dispatch;

const DEFAULT_AUTO_MIN_SAMPLES: usize = 224 * 224;
// Metal single-job Auto dispatch is disabled for the transcode paths whose
// current evidence is batch-shaped. Callers can opt in per stage with the
// public threshold setters when they have host-local evidence.
const DEFAULT_AUTO_DWT97_MIN_SAMPLES: usize = usize::MAX;
const DEFAULT_AUTO_REVERSIBLE_MIN_SAMPLES: usize = usize::MAX;
const DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_JOBS: usize = 32;
const DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_SAMPLES: usize = 224 * 224 * 32;
const DEFAULT_AUTO_DWT97_BATCH_MIN_JOBS: usize = 32;
const DEFAULT_AUTO_DWT97_BATCH_MIN_SAMPLES: usize = 224 * 224 * 32;
// Auto avoids the staged 9/7 Metal path for very large tile axes by default;
// strict Metal and caller-lowered thresholds remain explicit policy decisions.
const MAX_AUTO_DWT97_STAGED_BATCH_AXIS: usize = 1024;

/// Optional Metal accelerator for `j2k-transcode` transform stages.
#[derive(Debug, Clone)]
pub struct MetalDctToWaveletStageAccelerator {
    mode: TranscodeStageDispatchMode,
    min_auto_samples: usize,
    min_auto_dwt97_samples: usize,
    min_auto_reversible_samples: usize,
    min_auto_reversible_batch_jobs: usize,
    min_auto_reversible_batch_samples: usize,
    counters: DctToWaveletStageCounters,
    last_dwt97_batch_stage_timings: Option<Dwt97BatchStageTimings>,
    min_auto_dwt97_batch_jobs: usize,
    min_auto_dwt97_batch_samples: usize,
    #[cfg(target_os = "macos")]
    session: Option<MetalTranscodeSession>,
}

impl MetalDctToWaveletStageAccelerator {
    /// Create an accelerator that treats unsupported Metal dispatch as an error.
    #[must_use]
    pub const fn new_explicit() -> Self {
        Self {
            mode: TranscodeStageDispatchMode::Explicit,
            min_auto_samples: 0,
            min_auto_dwt97_samples: 0,
            min_auto_reversible_samples: 0,
            min_auto_reversible_batch_jobs: 0,
            min_auto_reversible_batch_samples: 0,
            counters: DctToWaveletStageCounters::new(),
            last_dwt97_batch_stage_timings: None,
            min_auto_dwt97_batch_jobs: 0,
            min_auto_dwt97_batch_samples: 0,
            #[cfg(target_os = "macos")]
            session: None,
        }
    }

    /// Create an accelerator that falls back to scalar CPU for small or unsupported jobs.
    #[must_use]
    pub const fn for_auto() -> Self {
        Self {
            mode: TranscodeStageDispatchMode::Auto,
            min_auto_samples: DEFAULT_AUTO_MIN_SAMPLES,
            min_auto_dwt97_samples: DEFAULT_AUTO_DWT97_MIN_SAMPLES,
            min_auto_reversible_samples: DEFAULT_AUTO_REVERSIBLE_MIN_SAMPLES,
            min_auto_reversible_batch_jobs: DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_JOBS,
            min_auto_reversible_batch_samples: DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_SAMPLES,
            counters: DctToWaveletStageCounters::new(),
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

    /// Override the minimum component sample count used before Auto mode dispatches non-reversible projection jobs to Metal.
    #[must_use]
    pub const fn with_auto_min_samples(mut self, min_samples: usize) -> Self {
        self.min_auto_samples = min_samples;
        self.min_auto_dwt97_samples = min_samples;
        self
    }

    /// Override the minimum component sample count used before Auto mode dispatches 9/7 transform jobs to Metal.
    #[must_use]
    pub const fn with_auto_dwt97_min_samples(mut self, min_samples: usize) -> Self {
        self.min_auto_dwt97_samples = min_samples;
        self
    }

    /// Override the 9/7 batch thresholds used before Auto mode dispatches a same-geometry batch to Metal.
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

    /// Override the minimum component sample count used before Auto mode dispatches single reversible 5/3 jobs to Metal.
    #[must_use]
    pub const fn with_auto_reversible_min_samples(mut self, min_samples: usize) -> Self {
        self.min_auto_reversible_samples = min_samples;
        self
    }

    /// Override the reversible 5/3 batch thresholds used before Auto mode dispatches a same-geometry batch to Metal.
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
        self.counters.reversible_dwt53_attempts()
    }

    /// Number of reversible integer 5/3 jobs handled by Metal.
    #[must_use]
    pub const fn reversible_dwt53_dispatches(&self) -> usize {
        self.counters.reversible_dwt53_dispatches()
    }

    /// Number of reversible integer 5/3 batches offered to this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_batch_attempts(&self) -> usize {
        self.counters.reversible_dwt53_batch_attempts()
    }

    /// Number of reversible integer 5/3 batches handled by Metal.
    #[must_use]
    pub const fn reversible_dwt53_batch_dispatches(&self) -> usize {
        self.counters.reversible_dwt53_batch_dispatches()
    }

    /// Number of 5/3 projection jobs offered to this accelerator.
    #[must_use]
    pub const fn dwt53_attempts(&self) -> usize {
        self.counters.dwt53_attempts()
    }

    /// Number of 5/3 projection jobs handled by Metal.
    #[must_use]
    pub const fn dwt53_dispatches(&self) -> usize {
        self.counters.dwt53_dispatches()
    }

    /// Number of 9/7 transform jobs offered to this accelerator.
    #[must_use]
    pub const fn dwt97_attempts(&self) -> usize {
        self.counters.dwt97_attempts()
    }

    /// Number of 9/7 transform jobs handled by Metal.
    #[must_use]
    pub const fn dwt97_dispatches(&self) -> usize {
        self.counters.dwt97_dispatches()
    }

    /// Number of 9/7 transform batches offered to this accelerator.
    #[must_use]
    pub const fn dwt97_batch_attempts(&self) -> usize {
        self.counters.dwt97_batch_attempts()
    }

    /// Number of 9/7 transform batches handled by Metal.
    #[must_use]
    pub const fn dwt97_batch_dispatches(&self) -> usize {
        self.counters.dwt97_batch_dispatches()
    }

    /// Number of 9/7 code-block-ready batches offered to this accelerator.
    #[must_use]
    pub const fn htj2k97_codeblock_batch_attempts(&self) -> usize {
        self.counters.htj2k97_codeblock_batch_attempts()
    }

    /// Number of 9/7 code-block-ready batches handled by Metal.
    #[must_use]
    pub const fn htj2k97_codeblock_batch_dispatches(&self) -> usize {
        self.counters.htj2k97_codeblock_batch_dispatches()
    }

    /// Backend stage timings for the most recent 9/7 batch dispatch.
    #[must_use]
    pub const fn last_dwt97_batch_stage_timings(&self) -> Option<Dwt97BatchStageTimings> {
        self.last_dwt97_batch_stage_timings
    }

    #[cfg(target_os = "macos")]
    fn recover<T>(&self, error: MetalTranscodeError) -> Result<Option<T>, TranscodeStageError> {
        self.mode
            .recover(error, MetalTranscodeError::is_recoverable)
    }

    /// Dispatch a same-geometry batch of reversible integer 5/3 DCT-grid projection jobs.
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
        self.counters
            .record(CounterEvent::ReversibleDwt53BatchAttempt, 1);
        if jobs.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let total_samples = jobs.iter().fold(0usize, |total, job| {
            total.saturating_add(job.width.saturating_mul(job.height))
        });
        if self.mode.is_auto()
            && (jobs.len() < self.min_auto_reversible_batch_jobs
                || total_samples < self.min_auto_reversible_batch_samples)
        {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            self.mode.unavailable()
        }
        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_reversible_dwt53_batch(self.metal_session(), jobs) {
                Ok(output) => {
                    self.counters
                        .record(CounterEvent::ReversibleDwt53BatchDispatch, 1);
                    Ok(Some(output))
                }
                Err(error) => self.recover(error),
            }
        }
    }
}

impl Default for MetalDctToWaveletStageAccelerator {
    fn default() -> Self {
        Self::for_auto()
    }
}
