// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_transcode::{
    DctGridToDwt53Job, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob,
    DctGridToReversibleDwt53Job, DctToWaveletStageAccelerator, Dwt53TwoDimensional,
    Dwt97BatchStageTimings, Dwt97TwoDimensional, Htj2k97CodeBlockOptions,
    PrequantizedHtj2k97Component, ReversibleDwt53FirstLevel, TranscodeStageError,
};

use super::{CounterEvent, MetalDctToWaveletStageAccelerator, MAX_AUTO_DWT97_STAGED_BATCH_AXIS};
#[cfg(target_os = "macos")]
use crate::metal;

#[doc(hidden)]
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
        self.counters
            .record(CounterEvent::ReversibleDwt53Attempt, 1);
        if self.mode.is_auto()
            && job.width.saturating_mul(job.height) < self.min_auto_reversible_samples
        {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            self.mode.unavailable()
        }
        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_reversible_dwt53(self.metal_session(), job) {
                Ok(output) => {
                    self.counters
                        .record(CounterEvent::ReversibleDwt53Dispatch, 1);
                    Ok(Some(output))
                }
                Err(error) => self.recover(error),
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
        self.counters.record(CounterEvent::Dwt53Attempt, 1);
        if self.mode.is_auto() && job.width.saturating_mul(job.height) < self.min_auto_samples {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            self.mode.unavailable()
        }
        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_dwt53(self.metal_session(), job) {
                Ok(output) => {
                    self.counters.record(CounterEvent::Dwt53Dispatch, 1);
                    Ok(Some(output))
                }
                Err(error) => self.recover(error),
            }
        }
    }

    fn dct_grid_to_dwt97(
        &mut self,
        job: DctGridToDwt97Job<'_>,
    ) -> Result<Option<Dwt97TwoDimensional<f64>>, TranscodeStageError> {
        self.counters.record(CounterEvent::Dwt97Attempt, 1);
        if self.mode.is_auto() && job.width.saturating_mul(job.height) < self.min_auto_dwt97_samples
        {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            self.mode.unavailable()
        }
        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_dwt97(self.metal_session(), job) {
                Ok(output) => {
                    self.counters.record(CounterEvent::Dwt97Dispatch, 1);
                    Ok(Some(output))
                }
                Err(error) => self.recover(error),
            }
        }
    }

    fn dct_grid_to_dwt97_batch(
        &mut self,
        jobs: &[DctGridToDwt97Job<'_>],
    ) -> Result<Option<Vec<Dwt97TwoDimensional<f64>>>, TranscodeStageError> {
        self.counters.record(CounterEvent::Dwt97BatchAttempt, 1);
        self.last_dwt97_batch_stage_timings = None;
        if jobs.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if self
            .auto_declines_dwt97_batch(jobs.iter().map(|job| (job.width, job.height)), jobs.len())
        {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = jobs;
            self.mode.unavailable()
        }
        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_dwt97_batch(self.metal_session(), jobs) {
                Ok((output, timings)) => {
                    self.counters.record(CounterEvent::Dwt97BatchDispatch, 1);
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
    ) -> Result<Option<Vec<PrequantizedHtj2k97Component>>, TranscodeStageError> {
        self.counters.record(CounterEvent::Dwt97BatchAttempt, 1);
        self.counters
            .record(CounterEvent::Htj2k97CodeblockBatchAttempt, 1);
        self.last_dwt97_batch_stage_timings = None;
        if jobs.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if self
            .auto_declines_dwt97_batch(jobs.iter().map(|job| (job.width, job.height)), jobs.len())
        {
            return Ok(None);
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (jobs, options);
            self.mode.unavailable()
        }
        #[cfg(target_os = "macos")]
        {
            match metal::dispatch_dct_grid_to_htj2k97_codeblock_batch(
                self.metal_session(),
                jobs,
                options,
            ) {
                Ok((output, timings)) => {
                    self.counters.record(CounterEvent::Dwt97BatchDispatch, 1);
                    self.counters
                        .record(CounterEvent::Htj2k97CodeblockBatchDispatch, 1);
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

impl MetalDctToWaveletStageAccelerator {
    fn auto_declines_dwt97_batch(
        &self,
        geometry: impl Iterator<Item = (usize, usize)>,
        job_count: usize,
    ) -> bool {
        if !self.mode.is_auto() {
            return false;
        }
        let mut total_samples = 0usize;
        let mut oversized_axis = false;
        for (width, height) in geometry {
            total_samples = total_samples.saturating_add(width.saturating_mul(height));
            oversized_axis |= width > MAX_AUTO_DWT97_STAGED_BATCH_AXIS
                || height > MAX_AUTO_DWT97_STAGED_BATCH_AXIS;
        }
        job_count < self.min_auto_dwt97_batch_jobs
            || total_samples < self.min_auto_dwt97_batch_samples
            || oversized_axis
    }
}
