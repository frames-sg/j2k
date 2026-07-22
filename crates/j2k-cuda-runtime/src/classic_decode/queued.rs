// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    bytes::classic_statuses_as_bytes_mut,
    context::CudaContext,
    error::{select_resource_release_error, select_uncertain_completion_error, CudaError},
    execution::CudaExecutionStats,
    memory::{CudaBufferPoolReuseGuard, CudaPooledDeviceBuffer},
};

use super::{CudaClassicDecodeStageTimings, CudaClassicStatus};

const CLASSIC_KERNEL_NAME: &str = "j2k_decode_classic_codeblocks_multi";

/// Enqueued classic Tier-1 work retained until one deferred status readback.
#[doc(hidden)]
#[derive(Debug)]
#[must_use = "queued classic decode must be finished or retained until Drop synchronizes it"]
pub struct CudaQueuedClassicDecode {
    pub(crate) context: CudaContext,
    pub(crate) resources: Vec<CudaPooledDeviceBuffer>,
    pub(crate) status_buffer: Option<CudaPooledDeviceBuffer>,
    pub(crate) status_count: usize,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) timings: CudaClassicDecodeStageTimings,
    pub(crate) pool_reuse_guard: Option<CudaBufferPoolReuseGuard>,
    pub(crate) finish_host_live_bytes: usize,
}

impl CudaQueuedClassicDecode {
    /// Number of descriptor statuses downloaded by completion.
    #[doc(hidden)]
    pub const fn status_count(&self) -> usize {
        self.status_count
    }

    /// CUDA execution counters for the enqueued classic work.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Finish the ordered graph with one status transfer and validate every job.
    pub fn finish(
        mut self,
    ) -> Result<(CudaExecutionStats, CudaClassicDecodeStageTimings), CudaError> {
        if self.status_count == 0 {
            self.release_after_stream_completion()?;
            return Ok((self.execution, self.timings));
        }
        let statuses_result = HostPhaseBudget::with_live_bytes(
            "CUDA queued classic Tier-1 status readback",
            self.finish_host_live_bytes,
        )
        .and_then(|mut budget| {
            budget.try_vec_filled(self.status_count, CudaClassicStatus::default())
        });
        let mut statuses = match statuses_result {
            Ok(statuses) => statuses,
            Err(error) => return Err(self.synchronize_release_after_error(error)),
        };
        let copy_result = self.status_buffer.as_ref().map_or_else(
            || {
                Err(CudaError::StatePoisoned {
                    message: "queued classic status buffer disappeared before readback".to_string(),
                })
            },
            |buffer| buffer.copy_to_host(classic_statuses_as_bytes_mut(&mut statuses)),
        );
        if let Err(error) = copy_result {
            return Err(self.release_after_recoverable_operation_error(error));
        }
        self.context.record_status_device_to_host_copy(
            self.status_count
                .saturating_mul(core::mem::size_of::<CudaClassicStatus>()),
        );
        let status_error = statuses
            .iter()
            .copied()
            .enumerate()
            .find(|(_, status)| status.code != 0)
            .map(|(job_index, status)| CudaError::KernelJobStatus {
                kernel: CLASSIC_KERNEL_NAME,
                job_index,
                code: status.code,
                detail: status.detail,
            });
        let release_result = self.release_after_stream_completion();
        match (status_error, release_result) {
            (Some(primary), Err(release)) => Err(select_resource_release_error(primary, release)),
            (Some(error), Ok(())) | (None, Err(error)) => Err(error),
            (None, Ok(())) => Ok((self.execution, self.timings)),
        }
    }

    fn release_after_stream_completion(&mut self) -> Result<(), CudaError> {
        self.status_buffer.take();
        self.resources.clear();
        if let Some(guard) = self.pool_reuse_guard.take() {
            guard.release()?;
        }
        Ok(())
    }

    fn abandon_resources(&mut self) {
        self.status_buffer.take();
        self.resources.clear();
        if let Some(guard) = self.pool_reuse_guard.take() {
            guard.abandon();
        }
    }

    fn release_after_recoverable_operation_error(&mut self, primary: CudaError) -> CudaError {
        if self.context.inner.resource_lifetimes_poisoned() {
            self.abandon_resources();
            return primary;
        }
        match self.release_after_stream_completion() {
            Ok(()) => primary,
            Err(release) => select_resource_release_error(primary, release),
        }
    }

    fn synchronize_release_after_error(&mut self, primary: CudaError) -> CudaError {
        let outcome = self.context.synchronize_for_resource_release();
        if let Err(completion) = outcome.into_result() {
            self.abandon_resources();
            return select_uncertain_completion_error(primary, Some(completion));
        }
        match self.release_after_stream_completion() {
            Ok(()) => primary,
            Err(release) => select_resource_release_error(primary, release),
        }
    }
}

impl Drop for CudaQueuedClassicDecode {
    fn drop(&mut self) {
        if self.pool_reuse_guard.is_some() {
            let outcome = self.context.synchronize_for_resource_release();
            if outcome.completion_established() {
                let _ = self.release_after_stream_completion();
            } else {
                self.abandon_resources();
            }
        }
    }
}
