// SPDX-License-Identifier: MIT OR Apache-2.0

mod drop_guard;

use crate::{
    allocation::HostPhaseBudget,
    bytes::htj2k_statuses_as_bytes_mut,
    context::CudaContext,
    error::{select_resource_release_error, select_uncertain_completion_error, CudaError},
    execution::CudaExecutionStats,
    memory::{CudaBufferPoolReuseGuard, CudaPooledDeviceBuffer},
};

use super::{
    status::{first_status_error, select_status_release_result},
    CudaHtj2kStatus,
};

/// Enqueued HTJ2K cleanup work plus pooled resources/statuses that must stay
/// live until `finish` validates kernel completion.
#[doc(hidden)]
#[derive(Debug)]
#[must_use = "queued HTJ2K cleanup must be finished or retained until Drop synchronizes it"]
pub struct CudaQueuedHtj2kCleanup {
    pub(crate) context: CudaContext,
    pub(crate) resources: Vec<CudaPooledDeviceBuffer>,
    pub(crate) status_buffer: Option<CudaPooledDeviceBuffer>,
    pub(crate) status_count: usize,
    pub(crate) kernel_name: &'static str,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) pool_reuse_guard: Option<CudaBufferPoolReuseGuard>,
    pub(crate) finish_host_live_bytes: usize,
}

impl CudaQueuedHtj2kCleanup {
    fn release_after_stream_completion(&mut self) -> Result<(), CudaError> {
        self.status_buffer.take();
        self.resources.clear();
        if let Some(guard) = self.pool_reuse_guard.take() {
            guard.release()?;
        }
        Ok(())
    }

    fn synchronize_and_release(&mut self) -> Result<(), CudaError> {
        if self.pool_reuse_guard.is_none() {
            return self.release_after_stream_completion();
        }
        let outcome = self.context.synchronize_for_resource_release();
        if !outcome.completion_established() {
            return outcome.into_result();
        }

        self.release_after_stream_completion()
    }

    fn abandon_resources(&mut self) {
        self.status_buffer.take();
        self.resources.clear();
        if let Some(guard) = self.pool_reuse_guard.take() {
            guard.abandon();
        }
    }

    fn release_after_recoverable_operation_error(&mut self, primary_error: CudaError) -> CudaError {
        if self.context.inner.resource_lifetimes_poisoned() {
            self.abandon_resources();
            return primary_error;
        }
        match self.release_after_stream_completion() {
            Ok(()) => primary_error,
            Err(release_error) => select_resource_release_error(primary_error, release_error),
        }
    }

    fn synchronize_release_after_error(&mut self, primary_error: CudaError) -> CudaError {
        let outcome = self.context.synchronize_for_resource_release();
        if let Err(completion_error) = outcome.into_result() {
            self.abandon_resources();
            return select_uncertain_completion_error(primary_error, Some(completion_error));
        }
        match self.release_after_stream_completion() {
            Ok(()) => primary_error,
            Err(release_error) => select_resource_release_error(primary_error, release_error),
        }
    }

    /// CUDA execution counters for the enqueued cleanup work.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Number of pooled resource buffers held live for the queued cleanup work.
    pub fn resource_count(&self) -> usize {
        self.resources.len() + usize::from(self.status_buffer.is_some())
    }

    /// Synchronize through status download and validate kernel statuses.
    pub fn finish(mut self) -> Result<CudaExecutionStats, CudaError> {
        if self.status_buffer.is_none() {
            if self.pool_reuse_guard.is_some() {
                self.synchronize_and_release()?;
            } else {
                self.release_after_stream_completion()?;
            }
            return Ok(self.execution);
        }

        let statuses_result = HostPhaseBudget::with_live_bytes(
            "CUDA queued HTJ2K cleanup status readback",
            self.finish_host_live_bytes,
        )
        .and_then(|mut budget| {
            budget.try_vec_filled(self.status_count, CudaHtj2kStatus::default())
        });
        let mut statuses = match statuses_result {
            Ok(statuses) => statuses,
            Err(primary_error) => {
                return Err(self.synchronize_release_after_error(primary_error));
            }
        };
        let copy_result = self.status_buffer.as_ref().map_or_else(
            || {
                Err(CudaError::StatePoisoned {
                    message: "queued HTJ2K status buffer disappeared before readback".to_string(),
                })
            },
            |status_buffer| status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses)),
        );
        if let Err(error) = copy_result {
            return Err(self.release_after_recoverable_operation_error(error));
        }
        let status_error = first_status_error(&statuses, self.kernel_name);
        let release_result = self.release_after_stream_completion();
        select_status_release_result(self.execution, status_error, release_result)
    }
}
