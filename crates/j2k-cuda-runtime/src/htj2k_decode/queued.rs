// SPDX-License-Identifier: MIT OR Apache-2.0

mod drop_guard;
mod lifecycle;

use crate::{
    allocation::HostPhaseBudget,
    bytes::htj2k_statuses_as_bytes_mut,
    context::CudaContext,
    error::CudaError,
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
    pub(crate) status_offset: usize,
    pub(crate) uses_external_status_group: bool,
    pub(crate) kernel_name: &'static str,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) pool_reuse_guard: Option<CudaBufferPoolReuseGuard>,
    pub(crate) finish_host_live_bytes: usize,
}

impl CudaQueuedHtj2kCleanup {
    /// Number of statuses downloaded when this guard is finished directly.
    #[doc(hidden)]
    pub const fn status_count(&self) -> usize {
        self.status_count
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
        if self.uses_external_status_group && self.status_count != 0 {
            let error = CudaError::InvalidArgument {
                message: "group-status HTJ2K cleanup must be finished by its status group"
                    .to_string(),
            };
            return Err(self.synchronize_release_after_error(error));
        }
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
        self.context.record_status_device_to_host_copy(
            self.status_count
                .saturating_mul(core::mem::size_of::<CudaHtj2kStatus>()),
        );
        let status_error = first_status_error(&statuses, self.kernel_name);
        let release_result = self.release_after_stream_completion();
        select_status_release_result(self.execution, status_error, release_result)
    }
}
