// SPDX-License-Identifier: MIT OR Apache-2.0

mod api;
mod enqueue;
mod plan;

use crate::{
    error::CudaError,
    execution::{CudaEvent, CudaExecutionStats},
    memory::CudaDeviceBuffer,
};

/// Completion owner for an asynchronously enqueued grayscale final-store
/// batch. Dropping unfinished work waits before releasing uploaded job data.
#[doc(hidden)]
#[derive(Debug)]
#[must_use = "queued J2K final store must be retained or finished"]
pub struct CudaQueuedJ2kStoreBatch {
    pub(super) completion: Option<CudaEvent>,
    pub(super) jobs: Option<CudaDeviceBuffer>,
    pub(super) execution: CudaExecutionStats,
}

impl CudaQueuedJ2kStoreBatch {
    /// Query final-store completion without waiting on the host.
    pub fn is_complete(&self) -> Result<bool, CudaError> {
        self.completion
            .as_ref()
            .map_or(Ok(true), CudaEvent::is_complete)
    }

    /// Wait for final store completion and release retained job metadata.
    pub fn finish(mut self) -> Result<CudaExecutionStats, CudaError> {
        if let Some(completion) = self.completion.take() {
            if let Err(error) = completion.synchronize() {
                if let Some(jobs) = self.jobs.take() {
                    // The driver may still reference uploaded metadata.
                    std::mem::forget(jobs);
                }
                return Err(error);
            }
        }
        self.jobs.take();
        Ok(self.execution)
    }

    /// Release final-store metadata after another ordered operation has
    /// already established completion of this context's default stream.
    ///
    /// # Safety
    ///
    /// The caller must prove that all default-stream work through this final
    /// store has completed, for example with a later synchronous status
    /// download on the same stream.
    #[doc(hidden)]
    pub unsafe fn release_after_stream_completion(mut self) -> CudaExecutionStats {
        self.completion.take();
        self.jobs.take();
        self.execution
    }

    /// Kernel dispatch counters for this queued final store.
    #[must_use]
    pub const fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

impl Drop for CudaQueuedJ2kStoreBatch {
    fn drop(&mut self) {
        let Some(completion) = self.completion.take() else {
            return;
        };
        if completion.synchronize().is_ok() {
            self.jobs.take();
        } else if let Some(jobs) = self.jobs.take() {
            // Completion is uncertain, so intentionally retain the allocation
            // rather than free metadata still reachable by the driver.
            std::mem::forget(jobs);
        }
    }
}
