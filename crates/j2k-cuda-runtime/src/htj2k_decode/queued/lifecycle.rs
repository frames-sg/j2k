// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{select_resource_release_error, select_uncertain_completion_error, CudaError};

use super::CudaQueuedHtj2kCleanup;

impl CudaQueuedHtj2kCleanup {
    pub(in crate::htj2k_decode) fn release_after_stream_completion(
        &mut self,
    ) -> Result<(), CudaError> {
        self.status_buffer.take();
        self.resources.clear();
        if let Some(guard) = self.pool_reuse_guard.take() {
            guard.release()?;
        }
        Ok(())
    }

    pub(in crate::htj2k_decode) fn synchronize_and_release(&mut self) -> Result<(), CudaError> {
        if self.pool_reuse_guard.is_none() {
            return self.release_after_stream_completion();
        }
        let outcome = self.context.synchronize_for_resource_release();
        if !outcome.completion_established() {
            return outcome.into_result();
        }

        self.release_after_stream_completion()
    }

    pub(super) fn abandon_resources(&mut self) {
        self.status_buffer.take();
        self.resources.clear();
        if let Some(guard) = self.pool_reuse_guard.take() {
            guard.abandon();
        }
    }

    pub(super) fn release_after_recoverable_operation_error(
        &mut self,
        primary_error: CudaError,
    ) -> CudaError {
        if self.context.inner.resource_lifetimes_poisoned() {
            self.abandon_resources();
            return primary_error;
        }
        match self.release_after_stream_completion() {
            Ok(()) => primary_error,
            Err(release_error) => select_resource_release_error(primary_error, release_error),
        }
    }

    pub(super) fn synchronize_release_after_error(
        &mut self,
        primary_error: CudaError,
    ) -> CudaError {
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
}
