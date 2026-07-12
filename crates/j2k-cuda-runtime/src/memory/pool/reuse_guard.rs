// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CudaBufferPoolInner;
use crate::{
    error::{select_resource_release_error, CudaError},
    execution::completion::select_uncertain_completion_error,
};
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct CudaBufferPoolReuseGuard {
    pub(super) pool: Arc<CudaBufferPoolInner>,
    pub(super) active: bool,
}

impl CudaBufferPoolReuseGuard {
    pub(crate) fn release(mut self) -> Result<(), CudaError> {
        if let Err(error) = self.release_inner() {
            if self.active {
                // A poisoned pool cannot safely transition deferred
                // allocations back to reusable state. Retain the pool and its
                // active hold.
                std::mem::forget(self);
            }
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn synchronize_pool_context(&self) -> crate::execution::CudaSynchronizationOutcome {
        self.pool.context.synchronize_for_resource_release()
    }

    pub(crate) fn synchronize_and_release(self) -> Result<(), CudaError> {
        let outcome = self.synchronize_pool_context();
        if !outcome.completion_established() {
            let result = outcome.into_result();
            self.abandon();
            return result;
        }

        self.release()
    }

    pub(crate) fn synchronize_then_error<T>(self, error: CudaError) -> Result<T, CudaError> {
        if self.pool.context.inner.resource_lifetimes_poisoned() {
            self.abandon();
            return Err(select_uncertain_completion_error(error, None));
        }
        if let Err(completion_error) = self.synchronize_pool_context().into_result() {
            self.abandon();
            return Err(select_uncertain_completion_error(
                error,
                Some(completion_error),
            ));
        }
        match self.release() {
            Ok(()) => Err(error),
            Err(release_error) => Err(select_resource_release_error(error, release_error)),
        }
    }

    pub(crate) fn release_after_recoverable_operation_error<T>(
        self,
        primary_error: CudaError,
    ) -> Result<T, CudaError> {
        // A failed recoverable driver operation already attempted context-wide
        // completion while holding the lifecycle gate. Do not synchronize a
        // second time after another thread could submit later work.
        if self.pool.context.inner.resource_lifetimes_poisoned() {
            self.abandon();
            return Err(primary_error);
        }
        match self.release() {
            Ok(()) => Err(primary_error),
            Err(release_error) => Err(select_resource_release_error(primary_error, release_error)),
        }
    }

    pub(crate) fn abandon(self) {
        // Completion could not be established. Leaking the guard keeps the
        // pool and its reuse hold alive, so deferred allocations cannot be
        // recycled or freed while CUDA might still reference them.
        std::mem::forget(self);
    }

    fn release_inner(&mut self) -> Result<(), CudaError> {
        if !self.active {
            return Ok(());
        }
        match self.pool.release_reuse_hold() {
            Ok(()) => {
                self.active = false;
                Ok(())
            }
            Err(error @ CudaError::HostAllocationFailed { .. }) => {
                // The hold reaches zero before deferred buffers enter the
                // completed-work cache. Cache-allocation failure may drop
                // those buffers safely, so this guard must not leak the pool.
                self.active = false;
                Err(error)
            }
            Err(error) => Err(error),
        }
    }
}

impl Drop for CudaBufferPoolReuseGuard {
    fn drop(&mut self) {
        if self.active {
            // Last-resort protection for abandoned queued work or unwinding.
            // Normal Result paths establish actual completion, then call
            // `release` so failures can be surfaced. If completion cannot be
            // established here, leave the hold active rather than make
            // possibly referenced allocations reusable.
            let outcome = self.pool.context.synchronize_for_resource_release();
            if outcome.completion_established() {
                if self.release_inner().is_err() && self.active {
                    std::mem::forget(self.pool.clone());
                }
            } else {
                std::mem::forget(self.pool.clone());
            }
        }
    }
}
