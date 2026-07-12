// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CudaError;

use super::ContextInner;

pub(crate) fn ensure_context_ownership(
    matches_context: impl IntoIterator<Item = bool>,
    mismatch_message: &'static str,
) -> Result<(), CudaError> {
    if matches_context.into_iter().all(|matches| matches) {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: mismatch_message.to_string(),
        })
    }
}

impl ContextInner {
    pub(crate) fn set_current(&self) -> Result<(), CudaError> {
        self.resource_lifecycle.run_recoverable(
            || self.set_current_for_resource_release(),
            || Ok(()),
            || self.synchronize_current_after_operation_error(),
        )
    }

    pub(crate) fn set_current_for_resource_release(&self) -> Result<(), CudaError> {
        // SAFETY: context is created by cuCtxCreate_v2 and remains valid while
        // ContextInner is alive.
        self.driver.check("cuCtxSetCurrent", unsafe {
            (self.driver.cu_ctx_set_current)(self.context)
        })
    }

    pub(crate) fn ensure_resource_lifetime_available(&self) -> Result<(), CudaError> {
        self.resource_lifecycle.ensure_available()
    }

    /// Serialize a context-bound driver operation and recover its context.
    ///
    /// A CUDA driver call may surface a failure from earlier asynchronous work.
    /// On failure, the lifecycle gate remains held while a context-wide
    /// synchronization establishes completion. The context is poisoned only
    /// when that recovery cannot establish completion.
    pub(crate) fn with_current_resource_operation<T>(
        &self,
        operation: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        self.resource_lifecycle.run_recoverable(
            || self.set_current_for_resource_release(),
            operation,
            || self.synchronize_current_after_operation_error(),
        )
    }

    pub(crate) fn with_current_completion_operation<T>(
        &self,
        operation: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        self.resource_lifecycle
            .run_completion(|| self.set_current_for_resource_release(), operation)
    }

    /// Run an operation that creates, destroys, or transfers ownership of a
    /// CUDA resource. Any failure quarantines the context because successful
    /// synchronization cannot prove whether that state transition committed.
    pub(crate) fn with_current_stateful_operation<T>(
        &self,
        operation: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        self.resource_lifecycle.run_stateful(
            || self.set_current_for_resource_release(),
            operation,
            || self.synchronize_current_after_operation_error(),
        )
    }

    fn synchronize_current_after_operation_error(&self) -> Result<(), CudaError> {
        // SAFETY: run_recoverable invokes this only while the lifecycle gate is
        // held and after this context was made current on the calling thread.
        let status = unsafe { (self.driver.cu_ctx_synchronize)() };
        self.driver.check("cuCtxSynchronize", status)
    }

    pub(crate) fn resource_lifetimes_poisoned(&self) -> bool {
        self.resource_lifecycle.is_poisoned()
    }
}
