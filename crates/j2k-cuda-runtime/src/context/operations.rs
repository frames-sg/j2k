// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CudaError;

use super::ContextInner;

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

    /// Run a resource state transition, quarantining uncertain failure.
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
