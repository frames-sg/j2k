// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::error::CudaError;

mod creation;
mod device;
mod diagnostics;
mod host_budget;
mod inner;
mod kernel_cache;
mod kernel_dispatch;
mod lifecycle;
mod operations;
mod pinned_host;
mod pointer;
mod resource_creation;
#[cfg(test)]
mod test_kernels;

pub use self::diagnostics::CudaContextDiagnostics;
#[doc(hidden)]
pub use self::host_budget::{CudaExternalHostOwner, CudaExternalHostReservation};
#[cfg(test)]
pub(crate) use self::pinned_host::validate_non_null_pinned_host_allocation;
#[cfg(test)]
pub(crate) use self::test_kernels::{CudaKernelModule, CudaKernelName};
pub(crate) use self::{
    inner::{ContextInner, ContextOwnership},
    kernel_cache::{CompiledKernel, CompiledKernelKey},
    lifecycle::ContextResourceLifecycle,
    pinned_host::PinnedUploadStaging,
    resource_creation::{validate_device_allocation, validate_resource_handle},
};

/// CUDA driver context shared by J2K CUDA adapter crates.
#[derive(Clone)]
pub struct CudaContext {
    pub(crate) inner: Arc<ContextInner>,
}

impl CudaContext {
    /// Bind this context for an engine operation without submitting work.
    #[doc(hidden)]
    pub fn prepare_operation(&self) -> Result<(), CudaError> {
        self.inner.set_current()
    }

    /// Return whether uncertain completion quarantined resource lifetimes.
    #[doc(hidden)]
    #[must_use]
    pub fn resource_lifetimes_poisoned(&self) -> bool {
        self.inner.resource_lifetimes_poisoned()
    }

    /// Returns whether both handles own the same CUDA driver context.
    #[doc(hidden)]
    #[must_use]
    pub fn is_same_context(&self, other: &Self) -> bool {
        self.inner.context == other.inner.context
    }

    /// Device ordinal associated with this context.
    #[doc(hidden)]
    #[must_use]
    pub fn device_ordinal(&self) -> usize {
        self.inner.device_ordinal
    }

    /// Validate and resolve a raw device pointer for this context.
    ///
    /// Stream-ordered allocations whose pointer attributes omit a direct
    /// context are resolved through the runtime's allocation provenance path.
    ///
    /// # Errors
    ///
    /// Returns a driver or validation error when the pointer is not a live
    /// allocation associated with this context.
    #[doc(hidden)]
    pub fn validate_device_pointer(&self, ptr: u64) -> Result<u64, CudaError> {
        self.inner.resolve_pointer_for_context(ptr)
    }
}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaContext").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod structure_tests;
