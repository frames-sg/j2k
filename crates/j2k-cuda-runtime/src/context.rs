// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::{
    error::CudaError,
    execution::CudaExecutionStats,
    htj2k_decode::{
        htj2k_decode_needs_zero_fill, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeOutput,
        CudaHtj2kDecodeStageTimings,
    },
};

mod band_transfer;
mod compact;
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

pub use self::compact::{CudaHtj2kCompactEncodedCodeBlock, CudaHtj2kCompactEncodedCodeBlocks};
pub use self::diagnostics::CudaContextDiagnostics;
#[doc(hidden)]
pub use self::host_budget::{CudaExternalHostOwner, CudaExternalHostReservation};
#[cfg(test)]
pub(crate) use self::pinned_host::validate_non_null_pinned_host_allocation;
#[cfg(test)]
pub(crate) use self::test_kernels::{CudaKernelModule, CudaKernelName};
pub(crate) use self::{
    band_transfer::cuda_idwt_trace_enabled,
    compact::HTJ2K_UVLC_ENCODE_TABLE_BYTES,
    inner::{ContextInner, ContextOwnership},
    kernel_cache::{CompiledKernel, CompiledKernelKey},
    lifecycle::ContextResourceLifecycle,
    operations::ensure_context_ownership,
    pinned_host::PinnedUploadStaging,
    resource_creation::{validate_device_allocation, validate_resource_handle},
};

/// CUDA driver context shared by J2K CUDA adapter crates.
#[derive(Clone)]
pub struct CudaContext {
    pub(crate) inner: Arc<ContextInner>,
}

impl CudaContext {
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

    pub(crate) fn decode_empty_htj2k_codeblocks(
        &self,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.inner.set_current()?;
        let output_bytes = output_words
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: output_words })?;
        let coefficients = self.allocate(output_bytes)?;
        if htj2k_decode_needs_zero_fill(jobs, output_words)? {
            self.memset_d32(&coefficients, 0, output_words)?;
            self.synchronize()?;
        }
        Ok(CudaHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats::default(),
            statuses: Vec::new(),
            stage_timings: CudaHtj2kDecodeStageTimings::default(),
        })
    }
}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaContext").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod structure_tests;
