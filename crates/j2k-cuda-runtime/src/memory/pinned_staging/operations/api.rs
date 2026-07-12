// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::CudaPinnedUploadStagingPoolDiagnostics;
use crate::{
    bytes::{f32_slice_as_bytes, i32_slice_as_bytes},
    context::CudaContext,
    memory::CudaDeviceBuffer,
    CudaError,
};

pub(super) fn pinned_upload_staging_pool_diagnostics(
    context: &CudaContext,
) -> Result<CudaPinnedUploadStagingPoolDiagnostics, CudaError> {
    context
        .inner
        .pinned_upload_staging
        .lock()
        .map_err(|error| CudaError::StatePoisoned {
            message: error.to_string(),
        })?
        .diagnostics()
}

impl CudaContext {
    #[doc(hidden)]
    /// Snapshot page-locked upload-staging retention for this CUDA context.
    ///
    /// The snapshot is exact for this context at the instant its pool mutex is
    /// held. It includes checked-out staging owned by this context, but excludes
    /// unrelated host owners. This is observability rather than a cross-owner admission transaction.
    pub fn pinned_upload_staging_pool_diagnostics(
        &self,
    ) -> Result<CudaPinnedUploadStagingPoolDiagnostics, CudaError> {
        let _operation = match self.inner.pinned_upload_operation.lock() {
            Ok(operation) => operation,
            Err(poisoned) => poisoned.into_inner(),
        };
        pinned_upload_staging_pool_diagnostics(self)
    }

    /// Upload host `f32` samples through a temporary page-locked staging buffer.
    pub fn upload_f32_pinned(&self, samples: &[f32]) -> Result<CudaDeviceBuffer, CudaError> {
        self.upload_pinned(f32_slice_as_bytes(samples))
    }

    /// Upload host `i32` samples through a temporary page-locked staging buffer.
    pub(crate) fn upload_i32_pinned(&self, samples: &[i32]) -> Result<CudaDeviceBuffer, CudaError> {
        self.upload_pinned(i32_slice_as_bytes(samples))
    }
}
