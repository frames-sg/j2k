// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    retain_pinned_upload_staging_after_abandoned_checkout, select_pinned_upload_result,
};
use super::CudaPinnedUploadOperationGuard;
use crate::{context::PinnedUploadStaging, error::select_resource_release_error};
use crate::{CudaDeviceBuffer, CudaError};

#[doc(hidden)]
/// RAII checkout of one page-locked upload allocation.
///
/// Normal completion uploads or recycles explicitly. Abandonment or unwinding
/// quarantines the raw allocation instead of losing ownership.
#[must_use = "the pinned staging checkout must be uploaded or recycled"]
pub struct CudaPinnedUploadStagingCheckout<'operation, 'context> {
    pub(super) operation: &'operation CudaPinnedUploadOperationGuard<'context>,
    pub(super) staging: Option<PinnedUploadStaging>,
    pub(super) requested_len: usize,
    pub(super) allocation_len: usize,
}

impl CudaPinnedUploadStagingCheckout<'_, '_> {
    /// Actual page-locked allocation bytes backing this checkout.
    #[must_use]
    pub fn allocation_byte_len(&self) -> usize {
        self.allocation_len
    }

    /// Total page-locked bytes retained by this context, including this checkout.
    pub fn retained_page_locked_bytes(&self) -> Result<usize, CudaError> {
        self.operation
            .context
            .inner
            .pinned_upload_staging
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?
            .diagnostics()
            .map(|diagnostics| diagnostics.retained_bytes)
    }

    /// Copy and upload the prepared byte count, then recycle staging.
    pub fn upload(mut self, bytes: &[u8]) -> Result<CudaDeviceBuffer, CudaError> {
        if bytes.len() != self.requested_len {
            let error = CudaError::InvalidArgument {
                message: "prepared CUDA pinned upload byte length changed".to_string(),
            };
            return self.recycle_with_primary_error(error);
        }
        self.copy_from_slice(bytes)?;
        let upload_result = self.operation.context.upload(self.as_slice()?);
        let recycle_result = self.recycle_inner();
        select_pinned_upload_result(upload_result, recycle_result)
    }

    /// Recycle prepared staging without uploading it.
    pub fn recycle(mut self) -> Result<(), CudaError> {
        self.recycle_inner()
    }

    pub(crate) fn copy_from_slice(&mut self, bytes: &[u8]) -> Result<(), CudaError> {
        if bytes.len() != self.requested_len {
            return Err(CudaError::InvalidArgument {
                message: "prepared CUDA pinned upload byte length changed".to_string(),
            });
        }
        let staging = self.staging.as_mut().ok_or(CudaError::InternalInvariant {
            what: "CUDA pinned upload staging checkout is empty",
        })?;
        staging.as_mut_slice()[..self.requested_len].copy_from_slice(bytes);
        Ok(())
    }

    pub(crate) fn as_slice(&self) -> Result<&[u8], CudaError> {
        let staging = self.staging.as_ref().ok_or(CudaError::InternalInvariant {
            what: "CUDA pinned upload staging checkout is empty",
        })?;
        Ok(&staging.as_slice()[..self.requested_len])
    }

    fn recycle_inner(&mut self) -> Result<(), CudaError> {
        let staging = self.staging.take().ok_or(CudaError::InternalInvariant {
            what: "CUDA pinned upload staging checkout is empty",
        })?;
        self.operation.recycle_pinned_upload_staging(staging)
    }

    fn recycle_with_primary_error<T>(&mut self, error: CudaError) -> Result<T, CudaError> {
        match self.recycle_inner() {
            Ok(()) => Err(error),
            Err(recycle_error) => Err(select_resource_release_error(error, recycle_error)),
        }
    }
}

impl Drop for CudaPinnedUploadStagingCheckout<'_, '_> {
    fn drop(&mut self) {
        if let Some(staging) = self.staging.take() {
            retain_pinned_upload_staging_after_abandoned_checkout(
                self.operation.context.inner.pinned_upload_staging.lock(),
                staging,
            );
        }
    }
}

#[cfg(test)]
mod tests;
