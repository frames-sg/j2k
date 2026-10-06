// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    retain_pinned_upload_staging_after_abandoned_checkout, select_pinned_upload_result,
};
use super::CudaPinnedUploadOperationGuard;
use crate::{context::PinnedUploadStaging, error::select_resource_release_error};
use crate::{CudaDeviceBuffer, CudaError};
use std::sync::OnceLock;

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
        self.copy_from_parts(&[bytes])?;
        let upload_result = self.operation.context.upload(self.as_slice()?);
        let recycle_result = self.recycle_inner();
        select_pinned_upload_result(upload_result, recycle_result)
    }

    /// Recycle prepared staging without uploading it.
    pub fn recycle(mut self) -> Result<(), CudaError> {
        self.recycle_inner()
    }

    /// Copy `parts` back to back into staging; their total must match the checkout.
    pub(crate) fn copy_from_parts(&mut self, parts: &[&[u8]]) -> Result<(), CudaError> {
        let total = parts
            .iter()
            .try_fold(0_usize, |total, part| total.checked_add(part.len()));
        if total != Some(self.requested_len) {
            return Err(CudaError::InvalidArgument {
                message: "prepared CUDA pinned upload byte length changed".to_string(),
            });
        }
        let staging = self.staging.as_mut().ok_or(CudaError::InternalInvariant {
            what: "CUDA pinned upload staging checkout is empty",
        })?;
        let target = &mut staging.as_mut_slice()[..self.requested_len];
        let workers = if self.requested_len >= PARALLEL_STAGING_FILL_MIN_BYTES {
            staging_fill_workers().min(parts.len())
        } else {
            1
        };
        if workers <= 1 {
            copy_parts(target, parts);
        } else {
            copy_parts_in_parallel(target, parts, workers);
        }
        Ok(())
    }

    pub(crate) fn as_slice(&self) -> Result<&[u8], CudaError> {
        let staging = self.staging.as_ref().ok_or(CudaError::InternalInvariant {
            what: "CUDA pinned upload staging checkout is empty",
        })?;
        Ok(&staging.as_slice()[..self.requested_len])
    }

    pub(crate) fn into_pending(
        mut self,
        completion: crate::CudaEvent,
    ) -> Result<super::PendingPinnedUpload, CudaError> {
        let staging = self.staging.take().ok_or(CudaError::InternalInvariant {
            what: "CUDA pinned upload staging checkout is empty",
        })?;
        Ok(super::PendingPinnedUpload::new(
            self.operation.context.clone(),
            staging,
            self.requested_len,
            completion,
        ))
    }

    fn recycle_inner(&mut self) -> Result<(), CudaError> {
        let staging = self.staging.take().ok_or(CudaError::InternalInvariant {
            what: "CUDA pinned upload staging checkout is empty",
        })?;
        self.operation.held().recycle_pinned_upload_staging(staging)
    }

    fn recycle_with_primary_error<T>(&mut self, error: CudaError) -> Result<T, CudaError> {
        match self.recycle_inner() {
            Ok(()) => Err(error),
            Err(recycle_error) => Err(select_resource_release_error(error, recycle_error)),
        }
    }
}

fn copy_parts(mut target: &mut [u8], parts: &[&[u8]]) {
    for part in parts {
        let (head, tail) = target.split_at_mut(part.len());
        head.copy_from_slice(part);
        target = tail;
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

/// Uploads at least this large fill disjoint pinned spans from several threads.
const PARALLEL_STAGING_FILL_MIN_BYTES: usize = 4 * 1024 * 1024;
/// Temporary fill threads per upload, bounded independently of the caller's
/// planning thread pool.
const MAX_STAGING_FILL_WORKERS: usize = 4;

fn staging_fill_workers() -> usize {
    static WORKERS: OnceLock<usize> = OnceLock::new();
    *WORKERS.get_or_init(|| {
        std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(MAX_STAGING_FILL_WORKERS)
    })
}

/// Copy `parts` into `target` with up to `workers` threads. A span whose
/// thread cannot be created is copied on this thread instead.
fn copy_parts_in_parallel(target: &mut [u8], parts: &[&[u8]], workers: usize) {
    let mut spans: [(&mut [u8], &[&[u8]]); MAX_STAGING_FILL_WORKERS] = Default::default();
    let mut span_count = 0;
    let mut remaining = target;
    for group in parts.chunks(parts.len().div_ceil(workers)) {
        let len = group.iter().map(|part| part.len()).sum();
        let (head, tail) = std::mem::take(&mut remaining).split_at_mut(len);
        remaining = tail;
        spans[span_count] = (head, group);
        span_count += 1;
    }
    let Some((last, spawned)) = spans[..span_count].split_last_mut() else {
        return;
    };
    let unspawned = std::thread::scope(|scope| {
        let mut unspawned = [false; MAX_STAGING_FILL_WORKERS];
        for (index, span) in spawned.iter_mut().enumerate() {
            let (head, group) = (&mut *span.0, span.1);
            unspawned[index] = std::thread::Builder::new()
                .spawn_scoped(scope, move || copy_parts(head, group))
                .is_err();
        }
        copy_parts(last.0, last.1);
        unspawned
    });
    for (span, failed) in spawned.iter_mut().zip(unspawned) {
        if failed {
            copy_parts(span.0, span.1);
        }
    }
}
