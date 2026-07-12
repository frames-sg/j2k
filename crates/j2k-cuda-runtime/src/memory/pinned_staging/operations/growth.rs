// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transactional checkout growth under the context-wide host authority.

use super::{policy::validate_pinned_upload_staging_len, CudaPinnedUploadOperationGuard};
use crate::{
    context::PinnedUploadStaging,
    error::{select_resource_release_error, CudaError},
};
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

impl CudaPinnedUploadOperationGuard<'_> {
    pub(super) fn take_pinned_upload_staging(
        &self,
        len: usize,
    ) -> Result<PinnedUploadStaging, CudaError> {
        validate_pinned_upload_staging_len(len, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
        self.context.inner.ensure_resource_lifetime_available()?;
        let mut pool = self
            .context
            .inner
            .pinned_upload_staging
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        pool.prepare_unwind_quarantine_slots()?;
        if let Some(staging) = pool.take_best_fit(len)? {
            return Ok(staging);
        }
        self.context.reserve_pinned_host_bytes(len)?;
        let pool_growth_fits =
            match pool.cached_plus_request_fits_host_cap(len, DEFAULT_MAX_HOST_ALLOCATION_BYTES) {
                Ok(fits) => fits,
                Err(error) => {
                    drop(pool);
                    return Err(match self.context.release_pinned_host_bytes(len) {
                        Ok(()) => error,
                        Err(release) => select_resource_release_error(error, release),
                    });
                }
            };
        if pool_growth_fits {
            if let Err(error) = pool.begin_new_active_checkout(len) {
                drop(pool);
                return Err(match self.context.release_pinned_host_bytes(len) {
                    Ok(()) => error,
                    Err(release) => select_resource_release_error(error, release),
                });
            }
            drop(pool);
        } else {
            drop(pool);
            let primary = CudaError::InternalInvariant {
                what: "CUDA pinned pool and context host authority disagree about growth admission",
            };
            return Err(match self.context.release_pinned_host_bytes(len) {
                Ok(()) => primary,
                Err(release) => select_resource_release_error(primary, release),
            });
        }

        let mut ptr = std::ptr::null_mut();
        let allocation_result = self.context.inner.with_current_stateful_operation(|| {
            // SAFETY: CUDA writes a page-locked host pointer for the requested
            // byte length while this context's lifecycle gate is held.
            self.context.inner.driver.check("cuMemHostAlloc", unsafe {
                (self.context.inner.driver.cu_mem_host_alloc)(&raw mut ptr, len, 0)
            })?;
            PinnedUploadStaging::from_raw(ptr.cast::<u8>(), len)
        });
        let staging = match allocation_result {
            Ok(staging) => staging,
            Err(error) => return Err(self.rollback_active_checkout_error(len, error)),
        };
        let confirmation = match self.context.inner.pinned_upload_staging.lock() {
            Ok(mut pool) => pool.confirm_new_active_checkout(),
            Err(poisoned) => Err(CudaError::StatePoisoned {
                message: poisoned.to_string(),
            }),
        };
        if let Err(error) = confirmation {
            return match self.release_active_pinned_upload_staging(staging, Some(error)) {
                Err(error) => Err(error),
                Ok(()) => Err(CudaError::InternalInvariant {
                    what: "CUDA pinned upload staging lost confirmation failure",
                }),
            };
        }
        Ok(staging)
    }

    fn rollback_active_checkout_error(&self, len: usize, error: CudaError) -> CudaError {
        let accounting_result = match self.context.inner.pinned_upload_staging.lock() {
            Ok(mut pool) => pool.finish_active_checkout(len),
            Err(poisoned) => Err(CudaError::StatePoisoned {
                message: poisoned.to_string(),
            }),
        };
        let authority_result = self.context.release_pinned_host_bytes(len);
        let rollback_result = match (accounting_result, authority_result) {
            (Ok(()), Ok(())) => return error,
            (Err(accounting), Ok(())) | (Ok(()), Err(accounting)) => accounting,
            (Err(accounting), Err(authority)) => {
                select_resource_release_error(accounting, authority)
            }
        };
        select_resource_release_error(error, rollback_result)
    }
}
