// SPDX-License-Identifier: MIT OR Apache-2.0

mod api;
mod checkout;
mod gate;
mod growth;
mod policy;
mod recycle;

use self::api::pinned_upload_staging_pool_diagnostics;
pub use self::checkout::CudaPinnedUploadStagingCheckout;
pub use self::gate::CudaPinnedUploadOperationGuard;
use self::policy::{lock_pinned_upload_operation, validate_pinned_upload_operation_context};
use super::{
    retain_pinned_upload_staging_after_lock_poison, CudaPinnedUploadStagingPoolDiagnostics,
    PinnedUploadStagingAdmission,
};
use crate::{
    context::{CudaContext, PinnedUploadStaging},
    error::CudaError,
    memory::CudaDeviceBuffer,
};

impl CudaContext {
    /// Upload host bytes through a temporary page-locked staging buffer.
    pub fn upload_pinned(&self, bytes: &[u8]) -> Result<CudaDeviceBuffer, CudaError> {
        if bytes.is_empty() {
            return self.upload(bytes);
        }
        self.begin_pinned_upload_operation()?.upload(bytes)
    }

    #[doc(hidden)]
    /// Begin a serialized page-locked upload-staging transaction.
    pub fn begin_pinned_upload_operation(
        &self,
    ) -> Result<CudaPinnedUploadOperationGuard<'_>, CudaError> {
        let gate = lock_pinned_upload_operation(&self.inner.pinned_upload_operation)?;
        Ok(CudaPinnedUploadOperationGuard {
            context: self,
            _gate: gate,
            _not_sync: std::marker::PhantomData,
        })
    }
}

impl CudaPinnedUploadOperationGuard<'_> {
    #[doc(hidden)]
    /// Return whether this transaction guard belongs to `context`.
    #[must_use]
    pub fn is_for_context(&self, context: &CudaContext) -> bool {
        self.context.is_same_context(context)
    }

    #[doc(hidden)]
    /// Reject use of this transaction with a different CUDA context.
    pub fn ensure_for_context(&self, context: &CudaContext) -> Result<(), CudaError> {
        validate_pinned_upload_operation_context(self.is_for_context(context))
    }

    #[doc(hidden)]
    /// Upload bytes while retaining this transaction through staging recycle.
    pub fn upload(&self, bytes: &[u8]) -> Result<CudaDeviceBuffer, CudaError> {
        if bytes.is_empty() {
            return self.context.upload(bytes);
        }
        self.prepare_upload(bytes.len())?.upload(bytes)
    }

    #[doc(hidden)]
    /// Check out staging while retaining this operation's serialization gate.
    pub fn prepare_upload(
        &self,
        requested_len: usize,
    ) -> Result<CudaPinnedUploadStagingCheckout<'_, '_>, CudaError> {
        let staging = self.take_pinned_upload_staging(requested_len)?;
        let allocation_len = staging.len;
        Ok(CudaPinnedUploadStagingCheckout {
            operation: self,
            staging: Some(staging),
            requested_len,
            allocation_len,
        })
    }

    pub(crate) fn recycle_pinned_upload_staging(
        &self,
        staging: PinnedUploadStaging,
    ) -> Result<(), CudaError> {
        let mut candidate = Some(staging);
        loop {
            let Some(candidate_bytes) = candidate.as_ref().map(|staging| staging.len) else {
                return Err(CudaError::InternalInvariant {
                    what: "CUDA pinned upload staging candidate ownership was lost",
                });
            };
            let mut pool = match self.context.inner.pinned_upload_staging.lock() {
                Ok(pool) => pool,
                Err(error) => {
                    let Some(candidate) = candidate.take() else {
                        return Err(CudaError::InternalInvariant {
                            what: "CUDA pinned upload staging candidate ownership was lost",
                        });
                    };
                    return Err(retain_pinned_upload_staging_after_lock_poison(
                        error, candidate,
                    ));
                }
            };
            let admission = match pool.admission(candidate_bytes) {
                Ok(admission) => admission,
                Err(error) => {
                    drop(pool);
                    let Some(candidate) = candidate.take() else {
                        return Err(CudaError::InternalInvariant {
                            what: "CUDA pinned upload staging candidate ownership was lost",
                        });
                    };
                    return self.release_active_pinned_upload_staging(candidate, Some(error));
                }
            };
            match admission {
                PinnedUploadStagingAdmission::Reject => {
                    pool.note_rejection();
                    drop(pool);
                    let Some(candidate) = candidate.take() else {
                        return Err(CudaError::InternalInvariant {
                            what: "CUDA pinned upload staging candidate ownership was lost",
                        });
                    };
                    return self.release_active_pinned_upload_staging(candidate, None);
                }
                PinnedUploadStagingAdmission::Evict => {
                    let evicted = match pool.evict_largest_oldest() {
                        Ok(Some(evicted)) => evicted,
                        Ok(None) => {
                            drop(pool);
                            let error = CudaError::InternalInvariant {
                                what:
                                    "CUDA pinned upload staging selected eviction without a victim",
                            };
                            let Some(candidate) = candidate.take() else {
                                return Err(error);
                            };
                            return self
                                .release_active_pinned_upload_staging(candidate, Some(error));
                        }
                        Err(error) => {
                            drop(pool);
                            let Some(candidate) = candidate.take() else {
                                return Err(error);
                            };
                            return self
                                .release_active_pinned_upload_staging(candidate, Some(error));
                        }
                    };
                    drop(pool);
                    if let Err(release_error) =
                        self.release_inactive_pinned_upload_staging(evicted, None)
                    {
                        let Some(candidate) = candidate.take() else {
                            return Err(release_error);
                        };
                        return self
                            .release_active_pinned_upload_staging(candidate, Some(release_error));
                    }
                }
                PinnedUploadStagingAdmission::Admit => {
                    let Some(admitted) = candidate.take() else {
                        drop(pool);
                        return Err(CudaError::InternalInvariant {
                            what: "CUDA pinned upload staging admitted a missing candidate",
                        });
                    };
                    return match pool.try_admit_active(admitted) {
                        Ok(()) => Ok(()),
                        Err((error, unretained)) => {
                            drop(pool);
                            self.release_active_pinned_upload_staging(unretained, Some(error))
                        }
                    };
                }
            }
        }
    }

    #[doc(hidden)]
    /// Snapshot retained staging while this transaction excludes peer uploads.
    pub fn diagnostics(&self) -> Result<CudaPinnedUploadStagingPoolDiagnostics, CudaError> {
        pinned_upload_staging_pool_diagnostics(self.context)
    }

    /// Verify that the context authority and pinned pool retain the same bytes.
    #[doc(hidden)]
    pub fn verify_host_budget(&self) -> Result<(), CudaError> {
        let authority_bytes = self.context.authority_pinned_host_bytes()?;
        let pool_bytes = self.diagnostics()?.retained_bytes;
        if authority_bytes == pool_bytes {
            Ok(())
        } else {
            self.context.poison_host_budget();
            Err(CudaError::InternalInvariant {
                what: "CUDA pinned pool and context host authority byte totals diverged",
            })
        }
    }
}
