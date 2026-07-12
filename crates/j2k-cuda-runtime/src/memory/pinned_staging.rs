// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{MutexGuard, PoisonError};

mod operations;
mod pool;

pub use self::operations::{CudaPinnedUploadOperationGuard, CudaPinnedUploadStagingCheckout};
pub use self::pool::{CudaPinnedUploadStagingPoolDiagnostics, CudaPinnedUploadStagingPoolLimits};
pub(crate) use self::pool::{PinnedUploadStagingAdmission, PinnedUploadStagingPool};

use crate::{
    context::PinnedUploadStaging,
    error::{select_resource_release_error, CudaError},
};

pub(crate) fn select_pinned_upload_result<T>(
    upload: Result<T, CudaError>,
    recycle: Result<(), CudaError>,
) -> Result<T, CudaError> {
    match (upload, recycle) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(primary_error), Err(release_error)) => {
            Err(select_resource_release_error(primary_error, release_error))
        }
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
    }
}

pub(super) fn retain_pinned_upload_staging_after_lock_poison(
    error: PoisonError<MutexGuard<'_, PinnedUploadStagingPool>>,
    staging: PinnedUploadStaging,
) -> CudaError {
    let message = error.to_string();
    let mut pool = error.into_inner();
    if let Err((retention_error, _unretained)) = pool.try_quarantine_active_checkout(staging) {
        // The poisoned lifetime state does not permit proving that the raw
        // pinned token is safe to release. `PinnedUploadStaging` intentionally
        // has no Drop, so returning without retaining it leaks rather than
        // freeing it early.
        // Returning consumes the no-Drop token without freeing its raw pointer.
        return select_resource_release_error(
            CudaError::StatePoisoned { message },
            retention_error,
        );
    }
    CudaError::StatePoisoned { message }
}

pub(super) fn retain_pinned_upload_staging_after_release_failure(
    pool: Result<
        MutexGuard<'_, PinnedUploadStagingPool>,
        PoisonError<MutexGuard<'_, PinnedUploadStagingPool>>,
    >,
    staging: PinnedUploadStaging,
) -> Result<(), CudaError> {
    let mut pool = match pool {
        Ok(pool) => pool,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Err((error, _unretained)) = pool.try_retain_after_uncertain_release(staging) {
        // The preceding CUDA release failed. This raw token has no Drop, so
        // returning without retaining it cannot accidentally free it early.
        // Returning consumes the no-Drop token without freeing its raw pointer.
        return Err(error);
    }
    Ok(())
}

pub(super) fn retain_pinned_upload_staging_after_active_release_failure(
    pool: Result<
        MutexGuard<'_, PinnedUploadStagingPool>,
        PoisonError<MutexGuard<'_, PinnedUploadStagingPool>>,
    >,
    staging: PinnedUploadStaging,
) -> Result<(), CudaError> {
    let mut pool = match pool {
        Ok(pool) => pool,
        Err(poisoned) => poisoned.into_inner(),
    };
    pool.try_quarantine_active_checkout(staging)
        .map_err(|(error, _untracked)| error)
}

pub(super) fn retain_pinned_upload_staging_after_abandoned_checkout(
    pool: Result<
        MutexGuard<'_, PinnedUploadStagingPool>,
        PoisonError<MutexGuard<'_, PinnedUploadStagingPool>>,
    >,
    staging: PinnedUploadStaging,
) {
    let mut pool = match pool {
        Ok(pool) => pool,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Err((_error, _untracked)) = pool.try_quarantine_active_checkout(staging) {
        // Checkout preflight reserves quarantine metadata before a raw token is
        // removed or allocated. If an invariant still prevents retention, the
        // pool marks its accounting poisoned and this no-Drop token leaks; all
        // later staging and exact diagnostics then fail closed.
    }
}

#[cfg(test)]
mod tests;
