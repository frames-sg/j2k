// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    retain_pinned_upload_staging_after_active_release_failure,
    retain_pinned_upload_staging_after_release_failure,
};
use super::CudaPinnedUploadOperationGuard;
use crate::{
    context::PinnedUploadStaging,
    error::{select_resource_release_error, CudaError},
};

#[derive(Clone, Copy)]
enum ReleaseAccounting {
    Active,
    Inactive,
}

impl CudaPinnedUploadOperationGuard<'_> {
    pub(super) fn release_active_pinned_upload_staging(
        &self,
        staging: PinnedUploadStaging,
        primary_error: Option<CudaError>,
    ) -> Result<(), CudaError> {
        self.release_pinned_upload_staging(staging, primary_error, ReleaseAccounting::Active)
    }

    pub(super) fn release_inactive_pinned_upload_staging(
        &self,
        staging: PinnedUploadStaging,
        primary_error: Option<CudaError>,
    ) -> Result<(), CudaError> {
        self.release_pinned_upload_staging(staging, primary_error, ReleaseAccounting::Inactive)
    }

    fn release_pinned_upload_staging(
        &self,
        mut staging: PinnedUploadStaging,
        primary_error: Option<CudaError>,
        accounting: ReleaseAccounting,
    ) -> Result<(), CudaError> {
        let staging_len = staging.len;
        match self
            .context
            .inner
            .with_current_stateful_operation(|| staging.free(&self.context.inner.driver))
        {
            Ok(()) => {
                let accounting_result = match accounting {
                    ReleaseAccounting::Active => {
                        match self.context.inner.pinned_upload_staging.lock() {
                            Ok(mut pool) => pool.finish_active_checkout(staging_len),
                            Err(error) => Err(CudaError::StatePoisoned {
                                message: error.to_string(),
                            }),
                        }
                    }
                    ReleaseAccounting::Inactive => Ok(()),
                };
                match (primary_error, accounting_result) {
                    (None, Ok(())) => Ok(()),
                    (Some(error), Ok(())) | (None, Err(error)) => Err(error),
                    (Some(primary), Err(release)) => {
                        Err(select_resource_release_error(primary, release))
                    }
                }
            }
            Err(release_error) => {
                // A failed ownership transition leaves the allocation's state
                // uncertain. Quarantine it under the context and retain every
                // primary/release diagnostic.
                let retention_result = match accounting {
                    ReleaseAccounting::Active => {
                        retain_pinned_upload_staging_after_active_release_failure(
                            self.context.inner.pinned_upload_staging.lock(),
                            staging,
                        )
                    }
                    ReleaseAccounting::Inactive => {
                        retain_pinned_upload_staging_after_release_failure(
                            self.context.inner.pinned_upload_staging.lock(),
                            staging,
                        )
                    }
                };
                let release_error = match retention_result {
                    Ok(()) => release_error,
                    Err(retention_error) => {
                        select_resource_release_error(release_error, retention_error)
                    }
                };
                Err(match primary_error {
                    Some(primary_error) => {
                        select_resource_release_error(primary_error, release_error)
                    }
                    None => release_error,
                })
            }
        }
    }
}
