// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{allocation::HostPhaseBudget, error::CudaError};
use std::sync::{Mutex, MutexGuard};

pub(super) const PINNED_UPLOAD_STAGING_ALLOCATION: &str = "CUDA pinned upload staging allocation";

pub(super) fn validate_pinned_upload_staging_len(len: usize, cap: usize) -> Result<(), CudaError> {
    if len == 0 {
        return Err(CudaError::InvalidArgument {
            message: "prepared CUDA pinned upload staging cannot be empty".to_string(),
        });
    }
    HostPhaseBudget::with_cap(PINNED_UPLOAD_STAGING_ALLOCATION, cap).account_bytes(len)
}

pub(super) fn lock_pinned_upload_operation(
    gate: &Mutex<()>,
) -> Result<MutexGuard<'_, ()>, CudaError> {
    gate.lock().map_err(|error| CudaError::StatePoisoned {
        message: error.to_string(),
    })
}

pub(super) fn validate_pinned_upload_operation_context(
    is_same_context: bool,
) -> Result<(), CudaError> {
    if is_same_context {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: "pinned upload transaction belongs to a different CUDA context".to_string(),
        })
    }
}

#[cfg(test)]
mod tests;
