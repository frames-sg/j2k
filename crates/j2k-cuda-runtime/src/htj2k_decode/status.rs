// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::{select_resource_release_error, CudaError},
    execution::CudaExecutionStats,
};

use super::CudaHtj2kStatus;

pub(super) fn first_status_error(
    statuses: &[CudaHtj2kStatus],
    kernel: &'static str,
) -> Option<CudaError> {
    statuses
        .iter()
        .copied()
        .find(|status| !status.is_ok())
        .map(|status| CudaError::KernelStatus {
            kernel,
            code: status.code,
            detail: status.detail,
        })
}

pub(super) fn select_status_release_result(
    execution: CudaExecutionStats,
    status_error: Option<CudaError>,
    release_result: Result<(), CudaError>,
) -> Result<CudaExecutionStats, CudaError> {
    match (status_error, release_result) {
        (Some(primary_error), Err(release_error)) => {
            Err(select_resource_release_error(primary_error, release_error))
        }
        (Some(error), Ok(())) | (None, Err(error)) => Err(error),
        (None, Ok(())) => Ok(execution),
    }
}

#[cfg(test)]
mod tests;
