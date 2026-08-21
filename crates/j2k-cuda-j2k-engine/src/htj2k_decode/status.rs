// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::{select_resource_release_error, CudaError},
    execution::CudaExecutionStats,
};

use super::CudaHtj2kStatus;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CudaHtj2kStatusSpan {
    pub(super) start: usize,
    pub(super) count: usize,
    pub(super) kernel: &'static str,
}

pub(super) fn first_status_error(
    statuses: &[CudaHtj2kStatus],
    kernel: &'static str,
) -> Option<CudaError> {
    statuses
        .iter()
        .copied()
        .enumerate()
        .find(|(_, status)| !status.is_ok())
        .map(|(job_index, status)| CudaError::KernelJobStatus {
            kernel,
            job_index,
            code: status.code,
            detail: status.detail,
        })
}

pub(super) fn first_group_status_error(
    statuses: &[CudaHtj2kStatus],
    spans: &[CudaHtj2kStatusSpan],
) -> Option<CudaError> {
    for span in spans {
        let end = span.start.checked_add(span.count)?;
        let span_statuses = statuses.get(span.start..end)?;
        if let Some((local_index, status)) = span_statuses
            .iter()
            .copied()
            .enumerate()
            .find(|(_, status)| !status.is_ok())
        {
            return Some(CudaError::KernelJobStatus {
                kernel: span.kernel,
                job_index: span.start.saturating_add(local_index),
                code: status.code,
                detail: status.detail,
            });
        }
    }
    None
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
