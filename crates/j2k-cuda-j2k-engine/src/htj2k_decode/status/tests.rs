// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn kernel_error() -> CudaError {
    CudaError::KernelStatus {
        kernel: "test_kernel",
        code: 7,
        detail: 11,
    }
}

fn release_error() -> CudaError {
    CudaError::StatePoisoned {
        message: "release failed".to_string(),
    }
}

#[test]
fn ht_cleanup_status_preserves_the_failing_descriptor_index() {
    let statuses = [
        CudaHtj2kStatus::default(),
        CudaHtj2kStatus {
            code: 7,
            detail: 11,
            ..CudaHtj2kStatus::default()
        },
    ];
    assert!(matches!(
        first_status_error(&statuses, "test_kernel"),
        Some(CudaError::KernelJobStatus {
            kernel: "test_kernel",
            job_index: 1,
            code: 7,
            detail: 11,
        })
    ));
}

#[test]
fn grouped_ht_cleanup_status_preserves_the_global_descriptor_index_and_kernel() {
    let statuses = [
        CudaHtj2kStatus::default(),
        CudaHtj2kStatus::default(),
        CudaHtj2kStatus {
            code: 9,
            detail: 17,
            ..CudaHtj2kStatus::default()
        },
    ];
    let spans = [
        CudaHtj2kStatusSpan {
            start: 0,
            count: 1,
            kernel: "cleanup_only",
        },
        CudaHtj2kStatusSpan {
            start: 1,
            count: 2,
            kernel: "refinement",
        },
    ];
    assert!(matches!(
        first_group_status_error(&statuses, &spans),
        Some(CudaError::KernelJobStatus {
            kernel: "refinement",
            job_index: 2,
            code: 9,
            detail: 17,
        })
    ));
}

#[test]
fn kernel_and_release_failures_are_both_preserved() {
    let error = select_status_release_result(
        CudaExecutionStats::default(),
        Some(kernel_error()),
        Err(release_error()),
    )
    .expect_err("compound failure");
    assert!(matches!(error, CudaError::ResourceReleaseFailed { .. }));
    let rendered = error.to_string();
    assert!(rendered.contains("status 7 detail 11"));
    assert!(rendered.contains("release failed"));
}

#[test]
fn single_failure_and_success_paths_keep_their_original_result() {
    assert!(matches!(
        select_status_release_result(CudaExecutionStats::default(), Some(kernel_error()), Ok(())),
        Err(CudaError::KernelStatus { .. })
    ));
    assert!(matches!(
        select_status_release_result(CudaExecutionStats::default(), None, Err(release_error())),
        Err(CudaError::StatePoisoned { .. })
    ));
    assert!(select_status_release_result(CudaExecutionStats::default(), None, Ok(())).is_ok());
}
