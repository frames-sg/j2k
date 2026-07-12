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
