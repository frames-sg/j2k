// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{map_classic_status_error, ClassicJobIdentity};

#[test]
fn classic_kernel_failure_maps_to_responsible_source_index() {
    let identities = [
        ClassicJobIdentity {
            source_index: 2,
            original_job_index: 0,
        },
        ClassicJobIdentity {
            source_index: 6,
            original_job_index: 1,
        },
    ];
    let mapped = map_classic_status_error(
        j2k_cuda_runtime::CudaError::KernelJobStatus {
            kernel: "injected",
            job_index: 1,
            code: 5,
            detail: 13,
        },
        &identities,
    );
    assert!(matches!(
        mapped,
        crate::Error::CudaTier1JobFailed {
            source_index: 6,
            original_job_index: 1,
            ..
        }
    ));
}
