// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) use j2k_cuda_runtime::CudaError;

pub(crate) fn select_resource_release_error(primary: CudaError, release: CudaError) -> CudaError {
    CudaError::ResourceReleaseFailed {
        primary: Box::new(primary),
        release: Box::new(release),
    }
}
