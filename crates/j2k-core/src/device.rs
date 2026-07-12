// SPDX-License-Identifier: MIT OR Apache-2.0
// j2k-coverage: shared-accelerator-host

use crate::backend::BackendRequest;

/// Validate a backend request for adapters that support CPU fallback and CUDA output.
#[doc(hidden)]
pub const fn validate_cuda_surface_backend_request(
    request: BackendRequest,
) -> Result<(), BackendRequest> {
    match request {
        BackendRequest::Cpu | BackendRequest::Auto | BackendRequest::Cuda => Ok(()),
        BackendRequest::Metal => Err(request),
    }
}
