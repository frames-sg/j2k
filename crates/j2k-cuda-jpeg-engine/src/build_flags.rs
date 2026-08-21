// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CudaError;

const REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR: &str = "J2K_REQUIRE_CUDA_OXIDE_BUILD";

fn ensure_cuda_oxide_ptx_built(built: bool, display_name: &str) -> Result<(), CudaError> {
    if built {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: format!(
                "{display_name} PTX was not built; set {REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR} on a Linux cuda-oxide host to require it"
            ),
        })
    }
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) fn ensure_jpeg_decode_ptx_built() -> Result<(), CudaError> {
    ensure_cuda_oxide_ptx_built(
        cfg!(j2k_cuda_oxide_jpeg_decode_built),
        "cuda-oxide JPEG decode",
    )
}

#[cfg(feature = "cuda-oxide-jpeg-encode")]
pub(crate) fn ensure_jpeg_encode_ptx_built() -> Result<(), CudaError> {
    ensure_cuda_oxide_ptx_built(
        cfg!(j2k_cuda_oxide_jpeg_encode_built),
        "cuda-oxide JPEG encode",
    )
}
