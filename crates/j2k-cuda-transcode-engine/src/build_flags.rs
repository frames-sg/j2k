// SPDX-License-Identifier: MIT OR Apache-2.0

use std::os::raw::c_uint;

pub(crate) const PINNED_POOLED_I16_UPLOAD_MAX_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const DWT97_ROW_LIFT_MAX_WIDTH: i32 = 1024;
pub(crate) const DWT97_ROW_LIFT_COOP_THREADS_X: c_uint = 128;
pub(crate) const DWT97_ROW_LIFT_COOP_ROWS_PER_BLOCK: c_uint = 4;
/// Whether the coefficient-domain transcode CUDA Oxide kernels were compiled.
#[must_use]
pub fn transcode_kernels_built() -> bool {
    #[cfg(feature = "cuda-oxide-transcode")]
    {
        cfg!(j2k_cuda_oxide_transcode_built)
    }
    #[cfg(not(feature = "cuda-oxide-transcode"))]
    {
        false
    }
}

pub(crate) fn ensure_transcode_ptx_built() -> Result<(), j2k_cuda_runtime::CudaError> {
    if transcode_kernels_built() {
        Ok(())
    } else {
        Err(j2k_cuda_runtime::CudaError::InvalidArgument {
            message: "CUDA Oxide transcode PTX was not built; enable j2k-cuda-transcode-engine/cuda-oxide-transcode or an adapter feature that implies it, and use J2K_REQUIRE_CUDA_OXIDE_BUILD=1 on CUDA hosts".to_string(),
        })
    }
}
