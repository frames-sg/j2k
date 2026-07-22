// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{driver::CuFunction, error::CudaError, kernels::CudaKernel};

#[cfg(feature = "cuda-oxide-j2k-classic-decode")]
use super::CompiledKernelKey;
use super::ContextInner;
#[cfg(feature = "cuda-oxide-j2k-classic-decode")]
use crate::build_flags::ensure_cuda_oxide_j2k_classic_decode_ptx_built;

impl ContextInner {
    #[cfg(feature = "cuda-oxide-j2k-classic-decode")]
    pub(crate) fn cuda_oxide_j2k_classic_decode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_j2k_classic_decode_ptx_built()?;
        if !kernel.is_j2k_classic_decode_stage() {
            return Err(CudaError::InvalidArgument {
                message: format!("kernel {kernel:?} is not a classic J2K decode stage"),
            });
        }
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideJ2kClassicDecode(kernel))
    }

    #[cfg(not(feature = "cuda-oxide-j2k-classic-decode"))]
    pub(crate) fn cuda_oxide_j2k_classic_decode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        let _ = (self, kernel);
        Err(Self::cuda_oxide_feature_missing(
            "classic J2K decode",
            "cuda-oxide-j2k-classic-decode",
        ))
    }
}
