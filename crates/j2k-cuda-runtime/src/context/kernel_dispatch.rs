// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-oxide-copy-u8")]
use crate::build_flags::ensure_cuda_oxide_copy_u8_ptx_built;
#[cfg(test)]
use crate::kernels::CudaKernel;
use crate::{driver::CuFunction, error::CudaError};

use super::inner::ContextInner;
#[cfg(j2k_cuda_oxide_enabled)]
use super::kernel_cache::CompiledKernelKey;

impl ContextInner {
    #[cfg(feature = "cuda-oxide-copy-u8")]
    pub(crate) fn cuda_oxide_copy_u8_kernel_function(&self) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_copy_u8_ptx_built()?;
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideCopyU8)
    }

    #[cfg(not(feature = "cuda-oxide-copy-u8"))]
    #[expect(
        clippy::unused_self,
        reason = "feature-disabled method preserves the enabled dispatch interface"
    )]
    pub(crate) fn cuda_oxide_copy_u8_kernel_function(&self) -> Result<CuFunction, CudaError> {
        Err(Self::cuda_oxide_feature_missing(
            "CopyU8",
            "cuda-oxide-copy-u8",
        ))
    }

    #[cfg(test)]
    pub(crate) fn cuda_oxide_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        if kernel == CudaKernel::CopyU8 {
            return self.cuda_oxide_copy_u8_kernel_function();
        }
        Err(CudaError::InvalidArgument {
            message: format!("kernel {kernel:?} is not mapped to a CUDA Oxide module family"),
        })
    }

    #[cfg(not(feature = "cuda-oxide-copy-u8"))]
    fn cuda_oxide_feature_missing(family: &str, feature: &str) -> CudaError {
        CudaError::InvalidArgument {
            message: format!(
                "CUDA Oxide PTX was not built for {family}; enable j2k-cuda-runtime/{feature} or a crate cuda-runtime feature that implies it. CUDA C/PTX fallback is no longer available."
            ),
        }
    }
}
