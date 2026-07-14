// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-oxide-copy-u8")]
use crate::build_flags::ensure_cuda_oxide_copy_u8_ptx_built;
#[cfg(feature = "cuda-oxide-htj2k-decode")]
use crate::build_flags::ensure_cuda_oxide_htj2k_decode_ptx_built;
#[cfg(feature = "cuda-oxide-htj2k-encode")]
use crate::build_flags::ensure_cuda_oxide_htj2k_encode_ptx_built;
#[cfg(feature = "cuda-oxide-j2k-decode-store")]
use crate::build_flags::ensure_cuda_oxide_j2k_decode_store_ptx_built;
#[cfg(feature = "cuda-oxide-j2k-dequantize")]
use crate::build_flags::ensure_cuda_oxide_j2k_dequantize_ptx_built;
#[cfg(feature = "cuda-oxide-j2k-encode")]
use crate::build_flags::ensure_cuda_oxide_j2k_encode_ptx_built;
#[cfg(feature = "cuda-oxide-j2k-idwt")]
use crate::build_flags::ensure_cuda_oxide_j2k_idwt_ptx_built;
#[cfg(feature = "cuda-oxide-j2k-ml")]
use crate::build_flags::ensure_cuda_oxide_j2k_ml_ptx_built;
#[cfg(feature = "cuda-oxide-jpeg-decode")]
use crate::build_flags::ensure_cuda_oxide_jpeg_decode_ptx_built;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
use crate::build_flags::ensure_cuda_oxide_jpeg_encode_ptx_built;
#[cfg(feature = "cuda-oxide-transcode")]
use crate::build_flags::ensure_cuda_oxide_transcode_ptx_built;
use crate::{driver::CuFunction, error::CudaError, kernels::CudaKernel};

use super::inner::ContextInner;
#[cfg(j2k_cuda_oxide_enabled)]
use super::kernel_cache::CompiledKernelKey;

mod classic;

impl ContextInner {
    #[cfg(feature = "cuda-oxide-j2k-ml")]
    pub(crate) fn cuda_oxide_j2k_ml_kernel_function(&self) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_j2k_ml_ptx_built()?;
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideJ2kMl)
    }

    #[cfg(not(feature = "cuda-oxide-j2k-ml"))]
    #[expect(
        clippy::unused_self,
        reason = "feature-disabled method preserves the enabled dispatch interface"
    )]
    pub(crate) fn cuda_oxide_j2k_ml_kernel_function(&self) -> Result<CuFunction, CudaError> {
        Err(Self::cuda_oxide_feature_missing(
            "j2k-ml",
            "cuda-oxide-j2k-ml",
        ))
    }

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

    #[cfg(feature = "cuda-oxide-j2k-encode")]
    pub(crate) fn cuda_oxide_j2k_encode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_j2k_encode_ptx_built()?;
        if !kernel.is_cuda_oxide_j2k_encode_stage() {
            return Err(CudaError::InvalidArgument {
                message: format!("kernel {kernel:?} is not a J2K encode cuda-oxide stage"),
            });
        }
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideJ2kEncode(kernel))
    }

    #[cfg(not(feature = "cuda-oxide-j2k-encode"))]
    #[expect(
        clippy::unused_self,
        reason = "feature-disabled method preserves the enabled dispatch interface"
    )]
    pub(crate) fn cuda_oxide_j2k_encode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        let _ = kernel;
        Err(Self::cuda_oxide_feature_missing(
            "J2K encode",
            "cuda-oxide-j2k-encode",
        ))
    }

    #[cfg(feature = "cuda-oxide-j2k-decode-store")]
    pub(crate) fn cuda_oxide_j2k_decode_store_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_j2k_decode_store_ptx_built()?;
        if !kernel.is_j2k_decode_store_stage() {
            return Err(CudaError::InvalidArgument {
                message: format!("kernel {kernel:?} is not a J2K decode-store cuda-oxide stage"),
            });
        }
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideJ2kDecodeStore(kernel))
    }

    #[cfg(not(feature = "cuda-oxide-j2k-decode-store"))]
    #[expect(
        clippy::unused_self,
        reason = "feature-disabled method preserves the enabled dispatch interface"
    )]
    pub(crate) fn cuda_oxide_j2k_decode_store_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        let _ = kernel;
        Err(Self::cuda_oxide_feature_missing(
            "J2K decode store",
            "cuda-oxide-j2k-decode-store",
        ))
    }

    #[cfg(feature = "cuda-oxide-j2k-dequantize")]
    pub(crate) fn cuda_oxide_j2k_dequantize_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_j2k_dequantize_ptx_built()?;
        if !kernel.is_j2k_dequantize_stage() {
            return Err(CudaError::InvalidArgument {
                message: format!("kernel {kernel:?} is not a J2K dequantize cuda-oxide stage"),
            });
        }
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideJ2kDequantize(kernel))
    }

    #[cfg(not(feature = "cuda-oxide-j2k-dequantize"))]
    #[expect(
        clippy::unused_self,
        reason = "feature-disabled method preserves the enabled dispatch interface"
    )]
    pub(crate) fn cuda_oxide_j2k_dequantize_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        let _ = kernel;
        Err(Self::cuda_oxide_feature_missing(
            "J2K dequantize",
            "cuda-oxide-j2k-dequantize",
        ))
    }

    #[cfg(feature = "cuda-oxide-j2k-idwt")]
    pub(crate) fn cuda_oxide_j2k_idwt_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_j2k_idwt_ptx_built()?;
        if !kernel.is_j2k_idwt_stage() {
            return Err(CudaError::InvalidArgument {
                message: format!("kernel {kernel:?} is not a J2K IDWT cuda-oxide stage"),
            });
        }
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideJ2kIdwt(kernel))
    }

    #[cfg(not(feature = "cuda-oxide-j2k-idwt"))]
    #[expect(
        clippy::unused_self,
        reason = "feature-disabled method preserves the enabled dispatch interface"
    )]
    pub(crate) fn cuda_oxide_j2k_idwt_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        let _ = kernel;
        Err(Self::cuda_oxide_feature_missing(
            "J2K IDWT",
            "cuda-oxide-j2k-idwt",
        ))
    }

    #[cfg(feature = "cuda-oxide-htj2k-decode")]
    pub(crate) fn cuda_oxide_htj2k_decode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_htj2k_decode_ptx_built()?;
        if !kernel.is_htj2k_decode_stage() {
            return Err(CudaError::InvalidArgument {
                message: format!("kernel {kernel:?} is not an HTJ2K decode cuda-oxide stage"),
            });
        }
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideHtj2kDecode(kernel))
    }

    #[cfg(not(feature = "cuda-oxide-htj2k-decode"))]
    #[expect(
        clippy::unused_self,
        reason = "feature-disabled method preserves the enabled dispatch interface"
    )]
    pub(crate) fn cuda_oxide_htj2k_decode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        let _ = kernel;
        Err(Self::cuda_oxide_feature_missing(
            "HTJ2K decode",
            "cuda-oxide-htj2k-decode",
        ))
    }

    #[cfg(feature = "cuda-oxide-htj2k-encode")]
    pub(crate) fn cuda_oxide_htj2k_encode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_htj2k_encode_ptx_built()?;
        if !kernel.is_htj2k_encode_codeblock_stage() {
            return Err(CudaError::InvalidArgument {
                message: format!("kernel {kernel:?} is not an HTJ2K encode cuda-oxide stage"),
            });
        }
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideHtj2kEncode(kernel))
    }

    #[cfg(not(feature = "cuda-oxide-htj2k-encode"))]
    #[expect(
        clippy::unused_self,
        reason = "feature-disabled method preserves the enabled dispatch interface"
    )]
    pub(crate) fn cuda_oxide_htj2k_encode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        let _ = kernel;
        Err(Self::cuda_oxide_feature_missing(
            "HTJ2K encode",
            "cuda-oxide-htj2k-encode",
        ))
    }

    #[cfg(feature = "cuda-oxide-transcode")]
    pub(crate) fn cuda_oxide_transcode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_transcode_ptx_built()?;
        if !kernel.is_cuda_oxide_transcode_stage() {
            return Err(CudaError::InvalidArgument {
                message: format!("kernel {kernel:?} is not a supported transcode cuda-oxide stage"),
            });
        }
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideTranscode(kernel))
    }

    #[cfg(not(feature = "cuda-oxide-transcode"))]
    #[expect(
        clippy::unused_self,
        reason = "feature-disabled method preserves the enabled dispatch interface"
    )]
    pub(crate) fn cuda_oxide_transcode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        let _ = kernel;
        Err(Self::cuda_oxide_feature_missing(
            "transcode",
            "cuda-oxide-transcode",
        ))
    }

    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    pub(crate) fn cuda_oxide_jpeg_decode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_jpeg_decode_ptx_built()?;
        if !kernel.is_cuda_oxide_jpeg_decode_stage() {
            return Err(CudaError::InvalidArgument {
                message: format!("kernel {kernel:?} is not a supported JPEG cuda-oxide stage"),
            });
        }
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideJpegDecode(kernel))
    }

    #[cfg(not(feature = "cuda-oxide-jpeg-decode"))]
    #[expect(
        clippy::unused_self,
        reason = "feature-disabled method preserves the enabled dispatch interface"
    )]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "JPEG decoder callers are absent when the feature is disabled"
        )
    )]
    pub(crate) fn cuda_oxide_jpeg_decode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        let _ = kernel;
        Err(Self::cuda_oxide_feature_missing(
            "JPEG decode",
            "cuda-oxide-jpeg-decode",
        ))
    }

    #[cfg(feature = "cuda-oxide-jpeg-encode")]
    pub(crate) fn cuda_oxide_jpeg_encode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_jpeg_encode_ptx_built()?;
        if !kernel.is_cuda_oxide_jpeg_encode_stage() {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "kernel {kernel:?} is not a supported JPEG encode cuda-oxide stage"
                ),
            });
        }
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideJpegEncode(kernel))
    }

    #[cfg(not(feature = "cuda-oxide-jpeg-encode"))]
    #[expect(
        clippy::unused_self,
        reason = "feature-disabled method preserves the enabled dispatch interface"
    )]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "JPEG encoder callers are absent when the feature is disabled"
        )
    )]
    pub(crate) fn cuda_oxide_jpeg_encode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        let _ = kernel;
        Err(Self::cuda_oxide_feature_missing(
            "JPEG encode",
            "cuda-oxide-jpeg-encode",
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
        if kernel.is_htj2k_decode_stage() {
            return self.cuda_oxide_htj2k_decode_kernel_function(kernel);
        }
        if kernel.is_j2k_dequantize_stage() {
            return self.cuda_oxide_j2k_dequantize_kernel_function(kernel);
        }
        if kernel.is_j2k_idwt_stage() {
            return self.cuda_oxide_j2k_idwt_kernel_function(kernel);
        }
        if kernel.is_j2k_decode_store_stage() {
            return self.cuda_oxide_j2k_decode_store_kernel_function(kernel);
        }
        if kernel.is_htj2k_encode_codeblock_stage() {
            return self.cuda_oxide_htj2k_encode_kernel_function(kernel);
        }
        if kernel.is_cuda_oxide_j2k_encode_stage() {
            return self.cuda_oxide_j2k_encode_kernel_function(kernel);
        }
        if kernel.is_cuda_oxide_transcode_stage() {
            return self.cuda_oxide_transcode_kernel_function(kernel);
        }
        if kernel.is_cuda_oxide_jpeg_decode_stage() {
            return self.cuda_oxide_jpeg_decode_kernel_function(kernel);
        }
        if kernel.is_cuda_oxide_jpeg_encode_stage() {
            return self.cuda_oxide_jpeg_encode_kernel_function(kernel);
        }
        Err(CudaError::InvalidArgument {
            message: format!("kernel {kernel:?} is not mapped to a CUDA Oxide module family"),
        })
    }

    #[cfg(any(
        not(feature = "cuda-oxide-copy-u8"),
        not(feature = "cuda-oxide-j2k-encode"),
        not(feature = "cuda-oxide-j2k-decode-store"),
        not(feature = "cuda-oxide-j2k-dequantize"),
        not(feature = "cuda-oxide-j2k-idwt"),
        not(feature = "cuda-oxide-htj2k-decode"),
        not(feature = "cuda-oxide-htj2k-encode"),
        not(feature = "cuda-oxide-transcode"),
        not(feature = "cuda-oxide-jpeg-decode"),
        not(feature = "cuda-oxide-jpeg-encode")
    ))]
    fn cuda_oxide_feature_missing(family: &str, feature: &str) -> CudaError {
        CudaError::InvalidArgument {
            message: format!(
                "CUDA Oxide PTX was not built for {family}; enable j2k-cuda-runtime/{feature} or a crate cuda-runtime feature that implies it. CUDA C/PTX fallback is no longer available."
            ),
        }
    }
}
