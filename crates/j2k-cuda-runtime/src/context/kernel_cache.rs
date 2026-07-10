// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(j2k_cuda_oxide_enabled)]
use std::ffi::{c_char, c_void};

use crate::driver::{CuFunction, CuModule};
#[cfg(j2k_cuda_oxide_enabled)]
use crate::kernels;
#[cfg(j2k_cuda_oxide_enabled)]
use crate::{error::CudaError, kernels::CudaKernel};

use super::inner::ContextInner;

#[cfg_attr(
    not(j2k_cuda_oxide_enabled),
    expect(
        dead_code,
        reason = "compiled module/function pair is used only by CUDA Oxide kernels"
    )
)]
#[derive(Debug)]
pub(crate) struct CompiledKernel {
    pub(crate) module: CuModule,
    pub(crate) function: CuFunction,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CompiledKernelKey {
    #[cfg(feature = "cuda-oxide-copy-u8")]
    CudaOxideCopyU8,
    #[cfg(feature = "cuda-oxide-j2k-encode")]
    CudaOxideJ2kEncode(CudaKernel),
    #[cfg(feature = "cuda-oxide-j2k-decode-store")]
    CudaOxideJ2kDecodeStore(CudaKernel),
    #[cfg(feature = "cuda-oxide-j2k-dequantize")]
    CudaOxideJ2kDequantize(CudaKernel),
    #[cfg(feature = "cuda-oxide-j2k-idwt")]
    CudaOxideJ2kIdwt(CudaKernel),
    #[cfg(feature = "cuda-oxide-htj2k-decode")]
    CudaOxideHtj2kDecode(CudaKernel),
    #[cfg(feature = "cuda-oxide-htj2k-encode")]
    CudaOxideHtj2kEncode(CudaKernel),
    #[cfg(feature = "cuda-oxide-transcode")]
    CudaOxideTranscode(CudaKernel),
    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    CudaOxideJpegDecode(CudaKernel),
    #[cfg(feature = "cuda-oxide-jpeg-encode")]
    CudaOxideJpegEncode(CudaKernel),
}

impl ContextInner {
    #[cfg(j2k_cuda_oxide_enabled)]
    pub(in crate::context) fn kernel_function_from_key(
        &self,
        key: CompiledKernelKey,
    ) -> Result<CuFunction, CudaError> {
        match key {
            #[cfg(feature = "cuda-oxide-copy-u8")]
            CompiledKernelKey::CudaOxideCopyU8 => {}
            #[cfg(feature = "cuda-oxide-j2k-encode")]
            CompiledKernelKey::CudaOxideJ2kEncode(_) => {}
            #[cfg(feature = "cuda-oxide-j2k-decode-store")]
            CompiledKernelKey::CudaOxideJ2kDecodeStore(_) => {}
            #[cfg(feature = "cuda-oxide-j2k-dequantize")]
            CompiledKernelKey::CudaOxideJ2kDequantize(_) => {}
            #[cfg(feature = "cuda-oxide-j2k-idwt")]
            CompiledKernelKey::CudaOxideJ2kIdwt(_) => {}
            #[cfg(feature = "cuda-oxide-htj2k-decode")]
            CompiledKernelKey::CudaOxideHtj2kDecode(_) => {}
            #[cfg(feature = "cuda-oxide-htj2k-encode")]
            CompiledKernelKey::CudaOxideHtj2kEncode(_) => {}
            #[cfg(feature = "cuda-oxide-transcode")]
            CompiledKernelKey::CudaOxideTranscode(_) => {}
            #[cfg(feature = "cuda-oxide-jpeg-decode")]
            CompiledKernelKey::CudaOxideJpegDecode(_) => {}
            #[cfg(feature = "cuda-oxide-jpeg-encode")]
            CompiledKernelKey::CudaOxideJpegEncode(_) => {}
        }
        self.set_current()?;
        let mut modules = self
            .modules
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        if let Some(compiled) = modules.get(&key) {
            return Ok(compiled.function);
        }

        let compiled = CompiledKernel::load(self, key)?;
        let function = compiled.function;
        modules.insert(key, compiled);
        Ok(function)
    }
}

#[cfg(j2k_cuda_oxide_enabled)]
impl CompiledKernelKey {
    pub(crate) fn kernel(self) -> CudaKernel {
        match self {
            #[cfg(feature = "cuda-oxide-copy-u8")]
            Self::CudaOxideCopyU8 => CudaKernel::CopyU8,
            #[cfg(feature = "cuda-oxide-j2k-encode")]
            Self::CudaOxideJ2kEncode(kernel) => kernel,
            #[cfg(feature = "cuda-oxide-j2k-decode-store")]
            Self::CudaOxideJ2kDecodeStore(kernel) => kernel,
            #[cfg(feature = "cuda-oxide-j2k-dequantize")]
            Self::CudaOxideJ2kDequantize(kernel) => kernel,
            #[cfg(feature = "cuda-oxide-j2k-idwt")]
            Self::CudaOxideJ2kIdwt(kernel) => kernel,
            #[cfg(feature = "cuda-oxide-htj2k-decode")]
            Self::CudaOxideHtj2kDecode(kernel) => kernel,
            #[cfg(feature = "cuda-oxide-htj2k-encode")]
            Self::CudaOxideHtj2kEncode(kernel) => kernel,
            #[cfg(feature = "cuda-oxide-transcode")]
            Self::CudaOxideTranscode(kernel) => kernel,
            #[cfg(feature = "cuda-oxide-jpeg-decode")]
            Self::CudaOxideJpegDecode(kernel) => kernel,
            #[cfg(feature = "cuda-oxide-jpeg-encode")]
            Self::CudaOxideJpegEncode(kernel) => kernel,
        }
    }

    pub(crate) fn ptx(self) -> &'static [u8] {
        match self {
            #[cfg(feature = "cuda-oxide-copy-u8")]
            Self::CudaOxideCopyU8 => kernels::cuda_oxide_copy_u8_ptx(),
            #[cfg(feature = "cuda-oxide-j2k-encode")]
            Self::CudaOxideJ2kEncode(_) => kernels::cuda_oxide_j2k_encode_ptx(),
            #[cfg(feature = "cuda-oxide-j2k-decode-store")]
            Self::CudaOxideJ2kDecodeStore(_) => kernels::cuda_oxide_j2k_decode_store_ptx(),
            #[cfg(feature = "cuda-oxide-j2k-dequantize")]
            Self::CudaOxideJ2kDequantize(_) => kernels::cuda_oxide_j2k_dequantize_ptx(),
            #[cfg(feature = "cuda-oxide-j2k-idwt")]
            Self::CudaOxideJ2kIdwt(_) => kernels::cuda_oxide_j2k_idwt_ptx(),
            #[cfg(feature = "cuda-oxide-htj2k-decode")]
            Self::CudaOxideHtj2kDecode(_) => kernels::cuda_oxide_htj2k_decode_ptx(),
            #[cfg(feature = "cuda-oxide-htj2k-encode")]
            Self::CudaOxideHtj2kEncode(_) => kernels::cuda_oxide_htj2k_encode_ptx(),
            #[cfg(feature = "cuda-oxide-transcode")]
            Self::CudaOxideTranscode(_) => kernels::cuda_oxide_transcode_ptx(),
            #[cfg(feature = "cuda-oxide-jpeg-decode")]
            Self::CudaOxideJpegDecode(_) => kernels::cuda_oxide_jpeg_decode_ptx(),
            #[cfg(feature = "cuda-oxide-jpeg-encode")]
            Self::CudaOxideJpegEncode(_) => kernels::cuda_oxide_jpeg_encode_ptx(),
        }
    }

    pub(crate) fn entrypoint(self) -> &'static [u8] {
        self.kernel().entrypoint()
    }
}

#[cfg(j2k_cuda_oxide_enabled)]
impl CompiledKernel {
    pub(crate) fn load(context: &ContextInner, key: CompiledKernelKey) -> Result<Self, CudaError> {
        context.set_current()?;
        let mut module = std::ptr::null_mut();
        // SAFETY: image is a NUL-terminated PTX string. CUDA copies or parses
        // it during module load, and the context cache unloads the module on
        // context drop.
        context.driver.check("cuModuleLoadData", unsafe {
            (context.driver.cu_module_load_data)(
                &raw mut module,
                key.ptx().as_ptr().cast::<c_void>(),
            )
        })?;
        let mut function = std::ptr::null_mut();
        // SAFETY: name is a NUL-terminated kernel symbol in this module.
        context.driver.check("cuModuleGetFunction", unsafe {
            (context.driver.cu_module_get_function)(
                &raw mut function,
                module,
                key.entrypoint().as_ptr().cast::<c_char>(),
            )
        })?;
        Ok(Self { module, function })
    }
}

// SAFETY: CompiledKernel stores opaque CUDA module/function handles. Lifetime
// and unloading are coordinated by ContextInner's module cache mutex.
unsafe impl Send for CompiledKernel {}
