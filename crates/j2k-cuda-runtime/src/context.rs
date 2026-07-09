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
#[cfg(feature = "cuda-oxide-jpeg-decode")]
use crate::build_flags::ensure_cuda_oxide_jpeg_decode_ptx_built;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
use crate::build_flags::ensure_cuda_oxide_jpeg_encode_ptx_built;
#[cfg(feature = "cuda-oxide-transcode")]
use crate::build_flags::ensure_cuda_oxide_transcode_ptx_built;
#[cfg(j2k_cuda_oxide_enabled)]
use crate::kernels;
use crate::{
    build_flags::CUDA_IDWT_TRACE_ENV_VAR,
    bytes::{f32_slice_as_bytes_mut, i32_slice_as_bytes_mut},
    driver::{CuContext, CuFunction, CuModule, Driver},
    error::CudaError,
    execution::{CudaExecutionStats, CudaLaunchMode},
    htj2k_decode::{
        htj2k_decode_needs_zero_fill, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeOutput,
        CudaHtj2kDecodeStageTimings, CudaQueuedHtj2kCleanup,
    },
    htj2k_encode::{
        htj2k_encoded_cleanup_length, htj2k_encoded_num_coding_passes,
        htj2k_encoded_num_zero_bitplanes, htj2k_encoded_refinement_length,
        CudaHtj2kEncodeStageTimings, CudaHtj2kEncodeStatus, CudaHtj2kEncodedCodeBlock,
        CudaHtj2kEncodedCodeBlocks,
    },
    kernels::CudaKernel,
    memory::{pooled_device_buffer, CudaDeviceBuffer, CudaPooledDeviceBuffer},
};
#[cfg(j2k_cuda_oxide_enabled)]
use std::ffi::{c_char, c_void};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

pub(crate) struct ContextInner {
    pub(crate) driver: Driver,
    pub(crate) context: CuContext,
    pub(crate) modules: Mutex<HashMap<CompiledKernelKey, CompiledKernel>>,
    pub(crate) pinned_upload_staging: Mutex<Vec<PinnedUploadStaging>>,
}

pub(crate) struct PinnedUploadStaging {
    pub(crate) ptr: *mut u8,
    pub(crate) len: usize,
}

impl PinnedUploadStaging {
    pub(crate) fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: ptr is a live pinned allocation of len bytes.
            unsafe { std::slice::from_raw_parts(self.ptr.cast_const(), self.len) }
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: ptr is uniquely borrowed through &mut self and covers len
            // bytes allocated by CUDA.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    pub(crate) fn free(self, driver: &Driver) -> Result<(), CudaError> {
        if self.ptr.is_null() {
            return Ok(());
        }
        // SAFETY: ptr was returned by cuMemHostAlloc for this process.
        driver.check("cuMemFreeHost", unsafe {
            (driver.cu_mem_free_host)(self.ptr.cast())
        })
    }
}

// SAFETY: The pinned allocation is owned by this value. Mutable access requires
// &mut self, and freeing is explicitly coordinated by the owning CudaContext.
unsafe impl Send for PinnedUploadStaging {}

impl ContextInner {
    pub(crate) fn set_current(&self) -> Result<(), CudaError> {
        // SAFETY: context is created by cuCtxCreate_v2 and remains valid while
        // ContextInner is alive.
        self.driver.check("cuCtxSetCurrent", unsafe {
            (self.driver.cu_ctx_set_current)(self.context)
        })
    }

    #[cfg(feature = "cuda-oxide-copy-u8")]
    pub(crate) fn cuda_oxide_copy_u8_kernel_function(&self) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_copy_u8_ptx_built()?;
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideCopyU8)
    }

    #[cfg(not(feature = "cuda-oxide-copy-u8"))]
    #[allow(clippy::unused_self)]
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
    #[allow(clippy::unused_self)]
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
    #[allow(clippy::unused_self)]
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
    #[allow(clippy::unused_self)]
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
    #[allow(clippy::unused_self)]
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
    #[allow(clippy::unused_self)]
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
    #[allow(clippy::unused_self)]
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
    #[allow(clippy::unused_self)]
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
    #[allow(clippy::unused_self)]
    #[allow(dead_code)]
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
    #[allow(clippy::unused_self)]
    #[allow(dead_code)]
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

    #[cfg(j2k_cuda_oxide_enabled)]
    fn kernel_function_from_key(&self, key: CompiledKernelKey) -> Result<CuFunction, CudaError> {
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

impl Drop for ContextInner {
    fn drop(&mut self) {
        if !self.context.is_null() {
            let _ = self.set_current();
            let pinned_upload_staging = match self.pinned_upload_staging.get_mut() {
                Ok(pinned_upload_staging) => pinned_upload_staging,
                Err(poisoned) => poisoned.into_inner(),
            };
            for staging in pinned_upload_staging.drain(..) {
                let _ = staging.free(&self.driver);
            }
            let modules = match self.modules.get_mut() {
                Ok(modules) => modules,
                Err(poisoned) => poisoned.into_inner(),
            };
            for compiled in modules.drain().map(|(_, compiled)| compiled) {
                // SAFETY: modules were loaded into this CUDA context. Drop
                // cannot surface errors, so cleanup failures are ignored.
                let _ = unsafe { (self.driver.cu_module_unload)(compiled.module) };
            }
            // SAFETY: context was created by this ContextInner and cached
            // modules have already been unloaded.
            let _ = unsafe { (self.driver.cu_ctx_destroy)(self.context) };
        }
    }
}

// SAFETY: ContextInner owns an opaque CUDA context handle and synchronizes its
// Rust-side mutable caches with mutexes.
unsafe impl Send for ContextInner {}

// SAFETY: All shared Rust state is mutex-protected, and CUDA operations set the
// current context before touching context-owned resources.
unsafe impl Sync for ContextInner {}

/// CUDA driver context shared by J2K CUDA adapter crates.
#[derive(Clone)]
pub struct CudaContext {
    pub(crate) inner: Arc<ContextInner>,
}

/// Host-visible compact HTJ2K cleanup-pass encode metadata for one code block.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaHtj2kCompactEncodedCodeBlock {
    pub(crate) payload_range: std::ops::Range<usize>,
    pub(crate) status: CudaHtj2kEncodeStatus,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) stage_timings: CudaHtj2kEncodeStageTimings,
}

impl CudaHtj2kCompactEncodedCodeBlock {
    /// Encoded cleanup-pass payload range in the batch payload.
    pub fn payload_range(&self) -> std::ops::Range<usize> {
        self.payload_range.clone()
    }

    impl_cuda_htj2k_encoded_status_accessors!();

    /// Consume this code block and return its payload range plus segment metadata.
    pub fn into_parts(self) -> (std::ops::Range<usize>, u32, u32, u8, u8) {
        (
            self.payload_range,
            htj2k_encoded_cleanup_length(self.status),
            htj2k_encoded_refinement_length(self.status),
            htj2k_encoded_num_coding_passes(self.status),
            htj2k_encoded_num_zero_bitplanes(self.status),
        )
    }
}

/// Host-visible compact HTJ2K cleanup-pass encode batch produced by one CUDA
/// kernel dispatch.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaHtj2kCompactEncodedCodeBlocks {
    pub(crate) payload: Vec<u8>,
    pub(crate) code_blocks: Vec<CudaHtj2kCompactEncodedCodeBlock>,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) stage_timings: CudaHtj2kEncodeStageTimings,
}

impl CudaHtj2kCompactEncodedCodeBlocks {
    /// Compact encoded payload shared by all code-block ranges.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Encoded cleanup code-block metadata, in submitted-job order.
    pub fn code_blocks(&self) -> &[CudaHtj2kCompactEncodedCodeBlock] {
        &self.code_blocks
    }

    /// Consume the batch and return its payload plus per-code-block metadata.
    pub fn into_payload_and_code_blocks(self) -> (Vec<u8>, Vec<CudaHtj2kCompactEncodedCodeBlock>) {
        (self.payload, self.code_blocks)
    }

    /// CUDA execution counters for the batch encode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// CUDA event timings for the batch encode dispatch.
    pub fn stage_timings(&self) -> CudaHtj2kEncodeStageTimings {
        self.stage_timings
    }

    pub(crate) fn into_owned_code_blocks(self) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let Self {
            payload,
            code_blocks,
            execution,
            stage_timings,
        } = self;
        let code_blocks = code_blocks
            .into_iter()
            .map(|block| {
                let CudaHtj2kCompactEncodedCodeBlock {
                    payload_range,
                    status,
                    execution,
                    stage_timings,
                } = block;
                if payload_range.start > payload_range.end || payload_range.end > payload.len() {
                    return Err(CudaError::LengthTooLarge {
                        len: payload_range.end,
                    });
                }
                Ok(CudaHtj2kEncodedCodeBlock {
                    data: payload[payload_range].to_vec(),
                    status,
                    execution,
                    stage_timings,
                })
            })
            .collect::<Result<Vec<_>, CudaError>>()?;

        Ok(CudaHtj2kEncodedCodeBlocks {
            code_blocks,
            execution,
            stage_timings,
        })
    }
}

pub(crate) const HTJ2K_UVLC_ENCODE_TABLE_BYTES: usize = 75 * 6;

impl CudaContext {
    /// Create a context for the system default CUDA device.
    pub fn system_default() -> Result<Self, CudaError> {
        let driver = Driver::load()?;

        // SAFETY: cuInit is the CUDA Driver API process initializer.
        driver.check("cuInit", unsafe { (driver.cu_init)(0) })?;

        let mut count = 0;
        // SAFETY: CUDA writes one integer device count to the provided pointer.
        driver.check("cuDeviceGetCount", unsafe {
            (driver.cu_device_get_count)(&raw mut count)
        })?;
        if count <= 0 {
            return Err(CudaError::Unavailable {
                message: "no CUDA devices reported by driver".to_string(),
            });
        }

        let mut device = 0;
        // SAFETY: device 0 is valid when count is greater than zero.
        driver.check("cuDeviceGet", unsafe {
            (driver.cu_device_get)(&raw mut device, 0)
        })?;

        let mut context = std::ptr::null_mut();
        // SAFETY: CUDA writes a newly-created context handle for a valid device.
        driver.check("cuCtxCreate_v2", unsafe {
            (driver.cu_ctx_create)(&raw mut context, 0, device)
        })?;

        Ok(Self {
            inner: Arc::new(ContextInner {
                driver,
                context,
                modules: Mutex::new(HashMap::new()),
                pinned_upload_staging: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Dequantize HTJ2K cleanup outputs using the metadata buffer already held
    /// live by a queued cleanup launch.
    #[doc(hidden)]
    pub fn j2k_dequantize_queued_htj2k_cleanup_with_pool(
        &self,
        cleanup: &CudaQueuedHtj2kCleanup,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.inner.set_current()?;
        if cleanup.status_count == 0 {
            return Ok(CudaExecutionStats::default());
        }
        let Some(jobs_buffer) = cleanup.resources.first() else {
            return Err(CudaError::InvalidArgument {
                message: "queued HTJ2K cleanup has no metadata buffer".to_string(),
            });
        };
        self.launch_j2k_dequantize_htj2k_cleanup_jobs_multi(
            pooled_device_buffer(jobs_buffer)?,
            cleanup.status_count,
            CudaLaunchMode::Sync,
        )?;
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }

    pub(crate) fn decode_empty_htj2k_codeblocks(
        &self,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.inner.set_current()?;
        let output_bytes = output_words
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: output_words })?;
        let coefficients = self.allocate(output_bytes)?;
        if htj2k_decode_needs_zero_fill(jobs, output_words)? {
            self.memset_d32(&coefficients, 0, output_words)?;
        }
        Ok(CudaHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats::default(),
            statuses: Vec::new(),
            stage_timings: CudaHtj2kDecodeStageTimings::default(),
        })
    }
}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaContext").finish_non_exhaustive()
    }
}

/// Bundled CUDA kernel identifiers that can be preloaded by runtime internals.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaKernelName {
    CopyU8,
    Htj2kDecodeCodeblocks,
    Htj2kDecodeCodeblocksMultiCleanupDequantize,
    J2kDequantizeHtj2kCodeblocks,
    J2kDequantizeHtj2kCodeblocksMulti,
    J2kDequantizeHtj2kCleanupJobsMulti,
    J2kIdwtInterleave,
    J2kIdwtInterleaveHorizontal53Multi,
    J2kIdwtInterleaveHorizontal97Multi,
    J2kIdwtHorizontal53,
    J2kIdwtHorizontal97,
    J2kIdwtVertical53Multi,
    J2kIdwtVertical97Multi,
    J2kIdwtVertical97MultiCols4,
    J2kIdwtVertical53,
    J2kIdwtVertical97,
    J2kInverseMct,
    J2kStoreGray8,
    J2kStoreGray16,
    J2kStoreRgb8,
    J2kStoreRgb8MctBatch,
    J2kStoreRgb16,
    J2kStoreRgb16Mct,
    Htj2kEncodeCodeblocks,
    Htj2kEncodeCodeblocksMultiInput,
    Htj2kEncodeCodeblocksMultiInputCleanup,
    Htj2kEncodeCodeblocksMultiInputCleanup64,
    Htj2kCompactCodeblocks,
    Htj2kPacketizeCleanup,
}

#[cfg(test)]
impl CudaKernelName {
    pub(crate) fn kernel(self) -> CudaKernel {
        match self {
            Self::CopyU8 => CudaKernel::CopyU8,
            Self::Htj2kDecodeCodeblocks => CudaKernel::Htj2kDecodeCodeblocks,
            Self::Htj2kDecodeCodeblocksMultiCleanupDequantize => {
                CudaKernel::Htj2kDecodeCodeblocksMultiCleanupDequantize
            }
            Self::J2kDequantizeHtj2kCodeblocks => CudaKernel::J2kDequantizeHtj2kCodeblocks,
            Self::J2kDequantizeHtj2kCodeblocksMulti => {
                CudaKernel::J2kDequantizeHtj2kCodeblocksMulti
            }
            Self::J2kDequantizeHtj2kCleanupJobsMulti => {
                CudaKernel::J2kDequantizeHtj2kCleanupJobsMulti
            }
            Self::J2kIdwtInterleave => CudaKernel::J2kIdwtInterleave,
            Self::J2kIdwtInterleaveHorizontal53Multi => {
                CudaKernel::J2kIdwtInterleaveHorizontal53Multi
            }
            Self::J2kIdwtInterleaveHorizontal97Multi => {
                CudaKernel::J2kIdwtInterleaveHorizontal97Multi
            }
            Self::J2kIdwtHorizontal53 => CudaKernel::J2kIdwtHorizontal53,
            Self::J2kIdwtHorizontal97 => CudaKernel::J2kIdwtHorizontal97,
            Self::J2kIdwtVertical53Multi => CudaKernel::J2kIdwtVertical53Multi,
            Self::J2kIdwtVertical97Multi => CudaKernel::J2kIdwtVertical97Multi,
            Self::J2kIdwtVertical97MultiCols4 => CudaKernel::J2kIdwtVertical97MultiCols4,
            Self::J2kIdwtVertical53 => CudaKernel::J2kIdwtVertical53,
            Self::J2kIdwtVertical97 => CudaKernel::J2kIdwtVertical97,
            Self::J2kInverseMct => CudaKernel::J2kInverseMct,
            Self::J2kStoreGray8 => CudaKernel::J2kStoreGray8,
            Self::J2kStoreGray16 => CudaKernel::J2kStoreGray16,
            Self::J2kStoreRgb8 => CudaKernel::J2kStoreRgb8,
            Self::J2kStoreRgb8MctBatch => CudaKernel::J2kStoreRgb8MctBatch,
            Self::J2kStoreRgb16 => CudaKernel::J2kStoreRgb16,
            Self::J2kStoreRgb16Mct => CudaKernel::J2kStoreRgb16Mct,
            Self::Htj2kEncodeCodeblocks => CudaKernel::Htj2kEncodeCodeblocks,
            Self::Htj2kEncodeCodeblocksMultiInput => CudaKernel::Htj2kEncodeCodeblocksMultiInput,
            Self::Htj2kEncodeCodeblocksMultiInputCleanup => {
                CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup
            }
            Self::Htj2kEncodeCodeblocksMultiInputCleanup64 => {
                CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup64
            }
            Self::Htj2kCompactCodeblocks => CudaKernel::Htj2kCompactCodeblocks,
            Self::Htj2kPacketizeCleanup => CudaKernel::Htj2kPacketizeCleanup,
        }
    }

    pub(crate) fn entrypoint(self) -> &'static str {
        match self {
            Self::CopyU8 => "j2k_copy_u8",
            Self::Htj2kDecodeCodeblocks => "j2k_htj2k_decode_codeblocks",
            Self::Htj2kDecodeCodeblocksMultiCleanupDequantize => {
                "j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize"
            }
            Self::J2kDequantizeHtj2kCodeblocks => "j2k_dequantize_htj2k_codeblocks",
            Self::J2kDequantizeHtj2kCodeblocksMulti => "j2k_dequantize_htj2k_codeblocks_multi",
            Self::J2kDequantizeHtj2kCleanupJobsMulti => "j2k_dequantize_htj2k_cleanup_jobs_multi",
            Self::J2kIdwtInterleave => "j2k_idwt_interleave",
            Self::J2kIdwtInterleaveHorizontal53Multi => "j2k_idwt_interleave_horizontal_53_multi",
            Self::J2kIdwtInterleaveHorizontal97Multi => "j2k_idwt_interleave_horizontal_97_multi",
            Self::J2kIdwtHorizontal53 => "j2k_idwt_horizontal_53",
            Self::J2kIdwtHorizontal97 => "j2k_idwt_horizontal_97",
            Self::J2kIdwtVertical53Multi => "j2k_idwt_vertical_53_multi",
            Self::J2kIdwtVertical97Multi => "j2k_idwt_vertical_97_multi",
            Self::J2kIdwtVertical97MultiCols4 => "j2k_idwt_vertical_97_multi_cols4",
            Self::J2kIdwtVertical53 => "j2k_idwt_vertical_53",
            Self::J2kIdwtVertical97 => "j2k_idwt_vertical_97",
            Self::J2kInverseMct => "j2k_inverse_mct",
            Self::J2kStoreGray8 => "j2k_store_gray8",
            Self::J2kStoreGray16 => "j2k_store_gray16",
            Self::J2kStoreRgb8 => "j2k_store_rgb8",
            Self::J2kStoreRgb8MctBatch => "j2k_store_rgb8_mct_batch",
            Self::J2kStoreRgb16 => "j2k_store_rgb16",
            Self::J2kStoreRgb16Mct => "j2k_store_rgb16_mct",
            Self::Htj2kEncodeCodeblocks => "j2k_htj2k_encode_codeblocks",
            Self::Htj2kEncodeCodeblocksMultiInput => "j2k_htj2k_encode_codeblocks_multi_input",
            Self::Htj2kEncodeCodeblocksMultiInputCleanup => {
                "j2k_htj2k_encode_codeblocks_multi_input_cleanup"
            }
            Self::Htj2kEncodeCodeblocksMultiInputCleanup64 => {
                "j2k_htj2k_encode_codeblocks_multi_input_cleanup_64"
            }
            Self::Htj2kCompactCodeblocks => "j2k_htj2k_compact_codeblocks",
            Self::Htj2kPacketizeCleanup => "j2k_htj2k_packetize_cleanup",
        }
    }
}

/// Metadata for a preloaded CUDA kernel module entry point.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaKernelModule {
    pub(crate) entrypoint: &'static str,
}

#[cfg(test)]
impl CudaKernelModule {
    pub(crate) fn entrypoint(&self) -> &'static str {
        self.entrypoint
    }
}

pub(crate) fn cuda_idwt_trace_enabled() -> bool {
    std::env::var_os(CUDA_IDWT_TRACE_ENV_VAR).is_some()
}

impl CudaContext {
    pub(crate) fn download_i32_band(
        buffer: &CudaDeviceBuffer,
        count: usize,
    ) -> Result<Vec<i32>, CudaError> {
        let mut out = vec![0i32; count];
        if count != 0 {
            buffer.copy_to_host(i32_slice_as_bytes_mut(&mut out))?;
        }
        Ok(out)
    }
}

impl CudaContext {
    pub(crate) fn download_f32_band(
        buffer: &CudaDeviceBuffer,
        count: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let mut out = vec![0f32; count];
        if count != 0 {
            buffer.copy_to_host(f32_slice_as_bytes_mut(&mut out))?;
        }
        Ok(out)
    }

    pub(crate) fn download_pooled_f32_band(
        buffer: &CudaPooledDeviceBuffer,
        count: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let mut out = vec![0f32; count];
        if count != 0 {
            buffer.copy_to_host(f32_slice_as_bytes_mut(&mut out))?;
        }
        Ok(out)
    }
}

#[cfg_attr(not(j2k_cuda_oxide_enabled), allow(dead_code))]
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
