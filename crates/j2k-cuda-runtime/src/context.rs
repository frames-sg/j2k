#[cfg(feature = "cuda-oxide-copy-u8")]
use crate::build_flags::ensure_cuda_oxide_copy_u8_ptx_built;
#[cfg(feature = "cuda-oxide-j2k-encode")]
use crate::build_flags::ensure_cuda_oxide_j2k_encode_ptx_built;
#[cfg(any(feature = "cuda-oxide-copy-u8", feature = "cuda-oxide-j2k-encode"))]
use crate::kernels;
use crate::{
    build_flags::{ensure_kernel_ptx_built, CUDA_IDWT_TRACE_ENV_VAR},
    bytes::{f32_slice_as_bytes_mut, i32_slice_as_bytes_mut},
    driver::{CuContext, CuFunction, CuModule, Driver},
    error::CudaError,
    execution::{CudaExecutionStats, CudaLaunchMode},
    htj2k_decode::{
        htj2k_decode_needs_zero_fill, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeOutput,
        CudaHtj2kDecodeStageTimings, CudaQueuedHtj2kCleanup,
    },
    htj2k_encode::{
        CudaHtj2kEncodeStageTimings, CudaHtj2kEncodeStatus, CudaHtj2kEncodedCodeBlock,
        CudaHtj2kEncodedCodeBlocks,
    },
    kernels::CudaKernel,
    memory::{pooled_device_buffer, CudaDeviceBuffer, CudaPooledDeviceBuffer},
};
use std::{
    collections::HashMap,
    ffi::{c_char, c_void},
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

    pub(crate) fn kernel_function(&self, kernel: CudaKernel) -> Result<CuFunction, CudaError> {
        self.kernel_function_from_key(CompiledKernelKey::Builtin(kernel))
    }

    #[cfg(feature = "cuda-oxide-copy-u8")]
    pub(crate) fn cuda_oxide_copy_u8_kernel_function(&self) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_copy_u8_ptx_built()?;
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideCopyU8)
    }

    #[cfg(feature = "cuda-oxide-j2k-encode")]
    pub(crate) fn cuda_oxide_j2k_encode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        ensure_cuda_oxide_j2k_encode_ptx_built()?;
        if !kernel.is_j2k_encode_stage() {
            return Err(CudaError::InvalidArgument {
                message: format!("kernel {kernel:?} is not a J2K encode cuda-oxide stage"),
            });
        }
        self.kernel_function_from_key(CompiledKernelKey::CudaOxideJ2kEncode(kernel))
    }

    fn kernel_function_from_key(&self, key: CompiledKernelKey) -> Result<CuFunction, CudaError> {
        match key {
            CompiledKernelKey::Builtin(kernel) => ensure_kernel_ptx_built(kernel)?,
            #[cfg(feature = "cuda-oxide-copy-u8")]
            CompiledKernelKey::CudaOxideCopyU8 => {}
            #[cfg(feature = "cuda-oxide-j2k-encode")]
            CompiledKernelKey::CudaOxideJ2kEncode(_) => {}
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

    /// HTJ2K cleanup segment length in bytes.
    pub fn cleanup_length(&self) -> u32 {
        if self.status.number_of_coding_passes <= 1 {
            self.status.data_len
        } else {
            self.status.reserved0
        }
    }

    /// HTJ2K refinement segment length in bytes.
    pub fn refinement_length(&self) -> u32 {
        if self.status.number_of_coding_passes <= 1 {
            0
        } else {
            self.status.reserved1
        }
    }

    /// Number of coding passes in the encoded payload.
    pub fn num_coding_passes(&self) -> u8 {
        u8::try_from(self.status.number_of_coding_passes).unwrap_or(u8::MAX)
    }

    /// Number of missing most-significant bitplanes.
    pub fn num_zero_bitplanes(&self) -> u8 {
        u8::try_from(self.status.missing_bit_planes).unwrap_or(u8::MAX)
    }

    /// Consume this code block and return its payload range plus segment metadata.
    pub fn into_parts(self) -> (std::ops::Range<usize>, u32, u32, u8, u8) {
        let cleanup_length = if self.status.number_of_coding_passes <= 1 {
            self.status.data_len
        } else {
            self.status.reserved0
        };
        let refinement_length = if self.status.number_of_coding_passes <= 1 {
            0
        } else {
            self.status.reserved1
        };
        (
            self.payload_range,
            cleanup_length,
            refinement_length,
            u8::try_from(self.status.number_of_coding_passes).unwrap_or(u8::MAX),
            u8::try_from(self.status.missing_bit_planes).unwrap_or(u8::MAX),
        )
    }

    /// Kernel status row downloaded after dispatch.
    pub fn status(&self) -> CudaHtj2kEncodeStatus {
        self.status
    }

    /// CUDA execution counters for the encode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// CUDA event timings for the encode dispatch.
    pub fn stage_timings(&self) -> CudaHtj2kEncodeStageTimings {
        self.stage_timings
    }
}

/// Host-visible compact HTJ2K cleanup-pass encode batch produced by one CUDA
/// kernel dispatch.
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

/// Bundled CUDA kernel identifiers that can be preloaded by adapters.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CudaKernelName {
    /// Byte-wise device copy kernel.
    CopyU8,
    /// JPEG 2000 pixel deinterleave/level-shift kernel.
    J2kDeinterleaveToF32,
    /// JPEG 2000 forward reversible color transform kernel.
    J2kForwardRct,
    /// JPEG 2000 forward irreversible color transform kernel.
    J2kForwardIct,
    /// JPEG 2000 forward 5/3 horizontal DWT kernel.
    J2kForwardDwt53Horizontal,
    /// JPEG 2000 forward 5/3 vertical DWT kernel.
    J2kForwardDwt53Vertical,
    /// JPEG 2000 forward 9/7 horizontal DWT kernel.
    J2kForwardDwt97Horizontal,
    /// JPEG 2000 forward 9/7 vertical DWT kernel.
    J2kForwardDwt97Vertical,
    /// JPEG 2000 sub-band quantization kernel.
    J2kQuantizeSubband,
    /// JPEG 2000 strided sub-band quantization kernel.
    J2kQuantizeSubbandStrided,
    /// HTJ2K entropy code-block decode kernel.
    Htj2kDecodeCodeblocks,
    /// HTJ2K cleanup-only decode plus dequantization kernel.
    Htj2kDecodeCodeblocksMultiCleanupDequantize,
    /// JPEG 2000 HTJ2K coefficient dequantization kernel.
    J2kDequantizeHtj2kCodeblocks,
    /// JPEG 2000 HTJ2K multi-buffer coefficient dequantization kernel.
    J2kDequantizeHtj2kCodeblocksMulti,
    /// JPEG 2000 HTJ2K multi-buffer dequantization from cleanup metadata.
    J2kDequantizeHtj2kCleanupJobsMulti,
    /// JPEG 2000 inverse DWT band interleave kernel.
    J2kIdwtInterleave,
    /// JPEG 2000 fused band interleave and reversible 5/3 horizontal lifting kernel.
    J2kIdwtInterleaveHorizontal53Multi,
    /// JPEG 2000 fused band interleave and irreversible 9/7 horizontal lifting kernel.
    J2kIdwtInterleaveHorizontal97Multi,
    /// JPEG 2000 inverse DWT horizontal lifting kernel.
    J2kIdwtHorizontal,
    /// JPEG 2000 inverse 5/3 DWT horizontal lifting kernel.
    J2kIdwtHorizontal53,
    /// JPEG 2000 inverse 9/7 DWT horizontal lifting kernel.
    J2kIdwtHorizontal97,
    /// JPEG 2000 inverse DWT vertical lifting kernel.
    J2kIdwtVertical,
    /// JPEG 2000 reversible 5/3 vertical lifting multi-target kernel.
    J2kIdwtVertical53Multi,
    /// JPEG 2000 irreversible 9/7 vertical lifting multi-target kernel.
    J2kIdwtVertical97Multi,
    /// JPEG 2000 irreversible 9/7 vertical lifting multi-target 4-column kernel.
    J2kIdwtVertical97MultiCols4,
    /// JPEG 2000 inverse 5/3 DWT vertical lifting kernel.
    J2kIdwtVertical53,
    /// JPEG 2000 inverse 9/7 DWT vertical lifting kernel.
    J2kIdwtVertical97,
    /// JPEG 2000 inverse DWT single-decomposition kernel.
    J2kInverseDwtSingle,
    /// JPEG 2000 inverse RCT/ICT color transform kernel.
    J2kInverseMct,
    /// JPEG 2000 grayscale f32-to-Gray8 store kernel.
    J2kStoreGray8,
    /// JPEG 2000 grayscale f32-to-Gray16 store kernel.
    J2kStoreGray16,
    /// JPEG 2000 RGB/RGBA 8-bit store kernel.
    J2kStoreRgb8,
    /// JPEG 2000 fused inverse MCT and RGB/RGBA 8-bit store kernel.
    J2kStoreRgb8Mct,
    /// JPEG 2000 batched fused inverse MCT and RGB/RGBA 8-bit store kernel.
    J2kStoreRgb8MctBatch,
    /// JPEG 2000 RGB/RGBA 16-bit store kernel.
    J2kStoreRgb16,
    /// JPEG 2000 fused inverse MCT and RGB/RGBA 16-bit store kernel.
    J2kStoreRgb16Mct,
    /// HTJ2K single code-block encode kernel.
    Htj2kEncodeCodeblock,
    /// HTJ2K batched code-block encode kernel.
    Htj2kEncodeCodeblocks,
    /// HTJ2K batched multi-input code-block encode kernel.
    Htj2kEncodeCodeblocksMultiInput,
    /// HTJ2K cleanup-only batched multi-input code-block encode kernel.
    Htj2kEncodeCodeblocksMultiInputCleanup,
    /// HTJ2K cleanup-only batched multi-input 64x64 code-block encode kernel.
    Htj2kEncodeCodeblocksMultiInputCleanup64,
    /// HTJ2K batched code-block output compaction kernel.
    Htj2kCompactCodeblocks,
    /// HTJ2K packet header/body assembly kernel.
    Htj2kPacketizeCleanup,
}

impl CudaKernelName {
    pub(crate) fn kernel(self) -> CudaKernel {
        match self {
            Self::CopyU8 => CudaKernel::CopyU8,
            Self::J2kDeinterleaveToF32 => CudaKernel::J2kDeinterleaveToF32,
            Self::J2kForwardRct => CudaKernel::J2kForwardRct,
            Self::J2kForwardIct => CudaKernel::J2kForwardIct,
            Self::J2kForwardDwt53Horizontal => CudaKernel::J2kForwardDwt53Horizontal,
            Self::J2kForwardDwt53Vertical => CudaKernel::J2kForwardDwt53Vertical,
            Self::J2kForwardDwt97Horizontal => CudaKernel::J2kForwardDwt97Horizontal,
            Self::J2kForwardDwt97Vertical => CudaKernel::J2kForwardDwt97Vertical,
            Self::J2kQuantizeSubband => CudaKernel::J2kQuantizeSubband,
            Self::J2kQuantizeSubbandStrided => CudaKernel::J2kQuantizeSubbandStrided,
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
            Self::J2kIdwtHorizontal => CudaKernel::J2kIdwtHorizontal,
            Self::J2kIdwtHorizontal53 => CudaKernel::J2kIdwtHorizontal53,
            Self::J2kIdwtHorizontal97 => CudaKernel::J2kIdwtHorizontal97,
            Self::J2kIdwtVertical => CudaKernel::J2kIdwtVertical,
            Self::J2kIdwtVertical53Multi => CudaKernel::J2kIdwtVertical53Multi,
            Self::J2kIdwtVertical97Multi => CudaKernel::J2kIdwtVertical97Multi,
            Self::J2kIdwtVertical97MultiCols4 => CudaKernel::J2kIdwtVertical97MultiCols4,
            Self::J2kIdwtVertical53 => CudaKernel::J2kIdwtVertical53,
            Self::J2kIdwtVertical97 => CudaKernel::J2kIdwtVertical97,
            Self::J2kInverseDwtSingle => CudaKernel::J2kInverseDwtSingle,
            Self::J2kInverseMct => CudaKernel::J2kInverseMct,
            Self::J2kStoreGray8 => CudaKernel::J2kStoreGray8,
            Self::J2kStoreGray16 => CudaKernel::J2kStoreGray16,
            Self::J2kStoreRgb8 => CudaKernel::J2kStoreRgb8,
            Self::J2kStoreRgb8Mct => CudaKernel::J2kStoreRgb8Mct,
            Self::J2kStoreRgb8MctBatch => CudaKernel::J2kStoreRgb8MctBatch,
            Self::J2kStoreRgb16 => CudaKernel::J2kStoreRgb16,
            Self::J2kStoreRgb16Mct => CudaKernel::J2kStoreRgb16Mct,
            Self::Htj2kEncodeCodeblock => CudaKernel::Htj2kEncodeCodeblock,
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
            Self::J2kDeinterleaveToF32 => "j2k_deinterleave_to_f32",
            Self::J2kForwardRct => "j2k_forward_rct",
            Self::J2kForwardIct => "j2k_forward_ict",
            Self::J2kForwardDwt53Horizontal => "j2k_forward_dwt53_horizontal",
            Self::J2kForwardDwt53Vertical => "j2k_forward_dwt53_vertical",
            Self::J2kForwardDwt97Horizontal => "j2k_forward_dwt97_horizontal",
            Self::J2kForwardDwt97Vertical => "j2k_forward_dwt97_vertical",
            Self::J2kQuantizeSubband => "j2k_quantize_subband",
            Self::J2kQuantizeSubbandStrided => "j2k_quantize_subband_strided",
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
            Self::J2kIdwtHorizontal => "j2k_idwt_horizontal",
            Self::J2kIdwtHorizontal53 => "j2k_idwt_horizontal_53",
            Self::J2kIdwtHorizontal97 => "j2k_idwt_horizontal_97",
            Self::J2kIdwtVertical => "j2k_idwt_vertical",
            Self::J2kIdwtVertical53Multi => "j2k_idwt_vertical_53_multi",
            Self::J2kIdwtVertical97Multi => "j2k_idwt_vertical_97_multi",
            Self::J2kIdwtVertical97MultiCols4 => "j2k_idwt_vertical_97_multi_cols4",
            Self::J2kIdwtVertical53 => "j2k_idwt_vertical_53",
            Self::J2kIdwtVertical97 => "j2k_idwt_vertical_97",
            Self::J2kInverseDwtSingle => "j2k_inverse_dwt_single",
            Self::J2kInverseMct => "j2k_inverse_mct",
            Self::J2kStoreGray8 => "j2k_store_gray8",
            Self::J2kStoreGray16 => "j2k_store_gray16",
            Self::J2kStoreRgb8 => "j2k_store_rgb8",
            Self::J2kStoreRgb8Mct => "j2k_store_rgb8_mct",
            Self::J2kStoreRgb8MctBatch => "j2k_store_rgb8_mct_batch",
            Self::J2kStoreRgb16 => "j2k_store_rgb16",
            Self::J2kStoreRgb16Mct => "j2k_store_rgb16_mct",
            Self::Htj2kEncodeCodeblock => "j2k_htj2k_encode_codeblock",
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaKernelModule {
    pub(crate) kernel: CudaKernelName,
    pub(crate) entrypoint: &'static str,
}

impl CudaKernelModule {
    /// Bundled kernel identifier.
    pub fn kernel(&self) -> CudaKernelName {
        self.kernel
    }

    /// Kernel entry point name.
    pub fn entrypoint(&self) -> &'static str {
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

#[derive(Debug)]
pub(crate) struct CompiledKernel {
    pub(crate) module: CuModule,
    pub(crate) function: CuFunction,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CompiledKernelKey {
    Builtin(CudaKernel),
    #[cfg(feature = "cuda-oxide-copy-u8")]
    CudaOxideCopyU8,
    #[cfg(feature = "cuda-oxide-j2k-encode")]
    CudaOxideJ2kEncode(CudaKernel),
}

impl CompiledKernelKey {
    pub(crate) fn kernel(self) -> CudaKernel {
        match self {
            Self::Builtin(kernel) => kernel,
            #[cfg(feature = "cuda-oxide-copy-u8")]
            Self::CudaOxideCopyU8 => CudaKernel::CopyU8,
            #[cfg(feature = "cuda-oxide-j2k-encode")]
            Self::CudaOxideJ2kEncode(kernel) => kernel,
        }
    }

    pub(crate) fn ptx(self) -> &'static [u8] {
        match self {
            Self::Builtin(kernel) => kernel.ptx(),
            #[cfg(feature = "cuda-oxide-copy-u8")]
            Self::CudaOxideCopyU8 => kernels::cuda_oxide_copy_u8_ptx(),
            #[cfg(feature = "cuda-oxide-j2k-encode")]
            Self::CudaOxideJ2kEncode(_) => kernels::cuda_oxide_j2k_encode_ptx(),
        }
    }

    pub(crate) fn entrypoint(self) -> &'static [u8] {
        self.kernel().entrypoint()
    }
}

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
