// SPDX-License-Identifier: Apache-2.0

//! Thin CUDA Driver API runtime used by signinum CUDA adapter crates.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![warn(unreachable_pub)]

mod kernels;

use std::{
    collections::{BTreeMap, HashMap},
    ffi::c_void,
    os::raw::{c_char, c_int, c_uint},
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

#[cfg(signinum_cuda_jpeg_decode_ptx_built)]
use kernels::CudaLaunchGeometry;
use kernels::{
    copy_u8_launch_geometry, htj2k_codeblock_launch_geometry,
    htj2k_codeblock_sample_launch_geometry, htj2k_encode_codeblock_launch_geometry,
    htj2k_packetize_launch_geometry, j2k_dwt53_launch_geometry, j2k_forward_rct_launch_geometry,
    j2k_idwt_multi_1d_launch_geometry, j2k_idwt_multi_coop_axis_launch_geometry,
    j2k_idwt_multi_coop_columns_launch_geometry, j2k_idwt_multi_coop_launch_geometry,
    j2k_store_batch_launch_geometry, CudaKernel,
};
use libloading::Library;

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuDevicePtr = u64;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;
type CuEvent = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const PINNED_UPLOAD_STAGING_POOL_MAX: usize = 8;
const PINNED_POOLED_I16_UPLOAD_MAX_BYTES: usize = 4 * 1024 * 1024;
const DWT97_ROW_LIFT_MAX_WIDTH: i32 = 1024;
const DWT97_ROW_LIFT_COOP_THREADS_X: c_uint = 128;
const DWT97_ROW_LIFT_COOP_ROWS_PER_BLOCK: c_uint = 4;
const CUDA_IDWT_TRACE_ENV_VAR: &str = "SIGNINUM_CUDA_IDWT_TRACE";
const DWT97_FUSED_COLUMN_QUANTIZE_DISABLE_ENV_VAR: &str =
    "SIGNINUM_CUDA_DISABLE_DWT97_FUSED_COLUMN_QUANTIZE";
static CUDA_STAGE_TIMINGS_DISABLED: OnceLock<bool> = OnceLock::new();
static DWT97_FUSED_COLUMN_QUANTIZE_DISABLED: OnceLock<bool> = OnceLock::new();

fn cuda_stage_timings_disabled() -> bool {
    *CUDA_STAGE_TIMINGS_DISABLED
        .get_or_init(|| std::env::var_os("SIGNINUM_CUDA_DISABLE_STAGE_TIMINGS").is_some())
}

fn dwt97_fused_column_quantize_disabled() -> bool {
    *DWT97_FUSED_COLUMN_QUANTIZE_DISABLED
        .get_or_init(|| std::env::var_os(DWT97_FUSED_COLUMN_QUANTIZE_DISABLE_ENV_VAR).is_some())
}

type CuInit = unsafe extern "C" fn(c_uint) -> CuResult;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
type CuCtxCreate = unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult;
type CuCtxDestroy = unsafe extern "C" fn(CuContext) -> CuResult;
type CuCtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type CuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;
type CuMemFree = unsafe extern "C" fn(CuDevicePtr) -> CuResult;
type CuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CuResult;
type CuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CuResult;
type CuMemcpyHtoD = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
type CuMemsetD32 = unsafe extern "C" fn(CuDevicePtr, c_uint, usize) -> CuResult;
type CuGetErrorName = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;
type CuModuleLoadData = unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult;
type CuModuleUnload = unsafe extern "C" fn(CuModule) -> CuResult;
type CuModuleGetFunction =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type CuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type CuCtxSynchronize = unsafe extern "C" fn() -> CuResult;
type CuStreamCreate = unsafe extern "C" fn(*mut CuStream, c_uint) -> CuResult;
type CuStreamDestroy = unsafe extern "C" fn(CuStream) -> CuResult;
type CuStreamSynchronize = unsafe extern "C" fn(CuStream) -> CuResult;
type CuEventCreate = unsafe extern "C" fn(*mut CuEvent, c_uint) -> CuResult;
type CuEventDestroy = unsafe extern "C" fn(CuEvent) -> CuResult;
type CuEventRecord = unsafe extern "C" fn(CuEvent, CuStream) -> CuResult;
type CuEventSynchronize = unsafe extern "C" fn(CuEvent) -> CuResult;
type CuEventElapsedTime = unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult;
#[cfg(feature = "cuda-profiling")]
type NvtxRangePushA = unsafe extern "C" fn(*const c_char) -> c_int;
#[cfg(feature = "cuda-profiling")]
type NvtxRangePop = unsafe extern "C" fn() -> c_int;

/// Error returned by CUDA driver and Signinum CUDA kernel helpers.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CudaError {
    /// CUDA driver library or device is unavailable.
    #[error("CUDA driver is unavailable: {message}")]
    Unavailable {
        /// Human-readable availability failure.
        message: String,
    },
    /// CUDA Driver API call failed.
    #[error("CUDA driver call {operation} failed with CUresult {code}{name}")]
    Driver {
        /// Driver operation name.
        operation: &'static str,
        /// Raw CUDA result code.
        code: CuResult,
        /// CUDA error name, when available.
        name: String,
    },
    /// Host output buffer is too small for a device download.
    #[error("CUDA copy output buffer too small: required {required}, have {have}")]
    OutputTooSmall {
        /// Required byte count.
        required: usize,
        /// Provided byte count.
        have: usize,
    },
    /// Byte length cannot be represented by the kernel ABI.
    #[error("CUDA byte length is too large for kernel launch: {len}")]
    LengthTooLarge {
        /// Byte length.
        len: usize,
    },
    /// Device byte length is not aligned to the requested typed view element.
    #[error("CUDA buffer length {bytes} is not a multiple of typed element size {element_size}")]
    LengthNotElementAligned {
        /// Byte length.
        bytes: usize,
        /// Requested element size.
        element_size: usize,
    },
    /// Image dimensions overflowed allocation or launch geometry.
    #[error("CUDA image allocation size overflow for {width}x{height}x{channels}")]
    ImageTooLarge {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
        /// Channel count.
        channels: usize,
    },
    /// Internal runtime state lock was poisoned.
    #[error("CUDA runtime state lock is poisoned: {message}")]
    StatePoisoned {
        /// Poison error message.
        message: String,
    },
    /// A Signinum CUDA kernel reported a validated runtime failure.
    #[error("CUDA kernel {kernel} reported status {code} detail {detail}")]
    KernelStatus {
        /// Kernel entry point or logical stage name.
        kernel: &'static str,
        /// Kernel-defined status code.
        code: u32,
        /// Kernel-defined detail code.
        detail: u32,
    },
    /// Caller supplied arguments that cannot be represented by this runtime API.
    #[error("CUDA invalid argument: {message}")]
    InvalidArgument {
        /// Human-readable validation failure.
        message: String,
    },
}

/// Prepared baseline JPEG Huffman table for CUDA JPEG decode kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaJpegHuffmanTable {
    /// Largest Huffman code for each bit length; negative means no codes of that length.
    pub max_code: [i32; 17],
    /// Value-index offset for each bit length.
    pub val_offset: [i32; 17],
    /// Huffman values in canonical order.
    pub values: [u8; 256],
    /// Number of valid entries in `values`.
    pub values_len: u32,
}

impl CudaJpegHuffmanTable {
    /// Prepare a CUDA Huffman table from JPEG BITS and HUFFVAL payloads.
    pub fn from_jpeg_bits_values(
        bits: [u8; 16],
        values_len: u16,
        values: [u8; 256],
    ) -> Result<Self, CudaError> {
        let values_len_usize = usize::from(values_len);
        let mut huffsize = [0u8; 256];
        let mut huffsize_len = 0usize;
        for (len_minus_1, &count) in bits.iter().enumerate() {
            let len = u8::try_from(len_minus_1 + 1).map_err(|_| CudaError::InvalidArgument {
                message: "JPEG Huffman code length exceeds u8".to_string(),
            })?;
            for _ in 0..count {
                if huffsize_len >= values_len_usize || huffsize_len >= huffsize.len() {
                    return Err(CudaError::InvalidArgument {
                        message: "JPEG Huffman BITS exceed values length".to_string(),
                    });
                }
                huffsize[huffsize_len] = len;
                huffsize_len += 1;
            }
        }
        if huffsize_len != values_len_usize {
            return Err(CudaError::InvalidArgument {
                message: "JPEG Huffman BITS do not match values length".to_string(),
            });
        }

        let mut huffcode = [0u16; 256];
        let mut code = 0u32;
        let mut si = huffsize.first().copied().unwrap_or(0);
        for (idx, &size) in huffsize[..huffsize_len].iter().enumerate() {
            while size != si {
                code <<= 1;
                si = si.saturating_add(1);
            }
            if si > 16 || code >= (1u32 << si) {
                return Err(CudaError::InvalidArgument {
                    message: "JPEG Huffman code overflow".to_string(),
                });
            }
            huffcode[idx] = u16::try_from(code).map_err(|_| CudaError::InvalidArgument {
                message: "JPEG Huffman code exceeds u16".to_string(),
            })?;
            code = code
                .checked_add(1)
                .ok_or_else(|| CudaError::InvalidArgument {
                    message: "JPEG Huffman code overflow".to_string(),
                })?;
        }

        let mut max_code = [-1i32; 17];
        let mut val_offset = [0i32; 17];
        let mut cursor = 0usize;
        for (len_minus_1, &count) in bits.iter().enumerate() {
            let len = len_minus_1 + 1;
            let count = usize::from(count);
            if count == 0 {
                continue;
            }
            let min_code = i32::from(huffcode[cursor]);
            max_code[len] = i32::from(huffcode[cursor + count - 1]);
            val_offset[len] = i32::try_from(cursor).map_err(|_| CudaError::InvalidArgument {
                message: "JPEG Huffman values length exceeds i32".to_string(),
            })? - min_code;
            cursor += count;
        }

        Ok(Self {
            max_code,
            val_offset,
            values,
            values_len: u32::from(values_len),
        })
    }
}

/// Entropy resume point for CUDA baseline JPEG decode.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaJpegEntropyCheckpoint {
    /// MCU index for this checkpoint.
    pub mcu_index: u32,
    /// Byte offset into the entropy payload.
    pub entropy_pos: u32,
    /// Left-aligned buffered entropy bits.
    pub bit_acc: u64,
    /// Number of valid buffered bits.
    pub bit_count: u32,
    /// Previous Y DC predictor.
    pub y_prev_dc: i32,
    /// Previous Cb DC predictor.
    pub cb_prev_dc: i32,
    /// Previous Cr DC predictor.
    pub cr_prev_dc: i32,
    /// Reserved for ABI-compatible expansion.
    pub reserved: u32,
}

/// Signinum-owned CUDA baseline JPEG RGB8 kernel shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaJpegRgb8Sampling {
    /// Fast 4:2:0 YCbCr shape: four Y blocks, then Cb and Cr per MCU.
    Fast420,
    /// Fast 4:2:2 YCbCr shape: two Y blocks, then Cb and Cr per MCU.
    Fast422,
    /// Fast 4:4:4 YCbCr shape: one Y block, then Cb and Cr per MCU.
    Fast444,
}

/// Signinum-owned CUDA baseline JPEG RGB8 decode plan.
#[derive(Debug)]
pub struct CudaJpegRgb8DecodePlan<'a> {
    /// MCU sampling/kernel shape.
    pub sampling: CudaJpegRgb8Sampling,
    /// Image dimensions as `(width, height)`.
    pub dimensions: (u32, u32),
    /// Number of MCUs per row.
    pub mcus_per_row: u32,
    /// Number of MCU rows.
    pub mcu_rows: u32,
    /// Entropy-coded scan payload with byte stuffing/restart markers removed.
    pub entropy_bytes: &'a [u8],
    /// Entropy resume checkpoints.
    pub entropy_checkpoints: &'a [CudaJpegEntropyCheckpoint],
    /// Luma quantization table in JPEG zigzag order.
    pub y_quant: [u16; 64],
    /// Cb quantization table in JPEG zigzag order.
    pub cb_quant: [u16; 64],
    /// Cr quantization table in JPEG zigzag order.
    pub cr_quant: [u16; 64],
    /// Y DC Huffman table.
    pub y_dc_table: CudaJpegHuffmanTable,
    /// Y AC Huffman table.
    pub y_ac_table: CudaJpegHuffmanTable,
    /// Cb DC Huffman table.
    pub cb_dc_table: CudaJpegHuffmanTable,
    /// Cb AC Huffman table.
    pub cb_ac_table: CudaJpegHuffmanTable,
    /// Cr DC Huffman table.
    pub cr_dc_table: CudaJpegHuffmanTable,
    /// Cr AC Huffman table.
    pub cr_ac_table: CudaJpegHuffmanTable,
}

/// Signinum-owned CUDA baseline JPEG 4:2:0 decode plan.
#[derive(Debug)]
pub struct CudaJpeg420Rgb8DecodePlan<'a> {
    /// Image dimensions as `(width, height)`.
    pub dimensions: (u32, u32),
    /// Number of MCUs per row.
    pub mcus_per_row: u32,
    /// Number of MCU rows.
    pub mcu_rows: u32,
    /// Entropy-coded scan payload with byte stuffing/restart markers removed.
    pub entropy_bytes: &'a [u8],
    /// Entropy resume checkpoints.
    pub entropy_checkpoints: &'a [CudaJpegEntropyCheckpoint],
    /// Luma quantization table in JPEG zigzag order.
    pub y_quant: [u16; 64],
    /// Cb quantization table in JPEG zigzag order.
    pub cb_quant: [u16; 64],
    /// Cr quantization table in JPEG zigzag order.
    pub cr_quant: [u16; 64],
    /// Y DC Huffman table.
    pub y_dc_table: CudaJpegHuffmanTable,
    /// Y AC Huffman table.
    pub y_ac_table: CudaJpegHuffmanTable,
    /// Cb DC Huffman table.
    pub cb_dc_table: CudaJpegHuffmanTable,
    /// Cb AC Huffman table.
    pub cb_ac_table: CudaJpegHuffmanTable,
    /// Cr DC Huffman table.
    pub cr_dc_table: CudaJpegHuffmanTable,
    /// Cr AC Huffman table.
    pub cr_ac_table: CudaJpegHuffmanTable,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
struct CudaJpeg420Params {
    width: u32,
    height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    entropy_len: u32,
    checkpoint_count: u32,
    out_stride: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
struct CudaJpegDecodeStatus {
    code: u32,
    detail: u32,
    position: u32,
    reserved: u32,
}

#[cfg(signinum_cuda_jpeg_decode_ptx_built)]
struct CudaJpegRgb8ValidatedPlan {
    params: CudaJpeg420Params,
    output_len: usize,
}

#[cfg(signinum_cuda_jpeg_decode_ptx_built)]
fn validate_jpeg_rgb8_plan(
    plan: &CudaJpegRgb8DecodePlan<'_>,
) -> Result<CudaJpegRgb8ValidatedPlan, CudaError> {
    let (width, _) = plan.dimensions;
    let out_stride = width.checked_mul(3).ok_or(CudaError::ImageTooLarge {
        width,
        height: plan.dimensions.1,
        channels: 3,
    })?;
    validate_jpeg_rgb8_plan_with_pitch(plan, out_stride as usize)
}

#[cfg(signinum_cuda_jpeg_decode_ptx_built)]
fn validate_jpeg_rgb8_plan_with_pitch(
    plan: &CudaJpegRgb8DecodePlan<'_>,
    pitch_bytes: usize,
) -> Result<CudaJpegRgb8ValidatedPlan, CudaError> {
    let (width, height) = plan.dimensions;
    if width == 0 || height == 0 {
        return Err(CudaError::InvalidArgument {
            message: "JPEG CUDA decode dimensions must be nonzero".to_string(),
        });
    }
    if plan.entropy_checkpoints.is_empty() {
        return Err(CudaError::InvalidArgument {
            message: "JPEG CUDA decode requires at least one entropy checkpoint".to_string(),
        });
    }
    let entropy_len =
        u32::try_from(plan.entropy_bytes.len()).map_err(|_| CudaError::LengthTooLarge {
            len: plan.entropy_bytes.len(),
        })?;
    let checkpoint_count =
        u32::try_from(plan.entropy_checkpoints.len()).map_err(|_| CudaError::LengthTooLarge {
            len: plan.entropy_checkpoints.len(),
        })?;
    let row_bytes = width.checked_mul(3).ok_or(CudaError::ImageTooLarge {
        width,
        height,
        channels: 3,
    })?;
    if pitch_bytes < row_bytes as usize {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "JPEG CUDA decode pitch {pitch_bytes} is smaller than row byte count {row_bytes}"
            ),
        });
    }
    let out_stride =
        u32::try_from(pitch_bytes).map_err(|_| CudaError::LengthTooLarge { len: pitch_bytes })?;
    let output_len = pitch_bytes
        .checked_mul(height as usize - 1)
        .and_then(|prefix| prefix.checked_add(row_bytes as usize))
        .ok_or(CudaError::ImageTooLarge {
            width,
            height,
            channels: 3,
        })?;

    Ok(CudaJpegRgb8ValidatedPlan {
        params: CudaJpeg420Params {
            width,
            height,
            mcus_per_row: plan.mcus_per_row,
            mcu_rows: plan.mcu_rows,
            entropy_len,
            checkpoint_count,
            out_stride,
            reserved: 0,
        },
        output_len,
    })
}

fn cuda_jpeg_rgb8_plan_from_420<'a>(
    plan: &CudaJpeg420Rgb8DecodePlan<'a>,
) -> CudaJpegRgb8DecodePlan<'a> {
    CudaJpegRgb8DecodePlan {
        sampling: CudaJpegRgb8Sampling::Fast420,
        dimensions: plan.dimensions,
        mcus_per_row: plan.mcus_per_row,
        mcu_rows: plan.mcu_rows,
        entropy_bytes: plan.entropy_bytes,
        entropy_checkpoints: plan.entropy_checkpoints,
        y_quant: plan.y_quant,
        cb_quant: plan.cb_quant,
        cr_quant: plan.cr_quant,
        y_dc_table: plan.y_dc_table,
        y_ac_table: plan.y_ac_table,
        cb_dc_table: plan.cb_dc_table,
        cb_ac_table: plan.cb_ac_table,
        cr_dc_table: plan.cr_dc_table,
        cr_ac_table: plan.cr_ac_table,
    }
}

#[cfg(signinum_cuda_jpeg_decode_ptx_built)]
fn jpeg_rgb8_kernel(sampling: CudaJpegRgb8Sampling) -> (CudaKernel, &'static str) {
    match sampling {
        CudaJpegRgb8Sampling::Fast420 => (
            CudaKernel::JpegDecodeFast420Rgb8,
            "signinum_jpeg_decode_fast420_rgb8",
        ),
        CudaJpegRgb8Sampling::Fast422 => (
            CudaKernel::JpegDecodeFast422Rgb8,
            "signinum_jpeg_decode_fast422_rgb8",
        ),
        CudaJpegRgb8Sampling::Fast444 => (
            CudaKernel::JpegDecodeFast444Rgb8,
            "signinum_jpeg_decode_fast444_rgb8",
        ),
    }
}

struct Driver {
    _library: Library,
    cu_init: CuInit,
    cu_device_get_count: CuDeviceGetCount,
    cu_device_get: CuDeviceGet,
    cu_ctx_create: CuCtxCreate,
    cu_ctx_destroy: CuCtxDestroy,
    cu_ctx_set_current: CuCtxSetCurrent,
    cu_mem_alloc: CuMemAlloc,
    cu_mem_free: CuMemFree,
    cu_mem_host_alloc: CuMemHostAlloc,
    cu_mem_free_host: CuMemFreeHost,
    cu_memcpy_htod: CuMemcpyHtoD,
    cu_memcpy_dtoh: CuMemcpyDtoH,
    cu_memset_d32: CuMemsetD32,
    cu_get_error_name: CuGetErrorName,
    cu_module_load_data: CuModuleLoadData,
    cu_module_unload: CuModuleUnload,
    cu_module_get_function: CuModuleGetFunction,
    cu_launch_kernel: CuLaunchKernel,
    cu_ctx_synchronize: CuCtxSynchronize,
    cu_stream_create: CuStreamCreate,
    cu_stream_destroy: CuStreamDestroy,
    cu_stream_synchronize: CuStreamSynchronize,
    cu_event_create: CuEventCreate,
    cu_event_destroy: CuEventDestroy,
    cu_event_record: CuEventRecord,
    cu_event_synchronize: CuEventSynchronize,
    cu_event_elapsed_time: CuEventElapsedTime,
}

impl Driver {
    fn load() -> Result<Self, CudaError> {
        #[cfg(target_os = "linux")]
        const LIBRARY_CANDIDATES: &[&str] = &["libcuda.so.1", "libcuda.so"];
        #[cfg(target_os = "windows")]
        const LIBRARY_CANDIDATES: &[&str] = &["nvcuda.dll"];
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        const LIBRARY_CANDIDATES: &[&str] = &[];

        let mut last_error = None;
        for candidate in LIBRARY_CANDIDATES {
            // SAFETY: Loading the CUDA driver library is required before symbol
            // lookup. The resulting Library is owned by Driver and outlives all
            // copied function pointers.
            match unsafe { Library::new(candidate) } {
                Ok(library) => return Self::from_library(library),
                Err(error) => last_error = Some(error.to_string()),
            }
        }

        Err(CudaError::Unavailable {
            message: last_error.unwrap_or_else(|| "unsupported CUDA host platform".to_string()),
        })
    }

    fn from_library(library: Library) -> Result<Self, CudaError> {
        Ok(Self {
            cu_init: load_symbol(&library, b"cuInit\0")?,
            cu_device_get_count: load_symbol(&library, b"cuDeviceGetCount\0")?,
            cu_device_get: load_symbol(&library, b"cuDeviceGet\0")?,
            cu_ctx_create: load_symbol(&library, b"cuCtxCreate_v2\0")?,
            cu_ctx_destroy: load_symbol(&library, b"cuCtxDestroy_v2\0")?,
            cu_ctx_set_current: load_symbol(&library, b"cuCtxSetCurrent\0")?,
            cu_mem_alloc: load_symbol(&library, b"cuMemAlloc_v2\0")?,
            cu_mem_free: load_symbol(&library, b"cuMemFree_v2\0")?,
            cu_mem_host_alloc: load_symbol(&library, b"cuMemHostAlloc\0")?,
            cu_mem_free_host: load_symbol(&library, b"cuMemFreeHost\0")?,
            cu_memcpy_htod: load_symbol(&library, b"cuMemcpyHtoD_v2\0")?,
            cu_memcpy_dtoh: load_symbol(&library, b"cuMemcpyDtoH_v2\0")?,
            cu_memset_d32: load_symbol(&library, b"cuMemsetD32_v2\0")?,
            cu_get_error_name: load_symbol(&library, b"cuGetErrorName\0")?,
            cu_module_load_data: load_symbol(&library, b"cuModuleLoadData\0")?,
            cu_module_unload: load_symbol(&library, b"cuModuleUnload\0")?,
            cu_module_get_function: load_symbol(&library, b"cuModuleGetFunction\0")?,
            cu_launch_kernel: load_symbol(&library, b"cuLaunchKernel\0")?,
            cu_ctx_synchronize: load_symbol(&library, b"cuCtxSynchronize\0")?,
            cu_stream_create: load_symbol(&library, b"cuStreamCreate\0")?,
            cu_stream_destroy: load_symbol(&library, b"cuStreamDestroy_v2\0")?,
            cu_stream_synchronize: load_symbol(&library, b"cuStreamSynchronize\0")?,
            cu_event_create: load_symbol(&library, b"cuEventCreate\0")?,
            cu_event_destroy: load_symbol(&library, b"cuEventDestroy_v2\0")?,
            cu_event_record: load_symbol(&library, b"cuEventRecord\0")?,
            cu_event_synchronize: load_symbol(&library, b"cuEventSynchronize\0")?,
            cu_event_elapsed_time: load_symbol(&library, b"cuEventElapsedTime\0")?,
            _library: library,
        })
    }

    fn check(&self, operation: &'static str, result: CuResult) -> Result<(), CudaError> {
        if result == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(CudaError::Driver {
                operation,
                code: result,
                name: self.error_name(result),
            })
        }
    }

    fn error_name(&self, result: CuResult) -> String {
        let mut name = std::ptr::null();
        // SAFETY: cuGetErrorName writes a borrowed static C string pointer for
        // a CUDA result code. A failure here is non-critical for diagnostics.
        let status = unsafe { (self.cu_get_error_name)(result, &raw mut name) };
        if status == CUDA_SUCCESS && !name.is_null() {
            // SAFETY: CUDA returns a NUL-terminated static string on success.
            let cstr = unsafe { std::ffi::CStr::from_ptr(name) };
            format!(" ({})", cstr.to_string_lossy())
        } else {
            String::new()
        }
    }
}

fn load_symbol<T: Copy>(library: &Library, name: &'static [u8]) -> Result<T, CudaError> {
    // SAFETY: Symbol names are NUL-terminated CUDA Driver API entry points. The
    // symbol value is copied, and Driver keeps the Library alive.
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| CudaError::Unavailable {
            message: format!(
                "missing CUDA driver symbol {}: {error}",
                String::from_utf8_lossy(name)
            ),
        })
}

struct CudaNvtxRange {
    #[cfg(feature = "cuda-profiling")]
    active: bool,
}

impl CudaNvtxRange {
    fn push(name: &str) -> Self {
        #[cfg(feature = "cuda-profiling")]
        {
            let Some(api) = nvtx_api() else {
                return Self { active: false };
            };
            let Ok(name) = std::ffi::CString::new(name) else {
                return Self { active: false };
            };
            // SAFETY: `name` is a NUL-terminated C string that lives for the
            // duration of the call. The NVTX function pointer is loaded from a
            // live library stored in NvtxApi.
            let depth = unsafe { (api.range_push_a)(name.as_ptr()) };
            Self { active: depth >= 0 }
        }
        #[cfg(not(feature = "cuda-profiling"))]
        {
            let _ = name;
            Self {}
        }
    }
}

impl Drop for CudaNvtxRange {
    fn drop(&mut self) {
        #[cfg(feature = "cuda-profiling")]
        if self.active {
            if let Some(api) = nvtx_api() {
                // SAFETY: Matching pop for a successful nvtxRangePushA in this
                // thread. NVTX returns a depth value that is not needed here.
                let _ = unsafe { (api.range_pop)() };
            }
        }
    }
}

#[cfg(feature = "cuda-profiling")]
struct NvtxApi {
    _library: Library,
    range_push_a: NvtxRangePushA,
    range_pop: NvtxRangePop,
}

#[cfg(feature = "cuda-profiling")]
fn nvtx_api() -> Option<&'static NvtxApi> {
    static API: OnceLock<Option<NvtxApi>> = OnceLock::new();
    API.get_or_init(load_optional_nvtx).as_ref()
}

#[cfg(feature = "cuda-profiling")]
fn load_optional_nvtx() -> Option<NvtxApi> {
    #[cfg(target_os = "linux")]
    const LIBRARY_CANDIDATES: &[&str] = &["libnvToolsExt.so.1", "libnvToolsExt.so"];
    #[cfg(target_os = "windows")]
    const LIBRARY_CANDIDATES: &[&str] = &["nvToolsExt64_1.dll", "nvToolsExt64_64_1.dll"];
    #[cfg(target_os = "macos")]
    const LIBRARY_CANDIDATES: &[&str] = &["libnvToolsExt.dylib"];
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    const LIBRARY_CANDIDATES: &[&str] = &[];

    for candidate in LIBRARY_CANDIDATES {
        // SAFETY: This optional profiling path only copies immutable NVTX
        // function pointers and stores the Library in NvtxApi for their
        // lifetime. Failure to load simply disables NVTX ranges.
        let Ok(library) = (unsafe { Library::new(candidate) }) else {
            continue;
        };
        let Ok(range_push_a) = load_symbol(&library, b"nvtxRangePushA\0") else {
            continue;
        };
        let Ok(range_pop) = load_symbol(&library, b"nvtxRangePop\0") else {
            continue;
        };
        return Some(NvtxApi {
            _library: library,
            range_push_a,
            range_pop,
        });
    }
    None
}

// SAFETY: CUDA Driver API handles are process resources guarded by the driver.
// The struct stores copied function pointers and owns the loaded library.
unsafe impl Send for Driver {}
// SAFETY: Driver entry points are immutable function pointers, and mutable CUDA
// state is always addressed through explicit CUDA context calls.
unsafe impl Sync for Driver {}

struct ContextInner {
    driver: Driver,
    context: CuContext,
    modules: Mutex<HashMap<CudaKernel, CompiledKernel>>,
    pinned_upload_staging: Mutex<Vec<PinnedUploadStaging>>,
}

struct PinnedUploadStaging {
    ptr: *mut u8,
    len: usize,
}

impl PinnedUploadStaging {
    fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: ptr is a live pinned allocation of len bytes.
            unsafe { std::slice::from_raw_parts(self.ptr.cast_const(), self.len) }
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: ptr is uniquely borrowed through &mut self and covers len
            // bytes allocated by CUDA.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    fn free(self, driver: &Driver) -> Result<(), CudaError> {
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
    fn set_current(&self) -> Result<(), CudaError> {
        // SAFETY: context is created by cuCtxCreate_v2 and remains valid while
        // ContextInner is alive.
        self.driver.check("cuCtxSetCurrent", unsafe {
            (self.driver.cu_ctx_set_current)(self.context)
        })
    }

    fn kernel_function(&self, kernel: CudaKernel) -> Result<CuFunction, CudaError> {
        ensure_kernel_ptx_built(kernel)?;
        self.set_current()?;
        let mut modules = self
            .modules
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        if let Some(compiled) = modules.get(&kernel) {
            return Ok(compiled.function);
        }

        let compiled = CompiledKernel::load(self, kernel)?;
        let function = compiled.function;
        modules.insert(kernel, compiled);
        Ok(function)
    }
}

fn ensure_kernel_ptx_built(kernel: CudaKernel) -> Result<(), CudaError> {
    let message = match kernel {
        CudaKernel::J2kDeinterleaveToF32
        | CudaKernel::J2kForwardRct
        | CudaKernel::J2kForwardIct
        | CudaKernel::J2kForwardDwt53Horizontal
        | CudaKernel::J2kForwardDwt53Vertical
        | CudaKernel::J2kForwardDwt97Horizontal
        | CudaKernel::J2kForwardDwt97Vertical
        | CudaKernel::J2kQuantizeSubband
        | CudaKernel::J2kQuantizeSubbandStrided
            if !J2K_ENCODE_PTX_BUILT_FROM_CUDA =>
        {
            Some("JPEG 2000 encode CUDA PTX was not built from j2k_encode_kernels.cu")
        }
        CudaKernel::Htj2kEncodeCodeblock
        | CudaKernel::Htj2kEncodeCodeblocks
        | CudaKernel::Htj2kPacketizeCleanup
            if !HTJ2K_ENCODE_PTX_BUILT_FROM_CUDA =>
        {
            Some("HTJ2K encode CUDA PTX was not built from htj2k_encode_kernels.cu")
        }
        CudaKernel::TranscodeReversible53Idct
        | CudaKernel::TranscodeReversible53VerticalLow
        | CudaKernel::TranscodeReversible53VerticalHigh
        | CudaKernel::TranscodeReversible53HorizontalLow
        | CudaKernel::TranscodeReversible53HorizontalHigh
        | CudaKernel::TranscodeDwt97Idct
        | CudaKernel::TranscodeDwt97RowLift
        | CudaKernel::TranscodeDwt97ColumnLift
        | CudaKernel::TranscodeDwt97IdctBatch
        | CudaKernel::TranscodeDwt97RowLiftBatch
        | CudaKernel::TranscodeDwt97ColumnLiftBatch
        | CudaKernel::TranscodeDwt97QuantizeCodeblocks
            if !TRANSCODE_PTX_BUILT_FROM_CUDA =>
        {
            Some("transcode CUDA PTX was not built from transcode_kernels.cu")
        }
        _ => None,
    };
    match message {
        Some(message) => Err(CudaError::InvalidArgument {
            message: message.to_string(),
        }),
        None => Ok(()),
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

/// CUDA driver context shared by Signinum CUDA adapter crates.
#[derive(Clone)]
pub struct CudaContext {
    inner: Arc<ContextInner>,
}

/// HTJ2K code-block decode job consumed by the CUDA entropy kernel launcher.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaHtj2kCodeBlockJob {
    /// Byte offset into the contiguous compressed payload buffer.
    pub payload_offset: u64,
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Combined cleanup/refinement byte length.
    pub payload_len: u32,
    /// Cleanup segment length in bytes.
    pub cleanup_length: u32,
    /// Refinement segment length in bytes.
    pub refinement_length: u32,
    /// Missing most-significant bit planes.
    pub missing_bit_planes: u8,
    /// Total coded bitplanes for this code block's sub-band.
    pub num_bitplanes: u8,
    /// Number of HT coding passes present.
    pub number_of_coding_passes: u8,
    /// Output row stride, in coefficients.
    pub output_stride: u32,
    /// Output offset, in coefficients, into the destination plane.
    pub output_offset: u32,
    /// Dequantization multiplier for decoded coefficient values.
    pub dequantization_step: f32,
    /// Vertically causal context mode flag.
    pub stripe_causal: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CudaHtj2kCodeBlockKernelJob {
    coded_offset: u32,
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
}

/// One output buffer and its code-block jobs for batched HTJ2K cleanup decode.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kCleanupTarget<'a> {
    /// Device buffer receiving decoded integer coefficient bits.
    pub coefficients: &'a CudaDeviceBuffer,
    /// Code-block jobs that write into `coefficients`.
    pub jobs: &'a [CudaHtj2kCodeBlockJob],
    /// Number of coefficient words available in `coefficients`.
    pub output_words: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CudaHtj2kCleanupMultiKernelJob {
    output_ptr: u64,
    coded_offset: u32,
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
}

/// One output buffer and its code-block jobs for batched HTJ2K dequantization.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kDequantizeTarget<'a> {
    /// Device buffer containing decoded integer coefficient bits.
    pub coefficients: &'a CudaDeviceBuffer,
    /// Code-block jobs that write into `coefficients`.
    pub jobs: &'a [CudaHtj2kCodeBlockJob],
    /// Number of coefficient words available in `coefficients`.
    pub output_words: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CudaHtj2kDequantizeKernelJob {
    output_ptr: u64,
    width: u32,
    height: u32,
    output_stride: u32,
    output_offset: u32,
    num_bitplanes: u32,
    reserved: u32,
    dequantization_step: f32,
}

/// Static HTJ2K entropy lookup tables uploaded for CUDA code-block decode.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kDecodeTables<'a> {
    /// HT cleanup VLC table for first quad row contexts.
    pub vlc_table0: &'a [u16; 1024],
    /// HT cleanup VLC table for subsequent quad row contexts.
    pub vlc_table1: &'a [u16; 1024],
    /// HT cleanup UVLC table for first quad row contexts.
    pub uvlc_table0: &'a [u16; 320],
    /// HT cleanup UVLC table for subsequent quad row contexts.
    pub uvlc_table1: &'a [u16; 256],
}

/// Static HTJ2K cleanup encoder lookup tables uploaded for CUDA code-block encode.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kEncodeTables<'a> {
    /// HT cleanup encoder VLC table for first quad row contexts.
    pub vlc_table0: &'a [u16; 2048],
    /// HT cleanup encoder VLC table for subsequent quad row contexts.
    pub vlc_table1: &'a [u16; 2048],
    /// Packed HT cleanup encoder UVLC table rows, six bytes per row.
    pub uvlc_table: &'a [u8],
}

/// Status written by the CUDA HTJ2K entropy decoder for one code-block job.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kStatus {
    /// Zero on success; nonzero values are kernel-defined failures.
    pub code: u32,
    /// Kernel-defined failure detail.
    pub detail: u32,
    /// Reserved for ABI stability.
    pub reserved0: u32,
    /// Reserved for ABI stability.
    pub reserved1: u32,
}

impl CudaHtj2kStatus {
    /// Return true when the CUDA kernel reported success.
    pub fn is_ok(self) -> bool {
        self.code == HTJ2K_STATUS_OK
    }
}

/// CUDA event timings for resident HTJ2K decode stages.
#[allow(clippy::struct_field_names)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kDecodeStageTimings {
    /// HT cleanup entropy decode dispatch time, in microseconds.
    pub ht_cleanup_us: u128,
    /// HT refinement work time, in microseconds.
    ///
    /// The current CUDA entropy kernel executes cleanup and refinement for a
    /// code-block in one dispatch. When a batch contains refinement segments,
    /// this records that fused dispatch time so higher-level profiles expose
    /// refinement-bearing work instead of silently reporting zero.
    pub ht_refine_us: u128,
    /// Sign/magnitude dequantization time, in microseconds.
    pub dequant_us: u128,
    /// Host-observed status download time, in microseconds.
    pub status_d2h_us: u128,
}

/// Device-resident HTJ2K entropy decode result.
#[derive(Debug)]
pub struct CudaHtj2kDecodeOutput {
    coefficients: CudaDeviceBuffer,
    execution: CudaExecutionStats,
    statuses: Vec<CudaHtj2kStatus>,
    stage_timings: CudaHtj2kDecodeStageTimings,
}

impl CudaHtj2kDecodeOutput {
    /// Device buffer containing decoded f32 coefficients.
    pub fn coefficients(&self) -> &CudaDeviceBuffer {
        &self.coefficients
    }

    /// CUDA execution counters for the decode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Per-code-block kernel status rows downloaded after dispatch.
    pub fn statuses(&self) -> &[CudaHtj2kStatus] {
        &self.statuses
    }

    /// CUDA event timings for the decode stages inside this output.
    pub fn stage_timings(&self) -> CudaHtj2kDecodeStageTimings {
        self.stage_timings
    }

    /// Split output into device coefficients, execution counters, and statuses.
    pub fn into_parts(self) -> (CudaDeviceBuffer, CudaExecutionStats, Vec<CudaHtj2kStatus>) {
        (self.coefficients, self.execution, self.statuses)
    }
}

/// Device-resident HTJ2K entropy decode result borrowed from a CUDA buffer pool.
#[derive(Debug)]
pub struct CudaPooledHtj2kDecodeOutput {
    coefficients: CudaPooledDeviceBuffer,
    execution: CudaExecutionStats,
    statuses: Vec<CudaHtj2kStatus>,
    stage_timings: CudaHtj2kDecodeStageTimings,
}

impl CudaPooledHtj2kDecodeOutput {
    /// Device buffer containing decoded f32 coefficients.
    pub fn coefficients(&self) -> Option<&CudaDeviceBuffer> {
        self.coefficients.as_device_buffer()
    }

    /// CUDA execution counters for the decode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Per-code-block kernel status rows downloaded after dispatch.
    pub fn statuses(&self) -> &[CudaHtj2kStatus] {
        &self.statuses
    }

    /// CUDA event timings for the decode stages inside this output.
    pub fn stage_timings(&self) -> CudaHtj2kDecodeStageTimings {
        self.stage_timings
    }

    /// Split output into pooled device coefficients, execution counters, and statuses.
    pub fn into_parts(
        self,
    ) -> (
        CudaPooledDeviceBuffer,
        CudaExecutionStats,
        Vec<CudaHtj2kStatus>,
    ) {
        (self.coefficients, self.execution, self.statuses)
    }
}

/// Device-resident static HTJ2K cleanup decode lookup tables.
#[derive(Clone, Debug)]
pub struct CudaHtj2kDecodeTableResources {
    inner: Arc<CudaHtj2kDecodeTableResourceInner>,
}

#[derive(Debug)]
struct CudaHtj2kDecodeTableResourceInner {
    vlc_table0: CudaDeviceBuffer,
    vlc_table1: CudaDeviceBuffer,
    uvlc_table0: CudaDeviceBuffer,
    uvlc_table1: CudaDeviceBuffer,
}

/// Device-resident HTJ2K decode payload plus shared lookup tables reused across sub-band dispatches.
#[derive(Debug)]
pub struct CudaHtj2kDecodeResources {
    payload: CudaHtj2kDecodePayload,
    payload_len: usize,
    tables: CudaHtj2kDecodeTableResources,
}

#[derive(Debug)]
enum CudaHtj2kDecodePayload {
    Owned(CudaDeviceBuffer),
    Pooled(CudaPooledDeviceBuffer),
}

impl CudaHtj2kDecodePayload {
    fn buffer(&self) -> Result<&CudaDeviceBuffer, CudaError> {
        match self {
            Self::Owned(buffer) => Ok(buffer),
            Self::Pooled(buffer) => pooled_device_buffer(buffer),
        }
    }
}

/// Device-resident HTJ2K cleanup encode lookup tables reused across sub-band dispatches.
#[derive(Debug)]
pub struct CudaHtj2kEncodeResources {
    vlc_table0: CudaDeviceBuffer,
    vlc_table1: CudaDeviceBuffer,
    uvlc_table: CudaDeviceBuffer,
}

const HTJ2K_STATUS_OK: u32 = 0;
const HTJ2K_STATUS_UNSUPPORTED: u32 = 2;
const J2K_ENCODE_PTX_BUILT_FROM_CUDA: bool = cfg!(signinum_cuda_j2k_encode_ptx_built);
const HTJ2K_ENCODE_PTX_BUILT_FROM_CUDA: bool = cfg!(signinum_cuda_htj2k_encode_ptx_built);
const HTJ2K_ENCODE_MAX_CODEBLOCK_WIDTH: u32 = 1024;
const HTJ2K_ENCODE_MAX_CODEBLOCK_SAMPLES: usize = 4096;
/// True when the coefficient-domain transcode kernels were compiled by nvcc
/// (the runner). When false, build.rs wrote a placeholder PTX, so dispatch
/// returns a typed error instead of loading a non-existent kernel.
const TRANSCODE_PTX_BUILT_FROM_CUDA: bool = cfg!(signinum_cuda_transcode_ptx_built);

/// Whether the coefficient-domain transcode kernels were compiled (runner).
/// Backends check this to fall back to the scalar oracle when the kernels are
/// unavailable (e.g. a non-nvcc build) instead of attempting a device launch.
#[must_use]
pub fn transcode_kernels_built() -> bool {
    TRANSCODE_PTX_BUILT_FROM_CUDA
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CudaHtj2kEncodeParams {
    width: u32,
    height: u32,
    coefficient_stride: u32,
    total_bitplanes: u32,
    output_capacity: u32,
    target_coding_passes: u32,
}

/// One HTJ2K code-block encode job consumed by the CUDA batch encoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHtj2kEncodeCodeBlockJob {
    /// Offset, in i32 coefficients, into the contiguous coefficient buffer.
    pub coefficient_offset: u32,
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Total coded bitplanes for this code block's sub-band.
    pub total_bitplanes: u8,
    /// Requested HT coding passes. `1` emits cleanup-only output; `2` emits a
    /// zero `SigProp` segment for exactly representable blocks; `3` emits
    /// `SigProp` bits for newly significant magnitude-3 samples plus `MagRef`
    /// bits for cleanup-significant samples.
    pub target_coding_passes: u8,
}

/// One HTJ2K code-block region consumed from a strided resident coefficient buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHtj2kEncodeCodeBlockRegionJob {
    /// Offset, in i32 coefficients, to the top-left coefficient of this code block.
    pub coefficient_offset: u32,
    /// Source row stride in i32 coefficients.
    pub coefficient_stride: u32,
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Total coded bitplanes for this code block's sub-band.
    pub total_bitplanes: u8,
    /// Requested HT coding passes. `1` emits cleanup-only output; `2` emits a
    /// zero `SigProp` segment for exactly representable blocks; `3` emits
    /// `SigProp` bits for newly significant magnitude-3 samples plus `MagRef`
    /// bits for cleanup-significant samples.
    pub target_coding_passes: u8,
}

/// Resident coefficient buffer and jobs for a multi-input HTJ2K encode batch.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kEncodeResidentTarget<'a> {
    /// Device buffer containing quantized i32 coefficients.
    pub coefficients: &'a CudaDeviceBuffer,
    /// Number of i32 coefficients available in `coefficients`.
    pub coefficient_count: usize,
    /// Code-block jobs that read from `coefficients`.
    pub jobs: &'a [CudaHtj2kEncodeCodeBlockJob],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CudaHtj2kEncodeKernelJob {
    coefficient_offset: u32,
    coefficient_stride: u32,
    width: u32,
    height: u32,
    total_bitplanes: u32,
    output_offset: u32,
    output_capacity: u32,
    target_coding_passes: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CudaHtj2kEncodeMultiInputKernelJob {
    coefficient_ptr: u64,
    coefficient_offset: u32,
    coefficient_stride: u32,
    width: u32,
    height: u32,
    total_bitplanes: u32,
    output_offset: u32,
    output_capacity: u32,
    target_coding_passes: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CudaHtj2kEncodeCompactJob {
    source_offset: u32,
    compact_offset: u32,
    data_len: u32,
    reserved: u32,
}

/// Status written by the CUDA HTJ2K code-block cleanup-pass encoder.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kEncodeStatus {
    /// Zero on success; nonzero values are kernel-defined failures.
    pub code: u32,
    /// Kernel-defined failure detail.
    pub detail: u32,
    /// Encoded payload byte length.
    pub data_len: u32,
    /// Number of coding passes in the encoded payload.
    pub number_of_coding_passes: u32,
    /// Number of missing most-significant bitplanes.
    pub missing_bit_planes: u32,
    /// Reserved for ABI stability.
    pub reserved0: u32,
    /// Reserved for ABI stability.
    pub reserved1: u32,
    /// Reserved for ABI stability.
    pub reserved2: u32,
}

impl CudaHtj2kEncodeStatus {
    /// Return true when the CUDA kernel reported success.
    pub fn is_ok(self) -> bool {
        self.code == HTJ2K_STATUS_OK
    }
}

/// CUDA event timings for HTJ2K cleanup-pass encode stages.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kEncodeStageTimings {
    /// Total HT cleanup-pass encode, compaction, and required result readback time, in microseconds.
    pub ht_encode_us: u128,
    /// HT cleanup-pass encode kernel time, in microseconds.
    pub ht_kernel_us: u128,
    /// Status-buffer device-to-host readback time, in microseconds.
    pub ht_status_readback_us: u128,
    /// Encoded-byte compaction kernel time, in microseconds.
    pub ht_compact_us: u128,
    /// Compacted encoded-byte device-to-host readback time, in microseconds.
    pub ht_output_readback_us: u128,
}

impl CudaHtj2kEncodeStageTimings {
    fn from_parts(
        ht_kernel_us: u128,
        ht_status_readback_us: u128,
        ht_compact_us: u128,
        ht_output_readback_us: u128,
    ) -> Self {
        Self {
            ht_encode_us: ht_kernel_us
                .saturating_add(ht_status_readback_us)
                .saturating_add(ht_compact_us)
                .saturating_add(ht_output_readback_us),
            ht_kernel_us,
            ht_status_readback_us,
            ht_compact_us,
            ht_output_readback_us,
        }
    }
}

/// Host-visible HTJ2K cleanup-pass encode result produced by a CUDA kernel.
#[derive(Debug)]
pub struct CudaHtj2kEncodedCodeBlock {
    data: Vec<u8>,
    status: CudaHtj2kEncodeStatus,
    execution: CudaExecutionStats,
    stage_timings: CudaHtj2kEncodeStageTimings,
}

impl CudaHtj2kEncodedCodeBlock {
    /// Encoded cleanup-pass payload bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
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

    /// Consume this code block and return its encoded payload plus segment
    /// metadata.
    pub fn into_parts(self) -> (Vec<u8>, u32, u32, u8, u8) {
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
            self.data,
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

/// Host-visible HTJ2K cleanup-pass encode batch produced by one CUDA kernel dispatch.
#[derive(Debug)]
pub struct CudaHtj2kEncodedCodeBlocks {
    code_blocks: Vec<CudaHtj2kEncodedCodeBlock>,
    execution: CudaExecutionStats,
    stage_timings: CudaHtj2kEncodeStageTimings,
}

impl CudaHtj2kEncodedCodeBlocks {
    /// Encoded cleanup code-block payloads, in the same order as the submitted jobs.
    pub fn code_blocks(&self) -> &[CudaHtj2kEncodedCodeBlock] {
        &self.code_blocks
    }

    /// Consume the batch and return its per-code-block outputs.
    pub fn into_code_blocks(self) -> Vec<CudaHtj2kEncodedCodeBlock> {
        self.code_blocks
    }

    /// CUDA execution counters for the batch encode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// CUDA event timings for the batch encode dispatch.
    pub fn stage_timings(&self) -> CudaHtj2kEncodeStageTimings {
        self.stage_timings
    }
}

/// Host-visible compact HTJ2K cleanup-pass encode metadata for one code block.
#[derive(Debug)]
pub struct CudaHtj2kCompactEncodedCodeBlock {
    payload_range: std::ops::Range<usize>,
    status: CudaHtj2kEncodeStatus,
    execution: CudaExecutionStats,
    stage_timings: CudaHtj2kEncodeStageTimings,
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
    payload: Vec<u8>,
    code_blocks: Vec<CudaHtj2kCompactEncodedCodeBlock>,
    execution: CudaExecutionStats,
    stage_timings: CudaHtj2kEncodeStageTimings,
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

    fn into_owned_code_blocks(self) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
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

/// One HTJ2K packet prepared for CUDA Tier-2 packetization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationPacket {
    /// First block metadata row for this packet.
    pub block_start: u32,
    /// Number of block metadata rows in this packet.
    pub block_count: u32,
    /// First subband metadata row for this packet.
    pub subband_start: u32,
    /// Number of subband metadata rows in this packet.
    pub subband_count: u32,
    /// Maximum bytes reserved for this packet's header and body.
    pub output_capacity: u32,
    /// Packet layer index used for first-inclusion tag-tree coding.
    pub layer: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CudaHtj2kPacketizationKernelPacket {
    block_start: u32,
    block_count: u32,
    subband_start: u32,
    subband_count: u32,
    output_offset: u32,
    output_capacity: u32,
    layer: u32,
}

/// One HTJ2K packet subband layout for CUDA packetization.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationSubband {
    /// First code-block metadata row for this subband.
    pub block_start: u32,
    /// Number of code-block metadata rows in this subband.
    pub block_count: u32,
    /// Number of code-blocks in the x direction.
    pub num_cbs_x: u32,
    /// Number of code-blocks in the y direction.
    pub num_cbs_y: u32,
}

/// Initial tag-tree state for one HTJ2K packet subband.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationSubbandTagState {
    /// First inclusion tag-tree node state row for this packet subband.
    pub inclusion_node_start: u32,
    /// First zero-bitplane tag-tree node state row for this packet subband.
    pub zero_bitplane_node_start: u32,
    /// Number of node state rows in each tree.
    pub node_count: u32,
    /// Reserved for ABI stability.
    pub reserved0: u32,
}

/// Current/known state for one HTJ2K packet tag-tree node.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationTagNodeState {
    /// Tag-tree current value before this packet is emitted.
    pub current: u32,
    /// Nonzero when this node value is already known before this packet.
    pub known: u32,
}

/// One HTJ2K code-block contribution for CUDA packetization.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationBlock {
    /// Byte offset into the contiguous encoded code-block payload.
    pub data_offset: u32,
    /// Encoded code-block payload length in bytes.
    pub data_len: u32,
    /// HTJ2K cleanup segment length in bytes.
    pub cleanup_length: u32,
    /// HTJ2K refinement segment length in bytes.
    pub refinement_length: u32,
    /// Number of coding passes in this contribution.
    pub num_coding_passes: u32,
    /// Number of zero most-significant bitplanes before first inclusion.
    pub num_zero_bitplanes: u32,
    /// L-block value for segment-length coding.
    pub l_block: u32,
    /// Nonzero when this code block was included in an earlier packet for the same packet state.
    pub previously_included: u32,
    /// First packet layer where this code block is included, or tag-tree infinity when absent.
    pub inclusion_layer: u32,
}

/// Status written by the CUDA HTJ2K packetizer for one packet.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationStatus {
    /// Zero on success; nonzero values are kernel-defined failures.
    pub code: u32,
    /// Kernel-defined failure detail.
    pub detail: u32,
    /// Number of packet bytes written into this packet slot.
    pub output_len: u32,
    /// Reserved for ABI stability.
    pub reserved0: u32,
}

impl CudaHtj2kPacketizationStatus {
    /// Return true when the CUDA kernel reported success.
    pub fn is_ok(self) -> bool {
        self.code == HTJ2K_STATUS_OK
    }
}

/// CUDA event timings for HTJ2K Tier-2 packetization stages.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationStageTimings {
    /// Cleanup packetization dispatch time, in microseconds.
    pub packetize_us: u128,
}

/// Host-visible HTJ2K packet payload produced by the CUDA Tier-2 packetizer.
#[derive(Debug)]
pub struct CudaHtj2kPacketizedTile {
    data: Vec<u8>,
    statuses: Vec<CudaHtj2kPacketizationStatus>,
    execution: CudaExecutionStats,
    stage_timings: CudaHtj2kPacketizationStageTimings,
}

impl CudaHtj2kPacketizedTile {
    /// Concatenated tile packet payload bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Per-packet kernel status rows downloaded after dispatch.
    pub fn statuses(&self) -> &[CudaHtj2kPacketizationStatus] {
        &self.statuses
    }

    /// CUDA execution counters for the packetization dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// CUDA event timings for the packetization dispatch.
    pub fn stage_timings(&self) -> CudaHtj2kPacketizationStageTimings {
        self.stage_timings
    }
}

const HTJ2K_ENCODE_OUTPUT_CAPACITY: usize = 24 * 1024;
const HTJ2K_UVLC_ENCODE_TABLE_BYTES: usize = 75 * 6;

/// CUDA-side integer rectangle for JPEG 2000 direct-plan kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaJ2kRect {
    /// Inclusive minimum x coordinate.
    pub x0: u32,
    /// Inclusive minimum y coordinate.
    pub y0: u32,
    /// Exclusive maximum x coordinate.
    pub x1: u32,
    /// Exclusive maximum y coordinate.
    pub y1: u32,
}

/// One single-decomposition inverse DWT dispatch over device coefficient bands.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaJ2kIdwtJob {
    /// Output rectangle produced by the IDWT stage.
    pub rect: CudaJ2kRect,
    /// LL input band rectangle.
    pub ll_rect: CudaJ2kRect,
    /// HL input band rectangle.
    pub hl_rect: CudaJ2kRect,
    /// LH input band rectangle.
    pub lh_rect: CudaJ2kRect,
    /// HH input band rectangle.
    pub hh_rect: CudaJ2kRect,
    /// Nonzero for irreversible 9/7; zero for reversible 5/3.
    pub irreversible97: u32,
}

/// One output buffer and input band set for batched inverse DWT.
#[derive(Clone, Copy, Debug)]
pub struct CudaJ2kIdwtTarget<'a> {
    /// LL input band.
    pub ll: &'a CudaDeviceBuffer,
    /// HL input band.
    pub hl: &'a CudaDeviceBuffer,
    /// LH input band.
    pub lh: &'a CudaDeviceBuffer,
    /// HH input band.
    pub hh: &'a CudaDeviceBuffer,
    /// Output buffer for the reconstructed band.
    pub output: &'a CudaDeviceBuffer,
    /// IDWT geometry and transform metadata.
    pub job: CudaJ2kIdwtJob,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CudaJ2kIdwtMultiKernelJob {
    ll_ptr: u64,
    hl_ptr: u64,
    lh_ptr: u64,
    hh_ptr: u64,
    output_ptr: u64,
    job: CudaJ2kIdwtJob,
}

/// Grayscale store dispatch from f32 component samples to tightly packed Gray8.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kStoreGray8Job {
    /// Source component buffer width in samples.
    pub input_width: u32,
    /// Source x offset in samples.
    pub source_x: u32,
    /// Source y offset in samples.
    pub source_y: u32,
    /// Number of samples copied per row.
    pub copy_width: u32,
    /// Number of rows copied.
    pub copy_height: u32,
    /// Destination output width in samples.
    pub output_width: u32,
    /// Destination output height in rows.
    pub output_height: u32,
    /// Destination x offset in samples.
    pub output_x: u32,
    /// Destination y offset in samples.
    pub output_y: u32,
    /// Level-shift addend applied before quantizing to Gray8.
    pub addend: f32,
    /// Source component bit depth.
    pub bit_depth: u32,
}

/// Grayscale store dispatch from f32 component samples to tightly packed Gray16.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kStoreGray16Job {
    /// Source component buffer width in samples.
    pub input_width: u32,
    /// Source x offset in samples.
    pub source_x: u32,
    /// Source y offset in samples.
    pub source_y: u32,
    /// Number of samples copied per row.
    pub copy_width: u32,
    /// Number of rows copied.
    pub copy_height: u32,
    /// Destination output width in samples.
    pub output_width: u32,
    /// Destination output height in rows.
    pub output_height: u32,
    /// Destination x offset in samples.
    pub output_x: u32,
    /// Destination y offset in samples.
    pub output_y: u32,
    /// Level-shift addend applied before quantizing to Gray16.
    pub addend: f32,
    /// Source component bit depth.
    pub bit_depth: u32,
}

/// In-place inverse MCT dispatch over three device f32 component planes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kInverseMctJob {
    /// Number of samples in each component plane.
    pub len: u32,
    /// Nonzero for irreversible ICT; zero for reversible RCT.
    pub irreversible97: u32,
    /// Addend applied to output channel 0 after inverse MCT.
    pub addend0: f32,
    /// Addend applied to output channel 1 after inverse MCT.
    pub addend1: f32,
    /// Addend applied to output channel 2 after inverse MCT.
    pub addend2: f32,
}

/// RGB/RGBA store dispatch from three f32 component planes to packed 8-bit pixels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kStoreRgb8Job {
    /// Source width for component 0.
    pub input_width0: u32,
    /// Source width for component 1.
    pub input_width1: u32,
    /// Source width for component 2.
    pub input_width2: u32,
    /// Source x offset for component 0.
    pub source_x0: u32,
    /// Source y offset for component 0.
    pub source_y0: u32,
    /// Source x offset for component 1.
    pub source_x1: u32,
    /// Source y offset for component 1.
    pub source_y1: u32,
    /// Source x offset for component 2.
    pub source_x2: u32,
    /// Source y offset for component 2.
    pub source_y2: u32,
    /// Number of pixels copied per row.
    pub copy_width: u32,
    /// Number of rows copied.
    pub copy_height: u32,
    /// Destination output width in pixels.
    pub output_width: u32,
    /// Destination output height in rows.
    pub output_height: u32,
    /// Destination x offset.
    pub output_x: u32,
    /// Destination y offset.
    pub output_y: u32,
    /// Addend applied to component 0 before quantizing.
    pub addend0: f32,
    /// Addend applied to component 1 before quantizing.
    pub addend1: f32,
    /// Addend applied to component 2 before quantizing.
    pub addend2: f32,
    /// Source bit depth for component 0.
    pub bit_depth0: u32,
    /// Source bit depth for component 1.
    pub bit_depth1: u32,
    /// Source bit depth for component 2.
    pub bit_depth2: u32,
    /// Nonzero to write RGBA8 with opaque alpha; zero writes RGB8.
    pub rgba: u32,
}

/// RGB/RGBA store dispatch from three f32 component planes to packed 16-bit pixels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kStoreRgb16Job {
    /// Source width for component 0.
    pub input_width0: u32,
    /// Source width for component 1.
    pub input_width1: u32,
    /// Source width for component 2.
    pub input_width2: u32,
    /// Source x offset for component 0.
    pub source_x0: u32,
    /// Source y offset for component 0.
    pub source_y0: u32,
    /// Source x offset for component 1.
    pub source_x1: u32,
    /// Source y offset for component 1.
    pub source_y1: u32,
    /// Source x offset for component 2.
    pub source_x2: u32,
    /// Source y offset for component 2.
    pub source_y2: u32,
    /// Number of pixels copied per row.
    pub copy_width: u32,
    /// Number of rows copied.
    pub copy_height: u32,
    /// Destination output width in pixels.
    pub output_width: u32,
    /// Destination output height in rows.
    pub output_height: u32,
    /// Destination x offset.
    pub output_x: u32,
    /// Destination y offset.
    pub output_y: u32,
    /// Addend applied to component 0 before quantizing.
    pub addend0: f32,
    /// Addend applied to component 1 before quantizing.
    pub addend1: f32,
    /// Addend applied to component 2 before quantizing.
    pub addend2: f32,
    /// Source bit depth for component 0.
    pub bit_depth0: u32,
    /// Source bit depth for component 1.
    pub bit_depth1: u32,
    /// Source bit depth for component 2.
    pub bit_depth2: u32,
    /// Nonzero to write RGBA16 with opaque alpha; zero writes RGB16.
    pub rgba: u32,
}

/// Fused inverse RCT/ICT and packed RGB8/RGBA8 store dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kStoreRgb8MctJob {
    /// RGB/RGBA store geometry, addends, bit depths, and alpha mode.
    pub store: CudaJ2kStoreRgb8Job,
    /// Nonzero for irreversible ICT; zero for reversible RCT.
    pub irreversible97: u32,
}

/// One fused inverse MCT plus RGB8/RGBA8 store item for a batched dispatch.
#[derive(Clone, Copy, Debug)]
pub struct CudaJ2kStoreRgb8MctTarget<'a> {
    /// Source component plane 0.
    pub plane0: &'a CudaDeviceBuffer,
    /// Source component plane 1.
    pub plane1: &'a CudaDeviceBuffer,
    /// Source component plane 2.
    pub plane2: &'a CudaDeviceBuffer,
    /// Store geometry and inverse MCT parameters.
    pub job: CudaJ2kStoreRgb8MctJob,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CudaJ2kStoreRgb8MctBatchJob {
    plane0_ptr: CuDevicePtr,
    plane1_ptr: CuDevicePtr,
    plane2_ptr: CuDevicePtr,
    output_ptr: CuDevicePtr,
    job: CudaJ2kStoreRgb8MctJob,
}

/// Fused inverse RCT/ICT and packed RGB16/RGBA16 store dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kStoreRgb16MctJob {
    /// RGB/RGBA store geometry, addends, bit depths, and alpha mode.
    pub store: CudaJ2kStoreRgb16Job,
    /// Nonzero for irreversible ICT; zero for reversible RCT.
    pub irreversible97: u32,
}

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

    /// Upload host bytes into a CUDA device buffer.
    pub fn upload(&self, bytes: &[u8]) -> Result<CudaDeviceBuffer, CudaError> {
        self.inner.set_current()?;

        let mut ptr = 0;
        let buffer = if bytes.is_empty() {
            CudaDeviceBuffer {
                context: self.clone(),
                ptr,
                len: bytes.len(),
            }
        } else {
            // SAFETY: CUDA writes a device pointer for the requested byte size.
            self.inner.driver.check("cuMemAlloc_v2", unsafe {
                (self.inner.driver.cu_mem_alloc)(&raw mut ptr, bytes.len())
            })?;

            CudaDeviceBuffer {
                context: self.clone(),
                ptr,
                len: bytes.len(),
            }
        };

        if !bytes.is_empty() {
            // SAFETY: ptr is a valid device allocation of bytes.len(), and the
            // host pointer is valid for bytes.len().
            self.inner.driver.check("cuMemcpyHtoD_v2", unsafe {
                (self.inner.driver.cu_memcpy_htod)(
                    ptr,
                    bytes.as_ptr().cast::<c_void>(),
                    bytes.len(),
                )
            })?;
        }

        Ok(buffer)
    }

    /// Upload host bytes through a temporary page-locked staging buffer.
    pub fn upload_pinned(&self, bytes: &[u8]) -> Result<CudaDeviceBuffer, CudaError> {
        if bytes.is_empty() {
            return self.upload(bytes);
        }
        let mut staging = self.take_pinned_upload_staging(bytes.len())?;
        staging.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        let upload_result = self.upload(&staging.as_slice()[..bytes.len()]);
        let recycle_result = self.recycle_pinned_upload_staging(staging);
        match (upload_result, recycle_result) {
            (Ok(buffer), Ok(())) => Ok(buffer),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }

    fn take_pinned_upload_staging(&self, len: usize) -> Result<PinnedUploadStaging, CudaError> {
        self.inner.set_current()?;
        let mut staging =
            self.inner
                .pinned_upload_staging
                .lock()
                .map_err(|error| CudaError::StatePoisoned {
                    message: error.to_string(),
                })?;
        if let Some(index) = staging.iter().position(|buffer| buffer.len >= len) {
            return Ok(staging.swap_remove(index));
        }
        drop(staging);

        let mut ptr = std::ptr::null_mut();
        // SAFETY: CUDA writes a page-locked host pointer for the requested byte
        // length. The allocation is freed by the context's staging pool cleanup.
        self.inner.driver.check("cuMemHostAlloc", unsafe {
            (self.inner.driver.cu_mem_host_alloc)(&raw mut ptr, len, 0)
        })?;
        Ok(PinnedUploadStaging {
            ptr: ptr.cast::<u8>(),
            len,
        })
    }

    fn recycle_pinned_upload_staging(&self, staging: PinnedUploadStaging) -> Result<(), CudaError> {
        let mut pool =
            self.inner
                .pinned_upload_staging
                .lock()
                .map_err(|error| CudaError::StatePoisoned {
                    message: error.to_string(),
                })?;
        if pool.len() < PINNED_UPLOAD_STAGING_POOL_MAX {
            pool.push(staging);
            return Ok(());
        }
        drop(pool);
        self.inner.set_current()?;
        staging.free(&self.inner.driver)
    }

    /// Upload host `f32` samples into a CUDA device buffer.
    pub fn upload_f32(&self, samples: &[f32]) -> Result<CudaDeviceBuffer, CudaError> {
        self.upload(f32_slice_as_bytes(samples))
    }

    /// Upload host `f32` samples through a temporary page-locked staging buffer.
    pub fn upload_f32_pinned(&self, samples: &[f32]) -> Result<CudaDeviceBuffer, CudaError> {
        self.upload_pinned(f32_slice_as_bytes(samples))
    }

    /// Upload host `i32` samples through a temporary page-locked staging buffer.
    pub fn upload_i32_pinned(&self, samples: &[i32]) -> Result<CudaDeviceBuffer, CudaError> {
        self.upload_pinned(i32_slice_as_bytes(samples))
    }

    /// Copy host bytes through a CUDA copy kernel and return device output.
    pub fn copy_with_kernel(&self, bytes: &[u8]) -> Result<CudaKernelOutput, CudaError> {
        let staging = self.upload(bytes)?;
        let output = self.copy_device_to_device_with_kernel(&staging)?;
        let copy_dispatches = usize::from(!bytes.is_empty());
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: copy_dispatches,
                copy_kernel_dispatches: copy_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    /// Decode one baseline JPEG 4:2:0 image to device-resident RGB8 using Signinum CUDA kernels.
    pub fn decode_jpeg_420_rgb8_owned(
        &self,
        plan: &CudaJpeg420Rgb8DecodePlan<'_>,
    ) -> Result<CudaKernelOutput, CudaError> {
        let plan = cuda_jpeg_rgb8_plan_from_420(plan);
        self.decode_jpeg_rgb8_owned(&plan)
    }

    /// Decode one baseline JPEG RGB8 image to device-resident RGB8 using Signinum CUDA kernels.
    pub fn decode_jpeg_rgb8_owned(
        &self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
    ) -> Result<CudaKernelOutput, CudaError> {
        #[cfg(not(signinum_cuda_jpeg_decode_ptx_built))]
        {
            let _ = plan;
            Err(CudaError::InvalidArgument {
                message: "Signinum CUDA JPEG decode PTX was not built from jpeg_decode_kernels.cu"
                    .to_string(),
            })
        }

        #[cfg(signinum_cuda_jpeg_decode_ptx_built)]
        {
            let validated = validate_jpeg_rgb8_plan(plan)?;
            self.inner.set_current()?;
            let output = self.allocate(validated.output_len)?;
            let execution = self.decode_jpeg_rgb8_owned_validated(plan, &output, validated)?;
            Ok(CudaKernelOutput {
                buffer: output,
                execution,
            })
        }
    }

    /// Decode one baseline JPEG 4:2:0 image into caller-owned CUDA RGB8 memory.
    pub fn decode_jpeg_420_rgb8_owned_into(
        &self,
        plan: &CudaJpeg420Rgb8DecodePlan<'_>,
        output: &CudaDeviceBuffer,
        pitch_bytes: usize,
    ) -> Result<CudaExecutionStats, CudaError> {
        let plan = cuda_jpeg_rgb8_plan_from_420(plan);
        self.decode_jpeg_rgb8_owned_into(&plan, output, pitch_bytes)
    }

    /// Decode one baseline JPEG RGB8 image into caller-owned CUDA RGB8 memory.
    pub fn decode_jpeg_rgb8_owned_into(
        &self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
        output: &CudaDeviceBuffer,
        pitch_bytes: usize,
    ) -> Result<CudaExecutionStats, CudaError> {
        #[cfg(not(signinum_cuda_jpeg_decode_ptx_built))]
        {
            let _ = (plan, output, pitch_bytes);
            Err(CudaError::InvalidArgument {
                message: "Signinum CUDA JPEG decode PTX was not built from jpeg_decode_kernels.cu"
                    .to_string(),
            })
        }

        #[cfg(signinum_cuda_jpeg_decode_ptx_built)]
        {
            let validated = validate_jpeg_rgb8_plan_with_pitch(plan, pitch_bytes)?;
            if output.byte_len() < validated.output_len {
                return Err(CudaError::OutputTooSmall {
                    required: validated.output_len,
                    have: output.byte_len(),
                });
            }
            self.inner.set_current()?;
            self.decode_jpeg_rgb8_owned_validated(plan, output, validated)
        }
    }

    #[cfg(signinum_cuda_jpeg_decode_ptx_built)]
    fn decode_jpeg_rgb8_owned_validated(
        &self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
        output: &CudaDeviceBuffer,
        validated: CudaJpegRgb8ValidatedPlan,
    ) -> Result<CudaExecutionStats, CudaError> {
        let (kernel, kernel_name) = jpeg_rgb8_kernel(plan.sampling);
        let entropy = self.upload(plan.entropy_bytes)?;
        let y_quant = self.upload(u16_slice_as_bytes(&plan.y_quant))?;
        let cb_quant = self.upload(u16_slice_as_bytes(&plan.cb_quant))?;
        let cr_quant = self.upload(u16_slice_as_bytes(&plan.cr_quant))?;
        let y_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.y_dc_table))?;
        let y_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.y_ac_table))?;
        let cb_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cb_dc_table))?;
        let cb_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cb_ac_table))?;
        let cr_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cr_dc_table))?;
        let cr_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cr_ac_table))?;
        let checkpoints = self.upload(cuda_jpeg_entropy_checkpoints_as_bytes(
            plan.entropy_checkpoints,
        ))?;
        let mut statuses = vec![CudaJpegDecodeStatus::default(); plan.entropy_checkpoints.len()];
        let status_buffer = self.upload(cuda_jpeg_decode_statuses_as_bytes(&statuses))?;
        self.launch_jpeg_decode_rgb8(
            kernel,
            &entropy,
            output,
            validated.params,
            &y_quant,
            &cb_quant,
            &cr_quant,
            &y_dc,
            &y_ac,
            &cb_dc,
            &cb_ac,
            &cr_dc,
            &cr_ac,
            &checkpoints,
            &status_buffer,
        )?;
        status_buffer.copy_to_host(cuda_jpeg_decode_statuses_as_bytes_mut(&mut statuses))?;
        for status in statuses {
            if status.code != 0 {
                return Err(CudaError::KernelStatus {
                    kernel: kernel_name,
                    code: status.code,
                    detail: status.detail,
                });
            }
        }
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }

    #[cfg(signinum_cuda_jpeg_decode_ptx_built)]
    #[allow(clippy::too_many_arguments)]
    fn launch_jpeg_decode_rgb8(
        &self,
        kernel: CudaKernel,
        entropy: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        mut params: CudaJpeg420Params,
        y_quant: &CudaDeviceBuffer,
        cb_quant: &CudaDeviceBuffer,
        cr_quant: &CudaDeviceBuffer,
        y_dc: &CudaDeviceBuffer,
        y_ac: &CudaDeviceBuffer,
        cb_dc: &CudaDeviceBuffer,
        cb_ac: &CudaDeviceBuffer,
        cr_dc: &CudaDeviceBuffer,
        cr_ac: &CudaDeviceBuffer,
        checkpoints: &CudaDeviceBuffer,
        status: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(kernel)?;
        let mut entropy_ptr = entropy.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut y_quant_ptr = y_quant.device_ptr();
        let mut cb_quant_ptr = cb_quant.device_ptr();
        let mut cr_quant_ptr = cr_quant.device_ptr();
        let mut y_dc_ptr = y_dc.device_ptr();
        let mut y_ac_ptr = y_ac.device_ptr();
        let mut cb_dc_ptr = cb_dc.device_ptr();
        let mut cb_ac_ptr = cb_ac.device_ptr();
        let mut cr_dc_ptr = cr_dc.device_ptr();
        let mut cr_ac_ptr = cr_ac.device_ptr();
        let mut checkpoints_ptr = checkpoints.device_ptr();
        let mut status_ptr = status.device_ptr();
        let mut kernel_params = [
            (&raw mut entropy_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut params).cast::<c_void>(),
            (&raw mut y_quant_ptr).cast::<c_void>(),
            (&raw mut cb_quant_ptr).cast::<c_void>(),
            (&raw mut cr_quant_ptr).cast::<c_void>(),
            (&raw mut y_dc_ptr).cast::<c_void>(),
            (&raw mut y_ac_ptr).cast::<c_void>(),
            (&raw mut cb_dc_ptr).cast::<c_void>(),
            (&raw mut cb_ac_ptr).cast::<c_void>(),
            (&raw mut cr_dc_ptr).cast::<c_void>(),
            (&raw mut cr_ac_ptr).cast::<c_void>(),
            (&raw mut checkpoints_ptr).cast::<c_void>(),
            (&raw mut status_ptr).cast::<c_void>(),
        ];
        let geometry = CudaLaunchGeometry {
            grid: (params.checkpoint_count, 1, 1),
            block: (1, 1, 1),
        };

        self.launch_kernel(function, geometry, &mut kernel_params)
    }

    /// Decode HTJ2K code blocks into a device-resident f32 coefficient plane.
    #[allow(clippy::similar_names)]
    pub fn decode_htj2k_codeblocks(
        &self,
        payload: &[u8],
        jobs: &[CudaHtj2kCodeBlockJob],
        tables: CudaHtj2kDecodeTables<'_>,
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        if jobs.is_empty() {
            return self.decode_empty_htj2k_codeblocks(jobs, output_words);
        }
        let resources = self.upload_htj2k_decode_resources(payload, tables)?;
        self.decode_htj2k_codeblocks_with_resources(&resources, jobs, output_words)
    }

    /// Decode HTJ2K code blocks without collecting CUDA event timings.
    #[allow(clippy::similar_names)]
    pub fn decode_htj2k_codeblocks_untimed(
        &self,
        payload: &[u8],
        jobs: &[CudaHtj2kCodeBlockJob],
        tables: CudaHtj2kDecodeTables<'_>,
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        if jobs.is_empty() {
            return self.decode_empty_htj2k_codeblocks(jobs, output_words);
        }
        let resources = self.upload_htj2k_decode_resources(payload, tables)?;
        self.decode_htj2k_codeblocks_with_resources_untimed(&resources, jobs, output_words)
    }

    /// Upload HTJ2K decode payload and lookup tables once for reuse by sub-band dispatches.
    pub fn upload_htj2k_decode_resources(
        &self,
        payload: &[u8],
        tables: CudaHtj2kDecodeTables<'_>,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        let tables = self.upload_htj2k_decode_table_resources(tables)?;
        self.upload_htj2k_decode_resources_with_tables(payload, &tables)
    }

    /// Upload static HTJ2K cleanup decode lookup tables once for reuse.
    pub fn upload_htj2k_decode_table_resources(
        &self,
        tables: CudaHtj2kDecodeTables<'_>,
    ) -> Result<CudaHtj2kDecodeTableResources, CudaError> {
        self.inner.set_current()?;
        Ok(CudaHtj2kDecodeTableResources {
            inner: Arc::new(CudaHtj2kDecodeTableResourceInner {
                vlc_table0: self.upload(u16_slice_as_bytes(tables.vlc_table0))?,
                vlc_table1: self.upload(u16_slice_as_bytes(tables.vlc_table1))?,
                uvlc_table0: self.upload(u16_slice_as_bytes(tables.uvlc_table0))?,
                uvlc_table1: self.upload(u16_slice_as_bytes(tables.uvlc_table1))?,
            }),
        })
    }

    /// Upload an HTJ2K decode payload while reusing already resident cleanup tables.
    pub fn upload_htj2k_decode_resources_with_tables(
        &self,
        payload: &[u8],
        tables: &CudaHtj2kDecodeTableResources,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        self.inner.set_current()?;
        Ok(CudaHtj2kDecodeResources {
            payload: CudaHtj2kDecodePayload::Owned(self.upload_pinned(payload)?),
            payload_len: payload.len(),
            tables: tables.clone(),
        })
    }

    /// Upload an HTJ2K decode payload into a pooled buffer while reusing already resident cleanup tables.
    pub fn upload_htj2k_decode_resources_with_tables_and_pool(
        &self,
        payload: &[u8],
        tables: &CudaHtj2kDecodeTableResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        self.inner.set_current()?;
        Ok(CudaHtj2kDecodeResources {
            payload: CudaHtj2kDecodePayload::Pooled(pool.upload_pinned(payload)?),
            payload_len: payload.len(),
            tables: tables.clone(),
        })
    }

    /// Upload static HTJ2K cleanup encoder lookup tables once for reuse.
    pub fn upload_htj2k_encode_resources(
        &self,
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodeResources, CudaError> {
        if tables.uvlc_table.len() != HTJ2K_UVLC_ENCODE_TABLE_BYTES {
            return Err(CudaError::LengthTooLarge {
                len: tables.uvlc_table.len(),
            });
        }
        self.inner.set_current()?;
        Ok(CudaHtj2kEncodeResources {
            vlc_table0: self.upload(u16_slice_as_bytes(tables.vlc_table0))?,
            vlc_table1: self.upload(u16_slice_as_bytes(tables.vlc_table1))?,
            uvlc_table: self.upload(tables.uvlc_table)?,
        })
    }

    /// Decode HTJ2K code blocks using already resident payload and lookup tables.
    pub fn decode_htj2k_codeblocks_with_resources(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_impl(resources, jobs, output_words, true)
    }

    /// Decode HTJ2K code blocks using resident resources without CUDA event timings.
    pub fn decode_htj2k_codeblocks_with_resources_untimed(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_impl(resources, jobs, output_words, false)
    }

    /// Decode HTJ2K code blocks using resident resources and caller-owned
    /// transient buffer reuse.
    pub fn decode_htj2k_codeblocks_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_and_pool_impl(
            resources,
            jobs,
            output_words,
            pool,
            true,
            true,
        )
    }

    /// Decode HTJ2K code blocks using resident resources and caller-owned
    /// transient buffer reuse, without CUDA event timings.
    pub fn decode_htj2k_codeblocks_with_resources_untimed_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_and_pool_impl(
            resources,
            jobs,
            output_words,
            pool,
            false,
            true,
        )
    }

    /// Decode HTJ2K cleanup passes into resident coefficient buffers using
    /// caller-owned transient buffer reuse. Dequantization is left to a later
    /// dispatch.
    pub fn decode_htj2k_codeblocks_cleanup_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_and_pool_impl(
            resources,
            jobs,
            output_words,
            pool,
            true,
            false,
        )
    }

    /// Decode HTJ2K cleanup passes into resident coefficient buffers using
    /// caller-owned transient buffer reuse, without CUDA event timings.
    pub fn decode_htj2k_codeblocks_cleanup_with_resources_untimed_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_and_pool_impl(
            resources,
            jobs,
            output_words,
            pool,
            false,
            false,
        )
    }

    /// Allocate and initialize an HTJ2K coefficient output buffer without
    /// launching entropy cleanup decode. This is used when cleanup work is
    /// batched across multiple output buffers.
    pub fn allocate_htj2k_codeblock_coefficients_with_pool(
        &self,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        self.inner.set_current()?;
        let output_bytes = output_words
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: output_words })?;
        let coefficients = pool.take(output_bytes)?;
        let coefficient_buffer = pooled_device_buffer(&coefficients)?;
        if htj2k_decode_needs_zero_fill(jobs, output_words)? {
            self.memset_d32(coefficient_buffer, 0, output_words)?;
        }
        Ok(CudaPooledHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats::default(),
            statuses: Vec::new(),
            stage_timings: CudaHtj2kDecodeStageTimings::default(),
        })
    }

    /// Decode HTJ2K cleanup passes for multiple output buffers with one CUDA
    /// dispatch. Dequantization is left to a later dispatch.
    pub fn decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed(
            resources, targets, pool, false,
        )
        .map(|(execution, _timings)| execution)
    }

    /// Enqueue HTJ2K cleanup passes for multiple output buffers with one CUDA
    /// dispatch. The returned value must be kept live until `finish` validates
    /// the kernel statuses after the default stream has completed.
    pub fn decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaQueuedHtj2kCleanup, CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = htj2k_cleanup_multi_kernel_jobs(targets, resources.payload_len)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaQueuedHtj2kCleanup {
                resources: Vec::new(),
                status_buffer: None,
                status_count: 0,
                kernel_name: "signinum_htj2k_decode_codeblocks_multi",
                execution: CudaExecutionStats::default(),
            });
        }
        let (decode_kernel, decode_kernel_name) = htj2k_decode_multi_kernel_for_jobs(&kernel_jobs);

        let jobs_buffer = pool.upload(htj2k_cleanup_multi_jobs_as_bytes(&kernel_jobs))?;
        let status_buffer = pool.take(htj2k_statuses_byte_len(kernel_jobs.len())?)?;
        let launch_result = self.launch_htj2k_decode_codeblocks_multi_async(
            decode_kernel,
            resources.payload.buffer()?,
            pooled_device_buffer(&jobs_buffer)?,
            &resources.tables.inner.vlc_table0,
            &resources.tables.inner.vlc_table1,
            &resources.tables.inner.uvlc_table0,
            &resources.tables.inner.uvlc_table1,
            pooled_device_buffer(&status_buffer)?,
            kernel_jobs.len(),
        );
        if let Err(error) = launch_result {
            let _ = self.synchronize();
            return Err(error);
        }

        Ok(CudaQueuedHtj2kCleanup {
            resources: vec![jobs_buffer],
            status_buffer: Some(status_buffer),
            status_count: kernel_jobs.len(),
            kernel_name: decode_kernel_name,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Decode HTJ2K cleanup passes for multiple output buffers with one CUDA
    /// dispatch and return optional host-side timing splits.
    ///
    /// Dequantization is left to a later dispatch. When `collect_stage_timings`
    /// is false, the cleanup kernel launch is left asynchronous and the
    /// mandatory status readback remains the completion point.
    pub fn decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
        collect_stage_timings: bool,
    ) -> Result<(CudaExecutionStats, CudaHtj2kDecodeStageTimings), CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = htj2k_cleanup_multi_kernel_jobs(targets, resources.payload_len)?;
        if kernel_jobs.is_empty() {
            return Ok((
                CudaExecutionStats::default(),
                CudaHtj2kDecodeStageTimings::default(),
            ));
        }

        let jobs_buffer = pool.upload(htj2k_cleanup_multi_jobs_as_bytes(&kernel_jobs))?;
        let status_buffer = pool.take(htj2k_statuses_byte_len(kernel_jobs.len())?)?;
        let (decode_kernel, decode_kernel_name) = htj2k_decode_multi_kernel_for_jobs(&kernel_jobs);
        if collect_stage_timings {
            self.launch_htj2k_decode_codeblocks_multi(
                decode_kernel,
                resources.payload.buffer()?,
                pooled_device_buffer(&jobs_buffer)?,
                &resources.tables.inner.vlc_table0,
                &resources.tables.inner.vlc_table1,
                &resources.tables.inner.uvlc_table0,
                &resources.tables.inner.uvlc_table1,
                pooled_device_buffer(&status_buffer)?,
                kernel_jobs.len(),
            )?;
        } else {
            self.launch_htj2k_decode_codeblocks_multi_async(
                decode_kernel,
                resources.payload.buffer()?,
                pooled_device_buffer(&jobs_buffer)?,
                &resources.tables.inner.vlc_table0,
                &resources.tables.inner.vlc_table1,
                &resources.tables.inner.uvlc_table0,
                &resources.tables.inner.uvlc_table1,
                pooled_device_buffer(&status_buffer)?,
                kernel_jobs.len(),
            )?;
        }

        let mut statuses = vec![CudaHtj2kStatus::default(); kernel_jobs.len()];
        let status_d2h_start = collect_stage_timings.then(Instant::now);
        status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses))?;
        let status_d2h_us = status_d2h_start.map_or(0, |start| start.elapsed().as_micros());
        if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
            return Err(CudaError::KernelStatus {
                kernel: decode_kernel_name,
                code: status.code,
                detail: status.detail,
            });
        }

        Ok((
            CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
            CudaHtj2kDecodeStageTimings {
                status_d2h_us,
                ..CudaHtj2kDecodeStageTimings::default()
            },
        ))
    }

    /// Decode HTJ2K cleanup-only passes and dequantize their coefficients in
    /// one CUDA dispatch. Targets containing refinement passes are rejected so
    /// callers can fall back to cleanup followed by dequantization.
    pub fn decode_htj2k_codeblocks_cleanup_dequantize_multi_with_resources_and_pool_timed(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
        collect_stage_timings: bool,
    ) -> Result<(CudaExecutionStats, CudaHtj2kDecodeStageTimings), CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = htj2k_cleanup_multi_kernel_jobs(targets, resources.payload_len)?;
        if kernel_jobs.is_empty() {
            return Ok((
                CudaExecutionStats::default(),
                CudaHtj2kDecodeStageTimings::default(),
            ));
        }
        let Some((decode_kernel, decode_kernel_name)) =
            htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&kernel_jobs)
        else {
            return Err(CudaError::InvalidArgument {
                message: "fused HTJ2K cleanup/dequantize requires cleanup-only jobs".to_string(),
            });
        };

        let jobs_buffer = pool.upload(htj2k_cleanup_multi_jobs_as_bytes(&kernel_jobs))?;
        let status_buffer = pool.take(htj2k_statuses_byte_len(kernel_jobs.len())?)?;
        if collect_stage_timings {
            self.launch_htj2k_decode_codeblocks_multi(
                decode_kernel,
                resources.payload.buffer()?,
                pooled_device_buffer(&jobs_buffer)?,
                &resources.tables.inner.vlc_table0,
                &resources.tables.inner.vlc_table1,
                &resources.tables.inner.uvlc_table0,
                &resources.tables.inner.uvlc_table1,
                pooled_device_buffer(&status_buffer)?,
                kernel_jobs.len(),
            )?;
        } else {
            self.launch_htj2k_decode_codeblocks_multi_async(
                decode_kernel,
                resources.payload.buffer()?,
                pooled_device_buffer(&jobs_buffer)?,
                &resources.tables.inner.vlc_table0,
                &resources.tables.inner.vlc_table1,
                &resources.tables.inner.uvlc_table0,
                &resources.tables.inner.uvlc_table1,
                pooled_device_buffer(&status_buffer)?,
                kernel_jobs.len(),
            )?;
        }

        let mut statuses = vec![CudaHtj2kStatus::default(); kernel_jobs.len()];
        let status_d2h_start = collect_stage_timings.then(Instant::now);
        status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses))?;
        let status_d2h_us = status_d2h_start.map_or(0, |start| start.elapsed().as_micros());
        if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
            return Err(CudaError::KernelStatus {
                kernel: decode_kernel_name,
                code: status.code,
                detail: status.detail,
            });
        }

        Ok((
            CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
            CudaHtj2kDecodeStageTimings {
                status_d2h_us,
                ..CudaHtj2kDecodeStageTimings::default()
            },
        ))
    }

    fn decode_htj2k_codeblocks_with_resources_impl(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        collect_stage_timings: bool,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.inner.set_current()?;
        let output_bytes = output_words
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: output_words })?;
        let coefficients = self.allocate(output_bytes)?;
        if htj2k_decode_needs_zero_fill(jobs, output_words)? {
            self.memset_d32(&coefficients, 0, output_words)?;
        }
        if jobs.is_empty() {
            return Ok(CudaHtj2kDecodeOutput {
                coefficients,
                execution: CudaExecutionStats::default(),
                statuses: Vec::new(),
                stage_timings: CudaHtj2kDecodeStageTimings::default(),
            });
        }

        let kernel_jobs = htj2k_kernel_jobs(jobs, resources.payload_len, output_words)?;
        let jobs_buffer = self.upload(htj2k_jobs_as_bytes(&kernel_jobs))?;
        let status_buffer = self.allocate(htj2k_statuses_byte_len(jobs.len())?)?;

        let has_refinement = jobs
            .iter()
            .any(|job| job.refinement_length > 0 || job.number_of_coding_passes > 1);
        let (ht_cleanup_us, dequant_us) = self.submit_htj2k_decode_and_dequantize(
            resources,
            &coefficients,
            &jobs_buffer,
            &status_buffer,
            jobs.len(),
            collect_stage_timings,
        )?;

        let mut statuses = vec![CudaHtj2kStatus::default(); jobs.len()];
        if let Err(error) = status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses)) {
            if !collect_stage_timings {
                let _ = self.synchronize();
            }
            return Err(error);
        }
        if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
            return Err(CudaError::KernelStatus {
                kernel: "signinum_htj2k_decode_codeblocks",
                code: status.code,
                detail: status.detail,
            });
        }

        Ok(CudaHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats {
                kernel_dispatches: 2,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 2,
                hardware_decode: false,
            },
            statuses,
            stage_timings: CudaHtj2kDecodeStageTimings {
                ht_cleanup_us,
                ht_refine_us: if has_refinement { ht_cleanup_us } else { 0 },
                dequant_us,
                ..CudaHtj2kDecodeStageTimings::default()
            },
        })
    }

    fn decode_htj2k_codeblocks_with_resources_and_pool_impl(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
        collect_stage_timings: bool,
        dequantize: bool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        self.inner.set_current()?;
        let output_bytes = output_words
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: output_words })?;
        let coefficients = pool.take(output_bytes)?;
        let coefficient_buffer = pooled_device_buffer(&coefficients)?;
        if htj2k_decode_needs_zero_fill(jobs, output_words)? {
            self.memset_d32(coefficient_buffer, 0, output_words)?;
        }
        if jobs.is_empty() {
            return Ok(CudaPooledHtj2kDecodeOutput {
                coefficients,
                execution: CudaExecutionStats::default(),
                statuses: Vec::new(),
                stage_timings: CudaHtj2kDecodeStageTimings::default(),
            });
        }

        let kernel_jobs = htj2k_kernel_jobs(jobs, resources.payload_len, output_words)?;
        let jobs_buffer = pool.upload(htj2k_jobs_as_bytes(&kernel_jobs))?;
        let status_buffer = pool.take(htj2k_statuses_byte_len(jobs.len())?)?;

        let has_refinement = jobs
            .iter()
            .any(|job| job.refinement_length > 0 || job.number_of_coding_passes > 1);
        let jobs_device = pooled_device_buffer(&jobs_buffer)?;
        let status_device = pooled_device_buffer(&status_buffer)?;
        let (ht_cleanup_us, dequant_us, kernel_dispatches) = if dequantize {
            let (ht_cleanup_us, dequant_us) = self.submit_htj2k_decode_and_dequantize(
                resources,
                coefficient_buffer,
                jobs_device,
                status_device,
                jobs.len(),
                collect_stage_timings,
            )?;
            (ht_cleanup_us, dequant_us, 2)
        } else {
            let ht_cleanup_us = self.submit_htj2k_decode_cleanup(
                resources,
                coefficient_buffer,
                jobs_device,
                status_device,
                jobs.len(),
                collect_stage_timings,
            )?;
            (ht_cleanup_us, 0, 1)
        };

        let mut statuses = vec![CudaHtj2kStatus::default(); jobs.len()];
        if let Err(error) = status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses)) {
            if !collect_stage_timings {
                let _ = self.synchronize();
            }
            return Err(error);
        }
        if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
            return Err(CudaError::KernelStatus {
                kernel: "signinum_htj2k_decode_codeblocks",
                code: status.code,
                detail: status.detail,
            });
        }

        Ok(CudaPooledHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats {
                kernel_dispatches,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: kernel_dispatches,
                hardware_decode: false,
            },
            statuses,
            stage_timings: CudaHtj2kDecodeStageTimings {
                ht_cleanup_us,
                ht_refine_us: if has_refinement { ht_cleanup_us } else { 0 },
                dequant_us,
                ..CudaHtj2kDecodeStageTimings::default()
            },
        })
    }

    fn submit_htj2k_decode_and_dequantize(
        &self,
        resources: &CudaHtj2kDecodeResources,
        coefficients: &CudaDeviceBuffer,
        jobs_buffer: &CudaDeviceBuffer,
        status_buffer: &CudaDeviceBuffer,
        job_count: usize,
        collect_stage_timings: bool,
    ) -> Result<(u128, u128), CudaError> {
        let ht_cleanup_us = self.submit_htj2k_decode_cleanup(
            resources,
            coefficients,
            jobs_buffer,
            status_buffer,
            job_count,
            collect_stage_timings,
        )?;
        let dequant_us = self.submit_htj2k_dequantize_htj2k_codeblocks(
            coefficients,
            jobs_buffer,
            job_count,
            collect_stage_timings,
        )?;
        Ok((ht_cleanup_us, dequant_us))
    }

    fn submit_htj2k_decode_cleanup(
        &self,
        resources: &CudaHtj2kDecodeResources,
        coefficients: &CudaDeviceBuffer,
        jobs_buffer: &CudaDeviceBuffer,
        status_buffer: &CudaDeviceBuffer,
        job_count: usize,
        collect_stage_timings: bool,
    ) -> Result<u128, CudaError> {
        let ((), ht_cleanup_us) = self.time_default_stream_named_us_if(
            collect_stage_timings,
            "signinum.htj2k.decode.cleanup",
            || {
                if !collect_stage_timings {
                    return self.launch_htj2k_decode_codeblocks_async(
                        resources.payload.buffer()?,
                        coefficients,
                        jobs_buffer,
                        &resources.tables.inner.vlc_table0,
                        &resources.tables.inner.vlc_table1,
                        &resources.tables.inner.uvlc_table0,
                        &resources.tables.inner.uvlc_table1,
                        status_buffer,
                        job_count,
                    );
                }
                self.launch_htj2k_decode_codeblocks(
                    resources.payload.buffer()?,
                    coefficients,
                    jobs_buffer,
                    &resources.tables.inner.vlc_table0,
                    &resources.tables.inner.vlc_table1,
                    &resources.tables.inner.uvlc_table0,
                    &resources.tables.inner.uvlc_table1,
                    status_buffer,
                    job_count,
                )
            },
        )?;
        Ok(ht_cleanup_us)
    }

    fn submit_htj2k_dequantize_htj2k_codeblocks(
        &self,
        coefficients: &CudaDeviceBuffer,
        jobs_buffer: &CudaDeviceBuffer,
        job_count: usize,
        collect_stage_timings: bool,
    ) -> Result<u128, CudaError> {
        let ((), dequant_us) = match self.time_default_stream_named_us_if(
            collect_stage_timings,
            "signinum.htj2k.decode.dequantize",
            || {
                if collect_stage_timings {
                    self.launch_j2k_dequantize_htj2k_codeblocks(
                        coefficients,
                        jobs_buffer,
                        job_count,
                    )
                } else {
                    self.launch_j2k_dequantize_htj2k_codeblocks_async(
                        coefficients,
                        jobs_buffer,
                        job_count,
                    )
                }
            },
        ) {
            Ok(result) => result,
            Err(error) => {
                if !collect_stage_timings {
                    let _ = self.synchronize();
                }
                return Err(error);
            }
        };
        Ok(dequant_us)
    }

    /// Dequantize HTJ2K code-block outputs that live in multiple device buffers
    /// with one CUDA dispatch.
    pub fn j2k_dequantize_htj2k_codeblocks_multi_device(
        &self,
        targets: &[CudaHtj2kDequantizeTarget<'_>],
    ) -> Result<CudaExecutionStats, CudaError> {
        let pool = self.buffer_pool();
        self.j2k_dequantize_htj2k_codeblocks_multi_device_with_pool(targets, &pool)
    }

    /// Dequantize HTJ2K code-block outputs that live in multiple device buffers
    /// with one CUDA dispatch, reusing caller-owned transient storage.
    pub fn j2k_dequantize_htj2k_codeblocks_multi_device_with_pool(
        &self,
        targets: &[CudaHtj2kDequantizeTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_impl(targets, pool, true)
    }

    /// Dequantize HTJ2K code-block outputs in multiple device buffers without
    /// CUDA event timings. The launch is still synchronized before returning
    /// so the pooled job upload cannot be reused while the kernel reads it.
    pub fn j2k_dequantize_htj2k_codeblocks_multi_device_untimed_with_pool(
        &self,
        targets: &[CudaHtj2kDequantizeTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_impl(targets, pool, true)
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
        self.launch_j2k_dequantize_htj2k_cleanup_jobs_multi_with_sync(
            pooled_device_buffer(jobs_buffer)?,
            cleanup.status_count,
            true,
        )?;
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }

    fn j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_impl(
        &self,
        targets: &[CudaHtj2kDequantizeTarget<'_>],
        pool: &CudaBufferPool,
        synchronize_each_launch: bool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = htj2k_dequantize_kernel_jobs(targets)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaExecutionStats::default());
        }
        let jobs_buffer = pool.upload(htj2k_dequantize_jobs_as_bytes(&kernel_jobs))?;
        self.launch_j2k_dequantize_htj2k_codeblocks_multi_with_sync(
            pooled_device_buffer(&jobs_buffer)?,
            kernel_jobs.len(),
            synchronize_each_launch,
        )?;
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }

    fn decode_empty_htj2k_codeblocks(
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

    /// Encode one HTJ2K cleanup-pass code block with CUDA.
    pub fn encode_htj2k_codeblock(
        &self,
        coefficients: &[i32],
        width: u32,
        height: u32,
        total_bitplanes: u8,
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodedCodeBlock, CudaError> {
        let resources = self.upload_htj2k_encode_resources(tables)?;
        self.encode_htj2k_codeblock_with_resources(
            coefficients,
            width,
            height,
            total_bitplanes,
            &resources,
        )
    }

    /// Encode one HTJ2K cleanup-pass code block with pre-uploaded lookup tables.
    pub fn encode_htj2k_codeblock_with_resources(
        &self,
        coefficients: &[i32],
        width: u32,
        height: u32,
        total_bitplanes: u8,
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlock, CudaError> {
        let expected_len = checked_image_words(width, height, 1)?;
        if coefficients.len() != expected_len {
            return Err(CudaError::LengthTooLarge {
                len: coefficients.len(),
            });
        }

        self.inner.set_current()?;
        let coefficient_buffer = self.upload_i32_pinned(coefficients)?;
        let output_buffer = self.allocate(HTJ2K_ENCODE_OUTPUT_CAPACITY)?;
        let params = CudaHtj2kEncodeParams {
            width,
            height,
            coefficient_stride: width,
            total_bitplanes: u32::from(total_bitplanes),
            output_capacity: u32::try_from(HTJ2K_ENCODE_OUTPUT_CAPACITY).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: HTJ2K_ENCODE_OUTPUT_CAPACITY,
                }
            })?,
            target_coding_passes: 1,
        };
        let params_buffer = self.upload(htj2k_encode_params_as_bytes(&params))?;
        let initial_status = CudaHtj2kEncodeStatus {
            code: HTJ2K_STATUS_UNSUPPORTED,
            ..CudaHtj2kEncodeStatus::default()
        };
        let status_buffer = self.upload(htj2k_encode_status_as_bytes(&initial_status))?;

        let ((), ht_encode_us) =
            self.time_default_stream_named_us("signinum.htj2k.encode.codeblock", || {
                self.launch_htj2k_encode_codeblock(
                    &coefficient_buffer,
                    &output_buffer,
                    &params_buffer,
                    &resources.vlc_table0,
                    &resources.vlc_table1,
                    &resources.uvlc_table,
                    &status_buffer,
                )
            })?;
        let (status, status_readback_us) = self.time_default_stream_named_us(
            "signinum.htj2k.encode.codeblock.status_readback",
            || {
                let mut status = CudaHtj2kEncodeStatus::default();
                status_buffer.copy_to_host(htj2k_encode_status_as_bytes_mut(&mut status))?;
                if !status.is_ok() {
                    return Err(CudaError::KernelStatus {
                        kernel: "signinum_htj2k_encode_codeblock",
                        code: status.code,
                        detail: status.detail,
                    });
                }
                Ok(status)
            },
        )?;
        let data_len = usize::try_from(status.data_len)
            .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
        if data_len > HTJ2K_ENCODE_OUTPUT_CAPACITY {
            return Err(CudaError::LengthTooLarge { len: data_len });
        }
        let (data, output_readback_us) = if data_len == 0 {
            (Vec::new(), 0)
        } else {
            self.time_default_stream_named_us(
                "signinum.htj2k.encode.codeblock.output_readback",
                || {
                    let mut data = vec![0u8; data_len];
                    output_buffer.copy_range_to_host(0, &mut data)?;
                    Ok(data)
                },
            )?
        };
        let stage_timings = CudaHtj2kEncodeStageTimings::from_parts(
            ht_encode_us,
            status_readback_us,
            0,
            output_readback_us,
        );

        Ok(CudaHtj2kEncodedCodeBlock {
            data,
            status,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
            stage_timings,
        })
    }

    /// Encode multiple HTJ2K cleanup-pass code blocks with one CUDA dispatch.
    pub fn encode_htj2k_codeblocks(
        &self,
        coefficients: &[i32],
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let resources = self.upload_htj2k_encode_resources(tables)?;
        self.encode_htj2k_codeblocks_with_resources(coefficients, jobs, &resources)
    }

    /// Encode multiple HTJ2K cleanup-pass code blocks with pre-uploaded lookup tables.
    pub fn encode_htj2k_codeblocks_with_resources(
        &self,
        coefficients: &[i32],
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        if jobs.is_empty() {
            return Ok(CudaHtj2kEncodedCodeBlocks {
                code_blocks: Vec::new(),
                execution: CudaExecutionStats::default(),
                stage_timings: CudaHtj2kEncodeStageTimings::default(),
            });
        }

        self.inner.set_current()?;
        let coefficient_buffer = self.upload_i32_pinned(coefficients)?;
        self.encode_htj2k_codeblocks_device_with_resources(
            &coefficient_buffer,
            coefficients.len(),
            jobs,
            resources,
        )
    }

    /// Encode multiple HTJ2K cleanup-pass code blocks from resident quantized coefficients.
    pub fn encode_htj2k_codeblocks_resident(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let resources = self.upload_htj2k_encode_resources(tables)?;
        self.encode_htj2k_codeblocks_resident_with_resources(
            coefficients,
            coefficient_count,
            jobs,
            &resources,
        )
    }

    /// Encode multiple cleanup-pass code blocks from resident coefficients with lookup table reuse.
    pub fn encode_htj2k_codeblocks_resident_with_resources(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let pool = self.buffer_pool();
        self.encode_htj2k_codeblocks_resident_with_resources_and_pool(
            coefficients,
            coefficient_count,
            jobs,
            resources,
            &pool,
        )
    }

    /// Encode multiple cleanup-pass code blocks from resident coefficients with
    /// lookup table reuse and caller-owned transient buffer reuse.
    pub fn encode_htj2k_codeblocks_resident_with_resources_and_pool(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        if jobs.is_empty() {
            return Ok(CudaHtj2kEncodedCodeBlocks {
                code_blocks: Vec::new(),
                execution: CudaExecutionStats::default(),
                stage_timings: CudaHtj2kEncodeStageTimings::default(),
            });
        }
        let available_coefficients = coefficients.typed_view::<i32>()?.len();
        if available_coefficients < coefficient_count {
            return Err(CudaError::OutputTooSmall {
                required: coefficient_count
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or(CudaError::LengthTooLarge {
                        len: coefficient_count,
                    })?,
                have: coefficients.byte_len(),
            });
        }

        let kernel_jobs = htj2k_encode_kernel_jobs(jobs, coefficient_count)?;
        self.inner.set_current()?;
        self.encode_htj2k_kernel_jobs_device_with_resources_and_pool(
            coefficients,
            &kernel_jobs,
            resources,
            pool,
        )
    }

    /// Encode multiple cleanup-pass code-block batches from independent
    /// resident coefficient buffers with one CUDA dispatch.
    pub fn encode_htj2k_codeblocks_multi_resident_with_resources_and_pool(
        &self,
        targets: &[CudaHtj2kEncodeResidentTarget<'_>],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        self.encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool(
            targets, resources, pool,
        )?
        .into_owned_code_blocks()
    }

    /// Encode multiple cleanup-pass code-block batches from independent resident
    /// coefficient buffers with one CUDA dispatch, returning one compact payload
    /// plus per-block ranges.
    pub fn encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool(
        &self,
        targets: &[CudaHtj2kEncodeResidentTarget<'_>],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kCompactEncodedCodeBlocks, CudaError> {
        let kernel_jobs = htj2k_encode_multi_input_kernel_jobs(targets)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaHtj2kCompactEncodedCodeBlocks {
                payload: Vec::new(),
                code_blocks: Vec::new(),
                execution: CudaExecutionStats::default(),
                stage_timings: CudaHtj2kEncodeStageTimings::default(),
            });
        }
        self.inner.set_current()?;
        self.encode_htj2k_multi_input_kernel_jobs_device_compact_with_resources_and_pool(
            &kernel_jobs,
            resources,
            pool,
        )
    }

    /// Encode cleanup-pass code blocks from strided resident coefficient regions.
    pub fn encode_htj2k_codeblock_regions_resident(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let resources = self.upload_htj2k_encode_resources(tables)?;
        self.encode_htj2k_codeblock_regions_resident_with_resources(
            coefficients,
            coefficient_count,
            jobs,
            &resources,
        )
    }

    /// Encode strided resident code-block regions with pre-uploaded lookup tables.
    pub fn encode_htj2k_codeblock_regions_resident_with_resources(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let pool = self.buffer_pool();
        self.encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
            coefficients,
            coefficient_count,
            jobs,
            resources,
            &pool,
        )
    }

    /// Encode strided resident code-block regions with pre-uploaded lookup
    /// tables and caller-owned transient buffer reuse.
    pub fn encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        if jobs.is_empty() {
            return Ok(CudaHtj2kEncodedCodeBlocks {
                code_blocks: Vec::new(),
                execution: CudaExecutionStats::default(),
                stage_timings: CudaHtj2kEncodeStageTimings::default(),
            });
        }
        let available_coefficients = coefficients.typed_view::<i32>()?.len();
        if available_coefficients < coefficient_count {
            return Err(CudaError::OutputTooSmall {
                required: coefficient_count
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or(CudaError::LengthTooLarge {
                        len: coefficient_count,
                    })?,
                have: coefficients.byte_len(),
            });
        }

        let kernel_jobs = htj2k_encode_region_kernel_jobs(jobs, coefficient_count)?;
        self.inner.set_current()?;
        self.encode_htj2k_kernel_jobs_device_with_resources_and_pool(
            coefficients,
            &kernel_jobs,
            resources,
            pool,
        )
    }

    fn encode_htj2k_codeblocks_device_with_resources(
        &self,
        coefficient_buffer: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let kernel_jobs = htj2k_encode_kernel_jobs(jobs, coefficient_count)?;
        self.encode_htj2k_kernel_jobs_device_with_resources(
            coefficient_buffer,
            &kernel_jobs,
            resources,
        )
    }

    #[allow(clippy::too_many_lines)]
    fn encode_htj2k_kernel_jobs_device_with_resources(
        &self,
        coefficient_buffer: &CudaDeviceBuffer,
        kernel_jobs: &[CudaHtj2kEncodeKernelJob],
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let pool = self.buffer_pool();
        self.encode_htj2k_kernel_jobs_device_with_resources_and_pool(
            coefficient_buffer,
            kernel_jobs,
            resources,
            &pool,
        )
    }

    #[allow(clippy::too_many_lines)]
    fn encode_htj2k_kernel_jobs_device_with_resources_and_pool(
        &self,
        coefficient_buffer: &CudaDeviceBuffer,
        kernel_jobs: &[CudaHtj2kEncodeKernelJob],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let output_bytes = kernel_jobs
            .last()
            .map(|job| {
                (job.output_offset as usize)
                    .checked_add(job.output_capacity as usize)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })
            })
            .transpose()?
            .unwrap_or(0);

        let jobs_buffer = pool.upload(htj2k_encode_jobs_as_bytes(kernel_jobs))?;
        let output_buffer = pool.take(output_bytes)?;
        let status_buffer = pool.take(htj2k_encode_statuses_byte_len(kernel_jobs.len())?)?;

        let ((), ht_encode_us) =
            self.time_default_stream_named_us("signinum.htj2k.encode.codeblocks", || {
                self.launch_htj2k_encode_codeblocks(
                    coefficient_buffer,
                    pooled_device_buffer(&output_buffer)?,
                    pooled_device_buffer(&jobs_buffer)?,
                    &resources.vlc_table0,
                    &resources.vlc_table1,
                    &resources.uvlc_table,
                    pooled_device_buffer(&status_buffer)?,
                    kernel_jobs.len(),
                )
            })?;
        let (statuses, status_readback_us) = self.time_default_stream_named_us(
            "signinum.htj2k.encode.codeblocks.status_readback",
            || {
                let mut statuses = vec![CudaHtj2kEncodeStatus::default(); kernel_jobs.len()];
                status_buffer.copy_to_host(htj2k_encode_statuses_as_bytes_mut(&mut statuses))?;
                if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
                    return Err(CudaError::KernelStatus {
                        kernel: "signinum_htj2k_encode_codeblocks",
                        code: status.code,
                        detail: status.detail,
                    });
                }
                Ok(statuses)
            },
        )?;

        let (compact_jobs, compact_output_bytes) =
            htj2k_encode_compact_jobs(&statuses, kernel_jobs)?;
        let compact_output_buffer = pool.take(compact_output_bytes)?;
        let compact_dispatched = compact_output_bytes != 0;
        let compact_us = if compact_dispatched {
            let compact_jobs_buffer =
                pool.upload(htj2k_encode_compact_jobs_as_bytes(&compact_jobs))?;
            let ((), compact_us) = self.time_default_stream_named_us(
                "signinum.htj2k.encode.codeblocks.compact",
                || {
                    self.launch_htj2k_compact_codeblocks(
                        pooled_device_buffer(&output_buffer)?,
                        pooled_device_buffer(&compact_output_buffer)?,
                        pooled_device_buffer(&compact_jobs_buffer)?,
                        compact_jobs.len(),
                    )
                },
            )?;
            compact_us
        } else {
            0
        };
        let (output, output_readback_us) = if compact_output_bytes == 0 {
            (Vec::new(), 0)
        } else {
            self.time_default_stream_named_us(
                "signinum.htj2k.encode.codeblocks.output_readback",
                || copy_pooled_bytes_to_vec_uninit(&compact_output_buffer, compact_output_bytes),
            )?
        };

        let mut code_blocks = statuses
            .into_iter()
            .zip(kernel_jobs.iter())
            .zip(compact_jobs.iter())
            .map(|((status, job), compact_job)| {
                let data_len = usize::try_from(status.data_len)
                    .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
                if data_len > job.output_capacity as usize {
                    return Err(CudaError::LengthTooLarge { len: data_len });
                }
                let start = compact_job.compact_offset as usize;
                let end = start
                    .checked_add(data_len)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
                if end > compact_output_bytes {
                    return Err(CudaError::LengthTooLarge { len: end });
                }
                let data = output[start..end].to_vec();
                Ok(CudaHtj2kEncodedCodeBlock {
                    data,
                    status,
                    execution: CudaExecutionStats {
                        kernel_dispatches: 1,
                        copy_kernel_dispatches: usize::from(compact_dispatched),
                        decode_kernel_dispatches: 0,
                        hardware_decode: false,
                    },
                    stage_timings: CudaHtj2kEncodeStageTimings::default(),
                })
            })
            .collect::<Result<Vec<_>, CudaError>>()?;
        let stage_timings = CudaHtj2kEncodeStageTimings::from_parts(
            ht_encode_us,
            status_readback_us,
            compact_us,
            output_readback_us,
        );
        for block in &mut code_blocks {
            block.stage_timings = stage_timings;
        }
        let copy_kernel_dispatches =
            usize::from(code_blocks.iter().any(|block| !block.data().is_empty()));

        Ok(CudaHtj2kEncodedCodeBlocks {
            code_blocks,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
            stage_timings,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn encode_htj2k_multi_input_kernel_jobs_device_compact_with_resources_and_pool(
        &self,
        kernel_jobs: &[CudaHtj2kEncodeMultiInputKernelJob],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kCompactEncodedCodeBlocks, CudaError> {
        let output_bytes = kernel_jobs
            .last()
            .map(|job| {
                (job.output_offset as usize)
                    .checked_add(job.output_capacity as usize)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })
            })
            .transpose()?
            .unwrap_or(0);

        let jobs_buffer = pool.upload(htj2k_encode_multi_input_jobs_as_bytes(kernel_jobs))?;
        let output_buffer = pool.take(output_bytes)?;
        let status_buffer = pool.take(htj2k_encode_statuses_byte_len(kernel_jobs.len())?)?;
        let cleanup_only = kernel_jobs.iter().all(|job| job.target_coding_passes == 1);
        let cleanup_only_64 = cleanup_only
            && kernel_jobs
                .iter()
                .all(|job| job.width == 64 && job.height == 64 && job.coefficient_stride == 64);
        let status_kernel = if cleanup_only_64 {
            "signinum_htj2k_encode_codeblocks_multi_input_cleanup_64"
        } else if cleanup_only {
            "signinum_htj2k_encode_codeblocks_multi_input_cleanup"
        } else {
            "signinum_htj2k_encode_codeblocks_multi_input"
        };

        let ((), ht_encode_us) = self.time_default_stream_named_us(
            "signinum.htj2k.encode.codeblocks.multi_input",
            || {
                if cleanup_only_64 {
                    self.launch_htj2k_encode_codeblocks_multi_input_cleanup_64(
                        pooled_device_buffer(&output_buffer)?,
                        pooled_device_buffer(&jobs_buffer)?,
                        &resources.vlc_table0,
                        &resources.vlc_table1,
                        &resources.uvlc_table,
                        pooled_device_buffer(&status_buffer)?,
                        kernel_jobs.len(),
                    )
                } else if cleanup_only {
                    self.launch_htj2k_encode_codeblocks_multi_input_cleanup(
                        pooled_device_buffer(&output_buffer)?,
                        pooled_device_buffer(&jobs_buffer)?,
                        &resources.vlc_table0,
                        &resources.vlc_table1,
                        &resources.uvlc_table,
                        pooled_device_buffer(&status_buffer)?,
                        kernel_jobs.len(),
                    )
                } else {
                    self.launch_htj2k_encode_codeblocks_multi_input(
                        pooled_device_buffer(&output_buffer)?,
                        pooled_device_buffer(&jobs_buffer)?,
                        &resources.vlc_table0,
                        &resources.vlc_table1,
                        &resources.uvlc_table,
                        pooled_device_buffer(&status_buffer)?,
                        kernel_jobs.len(),
                    )
                }
            },
        )?;
        let (statuses, status_readback_us) = self.time_default_stream_named_us(
            "signinum.htj2k.encode.codeblocks.multi_input.status_readback",
            || {
                let mut statuses = vec![CudaHtj2kEncodeStatus::default(); kernel_jobs.len()];
                status_buffer.copy_to_host(htj2k_encode_statuses_as_bytes_mut(&mut statuses))?;
                if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
                    return Err(CudaError::KernelStatus {
                        kernel: status_kernel,
                        code: status.code,
                        detail: status.detail,
                    });
                }
                Ok(statuses)
            },
        )?;

        let (compact_jobs, compact_output_bytes) =
            htj2k_encode_compact_jobs_multi_input(&statuses, kernel_jobs)?;
        let compact_output_buffer = pool.take(compact_output_bytes)?;
        let compact_dispatched = compact_output_bytes != 0;
        let compact_us = if compact_dispatched {
            let compact_jobs_buffer =
                pool.upload(htj2k_encode_compact_jobs_as_bytes(&compact_jobs))?;
            let ((), compact_us) = self.time_default_stream_named_us(
                "signinum.htj2k.encode.codeblocks.multi_input.compact",
                || {
                    self.launch_htj2k_compact_codeblocks(
                        pooled_device_buffer(&output_buffer)?,
                        pooled_device_buffer(&compact_output_buffer)?,
                        pooled_device_buffer(&compact_jobs_buffer)?,
                        compact_jobs.len(),
                    )
                },
            )?;
            compact_us
        } else {
            0
        };
        let (output, output_readback_us) = if compact_output_bytes == 0 {
            (Vec::new(), 0)
        } else {
            self.time_default_stream_named_us(
                "signinum.htj2k.encode.codeblocks.multi_input.output_readback",
                || copy_pooled_bytes_to_vec_uninit(&compact_output_buffer, compact_output_bytes),
            )?
        };

        let mut code_blocks = statuses
            .into_iter()
            .zip(kernel_jobs.iter())
            .zip(compact_jobs.iter())
            .map(|((status, job), compact_job)| {
                let data_len = usize::try_from(status.data_len)
                    .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
                if data_len > job.output_capacity as usize {
                    return Err(CudaError::LengthTooLarge { len: data_len });
                }
                let start = compact_job.compact_offset as usize;
                let end = start
                    .checked_add(data_len)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
                if end > compact_output_bytes {
                    return Err(CudaError::LengthTooLarge { len: end });
                }
                Ok(CudaHtj2kCompactEncodedCodeBlock {
                    payload_range: start..end,
                    status,
                    execution: CudaExecutionStats {
                        kernel_dispatches: 1,
                        copy_kernel_dispatches: usize::from(compact_dispatched),
                        decode_kernel_dispatches: 0,
                        hardware_decode: false,
                    },
                    stage_timings: CudaHtj2kEncodeStageTimings::default(),
                })
            })
            .collect::<Result<Vec<_>, CudaError>>()?;
        let stage_timings = CudaHtj2kEncodeStageTimings::from_parts(
            ht_encode_us,
            status_readback_us,
            compact_us,
            output_readback_us,
        );
        for block in &mut code_blocks {
            block.stage_timings = stage_timings;
        }
        let copy_kernel_dispatches = usize::from(!output.is_empty());

        Ok(CudaHtj2kCompactEncodedCodeBlocks {
            payload: output,
            code_blocks,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
            stage_timings,
        })
    }

    /// Packetize HTJ2K code-block payloads with CUDA.
    pub fn packetize_htj2k_cleanup_packets(
        &self,
        payload: &[u8],
        packets: &[CudaHtj2kPacketizationPacket],
        subbands: &[CudaHtj2kPacketizationSubband],
        blocks: &[CudaHtj2kPacketizationBlock],
    ) -> Result<CudaHtj2kPacketizedTile, CudaError> {
        self.packetize_htj2k_cleanup_packets_with_tag_state(
            payload,
            packets,
            subbands,
            blocks,
            &[],
            &[],
        )
    }

    /// Packetize HTJ2K code-block payloads with CUDA using caller-provided tag-tree state.
    pub fn packetize_htj2k_cleanup_packets_with_tag_state(
        &self,
        payload: &[u8],
        packets: &[CudaHtj2kPacketizationPacket],
        subbands: &[CudaHtj2kPacketizationSubband],
        blocks: &[CudaHtj2kPacketizationBlock],
        subband_tag_states: &[CudaHtj2kPacketizationSubbandTagState],
        tag_nodes: &[CudaHtj2kPacketizationTagNodeState],
    ) -> Result<CudaHtj2kPacketizedTile, CudaError> {
        self.inner.set_current()?;
        if !HTJ2K_ENCODE_PTX_BUILT_FROM_CUDA
            && blocks.iter().any(|block| block.num_coding_passes > 1)
        {
            return Err(CudaError::InvalidArgument {
                message: "multi-pass HTJ2K packetization requires CUDA PTX rebuilt from htj2k_encode_kernels.cu".to_string(),
            });
        }
        let kernel_packets =
            htj2k_packetization_kernel_packets(packets, subbands, blocks, payload.len())?;
        validate_htj2k_packetization_tag_state(subbands, subband_tag_states, tag_nodes)?;
        let total_output = kernel_packets.iter().try_fold(0usize, |acc, packet| {
            let end = usize::try_from(packet.output_offset)
                .ok()
                .and_then(|offset| offset.checked_add(packet.output_capacity as usize))
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            Ok::<usize, CudaError>(acc.max(end))
        })?;
        let output_buffer = self.allocate(total_output)?;
        if packets.is_empty() {
            return Ok(CudaHtj2kPacketizedTile {
                data: Vec::new(),
                statuses: Vec::new(),
                execution: CudaExecutionStats::default(),
                stage_timings: CudaHtj2kPacketizationStageTimings::default(),
            });
        }

        let payload_buffer = self.upload_pinned(payload)?;
        let packet_buffer = self.upload(htj2k_packetization_packets_as_bytes(&kernel_packets))?;
        let subband_buffer = self.upload(htj2k_packetization_subbands_as_bytes(subbands))?;
        let block_buffer = self.upload(htj2k_packetization_blocks_as_bytes(blocks))?;
        let subband_tag_state_buffer = self.upload(
            htj2k_packetization_subband_tag_states_as_bytes(subband_tag_states),
        )?;
        let tag_node_buffer = self.upload(htj2k_packetization_tag_nodes_as_bytes(tag_nodes))?;
        let initial_statuses = vec![
            CudaHtj2kPacketizationStatus {
                code: HTJ2K_STATUS_UNSUPPORTED,
                ..CudaHtj2kPacketizationStatus::default()
            };
            packets.len()
        ];
        let status_buffer =
            self.upload(htj2k_packetization_statuses_as_bytes(&initial_statuses))?;

        let ((), packetize_us) =
            self.time_default_stream_named_us("signinum.htj2k.encode.packetize", || {
                self.launch_htj2k_packetize_cleanup(
                    &payload_buffer,
                    payload.len(),
                    &packet_buffer,
                    &subband_buffer,
                    &block_buffer,
                    &subband_tag_state_buffer,
                    &tag_node_buffer,
                    subband_tag_states.len(),
                    tag_nodes.len(),
                    &output_buffer,
                    &status_buffer,
                    packets.len(),
                )
            })?;
        let stage_timings = CudaHtj2kPacketizationStageTimings { packetize_us };

        let mut statuses = vec![CudaHtj2kPacketizationStatus::default(); packets.len()];
        status_buffer.copy_to_host(htj2k_packetization_statuses_as_bytes_mut(&mut statuses))?;
        if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
            return Err(CudaError::KernelStatus {
                kernel: "signinum_htj2k_packetize_cleanup",
                code: status.code,
                detail: status.detail,
            });
        }

        let mut data = Vec::new();
        for (packet, status) in kernel_packets.iter().zip(&statuses) {
            if status.output_len > packet.output_capacity {
                return Err(CudaError::LengthTooLarge {
                    len: status.output_len as usize,
                });
            }
            let start = packet.output_offset as usize;
            let end = start
                .checked_add(status.output_len as usize)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if end > output_buffer.byte_len() {
                return Err(CudaError::LengthTooLarge { len: end });
            }
            let previous_len = data.len();
            data.resize(previous_len + status.output_len as usize, 0);
            output_buffer.copy_range_to_host(start, &mut data[previous_len..])?;
        }

        Ok(CudaHtj2kPacketizedTile {
            data,
            statuses,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
            stage_timings,
        })
    }

    /// Apply one inverse JPEG 2000 DWT decomposition to device coefficient bands.
    pub fn j2k_inverse_dwt_single_device(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
    ) -> Result<CudaKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_impl(ll, hl, lh, hh, job, true)
    }

    /// Apply one inverse JPEG 2000 DWT decomposition without per-kernel synchronizes.
    pub fn j2k_inverse_dwt_single_device_untimed(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
    ) -> Result<CudaKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_impl(ll, hl, lh, hh, job, false)
    }

    /// Apply one inverse JPEG 2000 DWT decomposition with caller-owned
    /// transient buffer reuse.
    pub fn j2k_inverse_dwt_single_device_with_pool(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_with_pool_impl(ll, hl, lh, hh, job, true, pool)
    }

    /// Apply one inverse JPEG 2000 DWT decomposition with caller-owned
    /// transient buffer reuse and without per-kernel synchronizes.
    pub fn j2k_inverse_dwt_single_device_untimed_with_pool(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_with_pool_impl(ll, hl, lh, hh, job, false, pool)
    }

    /// Apply inverse JPEG 2000 DWT decompositions for multiple independent
    /// targets using one dispatch per parallel stage.
    pub fn j2k_inverse_dwt_batch_device_with_pool(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_inverse_dwt_batch_device_with_pool_impl(targets, pool, true)
    }

    /// Apply inverse JPEG 2000 DWT decompositions for multiple independent
    /// targets without per-stage synchronizes.
    pub fn j2k_inverse_dwt_batch_device_untimed_with_pool(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_inverse_dwt_batch_device_with_pool_impl(targets, pool, false)
    }

    /// Enqueue batched inverse JPEG 2000 DWT decompositions without
    /// synchronizing. The returned value must be kept live until the default
    /// stream has been synchronized by the caller.
    pub fn j2k_inverse_dwt_batch_device_enqueue_with_pool(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaQueuedExecution, CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = j2k_idwt_multi_kernel_jobs(targets)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaQueuedExecution {
                resources: Vec::new(),
                execution: CudaExecutionStats::default(),
            });
        }
        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&kernel_jobs))?;
        let jobs_device = pooled_device_buffer(&jobs_buffer)?;
        let max_width = kernel_jobs
            .iter()
            .map(|job| job.job.rect.x1.saturating_sub(job.job.rect.x0))
            .max()
            .unwrap_or(0);
        let max_height = kernel_jobs
            .iter()
            .map(|job| job.job.rect.y1.saturating_sub(job.job.rect.y0))
            .max()
            .unwrap_or(0);
        let kernel_mode = idwt_batch_kernel_mode(&kernel_jobs, max_width, max_height);
        let interleave_horizontal_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                .launch_j2k_idwt_interleave_horizontal_53_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    false,
                ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    false,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self
                .launch_j2k_idwt_interleave_horizontal_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    false,
                ),
        };
        if let Err(error) = interleave_horizontal_result {
            let _ = self.synchronize();
            return Err(error);
        }
        let vertical_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self.launch_j2k_idwt_vertical_53_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                false,
            ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_vertical_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    false,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self.launch_j2k_idwt_vertical_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                false,
            ),
        };
        if let Err(error) = vertical_result {
            let _ = self.synchronize();
            return Err(error);
        }

        Ok(CudaQueuedExecution {
            resources: vec![jobs_buffer],
            execution: CudaExecutionStats {
                kernel_dispatches: 2,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 2,
                hardware_decode: false,
            },
        })
    }

    /// Enqueue a sequence of batched inverse JPEG 2000 DWT stages while
    /// uploading all stage job metadata in one device buffer. The returned
    /// value must be kept live until the default stream has been synchronized
    /// by the caller.
    #[allow(clippy::too_many_lines)]
    pub fn j2k_inverse_dwt_batch_sequence_enqueue_with_pool(
        &self,
        target_batches: &[&[CudaJ2kIdwtTarget<'_>]],
        pool: &CudaBufferPool,
    ) -> Result<CudaQueuedExecution, CudaError> {
        self.inner.set_current()?;
        let mut all_jobs = Vec::new();
        let mut batches = Vec::new();
        for targets in target_batches {
            let kernel_jobs = j2k_idwt_multi_kernel_jobs(targets)?;
            if kernel_jobs.is_empty() {
                continue;
            }
            let start = all_jobs.len();
            let count = kernel_jobs.len();
            let max_width = kernel_jobs
                .iter()
                .map(|job| job.job.rect.x1.saturating_sub(job.job.rect.x0))
                .max()
                .unwrap_or(0);
            let max_height = kernel_jobs
                .iter()
                .map(|job| job.job.rect.y1.saturating_sub(job.job.rect.y0))
                .max()
                .unwrap_or(0);
            let kernel_mode = idwt_batch_kernel_mode(&kernel_jobs, max_width, max_height);
            all_jobs.extend(kernel_jobs);
            batches.push((start, count, max_width, max_height, kernel_mode));
        }
        if all_jobs.is_empty() {
            return Ok(CudaQueuedExecution {
                resources: Vec::new(),
                execution: CudaExecutionStats::default(),
            });
        }

        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&all_jobs))?;
        let jobs_base = pooled_device_buffer(&jobs_buffer)?.device_ptr();
        let job_size = std::mem::size_of::<CudaJ2kIdwtMultiKernelJob>();
        let mut kernel_dispatches = 0usize;
        let trace_enabled = cuda_idwt_trace_enabled();
        for (stage_index, (start, count, max_width, max_height, kernel_mode)) in
            batches.into_iter().enumerate()
        {
            let byte_offset = start
                .checked_mul(job_size)
                .ok_or(CudaError::LengthTooLarge { len: start })?;
            let jobs_ptr = jobs_base
                .checked_add(byte_offset as u64)
                .ok_or(CudaError::LengthTooLarge { len: byte_offset })?;
            let trace_start = if trace_enabled {
                let event = self.create_event()?;
                event.record_default_stream()?;
                Some(event)
            } else {
                None
            };
            let interleave_horizontal_result = match kernel_mode {
                CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                    .launch_j2k_idwt_interleave_horizontal_53_multi_ptr(
                        jobs_ptr,
                        max_height as usize,
                        count,
                        false,
                    ),
                CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                    .launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
                        jobs_ptr,
                        max_width as usize,
                        max_height as usize,
                        count,
                        false,
                    ),
                CudaJ2kIdwtBatchKernelMode::Generic => self
                    .launch_j2k_idwt_interleave_horizontal_multi_ptr(
                        jobs_ptr,
                        max_height as usize,
                        count,
                        false,
                    ),
            };
            if let Err(error) = interleave_horizontal_result {
                let _ = self.synchronize();
                return Err(error);
            }
            kernel_dispatches = kernel_dispatches.saturating_add(1);

            let vertical_result = match kernel_mode {
                CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                    .launch_j2k_idwt_vertical_53_multi_ptr(
                        jobs_ptr,
                        max_width as usize,
                        count,
                        false,
                    ),
                CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                    .launch_j2k_idwt_vertical_97_multi_ptr(
                        jobs_ptr,
                        max_width as usize,
                        max_height as usize,
                        count,
                        false,
                    ),
                CudaJ2kIdwtBatchKernelMode::Generic => self.launch_j2k_idwt_vertical_multi_ptr(
                    jobs_ptr,
                    max_width as usize,
                    count,
                    false,
                ),
            };
            if let Err(error) = vertical_result {
                let _ = self.synchronize();
                return Err(error);
            }
            kernel_dispatches = kernel_dispatches.saturating_add(1);
            if let Some(trace_start) = trace_start {
                let trace_end = self.create_event()?;
                trace_end.record_default_stream()?;
                trace_end.synchronize()?;
                let elapsed_us = elapsed_event_us_ceil(&trace_start, &trace_end)?;
                let end = start.saturating_add(count);
                let row = idwt_batch_trace_row(
                    stage_index,
                    &all_jobs[start..end],
                    max_width,
                    max_height,
                    kernel_mode,
                    elapsed_us,
                );
                eprintln!("{}", format_idwt_batch_trace_row(row));
            }
        }

        Ok(CudaQueuedExecution {
            resources: vec![jobs_buffer],
            execution: CudaExecutionStats {
                kernel_dispatches,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: kernel_dispatches,
                hardware_decode: false,
            },
        })
    }

    fn j2k_inverse_dwt_batch_device_with_pool_impl(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
        synchronize_each_launch: bool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = j2k_idwt_multi_kernel_jobs(targets)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaExecutionStats::default());
        }
        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&kernel_jobs))?;
        let jobs_device = pooled_device_buffer(&jobs_buffer)?;
        let max_width = kernel_jobs
            .iter()
            .map(|job| job.job.rect.x1.saturating_sub(job.job.rect.x0))
            .max()
            .unwrap_or(0);
        let max_height = kernel_jobs
            .iter()
            .map(|job| job.job.rect.y1.saturating_sub(job.job.rect.y0))
            .max()
            .unwrap_or(0);
        let kernel_mode = idwt_batch_kernel_mode(&kernel_jobs, max_width, max_height);
        let interleave_horizontal_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                .launch_j2k_idwt_interleave_horizontal_53_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self
                .launch_j2k_idwt_interleave_horizontal_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    synchronize_each_launch,
                ),
        };
        if let Err(error) = interleave_horizontal_result {
            if !synchronize_each_launch {
                let _ = self.synchronize();
            }
            return Err(error);
        }
        let vertical_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self.launch_j2k_idwt_vertical_53_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                synchronize_each_launch,
            ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_vertical_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self.launch_j2k_idwt_vertical_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                synchronize_each_launch,
            ),
        };
        if let Err(error) = vertical_result {
            if !synchronize_each_launch {
                let _ = self.synchronize();
            }
            return Err(error);
        }
        if !synchronize_each_launch {
            self.synchronize()?;
        }

        Ok(CudaExecutionStats {
            kernel_dispatches: 2,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 2,
            hardware_decode: false,
        })
    }

    fn j2k_inverse_dwt_single_device_impl(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
        synchronize_each_launch: bool,
    ) -> Result<CudaKernelOutput, CudaError> {
        let width = job.rect.x1.saturating_sub(job.rect.x0);
        let height = job.rect.y1.saturating_sub(job.rect.y0);
        let output_words = checked_image_words(width, height, 1)?;
        let output = self.allocate(output_words * std::mem::size_of::<f32>())?;
        if output_words == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }

        let job_buffer = self.upload(idwt_job_as_bytes(&job))?;
        let (horizontal_kernel, vertical_kernel) = if job.irreversible97 == 0 {
            (
                CudaKernel::J2kIdwtHorizontal53,
                CudaKernel::J2kIdwtVertical53,
            )
        } else {
            (
                CudaKernel::J2kIdwtHorizontal97,
                CudaKernel::J2kIdwtVertical97,
            )
        };
        if synchronize_each_launch {
            self.launch_j2k_idwt_interleave([ll, hl, lh, hh], &output, &job_buffer, width, height)?;
            self.launch_j2k_idwt_horizontal(
                horizontal_kernel,
                &output,
                &job_buffer,
                height as usize,
            )?;
            self.launch_j2k_idwt_vertical(vertical_kernel, &output, &job_buffer, width as usize)?;
        } else {
            self.launch_j2k_idwt_interleave_async(
                [ll, hl, lh, hh],
                &output,
                &job_buffer,
                width,
                height,
            )?;
            if let Err(error) = self.launch_j2k_idwt_horizontal_async(
                horizontal_kernel,
                &output,
                &job_buffer,
                height as usize,
            ) {
                let _ = self.synchronize();
                return Err(error);
            }
            if let Err(error) = self.launch_j2k_idwt_vertical_async(
                vertical_kernel,
                &output,
                &job_buffer,
                width as usize,
            ) {
                let _ = self.synchronize();
                return Err(error);
            }
            self.synchronize()?;
        }
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 3,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 3,
                hardware_decode: false,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn j2k_inverse_dwt_single_device_with_pool_impl(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
        synchronize_each_launch: bool,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledKernelOutput, CudaError> {
        let width = job.rect.x1.saturating_sub(job.rect.x0);
        let height = job.rect.y1.saturating_sub(job.rect.y0);
        let output_words = checked_image_words(width, height, 1)?;
        let output = pool.take(output_words * std::mem::size_of::<f32>())?;
        let output_buffer = pooled_device_buffer(&output)?;
        if output_words == 0 {
            return Ok(CudaPooledKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }

        let job_buffer = pool.upload(idwt_job_as_bytes(&job))?;
        let job_device_buffer = pooled_device_buffer(&job_buffer)?;
        let (horizontal_kernel, vertical_kernel) = if job.irreversible97 == 0 {
            (
                CudaKernel::J2kIdwtHorizontal53,
                CudaKernel::J2kIdwtVertical53,
            )
        } else {
            (
                CudaKernel::J2kIdwtHorizontal97,
                CudaKernel::J2kIdwtVertical97,
            )
        };
        if synchronize_each_launch {
            self.launch_j2k_idwt_interleave(
                [ll, hl, lh, hh],
                output_buffer,
                job_device_buffer,
                width,
                height,
            )?;
            self.launch_j2k_idwt_horizontal(
                horizontal_kernel,
                output_buffer,
                job_device_buffer,
                height as usize,
            )?;
            self.launch_j2k_idwt_vertical(
                vertical_kernel,
                output_buffer,
                job_device_buffer,
                width as usize,
            )?;
        } else {
            self.launch_j2k_idwt_interleave_async(
                [ll, hl, lh, hh],
                output_buffer,
                job_device_buffer,
                width,
                height,
            )?;
            if let Err(error) = self.launch_j2k_idwt_horizontal_async(
                horizontal_kernel,
                output_buffer,
                job_device_buffer,
                height as usize,
            ) {
                let _ = self.synchronize();
                return Err(error);
            }
            if let Err(error) = self.launch_j2k_idwt_vertical_async(
                vertical_kernel,
                output_buffer,
                job_device_buffer,
                width as usize,
            ) {
                let _ = self.synchronize();
                return Err(error);
            }
            self.synchronize()?;
        }
        Ok(CudaPooledKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 3,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 3,
                hardware_decode: false,
            },
        })
    }

    /// Store a device f32 component plane as tightly packed Gray8 pixels.
    pub fn j2k_store_gray8_device(
        &self,
        input: &CudaDeviceBuffer,
        job: CudaJ2kStoreGray8Job,
    ) -> Result<CudaKernelOutput, CudaError> {
        let output_words = checked_image_words(job.output_width, job.output_height, 1)?;
        let output = self.allocate(output_words)?;
        if output_words == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
        if pixels == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        validate_store_rgb8_plane(
            input,
            job.input_width,
            job.source_x,
            job.source_y,
            job.copy_width,
            job.copy_height,
        )?;

        let job_buffer = self.upload(store_gray8_job_as_bytes(&job))?;
        self.launch_j2k_store_gray8(input, &output, &job_buffer, pixels)?;
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Store a device f32 component plane as tightly packed Gray16 pixels.
    pub fn j2k_store_gray16_device(
        &self,
        input: &CudaDeviceBuffer,
        job: CudaJ2kStoreGray16Job,
    ) -> Result<CudaKernelOutput, CudaError> {
        let output_words = checked_image_words(job.output_width, job.output_height, 1)?;
        let output = self.allocate(
            output_words
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or(CudaError::LengthTooLarge { len: output_words })?,
        )?;
        if output_words == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
        if pixels == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        validate_store_rgb8_plane(
            input,
            job.input_width,
            job.source_x,
            job.source_y,
            job.copy_width,
            job.copy_height,
        )?;

        let job_buffer = self.upload(store_gray16_job_as_bytes(&job))?;
        self.launch_j2k_store_gray16(input, &output, &job_buffer, pixels)?;
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Apply inverse RCT/ICT in place on three device f32 component planes.
    pub fn j2k_inverse_mct_device(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: CudaJ2kInverseMctJob,
    ) -> Result<CudaExecutionStats, CudaError> {
        let bytes = (job.len as usize)
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if bytes > plane0.byte_len() || bytes > plane1.byte_len() || bytes > plane2.byte_len() {
            return Err(CudaError::LengthTooLarge { len: bytes });
        }
        if job.len == 0 {
            return Ok(CudaExecutionStats::default());
        }

        let job_buffer = self.upload(inverse_mct_job_as_bytes(&job))?;
        self.launch_j2k_inverse_mct(plane0, plane1, plane2, &job_buffer, job.len as usize)?;
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }

    /// Store three device f32 component planes as tightly packed RGB8/RGBA8.
    pub fn j2k_store_rgb8_device(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: CudaJ2kStoreRgb8Job,
    ) -> Result<CudaKernelOutput, CudaError> {
        let channels = if job.rgba == 0 { 3 } else { 4 };
        let output_bytes = checked_image_words(job.output_width, job.output_height, channels)?;
        let output = self.allocate(output_bytes)?;
        let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
        if output_bytes == 0 || pixels == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        validate_store_rgb8_plane(
            plane0,
            job.input_width0,
            job.source_x0,
            job.source_y0,
            job.copy_width,
            job.copy_height,
        )?;
        validate_store_rgb8_plane(
            plane1,
            job.input_width1,
            job.source_x1,
            job.source_y1,
            job.copy_width,
            job.copy_height,
        )?;
        validate_store_rgb8_plane(
            plane2,
            job.input_width2,
            job.source_x2,
            job.source_y2,
            job.copy_width,
            job.copy_height,
        )?;
        let dst_end = (job.output_y as usize)
            .checked_add(job.copy_height as usize)
            .and_then(|end_y| {
                (job.output_x as usize)
                    .checked_add(job.copy_width as usize)
                    .map(|end_x| (end_x, end_y))
            })
            .ok_or(CudaError::LengthTooLarge { len: output_bytes })?;
        if dst_end.0 > job.output_width as usize || dst_end.1 > job.output_height as usize {
            return Err(CudaError::LengthTooLarge { len: output_bytes });
        }

        let job_buffer = self.upload(store_rgb8_job_as_bytes(&job))?;
        self.launch_j2k_store_rgb8(plane0, plane1, plane2, &output, &job_buffer, pixels)?;
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Store three device f32 component planes as tightly packed RGB16/RGBA16.
    pub fn j2k_store_rgb16_device(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: CudaJ2kStoreRgb16Job,
    ) -> Result<CudaKernelOutput, CudaError> {
        let channels = if job.rgba == 0 { 3 } else { 4 };
        let output_samples = checked_image_words(job.output_width, job.output_height, channels)?;
        let output_bytes = output_samples
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or(CudaError::LengthTooLarge {
                len: output_samples,
            })?;
        let output = self.allocate(output_bytes)?;
        let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
        if output_bytes == 0 || pixels == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        validate_store_rgb8_plane(
            plane0,
            job.input_width0,
            job.source_x0,
            job.source_y0,
            job.copy_width,
            job.copy_height,
        )?;
        validate_store_rgb8_plane(
            plane1,
            job.input_width1,
            job.source_x1,
            job.source_y1,
            job.copy_width,
            job.copy_height,
        )?;
        validate_store_rgb8_plane(
            plane2,
            job.input_width2,
            job.source_x2,
            job.source_y2,
            job.copy_width,
            job.copy_height,
        )?;
        let dst_end = (job.output_y as usize)
            .checked_add(job.copy_height as usize)
            .and_then(|end_y| {
                (job.output_x as usize)
                    .checked_add(job.copy_width as usize)
                    .map(|end_x| (end_x, end_y))
            })
            .ok_or(CudaError::LengthTooLarge { len: output_bytes })?;
        if dst_end.0 > job.output_width as usize || dst_end.1 > job.output_height as usize {
            return Err(CudaError::LengthTooLarge { len: output_bytes });
        }

        let job_buffer = self.upload(store_rgb16_job_as_bytes(&job))?;
        self.launch_j2k_store_rgb16(plane0, plane1, plane2, &output, &job_buffer, pixels)?;
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Apply inverse RCT/ICT and store tightly packed RGB8/RGBA8 in one dispatch.
    pub fn j2k_store_rgb8_mct_device(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: CudaJ2kStoreRgb8MctJob,
    ) -> Result<CudaKernelOutput, CudaError> {
        let batch = self.j2k_store_rgb8_mct_batch_device(&[CudaJ2kStoreRgb8MctTarget {
            plane0,
            plane1,
            plane2,
            job,
        }])?;
        let (mut outputs, execution) = batch.into_parts();
        let buffer = outputs.pop().ok_or_else(|| CudaError::InvalidArgument {
            message: "single RGB8 MCT batch store returned no output".to_string(),
        })?;
        Ok(CudaKernelOutput { buffer, execution })
    }

    /// Apply inverse RCT/ICT and store multiple tightly packed RGB8/RGBA8 images
    /// in one dispatch.
    pub fn j2k_store_rgb8_mct_batch_device(
        &self,
        targets: &[CudaJ2kStoreRgb8MctTarget<'_>],
    ) -> Result<CudaKernelBatchOutput, CudaError> {
        if targets.is_empty() {
            return Ok(CudaKernelBatchOutput {
                outputs: Vec::new(),
                execution: CudaExecutionStats::default(),
            });
        }

        let mut outputs = Vec::with_capacity(targets.len());
        let mut kernel_jobs = Vec::with_capacity(targets.len());
        let mut max_pixels = 0usize;
        for target in targets {
            let store = target.job.store;
            let channels = if store.rgba == 0 { 3 } else { 4 };
            let output_bytes =
                checked_image_words(store.output_width, store.output_height, channels)?;
            let output = self.allocate(output_bytes)?;
            let pixels = checked_image_words(store.copy_width, store.copy_height, 1)?;
            if output_bytes != 0 && pixels != 0 {
                validate_store_rgb8_plane(
                    target.plane0,
                    store.input_width0,
                    store.source_x0,
                    store.source_y0,
                    store.copy_width,
                    store.copy_height,
                )?;
                validate_store_rgb8_plane(
                    target.plane1,
                    store.input_width1,
                    store.source_x1,
                    store.source_y1,
                    store.copy_width,
                    store.copy_height,
                )?;
                validate_store_rgb8_plane(
                    target.plane2,
                    store.input_width2,
                    store.source_x2,
                    store.source_y2,
                    store.copy_width,
                    store.copy_height,
                )?;
                let dst_end = (store.output_y as usize)
                    .checked_add(store.copy_height as usize)
                    .and_then(|end_y| {
                        (store.output_x as usize)
                            .checked_add(store.copy_width as usize)
                            .map(|end_x| (end_x, end_y))
                    })
                    .ok_or(CudaError::LengthTooLarge { len: output_bytes })?;
                if dst_end.0 > store.output_width as usize
                    || dst_end.1 > store.output_height as usize
                {
                    return Err(CudaError::LengthTooLarge { len: output_bytes });
                }
                max_pixels = max_pixels.max(pixels);
            }
            kernel_jobs.push(CudaJ2kStoreRgb8MctBatchJob {
                plane0_ptr: target.plane0.device_ptr(),
                plane1_ptr: target.plane1.device_ptr(),
                plane2_ptr: target.plane2.device_ptr(),
                output_ptr: output.device_ptr(),
                job: target.job,
            });
            outputs.push(output);
        }
        if max_pixels == 0 {
            return Ok(CudaKernelBatchOutput {
                outputs,
                execution: CudaExecutionStats::default(),
            });
        }

        let jobs_buffer = self.upload(store_rgb8_mct_batch_jobs_as_bytes(&kernel_jobs))?;
        self.launch_j2k_store_rgb8_mct_batch(&jobs_buffer, max_pixels, kernel_jobs.len())?;
        Ok(CudaKernelBatchOutput {
            outputs,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Apply inverse RCT/ICT and store multiple tightly packed RGB8/RGBA8 images
    /// into one contiguous device allocation in one dispatch.
    pub fn j2k_store_rgb8_mct_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreRgb8MctTarget<'_>],
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        let mut ranges = Vec::with_capacity(targets.len());
        let mut total_bytes = 0usize;
        let mut max_pixels = 0usize;
        for target in targets {
            let store = target.job.store;
            let channels = if store.rgba == 0 { 3 } else { 4 };
            let output_bytes =
                checked_image_words(store.output_width, store.output_height, channels)?;
            let pixels = checked_image_words(store.copy_width, store.copy_height, 1)?;
            if output_bytes != 0 && pixels != 0 {
                validate_store_rgb8_plane(
                    target.plane0,
                    store.input_width0,
                    store.source_x0,
                    store.source_y0,
                    store.copy_width,
                    store.copy_height,
                )?;
                validate_store_rgb8_plane(
                    target.plane1,
                    store.input_width1,
                    store.source_x1,
                    store.source_y1,
                    store.copy_width,
                    store.copy_height,
                )?;
                validate_store_rgb8_plane(
                    target.plane2,
                    store.input_width2,
                    store.source_x2,
                    store.source_y2,
                    store.copy_width,
                    store.copy_height,
                )?;
                let dst_end = (store.output_y as usize)
                    .checked_add(store.copy_height as usize)
                    .and_then(|end_y| {
                        (store.output_x as usize)
                            .checked_add(store.copy_width as usize)
                            .map(|end_x| (end_x, end_y))
                    })
                    .ok_or(CudaError::LengthTooLarge { len: output_bytes })?;
                if dst_end.0 > store.output_width as usize
                    || dst_end.1 > store.output_height as usize
                {
                    return Err(CudaError::LengthTooLarge { len: output_bytes });
                }
                max_pixels = max_pixels.max(pixels);
            }
            let offset = total_bytes;
            total_bytes = total_bytes
                .checked_add(output_bytes)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            ranges.push(CudaDeviceBufferRange {
                offset,
                len: output_bytes,
            });
        }

        let output = self.allocate(total_bytes)?;
        if targets.is_empty() || max_pixels == 0 {
            return Ok(CudaKernelContiguousBatchOutput {
                output,
                ranges,
                execution: CudaExecutionStats::default(),
            });
        }

        let base_ptr = output.device_ptr();
        let kernel_jobs = targets
            .iter()
            .zip(ranges.iter())
            .map(|(target, range)| {
                let output_ptr = base_ptr
                    .checked_add(
                        u64::try_from(range.offset)
                            .map_err(|_| CudaError::LengthTooLarge { len: range.offset })?,
                    )
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
                Ok(CudaJ2kStoreRgb8MctBatchJob {
                    plane0_ptr: target.plane0.device_ptr(),
                    plane1_ptr: target.plane1.device_ptr(),
                    plane2_ptr: target.plane2.device_ptr(),
                    output_ptr,
                    job: target.job,
                })
            })
            .collect::<Result<Vec<_>, CudaError>>()?;
        let jobs_buffer = self.upload(store_rgb8_mct_batch_jobs_as_bytes(&kernel_jobs))?;
        self.launch_j2k_store_rgb8_mct_batch(&jobs_buffer, max_pixels, kernel_jobs.len())?;
        Ok(CudaKernelContiguousBatchOutput {
            output,
            ranges,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Apply inverse RCT/ICT and store tightly packed RGB16/RGBA16 in one dispatch.
    pub fn j2k_store_rgb16_mct_device(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: CudaJ2kStoreRgb16MctJob,
    ) -> Result<CudaKernelOutput, CudaError> {
        let store = job.store;
        let channels = if store.rgba == 0 { 3 } else { 4 };
        let output_samples =
            checked_image_words(store.output_width, store.output_height, channels)?;
        let output_bytes = output_samples
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or(CudaError::LengthTooLarge {
                len: output_samples,
            })?;
        let output = self.allocate(output_bytes)?;
        let pixels = checked_image_words(store.copy_width, store.copy_height, 1)?;
        if output_bytes == 0 || pixels == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        validate_store_rgb8_plane(
            plane0,
            store.input_width0,
            store.source_x0,
            store.source_y0,
            store.copy_width,
            store.copy_height,
        )?;
        validate_store_rgb8_plane(
            plane1,
            store.input_width1,
            store.source_x1,
            store.source_y1,
            store.copy_width,
            store.copy_height,
        )?;
        validate_store_rgb8_plane(
            plane2,
            store.input_width2,
            store.source_x2,
            store.source_y2,
            store.copy_width,
            store.copy_height,
        )?;
        let dst_end = (store.output_y as usize)
            .checked_add(store.copy_height as usize)
            .and_then(|end_y| {
                (store.output_x as usize)
                    .checked_add(store.copy_width as usize)
                    .map(|end_x| (end_x, end_y))
            })
            .ok_or(CudaError::LengthTooLarge { len: output_bytes })?;
        if dst_end.0 > store.output_width as usize || dst_end.1 > store.output_height as usize {
            return Err(CudaError::LengthTooLarge { len: output_bytes });
        }

        let job_buffer = self.upload(store_rgb16_mct_job_as_bytes(&job))?;
        self.launch_j2k_store_rgb16_mct(plane0, plane1, plane2, &output, &job_buffer, pixels)?;
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Deinterleave interleaved pixel bytes into f32 component planes.
    pub fn j2k_deinterleave_to_f32(
        &self,
        pixels: &[u8],
        num_pixels: usize,
        num_components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<CudaJ2kDeinterleavedComponents, CudaError> {
        let resident = self.j2k_deinterleave_to_f32_resident(
            pixels,
            num_pixels,
            num_components,
            bit_depth,
            signed,
        )?;
        let execution = resident.execution();
        let components = resident.download_components()?;
        Ok(CudaJ2kDeinterleavedComponents {
            components,
            execution,
        })
    }

    /// Deinterleave interleaved pixel bytes into resident f32 component planes.
    pub fn j2k_deinterleave_to_f32_resident(
        &self,
        pixels: &[u8],
        num_pixels: usize,
        num_components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<CudaJ2kResidentComponents, CudaError> {
        if num_components == 0 || num_components > 4 {
            return Err(CudaError::InvalidArgument {
                message: "component count must be between 1 and 4".to_string(),
            });
        }
        if bit_depth == 0 || bit_depth > 16 {
            return Err(CudaError::InvalidArgument {
                message: "bit depth must be between 1 and 16".to_string(),
            });
        }
        let bytes_per_sample = if bit_depth <= 8 { 1usize } else { 2usize };
        let expected_len = num_pixels
            .checked_mul(usize::from(num_components))
            .and_then(|len| len.checked_mul(bytes_per_sample))
            .ok_or(CudaError::LengthTooLarge { len: num_pixels })?;
        if pixels.len() < expected_len {
            return Err(CudaError::InvalidArgument {
                message: "pixel buffer is shorter than the requested image".to_string(),
            });
        }

        self.inner.set_current()?;
        let sample_count = num_pixels
            .checked_mul(usize::from(num_components))
            .ok_or(CudaError::LengthTooLarge { len: num_pixels })?;
        let output_bytes = sample_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: sample_count })?;
        let output = self.allocate(output_bytes)?;
        if num_pixels == 0 {
            return Ok(CudaJ2kResidentComponents {
                buffer: output,
                num_pixels,
                num_components,
                execution: CudaExecutionStats::default(),
            });
        }

        let pixels = self.upload(&pixels[..expected_len])?;
        self.launch_j2k_deinterleave_to_f32(
            &pixels,
            &output,
            num_pixels,
            num_components,
            bit_depth,
            signed,
        )?;

        Ok(CudaJ2kResidentComponents {
            buffer: output,
            num_pixels,
            num_components,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    /// Deinterleave strided device-resident pixel bytes into resident f32 component planes.
    pub fn j2k_deinterleave_strided_to_f32_resident(
        &self,
        image: CudaJ2kStridedInterleavedPixels<'_>,
    ) -> Result<CudaJ2kResidentComponents, CudaError> {
        let CudaJ2kStridedInterleavedPixels {
            buffer: pixels,
            byte_offset,
            width,
            height,
            pitch_bytes,
            num_components,
            bit_depth,
            signed,
        } = image;
        if width == 0 || height == 0 {
            return Err(CudaError::InvalidArgument {
                message: "image dimensions must be nonzero".to_string(),
            });
        }
        if num_components == 0 || num_components > 4 {
            return Err(CudaError::InvalidArgument {
                message: "component count must be between 1 and 4".to_string(),
            });
        }
        if bit_depth == 0 || bit_depth > 16 {
            return Err(CudaError::InvalidArgument {
                message: "bit depth must be between 1 and 16".to_string(),
            });
        }
        let bytes_per_sample = if bit_depth <= 8 { 1usize } else { 2usize };
        let bytes_per_pixel = usize::from(num_components)
            .checked_mul(bytes_per_sample)
            .ok_or(CudaError::LengthTooLarge {
                len: usize::from(num_components),
            })?;
        let row_bytes =
            (width as usize)
                .checked_mul(bytes_per_pixel)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: usize::from(num_components),
                })?;
        if pitch_bytes < row_bytes {
            return Err(CudaError::InvalidArgument {
                message: "pitch is shorter than one row".to_string(),
            });
        }
        let required_end = byte_offset
            .checked_add(
                pitch_bytes
                    .checked_mul(height.saturating_sub(1) as usize)
                    .and_then(|prefix| prefix.checked_add(row_bytes))
                    .ok_or(CudaError::LengthTooLarge { len: pitch_bytes })?,
            )
            .ok_or(CudaError::LengthTooLarge { len: byte_offset })?;
        if required_end > pixels.byte_len() {
            return Err(CudaError::OutputTooSmall {
                required: required_end,
                have: pixels.byte_len(),
            });
        }

        self.inner.set_current()?;
        let num_pixels =
            (width as usize)
                .checked_mul(height as usize)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: usize::from(num_components),
                })?;
        let sample_count = num_pixels
            .checked_mul(usize::from(num_components))
            .ok_or(CudaError::LengthTooLarge { len: num_pixels })?;
        let output_bytes = sample_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: sample_count })?;
        let output = self.allocate(output_bytes)?;
        self.launch_j2k_deinterleave_strided_to_f32(
            pixels,
            &output,
            width,
            height,
            byte_offset,
            pitch_bytes,
            num_components,
            bit_depth,
            signed,
        )?;

        Ok(CudaJ2kResidentComponents {
            buffer: output,
            num_pixels,
            num_components,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    /// Run the reversible color transform in place on resident component planes.
    ///
    /// The transform is applied to the first three planes (R, G, B → Y, Cb, Cr).
    /// Any additional plane (e.g. a 4th alpha/auxiliary component) is left
    /// untouched, matching the native reference which applies RCT to the first
    /// three of `&mut [Vec<f32>]` and passes the remainder through unchanged.
    pub fn j2k_forward_rct_resident(
        &self,
        components: &mut CudaJ2kResidentComponents,
    ) -> Result<CudaExecutionStats, CudaError> {
        if components.num_components < 3 {
            return Err(CudaError::InvalidArgument {
                message: "forward RCT requires at least three resident component planes"
                    .to_string(),
            });
        }
        if components.num_pixels == 0 {
            return Ok(CudaExecutionStats::default());
        }

        self.inner.set_current()?;
        let plane0 = components.component_plane_device_ptr(0)?;
        let plane1 = components.component_plane_device_ptr(1)?;
        let plane2 = components.component_plane_device_ptr(2)?;
        self.launch_j2k_forward_rct_ptrs(plane0, plane1, plane2, components.num_pixels)?;

        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        })
    }

    /// Run the irreversible color transform in place on resident component planes.
    ///
    /// The transform is applied to the first three planes (R, G, B → Y, Cb, Cr).
    /// Any additional plane is left untouched, matching the native reference
    /// which applies ICT to the first three of `&mut [Vec<f32>]` and passes the
    /// remainder through unchanged.
    pub fn j2k_forward_ict_resident(
        &self,
        components: &mut CudaJ2kResidentComponents,
    ) -> Result<CudaExecutionStats, CudaError> {
        if components.num_components < 3 {
            return Err(CudaError::InvalidArgument {
                message: "forward ICT requires at least three resident component planes"
                    .to_string(),
            });
        }
        if components.num_pixels == 0 {
            return Ok(CudaExecutionStats::default());
        }

        self.inner.set_current()?;
        let plane0 = components.component_plane_device_ptr(0)?;
        let plane1 = components.component_plane_device_ptr(1)?;
        let plane2 = components.component_plane_device_ptr(2)?;
        self.launch_j2k_forward_ict_ptrs(plane0, plane1, plane2, components.num_pixels)?;

        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        })
    }

    /// Run the reversible color transform stage on three component planes.
    pub fn j2k_forward_rct(
        &self,
        plane0: &mut [f32],
        plane1: &mut [f32],
        plane2: &mut [f32],
    ) -> Result<CudaExecutionStats, CudaError> {
        if plane0.len() != plane1.len() || plane0.len() != plane2.len() {
            return Err(CudaError::ImageTooLarge {
                width: u32::try_from(plane0.len()).unwrap_or(u32::MAX),
                height: 1,
                channels: 3,
            });
        }
        if plane0.is_empty() {
            return Ok(CudaExecutionStats::default());
        }

        self.inner.set_current()?;
        let buffer0 = self.upload(f32_slice_as_bytes(plane0))?;
        let buffer1 = self.upload(f32_slice_as_bytes(plane1))?;
        let buffer2 = self.upload(f32_slice_as_bytes(plane2))?;
        self.launch_j2k_forward_rct_buffers(&buffer0, &buffer1, &buffer2, plane0.len())?;
        buffer0.copy_to_host(f32_slice_as_bytes_mut(plane0))?;
        buffer1.copy_to_host(f32_slice_as_bytes_mut(plane1))?;
        buffer2.copy_to_host(f32_slice_as_bytes_mut(plane2))?;

        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        })
    }

    /// Run the irreversible color transform stage on three component planes.
    pub fn j2k_forward_ict(
        &self,
        plane0: &mut [f32],
        plane1: &mut [f32],
        plane2: &mut [f32],
    ) -> Result<CudaExecutionStats, CudaError> {
        if plane0.len() != plane1.len() || plane0.len() != plane2.len() {
            return Err(CudaError::ImageTooLarge {
                width: u32::try_from(plane0.len()).unwrap_or(u32::MAX),
                height: 1,
                channels: 3,
            });
        }
        if plane0.is_empty() {
            return Ok(CudaExecutionStats::default());
        }

        self.inner.set_current()?;
        let buffer0 = self.upload(f32_slice_as_bytes(plane0))?;
        let buffer1 = self.upload(f32_slice_as_bytes(plane1))?;
        let buffer2 = self.upload(f32_slice_as_bytes(plane2))?;
        self.launch_j2k_forward_ict_buffers(&buffer0, &buffer1, &buffer2, plane0.len())?;
        buffer0.copy_to_host(f32_slice_as_bytes_mut(plane0))?;
        buffer1.copy_to_host(f32_slice_as_bytes_mut(plane1))?;
        buffer2.copy_to_host(f32_slice_as_bytes_mut(plane2))?;

        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        })
    }

    /// Run the reversible 5/3 forward DWT stage on one component plane.
    pub fn j2k_forward_dwt53(
        &self,
        samples: &[f32],
        width: u32,
        height: u32,
        num_levels: u8,
    ) -> Result<CudaDwt53Output, CudaError> {
        let expected_len =
            (width as usize)
                .checked_mul(height as usize)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: 1,
                })?;
        if expected_len != samples.len() {
            return Err(CudaError::ImageTooLarge {
                width,
                height,
                channels: 1,
            });
        }
        if samples.is_empty() || num_levels == 0 {
            return Ok(CudaDwt53Output {
                transformed: samples.to_vec(),
                levels: Vec::new(),
                ll_width: width,
                ll_height: height,
                execution: CudaExecutionStats::default(),
            });
        }

        self.inner.set_current()?;
        let buffer_a = self.upload(f32_slice_as_bytes(samples))?;
        let resident = self.j2k_forward_dwt53_resident_buffer(
            buffer_a,
            samples.len(),
            width,
            height,
            num_levels,
            0,
        )?;
        let transformed = resident.download_transformed()?;
        Ok(CudaDwt53Output {
            transformed,
            levels: resident.levels().to_vec(),
            ll_width: resident.ll_dimensions().0,
            ll_height: resident.ll_dimensions().1,
            execution: resident.execution(),
        })
    }

    /// Run the reversible 5/3 forward DWT on one resident component plane.
    pub fn j2k_forward_dwt53_resident_component(
        &self,
        components: &CudaJ2kResidentComponents,
        component: u8,
        width: u32,
        height: u32,
        num_levels: u8,
    ) -> Result<CudaResidentDwt53Output, CudaError> {
        let expected_len =
            (width as usize)
                .checked_mul(height as usize)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: 1,
                })?;
        if expected_len != components.num_pixels {
            return Err(CudaError::ImageTooLarge {
                width,
                height,
                channels: 1,
            });
        }

        self.inner.set_current()?;
        let plane_ptr = components.component_plane_device_ptr(component)?;
        let byte_len = expected_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: expected_len })?;
        let buffer_a = self.copy_device_ptr_to_device_with_kernel(plane_ptr, byte_len)?;
        let copy_dispatches = usize::from(byte_len != 0);
        self.j2k_forward_dwt53_resident_buffer(
            buffer_a,
            expected_len,
            width,
            height,
            num_levels,
            copy_dispatches,
        )
    }

    fn j2k_forward_dwt53_resident_buffer(
        &self,
        buffer_a: CudaDeviceBuffer,
        sample_count: usize,
        width: u32,
        height: u32,
        num_levels: u8,
        initial_copy_dispatches: usize,
    ) -> Result<CudaResidentDwt53Output, CudaError> {
        if sample_count == 0 || num_levels == 0 {
            return Ok(CudaResidentDwt53Output {
                buffer: buffer_a,
                sample_count,
                levels: Vec::new(),
                ll_width: width,
                ll_height: height,
                execution: CudaExecutionStats {
                    kernel_dispatches: initial_copy_dispatches,
                    copy_kernel_dispatches: initial_copy_dispatches,
                    decode_kernel_dispatches: 0,
                    hardware_decode: false,
                },
            });
        }

        let buffer_b = self.allocate(
            sample_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: sample_count })?,
        )?;
        let mut current_width = width;
        let mut current_height = height;
        let mut levels = Vec::new();
        let mut dispatches = 0usize;
        let mut active_is_a = true;

        for _ in 0..num_levels {
            if current_width < 2 && current_height < 2 {
                break;
            }
            let (level_dispatches, level_shape) = self.launch_j2k_forward_dwt53_level(
                &buffer_a,
                &buffer_b,
                &mut active_is_a,
                CudaDwt53LevelPass {
                    full_width: width,
                    current_width,
                    current_height,
                },
            )?;
            dispatches = dispatches.saturating_add(level_dispatches);
            levels.push(level_shape);
            current_width = level_shape.low_width;
            current_height = level_shape.low_height;
        }

        let buffer = if active_is_a { buffer_a } else { buffer_b };
        Ok(CudaResidentDwt53Output {
            buffer,
            sample_count,
            levels,
            ll_width: current_width,
            ll_height: current_height,
            execution: CudaExecutionStats {
                kernel_dispatches: initial_copy_dispatches.saturating_add(dispatches),
                copy_kernel_dispatches: initial_copy_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    /// Run the irreversible 9/7 forward DWT stage on one component plane.
    pub fn j2k_forward_dwt97(
        &self,
        samples: &[f32],
        width: u32,
        height: u32,
        num_levels: u8,
    ) -> Result<CudaDwt97Output, CudaError> {
        let expected_len =
            (width as usize)
                .checked_mul(height as usize)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: 1,
                })?;
        if expected_len != samples.len() {
            return Err(CudaError::ImageTooLarge {
                width,
                height,
                channels: 1,
            });
        }
        if samples.is_empty() || num_levels == 0 {
            return Ok(CudaDwt97Output {
                transformed: samples.to_vec(),
                levels: Vec::new(),
                ll_width: width,
                ll_height: height,
                execution: CudaExecutionStats::default(),
            });
        }

        self.inner.set_current()?;
        let buffer_a = self.upload(f32_slice_as_bytes(samples))?;
        let resident = self.j2k_forward_dwt97_resident_buffer(
            buffer_a,
            samples.len(),
            width,
            height,
            num_levels,
            0,
        )?;
        let transformed = resident.download_transformed()?;
        Ok(CudaDwt97Output {
            transformed,
            levels: resident.levels().to_vec(),
            ll_width: resident.ll_dimensions().0,
            ll_height: resident.ll_dimensions().1,
            execution: resident.execution(),
        })
    }

    /// Run the irreversible 9/7 forward DWT on one resident component plane.
    pub fn j2k_forward_dwt97_resident_component(
        &self,
        components: &CudaJ2kResidentComponents,
        component: u8,
        width: u32,
        height: u32,
        num_levels: u8,
    ) -> Result<CudaResidentDwt97Output, CudaError> {
        let expected_len =
            (width as usize)
                .checked_mul(height as usize)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: 1,
                })?;
        if expected_len != components.num_pixels {
            return Err(CudaError::ImageTooLarge {
                width,
                height,
                channels: 1,
            });
        }

        self.inner.set_current()?;
        let plane_ptr = components.component_plane_device_ptr(component)?;
        let byte_len = expected_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: expected_len })?;
        let buffer_a = self.copy_device_ptr_to_device_with_kernel(plane_ptr, byte_len)?;
        let copy_dispatches = usize::from(byte_len != 0);
        self.j2k_forward_dwt97_resident_buffer(
            buffer_a,
            expected_len,
            width,
            height,
            num_levels,
            copy_dispatches,
        )
    }

    fn j2k_forward_dwt97_resident_buffer(
        &self,
        buffer_a: CudaDeviceBuffer,
        sample_count: usize,
        width: u32,
        height: u32,
        num_levels: u8,
        initial_copy_dispatches: usize,
    ) -> Result<CudaResidentDwt97Output, CudaError> {
        if sample_count == 0 || num_levels == 0 {
            return Ok(CudaResidentDwt97Output {
                buffer: buffer_a,
                sample_count,
                levels: Vec::new(),
                ll_width: width,
                ll_height: height,
                execution: CudaExecutionStats {
                    kernel_dispatches: initial_copy_dispatches,
                    copy_kernel_dispatches: initial_copy_dispatches,
                    decode_kernel_dispatches: 0,
                    hardware_decode: false,
                },
            });
        }

        let buffer_b = self.allocate(
            sample_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: sample_count })?,
        )?;
        let mut current_width = width;
        let mut current_height = height;
        let mut levels = Vec::new();
        let mut dispatches = 0usize;
        let mut active_is_a = true;

        for _ in 0..num_levels {
            if current_width < 2 && current_height < 2 {
                break;
            }
            let (level_dispatches, level_shape) = self.launch_j2k_forward_dwt97_level(
                &buffer_a,
                &buffer_b,
                &mut active_is_a,
                CudaDwt53LevelPass {
                    full_width: width,
                    current_width,
                    current_height,
                },
            )?;
            dispatches = dispatches.saturating_add(level_dispatches);
            levels.push(level_shape);
            current_width = level_shape.low_width;
            current_height = level_shape.low_height;
        }

        let buffer = if active_is_a { buffer_a } else { buffer_b };
        Ok(CudaResidentDwt97Output {
            buffer,
            sample_count,
            levels,
            ll_width: current_width,
            ll_height: current_height,
            execution: CudaExecutionStats {
                kernel_dispatches: initial_copy_dispatches.saturating_add(dispatches),
                copy_kernel_dispatches: initial_copy_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    /// Quantize one JPEG 2000 sub-band on the device.
    pub fn j2k_quantize_subband(
        &self,
        samples: &[f32],
        job: CudaJ2kQuantizeJob,
    ) -> Result<CudaJ2kQuantizedSubband, CudaError> {
        let sample_buffer = self.upload(f32_slice_as_bytes(samples))?;
        let resident = self.j2k_quantize_subband_resident(&sample_buffer, samples.len(), job)?;
        let coefficients = resident.download_coefficients()?;
        Ok(CudaJ2kQuantizedSubband {
            coefficients,
            execution: resident.execution(),
        })
    }

    /// Quantize a resident contiguous JPEG 2000 sub-band into resident `i32` coefficients.
    pub fn j2k_quantize_subband_resident(
        &self,
        samples: &CudaDeviceBuffer,
        sample_count: usize,
        job: CudaJ2kQuantizeJob,
    ) -> Result<CudaJ2kResidentQuantizedSubband, CudaError> {
        if sample_count == 0 {
            return Ok(CudaJ2kResidentQuantizedSubband {
                coefficients: self.allocate(0)?,
                coefficient_count: 0,
                execution: CudaExecutionStats::default(),
            });
        }

        let available_samples = samples.typed_view::<f32>()?.len();
        if available_samples < sample_count {
            return Err(CudaError::OutputTooSmall {
                required: sample_count
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or(CudaError::LengthTooLarge { len: sample_count })?,
                have: samples.byte_len(),
            });
        }

        self.inner.set_current()?;
        let coefficient_buffer = self.allocate(
            sample_count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: sample_count })?,
        )?;
        self.launch_j2k_quantize_subband(samples, &coefficient_buffer, sample_count, job)?;

        Ok(CudaJ2kResidentQuantizedSubband {
            coefficients: coefficient_buffer,
            coefficient_count: sample_count,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    /// Quantize a resident strided DWT sub-band rectangle into resident `i32` coefficients.
    pub fn j2k_quantize_subband_region_resident(
        &self,
        samples: &CudaDeviceBuffer,
        job: CudaJ2kQuantizeSubbandRegionJob,
    ) -> Result<CudaJ2kResidentQuantizedSubband, CudaError> {
        let coefficient_count = checked_image_words(job.width, job.height, 1)?;
        if coefficient_count == 0 {
            return Ok(CudaJ2kResidentQuantizedSubband {
                coefficients: self.allocate(0)?,
                coefficient_count: 0,
                execution: CudaExecutionStats::default(),
            });
        }

        let available_samples = samples.typed_view::<f32>()?.len();
        validate_quantize_region(job, available_samples)?;
        self.inner.set_current()?;
        let coefficient_buffer = self.allocate(
            coefficient_count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge {
                    len: coefficient_count,
                })?,
        )?;
        self.launch_j2k_quantize_subband_region(samples, &coefficient_buffer, job)?;

        Ok(CudaJ2kResidentQuantizedSubband {
            coefficients: coefficient_buffer,
            coefficient_count,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    fn launch_j2k_forward_dwt53_level(
        &self,
        buffer_a: &CudaDeviceBuffer,
        buffer_b: &CudaDeviceBuffer,
        active_is_a: &mut bool,
        pass: CudaDwt53LevelPass,
    ) -> Result<(usize, CudaDwt53LevelShape), CudaError> {
        let low_width = pass.current_width.div_ceil(2);
        let low_height = pass.current_height.div_ceil(2);
        let mut dispatches = 0usize;

        if pass.current_height >= 2 {
            let (input, output) = active_dwt53_buffers(buffer_a, buffer_b, *active_is_a);
            self.launch_j2k_forward_dwt53_pass(
                CudaKernel::J2kForwardDwt53Vertical,
                input,
                output,
                CudaDwt53Pass {
                    full_width: pass.full_width,
                    current_width: pass.current_width,
                    current_height: pass.current_height,
                    low_extent: low_height,
                },
            )?;
            *active_is_a = !*active_is_a;
            dispatches = dispatches.saturating_add(1);
        }

        if pass.current_width >= 2 {
            let (input, output) = active_dwt53_buffers(buffer_a, buffer_b, *active_is_a);
            self.launch_j2k_forward_dwt53_pass(
                CudaKernel::J2kForwardDwt53Horizontal,
                input,
                output,
                CudaDwt53Pass {
                    full_width: pass.full_width,
                    current_width: pass.current_width,
                    current_height: pass.current_height,
                    low_extent: low_width,
                },
            )?;
            *active_is_a = !*active_is_a;
            dispatches = dispatches.saturating_add(1);
        }

        Ok((
            dispatches,
            CudaDwt53LevelShape {
                width: pass.current_width,
                height: pass.current_height,
                low_width,
                low_height,
                high_width: pass.current_width / 2,
                high_height: pass.current_height / 2,
            },
        ))
    }

    fn launch_j2k_forward_dwt97_level(
        &self,
        buffer_a: &CudaDeviceBuffer,
        buffer_b: &CudaDeviceBuffer,
        active_is_a: &mut bool,
        pass: CudaDwt53LevelPass,
    ) -> Result<(usize, CudaDwt53LevelShape), CudaError> {
        let low_width = pass.current_width.div_ceil(2);
        let low_height = pass.current_height.div_ceil(2);
        let mut dispatches = 0usize;

        if pass.current_height >= 2 {
            let (input, output) = active_dwt53_buffers(buffer_a, buffer_b, *active_is_a);
            self.launch_j2k_forward_dwt53_pass(
                CudaKernel::J2kForwardDwt97Vertical,
                input,
                output,
                CudaDwt53Pass {
                    full_width: pass.full_width,
                    current_width: pass.current_width,
                    current_height: pass.current_height,
                    low_extent: low_height,
                },
            )?;
            *active_is_a = !*active_is_a;
            dispatches = dispatches.saturating_add(1);
        }

        if pass.current_width >= 2 {
            let (input, output) = active_dwt53_buffers(buffer_a, buffer_b, *active_is_a);
            self.launch_j2k_forward_dwt53_pass(
                CudaKernel::J2kForwardDwt97Horizontal,
                input,
                output,
                CudaDwt53Pass {
                    full_width: pass.full_width,
                    current_width: pass.current_width,
                    current_height: pass.current_height,
                    low_extent: low_width,
                },
            )?;
            *active_is_a = !*active_is_a;
            dispatches = dispatches.saturating_add(1);
        }

        Ok((
            dispatches,
            CudaDwt53LevelShape {
                width: pass.current_width,
                height: pass.current_height,
                low_width,
                low_height,
                high_width: pass.current_width / 2,
                high_height: pass.current_height / 2,
            },
        ))
    }

    fn launch_j2k_forward_rct_buffers(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        len: usize,
    ) -> Result<(), CudaError> {
        self.launch_j2k_forward_rct_ptrs(
            plane0.device_ptr(),
            plane1.device_ptr(),
            plane2.device_ptr(),
            len,
        )
    }

    fn launch_j2k_forward_rct_ptrs(
        &self,
        plane0: CuDevicePtr,
        plane1: CuDevicePtr,
        plane2: CuDevicePtr,
        len: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kForwardRct)?;
        let mut plane0_ptr = plane0;
        let mut plane1_ptr = plane1;
        let mut plane2_ptr = plane2;
        let mut len_u64 = u64::try_from(len).map_err(|_| CudaError::LengthTooLarge { len })?;
        let mut params = [
            (&raw mut plane0_ptr).cast::<c_void>(),
            (&raw mut plane1_ptr).cast::<c_void>(),
            (&raw mut plane2_ptr).cast::<c_void>(),
            (&raw mut len_u64).cast::<c_void>(),
        ];
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_deinterleave_to_f32(
        &self,
        pixels: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        num_pixels: usize,
        num_components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kDeinterleaveToF32)?;
        let mut pixels_ptr = pixels.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut num_pixels_u64 =
            u64::try_from(num_pixels).map_err(|_| CudaError::LengthTooLarge { len: num_pixels })?;
        let mut num_components_u32 = u32::from(num_components);
        let mut bit_depth_u32 = u32::from(bit_depth);
        let mut signed_u32 = u32::from(signed);
        let mut params = [
            (&raw mut pixels_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut num_pixels_u64).cast::<c_void>(),
            (&raw mut num_components_u32).cast::<c_void>(),
            (&raw mut bit_depth_u32).cast::<c_void>(),
            (&raw mut signed_u32).cast::<c_void>(),
        ];
        let geometry = j2k_forward_rct_launch_geometry(num_pixels)
            .ok_or(CudaError::LengthTooLarge { len: num_pixels })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_j2k_deinterleave_strided_to_f32(
        &self,
        pixels: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        width: u32,
        height: u32,
        byte_offset: usize,
        pitch_bytes: usize,
        num_components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kDeinterleaveStridedToF32)?;
        let mut pixels_ptr = pixels.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut width_u64 = u64::from(width);
        let mut height_u64 = u64::from(height);
        let mut byte_offset_u64 = u64::try_from(byte_offset)
            .map_err(|_| CudaError::LengthTooLarge { len: byte_offset })?;
        let mut pitch_bytes_u64 = u64::try_from(pitch_bytes)
            .map_err(|_| CudaError::LengthTooLarge { len: pitch_bytes })?;
        let mut num_components_u32 = u32::from(num_components);
        let mut bit_depth_u32 = u32::from(bit_depth);
        let mut signed_u32 = u32::from(signed);
        let mut params = [
            (&raw mut pixels_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut width_u64).cast::<c_void>(),
            (&raw mut height_u64).cast::<c_void>(),
            (&raw mut byte_offset_u64).cast::<c_void>(),
            (&raw mut pitch_bytes_u64).cast::<c_void>(),
            (&raw mut num_components_u32).cast::<c_void>(),
            (&raw mut bit_depth_u32).cast::<c_void>(),
            (&raw mut signed_u32).cast::<c_void>(),
        ];
        let num_pixels =
            (width as usize)
                .checked_mul(height as usize)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: usize::from(num_components),
                })?;
        let geometry = j2k_forward_rct_launch_geometry(num_pixels)
            .ok_or(CudaError::LengthTooLarge { len: num_pixels })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_forward_ict_buffers(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        len: usize,
    ) -> Result<(), CudaError> {
        self.launch_j2k_forward_ict_ptrs(
            plane0.device_ptr(),
            plane1.device_ptr(),
            plane2.device_ptr(),
            len,
        )
    }

    fn launch_j2k_forward_ict_ptrs(
        &self,
        plane0: CuDevicePtr,
        plane1: CuDevicePtr,
        plane2: CuDevicePtr,
        len: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kForwardIct)?;
        let mut plane0_ptr = plane0;
        let mut plane1_ptr = plane1;
        let mut plane2_ptr = plane2;
        let mut len_u64 = u64::try_from(len).map_err(|_| CudaError::LengthTooLarge { len })?;
        let mut params = [
            (&raw mut plane0_ptr).cast::<c_void>(),
            (&raw mut plane1_ptr).cast::<c_void>(),
            (&raw mut plane2_ptr).cast::<c_void>(),
            (&raw mut len_u64).cast::<c_void>(),
        ];
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_forward_dwt53_pass(
        &self,
        kernel: CudaKernel,
        input: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        pass: CudaDwt53Pass,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(kernel)?;
        let mut input_ptr = input.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut full_width = pass.full_width;
        let mut current_width = pass.current_width;
        let mut current_height = pass.current_height;
        let mut low_extent = pass.low_extent;
        let mut params = [
            (&raw mut input_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut full_width).cast::<c_void>(),
            (&raw mut current_width).cast::<c_void>(),
            (&raw mut current_height).cast::<c_void>(),
            (&raw mut low_extent).cast::<c_void>(),
        ];
        let geometry = j2k_dwt53_launch_geometry(current_width, current_height).ok_or(
            CudaError::ImageTooLarge {
                width: pass.current_width,
                height: pass.current_height,
                channels: 1,
            },
        )?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_quantize_subband(
        &self,
        samples: &CudaDeviceBuffer,
        coefficients: &CudaDeviceBuffer,
        len: usize,
        job: CudaJ2kQuantizeJob,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kQuantizeSubband)?;
        let mut samples_ptr = samples.device_ptr();
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut len_u64 = u64::try_from(len).map_err(|_| CudaError::LengthTooLarge { len })?;
        let mut step_exponent = u32::from(job.step_exponent);
        let mut step_mantissa = u32::from(job.step_mantissa);
        let mut range_bits = u32::from(job.range_bits);
        let mut reversible = u32::from(job.reversible);
        let mut params = [
            (&raw mut samples_ptr).cast::<c_void>(),
            (&raw mut coefficients_ptr).cast::<c_void>(),
            (&raw mut len_u64).cast::<c_void>(),
            (&raw mut step_exponent).cast::<c_void>(),
            (&raw mut step_mantissa).cast::<c_void>(),
            (&raw mut range_bits).cast::<c_void>(),
            (&raw mut reversible).cast::<c_void>(),
        ];
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_quantize_subband_region(
        &self,
        samples: &CudaDeviceBuffer,
        coefficients: &CudaDeviceBuffer,
        job: CudaJ2kQuantizeSubbandRegionJob,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kQuantizeSubbandStrided)?;
        let mut samples_ptr = samples.device_ptr();
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut x0 = job.x0;
        let mut y0 = job.y0;
        let mut width = job.width;
        let mut height = job.height;
        let mut stride = job.stride;
        let mut step_exponent = u32::from(job.quantization.step_exponent);
        let mut step_mantissa = u32::from(job.quantization.step_mantissa);
        let mut range_bits = u32::from(job.quantization.range_bits);
        let mut reversible = u32::from(job.quantization.reversible);
        let mut params = [
            (&raw mut samples_ptr).cast::<c_void>(),
            (&raw mut coefficients_ptr).cast::<c_void>(),
            (&raw mut x0).cast::<c_void>(),
            (&raw mut y0).cast::<c_void>(),
            (&raw mut width).cast::<c_void>(),
            (&raw mut height).cast::<c_void>(),
            (&raw mut stride).cast::<c_void>(),
            (&raw mut step_exponent).cast::<c_void>(),
            (&raw mut step_mantissa).cast::<c_void>(),
            (&raw mut range_bits).cast::<c_void>(),
            (&raw mut reversible).cast::<c_void>(),
        ];
        let geometry =
            j2k_dwt53_launch_geometry(job.width, job.height).ok_or(CudaError::ImageTooLarge {
                width: job.width,
                height: job.height,
                channels: 1,
            })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn launch_htj2k_decode_codeblocks(
        &self,
        payload: &CudaDeviceBuffer,
        coefficients: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table0: &CudaDeviceBuffer,
        uvlc_table1: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        self.launch_htj2k_decode_codeblocks_with_sync(
            payload,
            coefficients,
            jobs,
            vlc_table0,
            vlc_table1,
            uvlc_table0,
            uvlc_table1,
            statuses,
            job_count,
            true,
        )
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn launch_htj2k_decode_codeblocks_async(
        &self,
        payload: &CudaDeviceBuffer,
        coefficients: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table0: &CudaDeviceBuffer,
        uvlc_table1: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        self.launch_htj2k_decode_codeblocks_with_sync(
            payload,
            coefficients,
            jobs,
            vlc_table0,
            vlc_table1,
            uvlc_table0,
            uvlc_table1,
            statuses,
            job_count,
            false,
        )
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn launch_htj2k_decode_codeblocks_with_sync(
        &self,
        payload: &CudaDeviceBuffer,
        coefficients: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table0: &CudaDeviceBuffer,
        uvlc_table1: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kDecodeCodeblocks)?;
        let mut payload_ptr = payload.device_ptr();
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table0_ptr = uvlc_table0.device_ptr();
        let mut uvlc_table1_ptr = uvlc_table1.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count = c_uint::try_from(job_count)
            .map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = [
            (&raw mut payload_ptr).cast::<c_void>(),
            (&raw mut coefficients_ptr).cast::<c_void>(),
            (&raw mut jobs_ptr).cast::<c_void>(),
            (&raw mut vlc_table0_ptr).cast::<c_void>(),
            (&raw mut vlc_table1_ptr).cast::<c_void>(),
            (&raw mut uvlc_table0_ptr).cast::<c_void>(),
            (&raw mut uvlc_table1_ptr).cast::<c_void>(),
            (&raw mut statuses_ptr).cast::<c_void>(),
            (&raw mut job_count).cast::<c_void>(),
        ];
        let geometry = htj2k_codeblock_launch_geometry(job_count as usize).ok_or(
            CudaError::LengthTooLarge {
                len: job_count as usize,
            },
        )?;

        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn launch_htj2k_decode_codeblocks_multi(
        &self,
        kernel: CudaKernel,
        payload: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table0: &CudaDeviceBuffer,
        uvlc_table1: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        self.launch_htj2k_decode_codeblocks_multi_with_sync(
            kernel,
            payload,
            jobs,
            vlc_table0,
            vlc_table1,
            uvlc_table0,
            uvlc_table1,
            statuses,
            job_count,
            true,
        )
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn launch_htj2k_decode_codeblocks_multi_async(
        &self,
        kernel: CudaKernel,
        payload: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table0: &CudaDeviceBuffer,
        uvlc_table1: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        self.launch_htj2k_decode_codeblocks_multi_with_sync(
            kernel,
            payload,
            jobs,
            vlc_table0,
            vlc_table1,
            uvlc_table0,
            uvlc_table1,
            statuses,
            job_count,
            false,
        )
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn launch_htj2k_decode_codeblocks_multi_with_sync(
        &self,
        kernel: CudaKernel,
        payload: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table0: &CudaDeviceBuffer,
        uvlc_table1: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(kernel)?;
        let mut payload_ptr = payload.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table0_ptr = uvlc_table0.device_ptr();
        let mut uvlc_table1_ptr = uvlc_table1.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count = c_uint::try_from(job_count)
            .map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = [
            (&raw mut payload_ptr).cast::<c_void>(),
            (&raw mut jobs_ptr).cast::<c_void>(),
            (&raw mut vlc_table0_ptr).cast::<c_void>(),
            (&raw mut vlc_table1_ptr).cast::<c_void>(),
            (&raw mut uvlc_table0_ptr).cast::<c_void>(),
            (&raw mut uvlc_table1_ptr).cast::<c_void>(),
            (&raw mut statuses_ptr).cast::<c_void>(),
            (&raw mut job_count).cast::<c_void>(),
        ];
        let geometry = htj2k_codeblock_launch_geometry(job_count as usize).ok_or(
            CudaError::LengthTooLarge {
                len: job_count as usize,
            },
        )?;

        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    fn launch_j2k_dequantize_htj2k_codeblocks(
        &self,
        coefficients: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        self.launch_j2k_dequantize_htj2k_codeblocks_with_sync(coefficients, jobs, job_count, true)
    }

    fn launch_j2k_dequantize_htj2k_codeblocks_async(
        &self,
        coefficients: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        self.launch_j2k_dequantize_htj2k_codeblocks_with_sync(coefficients, jobs, job_count, false)
    }

    fn launch_j2k_dequantize_htj2k_codeblocks_with_sync(
        &self,
        coefficients: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kDequantizeHtj2kCodeblocks)?;
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = [
            (&raw mut coefficients_ptr).cast::<c_void>(),
            (&raw mut jobs_ptr).cast::<c_void>(),
        ];
        let geometry = htj2k_codeblock_sample_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;

        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    fn launch_j2k_dequantize_htj2k_codeblocks_multi_with_sync(
        &self,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kDequantizeHtj2kCodeblocksMulti)?;
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = [(&raw mut jobs_ptr).cast::<c_void>()];
        let geometry = htj2k_codeblock_sample_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;

        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    fn launch_j2k_dequantize_htj2k_cleanup_jobs_multi_with_sync(
        &self,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kDequantizeHtj2kCleanupJobsMulti)?;
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = [(&raw mut jobs_ptr).cast::<c_void>()];
        let geometry = htj2k_codeblock_sample_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;

        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_htj2k_encode_codeblock(
        &self,
        coefficients: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        params_buffer: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table: &CudaDeviceBuffer,
        status: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kEncodeCodeblock)?;
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut params_ptr = params_buffer.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table_ptr = uvlc_table.device_ptr();
        let mut status_ptr = status.device_ptr();
        let mut params = [
            (&raw mut coefficients_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut params_ptr).cast::<c_void>(),
            (&raw mut vlc_table0_ptr).cast::<c_void>(),
            (&raw mut vlc_table1_ptr).cast::<c_void>(),
            (&raw mut uvlc_table_ptr).cast::<c_void>(),
            (&raw mut status_ptr).cast::<c_void>(),
        ];
        let geometry = htj2k_encode_codeblock_launch_geometry(1)
            .ok_or(CudaError::LengthTooLarge { len: 1 })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_htj2k_encode_codeblocks(
        &self,
        coefficients: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kEncodeCodeblocks)?;
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table_ptr = uvlc_table.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count_u64 =
            u64::try_from(job_count).map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = [
            (&raw mut coefficients_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut jobs_ptr).cast::<c_void>(),
            (&raw mut vlc_table0_ptr).cast::<c_void>(),
            (&raw mut vlc_table1_ptr).cast::<c_void>(),
            (&raw mut uvlc_table_ptr).cast::<c_void>(),
            (&raw mut statuses_ptr).cast::<c_void>(),
            (&raw mut job_count_u64).cast::<c_void>(),
        ];
        let geometry = htj2k_encode_codeblock_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_htj2k_encode_codeblocks_multi_input(
        &self,
        output: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kEncodeCodeblocksMultiInput)?;
        let mut output_ptr = output.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table_ptr = uvlc_table.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count_u64 =
            u64::try_from(job_count).map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = [
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut jobs_ptr).cast::<c_void>(),
            (&raw mut vlc_table0_ptr).cast::<c_void>(),
            (&raw mut vlc_table1_ptr).cast::<c_void>(),
            (&raw mut uvlc_table_ptr).cast::<c_void>(),
            (&raw mut statuses_ptr).cast::<c_void>(),
            (&raw mut job_count_u64).cast::<c_void>(),
        ];
        let geometry = htj2k_encode_codeblock_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_htj2k_encode_codeblocks_multi_input_cleanup(
        &self,
        output: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup)?;
        let mut output_ptr = output.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table_ptr = uvlc_table.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count_u64 =
            u64::try_from(job_count).map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = [
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut jobs_ptr).cast::<c_void>(),
            (&raw mut vlc_table0_ptr).cast::<c_void>(),
            (&raw mut vlc_table1_ptr).cast::<c_void>(),
            (&raw mut uvlc_table_ptr).cast::<c_void>(),
            (&raw mut statuses_ptr).cast::<c_void>(),
            (&raw mut job_count_u64).cast::<c_void>(),
        ];
        let geometry = htj2k_encode_codeblock_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_htj2k_encode_codeblocks_multi_input_cleanup_64(
        &self,
        output: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup64)?;
        let mut output_ptr = output.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table_ptr = uvlc_table.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count_u64 =
            u64::try_from(job_count).map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = [
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut jobs_ptr).cast::<c_void>(),
            (&raw mut vlc_table0_ptr).cast::<c_void>(),
            (&raw mut vlc_table1_ptr).cast::<c_void>(),
            (&raw mut uvlc_table_ptr).cast::<c_void>(),
            (&raw mut statuses_ptr).cast::<c_void>(),
            (&raw mut job_count_u64).cast::<c_void>(),
        ];
        let geometry = htj2k_encode_codeblock_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    fn launch_htj2k_compact_codeblocks(
        &self,
        scratch: &CudaDeviceBuffer,
        compact: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kCompactCodeblocks)?;
        let mut scratch_ptr = scratch.device_ptr();
        let mut compact_ptr = compact.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut job_count_u64 =
            u64::try_from(job_count).map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = [
            (&raw mut scratch_ptr).cast::<c_void>(),
            (&raw mut compact_ptr).cast::<c_void>(),
            (&raw mut jobs_ptr).cast::<c_void>(),
            (&raw mut job_count_u64).cast::<c_void>(),
        ];
        let geometry = htj2k_codeblock_sample_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_htj2k_packetize_cleanup(
        &self,
        payload: &CudaDeviceBuffer,
        payload_len: usize,
        packets: &CudaDeviceBuffer,
        subbands: &CudaDeviceBuffer,
        blocks: &CudaDeviceBuffer,
        subband_tag_states: &CudaDeviceBuffer,
        tag_nodes: &CudaDeviceBuffer,
        subband_tag_state_count: usize,
        tag_node_count: usize,
        output: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        packet_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kPacketizeCleanup)?;
        let mut payload_ptr = payload.device_ptr();
        let mut payload_len_u64 = u64::try_from(payload_len)
            .map_err(|_| CudaError::LengthTooLarge { len: payload_len })?;
        let mut packets_ptr = packets.device_ptr();
        let mut subbands_ptr = subbands.device_ptr();
        let mut blocks_ptr = blocks.device_ptr();
        let mut subband_tag_states_ptr = subband_tag_states.device_ptr();
        let mut tag_nodes_ptr = tag_nodes.device_ptr();
        let mut subband_tag_state_count_u64 =
            u64::try_from(subband_tag_state_count).map_err(|_| CudaError::LengthTooLarge {
                len: subband_tag_state_count,
            })?;
        let mut tag_node_count_u64 =
            u64::try_from(tag_node_count).map_err(|_| CudaError::LengthTooLarge {
                len: tag_node_count,
            })?;
        let mut output_ptr = output.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut packet_count_u64 = u64::try_from(packet_count)
            .map_err(|_| CudaError::LengthTooLarge { len: packet_count })?;
        let mut params = [
            (&raw mut payload_ptr).cast::<c_void>(),
            (&raw mut payload_len_u64).cast::<c_void>(),
            (&raw mut packets_ptr).cast::<c_void>(),
            (&raw mut subbands_ptr).cast::<c_void>(),
            (&raw mut blocks_ptr).cast::<c_void>(),
            (&raw mut subband_tag_states_ptr).cast::<c_void>(),
            (&raw mut tag_nodes_ptr).cast::<c_void>(),
            (&raw mut subband_tag_state_count_u64).cast::<c_void>(),
            (&raw mut tag_node_count_u64).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut statuses_ptr).cast::<c_void>(),
            (&raw mut packet_count_u64).cast::<c_void>(),
        ];
        let geometry = htj2k_packetize_launch_geometry(packet_count)
            .ok_or(CudaError::LengthTooLarge { len: packet_count })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_idwt_interleave(
        &self,
        bands: [&CudaDeviceBuffer; 4],
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        width: u32,
        height: u32,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_interleave_with_sync(bands, output, job, width, height, true)
    }

    fn launch_j2k_idwt_interleave_async(
        &self,
        bands: [&CudaDeviceBuffer; 4],
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        width: u32,
        height: u32,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_interleave_with_sync(bands, output, job, width, height, false)
    }

    fn launch_j2k_idwt_interleave_horizontal_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_interleave_horizontal_multi_ptr(
            jobs.device_ptr(),
            max_rows,
            job_count,
            synchronize,
        )
    }

    fn launch_j2k_idwt_interleave_horizontal_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kIdwtInterleaveHorizontalMulti)?;
        let mut jobs_ptr = jobs_ptr;
        let mut params = [(&raw mut jobs_ptr).cast::<c_void>()];
        let geometry = j2k_idwt_multi_1d_launch_geometry(max_rows, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    fn launch_j2k_idwt_interleave_horizontal_53_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_interleave_horizontal_53_multi_ptr(
            jobs.device_ptr(),
            max_rows,
            job_count,
            synchronize,
        )
    }

    fn launch_j2k_idwt_interleave_horizontal_53_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kIdwtInterleaveHorizontal53Multi)?;
        let mut jobs_ptr = jobs_ptr;
        let mut params = [(&raw mut jobs_ptr).cast::<c_void>()];
        let geometry = j2k_idwt_multi_coop_launch_geometry(max_rows, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    fn launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_width: usize,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kIdwtInterleaveHorizontal97Multi)?;
        let mut jobs_ptr = jobs_ptr;
        let mut params = [(&raw mut jobs_ptr).cast::<c_void>()];
        let geometry = j2k_idwt_multi_coop_axis_launch_geometry(max_rows, max_width, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    fn launch_j2k_idwt_interleave_with_sync(
        &self,
        bands: [&CudaDeviceBuffer; 4],
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        width: u32,
        height: u32,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kIdwtInterleave)?;
        let [ll, hl, lh, hh] = bands;
        let mut low_low_ptr = ll.device_ptr();
        let mut high_low_ptr = hl.device_ptr();
        let mut low_high_ptr = lh.device_ptr();
        let mut high_high_ptr = hh.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = [
            (&raw mut low_low_ptr).cast::<c_void>(),
            (&raw mut high_low_ptr).cast::<c_void>(),
            (&raw mut low_high_ptr).cast::<c_void>(),
            (&raw mut high_high_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut job_ptr).cast::<c_void>(),
        ];
        let geometry =
            j2k_dwt53_launch_geometry(width, height).ok_or(CudaError::ImageTooLarge {
                width,
                height,
                channels: 1,
            })?;
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    fn launch_j2k_idwt_horizontal(
        &self,
        kernel: CudaKernel,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        rows: usize,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_horizontal_with_sync(kernel, output, job, rows, true)
    }

    fn launch_j2k_idwt_horizontal_async(
        &self,
        kernel: CudaKernel,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        rows: usize,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_horizontal_with_sync(kernel, output, job, rows, false)
    }

    fn launch_j2k_idwt_horizontal_with_sync(
        &self,
        kernel: CudaKernel,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        rows: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(kernel)?;
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = [
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut job_ptr).cast::<c_void>(),
        ];
        let geometry =
            j2k_forward_rct_launch_geometry(rows).ok_or(CudaError::LengthTooLarge { len: rows })?;
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    fn launch_j2k_idwt_vertical(
        &self,
        kernel: CudaKernel,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        columns: usize,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_vertical_with_sync(kernel, output, job, columns, true)
    }

    fn launch_j2k_idwt_vertical_async(
        &self,
        kernel: CudaKernel,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        columns: usize,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_vertical_with_sync(kernel, output, job, columns, false)
    }

    fn launch_j2k_idwt_vertical_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        max_columns: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_vertical_multi_ptr(
            jobs.device_ptr(),
            max_columns,
            job_count,
            synchronize,
        )
    }

    fn launch_j2k_idwt_vertical_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_columns: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kIdwtVerticalMulti)?;
        let mut jobs_ptr = jobs_ptr;
        let mut params = [(&raw mut jobs_ptr).cast::<c_void>()];
        let geometry = j2k_idwt_multi_1d_launch_geometry(max_columns, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    fn launch_j2k_idwt_vertical_53_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        max_columns: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_vertical_53_multi_ptr(
            jobs.device_ptr(),
            max_columns,
            job_count,
            synchronize,
        )
    }

    fn launch_j2k_idwt_vertical_53_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_columns: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kIdwtVertical53Multi)?;
        let mut jobs_ptr = jobs_ptr;
        let mut params = [(&raw mut jobs_ptr).cast::<c_void>()];
        let geometry = j2k_idwt_multi_coop_launch_geometry(max_columns, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    fn launch_j2k_idwt_vertical_97_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_columns: usize,
        max_height: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        const COLUMNS_PER_BLOCK: usize = 4;
        const MIN_COLS4_JOBS: usize = 64;
        let (kernel, geometry) = if job_count >= MIN_COLS4_JOBS && max_height <= 256 {
            let geometry = j2k_idwt_multi_coop_columns_launch_geometry(
                max_columns,
                max_height,
                job_count,
                COLUMNS_PER_BLOCK,
            )
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
            (CudaKernel::J2kIdwtVertical97MultiCols4, geometry)
        } else {
            let geometry =
                j2k_idwt_multi_coop_axis_launch_geometry(max_columns, max_height, job_count)
                    .ok_or(CudaError::LengthTooLarge { len: job_count })?;
            (CudaKernel::J2kIdwtVertical97Multi, geometry)
        };
        let function = self.inner.kernel_function(kernel)?;
        let mut jobs_ptr = jobs_ptr;
        let mut params = [(&raw mut jobs_ptr).cast::<c_void>()];
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    fn launch_j2k_idwt_vertical_with_sync(
        &self,
        kernel: CudaKernel,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        columns: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(kernel)?;
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = [
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut job_ptr).cast::<c_void>(),
        ];
        let geometry = j2k_forward_rct_launch_geometry(columns)
            .ok_or(CudaError::LengthTooLarge { len: columns })?;
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    fn launch_j2k_store_gray8(
        &self,
        input: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kStoreGray8)?;
        let mut input_ptr = input.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = [
            (&raw mut input_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut job_ptr).cast::<c_void>(),
        ];
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_store_gray16(
        &self,
        input: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kStoreGray16)?;
        let mut input_ptr = input.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = [
            (&raw mut input_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut job_ptr).cast::<c_void>(),
        ];
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_inverse_mct(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        len: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kInverseMct)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = [
            (&raw mut plane0_ptr).cast::<c_void>(),
            (&raw mut plane1_ptr).cast::<c_void>(),
            (&raw mut plane2_ptr).cast::<c_void>(),
            (&raw mut job_ptr).cast::<c_void>(),
        ];
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_store_rgb8(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kStoreRgb8)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = [
            (&raw mut plane0_ptr).cast::<c_void>(),
            (&raw mut plane1_ptr).cast::<c_void>(),
            (&raw mut plane2_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut job_ptr).cast::<c_void>(),
        ];
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_store_rgb16(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kStoreRgb16)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = [
            (&raw mut plane0_ptr).cast::<c_void>(),
            (&raw mut plane1_ptr).cast::<c_void>(),
            (&raw mut plane2_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut job_ptr).cast::<c_void>(),
        ];
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_store_rgb8_mct_batch(
        &self,
        jobs: &CudaDeviceBuffer,
        max_pixels: usize,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kStoreRgb8MctBatch)?;
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = [(&raw mut jobs_ptr).cast::<c_void>()];
        let geometry = j2k_store_batch_launch_geometry(max_pixels, job_count)
            .ok_or(CudaError::LengthTooLarge { len: max_pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_store_rgb16_mct(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kStoreRgb16Mct)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = [
            (&raw mut plane0_ptr).cast::<c_void>(),
            (&raw mut plane1_ptr).cast::<c_void>(),
            (&raw mut plane2_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut job_ptr).cast::<c_void>(),
        ];
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_kernel(
        &self,
        function: CuFunction,
        geometry: kernels::CudaLaunchGeometry,
        params: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        self.launch_kernel_async(function, geometry, params)?;
        // SAFETY: `function` was loaded from a live module in this context, and
        // the kernel was launched on the current context; synchronize waits for
        // completion before callers inspect outputs.
        self.synchronize()
    }

    fn launch_kernel_async(
        &self,
        function: CuFunction,
        geometry: kernels::CudaLaunchGeometry,
        params: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        // SAFETY: `function` was loaded from a live module in this context, and
        // `params` contains kernel argument pointers valid for the launch call.
        let launch_status = unsafe {
            (self.inner.driver.cu_launch_kernel)(
                function,
                geometry.grid.0,
                geometry.grid.1,
                geometry.grid.2,
                geometry.block.0,
                geometry.block.1,
                geometry.block.2,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.inner.driver.check("cuLaunchKernel", launch_status)
    }

    /// Copy one device buffer to another through a CUDA kernel.
    pub fn copy_device_to_device_with_kernel(
        &self,
        src: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.copy_device_ptr_to_device_with_kernel(src.device_ptr(), src.byte_len())
    }

    fn copy_device_ptr_to_device_with_kernel(
        &self,
        src_ptr: CuDevicePtr,
        byte_len: usize,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.inner.set_current()?;
        let dst = self.allocate(byte_len)?;
        if byte_len == 0 {
            return Ok(dst);
        }

        let function = self.inner.kernel_function(CudaKernel::CopyU8)?;
        let mut dst_ptr = dst.device_ptr();
        let mut src_ptr = src_ptr;
        let mut len =
            u64::try_from(byte_len).map_err(|_| CudaError::LengthTooLarge { len: byte_len })?;
        let mut params = [
            (&raw mut dst_ptr).cast::<c_void>(),
            (&raw mut src_ptr).cast::<c_void>(),
            (&raw mut len).cast::<c_void>(),
        ];
        let geometry =
            copy_u8_launch_geometry(byte_len).ok_or(CudaError::LengthTooLarge { len: byte_len })?;

        self.launch_kernel(function, geometry, &mut params)?;

        Ok(dst)
    }

    /// Allocate an uninitialized CUDA device buffer.
    pub fn allocate(&self, len: usize) -> Result<CudaDeviceBuffer, CudaError> {
        self.inner.set_current()?;
        let mut ptr = 0;
        if len != 0 {
            // SAFETY: CUDA writes a device pointer for the requested byte size.
            self.inner.driver.check("cuMemAlloc_v2", unsafe {
                (self.inner.driver.cu_mem_alloc)(&raw mut ptr, len)
            })?;
        }
        Ok(CudaDeviceBuffer {
            context: self.clone(),
            ptr,
            len,
        })
    }

    fn memset_d32(
        &self,
        dst: &CudaDeviceBuffer,
        value: c_uint,
        words: usize,
    ) -> Result<(), CudaError> {
        self.inner.set_current()?;
        let required = words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaError::LengthTooLarge { len: words })?;
        if required > dst.byte_len() {
            return Err(CudaError::OutputTooSmall {
                required,
                have: dst.byte_len(),
            });
        }
        if words == 0 {
            return Ok(());
        }
        // SAFETY: `dst` is a live CUDA allocation in this context and `words`
        // was bounds-checked against the allocation byte length above.
        self.inner.driver.check("cuMemsetD32_v2", unsafe {
            (self.inner.driver.cu_memset_d32)(dst.device_ptr(), value, words)
        })
    }

    /// Allocate page-locked host memory for host-to-device staging.
    pub fn pinned_host_buffer(&self, len: usize) -> Result<CudaPinnedHostBuffer, CudaError> {
        self.inner.set_current()?;
        let mut ptr = std::ptr::null_mut();
        if len != 0 {
            // SAFETY: CUDA writes a page-locked host pointer for the requested
            // byte length. The allocation is freed by CudaPinnedHostBuffer.
            self.inner.driver.check("cuMemHostAlloc", unsafe {
                (self.inner.driver.cu_mem_host_alloc)(&raw mut ptr, len, 0)
            })?;
        }
        Ok(CudaPinnedHostBuffer {
            context: self.clone(),
            ptr: ptr.cast::<u8>(),
            len,
        })
    }

    /// Create a CUDA stream owned by this context.
    pub fn create_stream(&self) -> Result<CudaStream, CudaError> {
        self.inner.set_current()?;
        let mut stream = std::ptr::null_mut();
        // SAFETY: CUDA writes a new stream handle, destroyed by CudaStream.
        self.inner.driver.check("cuStreamCreate", unsafe {
            (self.inner.driver.cu_stream_create)(&raw mut stream, 0)
        })?;
        Ok(CudaStream {
            context: self.clone(),
            stream,
        })
    }

    /// Create a CUDA timing event owned by this context.
    pub fn create_event(&self) -> Result<CudaEvent, CudaError> {
        self.inner.set_current()?;
        let mut event = std::ptr::null_mut();
        // SAFETY: CUDA writes a new event handle, destroyed by CudaEvent.
        self.inner.driver.check("cuEventCreate", unsafe {
            (self.inner.driver.cu_event_create)(&raw mut event, 0)
        })?;
        Ok(CudaEvent {
            context: self.clone(),
            event,
        })
    }

    /// Time work submitted to the default CUDA stream and return elapsed microseconds.
    pub fn time_default_stream_us<T>(
        &self,
        work: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<(T, u128), CudaError> {
        self.inner.set_current()?;
        if cuda_stage_timings_disabled() {
            return work().map(|output| (output, 0));
        }
        let start = self.create_event()?;
        let end = self.create_event()?;
        start.record_default_stream()?;
        let output = match work() {
            Ok(output) => output,
            Err(error) => {
                // Timed closures may submit asynchronous default-stream work.
                // On a later host-side error, wait before dropping any device
                // buffers captured by the closure.
                self.synchronize()?;
                return Err(error);
            }
        };
        end.record_default_stream()?;
        end.synchronize()?;
        Ok((output, elapsed_event_us_ceil(&start, &end)?))
    }

    /// Run work inside an optional NVTX profiling range.
    ///
    /// The range is a no-op unless the crate is built with `cuda-profiling`
    /// and an NVTX runtime library can be loaded dynamically.
    pub fn with_nvtx_range<T>(
        &self,
        name: &str,
        work: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        let _range = CudaNvtxRange::push(name);
        work()
    }

    /// Time work submitted to the default CUDA stream inside an optional NVTX range.
    ///
    /// The NVTX range is a no-op unless the crate is built with
    /// `cuda-profiling` and an NVTX runtime library can be loaded dynamically.
    pub fn time_default_stream_named_us<T>(
        &self,
        name: &str,
        work: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<(T, u128), CudaError> {
        self.with_nvtx_range(name, || self.time_default_stream_us(work))
    }

    /// Optionally time work submitted to the default CUDA stream inside an NVTX range.
    pub fn time_default_stream_named_us_if<T>(
        &self,
        collect_stage_timings: bool,
        name: &str,
        work: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<(T, u128), CudaError> {
        if collect_stage_timings {
            self.time_default_stream_named_us(name, work)
        } else {
            self.with_nvtx_range(name, || work().map(|output| (output, 0)))
        }
    }

    /// Synchronize all work submitted to this CUDA context.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.inner.set_current()?;
        // SAFETY: a CUDA context is current for this `CudaContext`.
        let status = unsafe { (self.inner.driver.cu_ctx_synchronize)() };
        self.inner.driver.check("cuCtxSynchronize", status)
    }

    /// Preload a bundled CUDA kernel module and return its metadata handle.
    pub fn preload_kernel_module(
        &self,
        kernel: CudaKernelName,
    ) -> Result<CudaKernelModule, CudaError> {
        let _ = self.inner.kernel_function(kernel.kernel())?;
        Ok(CudaKernelModule {
            kernel,
            entrypoint: kernel.entrypoint(),
        })
    }

    /// Create a reusable device-buffer pool for this context.
    pub fn buffer_pool(&self) -> CudaBufferPool {
        CudaBufferPool::new(self.clone())
    }

    /// Create a reusable best-fit device-buffer pool for workloads with many
    /// same-sized intermediate buffers.
    pub fn best_fit_buffer_pool(&self) -> CudaBufferPool {
        CudaBufferPool::new_size_buckets(self.clone())
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
    fn kernel(self) -> CudaKernel {
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

    fn entrypoint(self) -> &'static str {
        match self {
            Self::CopyU8 => "signinum_copy_u8",
            Self::J2kDeinterleaveToF32 => "signinum_j2k_deinterleave_to_f32",
            Self::J2kForwardRct => "signinum_j2k_forward_rct",
            Self::J2kForwardIct => "signinum_j2k_forward_ict",
            Self::J2kForwardDwt53Horizontal => "signinum_j2k_forward_dwt53_horizontal",
            Self::J2kForwardDwt53Vertical => "signinum_j2k_forward_dwt53_vertical",
            Self::J2kForwardDwt97Horizontal => "signinum_j2k_forward_dwt97_horizontal",
            Self::J2kForwardDwt97Vertical => "signinum_j2k_forward_dwt97_vertical",
            Self::J2kQuantizeSubband => "signinum_j2k_quantize_subband",
            Self::J2kQuantizeSubbandStrided => "signinum_j2k_quantize_subband_strided",
            Self::Htj2kDecodeCodeblocks => "signinum_htj2k_decode_codeblocks",
            Self::Htj2kDecodeCodeblocksMultiCleanupDequantize => {
                "signinum_htj2k_decode_codeblocks_multi_cleanup_dequantize"
            }
            Self::J2kDequantizeHtj2kCodeblocks => "signinum_j2k_dequantize_htj2k_codeblocks",
            Self::J2kDequantizeHtj2kCodeblocksMulti => {
                "signinum_j2k_dequantize_htj2k_codeblocks_multi"
            }
            Self::J2kDequantizeHtj2kCleanupJobsMulti => {
                "signinum_j2k_dequantize_htj2k_cleanup_jobs_multi"
            }
            Self::J2kIdwtInterleave => "signinum_j2k_idwt_interleave",
            Self::J2kIdwtInterleaveHorizontal53Multi => {
                "signinum_j2k_idwt_interleave_horizontal_53_multi"
            }
            Self::J2kIdwtInterleaveHorizontal97Multi => {
                "signinum_j2k_idwt_interleave_horizontal_97_multi"
            }
            Self::J2kIdwtHorizontal => "signinum_j2k_idwt_horizontal",
            Self::J2kIdwtHorizontal53 => "signinum_j2k_idwt_horizontal_53",
            Self::J2kIdwtHorizontal97 => "signinum_j2k_idwt_horizontal_97",
            Self::J2kIdwtVertical => "signinum_j2k_idwt_vertical",
            Self::J2kIdwtVertical53Multi => "signinum_j2k_idwt_vertical_53_multi",
            Self::J2kIdwtVertical97Multi => "signinum_j2k_idwt_vertical_97_multi",
            Self::J2kIdwtVertical97MultiCols4 => "signinum_j2k_idwt_vertical_97_multi_cols4",
            Self::J2kIdwtVertical53 => "signinum_j2k_idwt_vertical_53",
            Self::J2kIdwtVertical97 => "signinum_j2k_idwt_vertical_97",
            Self::J2kInverseDwtSingle => "signinum_j2k_inverse_dwt_single",
            Self::J2kInverseMct => "signinum_j2k_inverse_mct",
            Self::J2kStoreGray8 => "signinum_j2k_store_gray8",
            Self::J2kStoreGray16 => "signinum_j2k_store_gray16",
            Self::J2kStoreRgb8 => "signinum_j2k_store_rgb8",
            Self::J2kStoreRgb8Mct => "signinum_j2k_store_rgb8_mct",
            Self::J2kStoreRgb8MctBatch => "signinum_j2k_store_rgb8_mct_batch",
            Self::J2kStoreRgb16 => "signinum_j2k_store_rgb16",
            Self::J2kStoreRgb16Mct => "signinum_j2k_store_rgb16_mct",
            Self::Htj2kEncodeCodeblock => "signinum_htj2k_encode_codeblock",
            Self::Htj2kEncodeCodeblocks => "signinum_htj2k_encode_codeblocks",
            Self::Htj2kEncodeCodeblocksMultiInput => "signinum_htj2k_encode_codeblocks_multi_input",
            Self::Htj2kEncodeCodeblocksMultiInputCleanup => {
                "signinum_htj2k_encode_codeblocks_multi_input_cleanup"
            }
            Self::Htj2kEncodeCodeblocksMultiInputCleanup64 => {
                "signinum_htj2k_encode_codeblocks_multi_input_cleanup_64"
            }
            Self::Htj2kCompactCodeblocks => "signinum_htj2k_compact_codeblocks",
            Self::Htj2kPacketizeCleanup => "signinum_htj2k_packetize_cleanup",
        }
    }
}

/// Metadata for a preloaded CUDA kernel module entry point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaKernelModule {
    kernel: CudaKernelName,
    entrypoint: &'static str,
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

/// Page-locked host staging buffer.
#[derive(Debug)]
pub struct CudaPinnedHostBuffer {
    context: CudaContext,
    ptr: *mut u8,
    len: usize,
}

impl CudaPinnedHostBuffer {
    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this buffer has zero length.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Immutable byte view of the pinned allocation.
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: ptr is a live pinned allocation of len bytes.
            unsafe { std::slice::from_raw_parts(self.ptr.cast_const(), self.len) }
        }
    }

    /// Mutable byte view of the pinned allocation.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: ptr is uniquely borrowed through &mut self and covers len
            // bytes allocated by CUDA.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
}

impl Drop for CudaPinnedHostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = self.context.inner.set_current();
            // SAFETY: ptr was returned by cuMemHostAlloc for this process.
            let _ = unsafe { (self.context.inner.driver.cu_mem_free_host)(self.ptr.cast()) };
        }
    }
}

// SAFETY: The pinned allocation is owned by this value and CUDA frees it on
// drop. Mutable access still requires &mut self.
unsafe impl Send for CudaPinnedHostBuffer {}

/// CUDA stream RAII handle.
#[derive(Debug)]
pub struct CudaStream {
    context: CudaContext,
    stream: CuStream,
}

impl CudaStream {
    /// Synchronize all work submitted to this stream.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.context.inner.set_current()?;
        // SAFETY: stream is a live CUDA stream owned by this handle.
        self.context
            .inner
            .driver
            .check("cuStreamSynchronize", unsafe {
                (self.context.inner.driver.cu_stream_synchronize)(self.stream)
            })
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            let _ = self.context.inner.set_current();
            // SAFETY: stream was created by this context. Drop cannot surface
            // errors, so cleanup failures are ignored.
            let _ = unsafe { (self.context.inner.driver.cu_stream_destroy)(self.stream) };
        }
    }
}

// SAFETY: CUDA stream handles are driver-owned resources. The Rust handle owns
// destruction and does not expose mutable aliasing of Rust memory.
unsafe impl Send for CudaStream {}

/// CUDA event RAII handle for timing and synchronization.
#[derive(Debug)]
pub struct CudaEvent {
    context: CudaContext,
    event: CuEvent,
}

impl CudaEvent {
    /// Record this event on a CUDA stream.
    pub fn record(&self, stream: &CudaStream) -> Result<(), CudaError> {
        self.context.inner.set_current()?;
        // SAFETY: event and stream are live CUDA handles.
        self.context.inner.driver.check("cuEventRecord", unsafe {
            (self.context.inner.driver.cu_event_record)(self.event, stream.stream)
        })
    }

    fn record_default_stream(&self) -> Result<(), CudaError> {
        self.context.inner.set_current()?;
        // SAFETY: a null stream is CUDA's default stream for the current context.
        self.context.inner.driver.check("cuEventRecord", unsafe {
            (self.context.inner.driver.cu_event_record)(self.event, std::ptr::null_mut())
        })
    }

    /// Wait for this event to complete.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.context.inner.set_current()?;
        // SAFETY: event is a live CUDA event owned by this handle.
        self.context
            .inner
            .driver
            .check("cuEventSynchronize", unsafe {
                (self.context.inner.driver.cu_event_synchronize)(self.event)
            })
    }

    /// Elapsed time in microseconds from `start` to `end`.
    pub fn elapsed_time_us(start: &Self, end: &Self) -> Result<f32, CudaError> {
        end.context.inner.set_current()?;
        let mut millis = 0.0f32;
        // SAFETY: start and end are live CUDA events that have been recorded.
        let status = unsafe {
            (end.context.inner.driver.cu_event_elapsed_time)(
                &raw mut millis,
                start.event,
                end.event,
            )
        };
        end.context
            .inner
            .driver
            .check("cuEventElapsedTime", status)?;
        Ok(millis * 1000.0)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn elapsed_event_us_ceil(start: &CudaEvent, end: &CudaEvent) -> Result<u128, CudaError> {
    let elapsed = CudaEvent::elapsed_time_us(start, end)?;
    if elapsed <= 0.0 {
        return Ok(1);
    }
    Ok(elapsed.ceil() as u128)
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            let _ = self.context.inner.set_current();
            // SAFETY: event was created by this context. Drop cannot surface
            // errors, so cleanup failures are ignored.
            let _ = unsafe { (self.context.inner.driver.cu_event_destroy)(self.event) };
        }
    }
}

// SAFETY: CUDA event handles are driver-owned resources. The Rust handle owns
// destruction and does not expose mutable aliasing of Rust memory.
unsafe impl Send for CudaEvent {}

/// Owned CUDA device buffer.
#[derive(Debug)]
pub struct CudaDeviceBuffer {
    context: CudaContext,
    ptr: CuDevicePtr,
    len: usize,
}

/// Typed immutable device buffer view.
#[derive(Clone, Copy, Debug)]
pub struct CudaDeviceBufferView<'a, T> {
    ptr: CuDevicePtr,
    len: usize,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<T> CudaDeviceBufferView<'_, T> {
    /// Raw CUDA device pointer value for kernel argument binding.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Number of typed elements in this view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Typed mutable device buffer view.
#[derive(Debug)]
pub struct CudaDeviceBufferViewMut<'a, T> {
    ptr: CuDevicePtr,
    len: usize,
    _marker: std::marker::PhantomData<&'a mut T>,
}

impl<T> CudaDeviceBufferViewMut<'_, T> {
    /// Raw CUDA device pointer value for kernel argument binding.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Number of typed elements in this view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Reusable CUDA device-buffer pool for repeated adapter dispatches.
#[derive(Clone, Debug)]
pub struct CudaBufferPool {
    inner: Arc<CudaBufferPoolInner>,
}

#[derive(Debug)]
struct CudaBufferPoolInner {
    context: CudaContext,
    free: Mutex<CudaBufferPoolFree>,
}

#[derive(Debug)]
enum CudaBufferPoolFree {
    FirstFit(Vec<CudaDeviceBuffer>),
    SizeBuckets(BTreeMap<usize, Vec<CudaDeviceBuffer>>),
}

/// Diagnostics for one traced [`CudaBufferPool`] acquisition.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaBufferPoolTakeTrace {
    /// Requested byte length for the checkout.
    pub requested_len: usize,
    /// Number of cached free buffers before the checkout.
    pub free_count_before: usize,
    /// Number of cached entries examined while finding a reusable buffer or allocating.
    pub scanned_count: usize,
    /// Whether the checkout reused a cached allocation.
    pub reused: bool,
    /// Actual allocation byte length backing the checkout.
    pub allocation_byte_len: usize,
}

impl CudaBufferPool {
    /// Create a new pool for `context`.
    pub fn new(context: CudaContext) -> Self {
        Self {
            inner: Arc::new(CudaBufferPoolInner {
                context,
                free: Mutex::new(CudaBufferPoolFree::FirstFit(Vec::new())),
            }),
        }
    }

    fn new_size_buckets(context: CudaContext) -> Self {
        Self {
            inner: Arc::new(CudaBufferPoolInner {
                context,
                free: Mutex::new(CudaBufferPoolFree::SizeBuckets(BTreeMap::new())),
            }),
        }
    }

    /// Acquire a device buffer with at least `len` bytes.
    pub fn take(&self, len: usize) -> Result<CudaPooledDeviceBuffer, CudaError> {
        let mut free = self
            .inner
            .free
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        let (reusable_buffer, _) = pool_take_fit_buffer(&mut free, len);
        let buffer = if let Some(buffer) = reusable_buffer {
            buffer
        } else {
            drop(free);
            self.inner.context.allocate(len)?
        };
        Ok(CudaPooledDeviceBuffer {
            buffer: Some(buffer),
            requested_len: len,
            pool: self.inner.clone(),
        })
    }

    /// Acquire a device buffer with diagnostics for profiling pool behavior.
    pub fn take_with_trace(
        &self,
        len: usize,
    ) -> Result<(CudaPooledDeviceBuffer, CudaBufferPoolTakeTrace), CudaError> {
        let mut free = self
            .inner
            .free
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        let free_count_before = free.cached_count();
        let (reusable_buffer, scanned_count) = pool_take_fit_buffer(&mut free, len);
        let reused = reusable_buffer.is_some();
        let buffer = if let Some(buffer) = reusable_buffer {
            buffer
        } else {
            drop(free);
            self.inner.context.allocate(len)?
        };
        let allocation_byte_len = buffer.byte_len();
        let trace = CudaBufferPoolTakeTrace {
            requested_len: len,
            free_count_before,
            scanned_count,
            reused,
            allocation_byte_len,
        };
        Ok((
            CudaPooledDeviceBuffer {
                buffer: Some(buffer),
                requested_len: len,
                pool: self.inner.clone(),
            },
            trace,
        ))
    }

    /// Upload host bytes into a pooled device buffer.
    pub fn upload(&self, bytes: &[u8]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        let buffer = self.take(bytes.len())?;
        if !bytes.is_empty() {
            self.inner.context.inner.set_current()?;
            // SAFETY: `buffer` is a live device allocation with at least
            // `bytes.len()` bytes for this checkout, and `bytes` is valid for
            // that many host bytes.
            let result = unsafe {
                (self.inner.context.inner.driver.cu_memcpy_htod)(
                    buffer.device_ptr(),
                    bytes.as_ptr().cast::<c_void>(),
                    bytes.len(),
                )
            };
            self.inner
                .context
                .inner
                .driver
                .check("cuMemcpyHtoD_v2", result)?;
        }
        Ok(buffer)
    }

    /// Upload host bytes through temporary page-locked staging into a pooled device buffer.
    pub fn upload_pinned(&self, bytes: &[u8]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        if bytes.is_empty() {
            return self.upload(bytes);
        }

        let buffer = self.take(bytes.len())?;
        let mut staging = self.inner.context.take_pinned_upload_staging(bytes.len())?;
        staging.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        self.inner.context.inner.set_current()?;
        // SAFETY: `buffer` is a live device allocation with at least
        // `bytes.len()` bytes, and the pinned staging slice covers that range.
        let upload_result = unsafe {
            (self.inner.context.inner.driver.cu_memcpy_htod)(
                buffer.device_ptr(),
                staging.as_slice()[..bytes.len()].as_ptr().cast::<c_void>(),
                bytes.len(),
            )
        };
        let upload_result = self
            .inner
            .context
            .inner
            .driver
            .check("cuMemcpyHtoD_v2", upload_result);
        let recycle_result = self.inner.context.recycle_pinned_upload_staging(staging);
        match (upload_result, recycle_result) {
            (Ok(()), Ok(())) => Ok(buffer),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }

    /// Upload host `f32` samples into a pooled device buffer.
    pub fn upload_f32(&self, samples: &[f32]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        self.upload(f32_slice_as_bytes(samples))
    }

    /// Upload host `f32` samples through pinned staging into a pooled device buffer.
    pub fn upload_f32_pinned(&self, samples: &[f32]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        self.upload_pinned(f32_slice_as_bytes(samples))
    }

    /// Upload host `i16` samples into a pooled device buffer.
    pub fn upload_i16(&self, samples: &[i16]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        self.upload(i16_slice_as_bytes(samples))
    }

    /// Upload host `i16` samples through pinned staging into a pooled device buffer.
    pub fn upload_i16_pinned(&self, samples: &[i16]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        self.upload_pinned(i16_slice_as_bytes(samples))
    }

    /// Number of free buffers currently cached by the pool.
    pub fn cached_count(&self) -> Result<usize, CudaError> {
        Ok(self
            .inner
            .free
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?
            .cached_count())
    }
}

impl CudaBufferPoolFree {
    fn cached_count(&self) -> usize {
        match self {
            Self::FirstFit(free) => free.len(),
            Self::SizeBuckets(free) => free.values().map(Vec::len).sum(),
        }
    }
}

fn pool_take_fit_buffer(
    free: &mut CudaBufferPoolFree,
    len: usize,
) -> (Option<CudaDeviceBuffer>, usize) {
    match free {
        CudaBufferPoolFree::FirstFit(free) => pool_take_first_fit_buffer(free, len),
        CudaBufferPoolFree::SizeBuckets(free) => pool_take_size_bucket_buffer(free, len),
    }
}

fn pool_take_first_fit_buffer(
    free: &mut Vec<CudaDeviceBuffer>,
    len: usize,
) -> (Option<CudaDeviceBuffer>, usize) {
    let mut examined = 0usize;
    for (index, buffer) in free.iter().enumerate() {
        examined = examined.saturating_add(1);
        if buffer.byte_len() >= len {
            return (Some(free.swap_remove(index)), examined);
        }
    }
    (None, examined)
}

fn pool_take_size_bucket_buffer(
    free: &mut BTreeMap<usize, Vec<CudaDeviceBuffer>>,
    len: usize,
) -> (Option<CudaDeviceBuffer>, usize) {
    let Some(size) = free.range(len..).next().map(|(size, _)| *size) else {
        return (None, usize::from(!free.is_empty()));
    };
    let buffer = free
        .get_mut(&size)
        .expect("selected CUDA buffer pool size bucket must exist")
        .pop();
    if free.get(&size).is_some_and(Vec::is_empty) {
        free.remove(&size);
    }
    (buffer, 1)
}

#[cfg(test)]
fn pool_fit_buffer_index_by_len<I>(lengths: I, len: usize) -> Option<usize>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    let lengths = lengths.into_iter().collect::<Vec<_>>();
    let mut left = 0usize;
    let mut right = lengths.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if lengths[mid].1 < len {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    (left < lengths.len()).then_some(lengths[left].0)
}

/// Device buffer borrowed from a [`CudaBufferPool`].
#[derive(Debug)]
pub struct CudaPooledDeviceBuffer {
    buffer: Option<CudaDeviceBuffer>,
    requested_len: usize,
    pool: Arc<CudaBufferPoolInner>,
}

impl CudaPooledDeviceBuffer {
    /// Raw CUDA device pointer value for kernel argument binding.
    pub fn device_ptr(&self) -> u64 {
        self.buffer.as_ref().map_or(0, CudaDeviceBuffer::device_ptr)
    }

    /// Requested byte length for the current checkout.
    pub fn byte_len(&self) -> usize {
        self.requested_len
    }

    /// Actual device allocation byte length.
    pub fn allocation_byte_len(&self) -> usize {
        self.buffer.as_ref().map_or(0, CudaDeviceBuffer::byte_len)
    }

    /// Borrow the underlying device buffer while the checkout is live.
    pub fn as_device_buffer(&self) -> Option<&CudaDeviceBuffer> {
        self.buffer.as_ref()
    }

    /// Copy the requested bytes for this checkout into caller-owned host output.
    pub fn copy_to_host(&self, out: &mut [u8]) -> Result<(), CudaError> {
        if out.len() < self.requested_len {
            return Err(CudaError::OutputTooSmall {
                required: self.requested_len,
                have: out.len(),
            });
        }
        if self.requested_len == 0 {
            return Ok(());
        }
        let buffer = self
            .buffer
            .as_ref()
            .ok_or_else(|| CudaError::InvalidArgument {
                message: "pooled CUDA buffer checkout is empty".to_string(),
            })?;
        buffer.context.inner.set_current()?;
        // SAFETY: `buffer.ptr` is a live allocation with at least
        // `requested_len` bytes for this checkout, and `out` was validated.
        let result = unsafe {
            (buffer.context.inner.driver.cu_memcpy_dtoh)(
                out.as_mut_ptr().cast::<c_void>(),
                buffer.ptr,
                self.requested_len,
            )
        };
        buffer
            .context
            .inner
            .driver
            .check("cuMemcpyDtoH_v2", result)?;
        Ok(())
    }
}

impl Drop for CudaPooledDeviceBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let free = self.pool.free.lock();
            match free {
                Ok(mut free) => match &mut *free {
                    CudaBufferPoolFree::FirstFit(free) => free.push(buffer),
                    CudaBufferPoolFree::SizeBuckets(free) => {
                        free.entry(buffer.byte_len()).or_default().push(buffer);
                    }
                },
                Err(_) => drop(buffer),
            }
        }
    }
}

/// Device buffer plus execution metadata.
#[derive(Debug)]
pub struct CudaKernelOutput {
    buffer: CudaDeviceBuffer,
    execution: CudaExecutionStats,
}

/// Multiple device buffers plus shared execution metadata from one batched kernel.
#[derive(Debug)]
pub struct CudaKernelBatchOutput {
    outputs: Vec<CudaDeviceBuffer>,
    execution: CudaExecutionStats,
}

/// One byte range inside a contiguous CUDA batch output allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaDeviceBufferRange {
    /// Byte offset from the start of the contiguous allocation.
    pub offset: usize,
    /// Byte length for this output item.
    pub len: usize,
}

/// One contiguous device buffer plus per-item ranges from one batched kernel.
#[derive(Debug)]
pub struct CudaKernelContiguousBatchOutput {
    output: CudaDeviceBuffer,
    ranges: Vec<CudaDeviceBufferRange>,
    execution: CudaExecutionStats,
}

/// Pooled device buffer plus execution metadata.
#[derive(Debug)]
pub struct CudaPooledKernelOutput {
    buffer: CudaPooledDeviceBuffer,
    execution: CudaExecutionStats,
}

/// Enqueued CUDA work plus pooled resources that must stay live until the
/// default stream is synchronized.
#[derive(Debug)]
pub struct CudaQueuedExecution {
    resources: Vec<CudaPooledDeviceBuffer>,
    execution: CudaExecutionStats,
}

impl CudaQueuedExecution {
    /// CUDA execution counters for the enqueued work.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Number of pooled resource buffers held live for the queued work.
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }
}

/// Enqueued HTJ2K cleanup work plus pooled resources/statuses that must stay
/// live until `finish` validates kernel completion.
#[derive(Debug)]
pub struct CudaQueuedHtj2kCleanup {
    resources: Vec<CudaPooledDeviceBuffer>,
    status_buffer: Option<CudaPooledDeviceBuffer>,
    status_count: usize,
    kernel_name: &'static str,
    execution: CudaExecutionStats,
}

impl CudaQueuedHtj2kCleanup {
    /// CUDA execution counters for the enqueued cleanup work.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Number of pooled resource buffers held live for the queued cleanup work.
    pub fn resource_count(&self) -> usize {
        self.resources.len() + usize::from(self.status_buffer.is_some())
    }

    /// Synchronize through status download and validate kernel statuses.
    pub fn finish(self) -> Result<CudaExecutionStats, CudaError> {
        let Some(status_buffer) = self.status_buffer else {
            return Ok(self.execution);
        };

        let mut statuses = vec![CudaHtj2kStatus::default(); self.status_count];
        status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses))?;
        if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
            return Err(CudaError::KernelStatus {
                kernel: self.kernel_name,
                code: status.code,
                detail: status.detail,
            });
        }

        Ok(self.execution)
    }
}

/// Device-resident interleaved JPEG 2000 input pixels with row stride metadata.
#[derive(Clone, Copy, Debug)]
pub struct CudaJ2kStridedInterleavedPixels<'a> {
    /// Backing CUDA device byte buffer.
    pub buffer: &'a CudaDeviceBuffer,
    /// Byte offset to the first pixel in `buffer`.
    pub byte_offset: usize,
    /// Active input width in pixels.
    pub width: u32,
    /// Active input height in pixels.
    pub height: u32,
    /// Bytes between the start of consecutive rows.
    pub pitch_bytes: usize,
    /// Number of interleaved components per pixel.
    pub num_components: u8,
    /// Integer sample precision.
    pub bit_depth: u8,
    /// Whether integer samples are signed.
    pub signed: bool,
}

/// Resident f32 component planes produced by CUDA JPEG 2000 encode preparation.
#[derive(Debug)]
pub struct CudaJ2kResidentComponents {
    buffer: CudaDeviceBuffer,
    num_pixels: usize,
    num_components: u8,
    execution: CudaExecutionStats,
}

impl CudaJ2kResidentComponents {
    /// Contiguous component-major f32 device buffer.
    pub fn buffer(&self) -> &CudaDeviceBuffer {
        &self.buffer
    }

    /// Number of pixels in each component plane.
    pub fn num_pixels(&self) -> usize {
        self.num_pixels
    }

    /// Number of resident component planes.
    pub fn num_components(&self) -> u8 {
        self.num_components
    }

    /// CUDA execution counters for the producing dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Download component planes into host memory for verification or host APIs.
    pub fn download_components(&self) -> Result<Vec<Vec<f32>>, CudaError> {
        if self.num_pixels == 0 {
            return Ok(vec![Vec::new(); usize::from(self.num_components)]);
        }
        let sample_count = self
            .num_pixels
            .checked_mul(usize::from(self.num_components))
            .ok_or(CudaError::LengthTooLarge {
                len: self.num_pixels,
            })?;
        let mut flattened = vec![0.0f32; sample_count];
        self.buffer
            .copy_to_host(f32_slice_as_bytes_mut(&mut flattened))?;
        Ok(flattened
            .chunks_exact(self.num_pixels)
            .map(<[f32]>::to_vec)
            .collect())
    }

    fn component_plane_device_ptr(&self, component: u8) -> Result<CuDevicePtr, CudaError> {
        if component >= self.num_components {
            return Err(CudaError::InvalidArgument {
                message: "component plane index is out of range".to_string(),
            });
        }
        let plane_bytes = self
            .num_pixels
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge {
                len: self.num_pixels,
            })?;
        let offset = plane_bytes
            .checked_mul(usize::from(component))
            .ok_or(CudaError::LengthTooLarge { len: plane_bytes })?;
        let end = offset
            .checked_add(plane_bytes)
            .ok_or(CudaError::LengthTooLarge { len: offset })?;
        if end > self.buffer.byte_len() {
            return Err(CudaError::OutputTooSmall {
                required: end,
                have: self.buffer.byte_len(),
            });
        }
        let offset =
            u64::try_from(offset).map_err(|_| CudaError::LengthTooLarge { len: offset })?;
        self.buffer
            .device_ptr()
            .checked_add(offset)
            .ok_or(CudaError::LengthTooLarge {
                len: self.buffer.byte_len(),
            })
    }
}

/// Host-visible component planes produced by CUDA pixel deinterleave.
#[derive(Debug)]
pub struct CudaJ2kDeinterleavedComponents {
    components: Vec<Vec<f32>>,
    execution: CudaExecutionStats,
}

impl CudaJ2kDeinterleavedComponents {
    /// Per-component f32 sample planes in component order.
    pub fn components(&self) -> &[Vec<f32>] {
        &self.components
    }

    /// CUDA execution counters for the deinterleave dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Consume the output and return owned component planes.
    pub fn into_components(self) -> Vec<Vec<f32>> {
        self.components
    }
}

/// Forward 5/3 DWT output and level metadata.
#[derive(Debug)]
pub struct CudaDwt53Output {
    transformed: Vec<f32>,
    levels: Vec<CudaDwt53LevelShape>,
    ll_width: u32,
    ll_height: u32,
    execution: CudaExecutionStats,
}

impl CudaDwt53Output {
    /// Transformed coefficients downloaded to host memory.
    pub fn transformed(&self) -> &[f32] {
        &self.transformed
    }

    /// Per-level DWT shapes.
    pub fn levels(&self) -> &[CudaDwt53LevelShape] {
        &self.levels
    }

    /// Dimensions of the final low-low band.
    pub fn ll_dimensions(&self) -> (u32, u32) {
        (self.ll_width, self.ll_height)
    }

    /// CUDA execution counters for the transform.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

/// Resident forward 5/3 DWT output and level metadata.
#[derive(Debug)]
pub struct CudaResidentDwt53Output {
    buffer: CudaDeviceBuffer,
    sample_count: usize,
    levels: Vec<CudaDwt53LevelShape>,
    ll_width: u32,
    ll_height: u32,
    execution: CudaExecutionStats,
}

impl CudaResidentDwt53Output {
    /// Resident component-major transformed coefficient buffer.
    pub fn buffer(&self) -> &CudaDeviceBuffer {
        &self.buffer
    }

    /// Transformed coefficient count.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Download transformed coefficients into host memory.
    pub fn download_transformed(&self) -> Result<Vec<f32>, CudaError> {
        let mut transformed = vec![0f32; self.sample_count];
        self.buffer
            .copy_to_host(f32_slice_as_bytes_mut(&mut transformed))?;
        Ok(transformed)
    }

    /// Per-level DWT shapes.
    pub fn levels(&self) -> &[CudaDwt53LevelShape] {
        &self.levels
    }

    /// Dimensions of the final low-low band.
    pub fn ll_dimensions(&self) -> (u32, u32) {
        (self.ll_width, self.ll_height)
    }

    /// CUDA execution counters for the transform.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

/// Forward 9/7 DWT output and level metadata.
#[derive(Debug)]
pub struct CudaDwt97Output {
    transformed: Vec<f32>,
    levels: Vec<CudaDwt53LevelShape>,
    ll_width: u32,
    ll_height: u32,
    execution: CudaExecutionStats,
}

impl CudaDwt97Output {
    /// Transformed coefficients downloaded to host memory.
    pub fn transformed(&self) -> &[f32] {
        &self.transformed
    }

    /// Per-level DWT shapes.
    pub fn levels(&self) -> &[CudaDwt53LevelShape] {
        &self.levels
    }

    /// Dimensions of the final low-low band.
    pub fn ll_dimensions(&self) -> (u32, u32) {
        (self.ll_width, self.ll_height)
    }

    /// CUDA execution counters for the transform.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

/// Resident forward 9/7 DWT output and level metadata.
#[derive(Debug)]
pub struct CudaResidentDwt97Output {
    buffer: CudaDeviceBuffer,
    sample_count: usize,
    levels: Vec<CudaDwt53LevelShape>,
    ll_width: u32,
    ll_height: u32,
    execution: CudaExecutionStats,
}

impl CudaResidentDwt97Output {
    /// Resident component-major transformed coefficient buffer.
    pub fn buffer(&self) -> &CudaDeviceBuffer {
        &self.buffer
    }

    /// Transformed coefficient count.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Download transformed coefficients into host memory.
    pub fn download_transformed(&self) -> Result<Vec<f32>, CudaError> {
        let mut transformed = vec![0f32; self.sample_count];
        self.buffer
            .copy_to_host(f32_slice_as_bytes_mut(&mut transformed))?;
        Ok(transformed)
    }

    /// Per-level DWT shapes.
    pub fn levels(&self) -> &[CudaDwt53LevelShape] {
        &self.levels
    }

    /// Dimensions of the final low-low band.
    pub fn ll_dimensions(&self) -> (u32, u32) {
        (self.ll_width, self.ll_height)
    }

    /// CUDA execution counters for the transform.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

/// JPEG 2000 sub-band quantization parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaJ2kQuantizeJob {
    /// Quantization step-size exponent.
    pub step_exponent: u16,
    /// Quantization step-size mantissa.
    pub step_mantissa: u16,
    /// Nominal range bits for this sub-band.
    pub range_bits: u8,
    /// Whether to use reversible integer quantization.
    pub reversible: bool,
}

/// Resident strided sub-band rectangle and quantization parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaJ2kQuantizeSubbandRegionJob {
    /// X offset, in f32 samples, of the sub-band rectangle inside the resident plane.
    pub x0: u32,
    /// Y offset, in f32 samples, of the sub-band rectangle inside the resident plane.
    pub y0: u32,
    /// Sub-band rectangle width in samples.
    pub width: u32,
    /// Sub-band rectangle height in samples.
    pub height: u32,
    /// Resident source row stride in f32 samples.
    pub stride: u32,
    /// Quantization parameters applied to every source sample.
    pub quantization: CudaJ2kQuantizeJob,
}

/// Quantized JPEG 2000 sub-band coefficients and execution metadata.
#[derive(Debug)]
pub struct CudaJ2kQuantizedSubband {
    coefficients: Vec<i32>,
    execution: CudaExecutionStats,
}

impl CudaJ2kQuantizedSubband {
    /// Quantized sub-band coefficients downloaded to host memory.
    pub fn coefficients(&self) -> &[i32] {
        &self.coefficients
    }

    /// CUDA execution counters for the quantization stage.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

/// Device-resident quantized JPEG 2000 sub-band coefficients and execution metadata.
#[derive(Debug)]
pub struct CudaJ2kResidentQuantizedSubband {
    coefficients: CudaDeviceBuffer,
    coefficient_count: usize,
    execution: CudaExecutionStats,
}

impl CudaJ2kResidentQuantizedSubband {
    /// Device buffer containing row-major `i32` coefficients.
    pub fn buffer(&self) -> &CudaDeviceBuffer {
        &self.coefficients
    }

    /// Number of `i32` coefficients in the resident buffer.
    pub fn coefficient_count(&self) -> usize {
        self.coefficient_count
    }

    /// Copy quantized coefficients to host memory.
    pub fn download_coefficients(&self) -> Result<Vec<i32>, CudaError> {
        let mut coefficients = vec![0i32; self.coefficient_count];
        self.coefficients
            .copy_to_host(i32_slice_as_bytes_mut(&mut coefficients))?;
        Ok(coefficients)
    }

    /// CUDA execution counters for the quantization stage.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

/// Shape metadata for one forward 5/3 DWT level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaDwt53LevelShape {
    /// Input level width.
    pub width: u32,
    /// Input level height.
    pub height: u32,
    /// Low-pass width.
    pub low_width: u32,
    /// Low-pass height.
    pub low_height: u32,
    /// High-pass width.
    pub high_width: u32,
    /// High-pass height.
    pub high_height: u32,
}

#[derive(Clone, Copy, Debug)]
struct CudaDwt53Pass {
    full_width: u32,
    current_width: u32,
    current_height: u32,
    low_extent: u32,
}

#[derive(Clone, Copy, Debug)]
struct CudaDwt53LevelPass {
    full_width: u32,
    current_width: u32,
    current_height: u32,
}

fn active_dwt53_buffers<'a>(
    buffer_a: &'a CudaDeviceBuffer,
    buffer_b: &'a CudaDeviceBuffer,
    active_is_a: bool,
) -> (&'a CudaDeviceBuffer, &'a CudaDeviceBuffer) {
    if active_is_a {
        (buffer_a, buffer_b)
    } else {
        (buffer_b, buffer_a)
    }
}

fn j2k_idwt_multi_kernel_jobs(
    targets: &[CudaJ2kIdwtTarget<'_>],
) -> Result<Vec<CudaJ2kIdwtMultiKernelJob>, CudaError> {
    let mut kernel_jobs = Vec::with_capacity(targets.len());
    for target in targets {
        let width = target.job.rect.x1.saturating_sub(target.job.rect.x0);
        let height = target.job.rect.y1.saturating_sub(target.job.rect.y0);
        if width == 0 || height == 0 {
            continue;
        }
        ensure_idwt_buffer_len(target.output, target.job.rect)?;
        ensure_idwt_buffer_len(target.ll, target.job.ll_rect)?;
        ensure_idwt_buffer_len(target.hl, target.job.hl_rect)?;
        ensure_idwt_buffer_len(target.lh, target.job.lh_rect)?;
        ensure_idwt_buffer_len(target.hh, target.job.hh_rect)?;
        kernel_jobs.push(CudaJ2kIdwtMultiKernelJob {
            ll_ptr: target.ll.device_ptr(),
            hl_ptr: target.hl.device_ptr(),
            lh_ptr: target.lh.device_ptr(),
            hh_ptr: target.hh.device_ptr(),
            output_ptr: target.output.device_ptr(),
            job: target.job,
        });
    }
    Ok(kernel_jobs)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CudaJ2kIdwtBatchKernelMode {
    Generic,
    Cooperative53,
    Cooperative97,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CudaJ2kIdwtBatchTraceRow {
    stage_index: usize,
    mode: CudaJ2kIdwtBatchKernelMode,
    job_count: usize,
    max_width: u32,
    max_height: u32,
    min_width: u32,
    min_height: u32,
    total_pixels: u64,
    irreversible_jobs: usize,
    elapsed_us: u128,
}

fn idwt_batch_kernel_mode(
    kernel_jobs: &[CudaJ2kIdwtMultiKernelJob],
    max_width: u32,
    max_height: u32,
) -> CudaJ2kIdwtBatchKernelMode {
    const MAX_COOPERATIVE_DIMENSION: u32 = 512;
    const MIN_COOPERATIVE_53_DIMENSION: u32 = 128;
    const MIN_COOPERATIVE_97_DIMENSION: u32 = 64;
    let bounded_cooperative_shape =
        max_width <= MAX_COOPERATIVE_DIMENSION && max_height <= MAX_COOPERATIVE_DIMENSION;
    if !bounded_cooperative_shape {
        return CudaJ2kIdwtBatchKernelMode::Generic;
    }
    if kernel_jobs.iter().all(|job| job.job.irreversible97 == 0) {
        if max_width >= MIN_COOPERATIVE_53_DIMENSION && max_height >= MIN_COOPERATIVE_53_DIMENSION {
            CudaJ2kIdwtBatchKernelMode::Cooperative53
        } else {
            CudaJ2kIdwtBatchKernelMode::Generic
        }
    } else if kernel_jobs.iter().all(|job| job.job.irreversible97 != 0) {
        if max_width >= MIN_COOPERATIVE_97_DIMENSION && max_height >= MIN_COOPERATIVE_97_DIMENSION {
            CudaJ2kIdwtBatchKernelMode::Cooperative97
        } else {
            CudaJ2kIdwtBatchKernelMode::Generic
        }
    } else {
        CudaJ2kIdwtBatchKernelMode::Generic
    }
}

fn cuda_idwt_trace_enabled() -> bool {
    std::env::var_os(CUDA_IDWT_TRACE_ENV_VAR).is_some()
}

fn idwt_batch_trace_row(
    stage_index: usize,
    kernel_jobs: &[CudaJ2kIdwtMultiKernelJob],
    max_width: u32,
    max_height: u32,
    mode: CudaJ2kIdwtBatchKernelMode,
    elapsed_us: u128,
) -> CudaJ2kIdwtBatchTraceRow {
    let mut min_width = u32::MAX;
    let mut min_height = u32::MAX;
    let mut total_pixels = 0u64;
    let mut irreversible_jobs = 0usize;
    for kernel_job in kernel_jobs {
        let width = kernel_job
            .job
            .rect
            .x1
            .saturating_sub(kernel_job.job.rect.x0);
        let height = kernel_job
            .job
            .rect
            .y1
            .saturating_sub(kernel_job.job.rect.y0);
        min_width = min_width.min(width);
        min_height = min_height.min(height);
        total_pixels =
            total_pixels.saturating_add(u64::from(width).saturating_mul(u64::from(height)));
        if kernel_job.job.irreversible97 != 0 {
            irreversible_jobs = irreversible_jobs.saturating_add(1);
        }
    }
    if kernel_jobs.is_empty() {
        min_width = 0;
        min_height = 0;
    }
    CudaJ2kIdwtBatchTraceRow {
        stage_index,
        mode,
        job_count: kernel_jobs.len(),
        max_width,
        max_height,
        min_width,
        min_height,
        total_pixels,
        irreversible_jobs,
        elapsed_us,
    }
}

fn format_idwt_batch_trace_row(row: CudaJ2kIdwtBatchTraceRow) -> String {
    format!(
        "signinum_profile codec=j2k op=cuda_idwt_batch path=decode \
         stage_index={} mode={:?} job_count={} max_width={} max_height={} \
         min_width={} min_height={} total_pixels={} irreversible_jobs={} elapsed_us={}",
        row.stage_index,
        row.mode,
        row.job_count,
        row.max_width,
        row.max_height,
        row.min_width,
        row.min_height,
        row.total_pixels,
        row.irreversible_jobs,
        row.elapsed_us
    )
}

#[cfg(test)]
fn idwt_batch_uses_cooperative_53(
    kernel_jobs: &[CudaJ2kIdwtMultiKernelJob],
    max_width: u32,
    max_height: u32,
) -> bool {
    idwt_batch_kernel_mode(kernel_jobs, max_width, max_height)
        == CudaJ2kIdwtBatchKernelMode::Cooperative53
}

fn ensure_idwt_buffer_len(buffer: &CudaDeviceBuffer, rect: CudaJ2kRect) -> Result<(), CudaError> {
    let width = rect.x1.saturating_sub(rect.x0);
    let height = rect.y1.saturating_sub(rect.y0);
    let words = checked_image_words(width, height, 1)?;
    let bytes = words
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(CudaError::LengthTooLarge { len: words })?;
    if bytes > buffer.byte_len() {
        return Err(CudaError::OutputTooSmall {
            required: bytes,
            have: buffer.byte_len(),
        });
    }
    Ok(())
}

fn htj2k_kernel_jobs(
    jobs: &[CudaHtj2kCodeBlockJob],
    payload_len: usize,
    output_words: usize,
) -> Result<Vec<CudaHtj2kCodeBlockKernelJob>, CudaError> {
    jobs.iter()
        .map(|job| {
            let payload_offset = usize::try_from(job.payload_offset)
                .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
            let payload_end = payload_offset
                .checked_add(job.payload_len as usize)
                .ok_or(CudaError::LengthTooLarge { len: payload_len })?;
            let expected_payload_len = job
                .cleanup_length
                .checked_add(job.refinement_length)
                .ok_or(CudaError::LengthTooLarge {
                    len: job.payload_len as usize,
                })?;
            let output_stride = job.output_stride as usize;
            let output_offset = job.output_offset as usize;
            let output_end = if job.height == 0 {
                output_offset
            } else {
                output_offset
                    .checked_add(
                        output_stride
                            .checked_mul(job.height as usize - 1)
                            .ok_or(CudaError::LengthTooLarge { len: output_words })?,
                    )
                    .and_then(|last_row| last_row.checked_add(job.width as usize))
                    .ok_or(CudaError::LengthTooLarge { len: output_words })?
            };
            if payload_end > payload_len
                || expected_payload_len != job.payload_len
                || output_end > output_words
            {
                return Err(CudaError::LengthTooLarge {
                    len: payload_len.max(output_words),
                });
            }
            Ok(CudaHtj2kCodeBlockKernelJob {
                coded_offset: u32::try_from(payload_offset)
                    .map_err(|_| CudaError::LengthTooLarge { len: payload_len })?,
                width: job.width,
                height: job.height,
                coded_len: job.payload_len,
                cleanup_length: job.cleanup_length,
                refinement_length: job.refinement_length,
                missing_msbs: u32::from(job.missing_bit_planes),
                num_bitplanes: u32::from(job.num_bitplanes),
                number_of_coding_passes: u32::from(job.number_of_coding_passes),
                output_stride: job.output_stride,
                output_offset: job.output_offset,
                dequantization_step: job.dequantization_step,
                stripe_causal: u32::from(job.stripe_causal),
            })
        })
        .collect()
}

fn htj2k_dequantize_kernel_jobs(
    targets: &[CudaHtj2kDequantizeTarget<'_>],
) -> Result<Vec<CudaHtj2kDequantizeKernelJob>, CudaError> {
    let total_jobs = targets
        .iter()
        .try_fold(0usize, |count, target| count.checked_add(target.jobs.len()))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    let mut kernel_jobs = Vec::with_capacity(total_jobs);
    for target in targets {
        let output_bytes = target
            .output_words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaError::LengthTooLarge {
                len: target.output_words,
            })?;
        if output_bytes > target.coefficients.byte_len() {
            return Err(CudaError::LengthTooLarge { len: output_bytes });
        }
        for job in target.jobs {
            let output_stride = job.output_stride as usize;
            let output_offset = job.output_offset as usize;
            let output_end = if job.height == 0 {
                output_offset
            } else {
                output_offset
                    .checked_add(output_stride.checked_mul(job.height as usize - 1).ok_or(
                        CudaError::LengthTooLarge {
                            len: target.output_words,
                        },
                    )?)
                    .and_then(|last_row| last_row.checked_add(job.width as usize))
                    .ok_or(CudaError::LengthTooLarge {
                        len: target.output_words,
                    })?
            };
            if output_end > target.output_words {
                return Err(CudaError::LengthTooLarge {
                    len: target.output_words,
                });
            }
            kernel_jobs.push(CudaHtj2kDequantizeKernelJob {
                output_ptr: target.coefficients.device_ptr(),
                width: job.width,
                height: job.height,
                output_stride: job.output_stride,
                output_offset: job.output_offset,
                num_bitplanes: u32::from(job.num_bitplanes),
                reserved: 0,
                dequantization_step: job.dequantization_step,
            });
        }
    }
    Ok(kernel_jobs)
}

fn htj2k_cleanup_multi_kernel_jobs(
    targets: &[CudaHtj2kCleanupTarget<'_>],
    payload_len: usize,
) -> Result<Vec<CudaHtj2kCleanupMultiKernelJob>, CudaError> {
    let total_jobs = targets
        .iter()
        .try_fold(0usize, |count, target| count.checked_add(target.jobs.len()))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    let mut kernel_jobs = Vec::with_capacity(total_jobs);
    for target in targets {
        let output_bytes = target
            .output_words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaError::LengthTooLarge {
                len: target.output_words,
            })?;
        if output_bytes > target.coefficients.byte_len() {
            return Err(CudaError::LengthTooLarge { len: output_bytes });
        }
        for job in htj2k_kernel_jobs(target.jobs, payload_len, target.output_words)? {
            kernel_jobs.push(CudaHtj2kCleanupMultiKernelJob {
                output_ptr: target.coefficients.device_ptr(),
                coded_offset: job.coded_offset,
                width: job.width,
                height: job.height,
                coded_len: job.coded_len,
                cleanup_length: job.cleanup_length,
                refinement_length: job.refinement_length,
                missing_msbs: job.missing_msbs,
                num_bitplanes: job.num_bitplanes,
                number_of_coding_passes: job.number_of_coding_passes,
                output_stride: job.output_stride,
                output_offset: job.output_offset,
                dequantization_step: job.dequantization_step,
                stripe_causal: job.stripe_causal,
            });
        }
    }
    Ok(kernel_jobs)
}

fn htj2k_decode_multi_kernel_for_jobs(
    jobs: &[CudaHtj2kCleanupMultiKernelJob],
) -> (CudaKernel, &'static str) {
    let cleanup_only = jobs
        .iter()
        .all(|job| job.refinement_length == 0 && job.number_of_coding_passes <= 1);
    if cleanup_only {
        (
            CudaKernel::Htj2kDecodeCodeblocksMultiCleanupOnly,
            "signinum_htj2k_decode_codeblocks_multi_cleanup_only",
        )
    } else {
        (
            CudaKernel::Htj2kDecodeCodeblocksMulti,
            "signinum_htj2k_decode_codeblocks_multi",
        )
    }
}

fn htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(
    jobs: &[CudaHtj2kCleanupMultiKernelJob],
) -> Option<(CudaKernel, &'static str)> {
    let cleanup_only = jobs
        .iter()
        .all(|job| job.refinement_length == 0 && job.number_of_coding_passes <= 1);
    cleanup_only.then_some((
        CudaKernel::Htj2kDecodeCodeblocksMultiCleanupDequantize,
        "signinum_htj2k_decode_codeblocks_multi_cleanup_dequantize",
    ))
}

fn htj2k_decode_needs_zero_fill(
    jobs: &[CudaHtj2kCodeBlockJob],
    output_words: usize,
) -> Result<bool, CudaError> {
    let mut covered_words = 0usize;
    for job in jobs {
        let area = (job.width as usize)
            .checked_mul(job.height as usize)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        covered_words = covered_words
            .checked_add(area)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    }
    if covered_words > output_words {
        return Err(CudaError::LengthTooLarge { len: covered_words });
    }
    Ok(covered_words != output_words)
}

fn htj2k_encode_kernel_jobs(
    jobs: &[CudaHtj2kEncodeCodeBlockJob],
    coefficient_words: usize,
) -> Result<Vec<CudaHtj2kEncodeKernelJob>, CudaError> {
    let mut output_offset = 0usize;
    let mut kernel_jobs = Vec::with_capacity(jobs.len());
    for job in jobs {
        validate_htj2k_encode_codeblock_shape(job.width, job.height)?;
        let coefficient_offset = job.coefficient_offset as usize;
        let coefficient_len = checked_image_words(job.width, job.height, 1)?;
        let coefficient_end =
            coefficient_offset
                .checked_add(coefficient_len)
                .ok_or(CudaError::LengthTooLarge {
                    len: coefficient_words,
                })?;
        if coefficient_end > coefficient_words {
            return Err(CudaError::LengthTooLarge {
                len: coefficient_end,
            });
        }

        let output_end = output_offset
            .checked_add(HTJ2K_ENCODE_OUTPUT_CAPACITY)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if output_end > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge { len: output_end });
        }
        kernel_jobs.push(CudaHtj2kEncodeKernelJob {
            coefficient_offset: job.coefficient_offset,
            coefficient_stride: job.width,
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_offset: u32::try_from(output_offset)
                .map_err(|_| CudaError::LengthTooLarge { len: output_offset })?,
            output_capacity: u32::try_from(HTJ2K_ENCODE_OUTPUT_CAPACITY).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: HTJ2K_ENCODE_OUTPUT_CAPACITY,
                }
            })?,
            target_coding_passes: u32::from(job.target_coding_passes),
        });
        output_offset = output_end;
    }
    Ok(kernel_jobs)
}

fn htj2k_encode_multi_input_kernel_jobs(
    targets: &[CudaHtj2kEncodeResidentTarget<'_>],
) -> Result<Vec<CudaHtj2kEncodeMultiInputKernelJob>, CudaError> {
    let job_count = targets
        .iter()
        .try_fold(0usize, |sum, target| sum.checked_add(target.jobs.len()))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    let mut output_offset = 0usize;
    let mut kernel_jobs = Vec::with_capacity(job_count);
    for target in targets {
        let available_coefficients = target.coefficients.typed_view::<i32>()?.len();
        if available_coefficients < target.coefficient_count {
            return Err(CudaError::OutputTooSmall {
                required: target
                    .coefficient_count
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or(CudaError::LengthTooLarge {
                        len: target.coefficient_count,
                    })?,
                have: target.coefficients.byte_len(),
            });
        }
        for job in target.jobs {
            validate_htj2k_encode_codeblock_shape(job.width, job.height)?;
            let coefficient_offset = job.coefficient_offset as usize;
            let coefficient_len = checked_image_words(job.width, job.height, 1)?;
            let coefficient_end = coefficient_offset.checked_add(coefficient_len).ok_or(
                CudaError::LengthTooLarge {
                    len: target.coefficient_count,
                },
            )?;
            if coefficient_end > target.coefficient_count {
                return Err(CudaError::LengthTooLarge {
                    len: coefficient_end,
                });
            }

            let output_end = output_offset
                .checked_add(HTJ2K_ENCODE_OUTPUT_CAPACITY)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if output_end > u32::MAX as usize {
                return Err(CudaError::LengthTooLarge { len: output_end });
            }
            kernel_jobs.push(CudaHtj2kEncodeMultiInputKernelJob {
                coefficient_ptr: target.coefficients.device_ptr(),
                coefficient_offset: job.coefficient_offset,
                coefficient_stride: job.width,
                width: job.width,
                height: job.height,
                total_bitplanes: u32::from(job.total_bitplanes),
                output_offset: u32::try_from(output_offset)
                    .map_err(|_| CudaError::LengthTooLarge { len: output_offset })?,
                output_capacity: u32::try_from(HTJ2K_ENCODE_OUTPUT_CAPACITY).map_err(|_| {
                    CudaError::LengthTooLarge {
                        len: HTJ2K_ENCODE_OUTPUT_CAPACITY,
                    }
                })?,
                target_coding_passes: u32::from(job.target_coding_passes),
            });
            output_offset = output_end;
        }
    }
    Ok(kernel_jobs)
}

fn htj2k_encode_region_kernel_jobs(
    jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
    coefficient_words: usize,
) -> Result<Vec<CudaHtj2kEncodeKernelJob>, CudaError> {
    let mut output_offset = 0usize;
    let mut kernel_jobs = Vec::with_capacity(jobs.len());
    for job in jobs {
        validate_htj2k_encode_codeblock_shape(job.width, job.height)?;
        if job.width == 0 || job.height == 0 || job.coefficient_stride < job.width {
            return Err(CudaError::LengthTooLarge {
                len: coefficient_words,
            });
        }
        let row_offset = (job.height as usize - 1)
            .checked_mul(job.coefficient_stride as usize)
            .ok_or(CudaError::LengthTooLarge {
                len: coefficient_words,
            })?;
        let coefficient_end = job
            .coefficient_offset
            .try_into()
            .ok()
            .and_then(|offset: usize| offset.checked_add(row_offset))
            .and_then(|offset| offset.checked_add(job.width as usize))
            .ok_or(CudaError::LengthTooLarge {
                len: coefficient_words,
            })?;
        if coefficient_end > coefficient_words {
            return Err(CudaError::LengthTooLarge {
                len: coefficient_end,
            });
        }

        let output_end = output_offset
            .checked_add(HTJ2K_ENCODE_OUTPUT_CAPACITY)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if output_end > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge { len: output_end });
        }
        kernel_jobs.push(CudaHtj2kEncodeKernelJob {
            coefficient_offset: job.coefficient_offset,
            coefficient_stride: job.coefficient_stride,
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_offset: u32::try_from(output_offset)
                .map_err(|_| CudaError::LengthTooLarge { len: output_offset })?,
            output_capacity: u32::try_from(HTJ2K_ENCODE_OUTPUT_CAPACITY).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: HTJ2K_ENCODE_OUTPUT_CAPACITY,
                }
            })?,
            target_coding_passes: u32::from(job.target_coding_passes),
        });
        output_offset = output_end;
    }
    Ok(kernel_jobs)
}

fn htj2k_encode_compact_jobs(
    statuses: &[CudaHtj2kEncodeStatus],
    kernel_jobs: &[CudaHtj2kEncodeKernelJob],
) -> Result<(Vec<CudaHtj2kEncodeCompactJob>, usize), CudaError> {
    if statuses.len() != kernel_jobs.len() {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K encode status count does not match job count".to_string(),
        });
    }

    let mut compact_offset = 0usize;
    let mut compact_jobs = Vec::with_capacity(kernel_jobs.len());
    for (status, job) in statuses.iter().zip(kernel_jobs) {
        let data_len = usize::try_from(status.data_len)
            .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
        if data_len > job.output_capacity as usize {
            return Err(CudaError::LengthTooLarge { len: data_len });
        }
        let source_end = (job.output_offset as usize)
            .checked_add(data_len)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let job_output_end = (job.output_offset as usize)
            .checked_add(job.output_capacity as usize)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if source_end > job_output_end {
            return Err(CudaError::LengthTooLarge { len: source_end });
        }
        compact_jobs.push(CudaHtj2kEncodeCompactJob {
            source_offset: job.output_offset,
            compact_offset: u32::try_from(compact_offset).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: compact_offset,
                }
            })?,
            data_len: status.data_len,
            reserved: status.reserved2,
        });
        compact_offset = compact_offset
            .checked_add(data_len)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if compact_offset > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge {
                len: compact_offset,
            });
        }
    }

    Ok((compact_jobs, compact_offset))
}

fn htj2k_encode_compact_jobs_multi_input(
    statuses: &[CudaHtj2kEncodeStatus],
    kernel_jobs: &[CudaHtj2kEncodeMultiInputKernelJob],
) -> Result<(Vec<CudaHtj2kEncodeCompactJob>, usize), CudaError> {
    if statuses.len() != kernel_jobs.len() {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K encode status count does not match job count".to_string(),
        });
    }

    let mut compact_offset = 0usize;
    let mut compact_jobs = Vec::with_capacity(kernel_jobs.len());
    for (status, job) in statuses.iter().zip(kernel_jobs) {
        let data_len = usize::try_from(status.data_len)
            .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
        if data_len > job.output_capacity as usize {
            return Err(CudaError::LengthTooLarge { len: data_len });
        }
        let source_end = (job.output_offset as usize)
            .checked_add(data_len)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let job_output_end = (job.output_offset as usize)
            .checked_add(job.output_capacity as usize)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if source_end > job_output_end {
            return Err(CudaError::LengthTooLarge { len: source_end });
        }
        compact_jobs.push(CudaHtj2kEncodeCompactJob {
            source_offset: job.output_offset,
            compact_offset: u32::try_from(compact_offset).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: compact_offset,
                }
            })?,
            data_len: status.data_len,
            reserved: status.reserved2,
        });
        compact_offset = compact_offset
            .checked_add(data_len)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if compact_offset > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge {
                len: compact_offset,
            });
        }
    }

    Ok((compact_jobs, compact_offset))
}

fn validate_htj2k_encode_codeblock_shape(width: u32, height: u32) -> Result<(), CudaError> {
    let samples = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    if width == 0
        || height == 0
        || width > HTJ2K_ENCODE_MAX_CODEBLOCK_WIDTH
        || samples > HTJ2K_ENCODE_MAX_CODEBLOCK_SAMPLES
    {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K encode code-block dimensions exceed CUDA kernel limits".to_string(),
        });
    }
    Ok(())
}

impl CudaKernelOutput {
    /// Device buffer produced by the kernel.
    pub fn buffer(&self) -> &CudaDeviceBuffer {
        &self.buffer
    }

    /// CUDA execution counters for the kernel.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Split output into device buffer and execution metadata.
    pub fn into_parts(self) -> (CudaDeviceBuffer, CudaExecutionStats) {
        (self.buffer, self.execution)
    }
}

impl CudaKernelBatchOutput {
    /// Device buffers produced by the batched kernel.
    pub fn outputs(&self) -> &[CudaDeviceBuffer] {
        &self.outputs
    }

    /// CUDA execution counters for the batched kernel.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Split output into device buffers and execution metadata.
    pub fn into_parts(self) -> (Vec<CudaDeviceBuffer>, CudaExecutionStats) {
        (self.outputs, self.execution)
    }
}

impl CudaKernelContiguousBatchOutput {
    /// Contiguous device buffer produced by the batched kernel.
    pub fn output(&self) -> &CudaDeviceBuffer {
        &self.output
    }

    /// Per-item byte ranges inside the contiguous output buffer.
    pub fn ranges(&self) -> &[CudaDeviceBufferRange] {
        &self.ranges
    }

    /// CUDA execution counters for the batched kernel.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Split output into the contiguous buffer, per-item ranges, and execution metadata.
    pub fn into_parts(
        self,
    ) -> (
        CudaDeviceBuffer,
        Vec<CudaDeviceBufferRange>,
        CudaExecutionStats,
    ) {
        (self.output, self.ranges, self.execution)
    }
}

impl CudaPooledKernelOutput {
    /// Device buffer produced by the kernel.
    pub fn buffer(&self) -> Option<&CudaDeviceBuffer> {
        self.buffer.as_device_buffer()
    }

    /// CUDA execution counters for the kernel.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Split output into pooled device buffer and execution metadata.
    pub fn into_parts(self) -> (CudaPooledDeviceBuffer, CudaExecutionStats) {
        (self.buffer, self.execution)
    }
}

/// CUDA execution counters exposed for dispatch observability.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaExecutionStats {
    kernel_dispatches: usize,
    copy_kernel_dispatches: usize,
    decode_kernel_dispatches: usize,
    hardware_decode: bool,
}

impl CudaExecutionStats {
    /// Total kernel dispatch count.
    pub fn kernel_dispatches(self) -> usize {
        self.kernel_dispatches
    }

    /// Copy-kernel dispatch count.
    pub fn copy_kernel_dispatches(self) -> usize {
        self.copy_kernel_dispatches
    }

    /// Hardware decode dispatch count.
    pub fn decode_kernel_dispatches(self) -> usize {
        self.decode_kernel_dispatches
    }

    /// True when a hardware decode path was used.
    pub fn used_hardware_decode(self) -> bool {
        self.hardware_decode
    }
}

/// Reversible 5/3 transcode bands downloaded from the device. Layout matches
/// `signinum_transcode::accelerator::ReversibleDwt53FirstLevel`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaTranscodeReversible53Bands {
    /// Low-horizontal, low-vertical band (`low_width * low_height`).
    pub ll: Vec<i32>,
    /// High-horizontal, low-vertical band (`high_width * low_height`).
    pub hl: Vec<i32>,
    /// Low-horizontal, high-vertical band (`low_width * high_height`).
    pub lh: Vec<i32>,
    /// High-horizontal, high-vertical band (`high_width * high_height`).
    pub hh: Vec<i32>,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

#[derive(Clone, Copy)]
struct Reversible53Dims {
    block_cols: i32,
    width: i32,
    height: i32,
    low_width: i32,
    high_width: i32,
}

impl CudaContext {
    /// Compute one reversible integer 5/3 level directly from dequantized 8x8
    /// DCT blocks, bit-exact with the `signinum-transcode` scalar oracle.
    ///
    /// `dequantized_blocks` holds `block_cols * block_rows` natural-order blocks
    /// of 64 `i16` coefficients. `width`/`height` are the logical component
    /// dimensions (<= `block_cols*8` / `block_rows*8`).
    #[allow(clippy::too_many_lines)]
    pub fn j2k_transcode_reversible_dwt53(
        &self,
        dequantized_blocks: &[i16],
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
    ) -> Result<CudaTranscodeReversible53Bands, CudaError> {
        if !TRANSCODE_PTX_BUILT_FROM_CUDA {
            return Err(CudaError::InvalidArgument {
                message: "CUDA transcode kernels were not built (nvcc unavailable at build time)"
                    .to_string(),
            });
        }
        let block_count = block_cols
            .checked_mul(block_rows)
            .ok_or(CudaError::LengthTooLarge { len: block_cols })?;
        let covered_w = block_cols
            .checked_mul(8)
            .ok_or(CudaError::LengthTooLarge { len: block_cols })?;
        let covered_h = block_rows
            .checked_mul(8)
            .ok_or(CudaError::LengthTooLarge { len: block_rows })?;
        let expected_coeffs = block_count
            .checked_mul(64)
            .ok_or(CudaError::LengthTooLarge { len: block_count })?;
        if width == 0
            || height == 0
            || width > covered_w
            || height > covered_h
            || dequantized_blocks.len() != expected_coeffs
        {
            return Err(CudaError::InvalidArgument {
                message: "reversible 5/3 transcode job has unsupported grid geometry".to_string(),
            });
        }

        let low_width = width.div_ceil(2);
        let low_height = height.div_ceil(2);
        let high_width = width / 2;
        let high_height = height / 2;

        let to_i32 = |value: usize| -> Result<i32, CudaError> {
            i32::try_from(value).map_err(|_| CudaError::LengthTooLarge { len: value })
        };
        let dims = Reversible53Dims {
            block_cols: to_i32(block_cols)?,
            width: to_i32(width)?,
            height: to_i32(height)?,
            low_width: to_i32(low_width)?,
            high_width: to_i32(high_width)?,
        };

        self.inner.set_current()?;

        let alloc_i32 = |count: usize| -> Result<CudaDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            self.allocate(bytes)
        };
        let samples = alloc_i32(expected_coeffs)?;
        let v_low = alloc_i32(width * low_height)?;
        let v_high = alloc_i32(width * high_height)?;
        let ll = alloc_i32(low_width * low_height)?;
        let hl = alloc_i32(high_width * low_height)?;
        let lh = alloc_i32(low_width * high_height)?;
        let hh = alloc_i32(high_width * high_height)?;

        // SAFETY: `dequantized_blocks` is a live `&[i16]`; reinterpreting it as a
        // byte slice of `len * 2` bytes for upload is a read-only view with the
        // same lifetime and no alignment requirement on the destination.
        let block_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                dequantized_blocks.as_ptr().cast::<u8>(),
                std::mem::size_of_val(dequantized_blocks),
            )
        };
        let blocks_dev = self.upload(block_bytes)?;

        self.launch_transcode_reversible53_idct(&blocks_dev, &samples, block_count)?;
        if low_height > 0 {
            self.launch_transcode_reversible53_vertical(
                CudaKernel::TranscodeReversible53VerticalLow,
                &samples,
                dims,
                &v_low,
                to_i32(low_height)?,
            )?;
            self.launch_transcode_reversible53_horizontal(
                CudaKernel::TranscodeReversible53HorizontalLow,
                &v_low,
                dims,
                to_i32(low_height)?,
                &ll,
                &hl,
            )?;
        }
        if high_height > 0 {
            self.launch_transcode_reversible53_vertical(
                CudaKernel::TranscodeReversible53VerticalHigh,
                &samples,
                dims,
                &v_high,
                to_i32(high_height)?,
            )?;
            self.launch_transcode_reversible53_horizontal(
                CudaKernel::TranscodeReversible53HorizontalHigh,
                &v_high,
                dims,
                to_i32(high_height)?,
                &lh,
                &hh,
            )?;
        }

        Ok(CudaTranscodeReversible53Bands {
            ll: Self::download_i32_band(&ll, low_width * low_height)?,
            hl: Self::download_i32_band(&hl, high_width * low_height)?,
            lh: Self::download_i32_band(&lh, low_width * high_height)?,
            hh: Self::download_i32_band(&hh, high_width * high_height)?,
            low_width,
            low_height,
            high_width,
            high_height,
        })
    }

    fn download_i32_band(buffer: &CudaDeviceBuffer, count: usize) -> Result<Vec<i32>, CudaError> {
        let mut out = vec![0i32; count];
        if count != 0 {
            buffer.copy_to_host(i32_slice_as_bytes_mut(&mut out))?;
        }
        Ok(out)
    }

    fn launch_transcode_reversible53_idct(
        &self,
        blocks: &CudaDeviceBuffer,
        samples: &CudaDeviceBuffer,
        block_count: usize,
    ) -> Result<(), CudaError> {
        if block_count == 0 {
            return Ok(());
        }
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeReversible53Idct)?;
        let mut blocks_ptr = blocks.device_ptr();
        let mut samples_ptr = samples.device_ptr();
        let mut count = u32::try_from(block_count)
            .map_err(|_| CudaError::LengthTooLarge { len: block_count })?;
        let mut params = [
            (&raw mut blocks_ptr).cast::<c_void>(),
            (&raw mut samples_ptr).cast::<c_void>(),
            (&raw mut count).cast::<c_void>(),
        ];
        let geometry = copy_u8_launch_geometry(block_count)
            .ok_or(CudaError::LengthTooLarge { len: block_count })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_transcode_reversible53_vertical(
        &self,
        kernel: CudaKernel,
        samples: &CudaDeviceBuffer,
        dims: Reversible53Dims,
        out: &CudaDeviceBuffer,
        out_rows: i32,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(kernel)?;
        let mut samples_ptr = samples.device_ptr();
        let mut block_cols = dims.block_cols;
        let mut width = dims.width;
        let mut height = dims.height;
        let mut out_ptr = out.device_ptr();
        let mut rows = out_rows;
        let mut params = [
            (&raw mut samples_ptr).cast::<c_void>(),
            (&raw mut block_cols).cast::<c_void>(),
            (&raw mut width).cast::<c_void>(),
            (&raw mut height).cast::<c_void>(),
            (&raw mut out_ptr).cast::<c_void>(),
            (&raw mut rows).cast::<c_void>(),
        ];
        let grid_w = u32::try_from(dims.width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let grid_h = u32::try_from(out_rows).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let geometry = j2k_dwt53_launch_geometry(grid_w, grid_h)
            .ok_or(CudaError::LengthTooLarge { len: 0 })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_transcode_reversible53_horizontal(
        &self,
        kernel: CudaKernel,
        rows_buffer: &CudaDeviceBuffer,
        dims: Reversible53Dims,
        n_rows: i32,
        low_out: &CudaDeviceBuffer,
        high_out: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let row_count =
            usize::try_from(n_rows).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        if row_count == 0 {
            return Ok(());
        }
        let function = self.inner.kernel_function(kernel)?;
        let mut rows_ptr = rows_buffer.device_ptr();
        let mut width = dims.width;
        let mut rows = n_rows;
        let mut low_width = dims.low_width;
        let mut high_width = dims.high_width;
        let mut low_ptr = low_out.device_ptr();
        let mut high_ptr = high_out.device_ptr();
        let mut params = [
            (&raw mut rows_ptr).cast::<c_void>(),
            (&raw mut width).cast::<c_void>(),
            (&raw mut rows).cast::<c_void>(),
            (&raw mut low_width).cast::<c_void>(),
            (&raw mut high_width).cast::<c_void>(),
            (&raw mut low_ptr).cast::<c_void>(),
            (&raw mut high_ptr).cast::<c_void>(),
        ];
        let geometry = copy_u8_launch_geometry(row_count)
            .ok_or(CudaError::LengthTooLarge { len: row_count })?;
        self.launch_kernel(function, geometry, &mut params)
    }
}

/// Irreversible single-level 9/7 transcode bands downloaded from the device.
/// Device math is f32; callers widen to f64 (parity is within tolerance).
#[derive(Clone, Debug, PartialEq)]
pub struct CudaTranscodeDwt97Bands {
    /// Low-horizontal, low-vertical band (`low_width * low_height`).
    pub ll: Vec<f32>,
    /// High-horizontal, low-vertical band (`high_width * low_height`).
    pub hl: Vec<f32>,
    /// Low-horizontal, high-vertical band (`low_width * high_height`).
    pub lh: Vec<f32>,
    /// High-horizontal, high-vertical band (`high_width * high_height`).
    pub hh: Vec<f32>,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

/// Backend stage timings for a same-geometry 9/7 (or fused code-block) batch.
///
/// Mirrors `signinum-transcode`'s `Dwt97BatchStageTimings`; kept local because
/// `signinum-cuda-runtime` does not depend on `signinum-transcode`. The dispatch
/// layer maps this onto the transcode type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaDwt97BatchStageTimings {
    /// Buffer allocation plus host-to-device block upload time, microseconds.
    pub pack_upload_us: u128,
    /// IDCT plus horizontal 9/7 row-lift stage time, microseconds.
    pub idct_row_lift_us: u128,
    /// Vertical 9/7 column-lift stage time, microseconds.
    pub column_lift_us: u128,
    /// Code-block quantization stage time, microseconds (0 for the band path).
    pub quantize_codeblock_us: u128,
    /// Resident HT code-block encode time, microseconds.
    pub ht_encode_us: u128,
    /// Resident HT code-block encode dispatches.
    pub ht_codeblock_dispatches: usize,
    /// Device-to-host readback and unpack time, microseconds.
    pub readback_us: u128,
}

/// Per-subband inverse step sizes and code-block geometry for the fused 9/7
/// code-block quantization batch. The dispatch layer derives the deltas from
/// the `signinum-transcode` code-block oracle so the numbers stay authoritative.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2k97QuantizeParams {
    /// `1/Δ` for the LL subband.
    pub inv_delta_ll: f32,
    /// `1/Δ` for the HL subband.
    pub inv_delta_hl: f32,
    /// `1/Δ` for the LH subband.
    pub inv_delta_lh: f32,
    /// `1/Δ` for the HH subband.
    pub inv_delta_hh: f32,
    /// Code-block width in coefficients (`1 << (code_block_width_exp + 2)`).
    pub cb_width: usize,
    /// Code-block height in coefficients (`1 << (code_block_height_exp + 2)`).
    pub cb_height: usize,
}

/// Per-item raw code-block-major quantized 9/7 bands from the fused batch.
///
/// Each band concatenates `item_count` per-item subband buffers in code-block
/// -major order (outer code-block row, inner code-block column, each block
/// row-major), matching the `signinum-transcode` code-block oracle layout. The
/// dispatch layer reslices these into prequantized HTJ2K components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaHtj2k97CodeblockBands {
    /// LL subband (`item_count * low_width * low_height`).
    pub ll: Vec<i32>,
    /// HL subband (`item_count * high_width * low_height`).
    pub hl: Vec<i32>,
    /// LH subband (`item_count * low_width * high_height`).
    pub lh: Vec<i32>,
    /// HH subband (`item_count * high_width * high_height`).
    pub hh: Vec<i32>,
    /// Number of items in the batch.
    pub item_count: usize,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

/// Device-resident per-item raw code-block-major quantized 9/7 bands from the
/// fused transcode batch.
#[derive(Debug)]
pub struct CudaHtj2k97DeviceCodeblockBands {
    /// LL subband (`item_count * low_width * low_height`).
    pub ll: CudaPooledDeviceBuffer,
    /// HL subband (`item_count * high_width * low_height`).
    pub hl: CudaPooledDeviceBuffer,
    /// LH subband (`item_count * low_width * high_height`).
    pub lh: CudaPooledDeviceBuffer,
    /// HH subband (`item_count * high_width * high_height`).
    pub hh: CudaPooledDeviceBuffer,
    /// Number of items in the batch.
    pub item_count: usize,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

/// Device-resident 9/7 batch bands produced by the shared staged pipeline.
struct Dwt97BatchDeviceBands {
    ll: CudaPooledDeviceBuffer,
    lh: CudaPooledDeviceBuffer,
    hl: CudaPooledDeviceBuffer,
    hh: CudaPooledDeviceBuffer,
    low_width: usize,
    low_height: usize,
    high_width: usize,
    high_height: usize,
}

#[derive(Clone, Copy)]
enum Dwt97BatchInput<'a> {
    F32(&'a [f32]),
    I16(&'a [i16]),
}

impl Dwt97BatchInput<'_> {
    fn len(self) -> usize {
        match self {
            Self::F32(blocks) => blocks.len(),
            Self::I16(blocks) => blocks.len(),
        }
    }

    fn upload(self, pool: &CudaBufferPool) -> Result<CudaPooledDeviceBuffer, CudaError> {
        match self {
            Self::F32(blocks) => pool.upload_f32(blocks),
            Self::I16(blocks) => {
                let bytes = i16_slice_as_bytes(blocks);
                if should_use_pinned_pooled_i16_upload(bytes.len()) {
                    pool.upload_pinned(bytes)
                } else {
                    pool.upload(bytes)
                }
            }
        }
    }
}

fn should_use_pinned_pooled_i16_upload(byte_len: usize) -> bool {
    byte_len <= PINNED_POOLED_I16_UPLOAD_MAX_BYTES
}

fn pooled_device_buffer(buffer: &CudaPooledDeviceBuffer) -> Result<&CudaDeviceBuffer, CudaError> {
    buffer
        .as_device_buffer()
        .ok_or_else(|| CudaError::InvalidArgument {
            message: "pooled CUDA buffer checkout is empty".to_string(),
        })
}

fn copy_pooled_bytes_to_vec_uninit(
    buffer: &CudaPooledDeviceBuffer,
    byte_len: usize,
) -> Result<Vec<u8>, CudaError> {
    let mut out = Vec::with_capacity(byte_len);
    pooled_device_buffer(buffer)?.copy_range_to_host_uninit(0, out.spare_capacity_mut())?;
    // SAFETY: copy_range_to_host_uninit returned success after writing exactly
    // byte_len initialized bytes into the Vec spare capacity.
    unsafe {
        out.set_len(byte_len);
    }
    Ok(out)
}

impl CudaContext {
    /// Compute one irreversible single-level 9/7 transform directly from
    /// dequantized 8x8 DCT blocks (`block_cols * block_rows` blocks of 64 `f32`
    /// natural-order coefficients), matching the `signinum-transcode` scalar
    /// oracle within f32 tolerance.
    #[allow(clippy::too_many_lines)]
    pub fn j2k_transcode_dwt97(
        &self,
        blocks: &[f32],
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
    ) -> Result<CudaTranscodeDwt97Bands, CudaError> {
        if !TRANSCODE_PTX_BUILT_FROM_CUDA {
            return Err(CudaError::InvalidArgument {
                message: "CUDA transcode kernels were not built (nvcc unavailable at build time)"
                    .to_string(),
            });
        }
        let block_count = block_cols
            .checked_mul(block_rows)
            .ok_or(CudaError::LengthTooLarge { len: block_cols })?;
        let covered_w = block_cols
            .checked_mul(8)
            .ok_or(CudaError::LengthTooLarge { len: block_cols })?;
        let covered_h = block_rows
            .checked_mul(8)
            .ok_or(CudaError::LengthTooLarge { len: block_rows })?;
        let expected_coeffs = block_count
            .checked_mul(64)
            .ok_or(CudaError::LengthTooLarge { len: block_count })?;
        if width == 0
            || height == 0
            || width > covered_w
            || height > covered_h
            || blocks.len() != expected_coeffs
        {
            return Err(CudaError::InvalidArgument {
                message: "9/7 transcode job has unsupported grid geometry".to_string(),
            });
        }

        let low_width = width.div_ceil(2);
        let low_height = height.div_ceil(2);
        let high_width = width / 2;
        let high_height = height / 2;

        let to_i32 = |value: usize| -> Result<i32, CudaError> {
            i32::try_from(value).map_err(|_| CudaError::LengthTooLarge { len: value })
        };
        let dims = Reversible53Dims {
            block_cols: to_i32(block_cols)?,
            width: to_i32(width)?,
            height: to_i32(height)?,
            low_width: to_i32(low_width)?,
            high_width: to_i32(high_width)?,
        };

        self.inner.set_current()?;

        let alloc_f32 = |count: usize| -> Result<CudaDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            self.allocate(bytes)
        };
        let spatial = alloc_f32(width * height)?;
        let row_low = alloc_f32(height * low_width)?;
        let row_high = alloc_f32(height * high_width)?;
        let ll = alloc_f32(low_width * low_height)?;
        let lh = alloc_f32(low_width * high_height)?;
        let hl = alloc_f32(high_width * low_height)?;
        let hh = alloc_f32(high_width * high_height)?;

        let blocks_dev = self.upload_f32(blocks)?;

        self.launch_transcode_dwt97_idct(dims, &blocks_dev, &spatial)?;
        self.launch_transcode_dwt97_row_lift(dims, &spatial, &row_low, &row_high)?;
        if dims.low_width > 0 {
            self.launch_transcode_dwt97_column_lift(
                &row_low,
                dims.low_width,
                dims.height,
                &ll,
                &lh,
            )?;
        }
        if dims.high_width > 0 {
            self.launch_transcode_dwt97_column_lift(
                &row_high,
                dims.high_width,
                dims.height,
                &hl,
                &hh,
            )?;
        }

        Ok(CudaTranscodeDwt97Bands {
            ll: Self::download_f32_band(&ll, low_width * low_height)?,
            hl: Self::download_f32_band(&hl, high_width * low_height)?,
            lh: Self::download_f32_band(&lh, low_width * high_height)?,
            hh: Self::download_f32_band(&hh, high_width * high_height)?,
            low_width,
            low_height,
            high_width,
            high_height,
        })
    }

    fn download_f32_band(buffer: &CudaDeviceBuffer, count: usize) -> Result<Vec<f32>, CudaError> {
        let mut out = vec![0f32; count];
        if count != 0 {
            buffer.copy_to_host(f32_slice_as_bytes_mut(&mut out))?;
        }
        Ok(out)
    }

    fn download_pooled_f32_band(
        buffer: &CudaPooledDeviceBuffer,
        count: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let mut out = vec![0f32; count];
        if count != 0 {
            buffer.copy_to_host(f32_slice_as_bytes_mut(&mut out))?;
        }
        Ok(out)
    }

    fn launch_transcode_dwt97_idct(
        &self,
        dims: Reversible53Dims,
        blocks: &CudaDeviceBuffer,
        spatial: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::TranscodeDwt97Idct)?;
        let mut blocks_ptr = blocks.device_ptr();
        let mut block_cols = dims.block_cols;
        let mut width = dims.width;
        let mut height = dims.height;
        let mut spatial_ptr = spatial.device_ptr();
        let mut params = [
            (&raw mut blocks_ptr).cast::<c_void>(),
            (&raw mut block_cols).cast::<c_void>(),
            (&raw mut width).cast::<c_void>(),
            (&raw mut height).cast::<c_void>(),
            (&raw mut spatial_ptr).cast::<c_void>(),
        ];
        let grid_w = u32::try_from(dims.width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let grid_h =
            u32::try_from(dims.height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let geometry = j2k_dwt53_launch_geometry(grid_w, grid_h)
            .ok_or(CudaError::LengthTooLarge { len: 0 })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_transcode_dwt97_row_lift(
        &self,
        dims: Reversible53Dims,
        spatial: &CudaDeviceBuffer,
        row_low: &CudaDeviceBuffer,
        row_high: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97RowLift)?;
        let mut spatial_ptr = spatial.device_ptr();
        let mut width = dims.width;
        let mut height = dims.height;
        let mut low_width = dims.low_width;
        let mut high_width = dims.high_width;
        let mut low_ptr = row_low.device_ptr();
        let mut high_ptr = row_high.device_ptr();
        let mut params = [
            (&raw mut spatial_ptr).cast::<c_void>(),
            (&raw mut width).cast::<c_void>(),
            (&raw mut height).cast::<c_void>(),
            (&raw mut low_width).cast::<c_void>(),
            (&raw mut high_width).cast::<c_void>(),
            (&raw mut low_ptr).cast::<c_void>(),
            (&raw mut high_ptr).cast::<c_void>(),
        ];
        let rows =
            usize::try_from(dims.height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let geometry =
            copy_u8_launch_geometry(rows).ok_or(CudaError::LengthTooLarge { len: rows })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_transcode_dwt97_column_lift(
        &self,
        rows_buffer: &CudaDeviceBuffer,
        band_width: i32,
        height: i32,
        low_out: &CudaDeviceBuffer,
        high_out: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let columns =
            usize::try_from(band_width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        if columns == 0 {
            return Ok(());
        }
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97ColumnLift)?;
        let mut rows_ptr = rows_buffer.device_ptr();
        let mut band = band_width;
        let mut rows = height;
        let mut low_ptr = low_out.device_ptr();
        let mut high_ptr = high_out.device_ptr();
        let mut params = [
            (&raw mut rows_ptr).cast::<c_void>(),
            (&raw mut band).cast::<c_void>(),
            (&raw mut rows).cast::<c_void>(),
            (&raw mut low_ptr).cast::<c_void>(),
            (&raw mut high_ptr).cast::<c_void>(),
        ];
        let geometry =
            copy_u8_launch_geometry(columns).ok_or(CudaError::LengthTooLarge { len: columns })?;
        self.launch_kernel(function, geometry, &mut params)
    }
}

impl CudaContext {
    /// Compute a same-geometry batch of irreversible single-level 9/7 transforms
    /// with one batched launch per stage, returning per-item bands plus real
    /// backend stage timings. All jobs must share geometry (`block_cols`,
    /// `block_rows`, `width`, `height`); `blocks` is the items' natural-order
    /// `f32` coefficients laid out contiguously (`item_count * block_cols *
    /// block_rows * 64`). Bit-identical to running `j2k_transcode_dwt97` per item.
    #[allow(clippy::similar_names)]
    pub fn j2k_transcode_dwt97_batch(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
    ) -> Result<(Vec<CudaTranscodeDwt97Bands>, CudaDwt97BatchStageTimings), CudaError> {
        let pool = self.buffer_pool();
        self.j2k_transcode_dwt97_batch_with_pool(
            blocks, item_count, block_cols, block_rows, width, height, &pool,
        )
    }

    /// Compute a same-geometry batch of irreversible single-level 9/7 transforms
    /// while reusing device buffers from `pool` for transient stage storage.
    #[allow(clippy::too_many_arguments, clippy::similar_names)]
    pub fn j2k_transcode_dwt97_batch_with_pool(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        pool: &CudaBufferPool,
    ) -> Result<(Vec<CudaTranscodeDwt97Bands>, CudaDwt97BatchStageTimings), CudaError> {
        let (bands, pack_upload_us, idct_row_lift_us, column_lift_us) = self
            .transcode_dwt97_batch_to_device(
                blocks, item_count, block_cols, block_rows, width, height, pool,
            )?;
        let Dwt97BatchDeviceBands {
            ll,
            lh,
            hl,
            hh,
            low_width,
            low_height,
            high_width,
            high_height,
        } = bands;

        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let (outputs, readback_us) = self.time_default_stream_us(|| {
            let ll_all = Self::download_pooled_f32_band(&ll, item_count * ll_size)?;
            let lh_all = Self::download_pooled_f32_band(&lh, item_count * lh_size)?;
            let hl_all = Self::download_pooled_f32_band(&hl, item_count * hl_size)?;
            let hh_all = Self::download_pooled_f32_band(&hh, item_count * hh_size)?;
            let mut outputs = Vec::with_capacity(item_count);
            for item in 0..item_count {
                outputs.push(CudaTranscodeDwt97Bands {
                    ll: ll_all[item * ll_size..(item + 1) * ll_size].to_vec(),
                    hl: hl_all[item * hl_size..(item + 1) * hl_size].to_vec(),
                    lh: lh_all[item * lh_size..(item + 1) * lh_size].to_vec(),
                    hh: hh_all[item * hh_size..(item + 1) * hh_size].to_vec(),
                    low_width,
                    low_height,
                    high_width,
                    high_height,
                });
            }
            Ok(outputs)
        })?;

        Ok((
            outputs,
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us,
                quantize_codeblock_us: 0,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us,
            },
        ))
    }

    /// Compute a same-geometry batch directly into device-resident
    /// prequantized HTJ2K code-block coefficients: staged 9/7 followed by
    /// per-subband deadzone quantization into code-block-major `i32` layout.
    /// `params` carries the per-subband inverse step sizes (derived by the
    /// caller from the `signinum-transcode` code-block oracle) and the
    /// code-block geometry.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names
    )]
    pub fn j2k_transcode_htj2k97_codeblock_batch_resident(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        params: CudaHtj2k97QuantizeParams,
    ) -> Result<(CudaHtj2k97DeviceCodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        let pool = self.buffer_pool();
        self.j2k_transcode_htj2k97_codeblock_batch_resident_with_pool(
            blocks, item_count, block_cols, block_rows, width, height, params, &pool,
        )
    }

    /// Compute a same-geometry batch directly into device-resident
    /// prequantized HTJ2K code-block coefficients while reusing transient stage
    /// buffers from `pool`.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names
    )]
    pub fn j2k_transcode_htj2k97_codeblock_batch_resident_with_pool(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        params: CudaHtj2k97QuantizeParams,
        pool: &CudaBufferPool,
    ) -> Result<(CudaHtj2k97DeviceCodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        let (bands, pack_upload_us, idct_row_lift_us, column_lift_us) = self
            .transcode_dwt97_batch_to_device(
                blocks, item_count, block_cols, block_rows, width, height, pool,
            )?;
        let Dwt97BatchDeviceBands {
            ll,
            lh,
            hl,
            hh,
            low_width,
            low_height,
            high_width,
            high_height,
        } = bands;

        let to_i32 = |value: usize| -> Result<i32, CudaError> {
            i32::try_from(value).map_err(|_| CudaError::LengthTooLarge { len: value })
        };
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;
        let cb_w = to_i32(params.cb_width)?;
        let cb_h = to_i32(params.cb_height)?;

        let alloc_i32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };
        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let ll_q = alloc_i32(item_count * ll_size)?;
        let lh_q = alloc_i32(item_count * lh_size)?;
        let hl_q = alloc_i32(item_count * hl_size)?;
        let hh_q = alloc_i32(item_count * hh_size)?;

        let ((), quantize_codeblock_us) = self.time_default_stream_us(|| {
            // One launch per subband, each with its own dims and inverse delta.
            self.launch_transcode_dwt97_quantize_codeblocks(
                pooled_device_buffer(&ll)?,
                pooled_device_buffer(&ll_q)?,
                to_i32(low_width)?,
                to_i32(low_height)?,
                cb_w,
                cb_h,
                params.inv_delta_ll,
                items,
            )?;
            self.launch_transcode_dwt97_quantize_codeblocks(
                pooled_device_buffer(&hl)?,
                pooled_device_buffer(&hl_q)?,
                to_i32(high_width)?,
                to_i32(low_height)?,
                cb_w,
                cb_h,
                params.inv_delta_hl,
                items,
            )?;
            self.launch_transcode_dwt97_quantize_codeblocks(
                pooled_device_buffer(&lh)?,
                pooled_device_buffer(&lh_q)?,
                to_i32(low_width)?,
                to_i32(high_height)?,
                cb_w,
                cb_h,
                params.inv_delta_lh,
                items,
            )?;
            self.launch_transcode_dwt97_quantize_codeblocks(
                pooled_device_buffer(&hh)?,
                pooled_device_buffer(&hh_q)?,
                to_i32(high_width)?,
                to_i32(high_height)?,
                cb_w,
                cb_h,
                params.inv_delta_hh,
                items,
            )?;
            Ok(())
        })?;

        Ok((
            CudaHtj2k97DeviceCodeblockBands {
                ll: ll_q,
                hl: hl_q,
                lh: lh_q,
                hh: hh_q,
                item_count,
                low_width,
                low_height,
                high_width,
                high_height,
            },
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us,
                quantize_codeblock_us,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us: 0,
            },
        ))
    }

    /// Compute a same-geometry batch directly from `i16` dequantized DCT
    /// coefficients into device-resident prequantized HTJ2K code-block
    /// coefficients while reusing transient stage buffers from `pool`.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names
    )]
    pub fn j2k_transcode_htj2k97_codeblock_i16_batch_resident_with_pool(
        &self,
        blocks: &[i16],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        params: CudaHtj2k97QuantizeParams,
        pool: &CudaBufferPool,
    ) -> Result<(CudaHtj2k97DeviceCodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        if !dwt97_fused_column_quantize_disabled() {
            return self.j2k_transcode_htj2k97_codeblock_i16_batch_resident_fused_with_pool(
                blocks, item_count, block_cols, block_rows, width, height, params, pool,
            );
        }

        let (bands, pack_upload_us, idct_row_lift_us, column_lift_us) = self
            .transcode_dwt97_i16_batch_to_device(
                blocks, item_count, block_cols, block_rows, width, height, pool,
            )?;
        let Dwt97BatchDeviceBands {
            ll,
            lh,
            hl,
            hh,
            low_width,
            low_height,
            high_width,
            high_height,
        } = bands;

        let to_i32 = |value: usize| -> Result<i32, CudaError> {
            i32::try_from(value).map_err(|_| CudaError::LengthTooLarge { len: value })
        };
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;
        let cb_w = to_i32(params.cb_width)?;
        let cb_h = to_i32(params.cb_height)?;

        let alloc_i32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };
        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let ll_q = alloc_i32(item_count * ll_size)?;
        let lh_q = alloc_i32(item_count * lh_size)?;
        let hl_q = alloc_i32(item_count * hl_size)?;
        let hh_q = alloc_i32(item_count * hh_size)?;

        let ((), quantize_codeblock_us) = self.time_default_stream_us(|| {
            self.launch_transcode_dwt97_quantize_codeblocks(
                pooled_device_buffer(&ll)?,
                pooled_device_buffer(&ll_q)?,
                to_i32(low_width)?,
                to_i32(low_height)?,
                cb_w,
                cb_h,
                params.inv_delta_ll,
                items,
            )?;
            self.launch_transcode_dwt97_quantize_codeblocks(
                pooled_device_buffer(&hl)?,
                pooled_device_buffer(&hl_q)?,
                to_i32(high_width)?,
                to_i32(low_height)?,
                cb_w,
                cb_h,
                params.inv_delta_hl,
                items,
            )?;
            self.launch_transcode_dwt97_quantize_codeblocks(
                pooled_device_buffer(&lh)?,
                pooled_device_buffer(&lh_q)?,
                to_i32(low_width)?,
                to_i32(high_height)?,
                cb_w,
                cb_h,
                params.inv_delta_lh,
                items,
            )?;
            self.launch_transcode_dwt97_quantize_codeblocks(
                pooled_device_buffer(&hh)?,
                pooled_device_buffer(&hh_q)?,
                to_i32(high_width)?,
                to_i32(high_height)?,
                cb_w,
                cb_h,
                params.inv_delta_hh,
                items,
            )?;
            Ok(())
        })?;

        Ok((
            CudaHtj2k97DeviceCodeblockBands {
                ll: ll_q,
                hl: hl_q,
                lh: lh_q,
                hh: hh_q,
                item_count,
                low_width,
                low_height,
                high_width,
                high_height,
            },
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us,
                quantize_codeblock_us,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us: 0,
            },
        ))
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names
    )]
    fn j2k_transcode_htj2k97_codeblock_i16_batch_resident_fused_with_pool(
        &self,
        blocks: &[i16],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        params: CudaHtj2k97QuantizeParams,
        pool: &CudaBufferPool,
    ) -> Result<(CudaHtj2k97DeviceCodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        if !TRANSCODE_PTX_BUILT_FROM_CUDA {
            return Err(CudaError::InvalidArgument {
                message: "CUDA transcode kernels were not built (nvcc unavailable at build time)"
                    .to_string(),
            });
        }
        let block_count = block_cols
            .checked_mul(block_rows)
            .ok_or(CudaError::LengthTooLarge { len: block_cols })?;
        let covered_w = block_cols
            .checked_mul(8)
            .ok_or(CudaError::LengthTooLarge { len: block_cols })?;
        let covered_h = block_rows
            .checked_mul(8)
            .ok_or(CudaError::LengthTooLarge { len: block_rows })?;
        let per_item_coeffs = block_count
            .checked_mul(64)
            .ok_or(CudaError::LengthTooLarge { len: block_count })?;
        let expected_coeffs =
            per_item_coeffs
                .checked_mul(item_count)
                .ok_or(CudaError::LengthTooLarge {
                    len: per_item_coeffs,
                })?;
        if item_count == 0
            || width == 0
            || height == 0
            || width > covered_w
            || height > covered_h
            || blocks.len() != expected_coeffs
        {
            return Err(CudaError::InvalidArgument {
                message: "9/7 transcode batch has unsupported grid geometry".to_string(),
            });
        }

        let low_width = width.div_ceil(2);
        let low_height = height.div_ceil(2);
        let high_width = width / 2;
        let high_height = height / 2;

        let to_i32 = |value: usize| -> Result<i32, CudaError> {
            i32::try_from(value).map_err(|_| CudaError::LengthTooLarge { len: value })
        };
        let dims = Reversible53Dims {
            block_cols: to_i32(block_cols)?,
            width: to_i32(width)?,
            height: to_i32(height)?,
            low_width: to_i32(low_width)?,
            high_width: to_i32(high_width)?,
        };
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;
        let blocks_per_item = to_i32(block_count)?;
        let low_height_i32 = to_i32(low_height)?;
        let high_height_i32 = to_i32(high_height)?;
        let cb_w = to_i32(params.cb_width)?;
        let cb_h = to_i32(params.cb_height)?;

        self.inner.set_current()?;

        let alloc_f32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };
        let alloc_i32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };
        let (buffers, pack_upload_us) = self.time_default_stream_us(|| {
            let spatial = alloc_f32(item_count * width * height)?;
            let row_low = alloc_f32(item_count * height * low_width)?;
            let row_high = alloc_f32(item_count * height * high_width)?;
            let blocks_dev = Dwt97BatchInput::I16(blocks).upload(pool)?;
            Ok((spatial, row_low, row_high, blocks_dev))
        })?;
        let (spatial, row_low, row_high, blocks_dev) = buffers;

        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let ll_q = alloc_i32(item_count * ll_size)?;
        let lh_q = alloc_i32(item_count * lh_size)?;
        let hl_q = alloc_i32(item_count * hl_size)?;
        let hh_q = alloc_i32(item_count * hh_size)?;

        let ((), idct_row_lift_us) = self.time_default_stream_us(|| {
            self.launch_transcode_dwt97_idct_i16_batch(
                dims,
                blocks_per_item,
                items,
                pooled_device_buffer(&blocks_dev)?,
                pooled_device_buffer(&spatial)?,
            )?;
            self.launch_transcode_dwt97_row_lift_batch(
                dims,
                items,
                pooled_device_buffer(&spatial)?,
                pooled_device_buffer(&row_low)?,
                pooled_device_buffer(&row_high)?,
            )?;
            Ok(())
        })?;

        let ((), column_quantize_us) = self.time_default_stream_us(|| {
            if dims.low_width > 0 {
                self.launch_transcode_dwt97_column_lift_quantize_codeblocks_batch(
                    pooled_device_buffer(&row_low)?,
                    dims.low_width,
                    dims.height,
                    low_height_i32,
                    high_height_i32,
                    items,
                    pooled_device_buffer(&ll_q)?,
                    pooled_device_buffer(&lh_q)?,
                    cb_w,
                    cb_h,
                    params.inv_delta_ll,
                    params.inv_delta_lh,
                )?;
            }
            if dims.high_width > 0 {
                self.launch_transcode_dwt97_column_lift_quantize_codeblocks_batch(
                    pooled_device_buffer(&row_high)?,
                    dims.high_width,
                    dims.height,
                    low_height_i32,
                    high_height_i32,
                    items,
                    pooled_device_buffer(&hl_q)?,
                    pooled_device_buffer(&hh_q)?,
                    cb_w,
                    cb_h,
                    params.inv_delta_hl,
                    params.inv_delta_hh,
                )?;
            }
            Ok(())
        })?;

        Ok((
            CudaHtj2k97DeviceCodeblockBands {
                ll: ll_q,
                hl: hl_q,
                lh: lh_q,
                hh: hh_q,
                item_count,
                low_width,
                low_height,
                high_width,
                high_height,
            },
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us: 0,
                quantize_codeblock_us: column_quantize_us,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us: 0,
            },
        ))
    }

    /// Compute a same-geometry batch directly into prequantized HTJ2K code-block
    /// coefficients: staged 9/7 followed by per-subband deadzone quantization
    /// into code-block-major `i32` layout. `params` carries the per-subband
    /// inverse step sizes (derived by the caller from the `signinum-transcode`
    /// code-block oracle) and the code-block geometry.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names
    )]
    pub fn j2k_transcode_htj2k97_codeblock_batch(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        params: CudaHtj2k97QuantizeParams,
    ) -> Result<(CudaHtj2k97CodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        let pool = self.buffer_pool();
        self.j2k_transcode_htj2k97_codeblock_batch_with_pool(
            blocks, item_count, block_cols, block_rows, width, height, params, &pool,
        )
    }

    /// Compute a same-geometry batch directly into prequantized HTJ2K
    /// code-block coefficients while reusing transient stage buffers from
    /// `pool`.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names
    )]
    pub fn j2k_transcode_htj2k97_codeblock_batch_with_pool(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        params: CudaHtj2k97QuantizeParams,
        pool: &CudaBufferPool,
    ) -> Result<(CudaHtj2k97CodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        let (bands, pack_upload_us, idct_row_lift_us, column_lift_us) = self
            .transcode_dwt97_batch_to_device(
                blocks, item_count, block_cols, block_rows, width, height, pool,
            )?;
        let Dwt97BatchDeviceBands {
            ll,
            lh,
            hl,
            hh,
            low_width,
            low_height,
            high_width,
            high_height,
        } = bands;

        let to_i32 = |value: usize| -> Result<i32, CudaError> {
            i32::try_from(value).map_err(|_| CudaError::LengthTooLarge { len: value })
        };
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;
        let cb_w = to_i32(params.cb_width)?;
        let cb_h = to_i32(params.cb_height)?;

        let alloc_i32 = |count: usize| -> Result<CudaDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            self.allocate(bytes)
        };
        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let ll_q = alloc_i32(item_count * ll_size)?;
        let lh_q = alloc_i32(item_count * lh_size)?;
        let hl_q = alloc_i32(item_count * hl_size)?;
        let hh_q = alloc_i32(item_count * hh_size)?;

        let ((), quantize_codeblock_us) = self.time_default_stream_us(|| {
            // One launch per subband, each with its own dims and inverse delta.
            self.launch_transcode_dwt97_quantize_codeblocks(
                pooled_device_buffer(&ll)?,
                &ll_q,
                to_i32(low_width)?,
                to_i32(low_height)?,
                cb_w,
                cb_h,
                params.inv_delta_ll,
                items,
            )?;
            self.launch_transcode_dwt97_quantize_codeblocks(
                pooled_device_buffer(&hl)?,
                &hl_q,
                to_i32(high_width)?,
                to_i32(low_height)?,
                cb_w,
                cb_h,
                params.inv_delta_hl,
                items,
            )?;
            self.launch_transcode_dwt97_quantize_codeblocks(
                pooled_device_buffer(&lh)?,
                &lh_q,
                to_i32(low_width)?,
                to_i32(high_height)?,
                cb_w,
                cb_h,
                params.inv_delta_lh,
                items,
            )?;
            self.launch_transcode_dwt97_quantize_codeblocks(
                pooled_device_buffer(&hh)?,
                &hh_q,
                to_i32(high_width)?,
                to_i32(high_height)?,
                cb_w,
                cb_h,
                params.inv_delta_hh,
                items,
            )?;
            Ok(())
        })?;

        let (codeblocks, readback_us) = self.time_default_stream_us(|| {
            Ok(CudaHtj2k97CodeblockBands {
                ll: Self::download_i32_band(&ll_q, item_count * ll_size)?,
                hl: Self::download_i32_band(&hl_q, item_count * hl_size)?,
                lh: Self::download_i32_band(&lh_q, item_count * lh_size)?,
                hh: Self::download_i32_band(&hh_q, item_count * hh_size)?,
                item_count,
                low_width,
                low_height,
                high_width,
                high_height,
            })
        })?;

        Ok((
            codeblocks,
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us,
                quantize_codeblock_us,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us,
            },
        ))
    }

    /// Run the shared staged 9/7 batch pipeline (alloc + upload, batched IDCT +
    /// row lift, batched column lift) and return the device-resident bands plus
    /// the three pre-readback stage timings.
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::too_many_arguments)]
    fn transcode_dwt97_batch_to_device(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        pool: &CudaBufferPool,
    ) -> Result<(Dwt97BatchDeviceBands, u128, u128, u128), CudaError> {
        self.transcode_dwt97_batch_input_to_device(
            Dwt97BatchInput::F32(blocks),
            item_count,
            block_cols,
            block_rows,
            width,
            height,
            pool,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn transcode_dwt97_i16_batch_to_device(
        &self,
        blocks: &[i16],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        pool: &CudaBufferPool,
    ) -> Result<(Dwt97BatchDeviceBands, u128, u128, u128), CudaError> {
        self.transcode_dwt97_batch_input_to_device(
            Dwt97BatchInput::I16(blocks),
            item_count,
            block_cols,
            block_rows,
            width,
            height,
            pool,
        )
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::too_many_arguments)]
    fn transcode_dwt97_batch_input_to_device(
        &self,
        input: Dwt97BatchInput<'_>,
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        pool: &CudaBufferPool,
    ) -> Result<(Dwt97BatchDeviceBands, u128, u128, u128), CudaError> {
        if !TRANSCODE_PTX_BUILT_FROM_CUDA {
            return Err(CudaError::InvalidArgument {
                message: "CUDA transcode kernels were not built (nvcc unavailable at build time)"
                    .to_string(),
            });
        }
        let block_count = block_cols
            .checked_mul(block_rows)
            .ok_or(CudaError::LengthTooLarge { len: block_cols })?;
        let covered_w = block_cols
            .checked_mul(8)
            .ok_or(CudaError::LengthTooLarge { len: block_cols })?;
        let covered_h = block_rows
            .checked_mul(8)
            .ok_or(CudaError::LengthTooLarge { len: block_rows })?;
        let per_item_coeffs = block_count
            .checked_mul(64)
            .ok_or(CudaError::LengthTooLarge { len: block_count })?;
        let expected_coeffs =
            per_item_coeffs
                .checked_mul(item_count)
                .ok_or(CudaError::LengthTooLarge {
                    len: per_item_coeffs,
                })?;
        if item_count == 0
            || width == 0
            || height == 0
            || width > covered_w
            || height > covered_h
            || input.len() != expected_coeffs
        {
            return Err(CudaError::InvalidArgument {
                message: "9/7 transcode batch has unsupported grid geometry".to_string(),
            });
        }

        let low_width = width.div_ceil(2);
        let low_height = height.div_ceil(2);
        let high_width = width / 2;
        let high_height = height / 2;

        let to_i32 = |value: usize| -> Result<i32, CudaError> {
            i32::try_from(value).map_err(|_| CudaError::LengthTooLarge { len: value })
        };
        let dims = Reversible53Dims {
            block_cols: to_i32(block_cols)?,
            width: to_i32(width)?,
            height: to_i32(height)?,
            low_width: to_i32(low_width)?,
            high_width: to_i32(high_width)?,
        };
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;
        let blocks_per_item = to_i32(block_count)?;
        let low_height_i32 = to_i32(low_height)?;
        let high_height_i32 = to_i32(high_height)?;

        self.inner.set_current()?;

        let alloc_f32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };

        // Stage: allocate batch buffers and upload all blocks.
        let (buffers, pack_upload_us) = self.time_default_stream_us(|| {
            let spatial = alloc_f32(item_count * width * height)?;
            let row_low = alloc_f32(item_count * height * low_width)?;
            let row_high = alloc_f32(item_count * height * high_width)?;
            let ll = alloc_f32(item_count * low_width * low_height)?;
            let lh = alloc_f32(item_count * low_width * high_height)?;
            let hl = alloc_f32(item_count * high_width * low_height)?;
            let hh = alloc_f32(item_count * high_width * high_height)?;
            let blocks_dev = input.upload(pool)?;
            Ok((spatial, row_low, row_high, ll, lh, hl, hh, blocks_dev))
        })?;
        let (spatial, row_low, row_high, ll, lh, hl, hh, blocks_dev) = buffers;

        // Stage: batched separable IDCT then horizontal 9/7 row lift.
        let ((), idct_row_lift_us) = self.time_default_stream_us(|| {
            match input {
                Dwt97BatchInput::F32(_) => self.launch_transcode_dwt97_idct_batch(
                    dims,
                    blocks_per_item,
                    items,
                    pooled_device_buffer(&blocks_dev)?,
                    pooled_device_buffer(&spatial)?,
                )?,
                Dwt97BatchInput::I16(_) => self.launch_transcode_dwt97_idct_i16_batch(
                    dims,
                    blocks_per_item,
                    items,
                    pooled_device_buffer(&blocks_dev)?,
                    pooled_device_buffer(&spatial)?,
                )?,
            }
            self.launch_transcode_dwt97_row_lift_batch(
                dims,
                items,
                pooled_device_buffer(&spatial)?,
                pooled_device_buffer(&row_low)?,
                pooled_device_buffer(&row_high)?,
            )?;
            Ok(())
        })?;

        // Stage: batched vertical 9/7 column lift for both low and high rows.
        let ((), column_lift_us) = self.time_default_stream_us(|| {
            if dims.low_width > 0 {
                self.launch_transcode_dwt97_column_lift_batch(
                    pooled_device_buffer(&row_low)?,
                    dims.low_width,
                    dims.height,
                    low_height_i32,
                    high_height_i32,
                    items,
                    pooled_device_buffer(&ll)?,
                    pooled_device_buffer(&lh)?,
                )?;
            }
            if dims.high_width > 0 {
                self.launch_transcode_dwt97_column_lift_batch(
                    pooled_device_buffer(&row_high)?,
                    dims.high_width,
                    dims.height,
                    low_height_i32,
                    high_height_i32,
                    items,
                    pooled_device_buffer(&hl)?,
                    pooled_device_buffer(&hh)?,
                )?;
            }
            Ok(())
        })?;

        Ok((
            Dwt97BatchDeviceBands {
                ll,
                lh,
                hl,
                hh,
                low_width,
                low_height,
                high_width,
                high_height,
            },
            pack_upload_us,
            idct_row_lift_us,
            column_lift_us,
        ))
    }

    fn launch_transcode_dwt97_idct_batch(
        &self,
        dims: Reversible53Dims,
        blocks_per_item: i32,
        items: u32,
        blocks: &CudaDeviceBuffer,
        spatial: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97IdctBatch)?;
        let mut blocks_ptr = blocks.device_ptr();
        let mut block_cols = dims.block_cols;
        let mut width = dims.width;
        let mut height = dims.height;
        let mut blocks_per_item = blocks_per_item;
        let mut spatial_ptr = spatial.device_ptr();
        let mut params = [
            (&raw mut blocks_ptr).cast::<c_void>(),
            (&raw mut block_cols).cast::<c_void>(),
            (&raw mut width).cast::<c_void>(),
            (&raw mut height).cast::<c_void>(),
            (&raw mut blocks_per_item).cast::<c_void>(),
            (&raw mut spatial_ptr).cast::<c_void>(),
        ];
        let grid_w = u32::try_from(dims.width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let grid_h =
            u32::try_from(dims.height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let base = j2k_dwt53_launch_geometry(grid_w, grid_h)
            .ok_or(CudaError::LengthTooLarge { len: 0 })?;
        let geometry = kernels::CudaLaunchGeometry {
            grid: (base.grid.0, base.grid.1, items),
            block: base.block,
        };
        self.launch_kernel_async(function, geometry, &mut params)
    }

    fn launch_transcode_dwt97_idct_i16_batch(
        &self,
        dims: Reversible53Dims,
        blocks_per_item: i32,
        items: u32,
        blocks: &CudaDeviceBuffer,
        spatial: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97IdctI16Batch)?;
        let mut blocks_ptr = blocks.device_ptr();
        let mut block_cols = dims.block_cols;
        let mut width = dims.width;
        let mut height = dims.height;
        let mut blocks_per_item = blocks_per_item;
        let mut spatial_ptr = spatial.device_ptr();
        let mut params = [
            (&raw mut blocks_ptr).cast::<c_void>(),
            (&raw mut block_cols).cast::<c_void>(),
            (&raw mut width).cast::<c_void>(),
            (&raw mut height).cast::<c_void>(),
            (&raw mut blocks_per_item).cast::<c_void>(),
            (&raw mut spatial_ptr).cast::<c_void>(),
        ];
        let grid_w = u32::try_from(dims.width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let grid_h =
            u32::try_from(dims.height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let base = j2k_dwt53_launch_geometry(grid_w, grid_h)
            .ok_or(CudaError::LengthTooLarge { len: 0 })?;
        let geometry = kernels::CudaLaunchGeometry {
            grid: (base.grid.0, base.grid.1, items),
            block: base.block,
        };
        self.launch_kernel_async(function, geometry, &mut params)
    }

    fn launch_transcode_dwt97_row_lift_batch(
        &self,
        dims: Reversible53Dims,
        items: u32,
        spatial: &CudaDeviceBuffer,
        row_low: &CudaDeviceBuffer,
        row_high: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        if dims.width <= DWT97_ROW_LIFT_MAX_WIDTH {
            return self.launch_transcode_dwt97_row_lift_batch_coop(
                dims, items, spatial, row_low, row_high,
            );
        }

        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97RowLiftBatch)?;
        let mut spatial_ptr = spatial.device_ptr();
        let mut width = dims.width;
        let mut height = dims.height;
        let mut low_width = dims.low_width;
        let mut high_width = dims.high_width;
        let mut low_ptr = row_low.device_ptr();
        let mut high_ptr = row_high.device_ptr();
        let mut params = [
            (&raw mut spatial_ptr).cast::<c_void>(),
            (&raw mut width).cast::<c_void>(),
            (&raw mut height).cast::<c_void>(),
            (&raw mut low_width).cast::<c_void>(),
            (&raw mut high_width).cast::<c_void>(),
            (&raw mut low_ptr).cast::<c_void>(),
            (&raw mut high_ptr).cast::<c_void>(),
        ];
        let rows =
            usize::try_from(dims.height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let base = copy_u8_launch_geometry(rows).ok_or(CudaError::LengthTooLarge { len: rows })?;
        let geometry = kernels::CudaLaunchGeometry {
            grid: (base.grid.0, items, 1),
            block: base.block,
        };
        self.launch_kernel_async(function, geometry, &mut params)
    }

    fn launch_transcode_dwt97_row_lift_batch_coop(
        &self,
        dims: Reversible53Dims,
        items: u32,
        spatial: &CudaDeviceBuffer,
        row_low: &CudaDeviceBuffer,
        row_high: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97RowLiftBatchCoop)?;
        let mut spatial_ptr = spatial.device_ptr();
        let mut width = dims.width;
        let mut height = dims.height;
        let mut low_width = dims.low_width;
        let mut high_width = dims.high_width;
        let mut low_ptr = row_low.device_ptr();
        let mut high_ptr = row_high.device_ptr();
        let mut params = [
            (&raw mut spatial_ptr).cast::<c_void>(),
            (&raw mut width).cast::<c_void>(),
            (&raw mut height).cast::<c_void>(),
            (&raw mut low_width).cast::<c_void>(),
            (&raw mut high_width).cast::<c_void>(),
            (&raw mut low_ptr).cast::<c_void>(),
            (&raw mut high_ptr).cast::<c_void>(),
        ];
        let rows =
            usize::try_from(dims.height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let rows_per_block = DWT97_ROW_LIFT_COOP_ROWS_PER_BLOCK as usize;
        let grid_x = c_uint::try_from(rows.div_ceil(rows_per_block))
            .map_err(|_| CudaError::LengthTooLarge { len: rows })?;
        let geometry = kernels::CudaLaunchGeometry {
            grid: (grid_x, items, 1),
            block: (
                DWT97_ROW_LIFT_COOP_THREADS_X,
                DWT97_ROW_LIFT_COOP_ROWS_PER_BLOCK,
                1,
            ),
        };
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_transcode_dwt97_column_lift_batch(
        &self,
        rows_buffer: &CudaDeviceBuffer,
        band_width: i32,
        height: i32,
        low_height: i32,
        high_height: i32,
        items: u32,
        low_out: &CudaDeviceBuffer,
        high_out: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let columns =
            usize::try_from(band_width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        if columns == 0 {
            return Ok(());
        }
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97ColumnLiftBatch)?;
        let mut rows_ptr = rows_buffer.device_ptr();
        let mut band = band_width;
        let mut rows = height;
        let mut low_h = low_height;
        let mut high_h = high_height;
        let mut low_ptr = low_out.device_ptr();
        let mut high_ptr = high_out.device_ptr();
        let mut params = [
            (&raw mut rows_ptr).cast::<c_void>(),
            (&raw mut band).cast::<c_void>(),
            (&raw mut rows).cast::<c_void>(),
            (&raw mut low_h).cast::<c_void>(),
            (&raw mut high_h).cast::<c_void>(),
            (&raw mut low_ptr).cast::<c_void>(),
            (&raw mut high_ptr).cast::<c_void>(),
        ];
        let base =
            copy_u8_launch_geometry(columns).ok_or(CudaError::LengthTooLarge { len: columns })?;
        let geometry = kernels::CudaLaunchGeometry {
            grid: (base.grid.0, items, 1),
            block: base.block,
        };
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_transcode_dwt97_column_lift_quantize_codeblocks_batch(
        &self,
        rows_buffer: &CudaDeviceBuffer,
        band_width: i32,
        height: i32,
        low_height: i32,
        high_height: i32,
        items: u32,
        low_out: &CudaDeviceBuffer,
        high_out: &CudaDeviceBuffer,
        cb_width: i32,
        cb_height: i32,
        inv_delta_low: f32,
        inv_delta_high: f32,
    ) -> Result<(), CudaError> {
        let columns =
            usize::try_from(band_width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        if columns == 0 {
            return Ok(());
        }
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch)?;
        let mut rows_ptr = rows_buffer.device_ptr();
        let mut band = band_width;
        let mut rows = height;
        let mut low_h = low_height;
        let mut high_h = high_height;
        let mut low_ptr = low_out.device_ptr();
        let mut high_ptr = high_out.device_ptr();
        let mut cb_w = cb_width;
        let mut cb_h = cb_height;
        let mut inv_low = inv_delta_low;
        let mut inv_high = inv_delta_high;
        let mut params = [
            (&raw mut rows_ptr).cast::<c_void>(),
            (&raw mut band).cast::<c_void>(),
            (&raw mut rows).cast::<c_void>(),
            (&raw mut low_h).cast::<c_void>(),
            (&raw mut high_h).cast::<c_void>(),
            (&raw mut low_ptr).cast::<c_void>(),
            (&raw mut high_ptr).cast::<c_void>(),
            (&raw mut cb_w).cast::<c_void>(),
            (&raw mut cb_h).cast::<c_void>(),
            (&raw mut inv_low).cast::<c_void>(),
            (&raw mut inv_high).cast::<c_void>(),
        ];
        let base =
            copy_u8_launch_geometry(columns).ok_or(CudaError::LengthTooLarge { len: columns })?;
        let geometry = kernels::CudaLaunchGeometry {
            grid: (base.grid.0, items, 1),
            block: base.block,
        };
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_transcode_dwt97_quantize_codeblocks(
        &self,
        band: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        width: i32,
        height: i32,
        cb_width: i32,
        cb_height: i32,
        inv_delta: f32,
        items: u32,
    ) -> Result<(), CudaError> {
        if width <= 0 || height <= 0 {
            return Ok(());
        }
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97QuantizeCodeblocks)?;
        let mut band_ptr = band.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut width = width;
        let mut height = height;
        let mut cb_width = cb_width;
        let mut cb_height = cb_height;
        let mut inv_delta = inv_delta;
        let mut params = [
            (&raw mut band_ptr).cast::<c_void>(),
            (&raw mut output_ptr).cast::<c_void>(),
            (&raw mut width).cast::<c_void>(),
            (&raw mut height).cast::<c_void>(),
            (&raw mut cb_width).cast::<c_void>(),
            (&raw mut cb_height).cast::<c_void>(),
            (&raw mut inv_delta).cast::<c_void>(),
        ];
        let grid_w = u32::try_from(width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let grid_h = u32::try_from(height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let base = j2k_dwt53_launch_geometry(grid_w, grid_h)
            .ok_or(CudaError::LengthTooLarge { len: 0 })?;
        let geometry = kernels::CudaLaunchGeometry {
            grid: (base.grid.0, base.grid.1, items),
            block: base.block,
        };
        self.launch_kernel_async(function, geometry, &mut params)
    }
}

#[derive(Debug)]
struct CompiledKernel {
    module: CuModule,
    function: CuFunction,
}

impl CompiledKernel {
    fn load(context: &ContextInner, kernel: CudaKernel) -> Result<Self, CudaError> {
        context.set_current()?;
        let mut module = std::ptr::null_mut();
        // SAFETY: image is a NUL-terminated PTX string. CUDA copies or parses
        // it during module load, and the context cache unloads the module on
        // context drop.
        context.driver.check("cuModuleLoadData", unsafe {
            (context.driver.cu_module_load_data)(
                &raw mut module,
                kernel.ptx().as_ptr().cast::<c_void>(),
            )
        })?;
        let mut function = std::ptr::null_mut();
        // SAFETY: name is a NUL-terminated kernel symbol in this module.
        context.driver.check("cuModuleGetFunction", unsafe {
            (context.driver.cu_module_get_function)(
                &raw mut function,
                module,
                kernel.entrypoint().as_ptr().cast::<c_char>(),
            )
        })?;
        Ok(Self { module, function })
    }
}

// SAFETY: CompiledKernel stores opaque CUDA module/function handles. Lifetime
// and unloading are coordinated by ContextInner's module cache mutex.
unsafe impl Send for CompiledKernel {}

impl CudaDeviceBuffer {
    /// CUDA context that owns this allocation.
    pub fn context(&self) -> CudaContext {
        self.context.clone()
    }

    /// Raw CUDA device pointer value.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Device allocation length in bytes.
    pub fn byte_len(&self) -> usize {
        self.len
    }

    /// Borrow this allocation as a typed immutable device view.
    pub fn typed_view<T>(&self) -> Result<CudaDeviceBufferView<'_, T>, CudaError> {
        let element_size = std::mem::size_of::<T>();
        if element_size == 0 || !self.len.is_multiple_of(element_size) {
            return Err(CudaError::LengthNotElementAligned {
                bytes: self.len,
                element_size,
            });
        }
        Ok(CudaDeviceBufferView {
            ptr: self.ptr,
            len: self.len / element_size,
            _marker: std::marker::PhantomData,
        })
    }

    /// Borrow this allocation as a typed mutable device view.
    pub fn typed_view_mut<T>(&mut self) -> Result<CudaDeviceBufferViewMut<'_, T>, CudaError> {
        let element_size = std::mem::size_of::<T>();
        if element_size == 0 || !self.len.is_multiple_of(element_size) {
            return Err(CudaError::LengthNotElementAligned {
                bytes: self.len,
                element_size,
            });
        }
        Ok(CudaDeviceBufferViewMut {
            ptr: self.ptr,
            len: self.len / element_size,
            _marker: std::marker::PhantomData,
        })
    }

    /// Copy device bytes into caller-owned host output.
    pub fn copy_to_host(&self, out: &mut [u8]) -> Result<(), CudaError> {
        if out.len() < self.len {
            return Err(CudaError::OutputTooSmall {
                required: self.len,
                have: out.len(),
            });
        }
        if self.len == 0 {
            return Ok(());
        }

        self.context.inner.set_current()?;
        // SAFETY: ptr is a live device allocation of self.len bytes, and out is
        // valid for at least self.len bytes.
        self.context.inner.driver.check("cuMemcpyDtoH_v2", unsafe {
            (self.context.inner.driver.cu_memcpy_dtoh)(
                out.as_mut_ptr().cast::<c_void>(),
                self.ptr,
                self.len,
            )
        })
    }

    /// Copy a byte range from this device buffer into caller-owned host output.
    pub fn copy_range_to_host(&self, offset: usize, out: &mut [u8]) -> Result<(), CudaError> {
        let end = offset
            .checked_add(out.len())
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if end > self.len {
            return Err(CudaError::OutputTooSmall {
                required: end,
                have: self.len,
            });
        }
        if out.is_empty() {
            return Ok(());
        }

        self.context.inner.set_current()?;
        let source = self
            .ptr
            .checked_add(
                u64::try_from(offset).map_err(|_| CudaError::LengthTooLarge { len: offset })?,
            )
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        // SAFETY: `source` is inside this live device allocation, and `out`
        // is valid for the requested range length after the bounds check above.
        self.context.inner.driver.check("cuMemcpyDtoH_v2", unsafe {
            (self.context.inner.driver.cu_memcpy_dtoh)(
                out.as_mut_ptr().cast::<c_void>(),
                source,
                out.len(),
            )
        })
    }

    /// Copy a byte range from this device buffer into uninitialized host output.
    pub fn copy_range_to_host_uninit(
        &self,
        offset: usize,
        out: &mut [std::mem::MaybeUninit<u8>],
    ) -> Result<(), CudaError> {
        let end = offset
            .checked_add(out.len())
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if end > self.len {
            return Err(CudaError::OutputTooSmall {
                required: end,
                have: self.len,
            });
        }
        if out.is_empty() {
            return Ok(());
        }

        self.context.inner.set_current()?;
        let source = self
            .ptr
            .checked_add(
                u64::try_from(offset).map_err(|_| CudaError::LengthTooLarge { len: offset })?,
            )
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        // SAFETY: `source` is inside this live device allocation, and `out`
        // points at writable spare capacity for exactly the requested byte
        // count. The caller decides when those bytes become initialized.
        self.context.inner.driver.check("cuMemcpyDtoH_v2", unsafe {
            (self.context.inner.driver.cu_memcpy_dtoh)(
                out.as_mut_ptr().cast::<c_void>(),
                source,
                out.len(),
            )
        })
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let _ = self.context.inner.set_current();
            // SAFETY: ptr was allocated by this CUDA context. Drop cannot
            // surface errors, so failures are ignored during cleanup.
            let _ = unsafe { (self.context.inner.driver.cu_mem_free)(self.ptr) };
        }
    }
}

fn htj2k_packetization_kernel_packets(
    packets: &[CudaHtj2kPacketizationPacket],
    subbands: &[CudaHtj2kPacketizationSubband],
    blocks: &[CudaHtj2kPacketizationBlock],
    payload_len: usize,
) -> Result<Vec<CudaHtj2kPacketizationKernelPacket>, CudaError> {
    let mut output_offset = 0usize;
    let mut kernel_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let block_start = packet.block_start as usize;
        let block_count = packet.block_count as usize;
        let block_end = block_start
            .checked_add(block_count)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if block_end > blocks.len() {
            return Err(CudaError::LengthTooLarge { len: block_end });
        }
        let subband_start = packet.subband_start as usize;
        let subband_count = packet.subband_count as usize;
        let subband_end = subband_start
            .checked_add(subband_count)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if subband_end > subbands.len() {
            return Err(CudaError::LengthTooLarge { len: subband_end });
        }
        for subband in &subbands[subband_start..subband_end] {
            if subband.num_cbs_x == 0 || subband.num_cbs_y == 0 {
                return Err(CudaError::LengthTooLarge { len: 0 });
            }
            let subband_block_start = subband.block_start as usize;
            let subband_block_count = subband.block_count as usize;
            let subband_block_end = subband_block_start
                .checked_add(subband_block_count)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if subband_block_start < block_start || subband_block_end > block_end {
                return Err(CudaError::LengthTooLarge {
                    len: subband_block_end,
                });
            }
            let expected_blocks = (subband.num_cbs_x as usize)
                .checked_mul(subband.num_cbs_y as usize)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if expected_blocks != subband_block_count {
                return Err(CudaError::LengthTooLarge {
                    len: expected_blocks,
                });
            }
        }
        for block in &blocks[block_start..block_end] {
            let data_end = (block.data_offset as usize)
                .checked_add(block.data_len as usize)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if data_end > payload_len {
                return Err(CudaError::LengthTooLarge { len: data_end });
            }
        }
        let output_capacity = packet.output_capacity as usize;
        let next_output = output_offset
            .checked_add(output_capacity)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if next_output > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge { len: next_output });
        }
        kernel_packets.push(CudaHtj2kPacketizationKernelPacket {
            block_start: packet.block_start,
            block_count: packet.block_count,
            subband_start: packet.subband_start,
            subband_count: packet.subband_count,
            output_offset: u32::try_from(output_offset)
                .map_err(|_| CudaError::LengthTooLarge { len: output_offset })?,
            output_capacity: packet.output_capacity,
            layer: packet.layer,
        });
        output_offset = next_output;
    }
    Ok(kernel_packets)
}

fn validate_htj2k_packetization_tag_state(
    subbands: &[CudaHtj2kPacketizationSubband],
    subband_tag_states: &[CudaHtj2kPacketizationSubbandTagState],
    tag_nodes: &[CudaHtj2kPacketizationTagNodeState],
) -> Result<(), CudaError> {
    if subband_tag_states.is_empty() {
        if tag_nodes.is_empty() {
            return Ok(());
        }
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K packetization tag nodes require subband tag states".to_string(),
        });
    }
    if subband_tag_states.len() != subbands.len() {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K packetization subband tag-state count must match subband count"
                .to_string(),
        });
    }
    for (subband_index, (subband, state)) in subbands.iter().zip(subband_tag_states).enumerate() {
        let expected_node_count =
            htj2k_packetization_tag_tree_node_count(subband.num_cbs_x, subband.num_cbs_y)?;
        if state.node_count as usize != expected_node_count {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "HTJ2K packetization tag-state node count does not match subband {subband_index}"
                ),
            });
        }
        let node_count = state.node_count as usize;
        let inclusion_end = (state.inclusion_node_start as usize)
            .checked_add(node_count)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let zero_bitplane_end = (state.zero_bitplane_node_start as usize)
            .checked_add(node_count)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if inclusion_end > tag_nodes.len() || zero_bitplane_end > tag_nodes.len() {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "HTJ2K packetization tag-state offsets exceed tag node count at subband {subband_index}"
                ),
            });
        }
    }
    Ok(())
}

const HTJ2K_PACKET_MAX_TAG_NODES: usize = 2048;
const HTJ2K_PACKET_MAX_TAG_LEVELS: usize = 16;

fn htj2k_packetization_tag_tree_node_count(width: u32, height: u32) -> Result<usize, CudaError> {
    if width == 0 || height == 0 {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K packetization tag-tree dimensions must be nonzero".to_string(),
        });
    }
    let mut levels = 0usize;
    let mut total = 0usize;
    let mut w = width as usize;
    let mut h = height as usize;
    loop {
        if levels >= HTJ2K_PACKET_MAX_TAG_LEVELS {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K packetization tag-tree exceeds kernel level bounds".to_string(),
            });
        }
        let nodes = w
            .checked_mul(h)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        total = total
            .checked_add(nodes)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if total > HTJ2K_PACKET_MAX_TAG_NODES {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K packetization tag-tree exceeds kernel node bounds".to_string(),
            });
        }
        levels += 1;
        if w <= 1 && h <= 1 {
            return Ok(total);
        }
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
}

fn f32_slice_as_bytes(samples: &[f32]) -> &[u8] {
    // SAFETY: f32 has no invalid bit patterns, and the output byte slice is
    // read-only with the same lifetime as the input samples.
    unsafe {
        std::slice::from_raw_parts(
            samples.as_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
}

fn f32_slice_as_bytes_mut(samples: &mut [f32]) -> &mut [u8] {
    // SAFETY: the returned byte slice covers exactly the same initialized f32
    // storage and is used only for CUDA copies into the existing allocation.
    unsafe {
        std::slice::from_raw_parts_mut(
            samples.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
}

fn i16_slice_as_bytes(samples: &[i16]) -> &[u8] {
    // SAFETY: i16 has no invalid bit patterns, and the output byte slice is
    // read-only with the same lifetime as the input coefficients.
    unsafe {
        std::slice::from_raw_parts(
            samples.as_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
}

fn i32_slice_as_bytes(samples: &[i32]) -> &[u8] {
    // SAFETY: i32 has no invalid bit patterns, and the output byte slice is
    // read-only with the same lifetime as the input coefficients.
    unsafe {
        std::slice::from_raw_parts(
            samples.as_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
}

fn i32_slice_as_bytes_mut(samples: &mut [i32]) -> &mut [u8] {
    // SAFETY: the returned byte slice covers exactly the same initialized i32
    // storage and is used only for CUDA copies into the existing allocation.
    unsafe {
        std::slice::from_raw_parts_mut(
            samples.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
}

fn u16_slice_as_bytes(samples: &[u16]) -> &[u8] {
    // SAFETY: u16 has no invalid bit patterns, and the output byte slice is
    // read-only with the same lifetime as the input table.
    unsafe {
        std::slice::from_raw_parts(
            samples.as_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
}

#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
fn cuda_jpeg_huffman_table_as_bytes(table: &CudaJpegHuffmanTable) -> &[u8] {
    // SAFETY: CudaJpegHuffmanTable is repr(C), plain integer data copied to
    // CUDA and interpreted by the matching jpeg_decode_kernels.cu struct.
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(table).cast::<u8>(),
            std::mem::size_of::<CudaJpegHuffmanTable>(),
        )
    }
}

#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
fn cuda_jpeg_entropy_checkpoints_as_bytes(checkpoints: &[CudaJpegEntropyCheckpoint]) -> &[u8] {
    // SAFETY: CudaJpegEntropyCheckpoint is repr(C), plain integer data copied to
    // CUDA and interpreted by the matching jpeg_decode_kernels.cu struct.
    unsafe {
        std::slice::from_raw_parts(
            checkpoints.as_ptr().cast::<u8>(),
            std::mem::size_of_val(checkpoints),
        )
    }
}

#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
fn cuda_jpeg_decode_statuses_as_bytes(statuses: &[CudaJpegDecodeStatus]) -> &[u8] {
    // SAFETY: CudaJpegDecodeStatus is repr(C), plain integer data copied to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            statuses.as_ptr().cast::<u8>(),
            std::mem::size_of_val(statuses),
        )
    }
}

#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
fn cuda_jpeg_decode_statuses_as_bytes_mut(statuses: &mut [CudaJpegDecodeStatus]) -> &mut [u8] {
    // SAFETY: CudaJpegDecodeStatus is repr(C), plain integer data copied back
    // from CUDA into an identically-sized mutable slice.
    unsafe {
        std::slice::from_raw_parts_mut(
            statuses.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(statuses),
        )
    }
}

fn htj2k_encode_params_as_bytes(params: &CudaHtj2kEncodeParams) -> &[u8] {
    // SAFETY: CudaHtj2kEncodeParams is repr(C) POD data copied directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(params).cast::<u8>(),
            std::mem::size_of::<CudaHtj2kEncodeParams>(),
        )
    }
}

fn htj2k_encode_status_as_bytes(status: &CudaHtj2kEncodeStatus) -> &[u8] {
    // SAFETY: CudaHtj2kEncodeStatus is repr(C) integer POD data copied to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(status).cast::<u8>(),
            std::mem::size_of::<CudaHtj2kEncodeStatus>(),
        )
    }
}

fn htj2k_encode_status_as_bytes_mut(status: &mut CudaHtj2kEncodeStatus) -> &mut [u8] {
    // SAFETY: CudaHtj2kEncodeStatus is repr(C) integer POD data, and the byte
    // view is used only as a device-to-host copy target.
    unsafe {
        std::slice::from_raw_parts_mut(
            std::ptr::from_mut(status).cast::<u8>(),
            std::mem::size_of::<CudaHtj2kEncodeStatus>(),
        )
    }
}

fn htj2k_encode_statuses_byte_len(count: usize) -> Result<usize, CudaError> {
    count
        .checked_mul(std::mem::size_of::<CudaHtj2kEncodeStatus>())
        .ok_or(CudaError::LengthTooLarge { len: count })
}

fn htj2k_encode_jobs_as_bytes(jobs: &[CudaHtj2kEncodeKernelJob]) -> &[u8] {
    // SAFETY: CudaHtj2kEncodeKernelJob is repr(C) integer POD data copied
    // directly to CUDA.
    unsafe { std::slice::from_raw_parts(jobs.as_ptr().cast::<u8>(), std::mem::size_of_val(jobs)) }
}

fn htj2k_encode_multi_input_jobs_as_bytes(jobs: &[CudaHtj2kEncodeMultiInputKernelJob]) -> &[u8] {
    // SAFETY: CudaHtj2kEncodeMultiInputKernelJob is repr(C) integer POD data
    // copied directly to CUDA.
    unsafe { std::slice::from_raw_parts(jobs.as_ptr().cast::<u8>(), std::mem::size_of_val(jobs)) }
}

fn htj2k_encode_compact_jobs_as_bytes(jobs: &[CudaHtj2kEncodeCompactJob]) -> &[u8] {
    // SAFETY: CudaHtj2kEncodeCompactJob is repr(C) integer POD data copied
    // directly to CUDA.
    unsafe { std::slice::from_raw_parts(jobs.as_ptr().cast::<u8>(), std::mem::size_of_val(jobs)) }
}

fn htj2k_encode_statuses_as_bytes_mut(statuses: &mut [CudaHtj2kEncodeStatus]) -> &mut [u8] {
    // SAFETY: CudaHtj2kEncodeStatus is repr(C) integer POD data, and the byte
    // view is used only as a device-to-host copy target.
    unsafe {
        std::slice::from_raw_parts_mut(
            statuses.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(statuses),
        )
    }
}

fn htj2k_packetization_packets_as_bytes(packets: &[CudaHtj2kPacketizationKernelPacket]) -> &[u8] {
    // SAFETY: CudaHtj2kPacketizationKernelPacket is repr(C) integer POD data
    // copied directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            packets.as_ptr().cast::<u8>(),
            std::mem::size_of_val(packets),
        )
    }
}

fn htj2k_packetization_subbands_as_bytes(subbands: &[CudaHtj2kPacketizationSubband]) -> &[u8] {
    // SAFETY: CudaHtj2kPacketizationSubband is repr(C) integer POD data copied
    // directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            subbands.as_ptr().cast::<u8>(),
            std::mem::size_of_val(subbands),
        )
    }
}

fn htj2k_packetization_blocks_as_bytes(blocks: &[CudaHtj2kPacketizationBlock]) -> &[u8] {
    // SAFETY: CudaHtj2kPacketizationBlock is repr(C) integer POD data copied
    // directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(blocks.as_ptr().cast::<u8>(), std::mem::size_of_val(blocks))
    }
}

fn htj2k_packetization_subband_tag_states_as_bytes(
    states: &[CudaHtj2kPacketizationSubbandTagState],
) -> &[u8] {
    // SAFETY: CudaHtj2kPacketizationSubbandTagState is repr(C) integer POD
    // data copied directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(states.as_ptr().cast::<u8>(), std::mem::size_of_val(states))
    }
}

fn htj2k_packetization_tag_nodes_as_bytes(nodes: &[CudaHtj2kPacketizationTagNodeState]) -> &[u8] {
    // SAFETY: CudaHtj2kPacketizationTagNodeState is repr(C) integer POD data
    // copied directly to CUDA.
    unsafe { std::slice::from_raw_parts(nodes.as_ptr().cast::<u8>(), std::mem::size_of_val(nodes)) }
}

fn htj2k_packetization_statuses_as_bytes(statuses: &[CudaHtj2kPacketizationStatus]) -> &[u8] {
    // SAFETY: CudaHtj2kPacketizationStatus is repr(C) integer POD data copied
    // directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            statuses.as_ptr().cast::<u8>(),
            std::mem::size_of_val(statuses),
        )
    }
}

fn htj2k_packetization_statuses_as_bytes_mut(
    statuses: &mut [CudaHtj2kPacketizationStatus],
) -> &mut [u8] {
    // SAFETY: CudaHtj2kPacketizationStatus is repr(C) integer POD data, and
    // the byte view is used only as a device-to-host copy target.
    unsafe {
        std::slice::from_raw_parts_mut(
            statuses.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(statuses),
        )
    }
}

fn htj2k_jobs_as_bytes(jobs: &[CudaHtj2kCodeBlockKernelJob]) -> &[u8] {
    // SAFETY: CudaHtj2kCodeBlockKernelJob is repr(C), plain integer/f32 POD
    // data, and the byte view is used only for a host-to-device copy.
    unsafe { std::slice::from_raw_parts(jobs.as_ptr().cast::<u8>(), std::mem::size_of_val(jobs)) }
}

fn htj2k_cleanup_multi_jobs_as_bytes(jobs: &[CudaHtj2kCleanupMultiKernelJob]) -> &[u8] {
    // SAFETY: CudaHtj2kCleanupMultiKernelJob is repr(C), plain integer/f32 POD
    // data, and the byte view is used only for a host-to-device copy.
    unsafe { std::slice::from_raw_parts(jobs.as_ptr().cast::<u8>(), std::mem::size_of_val(jobs)) }
}

fn htj2k_dequantize_jobs_as_bytes(jobs: &[CudaHtj2kDequantizeKernelJob]) -> &[u8] {
    // SAFETY: CudaHtj2kDequantizeKernelJob is repr(C), plain integer/f32 POD
    // data, and the byte view is used only for a host-to-device copy.
    unsafe { std::slice::from_raw_parts(jobs.as_ptr().cast::<u8>(), std::mem::size_of_val(jobs)) }
}

fn htj2k_statuses_byte_len(count: usize) -> Result<usize, CudaError> {
    count
        .checked_mul(std::mem::size_of::<CudaHtj2kStatus>())
        .ok_or(CudaError::LengthTooLarge { len: count })
}

fn htj2k_statuses_as_bytes_mut(statuses: &mut [CudaHtj2kStatus]) -> &mut [u8] {
    // SAFETY: CudaHtj2kStatus is repr(C) integer POD data, and the byte view is
    // used only as a device-to-host copy target.
    unsafe {
        std::slice::from_raw_parts_mut(
            statuses.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(statuses),
        )
    }
}

fn idwt_job_as_bytes(job: &CudaJ2kIdwtJob) -> &[u8] {
    // SAFETY: CudaJ2kIdwtJob is repr(C) POD data copied directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(job).cast::<u8>(),
            std::mem::size_of::<CudaJ2kIdwtJob>(),
        )
    }
}

fn idwt_multi_jobs_as_bytes(jobs: &[CudaJ2kIdwtMultiKernelJob]) -> &[u8] {
    // SAFETY: CudaJ2kIdwtMultiKernelJob is repr(C), plain pointer/integer POD
    // data, and the byte view is used only for a host-to-device copy.
    unsafe { std::slice::from_raw_parts(jobs.as_ptr().cast::<u8>(), std::mem::size_of_val(jobs)) }
}

fn store_gray8_job_as_bytes(job: &CudaJ2kStoreGray8Job) -> &[u8] {
    // SAFETY: CudaJ2kStoreGray8Job is repr(C) POD data copied directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(job).cast::<u8>(),
            std::mem::size_of::<CudaJ2kStoreGray8Job>(),
        )
    }
}

fn store_gray16_job_as_bytes(job: &CudaJ2kStoreGray16Job) -> &[u8] {
    // SAFETY: CudaJ2kStoreGray16Job is repr(C) POD data copied directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(job).cast::<u8>(),
            std::mem::size_of::<CudaJ2kStoreGray16Job>(),
        )
    }
}

fn inverse_mct_job_as_bytes(job: &CudaJ2kInverseMctJob) -> &[u8] {
    // SAFETY: CudaJ2kInverseMctJob is repr(C) POD data copied directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(job).cast::<u8>(),
            std::mem::size_of::<CudaJ2kInverseMctJob>(),
        )
    }
}

fn store_rgb8_job_as_bytes(job: &CudaJ2kStoreRgb8Job) -> &[u8] {
    // SAFETY: CudaJ2kStoreRgb8Job is repr(C) POD data copied directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(job).cast::<u8>(),
            std::mem::size_of::<CudaJ2kStoreRgb8Job>(),
        )
    }
}

fn store_rgb16_job_as_bytes(job: &CudaJ2kStoreRgb16Job) -> &[u8] {
    // SAFETY: CudaJ2kStoreRgb16Job is repr(C) POD data copied directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(job).cast::<u8>(),
            std::mem::size_of::<CudaJ2kStoreRgb16Job>(),
        )
    }
}

fn store_rgb8_mct_batch_jobs_as_bytes(jobs: &[CudaJ2kStoreRgb8MctBatchJob]) -> &[u8] {
    // SAFETY: CudaJ2kStoreRgb8MctBatchJob is repr(C) POD data copied directly to CUDA.
    unsafe { std::slice::from_raw_parts(jobs.as_ptr().cast::<u8>(), std::mem::size_of_val(jobs)) }
}

fn store_rgb16_mct_job_as_bytes(job: &CudaJ2kStoreRgb16MctJob) -> &[u8] {
    // SAFETY: CudaJ2kStoreRgb16MctJob is repr(C) POD data copied directly to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(job).cast::<u8>(),
            std::mem::size_of::<CudaJ2kStoreRgb16MctJob>(),
        )
    }
}

fn validate_quantize_region(
    job: CudaJ2kQuantizeSubbandRegionJob,
    available_samples: usize,
) -> Result<(), CudaError> {
    if job.width == 0 || job.height == 0 {
        return Ok(());
    }
    if job.stride == 0
        || job
            .x0
            .checked_add(job.width)
            .is_none_or(|end_x| end_x > job.stride)
    {
        return Err(CudaError::LengthTooLarge {
            len: available_samples,
        });
    }

    let last_sample = (job.y0 as usize)
        .checked_add(job.height as usize - 1)
        .and_then(|row| row.checked_mul(job.stride as usize))
        .and_then(|row| row.checked_add(job.x0 as usize))
        .and_then(|row| row.checked_add(job.width as usize))
        .ok_or(CudaError::LengthTooLarge {
            len: available_samples,
        })?;
    if last_sample > available_samples {
        return Err(CudaError::OutputTooSmall {
            required: last_sample
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: last_sample })?,
            have: available_samples
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge {
                    len: available_samples,
                })?,
        });
    }
    Ok(())
}

fn validate_store_rgb8_plane(
    plane: &CudaDeviceBuffer,
    input_width: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
) -> Result<(), CudaError> {
    if source_x
        .checked_add(copy_width)
        .is_none_or(|end_x| end_x > input_width)
    {
        return Err(CudaError::LengthTooLarge {
            len: plane.byte_len(),
        });
    }
    let last_sample = if copy_height == 0 {
        0
    } else {
        (source_y as usize)
            .checked_add(copy_height as usize - 1)
            .and_then(|row| row.checked_mul(input_width as usize))
            .and_then(|row| row.checked_add(source_x as usize))
            .and_then(|row| row.checked_add(copy_width as usize))
            .ok_or(CudaError::LengthTooLarge {
                len: plane.byte_len(),
            })?
    };
    let required_bytes =
        last_sample
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge {
                len: plane.byte_len(),
            })?;
    if required_bytes > plane.byte_len() {
        return Err(CudaError::LengthTooLarge {
            len: required_bytes,
        });
    }
    Ok(())
}

fn checked_image_words(width: u32, height: u32, channels: usize) -> Result<usize, CudaError> {
    width
        .try_into()
        .ok()
        .and_then(|width: usize| width.checked_mul(height as usize))
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or(CudaError::ImageTooLarge {
            width,
            height,
            channels,
        })
}

#[cfg(test)]
mod tests {
    use super::{
        f32_slice_as_bytes_mut, format_idwt_batch_trace_row, idwt_batch_kernel_mode,
        idwt_batch_trace_row, idwt_batch_uses_cooperative_53, pool_fit_buffer_index_by_len,
        CudaContext, CudaError, CudaExecutionStats, CudaHtj2kCleanupMultiKernelJob,
        CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeTables,
        CudaHtj2kDequantizeTarget, CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob,
        CudaHtj2kEncodeResidentTarget, CudaHtj2kEncodeTables, CudaJ2kIdwtBatchKernelMode,
        CudaJ2kIdwtJob, CudaJ2kIdwtMultiKernelJob, CudaJ2kIdwtTarget, CudaJ2kQuantizeJob,
        CudaJ2kQuantizeSubbandRegionJob, CudaJ2kRect, CudaKernelName, CudaQueuedHtj2kCleanup,
    };

    fn cuda_runtime_required() -> bool {
        std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn runtime_raii_primitives_smoke_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let mut pinned = context.pinned_host_buffer(16).expect("pinned host buffer");
        pinned.as_mut_slice().copy_from_slice(&[7u8; 16]);
        assert_eq!(pinned.as_slice(), &[7u8; 16]);
        let pinned_upload = context
            .upload_pinned(&[1u8, 2, 3, 4])
            .expect("pinned upload");
        let mut uploaded = [0u8; 4];
        pinned_upload
            .copy_to_host(&mut uploaded)
            .expect("download pinned upload");
        assert_eq!(uploaded, [1, 2, 3, 4]);
        let pinned_float_upload = context
            .upload_f32_pinned(&[1.25, -2.5])
            .expect("pinned f32 upload");
        let mut downloaded_float_values = [0.0f32; 2];
        pinned_float_upload
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut downloaded_float_values))
            .expect("download pinned f32 upload");
        assert!((downloaded_float_values[0] - 1.25).abs() < f32::EPSILON);
        assert!((downloaded_float_values[1] + 2.5).abs() < f32::EPSILON);
        let pinned_integer_upload = context
            .upload_i32_pinned(&[7, -11])
            .expect("pinned i32 upload");
        let mut downloaded_integer_values = [0i32; 2];
        pinned_integer_upload
            .copy_to_host(super::i32_slice_as_bytes_mut(
                &mut downloaded_integer_values,
            ))
            .expect("download pinned i32 upload");
        assert_eq!(downloaded_integer_values, [7, -11]);
        let ranged_upload = context
            .upload(&[9u8, 8, 7, 6, 5, 4])
            .expect("range-copy upload");
        let mut range = [0u8; 3];
        ranged_upload
            .copy_range_to_host(2, &mut range)
            .expect("copy device range");
        assert_eq!(range, [7, 6, 5]);
        let mut uninit_range = Vec::with_capacity(3);
        ranged_upload
            .copy_range_to_host_uninit(1, uninit_range.spare_capacity_mut())
            .expect("copy device range into spare capacity");
        // SAFETY: copy_range_to_host_uninit returned success after writing
        // exactly three bytes into the Vec spare capacity.
        unsafe {
            uninit_range.set_len(3);
        }
        assert_eq!(uninit_range, [8, 7, 6]);
        let pool = context.buffer_pool();
        let pooled_upload = pool.upload(&[3u8, 1, 4, 1]).expect("pooled upload");
        let pooled_output = super::copy_pooled_bytes_to_vec_uninit(&pooled_upload, 4)
            .expect("copy pooled bytes into spare capacity");
        assert_eq!(pooled_output, [3, 1, 4, 1]);

        let module = context
            .preload_kernel_module(CudaKernelName::CopyU8)
            .expect("preload copy kernel");
        assert_eq!(module.entrypoint(), "signinum_copy_u8");

        let stream = context.create_stream().expect("CUDA stream");
        let start = context.create_event().expect("start event");
        let end = context.create_event().expect("end event");
        start.record(&stream).expect("record start");
        end.record(&stream).expect("record end");
        end.synchronize().expect("synchronize event");
        let elapsed = super::CudaEvent::elapsed_time_us(&start, &end).expect("elapsed time");
        assert!(elapsed >= 0.0);

        let pool = context.buffer_pool();
        {
            let buffer = pool.take(32).expect("pooled buffer");
            assert!(buffer.device_ptr() != 0);
            assert_eq!(buffer.byte_len(), 32);
            assert!(buffer.allocation_byte_len() >= 32);
        }
        let cached_count = pool.cached_count().expect("cached count");
        assert_eq!(cached_count, 1);
        {
            let buffer = pool.take(16).expect("reused pooled buffer");
            assert_eq!(buffer.byte_len(), 16);
            assert!(buffer.allocation_byte_len() >= 32);
        }

        let samples = [1.25f32, -2.5, 3.75, 4.5];
        {
            let buffer = pool.upload_f32(&samples).expect("pooled f32 upload");
            assert_eq!(
                buffer.byte_len(),
                samples.len() * std::mem::size_of::<f32>()
            );
            let mut downloaded = vec![0.0f32; samples.len()];
            buffer
                .copy_to_host(f32_slice_as_bytes_mut(&mut downloaded))
                .expect("download pooled f32 upload");
            assert_eq!(downloaded, samples);
        }
        let i16_samples = [-12i16, 7, 19, -4];
        {
            let buffer = pool
                .upload_i16_pinned(&i16_samples)
                .expect("pooled pinned i16 upload");
            assert_eq!(
                buffer.byte_len(),
                i16_samples.len() * std::mem::size_of::<i16>()
            );
            let mut downloaded_bytes = vec![0u8; std::mem::size_of_val(&i16_samples)];
            buffer
                .copy_to_host(&mut downloaded_bytes)
                .expect("download pooled pinned i16 upload");
            let downloaded = downloaded_bytes
                .chunks_exact(std::mem::size_of::<i16>())
                .map(|chunk| i16::from_ne_bytes([chunk[0], chunk[1]]))
                .collect::<Vec<_>>();
            assert_eq!(downloaded, i16_samples);
        }
        let cached_after_upload = pool.cached_count().expect("cached after upload");
        assert!(cached_after_upload >= cached_count);
    }

    #[test]
    fn pooled_i16_pinned_upload_is_size_gated() {
        assert!(super::should_use_pinned_pooled_i16_upload(4 * 1024 * 1024));
        assert!(!super::should_use_pinned_pooled_i16_upload(
            4 * 1024 * 1024 + 1
        ));
    }

    #[test]
    fn pooled_buffer_selection_uses_smallest_sufficient_fit() {
        let buffers = [(1usize, 32usize), (0, 64)];

        assert_eq!(
            pool_fit_buffer_index_by_len(buffers.iter().copied(), 16),
            Some(1)
        );
        let mut large_pool = (0..1024).map(|index| (index, 8usize)).collect::<Vec<_>>();
        large_pool[1022] = (1022, 32);
        large_pool[1023] = (1023, 64);

        assert_eq!(
            pool_fit_buffer_index_by_len(large_pool.iter().copied(), 16),
            Some(1022)
        );
        let mut recent_fit_pool = (0..4096).map(|index| (index, 8usize)).collect::<Vec<_>>();
        recent_fit_pool[4094] = (4094, 32);
        recent_fit_pool[4095] = (4095, 64);

        assert_eq!(
            pool_fit_buffer_index_by_len(recent_fit_pool.iter().copied(), 16),
            Some(4094)
        );
        let fallback_pool = (0..4096)
            .map(|index| match index.cmp(&3000) {
                std::cmp::Ordering::Less => (index, 8usize),
                std::cmp::Ordering::Equal => (index, 32),
                std::cmp::Ordering::Greater => (index, 64),
            })
            .collect::<Vec<_>>();

        assert_eq!(
            pool_fit_buffer_index_by_len(fallback_pool.iter().copied(), 16),
            Some(3000)
        );
    }

    #[test]
    fn pooled_take_with_trace_reports_allocation_and_reuse_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let (fresh, fresh_trace) = pool.take_with_trace(32).expect("fresh traced take");

        assert_eq!(fresh.byte_len(), 32);
        assert_eq!(fresh_trace.requested_len, 32);
        assert_eq!(fresh_trace.free_count_before, 0);
        assert_eq!(fresh_trace.scanned_count, 0);
        assert!(!fresh_trace.reused);
        assert!(fresh_trace.allocation_byte_len >= 32);
        drop(fresh);

        let (reused, reuse_trace) = pool.take_with_trace(16).expect("reused traced take");

        assert_eq!(reused.byte_len(), 16);
        assert_eq!(reuse_trace.requested_len, 16);
        assert_eq!(reuse_trace.free_count_before, 1);
        assert_eq!(reuse_trace.scanned_count, 1);
        assert!(reuse_trace.reused);
        assert!(reuse_trace.allocation_byte_len >= 32);
    }

    #[test]
    fn htj2k_encoded_codeblock_reports_segment_lengths_from_status() {
        let encoded = super::CudaHtj2kEncodedCodeBlock {
            data: vec![0u8; 10],
            status: super::CudaHtj2kEncodeStatus {
                code: super::HTJ2K_STATUS_OK,
                detail: 0,
                data_len: 10,
                number_of_coding_passes: 3,
                missing_bit_planes: 4,
                reserved0: 7,
                reserved1: 3,
                reserved2: 0,
            },
            execution: super::CudaExecutionStats::default(),
            stage_timings: super::CudaHtj2kEncodeStageTimings::default(),
        };

        assert_eq!(encoded.cleanup_length(), 7);
        assert_eq!(encoded.refinement_length(), 3);
    }

    #[test]
    fn htj2k_encode_compact_jobs_pack_actual_payloads() {
        let capacity = u32::try_from(super::HTJ2K_ENCODE_OUTPUT_CAPACITY)
            .expect("HTJ2K encode output capacity fits u32");
        let double_capacity = capacity
            .checked_mul(2)
            .expect("test output capacity fits u32");
        let kernel_jobs = [
            super::CudaHtj2kEncodeKernelJob {
                coefficient_offset: 0,
                coefficient_stride: 64,
                width: 64,
                height: 64,
                total_bitplanes: 8,
                output_offset: 0,
                output_capacity: capacity,
                target_coding_passes: 1,
            },
            super::CudaHtj2kEncodeKernelJob {
                coefficient_offset: 4096,
                coefficient_stride: 64,
                width: 64,
                height: 64,
                total_bitplanes: 8,
                output_offset: capacity,
                output_capacity: capacity,
                target_coding_passes: 1,
            },
            super::CudaHtj2kEncodeKernelJob {
                coefficient_offset: 8192,
                coefficient_stride: 64,
                width: 64,
                height: 64,
                total_bitplanes: 8,
                output_offset: double_capacity,
                output_capacity: capacity,
                target_coding_passes: 1,
            },
        ];
        let statuses = [
            super::CudaHtj2kEncodeStatus {
                code: super::HTJ2K_STATUS_OK,
                data_len: 12,
                reserved2: 0x8001_8002,
                ..super::CudaHtj2kEncodeStatus::default()
            },
            super::CudaHtj2kEncodeStatus {
                code: super::HTJ2K_STATUS_OK,
                data_len: 0,
                ..super::CudaHtj2kEncodeStatus::default()
            },
            super::CudaHtj2kEncodeStatus {
                code: super::HTJ2K_STATUS_OK,
                data_len: 7,
                ..super::CudaHtj2kEncodeStatus::default()
            },
        ];

        let (compact_jobs, compact_len) =
            super::htj2k_encode_compact_jobs(&statuses, &kernel_jobs).expect("valid compact jobs");

        assert_eq!(compact_len, 19);
        assert_eq!(
            compact_jobs,
            vec![
                super::CudaHtj2kEncodeCompactJob {
                    source_offset: 0,
                    compact_offset: 0,
                    data_len: 12,
                    reserved: 0x8001_8002,
                },
                super::CudaHtj2kEncodeCompactJob {
                    source_offset: capacity,
                    compact_offset: 12,
                    data_len: 0,
                    reserved: 0,
                },
                super::CudaHtj2kEncodeCompactJob {
                    source_offset: double_capacity,
                    compact_offset: 12,
                    data_len: 7,
                    reserved: 0,
                },
            ]
        );
    }

    #[test]
    fn htj2k_encode_resources_feed_resident_region_encode_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let vlc_table0 = [0u16; 2048];
        let vlc_table1 = [0u16; 2048];
        let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
        let resources = context
            .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
                vlc_table0: &vlc_table0,
                vlc_table1: &vlc_table1,
                uvlc_table: &uvlc_table,
            })
            .expect("encode resources");
        let coefficients = context
            .upload_i32_pinned(&[0, 0, 0, 0])
            .expect("resident coefficients");
        let jobs = [CudaHtj2kEncodeCodeBlockRegionJob {
            coefficient_offset: 0,
            coefficient_stride: 2,
            width: 2,
            height: 2,
            total_bitplanes: 1,
            target_coding_passes: 1,
        }];

        let encoded = context
            .encode_htj2k_codeblock_regions_resident_with_resources(
                &coefficients,
                4,
                &jobs,
                &resources,
            )
            .expect("resource-backed resident HTJ2K encode");

        assert_eq!(encoded.execution().kernel_dispatches(), 1);
        assert_eq!(encoded.code_blocks().len(), 1);
    }

    #[test]
    fn htj2k_encode_resident_region_reuses_pool_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let vlc_table0 = [0u16; 2048];
        let vlc_table1 = [0u16; 2048];
        let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
        let resources = context
            .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
                vlc_table0: &vlc_table0,
                vlc_table1: &vlc_table1,
                uvlc_table: &uvlc_table,
            })
            .expect("encode resources");
        let coefficients = context
            .upload_i32_pinned(&[0, 0, 0, 0])
            .expect("resident coefficients");
        let jobs = [CudaHtj2kEncodeCodeBlockRegionJob {
            coefficient_offset: 0,
            coefficient_stride: 2,
            width: 2,
            height: 2,
            total_bitplanes: 1,
            target_coding_passes: 1,
        }];

        let encoded = context
            .encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
                &coefficients,
                4,
                &jobs,
                &resources,
                &pool,
            )
            .expect("pooled resource-backed resident HTJ2K encode");

        assert_eq!(encoded.execution().kernel_dispatches(), 1);
        assert_eq!(encoded.code_blocks().len(), 1);
        assert!(pool.cached_count().expect("cached pooled encode buffers") >= 3);
    }

    #[test]
    fn htj2k_encode_codeblocks_resident_reuses_pool_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let vlc_table0 = [0u16; 2048];
        let vlc_table1 = [0u16; 2048];
        let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
        let resources = context
            .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
                vlc_table0: &vlc_table0,
                vlc_table1: &vlc_table1,
                uvlc_table: &uvlc_table,
            })
            .expect("encode resources");
        let coefficients = context
            .upload_i32_pinned(&[0, 0, 0, 0])
            .expect("resident coefficients");
        let jobs = [CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset: 0,
            width: 2,
            height: 2,
            total_bitplanes: 1,
            target_coding_passes: 1,
        }];

        let encoded = context
            .encode_htj2k_codeblocks_resident_with_resources_and_pool(
                &coefficients,
                4,
                &jobs,
                &resources,
                &pool,
            )
            .expect("pooled resource-backed resident HTJ2K codeblock encode");

        assert_eq!(encoded.execution().kernel_dispatches(), 1);
        assert_eq!(encoded.code_blocks().len(), 1);
        assert!(pool.cached_count().expect("cached pooled encode buffers") >= 3);
    }

    #[test]
    fn htj2k_encode_multi_resident_inputs_match_separate_batches_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let vlc_table0 = [0u16; 2048];
        let vlc_table1 = [0u16; 2048];
        let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
        let resources = context
            .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
                vlc_table0: &vlc_table0,
                vlc_table1: &vlc_table1,
                uvlc_table: &uvlc_table,
            })
            .expect("encode resources");
        let first = context
            .upload_i32_pinned(&[0, 0, 0, 0])
            .expect("first resident coefficients");
        let second = context
            .upload_i32_pinned(&[0, 0])
            .expect("second resident coefficients");
        let first_jobs = [CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset: 0,
            width: 2,
            height: 2,
            total_bitplanes: 1,
            target_coding_passes: 1,
        }];
        let second_jobs = [CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset: 0,
            width: 2,
            height: 1,
            total_bitplanes: 1,
            target_coding_passes: 1,
        }];

        let first_separate = context
            .encode_htj2k_codeblocks_resident_with_resources_and_pool(
                &first,
                4,
                &first_jobs,
                &resources,
                &pool,
            )
            .expect("first separate resident encode");
        let second_separate = context
            .encode_htj2k_codeblocks_resident_with_resources_and_pool(
                &second,
                2,
                &second_jobs,
                &resources,
                &pool,
            )
            .expect("second separate resident encode");

        let combined = context
            .encode_htj2k_codeblocks_multi_resident_with_resources_and_pool(
                &[
                    CudaHtj2kEncodeResidentTarget {
                        coefficients: &first,
                        coefficient_count: 4,
                        jobs: &first_jobs,
                    },
                    CudaHtj2kEncodeResidentTarget {
                        coefficients: &second,
                        coefficient_count: 2,
                        jobs: &second_jobs,
                    },
                ],
                &resources,
                &pool,
            )
            .expect("combined resident encode");

        assert_eq!(combined.execution().kernel_dispatches(), 1);
        assert_eq!(combined.code_blocks().len(), 2);
        assert_eq!(
            combined.code_blocks()[0].data(),
            first_separate.code_blocks()[0].data()
        );
        assert_eq!(
            combined.code_blocks()[1].data(),
            second_separate.code_blocks()[0].data()
        );
        let timings = combined.stage_timings();
        assert_eq!(
            timings.ht_encode_us,
            timings
                .ht_kernel_us
                .saturating_add(timings.ht_status_readback_us)
                .saturating_add(timings.ht_compact_us)
                .saturating_add(timings.ht_output_readback_us)
        );
        assert!(timings.ht_kernel_us > 0);
        assert!(timings.ht_status_readback_us > 0);
    }

    #[test]
    fn htj2k97_resident_batch_returns_pooled_quantized_bands_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let blocks = vec![0.0f32; 64];
        let params = super::CudaHtj2k97QuantizeParams {
            inv_delta_ll: 1.0,
            inv_delta_hl: 1.0,
            inv_delta_lh: 1.0,
            inv_delta_hh: 1.0,
            cb_width: 64,
            cb_height: 64,
        };

        let (bands, _) = context
            .j2k_transcode_htj2k97_codeblock_batch_resident_with_pool(
                &blocks, 1, 1, 1, 8, 8, params, &pool,
            )
            .expect("resident HTJ2K 9/7 codeblock batch");

        assert!(bands.ll.as_device_buffer().is_some());
        assert!(bands.hl.as_device_buffer().is_some());
        assert!(bands.lh.as_device_buffer().is_some());
        assert!(bands.hh.as_device_buffer().is_some());
        let cached_while_bands_live = pool.cached_count().expect("cached buffers while live");

        drop(bands);

        assert!(
            pool.cached_count().expect("cached buffers after drop") >= cached_while_bands_live + 4
        );
    }

    #[test]
    fn htj2k_encode_rejects_unsupported_refinement_pass_count_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let coefficients = [0, 2, -3, 1];
        let jobs = [CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset: 0,
            width: 2,
            height: 2,
            total_bitplanes: 3,
            target_coding_passes: 4,
        }];

        let error = context
            .encode_htj2k_codeblocks(
                &coefficients,
                &jobs,
                CudaHtj2kEncodeTables {
                    vlc_table0: &[0u16; 2048],
                    vlc_table1: &[0u16; 2048],
                    uvlc_table: &[0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES],
                },
            )
            .expect_err("unsupported HTJ2K encode pass count is explicit");

        match error {
            CudaError::KernelStatus {
                kernel,
                code,
                detail,
            } => {
                assert_eq!(kernel, "signinum_htj2k_encode_codeblocks");
                assert_eq!(code, super::HTJ2K_STATUS_UNSUPPORTED);
                assert_eq!(detail, 5);
            }
            other => panic!("unexpected CUDA encode error: {other:?}"),
        }
    }

    #[test]
    fn htj2k_encode_rejects_lossy_zero_sigprop_request_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let coefficients = [0, 2, -3, 4];
        let jobs = [CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset: 0,
            width: 2,
            height: 2,
            total_bitplanes: 3,
            target_coding_passes: 2,
        }];

        let error = context
            .encode_htj2k_codeblocks(
                &coefficients,
                &jobs,
                CudaHtj2kEncodeTables {
                    vlc_table0: &[0u16; 2048],
                    vlc_table1: &[0u16; 2048],
                    uvlc_table: &[0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES],
                },
            )
            .expect_err("target-2 zero SigProp cannot silently drop low coefficient bits");

        match error {
            CudaError::KernelStatus {
                kernel,
                code,
                detail,
            } => {
                assert_eq!(kernel, "signinum_htj2k_encode_codeblocks");
                assert_eq!(code, super::HTJ2K_STATUS_UNSUPPORTED);
                assert_eq!(detail, 6);
            }
            other => panic!("unexpected CUDA encode error: {other:?}"),
        }
    }

    #[test]
    fn htj2k_encode_rejects_unreachable_target_three_sigprop_coefficients_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let coefficients = [3, 0, 0, 0];
        let jobs = [CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset: 0,
            width: 2,
            height: 2,
            total_bitplanes: 4,
            target_coding_passes: 3,
        }];

        let error = context
            .encode_htj2k_codeblocks(
                &coefficients,
                &jobs,
                CudaHtj2kEncodeTables {
                    vlc_table0: &[0u16; 2048],
                    vlc_table1: &[0u16; 2048],
                    uvlc_table: &[0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES],
                },
            )
            .expect_err("isolated target-3 SigProp coefficient is explicitly unsupported");

        match error {
            CudaError::KernelStatus {
                kernel,
                code,
                detail,
            } => {
                assert_eq!(kernel, "signinum_htj2k_encode_codeblocks");
                assert_eq!(code, super::HTJ2K_STATUS_UNSUPPORTED);
                assert_eq!(detail, 6);
            }
            other => panic!("unexpected CUDA encode error: {other:?}"),
        }
    }

    #[test]
    fn htj2k_encode_resources_feed_single_codeblock_encode_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let vlc_table0 = [0u16; 2048];
        let vlc_table1 = [0u16; 2048];
        let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
        let resources = context
            .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
                vlc_table0: &vlc_table0,
                vlc_table1: &vlc_table1,
                uvlc_table: &uvlc_table,
            })
            .expect("encode resources");

        let encoded = context
            .encode_htj2k_codeblock_with_resources(&[0, 0, 0, 0], 2, 2, 1, &resources)
            .expect("resource-backed single HTJ2K encode");

        assert_eq!(encoded.execution().kernel_dispatches(), 1);
        // An all-zero codeblock has no significant bitplanes, so the encoder emits zero
        // coding passes (matching native ht_block_encode::encode_code_block).
        assert_eq!(encoded.num_coding_passes(), 0);
        assert_eq!(encoded.cleanup_length(), 0);
        assert_eq!(encoded.data().len(), 0);
        assert_eq!(encoded.refinement_length(), 0);
    }

    #[test]
    fn default_stream_timer_reports_elapsed_time_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let input = vec![17u8; 4096];
        let (output, elapsed_us) = context
            .time_default_stream_us(|| context.copy_with_kernel(&input))
            .expect("timed CUDA copy kernel");

        assert_eq!(output.execution().kernel_dispatches(), 1);
        assert!(elapsed_us > 0);
    }

    #[test]
    fn named_default_stream_timer_is_available_for_profiling_ranges_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let input = vec![23u8; 4096];
        let (output, elapsed_us) = context
            .time_default_stream_named_us("signinum.test.copy", || context.copy_with_kernel(&input))
            .expect("named timed CUDA copy kernel");

        assert_eq!(output.execution().kernel_dispatches(), 1);
        assert!(elapsed_us > 0);
    }

    #[test]
    fn typed_device_view_reports_element_count_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let mut aligned = context.allocate(16).expect("aligned buffer");
        let view = aligned.typed_view::<u32>().expect("typed immutable view");
        assert_eq!(view.len(), 4);
        let mut_view = aligned.typed_view_mut::<u64>().expect("typed mutable view");
        assert_eq!(mut_view.len(), 2);

        let unaligned = context.allocate(3).expect("unaligned buffer");
        let error = unaligned
            .typed_view::<u16>()
            .expect_err("unaligned typed view");
        assert!(matches!(
            error,
            CudaError::LengthNotElementAligned {
                bytes: 3,
                element_size: 2
            }
        ));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn kernel_module_names_cover_htj2k_decode_and_encode_stages() {
        let cases = [
            (
                CudaKernelName::Htj2kDecodeCodeblocks,
                "signinum_htj2k_decode_codeblocks",
            ),
            (
                CudaKernelName::Htj2kDecodeCodeblocksMultiCleanupDequantize,
                "signinum_htj2k_decode_codeblocks_multi_cleanup_dequantize",
            ),
            (
                CudaKernelName::J2kDequantizeHtj2kCodeblocks,
                "signinum_j2k_dequantize_htj2k_codeblocks",
            ),
            (
                CudaKernelName::J2kDequantizeHtj2kCodeblocksMulti,
                "signinum_j2k_dequantize_htj2k_codeblocks_multi",
            ),
            (
                CudaKernelName::J2kDequantizeHtj2kCleanupJobsMulti,
                "signinum_j2k_dequantize_htj2k_cleanup_jobs_multi",
            ),
            (
                CudaKernelName::J2kIdwtInterleave,
                "signinum_j2k_idwt_interleave",
            ),
            (
                CudaKernelName::J2kIdwtInterleaveHorizontal53Multi,
                "signinum_j2k_idwt_interleave_horizontal_53_multi",
            ),
            (
                CudaKernelName::J2kIdwtInterleaveHorizontal97Multi,
                "signinum_j2k_idwt_interleave_horizontal_97_multi",
            ),
            (
                CudaKernelName::J2kIdwtHorizontal,
                "signinum_j2k_idwt_horizontal",
            ),
            (
                CudaKernelName::J2kIdwtHorizontal53,
                "signinum_j2k_idwt_horizontal_53",
            ),
            (
                CudaKernelName::J2kIdwtHorizontal97,
                "signinum_j2k_idwt_horizontal_97",
            ),
            (
                CudaKernelName::J2kIdwtVertical,
                "signinum_j2k_idwt_vertical",
            ),
            (
                CudaKernelName::J2kIdwtVertical53Multi,
                "signinum_j2k_idwt_vertical_53_multi",
            ),
            (
                CudaKernelName::J2kIdwtVertical97Multi,
                "signinum_j2k_idwt_vertical_97_multi",
            ),
            (
                CudaKernelName::J2kIdwtVertical97MultiCols4,
                "signinum_j2k_idwt_vertical_97_multi_cols4",
            ),
            (
                CudaKernelName::J2kIdwtVertical53,
                "signinum_j2k_idwt_vertical_53",
            ),
            (
                CudaKernelName::J2kIdwtVertical97,
                "signinum_j2k_idwt_vertical_97",
            ),
            (
                CudaKernelName::J2kInverseDwtSingle,
                "signinum_j2k_inverse_dwt_single",
            ),
            (CudaKernelName::J2kInverseMct, "signinum_j2k_inverse_mct"),
            (CudaKernelName::J2kStoreGray8, "signinum_j2k_store_gray8"),
            (CudaKernelName::J2kStoreGray16, "signinum_j2k_store_gray16"),
            (CudaKernelName::J2kStoreRgb8, "signinum_j2k_store_rgb8"),
            (
                CudaKernelName::J2kStoreRgb8Mct,
                "signinum_j2k_store_rgb8_mct",
            ),
            (
                CudaKernelName::J2kStoreRgb8MctBatch,
                "signinum_j2k_store_rgb8_mct_batch",
            ),
            (CudaKernelName::J2kStoreRgb16, "signinum_j2k_store_rgb16"),
            (
                CudaKernelName::J2kStoreRgb16Mct,
                "signinum_j2k_store_rgb16_mct",
            ),
            (
                CudaKernelName::Htj2kEncodeCodeblock,
                "signinum_htj2k_encode_codeblock",
            ),
            (
                CudaKernelName::Htj2kEncodeCodeblocks,
                "signinum_htj2k_encode_codeblocks",
            ),
            (
                CudaKernelName::Htj2kEncodeCodeblocksMultiInput,
                "signinum_htj2k_encode_codeblocks_multi_input",
            ),
            (
                CudaKernelName::Htj2kEncodeCodeblocksMultiInputCleanup,
                "signinum_htj2k_encode_codeblocks_multi_input_cleanup",
            ),
            (
                CudaKernelName::Htj2kEncodeCodeblocksMultiInputCleanup64,
                "signinum_htj2k_encode_codeblocks_multi_input_cleanup_64",
            ),
            (
                CudaKernelName::Htj2kCompactCodeblocks,
                "signinum_htj2k_compact_codeblocks",
            ),
            (
                CudaKernelName::Htj2kPacketizeCleanup,
                "signinum_htj2k_packetize_cleanup",
            ),
        ];

        for (kernel, entrypoint) in cases {
            assert_eq!(kernel.entrypoint(), entrypoint);
            let raw_entrypoint = kernel.kernel().entrypoint();
            assert_eq!(
                &raw_entrypoint[..raw_entrypoint.len() - 1],
                entrypoint.as_bytes()
            );
            assert_eq!(raw_entrypoint.last(), Some(&0));
        }
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn htj2k_empty_codeblock_decode_zero_fills_coefficients_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let first_vlc = [0u16; 1024];
        let later_vlc = [0u16; 1024];
        let first_uvlc = [0u16; 320];
        let later_uvlc = [0u16; 256];
        let output = context
            .decode_htj2k_codeblocks(
                &[],
                &[],
                CudaHtj2kDecodeTables {
                    vlc_table0: &first_vlc,
                    vlc_table1: &later_vlc,
                    uvlc_table0: &first_uvlc,
                    uvlc_table1: &later_uvlc,
                },
                8,
            )
            .expect("empty HTJ2K decode");
        let mut actual = vec![f32::NAN; 8];
        output
            .coefficients()
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut actual))
            .expect("download coefficients");

        assert_eq!(actual, vec![0.0; 8]);
        assert_eq!(output.execution().kernel_dispatches(), 0);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn htj2k_empty_codeblock_decode_reuses_pool_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let first_vlc = [0u16; 1024];
        let later_vlc = [0u16; 1024];
        let first_uvlc = [0u16; 320];
        let later_uvlc = [0u16; 256];
        let tables = context
            .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
                vlc_table0: &first_vlc,
                vlc_table1: &later_vlc,
                uvlc_table0: &first_uvlc,
                uvlc_table1: &later_uvlc,
            })
            .expect("decode tables");
        let resources = context
            .upload_htj2k_decode_resources_with_tables(&[], &tables)
            .expect("decode resources");

        let output = context
            .decode_htj2k_codeblocks_with_resources_and_pool(&resources, &[], 8, &pool)
            .expect("pooled empty HTJ2K decode");
        let mut actual = vec![f32::NAN; 8];
        output
            .coefficients()
            .expect("pooled coefficients")
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut actual))
            .expect("download coefficients");

        assert_eq!(actual, vec![0.0; 8]);
        assert_eq!(output.execution().kernel_dispatches(), 0);
        let cached_while_live = pool.cached_count().expect("cached while live");

        drop(output);

        assert!(pool.cached_count().expect("cached after drop") > cached_while_live);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn htj2k_decode_table_resources_feed_multiple_payload_uploads_when_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let first_vlc = [0u16; 1024];
        let later_vlc = [0u16; 1024];
        let first_uvlc = [0u16; 320];
        let later_uvlc = [0u16; 256];
        let tables = context
            .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
                vlc_table0: &first_vlc,
                vlc_table1: &later_vlc,
                uvlc_table0: &first_uvlc,
                uvlc_table1: &later_uvlc,
            })
            .expect("decode table resources");

        let first_resources = context
            .upload_htj2k_decode_resources_with_tables(&[0xAA, 0x55], &tables)
            .expect("first payload resources");
        let second_resources = context
            .upload_htj2k_decode_resources_with_tables(&[0x11, 0x22, 0x33], &tables)
            .expect("second payload resources");

        assert!(std::sync::Arc::ptr_eq(
            &first_resources.tables.inner,
            &second_resources.tables.inner
        ));
        assert_eq!(first_resources.payload_len, 2);
        assert_eq!(second_resources.payload_len, 3);
    }

    #[test]
    fn j2k_inverse_dwt_single_dispatches_parallel_stages_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let ll = context
            .upload(super::f32_slice_as_bytes(&[10.0]))
            .expect("upload LL");
        let hl = context
            .upload(super::f32_slice_as_bytes(&[2.0]))
            .expect("upload HL");
        let lh = context
            .upload(super::f32_slice_as_bytes(&[4.0]))
            .expect("upload LH");
        let hh = context
            .upload(super::f32_slice_as_bytes(&[1.0]))
            .expect("upload HH");

        let output = context
            .j2k_inverse_dwt_single_device(
                &ll,
                &hl,
                &lh,
                &hh,
                CudaJ2kIdwtJob {
                    rect: CudaJ2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 2,
                        y1: 2,
                    },
                    ll_rect: CudaJ2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 1,
                        y1: 1,
                    },
                    hl_rect: CudaJ2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 1,
                        y1: 1,
                    },
                    lh_rect: CudaJ2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 1,
                        y1: 1,
                    },
                    hh_rect: CudaJ2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 1,
                        y1: 1,
                    },
                    irreversible97: 0,
                },
            )
            .expect("CUDA inverse DWT");

        assert_eq!(output.execution().kernel_dispatches(), 3);
        let mut actual = vec![0.0f32; 4];
        output
            .buffer()
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut actual))
            .expect("download inverse DWT");
        assert_eq!(actual, vec![7.0, 9.0, 10.0, 13.0]);
    }

    #[test]
    fn j2k_inverse_dwt_single_reuses_pool_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let ll = context
            .upload(super::f32_slice_as_bytes(&[10.0]))
            .expect("upload LL");
        let hl = context
            .upload(super::f32_slice_as_bytes(&[2.0]))
            .expect("upload HL");
        let lh = context
            .upload(super::f32_slice_as_bytes(&[4.0]))
            .expect("upload LH");
        let hh = context
            .upload(super::f32_slice_as_bytes(&[1.0]))
            .expect("upload HH");

        let output = context
            .j2k_inverse_dwt_single_device_with_pool(
                &ll,
                &hl,
                &lh,
                &hh,
                CudaJ2kIdwtJob {
                    rect: CudaJ2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 2,
                        y1: 2,
                    },
                    ll_rect: CudaJ2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 1,
                        y1: 1,
                    },
                    hl_rect: CudaJ2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 1,
                        y1: 1,
                    },
                    lh_rect: CudaJ2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 1,
                        y1: 1,
                    },
                    hh_rect: CudaJ2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 1,
                        y1: 1,
                    },
                    irreversible97: 0,
                },
                &pool,
            )
            .expect("pooled CUDA inverse DWT");

        assert_eq!(output.execution().kernel_dispatches(), 3);
        let cached_while_live = pool.cached_count().expect("cached while live");

        drop(output);

        assert!(pool.cached_count().expect("cached after drop") > cached_while_live);
    }

    #[test]
    fn idwt_cooperative_53_selection_requires_large_reversible_batches() {
        let mut kernel_job = CudaJ2kIdwtMultiKernelJob {
            ll_ptr: 0,
            hl_ptr: 0,
            lh_ptr: 0,
            hh_ptr: 0,
            output_ptr: 0,
            job: CudaJ2kIdwtJob {
                rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 0,
                    y1: 0,
                },
                ll_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 0,
                    y1: 0,
                },
                hl_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 0,
                    y1: 0,
                },
                lh_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 0,
                    y1: 0,
                },
                hh_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 0,
                    y1: 0,
                },
                irreversible97: 0,
            },
        };

        assert!(!idwt_batch_uses_cooperative_53(&[kernel_job], 127, 128));
        assert!(!idwt_batch_uses_cooperative_53(&[kernel_job], 128, 127));
        assert!(idwt_batch_uses_cooperative_53(&[kernel_job], 128, 128));
        assert!(idwt_batch_uses_cooperative_53(&[kernel_job], 512, 512));
        assert!(!idwt_batch_uses_cooperative_53(&[kernel_job], 513, 128));
        kernel_job.job.irreversible97 = 1;
        assert!(!idwt_batch_uses_cooperative_53(&[kernel_job], 128, 128));
    }

    #[test]
    fn idwt_cooperative_97_selection_requires_large_irreversible_batches() {
        let mut kernel_job = CudaJ2kIdwtMultiKernelJob {
            ll_ptr: 0,
            hl_ptr: 0,
            lh_ptr: 0,
            hh_ptr: 0,
            output_ptr: 0,
            job: CudaJ2kIdwtJob {
                rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 0,
                    y1: 0,
                },
                ll_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 0,
                    y1: 0,
                },
                hl_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 0,
                    y1: 0,
                },
                lh_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 0,
                    y1: 0,
                },
                hh_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 0,
                    y1: 0,
                },
                irreversible97: 1,
            },
        };

        assert_eq!(
            idwt_batch_kernel_mode(&[kernel_job], 128, 128),
            CudaJ2kIdwtBatchKernelMode::Cooperative97
        );
        assert_eq!(
            idwt_batch_kernel_mode(&[kernel_job], 64, 64),
            CudaJ2kIdwtBatchKernelMode::Cooperative97
        );
        assert_eq!(
            idwt_batch_kernel_mode(&[kernel_job], 512, 512),
            CudaJ2kIdwtBatchKernelMode::Cooperative97
        );
        assert_eq!(
            idwt_batch_kernel_mode(&[kernel_job], 63, 64),
            CudaJ2kIdwtBatchKernelMode::Generic
        );
        assert_eq!(
            idwt_batch_kernel_mode(&[kernel_job], 513, 128),
            CudaJ2kIdwtBatchKernelMode::Generic
        );
        kernel_job.job.irreversible97 = 0;
        assert_ne!(
            idwt_batch_kernel_mode(&[kernel_job], 128, 128),
            CudaJ2kIdwtBatchKernelMode::Cooperative97
        );
    }

    #[test]
    fn idwt_batch_trace_row_reports_stage_shape_and_mode() {
        let kernel_jobs = [
            CudaJ2kIdwtMultiKernelJob {
                ll_ptr: 0,
                hl_ptr: 0,
                lh_ptr: 0,
                hh_ptr: 0,
                output_ptr: 0,
                job: CudaJ2kIdwtJob {
                    rect: CudaJ2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 128,
                        y1: 96,
                    },
                    ll_rect: CudaJ2kRect::default(),
                    hl_rect: CudaJ2kRect::default(),
                    lh_rect: CudaJ2kRect::default(),
                    hh_rect: CudaJ2kRect::default(),
                    irreversible97: 1,
                },
            },
            CudaJ2kIdwtMultiKernelJob {
                ll_ptr: 0,
                hl_ptr: 0,
                lh_ptr: 0,
                hh_ptr: 0,
                output_ptr: 0,
                job: CudaJ2kIdwtJob {
                    rect: CudaJ2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 64,
                        y1: 48,
                    },
                    ll_rect: CudaJ2kRect::default(),
                    hl_rect: CudaJ2kRect::default(),
                    lh_rect: CudaJ2kRect::default(),
                    hh_rect: CudaJ2kRect::default(),
                    irreversible97: 1,
                },
            },
        ];

        let row = idwt_batch_trace_row(
            3,
            &kernel_jobs,
            128,
            96,
            CudaJ2kIdwtBatchKernelMode::Cooperative97,
            42,
        );

        assert_eq!(
            format_idwt_batch_trace_row(row),
            "signinum_profile codec=j2k op=cuda_idwt_batch path=decode stage_index=3 mode=Cooperative97 job_count=2 max_width=128 max_height=96 min_width=64 min_height=48 total_pixels=15360 irreversible_jobs=2 elapsed_us=42"
        );
    }

    #[test]
    fn j2k_inverse_dwt_batch_empty_uses_no_dispatch_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let execution = context
            .j2k_inverse_dwt_batch_device_with_pool(&[] as &[CudaJ2kIdwtTarget<'_>], &pool)
            .expect("empty batched CUDA inverse DWT");

        assert_eq!(execution.kernel_dispatches(), 0);
        assert_eq!(execution.decode_kernel_dispatches(), 0);
    }

    #[test]
    fn j2k_inverse_dwt_batch_matches_expected_outputs_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let ll = context
            .upload(super::f32_slice_as_bytes(&[10.0]))
            .expect("upload LL");
        let hl = context
            .upload(super::f32_slice_as_bytes(&[2.0]))
            .expect("upload HL");
        let lh = context
            .upload(super::f32_slice_as_bytes(&[4.0]))
            .expect("upload LH");
        let hh = context
            .upload(super::f32_slice_as_bytes(&[1.0]))
            .expect("upload HH");
        let first_output = pool
            .take(4 * std::mem::size_of::<f32>())
            .expect("first batched IDWT output");
        let second_output = pool
            .take(4 * std::mem::size_of::<f32>())
            .expect("second batched IDWT output");
        let job = CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 2,
                y1: 2,
            },
            ll_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            hl_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            lh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            hh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            irreversible97: 0,
        };

        let execution = context
            .j2k_inverse_dwt_batch_device_with_pool(
                &[
                    CudaJ2kIdwtTarget {
                        ll: &ll,
                        hl: &hl,
                        lh: &lh,
                        hh: &hh,
                        output: first_output
                            .as_device_buffer()
                            .expect("first output device buffer"),
                        job,
                    },
                    CudaJ2kIdwtTarget {
                        ll: &ll,
                        hl: &hl,
                        lh: &lh,
                        hh: &hh,
                        output: second_output
                            .as_device_buffer()
                            .expect("second output device buffer"),
                        job,
                    },
                ],
                &pool,
            )
            .expect("batched CUDA inverse DWT");
        assert_eq!(execution.kernel_dispatches(), 2);

        let mut first_actual = vec![0.0f32; 4];
        first_output
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut first_actual))
            .expect("download first batched IDWT");
        assert_eq!(first_actual, vec![7.0, 9.0, 10.0, 13.0]);
        let mut second_actual = vec![0.0f32; 4];
        second_output
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut second_actual))
            .expect("download second batched IDWT");
        assert_eq!(second_actual, vec![7.0, 9.0, 10.0, 13.0]);
    }

    #[test]
    fn j2k_inverse_dwt_batch_odd_origin_matches_single_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let ll = context
            .upload(super::f32_slice_as_bytes(&[10.0]))
            .expect("upload odd LL");
        let hl = context
            .upload(super::f32_slice_as_bytes(&[2.0, 5.0]))
            .expect("upload odd HL");
        let lh = context
            .upload(super::f32_slice_as_bytes(&[4.0, 7.0]))
            .expect("upload odd LH");
        let hh = context
            .upload(super::f32_slice_as_bytes(&[1.0, 3.0, 6.0, 8.0]))
            .expect("upload odd HH");
        let job = CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 1,
                y0: 1,
                x1: 4,
                y1: 4,
            },
            ll_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            hl_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 2,
                y1: 1,
            },
            lh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 2,
            },
            hh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 2,
                y1: 2,
            },
            irreversible97: 0,
        };

        let single = context
            .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
            .expect("single CUDA inverse DWT");
        assert_eq!(single.execution().kernel_dispatches(), 3);
        let batch_output = pool
            .take(9 * std::mem::size_of::<f32>())
            .expect("odd batched IDWT output");
        let execution = context
            .j2k_inverse_dwt_batch_device_with_pool(
                &[CudaJ2kIdwtTarget {
                    ll: &ll,
                    hl: &hl,
                    lh: &lh,
                    hh: &hh,
                    output: batch_output
                        .as_device_buffer()
                        .expect("odd batch output device buffer"),
                    job,
                }],
                &pool,
            )
            .expect("odd-origin batched CUDA inverse DWT");
        assert_eq!(execution.kernel_dispatches(), 2);

        let mut single_actual = vec![0.0f32; 9];
        single
            .buffer()
            .expect("single odd output device buffer")
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut single_actual))
            .expect("download single odd IDWT");
        let mut batch_actual = vec![0.0f32; 9];
        batch_output
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut batch_actual))
            .expect("download batch odd IDWT");
        assert_eq!(batch_actual, single_actual);
    }

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::similar_names)]
    fn j2k_inverse_dwt_batch_large_reversible_matches_single_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let band_len = 64 * 64;
        let ll_values: Vec<f32> = (0..band_len).map(|idx| (idx % 19) as f32).collect();
        let hl_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 3) % 23) as f32).collect();
        let lh_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 5) % 29) as f32).collect();
        let hh_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 7) % 31) as f32).collect();
        let ll = context
            .upload(super::f32_slice_as_bytes(&ll_values))
            .expect("upload large LL");
        let hl = context
            .upload(super::f32_slice_as_bytes(&hl_values))
            .expect("upload large HL");
        let lh = context
            .upload(super::f32_slice_as_bytes(&lh_values))
            .expect("upload large LH");
        let hh = context
            .upload(super::f32_slice_as_bytes(&hh_values))
            .expect("upload large HH");
        let job = CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 128,
                y1: 128,
            },
            ll_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 64,
                y1: 64,
            },
            hl_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 64,
                y1: 64,
            },
            lh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 64,
                y1: 64,
            },
            hh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 64,
                y1: 64,
            },
            irreversible97: 0,
        };

        let single = context
            .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
            .expect("large single CUDA inverse DWT");
        let batch_output = pool
            .take(128 * 128 * std::mem::size_of::<f32>())
            .expect("large batched IDWT output");
        let execution = context
            .j2k_inverse_dwt_batch_device_with_pool(
                &[CudaJ2kIdwtTarget {
                    ll: &ll,
                    hl: &hl,
                    lh: &lh,
                    hh: &hh,
                    output: batch_output
                        .as_device_buffer()
                        .expect("large batch output device buffer"),
                    job,
                }],
                &pool,
            )
            .expect("large batched CUDA inverse DWT");
        assert_eq!(execution.kernel_dispatches(), 2);

        let mut single_actual = vec![0.0f32; 128 * 128];
        single
            .buffer()
            .expect("large single output device buffer")
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut single_actual))
            .expect("download large single IDWT");
        let mut batch_actual = vec![0.0f32; 128 * 128];
        batch_output
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut batch_actual))
            .expect("download large batch IDWT");
        assert_eq!(batch_actual, single_actual);
    }

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::similar_names)]
    fn j2k_inverse_dwt_batch_large_irreversible_matches_single_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let band_len = 128 * 128;
        let ll_values: Vec<f32> = (0..band_len)
            .map(|idx| ((idx % 43) as f32) * 0.25)
            .collect();
        let hl_values: Vec<f32> = (0..band_len)
            .map(|idx| (((idx * 3) % 47) as f32) * 0.125)
            .collect();
        let lh_values: Vec<f32> = (0..band_len)
            .map(|idx| (((idx * 5) % 53) as f32) * 0.0625)
            .collect();
        let hh_values: Vec<f32> = (0..band_len)
            .map(|idx| (((idx * 7) % 59) as f32) * 0.03125)
            .collect();
        let ll = context
            .upload(super::f32_slice_as_bytes(&ll_values))
            .expect("upload large irreversible LL");
        let hl = context
            .upload(super::f32_slice_as_bytes(&hl_values))
            .expect("upload large irreversible HL");
        let lh = context
            .upload(super::f32_slice_as_bytes(&lh_values))
            .expect("upload large irreversible LH");
        let hh = context
            .upload(super::f32_slice_as_bytes(&hh_values))
            .expect("upload large irreversible HH");
        let job = CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 256,
                y1: 256,
            },
            ll_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 128,
                y1: 128,
            },
            hl_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 128,
                y1: 128,
            },
            lh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 128,
                y1: 128,
            },
            hh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 128,
                y1: 128,
            },
            irreversible97: 1,
        };

        let single = context
            .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
            .expect("large irreversible single CUDA inverse DWT");
        let batch_output = pool
            .take(256 * 256 * std::mem::size_of::<f32>())
            .expect("large irreversible batched IDWT output");
        let execution = context
            .j2k_inverse_dwt_batch_device_with_pool(
                &[CudaJ2kIdwtTarget {
                    ll: &ll,
                    hl: &hl,
                    lh: &lh,
                    hh: &hh,
                    output: batch_output
                        .as_device_buffer()
                        .expect("large irreversible batch output device buffer"),
                    job,
                }],
                &pool,
            )
            .expect("large irreversible batched CUDA inverse DWT");
        assert_eq!(execution.kernel_dispatches(), 2);

        let mut single_actual = vec![0.0f32; 256 * 256];
        single
            .buffer()
            .expect("large irreversible single output device buffer")
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut single_actual))
            .expect("download large irreversible single IDWT");
        let mut batch_actual = vec![0.0f32; 256 * 256];
        batch_output
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut batch_actual))
            .expect("download large irreversible batch IDWT");
        assert_eq!(batch_actual, single_actual);
    }

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::similar_names)]
    fn j2k_inverse_dwt_batch_512_reversible_matches_single_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let band_len = 256 * 256;
        let ll_values: Vec<f32> = (0..band_len).map(|idx| (idx % 43) as f32).collect();
        let hl_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 3) % 47) as f32).collect();
        let lh_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 5) % 53) as f32).collect();
        let hh_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 7) % 59) as f32).collect();
        let ll = context
            .upload(super::f32_slice_as_bytes(&ll_values))
            .expect("upload 512 LL");
        let hl = context
            .upload(super::f32_slice_as_bytes(&hl_values))
            .expect("upload 512 HL");
        let lh = context
            .upload(super::f32_slice_as_bytes(&lh_values))
            .expect("upload 512 LH");
        let hh = context
            .upload(super::f32_slice_as_bytes(&hh_values))
            .expect("upload 512 HH");
        let job = CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 512,
                y1: 512,
            },
            ll_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 256,
                y1: 256,
            },
            hl_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 256,
                y1: 256,
            },
            lh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 256,
                y1: 256,
            },
            hh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 256,
                y1: 256,
            },
            irreversible97: 0,
        };

        let single = context
            .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
            .expect("512 single CUDA inverse DWT");
        let batch_output = pool
            .take(512 * 512 * std::mem::size_of::<f32>())
            .expect("512 batched IDWT output");
        let execution = context
            .j2k_inverse_dwt_batch_device_with_pool(
                &[CudaJ2kIdwtTarget {
                    ll: &ll,
                    hl: &hl,
                    lh: &lh,
                    hh: &hh,
                    output: batch_output
                        .as_device_buffer()
                        .expect("512 batch output device buffer"),
                    job,
                }],
                &pool,
            )
            .expect("512 batched CUDA inverse DWT");
        assert_eq!(execution.kernel_dispatches(), 2);

        let mut single_actual = vec![0.0f32; 512 * 512];
        single
            .buffer()
            .expect("512 single output device buffer")
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut single_actual))
            .expect("download 512 single IDWT");
        let mut batch_actual = vec![0.0f32; 512 * 512];
        batch_output
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut batch_actual))
            .expect("download 512 batch IDWT");
        assert_eq!(batch_actual, single_actual);
    }

    #[test]
    fn j2k_inverse_dwt_batch_enqueue_matches_expected_outputs_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let ll = context
            .upload(super::f32_slice_as_bytes(&[10.0]))
            .expect("upload LL");
        let hl = context
            .upload(super::f32_slice_as_bytes(&[2.0]))
            .expect("upload HL");
        let lh = context
            .upload(super::f32_slice_as_bytes(&[4.0]))
            .expect("upload LH");
        let hh = context
            .upload(super::f32_slice_as_bytes(&[1.0]))
            .expect("upload HH");
        let output = pool
            .take(4 * std::mem::size_of::<f32>())
            .expect("batched IDWT output");
        let job = CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 2,
                y1: 2,
            },
            ll_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            hl_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            lh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            hh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            irreversible97: 0,
        };

        let queued = context
            .j2k_inverse_dwt_batch_device_enqueue_with_pool(
                &[CudaJ2kIdwtTarget {
                    ll: &ll,
                    hl: &hl,
                    lh: &lh,
                    hh: &hh,
                    output: output.as_device_buffer().expect("output device buffer"),
                    job,
                }],
                &pool,
            )
            .expect("enqueue batched CUDA inverse DWT");
        assert_eq!(queued.execution().kernel_dispatches(), 2);
        context.synchronize().expect("queued IDWT completion");
        drop(queued);

        let mut actual = vec![0.0f32; 4];
        output
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut actual))
            .expect("download queued batched IDWT");
        assert_eq!(actual, vec![7.0, 9.0, 10.0, 13.0]);
    }

    #[test]
    #[allow(clippy::similar_names, clippy::too_many_lines)]
    fn j2k_inverse_dwt_batch_sequence_enqueue_matches_two_stage_path_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let ll = context
            .upload(super::f32_slice_as_bytes(&[10.0]))
            .expect("upload LL");
        let hl = context
            .upload(super::f32_slice_as_bytes(&[2.0]))
            .expect("upload HL");
        let lh = context
            .upload(super::f32_slice_as_bytes(&[4.0]))
            .expect("upload LH");
        let hh = context
            .upload(super::f32_slice_as_bytes(&[1.0]))
            .expect("upload HH");
        let stage2_hl = context
            .upload(super::f32_slice_as_bytes(&[0.0, 1.0, 2.0, 3.0]))
            .expect("upload stage2 HL");
        let stage2_lh = context
            .upload(super::f32_slice_as_bytes(&[4.0, 5.0, 6.0, 7.0]))
            .expect("upload stage2 LH");
        let stage2_hh = context
            .upload(super::f32_slice_as_bytes(&[8.0, 9.0, 10.0, 11.0]))
            .expect("upload stage2 HH");
        let stage1_job = CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 2,
                y1: 2,
            },
            ll_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            hl_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            lh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            hh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            irreversible97: 0,
        };
        let stage2_job = CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 4,
                y1: 4,
            },
            ll_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 2,
                y1: 2,
            },
            hl_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 2,
                y1: 2,
            },
            lh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 2,
                y1: 2,
            },
            hh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 2,
                y1: 2,
            },
            irreversible97: 0,
        };
        let legacy_stage1 = pool
            .take(4 * std::mem::size_of::<f32>())
            .expect("legacy stage1 output");
        let legacy_stage2 = pool
            .take(16 * std::mem::size_of::<f32>())
            .expect("legacy stage2 output");
        let sequence_stage1 = pool
            .take(4 * std::mem::size_of::<f32>())
            .expect("sequence stage1 output");
        let sequence_stage2 = pool
            .take(16 * std::mem::size_of::<f32>())
            .expect("sequence stage2 output");

        context
            .j2k_inverse_dwt_batch_device_with_pool(
                &[CudaJ2kIdwtTarget {
                    ll: &ll,
                    hl: &hl,
                    lh: &lh,
                    hh: &hh,
                    output: legacy_stage1
                        .as_device_buffer()
                        .expect("legacy stage1 device buffer"),
                    job: stage1_job,
                }],
                &pool,
            )
            .expect("legacy stage1 IDWT");
        context
            .j2k_inverse_dwt_batch_device_with_pool(
                &[CudaJ2kIdwtTarget {
                    ll: legacy_stage1
                        .as_device_buffer()
                        .expect("legacy stage1 device buffer"),
                    hl: &stage2_hl,
                    lh: &stage2_lh,
                    hh: &stage2_hh,
                    output: legacy_stage2
                        .as_device_buffer()
                        .expect("legacy stage2 device buffer"),
                    job: stage2_job,
                }],
                &pool,
            )
            .expect("legacy stage2 IDWT");

        let sequence_stage1_targets = [CudaJ2kIdwtTarget {
            ll: &ll,
            hl: &hl,
            lh: &lh,
            hh: &hh,
            output: sequence_stage1
                .as_device_buffer()
                .expect("sequence stage1 device buffer"),
            job: stage1_job,
        }];
        let sequence_stage2_targets = [CudaJ2kIdwtTarget {
            ll: sequence_stage1
                .as_device_buffer()
                .expect("sequence stage1 device buffer"),
            hl: &stage2_hl,
            lh: &stage2_lh,
            hh: &stage2_hh,
            output: sequence_stage2
                .as_device_buffer()
                .expect("sequence stage2 device buffer"),
            job: stage2_job,
        }];
        let queued = context
            .j2k_inverse_dwt_batch_sequence_enqueue_with_pool(
                &[&sequence_stage1_targets, &sequence_stage2_targets],
                &pool,
            )
            .expect("queued IDWT sequence");
        assert_eq!(queued.execution().kernel_dispatches(), 4);
        assert_eq!(queued.resource_count(), 1);
        context
            .synchronize()
            .expect("queued IDWT sequence completion");
        drop(queued);

        let mut legacy_actual = vec![0.0f32; 16];
        legacy_stage2
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut legacy_actual))
            .expect("download legacy stage2 IDWT");
        let mut sequence_actual = vec![0.0f32; 16];
        sequence_stage2
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut sequence_actual))
            .expect("download sequence stage2 IDWT");
        assert_eq!(sequence_actual, legacy_actual);
    }

    #[test]
    fn j2k_store_rgb8_mct_matches_inverse_mct_plus_store_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let plane0 = [16.0f32, 18.0, 21.0, 24.0];
        let plane1 = [-3.0f32, 4.0, 5.0, -6.0];
        let plane2 = [2.0f32, -1.0, 7.0, 3.0];
        let legacy0 = context
            .upload(super::f32_slice_as_bytes(&plane0))
            .expect("upload legacy MCT plane 0");
        let legacy1 = context
            .upload(super::f32_slice_as_bytes(&plane1))
            .expect("upload legacy MCT plane 1");
        let legacy2 = context
            .upload(super::f32_slice_as_bytes(&plane2))
            .expect("upload legacy MCT plane 2");
        let fused0 = context
            .upload(super::f32_slice_as_bytes(&plane0))
            .expect("upload fused MCT plane 0");
        let fused1 = context
            .upload(super::f32_slice_as_bytes(&plane1))
            .expect("upload fused MCT plane 1");
        let fused2 = context
            .upload(super::f32_slice_as_bytes(&plane2))
            .expect("upload fused MCT plane 2");
        let addend = 128.0;

        let mct_stats = context
            .j2k_inverse_mct_device(
                &legacy0,
                &legacy1,
                &legacy2,
                super::CudaJ2kInverseMctJob {
                    len: 4,
                    irreversible97: 0,
                    addend0: addend,
                    addend1: addend,
                    addend2: addend,
                },
            )
            .expect("legacy inverse MCT");
        assert_eq!(mct_stats.kernel_dispatches(), 1);
        let store_job = super::CudaJ2kStoreRgb8Job {
            input_width0: 2,
            input_width1: 2,
            input_width2: 2,
            source_x0: 0,
            source_y0: 0,
            source_x1: 0,
            source_y1: 0,
            source_x2: 0,
            source_y2: 0,
            copy_width: 2,
            copy_height: 2,
            output_width: 2,
            output_height: 2,
            output_x: 0,
            output_y: 0,
            addend0: 0.0,
            addend1: 0.0,
            addend2: 0.0,
            bit_depth0: 8,
            bit_depth1: 8,
            bit_depth2: 8,
            rgba: 1,
        };
        let legacy_output = context
            .j2k_store_rgb8_device(&legacy0, &legacy1, &legacy2, store_job)
            .expect("legacy RGB8 store");
        let fused_output = context
            .j2k_store_rgb8_mct_device(
                &fused0,
                &fused1,
                &fused2,
                super::CudaJ2kStoreRgb8MctJob {
                    store: super::CudaJ2kStoreRgb8Job {
                        addend0: addend,
                        addend1: addend,
                        addend2: addend,
                        ..store_job
                    },
                    irreversible97: 0,
                },
            )
            .expect("fused RGB8 MCT store");

        assert_eq!(legacy_output.execution().kernel_dispatches(), 1);
        assert_eq!(fused_output.execution().kernel_dispatches(), 1);
        let mut legacy_bytes = vec![0u8; 16];
        legacy_output
            .buffer()
            .copy_to_host(&mut legacy_bytes)
            .expect("download legacy RGB8");
        let mut fused_bytes = vec![0u8; 16];
        fused_output
            .buffer()
            .copy_to_host(&mut fused_bytes)
            .expect("download fused RGB8");
        assert_eq!(fused_bytes, legacy_bytes);
    }

    #[test]
    #[allow(clippy::similar_names, clippy::too_many_lines)]
    fn j2k_store_rgb8_mct_batch_matches_separate_stores_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let plane0_a = [16.0f32, 18.0, 21.0, 24.0];
        let plane1_a = [-3.0f32, 4.0, 5.0, -6.0];
        let plane2_a = [2.0f32, -1.0, 7.0, 3.0];
        let plane0_b = [3.0f32, 7.0, 11.0, 13.0];
        let plane1_b = [5.0f32, -2.0, 9.0, 1.0];
        let plane2_b = [-4.0f32, 6.0, 0.0, 8.0];

        let plane0_a = context
            .upload(super::f32_slice_as_bytes(&plane0_a))
            .expect("upload plane 0 A");
        let plane1_a = context
            .upload(super::f32_slice_as_bytes(&plane1_a))
            .expect("upload plane 1 A");
        let plane2_a = context
            .upload(super::f32_slice_as_bytes(&plane2_a))
            .expect("upload plane 2 A");
        let plane0_b = context
            .upload(super::f32_slice_as_bytes(&plane0_b))
            .expect("upload plane 0 B");
        let plane1_b = context
            .upload(super::f32_slice_as_bytes(&plane1_b))
            .expect("upload plane 1 B");
        let plane2_b = context
            .upload(super::f32_slice_as_bytes(&plane2_b))
            .expect("upload plane 2 B");

        let store = super::CudaJ2kStoreRgb8Job {
            input_width0: 2,
            input_width1: 2,
            input_width2: 2,
            source_x0: 0,
            source_y0: 0,
            source_x1: 0,
            source_y1: 0,
            source_x2: 0,
            source_y2: 0,
            copy_width: 2,
            copy_height: 2,
            output_width: 2,
            output_height: 2,
            output_x: 0,
            output_y: 0,
            addend0: 128.0,
            addend1: 128.0,
            addend2: 128.0,
            bit_depth0: 8,
            bit_depth1: 8,
            bit_depth2: 8,
            rgba: 1,
        };
        let separate_a = context
            .j2k_store_rgb8_mct_device(
                &plane0_a,
                &plane1_a,
                &plane2_a,
                super::CudaJ2kStoreRgb8MctJob {
                    store,
                    irreversible97: 0,
                },
            )
            .expect("separate fused store A");
        let separate_b = context
            .j2k_store_rgb8_mct_device(
                &plane0_b,
                &plane1_b,
                &plane2_b,
                super::CudaJ2kStoreRgb8MctJob {
                    store,
                    irreversible97: 0,
                },
            )
            .expect("separate fused store B");

        let batched = context
            .j2k_store_rgb8_mct_batch_device(&[
                super::CudaJ2kStoreRgb8MctTarget {
                    plane0: &plane0_a,
                    plane1: &plane1_a,
                    plane2: &plane2_a,
                    job: super::CudaJ2kStoreRgb8MctJob {
                        store,
                        irreversible97: 0,
                    },
                },
                super::CudaJ2kStoreRgb8MctTarget {
                    plane0: &plane0_b,
                    plane1: &plane1_b,
                    plane2: &plane2_b,
                    job: super::CudaJ2kStoreRgb8MctJob {
                        store,
                        irreversible97: 0,
                    },
                },
            ])
            .expect("batched fused store");

        assert_eq!(batched.execution().kernel_dispatches(), 1);
        assert_eq!(batched.outputs().len(), 2);
        let mut separate_a_bytes = vec![0u8; 16];
        separate_a
            .buffer()
            .copy_to_host(&mut separate_a_bytes)
            .expect("download separate A");
        let mut separate_b_bytes = vec![0u8; 16];
        separate_b
            .buffer()
            .copy_to_host(&mut separate_b_bytes)
            .expect("download separate B");
        let mut batch_a_bytes = vec![0u8; 16];
        batched.outputs()[0]
            .copy_to_host(&mut batch_a_bytes)
            .expect("download batch A");
        let mut batch_b_bytes = vec![0u8; 16];
        batched.outputs()[1]
            .copy_to_host(&mut batch_b_bytes)
            .expect("download batch B");
        assert_eq!(batch_a_bytes, separate_a_bytes);
        assert_eq!(batch_b_bytes, separate_b_bytes);
    }

    #[test]
    fn j2k_store_rgb8_mct_single_matches_one_item_batch_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let plane0 = [16.0f32, 18.0, 21.0, 24.0];
        let plane1 = [-3.0f32, 4.0, 5.0, -6.0];
        let plane2 = [2.0f32, -1.0, 7.0, 3.0];
        let single0 = context
            .upload(super::f32_slice_as_bytes(&plane0))
            .expect("upload single plane 0");
        let single1 = context
            .upload(super::f32_slice_as_bytes(&plane1))
            .expect("upload single plane 1");
        let single2 = context
            .upload(super::f32_slice_as_bytes(&plane2))
            .expect("upload single plane 2");
        let batch0 = context
            .upload(super::f32_slice_as_bytes(&plane0))
            .expect("upload batch plane 0");
        let batch1 = context
            .upload(super::f32_slice_as_bytes(&plane1))
            .expect("upload batch plane 1");
        let batch2 = context
            .upload(super::f32_slice_as_bytes(&plane2))
            .expect("upload batch plane 2");

        let store = super::CudaJ2kStoreRgb8Job {
            input_width0: 2,
            input_width1: 2,
            input_width2: 2,
            source_x0: 0,
            source_y0: 0,
            source_x1: 0,
            source_y1: 0,
            source_x2: 0,
            source_y2: 0,
            copy_width: 2,
            copy_height: 2,
            output_width: 2,
            output_height: 2,
            output_x: 0,
            output_y: 0,
            addend0: 128.0,
            addend1: 128.0,
            addend2: 128.0,
            bit_depth0: 8,
            bit_depth1: 8,
            bit_depth2: 8,
            rgba: 1,
        };
        let job = super::CudaJ2kStoreRgb8MctJob {
            store,
            irreversible97: 0,
        };
        let single = context
            .j2k_store_rgb8_mct_device(&single0, &single1, &single2, job)
            .expect("single RGB8 MCT store");
        let batch = context
            .j2k_store_rgb8_mct_batch_device(&[super::CudaJ2kStoreRgb8MctTarget {
                plane0: &batch0,
                plane1: &batch1,
                plane2: &batch2,
                job,
            }])
            .expect("one-item batch RGB8 MCT store");

        assert_eq!(single.execution().kernel_dispatches(), 1);
        assert_eq!(batch.execution().kernel_dispatches(), 1);
        let mut single_bytes = vec![0u8; 16];
        single
            .buffer()
            .copy_to_host(&mut single_bytes)
            .expect("download single RGB8 MCT store");
        let mut batch_bytes = vec![0u8; 16];
        batch.outputs()[0]
            .copy_to_host(&mut batch_bytes)
            .expect("download one-item batch RGB8 MCT store");
        assert_eq!(single_bytes, batch_bytes);
    }

    #[test]
    fn j2k_store_rgb16_mct_matches_inverse_mct_plus_store_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let plane0 = [40.0f32, 44.0, 52.0, 55.0];
        let plane1 = [-3.5f32, 1.25, 2.75, -4.0];
        let plane2 = [5.0f32, -2.0, 1.5, 6.0];
        let legacy0 = context
            .upload(super::f32_slice_as_bytes(&plane0))
            .expect("upload legacy ICT plane 0");
        let legacy1 = context
            .upload(super::f32_slice_as_bytes(&plane1))
            .expect("upload legacy ICT plane 1");
        let legacy2 = context
            .upload(super::f32_slice_as_bytes(&plane2))
            .expect("upload legacy ICT plane 2");
        let fused0 = context
            .upload(super::f32_slice_as_bytes(&plane0))
            .expect("upload fused ICT plane 0");
        let fused1 = context
            .upload(super::f32_slice_as_bytes(&plane1))
            .expect("upload fused ICT plane 1");
        let fused2 = context
            .upload(super::f32_slice_as_bytes(&plane2))
            .expect("upload fused ICT plane 2");
        let addend = 32768.0;

        context
            .j2k_inverse_mct_device(
                &legacy0,
                &legacy1,
                &legacy2,
                super::CudaJ2kInverseMctJob {
                    len: 4,
                    irreversible97: 1,
                    addend0: addend,
                    addend1: addend,
                    addend2: addend,
                },
            )
            .expect("legacy inverse ICT");
        let store_job = super::CudaJ2kStoreRgb16Job {
            input_width0: 2,
            input_width1: 2,
            input_width2: 2,
            source_x0: 0,
            source_y0: 0,
            source_x1: 0,
            source_y1: 0,
            source_x2: 0,
            source_y2: 0,
            copy_width: 2,
            copy_height: 2,
            output_width: 2,
            output_height: 2,
            output_x: 0,
            output_y: 0,
            addend0: 0.0,
            addend1: 0.0,
            addend2: 0.0,
            bit_depth0: 16,
            bit_depth1: 16,
            bit_depth2: 16,
            rgba: 0,
        };
        let legacy_output = context
            .j2k_store_rgb16_device(&legacy0, &legacy1, &legacy2, store_job)
            .expect("legacy RGB16 store");
        let fused_output = context
            .j2k_store_rgb16_mct_device(
                &fused0,
                &fused1,
                &fused2,
                super::CudaJ2kStoreRgb16MctJob {
                    store: super::CudaJ2kStoreRgb16Job {
                        addend0: addend,
                        addend1: addend,
                        addend2: addend,
                        ..store_job
                    },
                    irreversible97: 1,
                },
            )
            .expect("fused RGB16 MCT store");

        assert_eq!(legacy_output.execution().kernel_dispatches(), 1);
        assert_eq!(fused_output.execution().kernel_dispatches(), 1);
        let mut legacy_bytes = vec![0u8; 24];
        legacy_output
            .buffer()
            .copy_to_host(&mut legacy_bytes)
            .expect("download legacy RGB16");
        let mut fused_bytes = vec![0u8; 24];
        fused_output
            .buffer()
            .copy_to_host(&mut fused_bytes)
            .expect("download fused RGB16");
        assert_eq!(fused_bytes, legacy_bytes);
    }

    #[test]
    fn j2k_dequantize_htj2k_codeblocks_multi_uses_one_dispatch_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let first = context
            .upload(super::i32_slice_as_bytes(&[0, 0, 0, 0]))
            .expect("upload first coefficients");
        let second = context
            .upload(super::i32_slice_as_bytes(&[0, 0]))
            .expect("upload second coefficients");
        let first_jobs = [CudaHtj2kCodeBlockJob {
            payload_offset: 0,
            width: 2,
            height: 2,
            payload_len: 0,
            cleanup_length: 0,
            refinement_length: 0,
            missing_bit_planes: 0,
            num_bitplanes: 1,
            number_of_coding_passes: 1,
            output_stride: 2,
            output_offset: 0,
            dequantization_step: 1.0,
            stripe_causal: false,
        }];
        let second_jobs = [CudaHtj2kCodeBlockJob {
            payload_offset: 0,
            width: 2,
            height: 1,
            payload_len: 0,
            cleanup_length: 0,
            refinement_length: 0,
            missing_bit_planes: 0,
            num_bitplanes: 1,
            number_of_coding_passes: 1,
            output_stride: 2,
            output_offset: 0,
            dequantization_step: 1.0,
            stripe_causal: false,
        }];

        let execution = context
            .j2k_dequantize_htj2k_codeblocks_multi_device(&[
                CudaHtj2kDequantizeTarget {
                    coefficients: &first,
                    jobs: &first_jobs,
                    output_words: 4,
                },
                CudaHtj2kDequantizeTarget {
                    coefficients: &second,
                    jobs: &second_jobs,
                    output_words: 2,
                },
            ])
            .expect("multi-buffer HTJ2K dequant");
        assert_eq!(execution.kernel_dispatches(), 1);

        let mut first_actual = vec![f32::NAN; 4];
        first
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut first_actual))
            .expect("download first coefficients");
        assert_eq!(first_actual, vec![0.0; 4]);
        let mut second_actual = vec![f32::NAN; 2];
        second
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut second_actual))
            .expect("download second coefficients");
        assert_eq!(second_actual, vec![0.0; 2]);
    }

    #[test]
    fn queued_cleanup_metadata_dequantizes_without_second_job_upload_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let first = context
            .upload(super::i32_slice_as_bytes(&[1, i32::MIN + 2, 0, 3]))
            .expect("upload first coefficients");
        let second = context
            .upload(super::i32_slice_as_bytes(&[4, i32::MIN + 5]))
            .expect("upload second coefficients");
        let jobs = [
            CudaHtj2kCleanupMultiKernelJob {
                output_ptr: first.device_ptr(),
                coded_offset: 0,
                width: 2,
                height: 2,
                coded_len: 0,
                cleanup_length: 0,
                refinement_length: 0,
                missing_msbs: 0,
                num_bitplanes: 31,
                number_of_coding_passes: 1,
                output_stride: 2,
                output_offset: 0,
                dequantization_step: 0.5,
                stripe_causal: 0,
            },
            CudaHtj2kCleanupMultiKernelJob {
                output_ptr: second.device_ptr(),
                coded_offset: 0,
                width: 2,
                height: 1,
                coded_len: 0,
                cleanup_length: 0,
                refinement_length: 0,
                missing_msbs: 0,
                num_bitplanes: 31,
                number_of_coding_passes: 1,
                output_stride: 2,
                output_offset: 0,
                dequantization_step: 0.25,
                stripe_causal: 0,
            },
        ];
        let jobs_buffer = pool
            .upload(super::htj2k_cleanup_multi_jobs_as_bytes(&jobs))
            .expect("upload cleanup metadata");
        let queued = CudaQueuedHtj2kCleanup {
            resources: vec![jobs_buffer],
            status_buffer: None,
            status_count: jobs.len(),
            kernel_name: "signinum_htj2k_decode_codeblocks_multi",
            execution: CudaExecutionStats::default(),
        };

        let execution = context
            .j2k_dequantize_queued_htj2k_cleanup_with_pool(&queued)
            .expect("dequant from queued cleanup metadata");
        assert_eq!(execution.kernel_dispatches(), 1);

        let mut first_actual = vec![f32::NAN; 4];
        first
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut first_actual))
            .expect("download first coefficients");
        assert_eq!(first_actual, vec![0.5, -1.0, 0.0, 1.5]);
        let mut second_actual = vec![f32::NAN; 2];
        second
            .copy_to_host(super::f32_slice_as_bytes_mut(&mut second_actual))
            .expect("download second coefficients");
        assert_eq!(second_actual, vec![1.0, -1.25]);
    }

    #[test]
    fn htj2k_decode_multi_kernel_routes_cleanup_only_jobs() {
        let cleanup_job = CudaHtj2kCleanupMultiKernelJob {
            output_ptr: 0,
            coded_offset: 0,
            width: 64,
            height: 64,
            coded_len: 8,
            cleanup_length: 8,
            refinement_length: 0,
            missing_msbs: 0,
            num_bitplanes: 8,
            number_of_coding_passes: 1,
            output_stride: 64,
            output_offset: 0,
            dequantization_step: 1.0,
            stripe_causal: 0,
        };
        let (_, cleanup_kernel_name) = super::htj2k_decode_multi_kernel_for_jobs(&[cleanup_job]);
        assert_eq!(
            cleanup_kernel_name,
            "signinum_htj2k_decode_codeblocks_multi_cleanup_only"
        );

        let mut refinement_job = cleanup_job;
        refinement_job.refinement_length = 4;
        refinement_job.number_of_coding_passes = 2;
        let (_, generic_kernel_name) = super::htj2k_decode_multi_kernel_for_jobs(&[refinement_job]);
        assert_eq!(
            generic_kernel_name,
            "signinum_htj2k_decode_codeblocks_multi"
        );
    }

    #[test]
    fn htj2k_decode_multi_cleanup_dequant_kernel_accepts_cleanup_only_jobs() {
        let cleanup_job = CudaHtj2kCleanupMultiKernelJob {
            output_ptr: 0,
            coded_offset: 0,
            width: 64,
            height: 64,
            coded_len: 8,
            cleanup_length: 8,
            refinement_length: 0,
            missing_msbs: 0,
            num_bitplanes: 8,
            number_of_coding_passes: 1,
            output_stride: 64,
            output_offset: 0,
            dequantization_step: 1.0,
            stripe_causal: 0,
        };
        let (_, cleanup_dequant_kernel_name) =
            super::htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&[cleanup_job])
                .expect("cleanup-only jobs use fused cleanup/dequant kernel");
        assert_eq!(
            cleanup_dequant_kernel_name,
            "signinum_htj2k_decode_codeblocks_multi_cleanup_dequantize"
        );
    }

    #[test]
    fn htj2k_decode_multi_cleanup_dequant_kernel_rejects_refinement_jobs() {
        let mut refinement_job = CudaHtj2kCleanupMultiKernelJob {
            output_ptr: 0,
            coded_offset: 0,
            width: 64,
            height: 64,
            coded_len: 12,
            cleanup_length: 8,
            refinement_length: 4,
            missing_msbs: 0,
            num_bitplanes: 8,
            number_of_coding_passes: 2,
            output_stride: 64,
            output_offset: 0,
            dequantization_step: 1.0,
            stripe_causal: 0,
        };
        assert!(
            super::htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&[refinement_job]).is_none()
        );

        refinement_job.refinement_length = 0;
        assert!(
            super::htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&[refinement_job]).is_none()
        );
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn htj2k_cleanup_multi_empty_targets_use_no_dispatch_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let first_vlc = [0u16; 1024];
        let later_vlc = [0u16; 1024];
        let first_uvlc = [0u16; 320];
        let later_uvlc = [0u16; 256];
        let tables = context
            .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
                vlc_table0: &first_vlc,
                vlc_table1: &later_vlc,
                uvlc_table0: &first_uvlc,
                uvlc_table1: &later_uvlc,
            })
            .expect("decode tables");
        let resources = context
            .upload_htj2k_decode_resources_with_tables(&[], &tables)
            .expect("decode resources");

        let execution = context
            .decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool(
                &resources,
                &[] as &[CudaHtj2kCleanupTarget<'_>],
                &pool,
            )
            .expect("empty cleanup batch");

        assert_eq!(execution.kernel_dispatches(), 0);
        assert_eq!(execution.decode_kernel_dispatches(), 0);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn htj2k_cleanup_multi_enqueue_empty_targets_finish_with_no_dispatch_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let pool = context.buffer_pool();
        let first_vlc = [0u16; 1024];
        let later_vlc = [0u16; 1024];
        let first_uvlc = [0u16; 320];
        let later_uvlc = [0u16; 256];
        let tables = context
            .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
                vlc_table0: &first_vlc,
                vlc_table1: &later_vlc,
                uvlc_table0: &first_uvlc,
                uvlc_table1: &later_uvlc,
            })
            .expect("decode tables");
        let resources = context
            .upload_htj2k_decode_resources_with_tables(&[], &tables)
            .expect("decode resources");

        let queued = context
            .decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool(
                &resources,
                &[] as &[CudaHtj2kCleanupTarget<'_>],
                &pool,
            )
            .expect("empty queued cleanup batch");
        assert_eq!(queued.execution().kernel_dispatches(), 0);
        assert_eq!(queued.execution().decode_kernel_dispatches(), 0);
        assert_eq!(queued.resource_count(), 0);

        let execution = queued.finish().expect("finish empty queued cleanup");
        assert_eq!(execution.kernel_dispatches(), 0);
        assert_eq!(execution.decode_kernel_dispatches(), 0);
    }

    #[test]
    fn j2k_forward_rct_matches_cpu_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let mut plane0 = vec![10.0, 1.0, 0.0, 255.0, 128.0];
        let mut plane1 = vec![20.0, 2.0, 255.0, 0.0, 64.0];
        let mut plane2 = vec![30.0, 3.0, 128.0, 127.0, 32.0];
        let mut expected0 = plane0.clone();
        let mut expected1 = plane1.clone();
        let mut expected2 = plane2.clone();
        for ((r, g), b) in expected0
            .iter_mut()
            .zip(expected1.iter_mut())
            .zip(expected2.iter_mut())
        {
            let r0 = *r;
            let g0 = *g;
            let b0 = *b;
            *r = ((r0 + 2.0_f32 * g0 + b0) * 0.25_f32).floor();
            *g = b0 - g0;
            *b = r0 - g0;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let execution = context
            .j2k_forward_rct(&mut plane0, &mut plane1, &mut plane2)
            .expect("CUDA forward RCT");

        assert_eq!(execution.kernel_dispatches(), 1);
        assert_eq!(plane0, expected0);
        assert_eq!(plane1, expected1);
        assert_eq!(plane2, expected2);
    }

    #[test]
    fn j2k_deinterleave_to_f32_matches_cpu_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let pixels = [0u8, 128, 255, 64, 32, 16];
        let context = CudaContext::system_default().expect("CUDA context");
        let output = context
            .j2k_deinterleave_to_f32(&pixels, 2, 3, 8, false)
            .expect("CUDA deinterleave");

        assert_eq!(output.execution().kernel_dispatches(), 1);
        assert_eq!(
            output.components(),
            &[vec![-128.0, -64.0], vec![0.0, -96.0], vec![127.0, -112.0],]
        );
    }

    #[test]
    fn j2k_deinterleave_then_rct_can_stay_resident_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let pixels = [10u8, 20, 30, 40, 50, 60];
        let context = CudaContext::system_default().expect("CUDA context");
        let mut components = context
            .j2k_deinterleave_to_f32_resident(&pixels, 2, 3, 8, false)
            .expect("resident CUDA deinterleave");

        assert_eq!(components.num_components(), 3);
        assert_eq!(components.num_pixels(), 2);
        assert_eq!(components.execution().kernel_dispatches(), 1);

        let rct_execution = context
            .j2k_forward_rct_resident(&mut components)
            .expect("resident CUDA forward RCT");

        assert_eq!(rct_execution.kernel_dispatches(), 1);
        assert_eq!(
            components
                .download_components()
                .expect("download resident components"),
            vec![vec![-108.0, -78.0], vec![10.0, 10.0], vec![-10.0, -10.0]]
        );
    }

    #[test]
    fn j2k_deinterleave_then_ict_can_stay_resident_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let pixels = [10u8, 20, 30, 40, 50, 60];
        let context = CudaContext::system_default().expect("CUDA context");
        let mut components = context
            .j2k_deinterleave_to_f32_resident(&pixels, 2, 3, 8, false)
            .expect("resident CUDA deinterleave");

        let ict_execution = context
            .j2k_forward_ict_resident(&mut components)
            .expect("resident CUDA forward ICT");

        assert_eq!(ict_execution.kernel_dispatches(), 1);
        let actual = components
            .download_components()
            .expect("download resident components");
        let expected = [[-118.0f32, -88.0], [-108.0, -78.0], [-98.0, -68.0]];
        for idx in 0..2 {
            let r = expected[0][idx];
            let g = expected[1][idx];
            let b = expected[2][idx];
            let expected_y = 0.299 * r + 0.587 * g + 0.114 * b;
            let blue_chroma = -0.16875 * r - 0.33126 * g + 0.5 * b;
            let red_chroma = 0.5 * r - 0.41869 * g - 0.08131 * b;
            assert!((actual[0][idx] - expected_y).abs() < 0.000_1);
            assert!((actual[1][idx] - blue_chroma).abs() < 0.000_1);
            assert!((actual[2][idx] - red_chroma).abs() < 0.000_1);
        }
    }

    #[test]
    fn j2k_resident_deinterleave_can_feed_resident_dwt53_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let pixels = [0u8, 64, 128, 255];
        let context = CudaContext::system_default().expect("CUDA context");
        let components = context
            .j2k_deinterleave_to_f32_resident(&pixels, 4, 1, 8, false)
            .expect("resident CUDA deinterleave");
        let host_component = components
            .download_components()
            .expect("download source component")[0]
            .clone();
        let expected = context
            .j2k_forward_dwt53(&host_component, 2, 2, 1)
            .expect("host-staged CUDA DWT");

        let resident = context
            .j2k_forward_dwt53_resident_component(&components, 0, 2, 2, 1)
            .expect("resident CUDA DWT");

        assert_eq!(resident.levels(), expected.levels());
        assert_eq!(resident.ll_dimensions(), expected.ll_dimensions());
        assert_eq!(resident.execution().copy_kernel_dispatches, 1);
        assert_eq!(
            resident
                .download_transformed()
                .expect("download resident DWT"),
            expected.transformed()
        );
    }

    #[test]
    fn j2k_resident_deinterleave_can_feed_resident_dwt97_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let pixels = [0u8, 64, 128, 255];
        let context = CudaContext::system_default().expect("CUDA context");
        let components = context
            .j2k_deinterleave_to_f32_resident(&pixels, 4, 1, 8, false)
            .expect("resident CUDA deinterleave");
        let host_component = components
            .download_components()
            .expect("download source component")[0]
            .clone();
        let expected = context
            .j2k_forward_dwt97(&host_component, 2, 2, 1)
            .expect("host-staged CUDA DWT");

        let resident = context
            .j2k_forward_dwt97_resident_component(&components, 0, 2, 2, 1)
            .expect("resident CUDA DWT");

        assert_eq!(resident.levels(), expected.levels());
        assert_eq!(resident.ll_dimensions(), expected.ll_dimensions());
        assert_eq!(resident.execution().copy_kernel_dispatches, 1);
        assert_eq!(
            resident
                .download_transformed()
                .expect("download resident DWT"),
            expected.transformed()
        );
    }

    #[test]
    fn j2k_forward_ict_matches_cpu_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let mut plane0 = vec![10.0, 1.0, 0.0, 255.0, 128.0];
        let mut plane1 = vec![20.0, 2.0, 255.0, 0.0, 64.0];
        let mut plane2 = vec![30.0, 3.0, 128.0, 127.0, 32.0];
        let mut expected0 = plane0.clone();
        let mut expected1 = plane1.clone();
        let mut expected2 = plane2.clone();
        for ((r, g), b) in expected0
            .iter_mut()
            .zip(expected1.iter_mut())
            .zip(expected2.iter_mut())
        {
            let r0 = *r;
            let g0 = *g;
            let b0 = *b;
            *r = 0.299 * r0 + 0.587 * g0 + 0.114 * b0;
            *g = -0.16875 * r0 - 0.33126 * g0 + 0.5 * b0;
            *b = 0.5 * r0 - 0.41869 * g0 - 0.08131 * b0;
        }

        let context = CudaContext::system_default().expect("CUDA context");
        let execution = context
            .j2k_forward_ict(&mut plane0, &mut plane1, &mut plane2)
            .expect("CUDA forward ICT");

        assert_eq!(execution.kernel_dispatches(), 1);
        for (actual, expected) in plane0.iter().zip(expected0) {
            assert!((*actual - expected).abs() < 0.0001);
        }
        for (actual, expected) in plane1.iter().zip(expected1) {
            assert!((*actual - expected).abs() < 0.0001);
        }
        for (actual, expected) in plane2.iter().zip(expected2) {
            assert!((*actual - expected).abs() < 0.0001);
        }
    }

    #[test]
    fn j2k_forward_dwt53_matches_cpu_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let width = 5usize;
        let height = 3usize;
        let samples: Vec<f32> = (0..width * height)
            .map(|value| {
                let sample = u16::try_from((value * 7 + 3) % 19).expect("sample fits in u16");
                f32::from(sample)
            })
            .collect();
        let expected = cpu_forward_dwt53_buffer(&samples, width, height, 1);

        let context = CudaContext::system_default().expect("CUDA context");
        let output = context
            .j2k_forward_dwt53(
                &samples,
                u32::try_from(width).expect("width fits in u32"),
                u32::try_from(height).expect("height fits in u32"),
                1,
            )
            .expect("CUDA forward 5/3 DWT");

        assert_eq!(output.execution().kernel_dispatches(), 2);
        assert_eq!(output.transformed(), expected.as_slice());
        assert_eq!(output.ll_dimensions(), (3, 2));
    }

    #[test]
    fn j2k_forward_dwt97_matches_cpu_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let width = 5usize;
        let height = 3usize;
        let samples: Vec<f32> = (0..width * height)
            .map(|value| {
                let sample = u16::try_from((value * 11 + 5) % 31).expect("sample fits in u16");
                f32::from(sample) - 12.0
            })
            .collect();
        let expected = cpu_forward_dwt97_buffer(&samples, width, height, 1);

        let context = CudaContext::system_default().expect("CUDA context");
        let output = context
            .j2k_forward_dwt97(
                &samples,
                u32::try_from(width).expect("width fits in u32"),
                u32::try_from(height).expect("height fits in u32"),
                1,
            )
            .expect("CUDA forward 9/7 DWT");

        assert_eq!(output.execution().kernel_dispatches(), 2);
        for (actual, expected) in output.transformed().iter().zip(expected) {
            assert!((*actual - expected).abs() < 0.001);
        }
        assert_eq!(output.ll_dimensions(), (3, 2));
    }

    #[test]
    fn j2k_quantize_subband_matches_cpu_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let samples = [-3.6f32, -2.5, -0.4, 0.0, 0.49, 1.5, 3.2, 9.9];
        let context = CudaContext::system_default().expect("CUDA context");
        let reversible = context
            .j2k_quantize_subband(
                &samples,
                CudaJ2kQuantizeJob {
                    step_exponent: 8,
                    step_mantissa: 0,
                    range_bits: 8,
                    reversible: true,
                },
            )
            .expect("CUDA reversible quantize");
        assert_eq!(reversible.execution().kernel_dispatches(), 1);
        assert_eq!(reversible.coefficients(), &[-4, -3, 0, 0, 0, 2, 3, 10]);

        let irreversible = context
            .j2k_quantize_subband(
                &samples,
                CudaJ2kQuantizeJob {
                    step_exponent: 9,
                    step_mantissa: 0,
                    range_bits: 8,
                    reversible: false,
                },
            )
            .expect("CUDA irreversible quantize");
        assert_eq!(irreversible.execution().kernel_dispatches(), 1);
        // delta = 2^(range_bits - step_exponent) = 2^(8 - 9) = 0.5, so q = sign*floor(|s|/0.5).
        // Matches native QuantStepSize::delta and JPEG 2000 T.800 Annex E.
        assert_eq!(irreversible.coefficients(), &[-7, -5, 0, 0, 0, 3, 6, 19]);
    }

    #[test]
    fn j2k_quantize_strided_resident_subband_matches_contiguous_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let samples: Vec<f32> = (0u16..12).map(|value| f32::from(value) - 6.0).collect();
        let context = CudaContext::system_default().expect("CUDA context");
        let sample_buffer = context.upload_f32(&samples).expect("resident samples");
        let quantization = CudaJ2kQuantizeJob {
            step_exponent: 8,
            step_mantissa: 0,
            range_bits: 8,
            reversible: true,
        };
        let resident = context
            .j2k_quantize_subband_region_resident(
                &sample_buffer,
                CudaJ2kQuantizeSubbandRegionJob {
                    x0: 1,
                    y0: 1,
                    width: 2,
                    height: 2,
                    stride: 4,
                    quantization,
                },
            )
            .expect("resident strided quantize");
        let contiguous = [samples[5], samples[6], samples[9], samples[10]];
        let expected = context
            .j2k_quantize_subband(&contiguous, quantization)
            .expect("contiguous quantize");

        assert_eq!(resident.coefficient_count(), 4);
        assert_eq!(resident.execution().kernel_dispatches(), 1);
        assert_eq!(
            resident
                .download_coefficients()
                .expect("download resident quantized coefficients"),
            expected.coefficients()
        );
    }

    fn cpu_forward_dwt53_buffer(
        samples: &[f32],
        width: usize,
        height: usize,
        levels: u8,
    ) -> Vec<f32> {
        let mut buffer = samples.to_vec();
        let mut current_width = width;
        let mut current_height = height;

        for _ in 0..levels {
            if current_width < 2 && current_height < 2 {
                break;
            }
            if current_height >= 2 {
                let low_height = current_height.div_ceil(2);
                let mut col = vec![0.0; current_height];
                for x in 0..current_width {
                    for y in 0..current_height {
                        col[y] = buffer[y * width + x];
                    }
                    forward_lift_53(&mut col);
                    for y in 0..low_height {
                        buffer[y * width + x] = col[y * 2];
                    }
                    for y in 0..current_height / 2 {
                        buffer[(low_height + y) * width + x] = col[y * 2 + 1];
                    }
                }
            }
            if current_width >= 2 {
                let mut row = vec![0.0; current_width];
                for y in 0..current_height {
                    let row_start = y * width;
                    row.copy_from_slice(&buffer[row_start..row_start + current_width]);
                    forward_lift_53(&mut row);
                    let low_width = current_width.div_ceil(2);
                    for x in 0..low_width {
                        buffer[row_start + x] = row[x * 2];
                    }
                    for x in 0..current_width / 2 {
                        buffer[row_start + low_width + x] = row[x * 2 + 1];
                    }
                }
            }
            current_width = current_width.div_ceil(2);
            current_height = current_height.div_ceil(2);
        }

        buffer
    }

    fn cpu_forward_dwt97_buffer(
        samples: &[f32],
        width: usize,
        height: usize,
        levels: u8,
    ) -> Vec<f32> {
        let mut buffer = samples.to_vec();
        let mut current_width = width;
        let mut current_height = height;

        for _ in 0..levels {
            if current_width < 2 && current_height < 2 {
                break;
            }
            if current_height >= 2 {
                let low_height = current_height.div_ceil(2);
                let mut col = vec![0.0; current_height];
                for x in 0..current_width {
                    for y in 0..current_height {
                        col[y] = buffer[y * width + x];
                    }
                    forward_lift_97(&mut col);
                    for y in 0..low_height {
                        buffer[y * width + x] = col[y * 2];
                    }
                    for y in 0..current_height / 2 {
                        buffer[(low_height + y) * width + x] = col[y * 2 + 1];
                    }
                }
            }
            if current_width >= 2 {
                let mut row = vec![0.0; current_width];
                for y in 0..current_height {
                    let row_start = y * width;
                    row.copy_from_slice(&buffer[row_start..row_start + current_width]);
                    forward_lift_97(&mut row);
                    let low_width = current_width.div_ceil(2);
                    for x in 0..low_width {
                        buffer[row_start + x] = row[x * 2];
                    }
                    for x in 0..current_width / 2 {
                        buffer[row_start + low_width + x] = row[x * 2 + 1];
                    }
                }
            }
            current_width = current_width.div_ceil(2);
            current_height = current_height.div_ceil(2);
        }

        buffer
    }

    fn forward_lift_53(data: &mut [f32]) {
        let n = data.len();
        if n < 2 {
            return;
        }

        let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
        for i in (1..n).step_by(2) {
            let left = data[i - 1];
            let right = if i + 1 < n {
                data[i + 1]
            } else {
                data[last_even]
            };
            data[i] -= ((left + right) * 0.5).floor();
        }

        for i in (0..n).step_by(2) {
            let left = if i > 0 { data[i - 1] } else { data[1] };
            let right = if i + 1 < n { data[i + 1] } else { left };
            data[i] += ((left + right) * 0.25 + 0.5).floor();
        }
    }

    fn forward_lift_97(data: &mut [f32]) {
        const ALPHA: f32 = -1.586_134_3;
        const BETA: f32 = -0.052_980_117;
        const GAMMA: f32 = 0.882_911_1;
        const DELTA: f32 = 0.443_506_87;
        const KAPPA: f32 = 1.230_174_1;
        const INV_KAPPA: f32 = 1.0 / KAPPA;

        let n = data.len();
        if n < 2 {
            return;
        }

        let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
        for i in (1..n).step_by(2) {
            let left = data[i - 1];
            let right = if i + 1 < n {
                data[i + 1]
            } else {
                data[last_even]
            };
            data[i] += ALPHA * (left + right);
        }
        for i in (0..n).step_by(2) {
            let left = if i > 0 { data[i - 1] } else { data[1] };
            let right = if i + 1 < n { data[i + 1] } else { left };
            data[i] += BETA * (left + right);
        }
        for i in (1..n).step_by(2) {
            let left = data[i - 1];
            let right = if i + 1 < n {
                data[i + 1]
            } else {
                data[last_even]
            };
            data[i] += GAMMA * (left + right);
        }
        for i in (0..n).step_by(2) {
            let left = if i > 0 { data[i - 1] } else { data[1] };
            let right = if i + 1 < n { data[i + 1] } else { left };
            data[i] += DELTA * (left + right);
        }
        for i in (0..n).step_by(2) {
            data[i] *= INV_KAPPA;
        }
        for i in (1..n).step_by(2) {
            data[i] *= KAPPA;
        }
    }
}
