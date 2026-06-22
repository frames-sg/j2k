use crate::{driver::CuResult, error::CudaError, kernels::CudaKernel};
use std::{os::raw::c_uint, sync::OnceLock};

pub(crate) const CUDA_SUCCESS: CuResult = 0;

pub(crate) const PINNED_UPLOAD_STAGING_POOL_MAX: usize = 8;

pub(crate) const PINNED_POOLED_I16_UPLOAD_MAX_BYTES: usize = 4 * 1024 * 1024;

pub(crate) const DWT97_ROW_LIFT_MAX_WIDTH: i32 = 1024;

pub(crate) const DWT97_ROW_LIFT_COOP_THREADS_X: c_uint = 128;

pub(crate) const DWT97_ROW_LIFT_COOP_ROWS_PER_BLOCK: c_uint = 4;

pub(crate) const CUDA_IDWT_TRACE_ENV_VAR: &str = "J2K_CUDA_IDWT_TRACE";

#[cfg(feature = "cuda-oxide-j2k-encode")]
pub(crate) const CUDA_OXIDE_J2K_ENCODE_ENV_VAR: &str = "J2K_CUDA_USE_OXIDE_J2K_ENCODE";

#[cfg(feature = "cuda-oxide-j2k-decode-store")]
pub(crate) const CUDA_OXIDE_J2K_DECODE_STORE_ENV_VAR: &str = "J2K_CUDA_USE_OXIDE_J2K_DECODE_STORE";

#[cfg(feature = "cuda-oxide-j2k-dequantize")]
pub(crate) const CUDA_OXIDE_J2K_DEQUANTIZE_ENV_VAR: &str = "J2K_CUDA_USE_OXIDE_J2K_DEQUANTIZE";

#[cfg(feature = "cuda-oxide-j2k-idwt")]
pub(crate) const CUDA_OXIDE_J2K_IDWT_ENV_VAR: &str = "J2K_CUDA_USE_OXIDE_J2K_IDWT";

#[cfg(feature = "cuda-oxide-transcode")]
pub(crate) const CUDA_OXIDE_TRANSCODE_ENV_VAR: &str = "J2K_CUDA_USE_OXIDE_TRANSCODE";

pub(crate) const DWT97_FUSED_COLUMN_QUANTIZE_DISABLE_ENV_VAR: &str =
    "J2K_CUDA_DISABLE_DWT97_FUSED_COLUMN_QUANTIZE";

pub(crate) static CUDA_STAGE_TIMINGS_DISABLED: OnceLock<bool> = OnceLock::new();

pub(crate) static DWT97_FUSED_COLUMN_QUANTIZE_DISABLED: OnceLock<bool> = OnceLock::new();

#[cfg(feature = "cuda-oxide-j2k-encode")]
pub(crate) static CUDA_OXIDE_J2K_ENCODE_ENABLED: OnceLock<bool> = OnceLock::new();

#[cfg(feature = "cuda-oxide-j2k-decode-store")]
pub(crate) static CUDA_OXIDE_J2K_DECODE_STORE_ENABLED: OnceLock<bool> = OnceLock::new();

#[cfg(feature = "cuda-oxide-j2k-dequantize")]
pub(crate) static CUDA_OXIDE_J2K_DEQUANTIZE_ENABLED: OnceLock<bool> = OnceLock::new();

#[cfg(feature = "cuda-oxide-j2k-idwt")]
pub(crate) static CUDA_OXIDE_J2K_IDWT_ENABLED: OnceLock<bool> = OnceLock::new();

#[cfg(feature = "cuda-oxide-transcode")]
pub(crate) static CUDA_OXIDE_TRANSCODE_ENABLED: OnceLock<bool> = OnceLock::new();

pub(crate) fn cuda_stage_timings_disabled() -> bool {
    *CUDA_STAGE_TIMINGS_DISABLED
        .get_or_init(|| std::env::var_os("J2K_CUDA_DISABLE_STAGE_TIMINGS").is_some())
}

pub(crate) fn dwt97_fused_column_quantize_disabled() -> bool {
    *DWT97_FUSED_COLUMN_QUANTIZE_DISABLED
        .get_or_init(|| std::env::var_os(DWT97_FUSED_COLUMN_QUANTIZE_DISABLE_ENV_VAR).is_some())
}

#[cfg(feature = "cuda-oxide-j2k-encode")]
pub(crate) fn cuda_oxide_j2k_encode_enabled() -> bool {
    *CUDA_OXIDE_J2K_ENCODE_ENABLED
        .get_or_init(|| std::env::var_os(CUDA_OXIDE_J2K_ENCODE_ENV_VAR).is_some())
}

#[cfg(feature = "cuda-oxide-j2k-decode-store")]
pub(crate) fn cuda_oxide_j2k_decode_store_enabled() -> bool {
    *CUDA_OXIDE_J2K_DECODE_STORE_ENABLED
        .get_or_init(|| std::env::var_os(CUDA_OXIDE_J2K_DECODE_STORE_ENV_VAR).is_some())
}

#[cfg(feature = "cuda-oxide-j2k-dequantize")]
pub(crate) fn cuda_oxide_j2k_dequantize_enabled() -> bool {
    *CUDA_OXIDE_J2K_DEQUANTIZE_ENABLED
        .get_or_init(|| std::env::var_os(CUDA_OXIDE_J2K_DEQUANTIZE_ENV_VAR).is_some())
}

#[cfg(feature = "cuda-oxide-j2k-idwt")]
pub(crate) fn cuda_oxide_j2k_idwt_enabled() -> bool {
    *CUDA_OXIDE_J2K_IDWT_ENABLED
        .get_or_init(|| std::env::var_os(CUDA_OXIDE_J2K_IDWT_ENV_VAR).is_some())
}

#[cfg(feature = "cuda-oxide-transcode")]
pub(crate) fn cuda_oxide_transcode_enabled() -> bool {
    *CUDA_OXIDE_TRANSCODE_ENABLED
        .get_or_init(|| std::env::var_os(CUDA_OXIDE_TRANSCODE_ENV_VAR).is_some())
}

pub(crate) fn ensure_kernel_ptx_built(kernel: CudaKernel) -> Result<(), CudaError> {
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
        | CudaKernel::TranscodeDwt97IdctI16Batch
        | CudaKernel::TranscodeDwt97RowLiftBatch
        | CudaKernel::TranscodeDwt97RowLiftBatchCoop
        | CudaKernel::TranscodeDwt97ColumnLiftBatch
        | CudaKernel::TranscodeDwt97QuantizeCodeblocks
        | CudaKernel::TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch
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

#[cfg(feature = "cuda-oxide-copy-u8")]
pub(crate) fn ensure_cuda_oxide_copy_u8_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_COPY_U8_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: "cuda-oxide CopyU8 PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_COPY_U8 on a Linux cuda-oxide host to require it".to_string(),
        })
    }
}

#[cfg(feature = "cuda-oxide-j2k-encode")]
pub(crate) fn ensure_cuda_oxide_j2k_encode_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_J2K_ENCODE_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: "cuda-oxide J2K encode PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_J2K_ENCODE on a Linux cuda-oxide host to require it".to_string(),
        })
    }
}

#[cfg(feature = "cuda-oxide-j2k-decode-store")]
pub(crate) fn ensure_cuda_oxide_j2k_decode_store_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_J2K_DECODE_STORE_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: "cuda-oxide J2K decode store PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_J2K_DECODE_STORE on a Linux cuda-oxide host to require it".to_string(),
        })
    }
}

#[cfg(feature = "cuda-oxide-j2k-dequantize")]
pub(crate) fn ensure_cuda_oxide_j2k_dequantize_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_J2K_DEQUANTIZE_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: "cuda-oxide J2K dequantize PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_J2K_DEQUANTIZE on a Linux cuda-oxide host to require it".to_string(),
        })
    }
}

#[cfg(feature = "cuda-oxide-j2k-idwt")]
pub(crate) fn ensure_cuda_oxide_j2k_idwt_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_J2K_IDWT_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: "cuda-oxide J2K IDWT PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_J2K_IDWT on a Linux cuda-oxide host to require it".to_string(),
        })
    }
}

#[cfg(feature = "cuda-oxide-transcode")]
pub(crate) fn ensure_cuda_oxide_transcode_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_TRANSCODE_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: "cuda-oxide transcode PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_TRANSCODE on a Linux cuda-oxide host to require it".to_string(),
        })
    }
}

pub(crate) const J2K_ENCODE_PTX_BUILT_FROM_CUDA: bool = cfg!(j2k_cuda_j2k_encode_ptx_built);

pub(crate) const HTJ2K_ENCODE_PTX_BUILT_FROM_CUDA: bool = cfg!(j2k_cuda_htj2k_encode_ptx_built);

/// True when the coefficient-domain transcode kernels were compiled by nvcc
/// (the runner). When false, build.rs wrote a placeholder PTX, so dispatch
/// returns a typed error instead of loading a non-existent kernel.
pub(crate) const TRANSCODE_PTX_BUILT_FROM_CUDA: bool = cfg!(j2k_cuda_transcode_ptx_built);

#[cfg(feature = "cuda-oxide-copy-u8")]
pub(crate) const CUDA_OXIDE_COPY_U8_PTX_BUILT: bool = cfg!(j2k_cuda_oxide_copy_u8_built);

#[cfg(feature = "cuda-oxide-j2k-encode")]
pub(crate) const CUDA_OXIDE_J2K_ENCODE_PTX_BUILT: bool = cfg!(j2k_cuda_oxide_j2k_encode_built);

#[cfg(feature = "cuda-oxide-j2k-decode-store")]
pub(crate) const CUDA_OXIDE_J2K_DECODE_STORE_PTX_BUILT: bool =
    cfg!(j2k_cuda_oxide_j2k_decode_store_built);

#[cfg(feature = "cuda-oxide-j2k-dequantize")]
pub(crate) const CUDA_OXIDE_J2K_DEQUANTIZE_PTX_BUILT: bool =
    cfg!(j2k_cuda_oxide_j2k_dequantize_built);

#[cfg(feature = "cuda-oxide-j2k-idwt")]
pub(crate) const CUDA_OXIDE_J2K_IDWT_PTX_BUILT: bool = cfg!(j2k_cuda_oxide_j2k_idwt_built);

#[cfg(feature = "cuda-oxide-transcode")]
pub(crate) const CUDA_OXIDE_TRANSCODE_PTX_BUILT: bool = cfg!(j2k_cuda_oxide_transcode_built);

/// Whether the coefficient-domain transcode kernels were compiled (runner).
/// Backends check this to fall back to the scalar oracle when the kernels are
/// unavailable (e.g. a non-nvcc build) instead of attempting a device launch.
#[must_use]
pub fn transcode_kernels_built() -> bool {
    if TRANSCODE_PTX_BUILT_FROM_CUDA {
        return true;
    }
    #[cfg(feature = "cuda-oxide-transcode")]
    {
        cuda_oxide_transcode_enabled() && CUDA_OXIDE_TRANSCODE_PTX_BUILT
    }
    #[cfg(not(feature = "cuda-oxide-transcode"))]
    {
        false
    }
}
