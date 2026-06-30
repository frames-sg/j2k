use crate::driver::CuResult;
#[cfg(any(
    feature = "cuda-oxide-copy-u8",
    feature = "cuda-oxide-j2k-encode",
    feature = "cuda-oxide-j2k-decode-store",
    feature = "cuda-oxide-j2k-dequantize",
    feature = "cuda-oxide-j2k-idwt",
    feature = "cuda-oxide-htj2k-decode",
    feature = "cuda-oxide-htj2k-encode",
    feature = "cuda-oxide-transcode",
    feature = "cuda-oxide-jpeg-decode",
    feature = "cuda-oxide-jpeg-encode"
))]
use crate::error::CudaError;
use std::{os::raw::c_uint, sync::OnceLock};

pub(crate) const CUDA_SUCCESS: CuResult = 0;

pub(crate) const PINNED_UPLOAD_STAGING_POOL_MAX: usize = 8;

pub(crate) const PINNED_POOLED_I16_UPLOAD_MAX_BYTES: usize = 4 * 1024 * 1024;

pub(crate) const DWT97_ROW_LIFT_MAX_WIDTH: i32 = 1024;

pub(crate) const DWT97_ROW_LIFT_COOP_THREADS_X: c_uint = 128;

pub(crate) const DWT97_ROW_LIFT_COOP_ROWS_PER_BLOCK: c_uint = 4;

pub(crate) const CUDA_IDWT_TRACE_ENV_VAR: &str = "J2K_CUDA_IDWT_TRACE";

#[cfg(any(
    feature = "cuda-oxide-copy-u8",
    feature = "cuda-oxide-j2k-encode",
    feature = "cuda-oxide-j2k-decode-store",
    feature = "cuda-oxide-j2k-dequantize",
    feature = "cuda-oxide-j2k-idwt",
    feature = "cuda-oxide-htj2k-decode",
    feature = "cuda-oxide-htj2k-encode",
    feature = "cuda-oxide-transcode",
    feature = "cuda-oxide-jpeg-decode",
    feature = "cuda-oxide-jpeg-encode"
))]
pub(crate) const REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR: &str = "J2K_REQUIRE_CUDA_OXIDE_BUILD";

pub(crate) const DWT97_FUSED_COLUMN_QUANTIZE_DISABLE_ENV_VAR: &str =
    "J2K_CUDA_DISABLE_DWT97_FUSED_COLUMN_QUANTIZE";

pub(crate) static CUDA_STAGE_TIMINGS_DISABLED: OnceLock<bool> = OnceLock::new();

pub(crate) static DWT97_FUSED_COLUMN_QUANTIZE_DISABLED: OnceLock<bool> = OnceLock::new();

pub(crate) fn cuda_stage_timings_disabled() -> bool {
    *CUDA_STAGE_TIMINGS_DISABLED
        .get_or_init(|| std::env::var_os("J2K_CUDA_DISABLE_STAGE_TIMINGS").is_some())
}

pub(crate) fn dwt97_fused_column_quantize_disabled() -> bool {
    *DWT97_FUSED_COLUMN_QUANTIZE_DISABLED
        .get_or_init(|| std::env::var_os(DWT97_FUSED_COLUMN_QUANTIZE_DISABLE_ENV_VAR).is_some())
}

#[cfg(feature = "cuda-oxide-copy-u8")]
pub(crate) fn ensure_cuda_oxide_copy_u8_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_COPY_U8_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: format!(
                "cuda-oxide CopyU8 PTX was not built; set {REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR} on a Linux cuda-oxide host to require it"
            ),
        })
    }
}

#[cfg(feature = "cuda-oxide-j2k-encode")]
pub(crate) fn ensure_cuda_oxide_j2k_encode_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_J2K_ENCODE_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: format!(
                "cuda-oxide J2K encode PTX was not built; set {REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR} on a Linux cuda-oxide host to require it"
            ),
        })
    }
}

#[cfg(feature = "cuda-oxide-j2k-decode-store")]
pub(crate) fn ensure_cuda_oxide_j2k_decode_store_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_J2K_DECODE_STORE_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: format!(
                "cuda-oxide J2K decode store PTX was not built; set {REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR} on a Linux cuda-oxide host to require it"
            ),
        })
    }
}

#[cfg(feature = "cuda-oxide-j2k-dequantize")]
pub(crate) fn ensure_cuda_oxide_j2k_dequantize_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_J2K_DEQUANTIZE_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: format!(
                "cuda-oxide J2K dequantize PTX was not built; set {REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR} on a Linux cuda-oxide host to require it"
            ),
        })
    }
}

#[cfg(feature = "cuda-oxide-j2k-idwt")]
pub(crate) fn ensure_cuda_oxide_j2k_idwt_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_J2K_IDWT_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: format!(
                "cuda-oxide J2K IDWT PTX was not built; set {REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR} on a Linux cuda-oxide host to require it"
            ),
        })
    }
}

#[cfg(feature = "cuda-oxide-transcode")]
pub(crate) fn ensure_cuda_oxide_transcode_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_TRANSCODE_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: format!(
                "cuda-oxide transcode PTX was not built; set {REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR} on a Linux cuda-oxide host to require it"
            ),
        })
    }
}

#[cfg(feature = "cuda-oxide-htj2k-decode")]
pub(crate) fn ensure_cuda_oxide_htj2k_decode_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_HTJ2K_DECODE_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: format!(
                "cuda-oxide HTJ2K decode PTX was not built; set {REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR} on a Linux cuda-oxide host to require it"
            ),
        })
    }
}

#[cfg(feature = "cuda-oxide-htj2k-encode")]
pub(crate) fn ensure_cuda_oxide_htj2k_encode_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_HTJ2K_ENCODE_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: format!(
                "cuda-oxide HTJ2K encode PTX was not built; set {REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR} on a Linux cuda-oxide host to require it"
            ),
        })
    }
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) fn ensure_cuda_oxide_jpeg_decode_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_JPEG_DECODE_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: format!(
                "cuda-oxide JPEG decode PTX was not built; set {REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR} on a Linux cuda-oxide host to require it"
            ),
        })
    }
}

#[cfg(feature = "cuda-oxide-jpeg-encode")]
pub(crate) fn ensure_cuda_oxide_jpeg_encode_ptx_built() -> Result<(), CudaError> {
    if CUDA_OXIDE_JPEG_ENCODE_PTX_BUILT {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: format!(
                "cuda-oxide JPEG encode PTX was not built; set {REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR} on a Linux cuda-oxide host to require it"
            ),
        })
    }
}

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

#[cfg(feature = "cuda-oxide-htj2k-decode")]
pub(crate) const CUDA_OXIDE_HTJ2K_DECODE_PTX_BUILT: bool = cfg!(j2k_cuda_oxide_htj2k_decode_built);

#[cfg(feature = "cuda-oxide-htj2k-encode")]
pub(crate) const CUDA_OXIDE_HTJ2K_ENCODE_PTX_BUILT: bool = cfg!(j2k_cuda_oxide_htj2k_encode_built);

#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) const CUDA_OXIDE_JPEG_DECODE_PTX_BUILT: bool = cfg!(j2k_cuda_oxide_jpeg_decode_built);

#[cfg(feature = "cuda-oxide-jpeg-encode")]
pub(crate) const CUDA_OXIDE_JPEG_ENCODE_PTX_BUILT: bool = cfg!(j2k_cuda_oxide_jpeg_encode_built);

/// Whether the coefficient-domain transcode CUDA Oxide kernels were compiled.
/// Backends check this to return a structured unavailable error on non-strict
/// non-CUDA/doc builds instead of attempting a device launch.
#[must_use]
pub fn transcode_kernels_built() -> bool {
    #[cfg(feature = "cuda-oxide-transcode")]
    {
        CUDA_OXIDE_TRANSCODE_PTX_BUILT
    }
    #[cfg(not(feature = "cuda-oxide-transcode"))]
    {
        false
    }
}
