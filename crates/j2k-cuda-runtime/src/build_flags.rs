use crate::driver::CuResult;
#[cfg(j2k_cuda_oxide_enabled)]
use crate::error::CudaError;
use std::{os::raw::c_uint, sync::OnceLock};

pub(crate) const CUDA_SUCCESS: CuResult = 0;

pub(crate) const PINNED_POOLED_I16_UPLOAD_MAX_BYTES: usize = 4 * 1024 * 1024;

pub(crate) const DWT97_ROW_LIFT_MAX_WIDTH: i32 = 1024;

pub(crate) const DWT97_ROW_LIFT_COOP_THREADS_X: c_uint = 128;

pub(crate) const DWT97_ROW_LIFT_COOP_ROWS_PER_BLOCK: c_uint = 4;

pub(crate) const CUDA_IDWT_TRACE_ENV_VAR: &str = "J2K_CUDA_IDWT_TRACE";

#[cfg(j2k_cuda_oxide_enabled)]
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

#[cfg(j2k_cuda_oxide_enabled)]
fn ensure_cuda_oxide_ptx_built(built: bool, display_name: &str) -> Result<(), CudaError> {
    if built {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: format!(
                "{display_name} PTX was not built; set {REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR} on a Linux cuda-oxide host to require it"
            ),
        })
    }
}

macro_rules! cuda_oxide_ptx_guard {
    (feature = $feature:literal, $ensure_fn:ident, $built_const:ident, $display_name:literal, $built_cfg:meta) => {
        #[cfg(feature = $feature)]
        pub(crate) fn $ensure_fn() -> Result<(), CudaError> {
            ensure_cuda_oxide_ptx_built($built_const, $display_name)
        }

        #[cfg(feature = $feature)]
        pub(crate) const $built_const: bool = cfg!($built_cfg);
    };
}

cuda_oxide_ptx_guard!(
    feature = "cuda-oxide-copy-u8",
    ensure_cuda_oxide_copy_u8_ptx_built,
    CUDA_OXIDE_COPY_U8_PTX_BUILT,
    "cuda-oxide CopyU8",
    j2k_cuda_oxide_copy_u8_built
);
cuda_oxide_ptx_guard!(
    feature = "cuda-oxide-j2k-encode",
    ensure_cuda_oxide_j2k_encode_ptx_built,
    CUDA_OXIDE_J2K_ENCODE_PTX_BUILT,
    "cuda-oxide J2K encode",
    j2k_cuda_oxide_j2k_encode_built
);
cuda_oxide_ptx_guard!(
    feature = "cuda-oxide-j2k-decode-store",
    ensure_cuda_oxide_j2k_decode_store_ptx_built,
    CUDA_OXIDE_J2K_DECODE_STORE_PTX_BUILT,
    "cuda-oxide J2K decode store",
    j2k_cuda_oxide_j2k_decode_store_built
);
cuda_oxide_ptx_guard!(
    feature = "cuda-oxide-j2k-classic-decode",
    ensure_cuda_oxide_j2k_classic_decode_ptx_built,
    CUDA_OXIDE_J2K_CLASSIC_DECODE_PTX_BUILT,
    "cuda-oxide classic J2K decode",
    j2k_cuda_oxide_j2k_classic_decode_built
);
cuda_oxide_ptx_guard!(
    feature = "cuda-oxide-j2k-dequantize",
    ensure_cuda_oxide_j2k_dequantize_ptx_built,
    CUDA_OXIDE_J2K_DEQUANTIZE_PTX_BUILT,
    "cuda-oxide J2K dequantize",
    j2k_cuda_oxide_j2k_dequantize_built
);
cuda_oxide_ptx_guard!(
    feature = "cuda-oxide-j2k-idwt",
    ensure_cuda_oxide_j2k_idwt_ptx_built,
    CUDA_OXIDE_J2K_IDWT_PTX_BUILT,
    "cuda-oxide J2K IDWT",
    j2k_cuda_oxide_j2k_idwt_built
);
cuda_oxide_ptx_guard!(
    feature = "cuda-oxide-j2k-ml",
    ensure_cuda_oxide_j2k_ml_ptx_built,
    CUDA_OXIDE_J2K_ML_PTX_BUILT,
    "cuda-oxide j2k-ml",
    j2k_cuda_oxide_j2k_ml_built
);
cuda_oxide_ptx_guard!(
    feature = "cuda-oxide-transcode",
    ensure_cuda_oxide_transcode_ptx_built,
    CUDA_OXIDE_TRANSCODE_PTX_BUILT,
    "cuda-oxide transcode",
    j2k_cuda_oxide_transcode_built
);
cuda_oxide_ptx_guard!(
    feature = "cuda-oxide-htj2k-decode",
    ensure_cuda_oxide_htj2k_decode_ptx_built,
    CUDA_OXIDE_HTJ2K_DECODE_PTX_BUILT,
    "cuda-oxide HTJ2K decode",
    j2k_cuda_oxide_htj2k_decode_built
);
cuda_oxide_ptx_guard!(
    feature = "cuda-oxide-htj2k-encode",
    ensure_cuda_oxide_htj2k_encode_ptx_built,
    CUDA_OXIDE_HTJ2K_ENCODE_PTX_BUILT,
    "cuda-oxide HTJ2K encode",
    j2k_cuda_oxide_htj2k_encode_built
);
cuda_oxide_ptx_guard!(
    feature = "cuda-oxide-jpeg-decode",
    ensure_cuda_oxide_jpeg_decode_ptx_built,
    CUDA_OXIDE_JPEG_DECODE_PTX_BUILT,
    "cuda-oxide JPEG decode",
    j2k_cuda_oxide_jpeg_decode_built
);
cuda_oxide_ptx_guard!(
    feature = "cuda-oxide-jpeg-encode",
    ensure_cuda_oxide_jpeg_encode_ptx_built,
    CUDA_OXIDE_JPEG_ENCODE_PTX_BUILT,
    "cuda-oxide JPEG encode",
    j2k_cuda_oxide_jpeg_encode_built
);

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
