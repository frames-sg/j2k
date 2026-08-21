use crate::driver::CuResult;
#[cfg(j2k_cuda_oxide_enabled)]
use crate::error::CudaError;
use std::sync::OnceLock;

pub(crate) const CUDA_SUCCESS: CuResult = 0;

pub(crate) const CUDA_ERROR_NOT_READY: CuResult = 600;

#[cfg(j2k_cuda_oxide_enabled)]
pub(crate) const REQUIRE_CUDA_OXIDE_BUILD_ENV_VAR: &str = "J2K_REQUIRE_CUDA_OXIDE_BUILD";

pub(crate) static CUDA_STAGE_TIMINGS_DISABLED: OnceLock<bool> = OnceLock::new();

pub(crate) fn cuda_stage_timings_disabled() -> bool {
    *CUDA_STAGE_TIMINGS_DISABLED
        .get_or_init(|| std::env::var_os("J2K_CUDA_DISABLE_STAGE_TIMINGS").is_some())
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
