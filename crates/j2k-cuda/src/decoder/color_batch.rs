// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(test, feature = "cuda-runtime"))]
use core::cell::Cell;

#[cfg(feature = "cuda-runtime")]
mod batch_execution;
#[cfg(feature = "cuda-runtime")]
mod finish;
#[cfg(feature = "cuda-runtime")]
mod host_owners;
#[cfg(feature = "cuda-runtime")]
pub(crate) mod native_batch;
#[cfg(feature = "cuda-runtime")]
mod single;
#[cfg(feature = "cuda-runtime")]
mod store;

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) use self::batch_execution::{
    decode_color_cuda_resident_batch_surfaces_with_profile, finalize_color_batch_decode_report,
};
#[cfg(feature = "cuda-runtime")]
use self::finish::{
    finish_color_cuda_resident_surface_with_component_work, FinishColorCudaResidentSurfaceRequest,
};
#[cfg(feature = "cuda-runtime")]
use self::host_owners::{append_color_payload_to_shared, take_component_work};
#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) use self::single::{
    decode_color_cuda_resident_region_scaled_surface, decode_color_cuda_resident_region_surface,
    decode_color_cuda_resident_scaled_surface, decode_color_cuda_resident_surface_with_profile,
};
#[cfg(feature = "cuda-runtime")]
use self::store::{
    can_fuse_mct_store_for_stores, dispatch_color_store, prepare_rgb8_mct_batch_store,
    rgb8_mct_batch_store_target, run_color_mct, validate_color_stores, ColorStoreInputs,
};
#[cfg(feature = "cuda-runtime")]
use super::decode_profile::aggregate_decode_reports;
#[cfg(feature = "cuda-runtime")]
use super::plan::build_cuda_htj2k_color_plans_from_bytes_with_profile;
#[cfg(feature = "cuda-runtime")]
use super::resident::{
    can_batch_color_idwt, decode_cuda_component_subbands_with_resources,
    finish_cuda_component_decode, pooled_cuda_buffer, run_color_component_idwt_batches,
    run_component_cleanup_dequant_batches, run_cuda_component_idwt_steps,
};
#[cfg(feature = "cuda-runtime")]
use super::{
    cuda_error, cuda_range_storage, profile, Arc, BackendKind, CudaBufferPool,
    CudaComponentDecodeWork, CudaDecodedComponent, CudaDeviceBuffer, CudaExecutionStats,
    CudaHtj2kColorDecodePlans, CudaHtj2kProfileReport, CudaQueuedIdwtBatch, CudaSession,
    CudaSurfaceStats, Error, J2kDecoder, NativeDecoderContext, PixelFormat, Rect, Storage, Surface,
    SurfaceResidency, CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE, CUDA_HTJ2K_KERNELS_NOT_READY,
};
#[cfg(feature = "cuda-runtime")]
use crate::allocation::HostPhaseBudget;

#[cfg(all(test, feature = "cuda-runtime"))]
std::thread_local! {
    pub(super) static CUDA_HTJ2K_BATCH_DECODE_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(all(test, feature = "cuda-runtime"))]
pub(crate) fn testing_reset_cuda_htj2k_batch_decode_calls() {
    CUDA_HTJ2K_BATCH_DECODE_CALLS.with(|calls| calls.set(0));
}

#[cfg(all(test, feature = "cuda-runtime"))]
pub(crate) fn testing_cuda_htj2k_batch_decode_calls() -> usize {
    CUDA_HTJ2K_BATCH_DECODE_CALLS.with(Cell::get)
}
