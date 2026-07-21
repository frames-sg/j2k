//! Shared traits and value types for the `j2k` workspace.
//!
//! Codec crates use this crate to expose common pixel formats, decode
//! outcomes, row sinks, caller-owned scratch pools, and CPU/GPU backend
//! selection contracts without depending on each other.

#![no_std]
#![warn(unreachable_pub)]

extern crate alloc;

#[doc(hidden)]
#[macro_export]
macro_rules! __j2k_fnv1a64_init {
    () => {
        0xcbf2_9ce4_8422_2325_u64
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __j2k_fnv1a64_update {
    ($hash:ident, $byte:expr) => {{
        $hash ^= u64::from($byte);
        $hash = $hash.wrapping_mul(0x0000_0100_0000_01B3_u64);
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __j2k_fnv1a64_bytes {
    ($bytes:expr) => {{
        let mut hash = $crate::__j2k_fnv1a64_init!();
        for &byte in $bytes {
            $crate::__j2k_fnv1a64_update!(hash, byte);
        }
        hash
    }};
}

/// Shared accelerator runtime contracts.
pub mod accelerator;
/// Backend selection and capability discovery.
mod backend;
/// Shared helpers for ordered tile-batch work.
mod batch;
mod buffer;
/// Reusable codec context wrappers.
mod context;
/// Shared device-output request policies.
mod device;
/// Common error classifications and helper error types.
mod error;
mod host_allocation;
mod passthrough;
/// Pixel layout and format descriptors.
mod pixel;
mod row_sink;
/// Integer sample type descriptors.
mod sample;
/// Decode downscale options.
mod scale;
mod scratch;
/// Facade traits implemented by codec crates.
mod traits;
/// Shared metadata and geometry types.
mod types;

pub use accelerator::{
    AcceleratorSession, DeviceMemoryRange, ExecutionStats, SurfaceMetadata, SurfaceResidency,
};
pub use backend::{BackendCapabilities, BackendKind, BackendRequest, CpuFeatures};
#[doc(hidden)]
pub use batch::plan_ht_gpu_job_chunks;
pub use batch::{
    checked_batch_count_product, checked_batch_count_sum, tile_batch_worker_count,
    try_batch_reserve_for_push, try_batch_reserve_to, try_collect_indexed_batch_results,
    try_collect_ordered_batch_results_with_limits, BatchAllocationBudget, BatchAllocationRequest,
    BatchDecodeError, BatchInfrastructureError, BatchResultSlot, HtGpuJobChunk, HtGpuJobChunkEntry,
    HtGpuJobChunkLimit, HtGpuJobChunkLimits, HtGpuJobChunkPlan, HtGpuJobChunkPlanError,
    HtGpuJobChunkRequest, HtGpuJobPassBucket, IndexedBatchResult, TileBatchError, TileBatchOptions,
    TileDecodeJob, TileRegionDecodeJob, TileRegionScaledDecodeJob,
    TileRegionScaledDeviceDecodeRequest, TileScaledDecodeJob,
};
pub use buffer::{
    checked_surface_len, copy_tight_pixels_to_strided_output, ensure_allocation_within_cap,
    strided_output_len, strided_output_len_capped, validate_strided_output_buffer,
    DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
pub use context::{CacheStats, CodecContext};
pub use device::validate_cuda_surface_backend_request;
pub use error::{
    adapter_error_is_buffer_error, adapter_error_is_not_implemented, adapter_error_is_truncated,
    adapter_error_is_unsupported, AdapterErrorKind, AdapterErrorParts, BufferError, CodecError,
    InputError, NotImplemented, Unsupported,
};
#[doc(hidden)]
pub use host_allocation::{
    host_capacity_bytes, try_host_vec_filled, try_host_vec_from_slice, try_host_vec_resize,
    try_host_vec_with_capacity, HostAllocationBudget, HostAllocationError,
    HostAllocationLimitError,
};
pub use passthrough::{
    CompressedPayloadKind, CompressedTransferSyntax, PassthroughCandidate, PassthroughDecision,
    PassthroughRejectReason, PassthroughRequirements,
};
pub use pixel::{PixelFormat, PixelLayout};
pub use row_sink::RowSink;
pub use sample::{Sample, SampleType};
pub use scale::Downscale;
pub use scratch::ScratchPool;
pub use traits::{
    submit_ready_device, CpuBackedImageDecode, DecodeRowsError, DeviceSubmission,
    DeviceSubmitSession, DeviceSurface, ImageCodec, ImageDecode, ImageDecodeDevice,
    ImageDecodeRows, ImageDecodeSubmit, ReadySubmission, TileBatchDecode, TileBatchDecodeDevice,
    TileBatchDecodeManyDevice, TileBatchDecodeSubmit, TileDecompress,
};
pub use types::{CodedUnitLayout, Colorspace, DecodeOutcome, Info, Rect, TileLayout};
