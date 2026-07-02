//! Shared traits and value types for the `j2k` workspace.
//!
//! Codec crates use this crate to expose common pixel formats, decode
//! outcomes, row sinks, caller-owned scratch pools, and CPU/GPU backend
//! selection contracts without depending on each other.

#![no_std]
#![warn(unreachable_pub)]

extern crate alloc;

/// Shared accelerator runtime contracts.
pub mod accelerator;
/// Backend selection and capability discovery.
pub mod backend;
/// Shared helpers for ordered tile-batch work.
pub mod batch;
mod buffer;
/// Reusable codec context wrappers.
pub mod context;
/// Shared device-output request policies.
pub mod device;
/// Common error classifications and helper error types.
pub mod error;
/// Compressed-byte passthrough eligibility checks.
pub mod passthrough;
/// Pixel layout and format descriptors.
pub mod pixel;
/// Row-streaming output sink trait.
pub mod row_sink;
/// Integer sample type descriptors.
pub mod sample;
/// Decode downscale options.
pub mod scale;
/// Caller-owned scratch pool trait.
pub mod scratch;
/// Facade traits implemented by codec crates.
pub mod traits;
/// Shared metadata and geometry types.
pub mod types;

pub use accelerator::{
    AcceleratorSession, DeviceMemoryRange, ExecutionStats, GpuAbi, SurfaceResidency,
};
pub use backend::{BackendCapabilities, BackendKind, BackendRequest, CpuFeatures};
pub use batch::{
    collect_indexed_batch_results, tile_batch_worker_count, IndexedBatchResult, TileBatchError,
    TileBatchOptions, TileDecodeJob, TileRegionDecodeJob, TileRegionScaledDecodeJob,
    TileScaledDecodeJob,
};
pub use buffer::{
    copy_tight_pixels_to_strided_output, ensure_allocation_within_cap, strided_output_len,
    strided_output_len_capped, validate_strided_output_buffer, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
pub use context::{CacheStats, CodecContext, DecoderContext};
pub use device::validate_cuda_surface_backend_request;
pub use error::{BufferError, CodecError, InputError, NotImplemented, Unsupported};
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
    submit_ready_device, DecodeRowsError, DeviceSubmission, DeviceSubmitSession, DeviceSurface,
    ImageCodec, ImageDecode, ImageDecodeDevice, ImageDecodeRows, ImageDecodeSubmit,
    ReadySubmission, TileBatchDecode, TileBatchDecodeDevice, TileBatchDecodeManyDevice,
    TileBatchDecodeSubmit, TileDecompress,
};
pub use types::{CodedUnitLayout, Colorspace, DecodeOutcome, Info, Rect, TileLayout};
