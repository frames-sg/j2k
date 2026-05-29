//! Shared traits and value types for the `signinum` workspace.
//!
//! Codec crates use this crate to expose common pixel formats, decode
//! outcomes, row sinks, caller-owned scratch pools, and CPU/GPU backend
//! selection contracts without depending on each other.

#![no_std]
#![deny(missing_docs)]
#![warn(unreachable_pub)]

extern crate alloc;

/// Backend selection and host capability detection.
pub mod backend;
/// Tile-batch ordering and worker-count helpers.
pub mod batch;
mod buffer;
/// Reusable codec context traits and wrappers.
pub mod context;
/// Shared error traits and buffer/input error types.
pub mod error;
/// Compressed-payload passthrough eligibility types.
pub mod passthrough;
/// Pixel format and layout descriptors.
pub mod pixel;
/// Row-streaming output sink trait.
pub mod row_sink;
/// Sample type markers for typed pixel rows.
pub mod sample;
/// Reduced-resolution decode scale factors.
pub mod scale;
/// Caller-owned scratch pool trait.
pub mod scratch;
/// Codec, decode, device, batch, and decompression traits.
pub mod traits;
/// Shared image metadata, rectangles, and decode outcomes.
pub mod types;

pub use backend::{BackendCapabilities, BackendKind, BackendRequest, CpuFeatures};
pub use batch::{
    collect_indexed_batch_results, tile_batch_worker_count, IndexedBatchResult, TileBatchOptions,
};
pub use buffer::{
    copy_tight_pixels_to_strided_output, strided_output_len, validate_strided_output_buffer,
};
pub use context::{CacheStats, CodecContext, DecoderContext};
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
    DecodeRowsError, DeviceSubmission, DeviceSurface, ImageCodec, ImageDecode, ImageDecodeDevice,
    ImageDecodeRows, ImageDecodeSubmit, ReadySubmission, TileBatchDecode, TileBatchDecodeDevice,
    TileBatchDecodeManyDevice, TileBatchDecodeSubmit, TileDecompress,
};
pub use types::{CodedUnitLayout, Colorspace, DecodeOutcome, Info, Rect, TileLayout, WarningKind};
