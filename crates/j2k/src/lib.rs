// SPDX-License-Identifier: Apache-2.0

//! JPEG 2000 / HTJ2K codec APIs for the Rust programming language, including
//! inspect, decode, encode, recode, and GPU-aware backend dispatch.
//!
//! This is the public `j2k` facade crate. CPU JPEG 2000 / HTJ2K decode and
//! encode APIs live here, while CUDA and Apple Metal GPU adapters use the
//! backend traits and encode-stage contracts re-exported by this crate.

#![deny(missing_docs)]

extern crate alloc;

mod backend;
mod batch;
mod decode;
mod encode;
mod parallelism;
mod recode;

/// Reusable JPEG 2000 decode context.
pub mod context;
pub use context::J2kContext;

/// JPEG 2000 error type.
pub mod error;
pub use error::J2kError;

/// Caller-owned JPEG 2000 scratch pool.
pub mod scratch;
pub use scratch::J2kScratchPool;

/// Adapter-facing planning APIs shared with GPU crates.
///
/// This module is public so device adapters can use the same route planning and
/// encode-stage contracts as the facade without depending on root re-exports.
pub mod adapter;

/// Borrowed view and decoder entry points.
pub mod view;
pub use view::{J2kCodec, J2kDecoder, J2kRowDecodeOptions, J2kView};

pub use batch::{
    decode_tile_into_in_context, decode_tile_region_into_in_context,
    decode_tile_region_scaled_into_in_context, decode_tile_scaled_into_in_context,
    decode_tiles_into, decode_tiles_region_into, decode_tiles_region_scaled_into,
    decode_tiles_scaled_into, TileBatchError, TileBatchOptions, TileDecodeJob, TileRegionDecodeJob,
    TileRegionScaledDecodeJob, TileScaledDecodeJob,
};

pub use parallelism::CpuDecodeParallelism;

pub use encode::{
    encode_j2k_lossless, encode_j2k_lossless_with_accelerator, encode_j2k_lossy,
    encode_j2k_lossy_with_accelerator, j2k_lossless_decomposition_levels,
    j2k_lossless_decomposition_levels_for_options,
    j2k_lossless_decomposition_levels_for_progression, EncodeBackendPreference, EncodedJ2k,
    EncodedLossyJ2k, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
    J2kLosslessSamples, J2kLossyEncodeOptions, J2kLossyEncodeReport, J2kLossySamples,
    J2kMarkerSegment, J2kProgressionOrder, J2kQualityLayer, J2kRateTarget, ReversibleTransform,
};

pub use recode::{
    recode_j2k_to_htj2k_lossless, J2kToHtj2kMode, J2kToHtj2kOptions, J2kToHtj2kReport,
    ReencodedHtj2k,
};

pub use j2k_core::{
    BackendKind, BackendRequest, BufferError, CodecError, CompressedPayloadKind,
    CompressedTransferSyntax, DecodeOutcome, DecodeRowsError, DecoderContext, Downscale,
    ImageCodec, ImageDecode, ImageDecodeRows, PassthroughCandidate, PassthroughDecision,
    PassthroughRejectReason, PassthroughRequirements, PixelFormat, Rect, RowSink, TileBatchDecode,
};

pub(crate) mod parse;
