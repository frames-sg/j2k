// SPDX-License-Identifier: Apache-2.0

//! JPEG 2000 inspect support for signinum.

#![deny(missing_docs)]

extern crate alloc;

mod backend;
mod batch;
mod decode;
mod encode;
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
pub mod adapter;

/// Borrowed view and decoder entry points.
pub mod view;
pub use view::{J2kCodec, J2kDecoder, J2kView};

pub use batch::{
    decode_tile_into_in_context, decode_tile_region_scaled_into_in_context, decode_tiles_into,
    decode_tiles_region_scaled_into, TileBatchError, TileBatchOptions, TileDecodeJob,
    TileRegionScaledDecodeJob,
};

pub use signinum_j2k_native::CpuDecodeParallelism;

pub use encode::{
    encode_j2k_lossless, encode_j2k_lossless_with_accelerator, j2k_lossless_decomposition_levels,
    j2k_lossless_decomposition_levels_for_options,
    j2k_lossless_decomposition_levels_for_progression, EncodeBackendPreference, EncodedJ2k,
    J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kProgressionOrder, ReversibleTransform,
};

pub use recode::{
    recode_j2k_to_htj2k_lossless, J2kToHtj2kMode, J2kToHtj2kOptions, J2kToHtj2kReport,
    ReencodedHtj2k,
};

pub use signinum_j2k_native::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kDeinterleaveToF32Job, J2kEncodeDispatchReport,
    J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardDwt53Level, J2kForwardDwt53Output,
    J2kForwardRctJob, J2kHtCodeBlockEncodeJob, J2kPacketizationBlockCodingMode,
    J2kPacketizationCodeBlock, J2kPacketizationEncodeJob, J2kPacketizationProgressionOrder,
    J2kPacketizationResolution, J2kPacketizationSubband, J2kQuantizeSubbandJob,
    J2kTier1CodeBlockEncodeJob,
};

pub use signinum_core::{
    BackendKind, BackendRequest, BufferError, CodecError, CompressedPayloadKind,
    CompressedTransferSyntax, DecodeOutcome, DecodeRowsError, DecoderContext, Downscale,
    ImageCodec, ImageDecode, ImageDecodeRows, PassthroughCandidate, PassthroughDecision,
    PassthroughRejectReason, PassthroughRequirements, PixelFormat, Rect, RowSink, TileBatchDecode,
};

pub(crate) mod parse;
