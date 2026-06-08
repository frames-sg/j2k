// SPDX-License-Identifier: Apache-2.0

//! JPEG 2000 inspect support for signinum.

#![deny(missing_docs)]

extern crate alloc;

mod backend;
mod batch;
mod decode;
mod encode;
#[doc(hidden)]
pub mod native_bridge;
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

pub use adapter::adaptive_route::{
    J2kAdaptiveBackendRequest, J2kAdaptiveBenchmarkEvidence, J2kAdaptiveBenchmarkScope,
    J2kAdaptiveBenchmarks, J2kAdaptiveCodecMode, J2kAdaptiveGatePolicy, J2kAdaptiveOperation,
    J2kAdaptiveOutputResidency, J2kAdaptiveQualityMode, J2kAdaptiveRcaFinding,
    J2kAdaptiveRcaReason, J2kAdaptiveRouteKind, J2kAdaptiveRoutePlanner, J2kAdaptiveRouteReport,
    J2kAdaptiveStage, J2kAdaptiveStageDecision, J2kAdaptiveStageGateStatus, J2kAdaptiveStageOwner,
    J2kAdaptiveWorkload,
};

pub use adapter::encode_stage::{
    CpuOnlyJ2kEncodeStageAccelerator, EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock,
    IrreversibleQuantizationStep, IrreversibleQuantizationSubbandScales, J2kCodeBlockSegment,
    J2kCodeBlockStyle, J2kDeinterleaveToF32Job, J2kEncodeDispatchReport, J2kEncodeStageAccelerator,
    J2kForwardDwt53Job, J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Job,
    J2kForwardDwt97Level, J2kForwardDwt97Output, J2kForwardIctJob, J2kForwardRctJob,
    J2kHtCodeBlockEncodeJob, J2kHtSubbandEncodeJob, J2kHtj2kTileEncodeJob,
    J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationEncodeJob,
    J2kPacketizationPacketDescriptor, J2kPacketizationProgressionOrder, J2kPacketizationResolution,
    J2kPacketizationSubband, J2kQuantizeSubbandJob, J2kSubBandType, J2kTier1CodeBlockEncodeJob,
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image, PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
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

pub use signinum_core::{
    BackendKind, BackendRequest, BufferError, CodecError, CompressedPayloadKind,
    CompressedTransferSyntax, DecodeOutcome, DecodeRowsError, DecoderContext, Downscale,
    ImageCodec, ImageDecode, ImageDecodeRows, PassthroughCandidate, PassthroughDecision,
    PassthroughRejectReason, PassthroughRequirements, PixelFormat, Rect, RowSink, TileBatchDecode,
};

pub(crate) mod parse;
