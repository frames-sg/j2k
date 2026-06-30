// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG 2000 inspect support for j2k.

#![deny(missing_docs)]

extern crate alloc;

mod backend;
mod batch;
mod decode;
mod encode;
mod metadata;
mod parallelism;
mod recode;
mod wrap;

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

pub use decode::{
    J2kComponentPlane, J2kDecodedColorSpace, J2kDecodedComponents, J2kDecodedNativeComponents,
    J2kNativeComponentPlane,
};

pub use parallelism::CpuDecodeParallelism;

pub use metadata::{
    J2kChannelAssociation, J2kChannelDefinition, J2kChannelType, J2kColorSpec, J2kComponentInfo,
    J2kComponentMapping, J2kComponentMappingType, J2kFileMetadata, J2kPaletteColumn,
    J2kPaletteMetadata, J2kSupportInfo,
};

pub use encode::{
    encode_j2k_lossless, encode_j2k_lossless_components, encode_j2k_lossless_typed_components,
    encode_j2k_lossless_with_accelerator, encode_j2k_lossless_with_roi_regions, encode_j2k_lossy,
    encode_j2k_lossy_with_accelerator, encode_j2k_lossy_with_roi_regions,
    j2k_lossless_decomposition_levels, j2k_lossless_decomposition_levels_for_options,
    j2k_lossless_decomposition_levels_for_progression, EncodeBackendPreference, EncodedJ2k,
    EncodedLossyJ2k, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessComponentPlane,
    J2kLosslessComponentSamples, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kLosslessTypedComponentPlane, J2kLosslessTypedComponentSamples, J2kLossyEncodeOptions,
    J2kLossyEncodeReport, J2kLossySamples, J2kMarkerSegment, J2kProgressionOrder, J2kQualityLayer,
    J2kRateTarget, J2kRoiRegion, ReversibleTransform,
};

pub use recode::{
    recode_j2k_to_htj2k_lossless, J2kToHtj2kMode, J2kToHtj2kOptions, J2kToHtj2kReport,
    ReencodedHtj2k,
};

pub use wrap::{wrap_j2k_codestream, J2kFileBoxMetadata, J2kFileColorSpec, J2kFileWrapOptions};

pub use parse::{extract_j2k_codestream_payload, J2kCodestreamPayload};

pub use j2k_core::{
    BackendKind, BackendRequest, BufferError, CodecError, CompressedPayloadKind,
    CompressedTransferSyntax, DecodeOutcome, DecodeRowsError, DecoderContext, Downscale,
    ImageCodec, ImageDecode, ImageDecodeRows, PassthroughCandidate, PassthroughDecision,
    PassthroughRejectReason, PassthroughRequirements, PixelFormat, Rect, RowSink, TileBatchDecode,
};

pub(crate) mod parse;
