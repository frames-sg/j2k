// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal-backed JPEG 2000 and HTJ2K decode and encode adapters.
//!
//! This crate wraps the CPU/native J2K implementation with optional
//! Metal-resident decode surfaces, batch decode sessions, and lossless encode
//! helpers on macOS. Non-macOS builds keep the same API surface and return
//! `Error::MetalUnavailable` for explicit Metal-only requests.
//!
//! The 0.9 expert surface uses `objc2-metal` directly: owned devices, queues,
//! command buffers, and resources are retained protocol objects, while
//! borrowed parameters are protocol-object references. This intentionally
//! replaces the former `metal-rs` types without an adapter layer.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

mod batch;
mod batch_allocation;
mod batch_decoder;
mod bench_support;
#[cfg(target_os = "macos")]
mod buffer_pool;
#[cfg(any(test, target_os = "macos"))]
mod classic;
mod decoder;
#[cfg(target_os = "macos")]
mod direct;
mod encode;
#[cfg(target_os = "macos")]
mod engine;
mod error;
mod generated;
#[cfg(any(test, target_os = "macos"))]
mod ht;
#[cfg(target_os = "macos")]
mod hybrid;
#[cfg(any(test, target_os = "macos"))]
mod idwt;
#[cfg(any(test, target_os = "macos"))]
mod mct;
#[cfg(target_os = "macos")]
mod metal_types;
mod profile;
#[cfg(target_os = "macos")]
mod profile_env;
#[cfg(any(test, target_os = "macos"))]
mod resident_limits;
mod routing;
mod session;
#[cfg(any(test, target_os = "macos"))]
mod store;
mod surface;
mod tile_batch;

pub use j2k_core::SurfaceResidency;
#[cfg(target_os = "macos")]
pub use j2k_metal_support::{MetalImageDestination, MetalImageLayout};

#[doc(hidden)]
pub use self::batch::MetalSubmission;
pub use self::batch_decoder::{
    MetalBatchDecodeResult, MetalBatchDecoder, MetalBatchGroup, MetalBatchGroupCompletion,
    MetalBatchGroupError, MetalBatchGroupParts,
};
#[cfg(target_os = "macos")]
pub use self::batch_decoder::{
    MetalResidentBatch, SubmittedMetalGroupDecodeInto, SubmittedMetalPreparedBatch,
};
pub use self::decoder::{
    Codec, DecodeOperation, DecodeRouteReport, DecodeSurfaceWithReport, J2kDecoder, MetalDecodeOp,
    MetalDecodeRequest,
};
pub use self::error::{
    Error, MetalDirectFallbackReason, MetalKernelRetryClass, NativeBackendError,
};
#[doc(hidden)]
pub use self::profile::MetalDecodeDispatchReport;
pub use self::session::{MetalBackendSession, MetalSession};
pub use self::surface::download_surfaces_packed;
pub(crate) use self::surface::Storage;
pub use self::surface::Surface;
pub use self::tile_batch::MetalTileBatch;
#[cfg(target_os = "macos")]
pub use buffer_pool::{MetalBufferPoolDiagnostics, MetalBufferPoolsDiagnostics};

#[doc(hidden)]
pub use batch::{benchmark_group_region_scaled_requests, BenchmarkGroupedRequests};
#[doc(hidden)]
pub use encode::{
    encode_lossless_batch_with_report, MetalLosslessBufferEncodeBatchOutcome,
    MetalLosslessBufferEncodeOutcome, MetalLosslessEncodeBatchStats, MetalLosslessEncodeOutcome,
    MetalLosslessEncodeStageStats,
};
pub use encode::{
    submit_lossless_batch, submit_lossless_batch_to_metal, validate_lossless_roundtrip_on_metal,
    validate_lossless_roundtrip_on_metal_with_session, MetalEncodeInputStaging,
    MetalEncodeStageAccelerator, MetalEncodedJ2k, MetalLosslessEncodeBatchRequest,
    MetalLosslessEncodeConfig, MetalLosslessEncodeResidency, MetalLosslessEncodeTile,
    SubmittedJ2kLosslessMetalBufferEncodeBatch, SubmittedJ2kLosslessMetalEncodeBatch,
};

#[doc(hidden)]
pub use bench_support::benchmark_region_scaled_direct_plan_prepare;
#[cfg(target_os = "macos")]
#[doc(hidden)]
pub use bench_support::{
    benchmark_overwrite_private_buffer_with_bytes, benchmark_private_buffer_with_bytes,
};

pub use j2k::{J2kContext, J2kScratchPool};
