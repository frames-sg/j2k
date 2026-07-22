// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal-backed JPEG 2000 and HTJ2K decode and encode adapters.
//!
//! This crate wraps the CPU/native J2K implementation with optional
//! Metal-resident decode surfaces, batch decode sessions, and lossless encode
//! helpers on macOS. Non-macOS builds keep the same API surface and return
//! `Error::MetalUnavailable` for explicit Metal-only requests.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

mod batch;
mod batch_allocation;
mod batch_decoder;
#[cfg(target_os = "macos")]
mod buffer_pool;
#[cfg(any(test, target_os = "macos"))]
mod classic;
#[cfg(target_os = "macos")]
mod compute;
mod decoder;
#[cfg(target_os = "macos")]
mod direct;
mod encode;
mod error;
#[cfg(any(test, target_os = "macos"))]
mod ht;
#[cfg(target_os = "macos")]
mod hybrid;
#[cfg(any(test, target_os = "macos"))]
mod idwt;
#[cfg(any(test, target_os = "macos"))]
mod mct;
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

use j2k_core::{Downscale, PixelFormat, Rect};

#[cfg(target_os = "macos")]
use j2k_metal_support::{
    checked_blit_command_encoder, checked_command_buffer, checked_private_buffer,
    checked_shared_buffer_with_bytes, commit_and_wait,
};
#[cfg(target_os = "macos")]
use metal::Buffer;

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

#[cfg(target_os = "macos")]
#[doc(hidden)]
pub fn benchmark_region_scaled_direct_plan_prepare(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<(), Error> {
    hybrid::benchmark_region_scaled_direct_plan_prepare(input, fmt, roi, scale)
}

#[cfg(not(target_os = "macos"))]
#[doc(hidden)]
pub fn benchmark_region_scaled_direct_plan_prepare(
    _input: &[u8],
    _fmt: PixelFormat,
    _roi: Rect,
    _scale: Downscale,
) -> Result<(), Error> {
    Err(Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
#[doc(hidden)]
pub fn benchmark_private_buffer_with_bytes(
    session: &MetalBackendSession,
    bytes: &[u8],
) -> Result<Buffer, Error> {
    if bytes.is_empty() {
        return Err(Error::MetalKernel {
            message: "J2K Metal benchmark private input upload is empty".to_string(),
        });
    }
    let byte_len = u64::try_from(bytes.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal benchmark private input length exceeds u64".to_string(),
    })?;
    let map_allocation =
        |source| error::metal_kernel_support_error("J2K Metal benchmark buffer allocation", source);
    let upload =
        checked_shared_buffer_with_bytes(session.device(), bytes).map_err(map_allocation)?;
    let private = checked_private_buffer(session.device(), bytes.len()).map_err(map_allocation)?;
    let runtime = session.runtime()?;
    let command_buffer = checked_command_buffer(runtime.command_queue()).map_err(map_allocation)?;
    let blit = checked_blit_command_encoder(&command_buffer).map_err(map_allocation)?;
    blit.copy_from_buffer(&upload, 0, &private, 0, byte_len);
    blit.end_encoding();
    commit_and_wait(&command_buffer).map_err(|error| {
        error::metal_kernel_support_error(
            format!("J2K Metal benchmark private input upload failed: {error}"),
            error,
        )
    })?;
    Ok(private)
}

#[cfg(target_os = "macos")]
#[doc(hidden)]
pub fn benchmark_overwrite_private_buffer_with_bytes(
    session: &MetalBackendSession,
    dst: &Buffer,
    bytes: &[u8],
) -> Result<(), Error> {
    if bytes.is_empty() {
        return Err(Error::MetalKernel {
            message: "J2K Metal benchmark private input overwrite is empty".to_string(),
        });
    }
    let byte_len = u64::try_from(bytes.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal benchmark private input overwrite length exceeds u64".to_string(),
    })?;
    if byte_len > dst.length() {
        return Err(Error::MetalKernel {
            message: "J2K Metal benchmark private input overwrite exceeds destination buffer"
                .to_string(),
        });
    }
    let map_allocation =
        |source| error::metal_kernel_support_error("J2K Metal benchmark buffer allocation", source);
    let upload =
        checked_shared_buffer_with_bytes(session.device(), bytes).map_err(map_allocation)?;
    let runtime = session.runtime()?;
    let command_buffer = checked_command_buffer(runtime.command_queue()).map_err(map_allocation)?;
    let blit = checked_blit_command_encoder(&command_buffer).map_err(map_allocation)?;
    blit.copy_from_buffer(&upload, 0, dst, 0, byte_len);
    blit.end_encoding();
    commit_and_wait(&command_buffer).map_err(|error| {
        error::metal_kernel_support_error(
            format!("J2K Metal benchmark private input overwrite failed: {error}"),
            error,
        )
    })?;
    Ok(())
}

pub use j2k::{J2kContext, J2kScratchPool};
