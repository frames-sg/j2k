// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use j2k_core::{BackendKind, PixelFormat};

use crate::{CudaSession, Error, Surface};

#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{CudaDeviceBuffer, CudaError};
#[cfg(all(test, feature = "cuda-runtime"))]
use j2k_cuda_runtime::{CudaJpegEntropyCheckpoint, CudaJpegRgb8Sampling};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::ColorSpace;
use j2k_jpeg::Decoder as CpuDecoder;

#[cfg(feature = "cuda-runtime")]
use crate::session::LeasedOwnedPacket;
#[cfg(feature = "cuda-runtime")]
use crate::surface::{CudaJpegDecodePath, CudaSurfaceStats, Storage};

pub(crate) fn unsupported_owned_cuda_output_format() -> Error {
    Error::UnsupportedCudaRequest {
        reason: "J2K CUDA JPEG owned decode currently supports full-frame RGB8 output only",
    }
}

#[cfg(feature = "cuda-runtime")]
const INVALID_CHUNKED_ENTROPY_DIAGNOSTIC_ARGUMENT: &str =
    "J2K CUDA JPEG chunked entropy diagnostic config or input is invalid";

#[cfg(feature = "cuda-runtime")]
mod plan;
#[cfg(all(test, feature = "cuda-runtime"))]
use plan::cuda_entropy_checkpoints_with_cap;
#[cfg(feature = "cuda-runtime")]
use plan::{build_cuda_rgb8_plan_data, fast_rgb8_packet_parts};

#[cfg(feature = "cuda-runtime")]
mod diagnostic;
#[cfg(feature = "cuda-runtime")]
pub(crate) use diagnostic::diagnose_owned_cuda_420_entropy;
#[cfg(feature = "cuda-runtime")]
pub use diagnostic::CudaJpegChunkedEntropyReport;

#[cfg(feature = "cuda-runtime")]
pub(crate) fn decode_owned_cuda_rgb8_from_decoder(
    decoder: &CpuDecoder<'_>,
    session: &mut CudaSession,
) -> Result<Surface, Error> {
    let info = decoder.info();
    validate_owned_cuda_rgb8_preflight(info.dimensions, info.color_space)?;
    let operation_gate = session.jpeg_host_operation_gate();
    let _operation = operation_gate
        .lock()
        .map_err(|_| Error::JpegHostOperationPoisoned)?;
    let context = session.cuda_context()?;
    let pinned_upload = context
        .begin_pinned_upload_operation()
        .map_err(crate::runtime::cuda_error)?;
    let pinned_accounting = session.reserve_pinned_upload_retention(&context, &pinned_upload)?;
    let packet = resolve_owned_rgb8_packet_from_decoder(decoder, session)?;
    let result = decode_owned_cuda_rgb8_from_packet(&packet, info.dimensions, session, &context);
    pinned_accounting.finish(result)
}

#[cfg(feature = "cuda-runtime")]
fn decode_owned_cuda_rgb8_from_packet(
    packet: &LeasedOwnedPacket,
    dimensions: (u32, u32),
    session: &mut CudaSession,
    context: &j2k_cuda_runtime::CudaContext,
) -> Result<Surface, Error> {
    let packet_parts = fast_rgb8_packet_parts(&packet.packet);
    let plan_data = build_cuda_rgb8_plan_data(&packet_parts, dimensions, session)?;
    let plan = plan_data.as_plan();
    let runtime_external_live = session.owned_host_live_bytes()?;
    let output = context
        .decode_jpeg_rgb8_owned_with_external_live(&plan, runtime_external_live)
        .map_err(cuda_owned_decode_error)?;
    let (buffer, stats) = output.into_parts();
    Ok(Surface {
        backend: BackendKind::Cuda,
        dimensions,
        fmt: PixelFormat::Rgb8,
        pitch_bytes: dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel(),
        stats: CudaSurfaceStats {
            kernel_dispatches: stats.kernel_dispatches(),
            copy_kernel_dispatches: stats.copy_kernel_dispatches(),
            decode_kernel_dispatches: stats.decode_kernel_dispatches(),
            hardware_decode: false,
            decode_path: CudaJpegDecodePath::OwnedCuda,
        },
        storage: Storage::Cuda(buffer),
    })
}

#[cfg(feature = "cuda-runtime")]
pub(crate) fn decode_owned_cuda_rgb8_from_decoder_into(
    decoder: &CpuDecoder<'_>,
    session: &mut CudaSession,
    output: &CudaDeviceBuffer,
    pitch_bytes: usize,
) -> Result<CudaSurfaceStats, Error> {
    let info = decoder.info();
    validate_owned_cuda_rgb8_preflight(info.dimensions, info.color_space)?;
    validate_owned_cuda_output_layout(info.dimensions, output, pitch_bytes)?;
    let operation_gate = session.jpeg_host_operation_gate();
    let _operation = operation_gate
        .lock()
        .map_err(|_| Error::JpegHostOperationPoisoned)?;
    let context = session.cuda_context()?;
    let pinned_upload = context
        .begin_pinned_upload_operation()
        .map_err(crate::runtime::cuda_error)?;
    let pinned_accounting = session.reserve_pinned_upload_retention(&context, &pinned_upload)?;
    context
        .validate_jpeg_output_buffer_context(output)
        .map_err(cuda_owned_decode_error)?;
    let packet = resolve_owned_rgb8_packet_from_decoder(decoder, session)?;
    let packet_parts = fast_rgb8_packet_parts(&packet.packet);
    let plan_data = build_cuda_rgb8_plan_data(&packet_parts, info.dimensions, session)?;
    let runtime_external_live = session.owned_host_live_bytes()?;
    let stats = context
        .decode_jpeg_rgb8_owned_into_with_external_live(
            &plan_data.as_plan(),
            output,
            pitch_bytes,
            runtime_external_live,
        )
        .map_err(cuda_owned_decode_error);
    let stats = pinned_accounting.finish(stats)?;
    Ok(CudaSurfaceStats {
        kernel_dispatches: stats.kernel_dispatches(),
        copy_kernel_dispatches: stats.copy_kernel_dispatches(),
        decode_kernel_dispatches: stats.decode_kernel_dispatches(),
        hardware_decode: false,
        decode_path: CudaJpegDecodePath::OwnedCuda,
    })
}

#[cfg(not(feature = "cuda-runtime"))]
pub(crate) fn decode_owned_cuda_rgb8_from_decoder(
    _decoder: &CpuDecoder<'_>,
    _session: &mut CudaSession,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(feature = "cuda-runtime")]
fn resolve_owned_rgb8_packet(
    bytes: &[u8],
    session: &mut CudaSession,
) -> Result<LeasedOwnedPacket, Error> {
    require_ready_packet(session.resolve_owned_packet(bytes)?)
}

#[cfg(feature = "cuda-runtime")]
fn resolve_owned_rgb8_packet_from_decoder(
    decoder: &CpuDecoder<'_>,
    session: &mut CudaSession,
) -> Result<LeasedOwnedPacket, Error> {
    require_ready_packet(session.resolve_owned_packet_from_decoder(decoder)?)
}

#[cfg(feature = "cuda-runtime")]
fn require_ready_packet(packet: Option<LeasedOwnedPacket>) -> Result<LeasedOwnedPacket, Error> {
    packet.ok_or(Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG decode currently supports baseline 8-bit YCbCr 4:2:0, 4:2:2, or 4:4:4 RGB8 output",
        })
}

#[cfg(feature = "cuda-runtime")]
fn validate_owned_cuda_rgb8_preflight(
    dimensions: (u32, u32),
    color_space: ColorSpace,
) -> Result<(), Error> {
    if color_space != ColorSpace::YCbCr {
        return Err(Error::UnsupportedCudaRequest {
            reason: "J2K-owned CUDA JPEG decode currently requires a YCbCr 4:2:0, 4:2:2, or 4:4:4 fast packet shape",
        });
    }
    let addressable = u64::from(dimensions.0)
        .checked_mul(u64::from(dimensions.1))
        .and_then(|pixels| pixels.checked_mul(3))
        .is_some_and(|bytes| bytes <= u64::from(u32::MAX) + 1);
    if !addressable {
        return Err(Error::UnsupportedCudaRequest {
            reason:
                "J2K-owned CUDA JPEG decode requires RGB8 output addressable by u32 byte offsets",
        });
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
fn validate_owned_cuda_output_layout(
    dimensions: (u32, u32),
    output: &CudaDeviceBuffer,
    pitch_bytes: usize,
) -> Result<(), Error> {
    let row_bytes = usize::try_from(dimensions.0)
        .ok()
        .and_then(|width| width.checked_mul(PixelFormat::Rgb8.bytes_per_pixel()))
        .ok_or(Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG RGB8 output row size overflows host addressability",
        })?;
    if pitch_bytes < row_bytes {
        return Err(Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG RGB8 output pitch is smaller than one packed row",
        });
    }
    if pitch_bytes > u32::MAX as usize {
        return Err(Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG RGB8 output pitch exceeds kernel u32 addressing",
        });
    }
    let required = usize::try_from(dimensions.1.saturating_sub(1))
        .ok()
        .and_then(|rows| rows.checked_mul(pitch_bytes))
        .and_then(|prefix| prefix.checked_add(row_bytes))
        .ok_or(Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG RGB8 output extent overflows host addressability",
        })?;
    if required > (u32::MAX as usize).saturating_add(1) {
        return Err(Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG RGB8 pitched output exceeds kernel u32 addressing",
        });
    }
    if output.byte_len() < required {
        return Err(Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG RGB8 output buffer is too small for the requested pitch",
        });
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
fn cuda_owned_decode_error(error: CudaError) -> Error {
    match error {
        CudaError::Unavailable { .. } => Error::CudaUnavailable,
        CudaError::InvalidArgument { .. } => Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG owned decode cannot handle this image or runtime build",
        },
        other => crate::runtime::cuda_error(other),
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_chunked_entropy_diagnostic_error(error: CudaError) -> Error {
    match error {
        CudaError::Unavailable { .. } => Error::CudaUnavailable,
        CudaError::InvalidArgument { .. } => Error::UnsupportedCudaRequest {
            reason: INVALID_CHUNKED_ENTROPY_DIAGNOSTIC_ARGUMENT,
        },
        other => crate::runtime::cuda_error(other),
    }
}

#[cfg(all(test, feature = "cuda-runtime"))]
mod tests;
