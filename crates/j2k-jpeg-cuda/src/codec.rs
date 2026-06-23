// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    submit_ready_device, BackendRequest, Downscale, ImageCodec, PixelFormat, ReadySubmission, Rect,
    TileBatchDecodeDevice, TileBatchDecodeManyDevice, TileBatchDecodeSubmit,
};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::CudaDeviceBuffer;
use j2k_jpeg::{
    decode_tile_into_in_context, decode_tile_region_into_in_context,
    decode_tile_region_scaled_into_in_context, decode_tile_scaled_into_in_context,
    Decoder as CpuDecoder, DecoderContext as CpuDecoderContext, JpegDecodeOp, JpegDecodeRequest,
    JpegResolvedDecode, JpegResolvedDecodePath, ScratchPool as CpuScratchPool,
    Warning as CpuWarning,
};

use crate::owned_decode::decode_owned_cuda_rgb8;
#[cfg(feature = "cuda-runtime")]
use crate::owned_decode::decode_owned_cuda_rgb8_into;
use crate::runtime::{validate_surface_request, wrap_surface};
use crate::{CudaSession, Error, Surface};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// JPEG codec marker used by J2K's generic CUDA decode traits.
pub struct Codec;

impl ImageCodec for Codec {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = CpuScratchPool;
}

fn rejected_decode_path_error(backend: BackendRequest, reason: &'static str) -> Error {
    match backend {
        BackendRequest::Cuda => Error::UnsupportedCudaRequest { reason },
        other => Error::UnsupportedBackend { request: other },
    }
}

impl Codec {
    #[cfg(feature = "cuda-runtime")]
    /// Run experimental chunked JPEG entropy self-sync diagnostics for a 4:2:0 RGB8 tile.
    ///
    /// This does not decode pixels and does not affect production CUDA routing.
    pub fn diagnose_tile_rgb8_chunked_entropy_with_session(
        input: &[u8],
        config: j2k_cuda_runtime::CudaJpegChunkedEntropyConfig,
        session: &mut CudaSession,
    ) -> Result<j2k_cuda_runtime::CudaJpegChunkedEntropyReport, Error> {
        crate::owned_decode::diagnose_owned_cuda_420_entropy(input, config, session)
    }

    #[cfg(feature = "cuda-runtime")]
    /// Decode one full JPEG tile to caller-owned CUDA RGB8 memory using a session.
    ///
    /// This is a strict J2K-owned CUDA-kernel path and currently supports
    /// full-tile RGB8 fast 4:2:0, 4:2:2, and 4:4:4 YCbCr JPEG inputs.
    pub fn decode_tile_rgb8_into_cuda_buffer_with_session(
        input: &[u8],
        output: &CudaDeviceBuffer,
        pitch_bytes: usize,
        session: &mut CudaSession,
    ) -> Result<crate::CudaSurfaceStats, Error> {
        let dimensions = CpuDecoder::inspect(input)?.dimensions;
        decode_owned_cuda_rgb8_into(input, dimensions, session, output, pitch_bytes)
    }

    /// Decode many JPEG tiles to J2K surfaces using a caller-owned CUDA session.
    pub fn decode_tiles_to_device_with_session(
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
        session: &mut CudaSession,
    ) -> Result<Vec<Surface>, Error> {
        let mut ctx = j2k_core::DecoderContext::<CpuDecoderContext>::new();
        let mut pool = CpuScratchPool::new();
        Self::decode_tiles_to_device_with_session_in_context(
            &mut ctx, &mut pool, inputs, fmt, backend, session,
        )
    }

    fn decode_tiles_to_device_with_session_in_context(
        ctx: &mut j2k_core::DecoderContext<CpuDecoderContext>,
        pool: &mut CpuScratchPool,
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
        session: &mut CudaSession,
    ) -> Result<Vec<Surface>, Error> {
        validate_surface_request(backend)?;
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        inputs
            .iter()
            .map(|input| Self::decode_tile_to_surface_impl(ctx, session, pool, input, fmt, backend))
            .collect()
    }

    fn decode_tile_to_surface_impl(
        ctx: &mut j2k_core::DecoderContext<CpuDecoderContext>,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let resolved = JpegResolvedDecode::inspect(
            input,
            JpegDecodeRequest {
                backend,
                fmt,
                op: JpegDecodeOp::Full,
            },
        )?;
        if resolved.path == JpegResolvedDecodePath::OwnedCudaRgb8 {
            return decode_owned_cuda_rgb8(input, resolved.capabilities.info.dimensions, session);
        }
        if let JpegResolvedDecodePath::Rejected { backend, reason } = resolved.path {
            return Err(rejected_decode_path_error(backend, reason));
        }
        let dims = (resolved.output_rect.w, resolved.output_rect.h);
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        decode_tile_into_in_context(input, ctx.codec_mut(), pool, &mut out, stride, fmt)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_tile_region_to_surface_impl(
        ctx: &mut j2k_core::DecoderContext<CpuDecoderContext>,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if backend == BackendRequest::Cuda {
            return Err(Error::UnsupportedCudaRequest {
                reason: "J2K CUDA JPEG owned decode does not support region output",
            });
        }
        let dims = (roi.w, roi.h);
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        decode_tile_region_into_in_context(
            input,
            ctx.codec_mut(),
            pool,
            &mut out,
            stride,
            fmt,
            roi.into(),
        )?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_tile_scaled_to_surface_impl(
        ctx: &mut j2k_core::DecoderContext<CpuDecoderContext>,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if backend == BackendRequest::Cuda {
            return Err(Error::UnsupportedCudaRequest {
                reason: "J2K CUDA JPEG owned decode does not support scaled output",
            });
        }
        let source_dims = CpuDecoder::inspect(input)?.dimensions;
        let dims = (
            source_dims.0.div_ceil(scale.denominator()),
            source_dims.1.div_ceil(scale.denominator()),
        );
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        decode_tile_scaled_into_in_context(
            input,
            ctx.codec_mut(),
            pool,
            &mut out,
            stride,
            fmt,
            scale,
        )?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_tile_region_scaled_to_surface_impl(
        ctx: &mut j2k_core::DecoderContext<CpuDecoderContext>,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if backend == BackendRequest::Cuda {
            return Err(Error::UnsupportedCudaRequest {
                reason: "J2K CUDA JPEG owned decode does not support scaled region output",
            });
        }
        let dims = {
            let scaled = roi.scaled_covering(scale);
            (scaled.w, scaled.h)
        };
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        decode_tile_region_scaled_into_in_context(
            input,
            ctx.codec_mut(),
            pool,
            &mut out,
            stride,
            fmt,
            roi.into(),
            scale,
        )?;
        wrap_surface(out, dims, fmt, backend, session)
    }
}

impl TileBatchDecodeSubmit for Codec {
    type Context = CpuDecoderContext;
    type Session = CudaSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = ReadySubmission<Surface, Error>;

    fn submit_tile_to_device(
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        Ok(submit_ready_device(session, |session| {
            Self::decode_tile_to_surface_impl(ctx, session, pool, input, fmt, backend)
        }))
    }

    fn submit_tile_region_to_device(
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        Ok(submit_ready_device(session, |session| {
            Self::decode_tile_region_to_surface_impl(ctx, session, pool, input, fmt, roi, backend)
        }))
    }

    fn submit_tile_scaled_to_device(
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        Ok(submit_ready_device(session, |session| {
            Self::decode_tile_scaled_to_surface_impl(ctx, session, pool, input, fmt, scale, backend)
        }))
    }

    fn submit_tile_region_scaled_to_device(
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        Ok(submit_ready_device(session, |session| {
            Self::decode_tile_region_scaled_to_surface_impl(
                ctx, session, pool, input, fmt, roi, scale, backend,
            )
        }))
    }
}

impl TileBatchDecodeDevice for Codec {
    type Context = CpuDecoderContext;
    type DeviceSurface = Surface;
}

impl TileBatchDecodeManyDevice for Codec {
    type Context = CpuDecoderContext;
    type DeviceSurface = Surface;

    fn decode_tiles_to_device(
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Vec<Self::DeviceSurface>, Self::Error> {
        let mut session = CudaSession::default();
        Self::decode_tiles_to_device_with_session_in_context(
            ctx,
            pool,
            inputs,
            fmt,
            backend,
            &mut session,
        )
    }
}
