// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    checked_surface_len, submit_ready_device, BackendRequest, Downscale, ImageCodec, PixelFormat,
    ReadySubmission, Rect, TileBatchDecodeDevice, TileBatchDecodeManyDevice, TileBatchDecodeSubmit,
    TileRegionScaledDeviceDecodeRequest, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::CudaDeviceBuffer;
use j2k_jpeg::{
    decode_tile_region_into_in_context, decode_tile_region_scaled_into_in_context,
    decode_tile_scaled_into_in_context, Decoder as CpuDecoder, DecoderContext as CpuDecoderContext,
    JpegCapabilityReport, JpegDecodeOp, JpegDecodeRequest, JpegResolvedDecode,
    JpegResolvedDecodePath, ScratchPool as CpuScratchPool, Warning as CpuWarning,
};

use crate::allocation::{try_collect_results_exact, try_vec_filled};
use crate::batch::CudaJpegBatch;
use crate::owned_decode::decode_owned_cuda_rgb8_from_decoder;
#[cfg(feature = "cuda-runtime")]
use crate::owned_decode::decode_owned_cuda_rgb8_from_decoder_into;
use crate::runtime::{validate_surface_request, wrap_surface};
use crate::{CudaSession, Error, Surface};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// JPEG codec marker used by J2K's generic CUDA decode traits.
pub struct Codec;

struct RegionScaledSurfaceRequest<'a> {
    ctx: &'a mut CpuDecoderContext,
    session: &'a mut CudaSession,
    pool: &'a mut CpuScratchPool,
    input: &'a [u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    backend: BackendRequest,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Debug, Clone, Copy)]
/// Caller-owned CUDA output target for one full-frame RGB8 JPEG decode.
pub struct CudaJpegDecodeOutputTile<'a> {
    /// Baseline JPEG input bytes.
    pub input: &'a [u8],
    /// CUDA output buffer that receives tightly packed RGB8 rows.
    pub output: &'a CudaDeviceBuffer,
    /// Number of bytes between consecutive output rows.
    pub pitch_bytes: usize,
}

#[doc(hidden)]
impl ImageCodec for Codec {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = crate::ScratchPool;
}

fn rejected_decode_path_error(backend: BackendRequest, reason: &'static str) -> Error {
    match backend {
        BackendRequest::Cuda => Error::UnsupportedCudaRequest { reason },
        other => Error::UnsupportedBackend { request: other },
    }
}

impl Codec {
    #[cfg(feature = "cuda-runtime")]
    #[doc(hidden)]
    /// Run experimental chunked JPEG entropy self-sync diagnostics for a 4:2:0 RGB8 tile.
    ///
    /// This does not decode pixels and does not affect production CUDA routing.
    pub fn diagnose_tile_rgb8_chunked_entropy_with_session(
        input: &[u8],
        config: j2k_cuda_runtime::CudaJpegChunkedEntropyConfig,
        session: &mut CudaSession,
    ) -> Result<crate::CudaJpegChunkedEntropyReport, Error> {
        crate::owned_decode::diagnose_owned_cuda_420_entropy(input, config, session)
    }

    #[cfg(feature = "cuda-runtime")]
    /// Decode one full JPEG tile to caller-owned CUDA RGB8 memory using a session.
    ///
    /// This is a strict J2K-owned CUDA-kernel path and currently supports
    /// full-tile RGB8 fast 4:2:0, 4:2:2, and 4:4:4 YCbCr JPEG inputs.
    #[doc(hidden)]
    pub fn decode_tile_rgb8_into_cuda_buffer_with_session(
        input: &[u8],
        output: &CudaDeviceBuffer,
        pitch_bytes: usize,
        session: &mut CudaSession,
    ) -> Result<crate::CudaSurfaceStats, Error> {
        let decoder = CpuDecoder::new(input)?;
        decode_owned_cuda_rgb8_from_decoder_into(&decoder, session, output, pitch_bytes)
    }

    #[cfg(feature = "cuda-runtime")]
    /// Decode full JPEG tiles to caller-owned CUDA RGB8 memory using a session.
    ///
    /// This is a strict J2K-owned CUDA-kernel path and currently supports
    /// full-tile RGB8 fast 4:2:0, 4:2:2, and 4:4:4 YCbCr JPEG inputs. Returned
    /// stats preserve the input tile order. The returned collection keeps its
    /// exact vector capacity charged to `session` until it is dropped or its
    /// owning iterator is dropped.
    #[doc(hidden)]
    pub fn decode_tiles_rgb8_into_cuda_buffers_with_session(
        tiles: &[CudaJpegDecodeOutputTile<'_>],
        session: &mut CudaSession,
    ) -> Result<CudaJpegBatch<crate::CudaSurfaceStats>, Error> {
        let mut batch = CudaJpegBatch::try_with_capacity(
            session,
            tiles.len(),
            "CUDA JPEG decode batch statistics",
        )?;
        for tile in tiles {
            let stats = Self::decode_tile_rgb8_into_cuda_buffer_with_session(
                tile.input,
                tile.output,
                tile.pitch_bytes,
                session,
            )?;
            batch.try_push(stats)?;
        }
        Ok(batch)
    }

    /// Decode many JPEG tiles to J2K surfaces using a caller-owned CUDA session.
    ///
    /// The returned collection keeps its exact vector capacity charged to
    /// `session`; each host-backed surface separately retains its exact host
    /// allocation charge until that surface is dropped.
    pub fn decode_tiles_to_device_with_session(
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
        session: &mut CudaSession,
    ) -> Result<CudaJpegBatch<Surface>, Error> {
        let mut ctx = CpuDecoderContext::default();
        let mut pool = CpuScratchPool::new();
        Self::decode_tiles_to_device_with_session_in_context(
            &mut ctx, &mut pool, inputs, fmt, backend, session,
        )
    }

    fn decode_tiles_to_device_with_session_in_context(
        ctx: &mut CpuDecoderContext,
        pool: &mut CpuScratchPool,
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
        session: &mut CudaSession,
    ) -> Result<CudaJpegBatch<Surface>, Error> {
        validate_surface_request(backend)?;
        let mut batch = CudaJpegBatch::try_with_capacity(
            session,
            inputs.len(),
            "CUDA JPEG decode batch surfaces",
        )?;
        for input in inputs {
            let surface =
                Self::decode_tile_to_surface_impl(ctx, session, pool, input, fmt, backend)?;
            batch.try_push(surface)?;
        }
        Ok(batch)
    }

    fn decode_tile_to_surface_impl(
        ctx: &mut CpuDecoderContext,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let request = JpegDecodeRequest {
            backend,
            fmt,
            op: JpegDecodeOp::Full,
        };
        let decoder = CpuDecoder::from_view_in_context(j2k_jpeg::JpegView::parse(input)?, ctx)?;
        let resolved = JpegResolvedDecode::from_capabilities(
            JpegCapabilityReport::for_decoder(&decoder, request.capability()),
            request,
        );
        if resolved.path == JpegResolvedDecodePath::OwnedCudaRgb8 {
            return decode_owned_cuda_rgb8_from_decoder(&decoder, session);
        }
        if let JpegResolvedDecodePath::Rejected { backend, reason } = resolved.path {
            return Err(rejected_decode_path_error(backend, reason));
        }
        let dims = (resolved.output_rect.w, resolved.output_rect.h);
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        decoder.decode_into_with_scratch(pool, &mut out, stride, fmt)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_tile_region_to_surface_impl(
        ctx: &mut CpuDecoderContext,
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
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        decode_tile_region_into_in_context(
            input,
            ctx,
            pool,
            j2k_jpeg::TileDecodeOutput {
                out: &mut out,
                stride,
                fmt,
            },
            roi.into(),
        )?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_tile_scaled_to_surface_impl(
        ctx: &mut CpuDecoderContext,
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
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        decode_tile_scaled_into_in_context(
            input,
            ctx,
            pool,
            j2k_jpeg::TileDecodeOutput {
                out: &mut out,
                stride,
                fmt,
            },
            scale,
        )?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_tile_region_scaled_to_surface_impl(
        request: RegionScaledSurfaceRequest<'_>,
    ) -> Result<Surface, Error> {
        let RegionScaledSurfaceRequest {
            ctx,
            session,
            pool,
            input,
            fmt,
            roi,
            scale,
            backend,
        } = request;
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
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        decode_tile_region_scaled_into_in_context(
            input,
            ctx,
            pool,
            j2k_jpeg::TileDecodeOutput {
                out: &mut out,
                stride,
                fmt,
            },
            roi.into(),
            scale,
        )?;
        wrap_surface(out, dims, fmt, backend, session)
    }
}

fn allocate_cpu_surface(dims: (u32, u32), fmt: PixelFormat) -> Result<(Vec<u8>, usize), Error> {
    let (stride, len) = checked_surface_len(
        dims,
        fmt.bytes_per_pixel(),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        "JPEG CUDA CPU fallback surface",
    )?;
    Ok((
        try_vec_filled(len, 0u8, "JPEG CUDA CPU fallback surface")?,
        stride,
    ))
}

#[doc(hidden)]
impl TileBatchDecodeSubmit for Codec {
    type Context = CpuDecoderContext;
    type Session = CudaSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = ReadySubmission<Surface, Error>;

    fn submit_tile_to_device(
        ctx: &mut Self::Context,
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
        ctx: &mut Self::Context,
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
        ctx: &mut Self::Context,
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
        ctx: &mut Self::Context,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        request: TileRegionScaledDeviceDecodeRequest<'_>,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let TileRegionScaledDeviceDecodeRequest {
            input,
            fmt,
            roi,
            scale,
            backend,
        } = request;
        validate_surface_request(backend)?;
        Ok(submit_ready_device(session, |session| {
            Self::decode_tile_region_scaled_to_surface_impl(RegionScaledSurfaceRequest {
                ctx,
                session,
                pool,
                input,
                fmt,
                roi,
                scale,
                backend,
            })
        }))
    }
}

#[doc(hidden)]
impl TileBatchDecodeDevice for Codec {
    type Context = CpuDecoderContext;
    type DeviceSurface = Surface;
}

#[doc(hidden)]
impl TileBatchDecodeManyDevice for Codec {
    type Context = CpuDecoderContext;
    type DeviceSurface = Surface;

    fn decode_tiles_to_device(
        ctx: &mut Self::Context,
        pool: &mut Self::Pool,
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Vec<Self::DeviceSurface>, Self::Error> {
        let mut session = CudaSession::default();
        validate_surface_request(backend)?;
        try_collect_results_exact(
            inputs.iter().map(|input| {
                Self::decode_tile_to_surface_impl(ctx, &mut session, pool, input, fmt, backend)
            }),
            "CUDA JPEG decode batch surfaces",
        )
    }
}
