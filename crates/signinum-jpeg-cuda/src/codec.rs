// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "cuda-runtime")]
use signinum_core::BackendKind;
use signinum_core::{
    BackendRequest, Downscale, ImageCodec, PixelFormat, ReadySubmission, Rect,
    TileBatchDecodeDevice, TileBatchDecodeManyDevice, TileBatchDecodeSubmit,
};
#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::CudaError;
use signinum_jpeg::{
    decode_tile_into_in_context, decode_tile_region_into_in_context,
    decode_tile_region_scaled_into_in_context, decode_tile_scaled_into_in_context,
    Decoder as CpuDecoder, DecoderContext as CpuDecoderContext, ScratchPool as CpuScratchPool,
    Warning as CpuWarning,
};

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
use crate::runtime::{validate_surface_request, wrap_surface};
#[cfg(feature = "cuda-runtime")]
use crate::surface::{CudaSurfaceStats, Storage};
use crate::{profile, CudaSession, Error, Surface};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// Marker type implementing tile-batch CUDA surface decode traits.
pub struct Codec;

impl ImageCodec for Codec {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = CpuScratchPool;
}

impl Codec {
    fn decode_tile_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuDecoderContext>,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let dims = CpuDecoder::inspect(input)?.dimensions;
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        decode_tile_into_in_context(input, ctx.codec_mut(), pool, &mut out, stride, fmt)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_tile_region_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuDecoderContext>,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
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
        ctx: &mut signinum_core::DecoderContext<CpuDecoderContext>,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let dims = (
            CpuDecoder::inspect(input)?
                .dimensions
                .0
                .div_ceil(scale.denominator()),
            CpuDecoder::inspect(input)?
                .dimensions
                .1
                .div_ceil(scale.denominator()),
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
        ctx: &mut signinum_core::DecoderContext<CpuDecoderContext>,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_to_surface_impl(ctx, session, pool, input, fmt, backend),
        ))
    }

    fn submit_tile_region_to_device(
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_region_to_surface_impl(ctx, session, pool, input, fmt, roi, backend),
        ))
    }

    fn submit_tile_scaled_to_device(
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_scaled_to_surface_impl(
                ctx, session, pool, input, fmt, scale, backend,
            ),
        ))
    }

    fn submit_tile_region_scaled_to_device(
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_region_scaled_to_surface_impl(
                ctx, session, pool, input, fmt, roi, scale, backend,
            ),
        ))
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Vec<Self::DeviceSurface>, Self::Error> {
        validate_surface_request(backend)?;
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let mut session = CudaSession::default();
        if let Some(surfaces) = try_decode_tiles_nvjpeg_batch(inputs, fmt, backend, &mut session)? {
            return Ok(surfaces);
        }

        inputs
            .iter()
            .map(|input| {
                Self::decode_tile_to_surface_impl(ctx, &mut session, pool, input, fmt, backend)
            })
            .collect()
    }
}

#[cfg(feature = "cuda-runtime")]
fn try_decode_tiles_nvjpeg_batch(
    inputs: &[&[u8]],
    fmt: PixelFormat,
    backend: BackendRequest,
    session: &mut CudaSession,
) -> Result<Option<Vec<Surface>>, Error> {
    if fmt != PixelFormat::Rgb8 || !matches!(backend, BackendRequest::Auto | BackendRequest::Cuda) {
        if profile::gpu_route_profile_enabled() {
            let request_s = format!("{backend:?}");
            let fmt_s = format!("{fmt:?}");
            let tiles_s = inputs.len().to_string();
            profile::emit_gpu_route_profile(
                "jpeg",
                "gpu_route",
                "cuda",
                &[
                    ("op", "batch_full"),
                    ("request", request_s.as_str()),
                    ("fmt", fmt_s.as_str()),
                    ("tiles", tiles_s.as_str()),
                    ("decision", "nvjpeg_batch_ineligible"),
                ],
            );
        }
        return Ok(None);
    }

    let mut batch_inputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let dimensions = CpuDecoder::inspect(input)?.dimensions;
        batch_inputs.push((*input, dimensions));
    }

    let context = match session.cuda_context() {
        Ok(context) => context,
        Err(_) if backend == BackendRequest::Auto => {
            if profile::gpu_route_profile_enabled() {
                let tiles_s = inputs.len().to_string();
                profile::emit_gpu_route_profile(
                    "jpeg",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "batch_full"),
                        ("request", "Auto"),
                        ("fmt", "Rgb8"),
                        ("tiles", tiles_s.as_str()),
                        ("decision", "nvjpeg_batch_fallback"),
                        ("reason", "cuda_unavailable"),
                    ],
                );
            }
            return Ok(None);
        }
        Err(error) => return Err(error),
    };

    match context.decode_jpeg_rgb8_batch_with_nvjpeg(&batch_inputs) {
        Ok(outputs) => {
            if profile::gpu_route_profile_enabled() {
                let tiles_s = outputs.len().to_string();
                profile::emit_gpu_route_profile(
                    "jpeg",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "batch_full"),
                        ("request", "AutoOrCuda"),
                        ("fmt", "Rgb8"),
                        ("tiles", tiles_s.as_str()),
                        ("decision", "nvjpeg_batch"),
                    ],
                );
            }
            let mut surfaces = Vec::with_capacity(outputs.len());
            for (output, (_, dimensions)) in outputs.into_iter().zip(batch_inputs) {
                let pitch_bytes = dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
                let (buffer, stats) = output.into_parts();
                surfaces.push(Surface {
                    backend: BackendKind::Cuda,
                    dimensions,
                    fmt: PixelFormat::Rgb8,
                    pitch_bytes,
                    stats: CudaSurfaceStats {
                        kernel_dispatches: stats.kernel_dispatches(),
                        copy_kernel_dispatches: stats.copy_kernel_dispatches(),
                        decode_kernel_dispatches: stats.decode_kernel_dispatches(),
                        hardware_decode: stats.used_hardware_decode(),
                    },
                    storage: Storage::Cuda(buffer),
                });
            }
            Ok(Some(surfaces))
        }
        Err(
            CudaError::NvjpegUnavailable { .. }
            | CudaError::Nvjpeg { .. }
            | CudaError::NvjpegDimensions { .. },
        ) => {
            if profile::gpu_route_profile_enabled() {
                let tiles_s = inputs.len().to_string();
                profile::emit_gpu_route_profile(
                    "jpeg",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "batch_full"),
                        ("request", "AutoOrCuda"),
                        ("fmt", "Rgb8"),
                        ("tiles", tiles_s.as_str()),
                        ("decision", "nvjpeg_batch_fallback"),
                        ("reason", "nvjpeg_unavailable_or_rejected"),
                    ],
                );
            }
            Ok(None)
        }
        Err(error) => Err(cuda_error(error)),
    }
}

#[cfg(not(feature = "cuda-runtime"))]
#[allow(clippy::unnecessary_wraps)]
fn try_decode_tiles_nvjpeg_batch(
    inputs: &[&[u8]],
    fmt: PixelFormat,
    backend: BackendRequest,
    _session: &mut CudaSession,
) -> Result<Option<Vec<Surface>>, Error> {
    if profile::gpu_route_profile_enabled() {
        let request_s = format!("{backend:?}");
        let fmt_s = format!("{fmt:?}");
        let tiles_s = inputs.len().to_string();
        profile::emit_gpu_route_profile(
            "jpeg",
            "gpu_route",
            "cuda",
            &[
                ("op", "batch_full"),
                ("request", request_s.as_str()),
                ("fmt", fmt_s.as_str()),
                ("tiles", tiles_s.as_str()),
                ("decision", "nvjpeg_batch_fallback"),
                ("reason", "cuda_runtime_feature_disabled"),
            ],
        );
    }
    Ok(None)
}
