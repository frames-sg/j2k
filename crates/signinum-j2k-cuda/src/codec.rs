// SPDX-License-Identifier: Apache-2.0

use core::convert::Infallible;

use signinum_core::{
    BackendRequest, Downscale, ImageCodec, PixelFormat, ReadySubmission, Rect, TileBatchDecode,
    TileBatchDecodeDevice, TileBatchDecodeManyDevice, TileBatchDecodeSubmit,
};
use signinum_j2k::{
    adapter::device_plan::{DeviceDecodePlan, DeviceDecodeRequest},
    J2kCodec as CpuCodec, J2kContext as CpuJ2kContext, J2kDecoder as CpuDecoder,
    J2kScratchPool as CpuJ2kScratchPool,
};

use crate::runtime::{validate_surface_request, wrap_surface};
use crate::{CudaSession, Error, J2kDecoder, Surface};

/// Marker type implementing tile-batch CUDA surface decode traits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Codec;

impl ImageCodec for Codec {
    type Error = Error;
    type Warning = Infallible;
    type Pool = CpuJ2kScratchPool;
}

impl Codec {
    fn decode_tile_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuJ2kContext>,
        session: &mut CudaSession,
        pool: &mut CpuJ2kScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if matches!(backend, BackendRequest::Cuda) {
            let mut decoder = J2kDecoder::new(input)?;
            return decoder.decode_to_device_with_session(fmt, session);
        }
        let dims = CpuDecoder::inspect(input)?.dimensions;
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        CpuCodec::decode_tile(ctx, pool, input, &mut out, stride, fmt)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_tile_region_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuJ2kContext>,
        session: &mut CudaSession,
        pool: &mut CpuJ2kScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if matches!(backend, BackendRequest::Cuda) {
            let mut decoder = J2kDecoder::new(input)?;
            return decoder.decode_region_to_device_with_session(fmt, roi, session);
        }
        let dims = DeviceDecodePlan::for_image(
            CpuDecoder::inspect(input)?.dimensions,
            DeviceDecodeRequest::Region { roi },
        )?
        .output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        CpuCodec::decode_tile_region(ctx, pool, input, &mut out, stride, fmt, roi)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_tile_scaled_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuJ2kContext>,
        session: &mut CudaSession,
        pool: &mut CpuJ2kScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if matches!(backend, BackendRequest::Cuda) {
            let mut decoder = J2kDecoder::new(input)?;
            return decoder.decode_scaled_to_device_with_session(fmt, scale, session);
        }
        let dims = DeviceDecodePlan::for_image(
            CpuDecoder::inspect(input)?.dimensions,
            DeviceDecodeRequest::Scaled { scale },
        )?
        .output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        CpuCodec::decode_tile_scaled(ctx, pool, input, &mut out, stride, fmt, scale)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_tile_region_scaled_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuJ2kContext>,
        session: &mut CudaSession,
        pool: &mut CpuJ2kScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if matches!(backend, BackendRequest::Cuda) {
            let mut decoder = J2kDecoder::new(input)?;
            return decoder.decode_region_scaled_to_device_with_session(fmt, roi, scale, session);
        }
        let dims = DeviceDecodePlan::for_image(
            CpuDecoder::inspect(input)?.dimensions,
            DeviceDecodeRequest::RegionScaled { roi, scale },
        )?
        .output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        CpuCodec::decode_tile_region_scaled(ctx, pool, input, &mut out, stride, fmt, roi, scale)?;
        wrap_surface(out, dims, fmt, backend, session)
    }
}

impl TileBatchDecodeSubmit for Codec {
    type Context = CpuJ2kContext;
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
    type Context = CpuJ2kContext;
    type DeviceSurface = Surface;
}

impl TileBatchDecodeManyDevice for Codec {
    type Context = CpuJ2kContext;
    type DeviceSurface = Surface;

    fn decode_tiles_to_device(
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Vec<Self::DeviceSurface>, Self::Error> {
        validate_surface_request(backend)?;
        let mut session = CudaSession::default();
        inputs
            .iter()
            .map(|input| {
                Self::decode_tile_to_surface_impl(ctx, &mut session, pool, input, fmt, backend)
            })
            .collect()
    }
}
