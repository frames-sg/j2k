// SPDX-License-Identifier: MIT OR Apache-2.0

use core::convert::Infallible;

use j2k::{
    adapter::device_plan::{DeviceDecodePlan, DeviceDecodeRequest},
    J2kCodec as CpuCodec, J2kContext as CpuJ2kContext, J2kDecoder as CpuDecoder,
    J2kScratchPool as CpuJ2kScratchPool,
};
use j2k_core::{
    submit_ready_device, BackendRequest, Downscale, ImageCodec, PixelFormat, ReadySubmission, Rect,
    TileBatchDecode, TileBatchDecodeDevice, TileBatchDecodeManyDevice, TileBatchDecodeSubmit,
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
    fn supports_cuda_batch_format(fmt: PixelFormat) -> bool {
        matches!(
            fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16
        )
    }

    #[cfg(feature = "cuda-runtime")]
    fn decode_tiles_to_cuda_batch(
        inputs: &[&[u8]],
        fmt: PixelFormat,
        session: &mut CudaSession,
    ) -> Result<Vec<Surface>, Error> {
        J2kDecoder::decode_batch_to_device_with_session(inputs, fmt, session)
    }

    #[cfg(not(feature = "cuda-runtime"))]
    fn decode_tiles_to_cuda_batch(
        _inputs: &[&[u8]],
        _fmt: PixelFormat,
        _session: &mut CudaSession,
    ) -> Result<Vec<Surface>, Error> {
        Err(Error::CudaUnavailable)
    }

    fn decode_tile_to_surface_impl(
        ctx: &mut j2k_core::DecoderContext<CpuJ2kContext>,
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
        ctx: &mut j2k_core::DecoderContext<CpuJ2kContext>,
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
        ctx: &mut j2k_core::DecoderContext<CpuJ2kContext>,
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
        ctx: &mut j2k_core::DecoderContext<CpuJ2kContext>,
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
    type Context = CpuJ2kContext;
    type DeviceSurface = Surface;
}

impl TileBatchDecodeManyDevice for Codec {
    type Context = CpuJ2kContext;
    type DeviceSurface = Surface;

    fn decode_tiles_to_device(
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
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
        if matches!(backend, BackendRequest::Cuda) && Self::supports_cuda_batch_format(fmt) {
            return Self::decode_tiles_to_cuda_batch(inputs, fmt, &mut session);
        }

        inputs
            .iter()
            .map(|input| {
                Self::decode_tile_to_surface_impl(ctx, &mut session, pool, input, fmt, backend)
            })
            .collect()
    }
}

#[cfg(all(test, feature = "cuda-runtime"))]
mod tests {
    use j2k_core::{BackendRequest, DecoderContext, PixelFormat, TileBatchDecodeManyDevice};
    use j2k_test_support::{cuda_runtime_required, htj2k_rgb8_pattern_fixture};

    use super::{Codec, CpuJ2kContext, CpuJ2kScratchPool};
    use crate::decoder::{
        testing_cuda_htj2k_batch_decode_calls, testing_reset_cuda_htj2k_batch_decode_calls,
    };
    use crate::{Error, SurfaceResidency};

    #[test]
    fn explicit_cuda_rgb_many_decode_uses_batch_api_once() {
        testing_reset_cuda_htj2k_batch_decode_calls();
        let fixture = rgb8_htj2k_fixture(32, 32);
        let inputs = [fixture.as_slice(), fixture.as_slice()];
        let mut ctx = DecoderContext::<CpuJ2kContext>::new();
        let mut pool = CpuJ2kScratchPool::new();

        let result = Codec::decode_tiles_to_device(
            &mut ctx,
            &mut pool,
            &inputs,
            PixelFormat::Rgb8,
            BackendRequest::Cuda,
        );

        assert_eq!(testing_cuda_htj2k_batch_decode_calls(), 1);
        match result {
            Ok(surfaces) => {
                assert_eq!(surfaces.len(), inputs.len());
                for surface in surfaces {
                    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
                    assert_eq!(surface.as_host_bytes(), None);
                }
            }
            Err(Error::CudaUnavailable) => {
                assert!(!cuda_runtime_required());
            }
            Err(error) => panic!("unexpected strict CUDA RGB batch error: {error}"),
        }
    }

    fn rgb8_htj2k_fixture(width: u32, height: u32) -> Vec<u8> {
        htj2k_rgb8_pattern_fixture(width, height, 17)
    }
}
