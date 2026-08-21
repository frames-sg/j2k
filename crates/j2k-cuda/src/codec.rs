// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    DeviceDecodePlan, DeviceDecodeRequest, J2kCodec as CpuCodec, J2kContext as CpuJ2kContext,
    J2kDecodeWarning, J2kDecoder as CpuDecoder, J2kScratchPool as CpuJ2kScratchPool,
};
use j2k_core::{
    checked_surface_len, submit_ready_device, BackendRequest, Downscale, ImageCodec, PixelFormat,
    ReadySubmission, Rect, TileBatchDecode, TileBatchDecodeDevice, TileBatchDecodeManyDevice,
    TileBatchDecodeSubmit, TileRegionScaledDecodeJob, TileRegionScaledDeviceDecodeRequest,
    DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};

use crate::{
    allocation::{try_collect_results_exact, try_vec_filled},
    routing::{auto_cuda_available, auto_repeated_decode_uses_cuda, inputs_repeat_one_slice},
    runtime::{validate_surface_request, wrap_surface},
};
use crate::{CudaSession, Error, J2kDecoder, Surface};

/// Marker type implementing tile-batch CUDA surface decode traits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Codec;

struct TileSurfaceRequest<'a> {
    ctx: &'a mut CpuJ2kContext,
    session: &'a mut CudaSession,
    pool: &'a mut CpuJ2kScratchPool,
    input: &'a [u8],
    fmt: PixelFormat,
    operation: DeviceDecodeRequest,
    backend: BackendRequest,
}

#[doc(hidden)]
impl ImageCodec for Codec {
    type Error = Error;
    type Warning = J2kDecodeWarning;
    type Pool = crate::J2kScratchPool;
}

impl Codec {
    fn supports_cuda_batch_format(fmt: PixelFormat) -> bool {
        matches!(
            fmt,
            PixelFormat::Gray8
                | PixelFormat::Gray16
                | PixelFormat::GrayI16
                | PixelFormat::Rgb8
                | PixelFormat::Rgba8
                | PixelFormat::Rgb16
                | PixelFormat::Rgba16
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

    fn decode_tile_op_to_surface_impl(request: TileSurfaceRequest<'_>) -> Result<Surface, Error> {
        let TileSurfaceRequest {
            ctx,
            session,
            pool,
            input,
            fmt,
            operation,
            backend,
        } = request;
        validate_surface_request(backend)?;
        if backend == BackendRequest::Cuda {
            let mut decoder = J2kDecoder::new(input)?;
            return decoder.decode_request_to_device_with_session(fmt, operation, session);
        }

        let plan = DeviceDecodePlan::for_image(CpuDecoder::inspect(input)?.dimensions, operation)?;
        let dims = plan.output_dims();
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        match operation {
            DeviceDecodeRequest::Full => {
                CpuCodec::decode_tile(ctx, pool, input, &mut out, stride, fmt)?;
            }
            DeviceDecodeRequest::Region { .. } => {
                CpuCodec::decode_tile_region(
                    ctx,
                    pool,
                    input,
                    &mut out,
                    stride,
                    fmt,
                    plan.source_rect(),
                )?;
            }
            DeviceDecodeRequest::Scaled { scale } => {
                CpuCodec::decode_tile_scaled(ctx, pool, input, &mut out, stride, fmt, scale)?;
            }
            DeviceDecodeRequest::RegionScaled { scale, .. } => {
                CpuCodec::decode_tile_region_scaled(
                    ctx,
                    pool,
                    fmt,
                    TileRegionScaledDecodeJob {
                        input,
                        out: &mut out,
                        stride,
                        roi: plan.source_rect(),
                        scale,
                    },
                )?;
            }
        }
        wrap_surface(out, dims, fmt, backend, session)
    }
}

fn allocate_cpu_surface(dims: (u32, u32), fmt: PixelFormat) -> Result<(Vec<u8>, usize), Error> {
    let (stride, len) = checked_surface_len(
        dims,
        fmt.bytes_per_pixel(),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        "j2k CUDA CPU fallback surface",
    )?;
    Ok((
        try_vec_filled(len, 0u8, "j2k CUDA CPU fallback surface")?,
        stride,
    ))
}

#[doc(hidden)]
impl TileBatchDecodeSubmit for Codec {
    type Context = CpuJ2kContext;
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
            Self::decode_tile_op_to_surface_impl(TileSurfaceRequest {
                ctx,
                session,
                pool,
                input,
                fmt,
                operation: DeviceDecodeRequest::Full,
                backend,
            })
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
            Self::decode_tile_op_to_surface_impl(TileSurfaceRequest {
                ctx,
                session,
                pool,
                input,
                fmt,
                operation: DeviceDecodeRequest::Region { roi },
                backend,
            })
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
            Self::decode_tile_op_to_surface_impl(TileSurfaceRequest {
                ctx,
                session,
                pool,
                input,
                fmt,
                operation: DeviceDecodeRequest::Scaled { scale },
                backend,
            })
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
            Self::decode_tile_op_to_surface_impl(TileSurfaceRequest {
                ctx,
                session,
                pool,
                input,
                fmt,
                operation: DeviceDecodeRequest::RegionScaled { roi, scale },
                backend,
            })
        }))
    }
}

#[doc(hidden)]
impl TileBatchDecodeDevice for Codec {
    type Context = CpuJ2kContext;
    type DeviceSurface = Surface;
}

#[doc(hidden)]
impl TileBatchDecodeManyDevice for Codec {
    type Context = CpuJ2kContext;
    type DeviceSurface = Surface;

    fn decode_tiles_to_device(
        ctx: &mut Self::Context,
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
        if backend == BackendRequest::Auto
            && Self::supports_cuda_batch_format(fmt)
            && inputs_repeat_one_slice(inputs)
        {
            let support = CpuDecoder::inspect_support(inputs[0])?;
            if auto_repeated_decode_uses_cuda(
                support.info.dimensions,
                support.info.components,
                fmt,
                support.transfer_syntax,
                support.payload_kind,
                inputs.len(),
            ) && auto_cuda_available(&mut session)?
            {
                return Self::decode_tiles_to_cuda_batch(inputs, fmt, &mut session);
            }
        }

        try_collect_results_exact(
            inputs.iter().map(|input| {
                Self::decode_tile_op_to_surface_impl(TileSurfaceRequest {
                    ctx,
                    session: &mut session,
                    pool,
                    input,
                    fmt,
                    operation: DeviceDecodeRequest::Full,
                    backend,
                })
            }),
            "j2k CUDA decode batch surfaces",
        )
    }
}

#[cfg(all(test, feature = "cuda-runtime"))]
mod tests {
    use j2k_core::{BackendRequest, PixelFormat, TileBatchDecodeManyDevice};
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
        let mut ctx = CpuJ2kContext::default();
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
