// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{J2kContext as CpuJ2kContext, J2kDecodeWarning, J2kDecoder as CpuDecoder, J2kView};
use j2k_core::{
    BackendRequest, CpuBackedImageDecode, DecodeOutcome, Downscale, ImageCodec, ImageDecodeDevice,
    ImageDecodeSubmit, PixelFormat, ReadySubmission, Rect, TileBatchDecodeDevice,
    TileBatchDecodeManyDevice, TileBatchDecodeSubmit, TileRegionScaledDeviceDecodeRequest,
};
use j2k_metal_support::FallibleSubmissionQueue;

use super::{J2kDecoder, MetalDecodeRequest};
use crate::{batch, Error, MetalSession, Surface};

#[doc(hidden)]
impl ImageCodec for J2kDecoder<'_> {
    type Error = Error;
    type Warning = J2kDecodeWarning;
    type Pool = crate::J2kScratchPool;
}

impl<'a> CpuBackedImageDecode<'a> for J2kDecoder<'a> {
    type Cpu = CpuDecoder<'a>;
    type View = J2kView<'a>;

    fn inspect_cpu(input: &'a [u8]) -> Result<j2k_core::Info, Self::Error> {
        Ok(CpuDecoder::inspect(input)?)
    }

    fn parse_cpu(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(J2kView::parse(input)?)
    }

    fn from_cpu_view(view: Self::View) -> Result<Self, Self::Error> {
        Self::from_view(view)
    }

    fn cpu_decoder_mut(&mut self) -> &mut Self::Cpu {
        &mut self.inner
    }

    fn map_cpu_outcome(
        outcome: DecodeOutcome<<Self::Cpu as ImageCodec>::Warning>,
    ) -> DecodeOutcome<Self::Warning> {
        outcome
    }
}

#[doc(hidden)]
impl<'a> ImageDecodeDevice<'a> for J2kDecoder<'a> {
    type DeviceSurface = Surface;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// J2K codec marker used by J2K's generic decode traits.
pub struct Codec;

#[doc(hidden)]
impl ImageCodec for Codec {
    type Error = Error;
    type Warning = J2kDecodeWarning;
    type Pool = crate::J2kScratchPool;
}

#[doc(hidden)]
impl<'a> ImageDecodeSubmit<'a> for J2kDecoder<'a> {
    type Session = MetalSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = ReadySubmission<Surface, Error>;

    fn submit_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit()?;
        Ok(ReadySubmission::from_result(self.decode_request_to_device(
            MetalDecodeRequest::full(fmt, backend),
        )))
    }

    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit()?;
        Ok(ReadySubmission::from_result(self.decode_request_to_device(
            MetalDecodeRequest::region(fmt, roi, backend),
        )))
    }

    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit()?;
        Ok(ReadySubmission::from_result(self.decode_request_to_device(
            MetalDecodeRequest::scaled(fmt, scale, backend),
        )))
    }

    fn submit_region_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit()?;
        Ok(ReadySubmission::from_result(self.decode_request_to_device(
            MetalDecodeRequest::region_scaled(fmt, roi, scale, backend),
        )))
    }
}

#[doc(hidden)]
impl TileBatchDecodeSubmit for Codec {
    type Context = CpuJ2kContext;
    type Session = MetalSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = batch::MetalSubmission;

    fn submit_tile_to_device(
        ctx: &mut Self::Context,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let _ = (ctx, pool);
        let request = MetalDecodeRequest::full(fmt, backend);
        batch::queue_tile_request(
            session,
            input,
            request.fmt,
            request.backend,
            request.op.batch_op(),
        )
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
        let _ = (ctx, pool);
        let request = MetalDecodeRequest::region(fmt, roi, backend);
        batch::queue_tile_request(
            session,
            input,
            request.fmt,
            request.backend,
            request.op.batch_op(),
        )
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
        let _ = (ctx, pool);
        let request = MetalDecodeRequest::scaled(fmt, scale, backend);
        batch::queue_tile_request(
            session,
            input,
            request.fmt,
            request.backend,
            request.op.batch_op(),
        )
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
        let _ = (ctx, pool);
        let request = MetalDecodeRequest::region_scaled(fmt, roi, scale, backend);
        batch::queue_tile_request(
            session,
            input,
            request.fmt,
            request.backend,
            request.op.batch_op(),
        )
    }
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
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let mut session = MetalSession::default();
        let mut submissions = FallibleSubmissionQueue::with_capacity_hint(inputs.len());
        for input in inputs {
            submissions.try_push_with("J2K Metal decode-many submissions", |_, _| {
                <Self as TileBatchDecodeSubmit>::submit_tile_to_device(
                    ctx,
                    &mut session,
                    pool,
                    input,
                    fmt,
                    backend,
                )
            })?;
        }

        submissions.try_finish(
            "J2K Metal decode-many submission and surface metadata",
            "J2K Metal decode-many surfaces",
            j2k_core::DeviceSubmission::wait,
        )
    }
}

#[doc(hidden)]
impl TileBatchDecodeDevice for Codec {
    type Context = CpuJ2kContext;
    type DeviceSurface = Surface;
}
