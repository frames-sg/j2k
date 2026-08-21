// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic J2K codec trait implementations for JPEG Metal.

use crate::{batch, batch_allocation, Codec, Decoder, MetalDecodeRequest, MetalSession, Surface};
use j2k_core::{
    BackendRequest, DeviceSubmission, Downscale, ImageDecodeSubmit, PixelFormat, Rect,
    TileBatchDecodeDevice, TileBatchDecodeManyDevice, TileBatchDecodeSubmit,
    TileRegionScaledDeviceDecodeRequest,
};
use j2k_jpeg::DecoderContext as CpuDecoderContext;

#[doc(hidden)]
impl<'a> ImageDecodeSubmit<'a> for Decoder<'a> {
    type Session = MetalSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = batch::MetalSubmission;

    fn submit_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let fast_packet = self.fast_packet_for_backend(backend);
        let slot = session
            .shared
            .lock()?
            .queue_request(batch::QueuedRequest::new_shared(
                self.source.clone(),
                fmt,
                backend,
                batch::BatchOp::Full,
                fast_packet,
                self.batch_shape_for_backend(backend),
            ))?;
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }

    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let fast_packet = self.fast_packet_for_backend(backend);
        let slot = session
            .shared
            .lock()?
            .queue_request(batch::QueuedRequest::new_shared(
                self.source.clone(),
                fmt,
                backend,
                batch::BatchOp::Region(roi),
                fast_packet,
                self.batch_shape_for_backend(backend),
            ))?;
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }

    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let fast_packet = self.fast_packet_for_backend(backend);
        let slot = session
            .shared
            .lock()?
            .queue_request(batch::QueuedRequest::new_shared(
                self.source.clone(),
                fmt,
                backend,
                batch::BatchOp::Scaled(scale),
                fast_packet,
                self.batch_shape_for_backend(backend),
            ))?;
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }

    fn submit_region_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let fast_packet = self.fast_packet_for_backend(backend);
        let slot = session
            .shared
            .lock()?
            .queue_request(batch::QueuedRequest::new_shared(
                self.source.clone(),
                fmt,
                backend,
                batch::BatchOp::RegionScaled { roi, scale },
                fast_packet,
                self.batch_shape_for_backend(backend),
            ))?;
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }
}

#[doc(hidden)]
impl TileBatchDecodeSubmit for Codec {
    type Context = CpuDecoderContext;
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
        Self::submit_tile_request_to_device(
            ctx,
            session,
            pool,
            input,
            MetalDecodeRequest::full(fmt, backend),
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
        Self::submit_tile_request_to_device(
            ctx,
            session,
            pool,
            input,
            MetalDecodeRequest::region(fmt, roi, backend),
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
        Self::submit_tile_request_to_device(
            ctx,
            session,
            pool,
            input,
            MetalDecodeRequest::scaled(fmt, scale, backend),
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
        Self::submit_tile_request_to_device(
            ctx,
            session,
            pool,
            input,
            MetalDecodeRequest::region_scaled(fmt, roi, scale, backend),
        )
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
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let _ = (ctx, pool);

        let session = MetalSession::default();
        let mut budget =
            batch_allocation::BatchMetadataBudget::new("JPEG Metal generic tile batch submission");
        let mut submissions =
            budget.try_vec(inputs.len(), "JPEG Metal generic tile batch submissions")?;
        let mut surfaces =
            budget.try_vec(inputs.len(), "JPEG Metal generic tile batch surfaces")?;
        let retained_metadata_bytes = budget.live_bytes();
        for input in inputs {
            let slot = {
                let mut state = session.shared.lock()?;
                let resolved = state.resolve_jpeg_plan_with_external_live(
                    input,
                    backend,
                    retained_metadata_bytes,
                )?;
                state.queue_request_with_retained_metadata(
                    batch::QueuedRequest::new_shared(
                        resolved.input,
                        fmt,
                        backend,
                        batch::BatchOp::Full,
                        resolved.fast_packet,
                        resolved.shape,
                    ),
                    retained_metadata_bytes,
                )?
            };
            submissions.push(batch::MetalSubmission {
                session: session.shared.clone(),
                slot,
            });
        }

        for submission in submissions {
            surfaces.push(submission.wait()?);
        }
        Ok(surfaces)
    }
}
