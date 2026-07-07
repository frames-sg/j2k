// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::{
    BackendRequest, CpuBackedImageDecode, DecodeOutcome, ImageCodec, ImageDecodeDevice, PixelFormat,
};
#[cfg(all(target_os = "macos", test))]
use j2k_core::{Downscale, Rect};
use j2k_jpeg::{
    adapter::{
        build_fast420_packet, build_fast422_packet, build_fast444_packet, decoder_bytes,
        JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1,
    },
    Decoder as CpuDecoder, JpegView, ScratchPool as CpuScratchPool, Warning as CpuWarning,
};

use crate::{
    batch, decode_surface_from_decoder, routing, Error, JpegFastPackets, MetalBackendSession,
    MetalDecodeRequest, ResidentPrivateJpegTile, Surface,
};
#[cfg(target_os = "macos")]
use crate::{compute, reject_cpu_staged_metal_upload};

/// JPEG decoder that can return host or Metal-resident surfaces.
pub struct Decoder<'a> {
    pub(crate) inner: CpuDecoder<'a>,
    pub(crate) source: Arc<[u8]>,
    pub(crate) fast444_packet: Option<Arc<JpegFast444PacketV1>>,
    pub(crate) fast422_packet: Option<Arc<JpegFast422PacketV1>>,
    pub(crate) fast420_packet: Option<Arc<JpegFast420PacketV1>>,
}

impl<'a> Decoder<'a> {
    /// Parse a JPEG byte slice into a decoder with any available Metal packets.
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        let inner = CpuDecoder::new(input)?;
        Ok(Self {
            fast444_packet: build_fast444_packet(input).ok().map(Arc::new),
            fast422_packet: build_fast422_packet(input).ok().map(Arc::new),
            fast420_packet: build_fast420_packet(input).ok().map(Arc::new),
            inner,
            source: Arc::<[u8]>::from(input),
        })
    }

    /// Create a decoder from an already parsed JPEG view.
    pub fn from_view(view: JpegView<'a>) -> Result<Self, Error> {
        let inner = CpuDecoder::from_view(view)?;
        let source_bytes = decoder_bytes(&inner);
        let source = Arc::<[u8]>::from(source_bytes);
        let fast444_packet = build_fast444_packet(source_bytes).ok().map(Arc::new);
        let fast422_packet = build_fast422_packet(source_bytes).ok().map(Arc::new);
        let fast420_packet = build_fast420_packet(source_bytes).ok().map(Arc::new);
        Ok(Self {
            inner,
            source,
            fast444_packet,
            fast422_packet,
            fast420_packet,
        })
    }

    /// Borrow the underlying CPU JPEG decoder.
    pub fn inner(&self) -> &CpuDecoder<'a> {
        &self.inner
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn fast444_packet(&self) -> Option<&JpegFast444PacketV1> {
        self.fast444_packet.as_deref()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn fast422_packet(&self) -> Option<&JpegFast422PacketV1> {
        self.fast422_packet.as_deref()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn fast420_packet(&self) -> Option<&JpegFast420PacketV1> {
        self.fast420_packet.as_deref()
    }

    pub(crate) fn fast_packets(&self) -> JpegFastPackets<'_> {
        JpegFastPackets::new(
            self.fast444_packet.as_deref(),
            self.fast422_packet.as_deref(),
            self.fast420_packet.as_deref(),
        )
    }

    #[cfg(all(target_os = "macos", test))]
    pub(crate) fn rgb8_region_scaled_metal_request(
        &self,
        roi: Rect,
        scale: Downscale,
    ) -> batch::QueuedRequest {
        self.rgb8_metal_request(batch::BatchOp::RegionScaled { roi, scale })
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn rgb8_metal_request(&self, op: batch::BatchOp) -> batch::QueuedRequest {
        batch::QueuedRequest::new_shared(
            Arc::clone(&self.source),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            op,
            self.fast444_packet.clone(),
            self.fast422_packet.clone(),
            self.fast420_packet.clone(),
        )
    }

    /// Consume this wrapper and return the underlying CPU JPEG decoder.
    pub fn into_inner(self) -> CpuDecoder<'a> {
        self.inner
    }

    /// Decode into a device surface using a request object instead of a
    /// geometry-specific method.
    pub fn decode_request_to_device(
        &mut self,
        request: MetalDecodeRequest,
    ) -> Result<Surface, Error> {
        let mut pool = CpuScratchPool::new();
        decode_surface_from_decoder(
            &self.inner,
            &mut pool,
            request.fmt,
            request.backend,
            request.op.batch_op(),
            self.fast_packets(),
        )
    }

    /// Decode a full image into a device surface using a reusable Metal session.
    pub fn decode_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        #[cfg(target_os = "macos")]
        {
            let mut pool = CpuScratchPool::new();
            let decision = crate::choose_route(
                &self.inner,
                BackendRequest::Metal,
                fmt,
                batch::BatchOp::Full,
                self.fast_packets(),
            );
            if let Some(err) = routing::decision_error(decision) {
                return Err(err);
            }
            match decision {
                routing::RouteDecision::MetalKernel => {
                    reject_cpu_staged_metal_upload(compute::decode_to_surface_with_session(
                        &self.inner,
                        &mut pool,
                        fmt,
                        self.fast444_packet.as_deref(),
                        self.fast422_packet.as_deref(),
                        self.fast420_packet.as_deref(),
                        session,
                    )?)
                }
                routing::RouteDecision::CpuHost
                | routing::RouteDecision::RejectExplicitMetal { .. }
                | routing::RouteDecision::RejectUnsupportedBackend { .. }
                | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = session;
            let decision = crate::choose_route(
                &self.inner,
                BackendRequest::Metal,
                fmt,
                batch::BatchOp::Full,
                self.fast_packets(),
            );
            if let Some(err) = routing::decision_error(decision) {
                return Err(err);
            }
            Err(Error::MetalUnavailable)
        }
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_private_rgb8_tile_with_session(
        &mut self,
        session: &MetalBackendSession,
    ) -> Result<ResidentPrivateJpegTile, Error> {
        let decision = crate::choose_route(
            &self.inner,
            BackendRequest::Metal,
            PixelFormat::Rgb8,
            batch::BatchOp::Full,
            self.fast_packets(),
        );
        if let Some(err) = routing::decision_error(decision) {
            return Err(err);
        }
        match decision {
            routing::RouteDecision::MetalKernel => compute::decode_private_rgb8_tile_with_session(
                &self.inner,
                self.fast444_packet.as_deref(),
                self.fast422_packet.as_deref(),
                self.fast420_packet.as_deref(),
                session,
            ),
            routing::RouteDecision::CpuHost
            | routing::RouteDecision::RejectExplicitMetal { .. }
            | routing::RouteDecision::RejectUnsupportedBackend { .. }
            | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
        }
    }
}

#[doc(hidden)]
impl ImageCodec for Decoder<'_> {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = crate::ScratchPool;
}

impl<'a> CpuBackedImageDecode<'a> for Decoder<'a> {
    type Cpu = CpuDecoder<'a>;
    type View = JpegView<'a>;

    fn inspect_cpu(input: &'a [u8]) -> Result<j2k_core::Info, Self::Error> {
        Ok(CpuDecoder::inspect(input)?.to_core_info())
    }

    fn parse_cpu(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(JpegView::parse(input)?)
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
impl<'a> ImageDecodeDevice<'a> for Decoder<'a> {
    type DeviceSurface = Surface;
}
