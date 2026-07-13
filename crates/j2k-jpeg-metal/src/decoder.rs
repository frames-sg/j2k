// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    BackendRequest, CpuBackedImageDecode, DecodeOutcome, ImageCodec, ImageDecodeDevice, PixelFormat,
};
#[cfg(all(target_os = "macos", test))]
use j2k_core::{Downscale, Rect};
#[cfg(target_os = "macos")]
use j2k_jpeg::adapter::{JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1};
use j2k_jpeg::{
    adapter::{JpegCachedPlan, JpegPlanCache},
    Decoder as CpuDecoder, JpegView, ScratchPool as CpuScratchPool, Warning as CpuWarning,
};

use crate::{
    batch, decode_surface_from_decoder, routing, Error, JpegFastPackets, MetalBackendSession,
    MetalDecodeRequest, SharedJpegFastPacket, SharedJpegInput, Surface,
};
#[cfg(target_os = "macos")]
use crate::{compute, reject_cpu_staged_metal_upload, ResidentPrivateJpegTile};

/// JPEG decoder that can return host or Metal-resident surfaces.
pub struct Decoder<'a> {
    pub(crate) inner: CpuDecoder<'a>,
    pub(crate) source: SharedJpegInput,
    pub(crate) fast_packet: Option<SharedJpegFastPacket>,
    #[cfg(target_os = "macos")]
    pub(crate) batch_shape: batch::BatchShape,
}

impl<'a> Decoder<'a> {
    /// Parse a JPEG byte slice into a decoder with any available Metal packets.
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        let mut cache = JpegPlanCache::default();
        let (plan, inner) = cache.resolve_with_decoder_and_external_live(input, 0)?;
        let source = plan.input().clone();
        let fast_packet = plan.fast_packet().cloned();
        #[cfg(target_os = "macos")]
        let batch_shape = batch::BatchShape::from_summary(plan.batch_summary(), plan.color_space());
        Ok(Self {
            inner,
            source,
            fast_packet,
            #[cfg(target_os = "macos")]
            batch_shape,
        })
    }

    /// Create a decoder from an already parsed JPEG view.
    pub fn from_view(view: JpegView<'a>) -> Result<Self, Error> {
        let (plan, inner) = JpegCachedPlan::build_from_view_with_decoder(view, 0)?;
        let source = plan.input().clone();
        let fast_packet = plan.fast_packet().cloned();
        #[cfg(target_os = "macos")]
        let batch_shape = batch::BatchShape::from_summary(plan.batch_summary(), plan.color_space());
        Ok(Self {
            inner,
            source,
            fast_packet,
            #[cfg(target_os = "macos")]
            batch_shape,
        })
    }

    /// Borrow the underlying CPU JPEG decoder.
    pub fn inner(&self) -> &CpuDecoder<'a> {
        &self.inner
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn fast444_packet(&self) -> Option<&JpegFast444PacketV1> {
        self.fast_packet
            .as_ref()
            .and_then(SharedJpegFastPacket::fast444)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn fast422_packet(&self) -> Option<&JpegFast422PacketV1> {
        self.fast_packet
            .as_ref()
            .and_then(SharedJpegFastPacket::fast422)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn fast420_packet(&self) -> Option<&JpegFast420PacketV1> {
        self.fast_packet
            .as_ref()
            .and_then(SharedJpegFastPacket::fast420)
    }

    pub(crate) fn fast_packets(&self) -> JpegFastPackets<'_> {
        JpegFastPackets::from_shared(self.fast_packet.as_ref())
    }

    pub(crate) fn retained_host_bytes(&self) -> Result<usize, Error> {
        let decoder_bytes = j2k_jpeg::adapter::decoder_retained_allocation_bytes(&self.inner)?;
        let input_bytes = self.source.retained_cache_bytes()?;
        let packet_bytes = self
            .fast_packet
            .as_ref()
            .map_or(Ok(0), SharedJpegFastPacket::retained_cache_bytes)?;
        decoder_bytes
            .checked_add(input_bytes)
            .and_then(|bytes| bytes.checked_add(packet_bytes))
            .ok_or_else(|| {
                j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                    "JPEG Metal decoder retained host-byte count overflow",
                )
                .into()
            })
    }

    pub(crate) fn fast_packet_for_backend(
        &self,
        backend: BackendRequest,
    ) -> Option<SharedJpegFastPacket> {
        if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast_packet.clone()
        } else {
            None
        }
    }

    pub(crate) const fn batch_shape_for_backend(
        &self,
        backend: BackendRequest,
    ) -> batch::BatchShape {
        #[cfg(target_os = "macos")]
        {
            if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
                return self.batch_shape;
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (self, backend);
        batch::BatchShape::unknown()
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
            self.source.clone(),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            op,
            self.fast_packet.clone(),
            self.batch_shape,
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
        let external_live_bytes = self.retained_host_bytes()?;
        let mut pool = CpuScratchPool::new();
        decode_surface_from_decoder(
            &self.inner,
            &mut pool,
            request.fmt,
            request.backend,
            request.op.batch_op(),
            self.fast_packets(),
            external_live_bytes,
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
            let external_live_bytes = self.retained_host_bytes()?;
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
                        self.fast_packets(),
                        external_live_bytes,
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
        crate::batch_allocation::BatchMetadataBudget::with_external_live(
            "JPEG Metal private tile decoder owners",
            self.retained_host_bytes()?,
        )
        .preflight(&[])?;
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
                self.fast444_packet(),
                self.fast422_packet(),
                self.fast420_packet(),
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
