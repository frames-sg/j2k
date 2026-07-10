// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal-backed JPEG decode and encode adapters.
//!
//! The crate exposes the same CPU-visible JPEG decode surface as
//! `j2k-jpeg`, with optional Metal-resident surfaces and batch submission
//! helpers on macOS. Non-macOS builds return `Error::MetalUnavailable`.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

#[cfg(target_os = "macos")]
mod abi;
mod batch;
#[cfg(target_os = "macos")]
mod buffers;
mod codec_batch;
#[cfg(target_os = "macos")]
mod compute;
mod decode_request;
mod decoder;
mod encode;
mod error;
mod fast_packets;
mod routing;
mod session;
mod surface;
mod tile_batch;
mod viewport;
#[cfg(test)]
mod viewport_tests;

use std::sync::Arc;

pub use encode::{
    encode_jpeg_baseline_batch_from_metal_buffers, encode_jpeg_baseline_from_metal_buffer,
    JpegBaselineMetalEncodeTile,
};
use j2k_core::{
    checked_surface_len, BackendKind, BackendRequest, DeviceSubmission, Downscale, ImageCodec,
    ImageDecodeSubmit, PixelFormat, Rect, TileBatchDecodeDevice, TileBatchDecodeManyDevice,
    TileBatchDecodeSubmit, TileRegionScaledDeviceDecodeRequest, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
use j2k_jpeg::{
    adapter::{
        build_fast420_packet, build_fast422_packet, build_fast444_packet, decoder_bytes,
        JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1,
    },
    Decoder as CpuDecoder, DecoderContext as CpuDecoderContext, ScratchPool as CpuScratchPool,
    Warning as CpuWarning,
};

#[cfg(target_os = "macos")]
use metal::{Device, MTLResourceOptions};

#[cfg(target_os = "macos")]
pub use codec_batch::{
    MetalBufferBatchTarget, MetalTextureBatchTarget, Rgb8MetalBatchOp, Rgb8MetalBatchRequest,
    Rgb8MetalBatchSource,
};
pub use decode_request::{MetalDecodeOp, MetalDecodeRequest};
pub use decoder::Decoder;
pub use error::Error;
pub(crate) use fast_packets::JpegFastPackets;
pub use j2k_core::SurfaceResidency;
pub use session::{MetalBackendSession, MetalSession};
pub(crate) use surface::Storage;
pub use surface::{
    MetalBatchOutputBuffer, MetalBatchTextureOutput, MetalTextureTile, ResidentPrivateJpegTile,
    Surface,
};
pub use tile_batch::JpegTileBatch;
pub use viewport::{
    choose_viewport_surface_strategy, decode_viewport_to_surface, is_contiguous_viewport_workload,
    suggest_viewport_workload, viewport_source_bounds, ViewportSurfaceStrategy, ViewportTile,
    ViewportWorkload,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// JPEG codec marker used by J2K's generic decode traits.
pub struct Codec;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Preflight report for RGB8 JPEG Metal resident decoder batches.
#[doc(hidden)]
pub struct JpegMetalResidentBatchReport {
    /// Requested decode operation.
    pub op: j2k_jpeg::JpegDecodeOp,
    /// Number of decoder tiles in the batch.
    pub tile_count: usize,
    /// Required output dimensions when the batch is eligible and shape-compatible.
    pub output_dimensions: Option<(u32, u32)>,
    /// Whether the batch can use reusable RGB8 Metal resident output.
    pub eligibility: j2k_jpeg::JpegBackendEligibility,
}

#[cfg(target_os = "macos")]
impl JpegMetalResidentBatchReport {
    /// Required number of tile slots in caller-owned Metal output.
    #[must_use]
    pub fn required_tile_capacity(&self) -> usize {
        self.tile_count
    }
}

#[cfg(target_os = "macos")]
fn report_required_output_dimensions(
    report: &JpegMetalResidentBatchReport,
) -> Result<Option<(u32, u32)>, Error> {
    if !report.eligibility.eligible {
        return Err(Error::UnsupportedMetalRequest {
            reason: report
                .eligibility
                .reason
                .unwrap_or("JPEG Metal resident batch report is not eligible"),
        });
    }
    if report.tile_count == 0 {
        return Ok(None);
    }
    report
        .output_dimensions
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal resident batch report is missing output dimensions",
        })
        .map(Some)
}

#[doc(hidden)]
impl ImageCodec for Codec {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = ScratchPool;
}

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
        let fast444_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast444_packet.clone()
        } else {
            None
        };
        let fast422_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast422_packet.clone()
        } else {
            None
        };
        let fast420_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast420_packet.clone()
        } else {
            None
        };
        let slot = session
            .shared
            .lock()?
            .queue_request(batch::QueuedRequest::new_shared(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::Full,
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ));
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
        let fast444_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast444_packet.clone()
        } else {
            None
        };
        let fast422_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast422_packet.clone()
        } else {
            None
        };
        let fast420_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast420_packet.clone()
        } else {
            None
        };
        let slot = session
            .shared
            .lock()?
            .queue_request(batch::QueuedRequest::new_shared(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::Region(roi),
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ));
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
        let fast444_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast444_packet.clone()
        } else {
            None
        };
        let fast422_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast422_packet.clone()
        } else {
            None
        };
        let fast420_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast420_packet.clone()
        } else {
            None
        };
        let slot = session
            .shared
            .lock()?
            .queue_request(batch::QueuedRequest::new_shared(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::Scaled(scale),
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ));
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
        let fast444_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast444_packet.clone()
        } else {
            None
        };
        let fast422_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast422_packet.clone()
        } else {
            None
        };
        let fast420_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast420_packet.clone()
        } else {
            None
        };
        let slot = session
            .shared
            .lock()?
            .queue_request(batch::QueuedRequest::new_shared(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::RegionScaled { roi, scale },
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ));
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
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
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
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
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
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
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
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
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
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Vec<Self::DeviceSurface>, Self::Error> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let mut session = MetalSession::default();
        let submissions = inputs
            .iter()
            .map(|input| {
                <Self as TileBatchDecodeSubmit>::submit_tile_to_device(
                    ctx,
                    &mut session,
                    pool,
                    input,
                    fmt,
                    backend,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        submissions
            .into_iter()
            .map(DeviceSubmission::wait)
            .collect()
    }
}

pub(crate) fn decode_surface_from_bytes(
    input: &[u8],
    fmt: PixelFormat,
    backend: BackendRequest,
    op: batch::BatchOp,
    fast444_packet: Option<Arc<JpegFast444PacketV1>>,
    fast422_packet: Option<Arc<JpegFast422PacketV1>>,
    fast420_packet: Option<Arc<JpegFast420PacketV1>>,
) -> Result<Surface, Error> {
    let decoder = CpuDecoder::new(input)?;
    let mut pool = CpuScratchPool::new();
    let build_auto_packets =
        matches!(backend, BackendRequest::Auto) && decoder.info().restart_interval.is_some();
    let build_metal_packets = matches!(backend, BackendRequest::Metal);
    let fast444_packet = if build_auto_packets || build_metal_packets {
        fast444_packet.or_else(|| {
            build_fast444_packet(decoder_bytes(&decoder))
                .ok()
                .map(Arc::new)
        })
    } else {
        None
    };
    let fast422_packet = if build_auto_packets || build_metal_packets {
        fast422_packet.or_else(|| {
            build_fast422_packet(decoder_bytes(&decoder))
                .ok()
                .map(Arc::new)
        })
    } else {
        None
    };
    let fast420_packet = if build_auto_packets || build_metal_packets {
        fast420_packet.or_else(|| {
            build_fast420_packet(decoder_bytes(&decoder))
                .ok()
                .map(Arc::new)
        })
    } else {
        None
    };
    let packets = JpegFastPackets::new(
        fast444_packet.as_deref(),
        fast422_packet.as_deref(),
        fast420_packet.as_deref(),
    );
    decode_surface_from_decoder(&decoder, &mut pool, fmt, backend, op, packets)
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn decode_compatible_batch(
    requests: &[batch::QueuedRequest],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let _ = requests;
    Ok(None)
}

#[allow(clippy::unnecessary_wraps)]
pub(crate) fn decode_compatible_batch_with_session(
    requests: &[batch::QueuedRequest],
    session: &mut session::SessionState,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    #[cfg(target_os = "macos")]
    {
        compute::decode_full_batch_to_surfaces_with_session_state(requests, session)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = session;
        decode_compatible_batch(requests)
    }
}

#[cfg(target_os = "macos")]
#[doc(hidden)]
pub fn decode_rgb8_batch_to_device_with_session(
    inputs: &[&[u8]],
    session: &MetalBackendSession,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if inputs.len() < 2 {
        return Ok(None);
    }

    let mut state = session::SessionState::default();
    let mut requests = Vec::with_capacity(inputs.len());
    for input in inputs {
        let input = state.intern_input_slice(input);
        let (fast444_packet, fast422_packet, fast420_packet) =
            state.resolve_fast_packets(&input, BackendRequest::Metal);
        requests.push(batch::QueuedRequest::new_shared(
            input,
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            fast444_packet,
            fast422_packet,
            fast420_packet,
        ));
    }

    compute::decode_full_batch_to_surfaces_with_session(&requests, session)
}

#[expect(
    clippy::too_many_lines,
    reason = "the decoder dispatcher keeps the ordered fast-packet routes and CPU fallback together so backend selection stays deterministic"
)]
fn decode_surface_from_decoder(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: batch::BatchOp,
    packets: JpegFastPackets<'_>,
) -> Result<Surface, Error> {
    match op {
        batch::BatchOp::Full => match backend {
            BackendRequest::Cpu => decode_full_cpu_upload(decoder, pool, fmt),
            BackendRequest::Auto | BackendRequest::Metal => {
                let decision = choose_route(decoder, backend, fmt, op, packets);
                if let Some(err) = routing::decision_error(decision) {
                    return Err(err);
                }
                match decision {
                    routing::RouteDecision::CpuHost => decode_full_cpu_upload(decoder, pool, fmt),
                    routing::RouteDecision::MetalKernel => {
                        #[cfg(target_os = "macos")]
                        {
                            reject_cpu_staged_metal_upload(compute::decode_to_surface(
                                decoder,
                                pool,
                                fmt,
                                packets.fast444,
                                packets.fast422,
                                packets.fast420,
                            )?)
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            let _ = (decoder, pool, fmt, packets);
                            Err(Error::MetalUnavailable)
                        }
                    }
                    routing::RouteDecision::RejectExplicitMetal { .. }
                    | routing::RouteDecision::RejectUnsupportedBackend { .. }
                    | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        },
        batch::BatchOp::Region(roi) => match backend {
            BackendRequest::Cpu => decode_region_cpu_upload(decoder, pool, fmt, roi),
            BackendRequest::Auto | BackendRequest::Metal => {
                let decision = choose_route(decoder, backend, fmt, op, packets);
                if let Some(err) = routing::decision_error(decision) {
                    return Err(err);
                }
                match decision {
                    routing::RouteDecision::CpuHost => {
                        decode_region_cpu_upload(decoder, pool, fmt, roi)
                    }
                    routing::RouteDecision::MetalKernel => {
                        #[cfg(target_os = "macos")]
                        {
                            reject_cpu_staged_metal_upload(compute::decode_region_to_surface(
                                decoder,
                                pool,
                                fmt,
                                roi.into(),
                                packets.fast444,
                                packets.fast422,
                                packets.fast420,
                            )?)
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            let _ = (decoder, pool, fmt, roi, packets);
                            Err(Error::MetalUnavailable)
                        }
                    }
                    routing::RouteDecision::RejectExplicitMetal { .. }
                    | routing::RouteDecision::RejectUnsupportedBackend { .. }
                    | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        },
        batch::BatchOp::Scaled(scale) => match backend {
            BackendRequest::Cpu => decode_scaled_cpu_upload(decoder, pool, fmt, scale),
            BackendRequest::Auto | BackendRequest::Metal => {
                let decision = choose_route(decoder, backend, fmt, op, packets);
                if let Some(err) = routing::decision_error(decision) {
                    return Err(err);
                }
                match decision {
                    routing::RouteDecision::CpuHost => {
                        decode_scaled_cpu_upload(decoder, pool, fmt, scale)
                    }
                    routing::RouteDecision::MetalKernel => {
                        #[cfg(target_os = "macos")]
                        {
                            reject_cpu_staged_metal_upload(compute::decode_scaled_to_surface(
                                decoder,
                                pool,
                                fmt,
                                scale,
                                packets.fast444,
                                packets.fast422,
                                packets.fast420,
                            )?)
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            let _ = (decoder, pool, fmt, scale, packets);
                            Err(Error::MetalUnavailable)
                        }
                    }
                    routing::RouteDecision::RejectExplicitMetal { .. }
                    | routing::RouteDecision::RejectUnsupportedBackend { .. }
                    | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        },
        batch::BatchOp::RegionScaled { roi, scale } => decode_region_scaled_surface_from_decoder(
            decoder, pool, fmt, roi, scale, backend, packets,
        ),
    }
}

fn decode_full_cpu_upload(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let dims = decoder.info().dimensions;
    let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
    decoder.decode_into_with_scratch(pool, &mut out, stride, fmt)?;
    upload_surface(out, dims, fmt, BackendRequest::Cpu)
}

fn decode_region_cpu_upload(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let dims = (roi.w, roi.h);
    let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
    decoder.decode_region_scaled_into_with_scratch(
        pool,
        &mut out,
        stride,
        fmt,
        roi.into(),
        Downscale::None,
    )?;
    upload_surface(out, dims, fmt, BackendRequest::Cpu)
}

fn decode_scaled_cpu_upload(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<Surface, Error> {
    let dims = scaled_dims(decoder.info().dimensions, scale);
    let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
    decoder.decode_scaled_into_with_scratch(pool, &mut out, stride, fmt, scale)?;
    upload_surface(out, dims, fmt, BackendRequest::Cpu)
}

fn decode_region_scaled_surface_from_decoder(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    backend: BackendRequest,
    packets: JpegFastPackets<'_>,
) -> Result<Surface, Error> {
    match backend {
        BackendRequest::Cpu => {
            decode_region_scaled_cpu_upload(decoder, pool, fmt, roi, scale, BackendRequest::Cpu)
        }
        BackendRequest::Auto | BackendRequest::Metal => {
            let decision = choose_route(
                decoder,
                backend,
                fmt,
                batch::BatchOp::RegionScaled { roi, scale },
                packets,
            );
            if let Some(err) = routing::decision_error(decision) {
                return Err(err);
            }
            match decision {
                routing::RouteDecision::CpuHost => decode_region_scaled_cpu_upload(
                    decoder,
                    pool,
                    fmt,
                    roi,
                    scale,
                    BackendRequest::Cpu,
                ),
                routing::RouteDecision::MetalKernel => {
                    #[cfg(target_os = "macos")]
                    {
                        reject_cpu_staged_metal_upload(compute::decode_region_scaled_to_surface(
                            decoder,
                            pool,
                            fmt,
                            roi.into(),
                            scale,
                            packets,
                        )?)
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        let _ = (decoder, pool, fmt, roi, scale, packets);
                        Err(Error::MetalUnavailable)
                    }
                }
                routing::RouteDecision::RejectExplicitMetal { .. }
                | routing::RouteDecision::RejectUnsupportedBackend { .. }
                | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    }
}

#[cfg(target_os = "macos")]
fn reject_cpu_staged_metal_upload(surface: Surface) -> Result<Surface, Error> {
    if surface.residency() == SurfaceResidency::CpuStagedMetalUpload {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal explicit device decode requires a direct resident Metal decode; use the CPU path for CPU-staged output",
        });
    }
    Ok(surface)
}

fn choose_route(
    decoder: &CpuDecoder<'_>,
    backend: BackendRequest,
    fmt: PixelFormat,
    op: batch::BatchOp,
    packets: JpegFastPackets<'_>,
) -> routing::RouteDecision {
    let capabilities = routing::JpegMetalCapabilities::for_request(
        decoder,
        fmt,
        op,
        packets.fast444,
        packets.fast422,
        packets.fast420,
    );
    let decision = routing::decide_route(backend, capabilities);
    if j2k_profile::gpu_route_profile_enabled() {
        let labels = decision.profile_labels();
        j2k_profile::emit_gpu_route_fields(
            "jpeg",
            "metal",
            &[
                j2k_profile::ProfileField::label("request", format_args!("{backend:?}")),
                j2k_profile::ProfileField::label("fmt", format_args!("{fmt:?}")),
                j2k_profile::ProfileField::label("op", jpeg_batch_op_profile(op)),
                j2k_profile::ProfileField::label("has_fast_packet", capabilities.has_fast_packet()),
                j2k_profile::ProfileField::label(
                    "supports_output_format",
                    capabilities.supports_output_format(),
                ),
                j2k_profile::ProfileField::label("decision", labels.decision),
                j2k_profile::ProfileField::label("reason", labels.reason),
            ],
        );
    }
    decision
}

fn jpeg_batch_op_profile(op: batch::BatchOp) -> &'static str {
    match op {
        batch::BatchOp::Full => "full",
        batch::BatchOp::Region(_) => "region",
        batch::BatchOp::Scaled(_) => "scaled",
        batch::BatchOp::RegionScaled { .. } => "region_scaled",
    }
}

fn decode_region_scaled_cpu_upload(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    let scaled = roi.scaled_covering(scale);
    let dims = (scaled.w, scaled.h);
    let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
    decoder.decode_region_scaled_into_with_scratch(
        pool,
        &mut out,
        stride,
        fmt,
        roi.into(),
        scale,
    )?;
    upload_surface(out, dims, fmt, backend)
}

fn allocate_cpu_surface(dims: (u32, u32), fmt: PixelFormat) -> Result<(Vec<u8>, usize), Error> {
    let (stride, len) = checked_surface_len(
        dims,
        fmt.bytes_per_pixel(),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        "JPEG Metal CPU fallback surface",
    )?;
    Ok((vec![0u8; len], stride))
}

fn scaled_dims(full: (u32, u32), scale: Downscale) -> (u32, u32) {
    (
        full.0.div_ceil(scale.denominator()),
        full.1.div_ceil(scale.denominator()),
    )
}

pub(crate) fn upload_surface(
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    let pitch_bytes = dimensions.0 as usize * fmt.bytes_per_pixel();
    match backend {
        BackendRequest::Cpu => Ok(Surface {
            backend: BackendKind::Cpu,
            residency: SurfaceResidency::Host,
            dimensions,
            fmt,
            pitch_bytes,
            storage: Storage::Host(bytes),
        }),
        BackendRequest::Auto | BackendRequest::Metal => {
            #[cfg(target_os = "macos")]
            {
                let device = Device::system_default().ok_or(Error::MetalUnavailable)?;
                let buffer = device.new_buffer_with_data(
                    bytes.as_ptr().cast(),
                    bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                Ok(Surface {
                    backend: BackendKind::Metal,
                    residency: SurfaceResidency::CpuStagedMetalUpload,
                    dimensions,
                    fmt,
                    pitch_bytes,
                    storage: Storage::Metal {
                        buffer,
                        offset: 0,
                        access_gate: None,
                    },
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                if matches!(backend, BackendRequest::Auto) {
                    Ok(Surface {
                        backend: BackendKind::Cpu,
                        residency: SurfaceResidency::Host,
                        dimensions,
                        fmt,
                        pitch_bytes,
                        storage: Storage::Host(bytes),
                    })
                } else {
                    Err(Error::MetalUnavailable)
                }
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    }
}

pub use j2k_jpeg::{
    DecoderContext, Downscale as JpegDownscale, Info, PixelFormat as JpegPixelFormat,
    Rect as JpegRectPublic, ScratchPool,
};

#[cfg(test)]
mod tests;
