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
mod batch_allocation;
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
mod plan_owner_ledger;
#[cfg(target_os = "macos")]
mod resident_batch;
mod routing;
mod session;
mod surface;
mod tile_batch;
mod viewport;
#[cfg(test)]
mod viewport_tests;

pub use encode::{
    encode_jpeg_baseline_batch_from_metal_buffers, encode_jpeg_baseline_from_metal_buffer,
    JpegBaselineMetalEncodeTile,
};
use j2k_core::{
    BackendKind, BackendRequest, DeviceSubmission, Downscale, ImageCodec, ImageDecodeSubmit,
    PixelFormat, Rect, TileBatchDecodeDevice, TileBatchDecodeManyDevice, TileBatchDecodeSubmit,
    TileRegionScaledDeviceDecodeRequest,
};
use j2k_jpeg::{
    DecodeRequest as CpuDecodeRequest, Decoder as CpuDecoder, DecoderContext as CpuDecoderContext,
    ScratchPool as CpuScratchPool, Warning as CpuWarning,
};

#[cfg(target_os = "macos")]
use metal::Device;

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
pub(crate) use j2k_jpeg::adapter::{SharedJpegFastPacket, SharedJpegInput};
#[cfg(target_os = "macos")]
pub(crate) use resident_batch::report_required_output_dimensions;
#[cfg(target_os = "macos")]
pub use resident_batch::JpegMetalResidentBatchReport;
pub use session::{MetalBackendSession, MetalSession};
pub(crate) use surface::Storage;
pub use surface::Surface;
#[cfg(target_os = "macos")]
pub use surface::{
    MetalBatchOutputBuffer, MetalBatchTextureOutput, MetalTextureTile, ResidentPrivateJpegTile,
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

pub(crate) fn decode_surface_from_shared_input(
    input: &SharedJpegInput,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: batch::BatchOp,
    fast_packet: Option<&SharedJpegFastPacket>,
    decoder_baseline_bytes: usize,
    fallback_live_bytes: usize,
) -> Result<Surface, Error> {
    let decoder = input.decoder_with_external_live(decoder_baseline_bytes)?;
    let external_live_bytes = fallback_live_bytes
        .checked_add(j2k_jpeg::adapter::decoder_retained_allocation_bytes(
            &decoder,
        )?)
        .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
            "JPEG Metal fallback decoder owner baseline overflow",
        ))?;
    let mut pool = CpuScratchPool::new();
    let build_auto_packets =
        matches!(backend, BackendRequest::Auto) && decoder.info().restart_interval.is_some();
    let build_metal_packets = matches!(backend, BackendRequest::Metal);
    let fast_packet = if build_auto_packets || build_metal_packets {
        fast_packet
    } else {
        None
    };
    let packets = JpegFastPackets::from_shared(fast_packet);
    decode_surface_from_decoder(
        &decoder,
        &mut pool,
        fmt,
        backend,
        op,
        packets,
        external_live_bytes,
    )
}

#[cfg(not(target_os = "macos"))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "the non-Metal stub preserves the cross-platform batch result contract"
)]
pub(crate) fn decode_compatible_batch(
    requests: &[batch::QueuedRequest],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let _ = requests;
    Ok(None)
}

#[cfg_attr(
    not(target_os = "macos"),
    expect(
        clippy::unnecessary_wraps,
        reason = "the non-Metal branch preserves the cross-platform session result contract"
    )
)]
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
    let mut budget =
        batch_allocation::BatchMetadataBudget::new("JPEG Metal device batch request plan");
    let mut requests = budget.try_vec(inputs.len(), "JPEG Metal device batch requests")?;
    let mut plan_owners = plan_owner_ledger::PlanOwnerLedger::default();
    for input in inputs {
        let external_live_bytes = plan_owners.external_live_bytes(budget.live_bytes())?;
        let resolved = state.resolve_jpeg_plan_with_external_live(
            input,
            BackendRequest::Metal,
            external_live_bytes,
        )?;
        let request = batch::QueuedRequest::new_shared(
            resolved.input,
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            resolved.fast_packet,
            resolved.shape,
        );
        let admission = plan_owners.preflight(
            &requests,
            &request,
            state.jpeg_plan_cache_diagnostics().retained_bytes,
        )?;
        plan_owner_ledger::preflight_collective_metadata(
            "JPEG Metal direct device request owners and metadata",
            admission.retained_bytes(),
            state.jpeg_plan_cache_diagnostics().retained_bytes,
            budget.live_bytes(),
        )?;
        requests.push(request);
        plan_owners.commit(admission);
    }
    batch::stamp_execution_owner_baseline(&mut requests, 0, budget.live_bytes());
    drop(state);

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
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    match op {
        batch::BatchOp::Full => match backend {
            BackendRequest::Cpu => decode_full_cpu_upload(decoder, pool, fmt, external_live_bytes),
            BackendRequest::Auto | BackendRequest::Metal => {
                let decision = choose_route(decoder, backend, fmt, op, packets);
                if let Some(err) = routing::decision_error(decision) {
                    return Err(err);
                }
                match decision {
                    routing::RouteDecision::CpuHost => {
                        decode_full_cpu_upload(decoder, pool, fmt, external_live_bytes)
                    }
                    routing::RouteDecision::MetalKernel => {
                        #[cfg(target_os = "macos")]
                        {
                            reject_cpu_staged_metal_upload(compute::decode_to_surface(
                                decoder,
                                pool,
                                fmt,
                                packets,
                                external_live_bytes,
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
            BackendRequest::Cpu => {
                decode_region_cpu_upload(decoder, pool, fmt, roi, external_live_bytes)
            }
            BackendRequest::Auto | BackendRequest::Metal => {
                let decision = choose_route(decoder, backend, fmt, op, packets);
                if let Some(err) = routing::decision_error(decision) {
                    return Err(err);
                }
                match decision {
                    routing::RouteDecision::CpuHost => {
                        decode_region_cpu_upload(decoder, pool, fmt, roi, external_live_bytes)
                    }
                    routing::RouteDecision::MetalKernel => {
                        #[cfg(target_os = "macos")]
                        {
                            reject_cpu_staged_metal_upload(compute::decode_region_to_surface(
                                decoder,
                                pool,
                                fmt,
                                roi.into(),
                                packets,
                                external_live_bytes,
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
            BackendRequest::Cpu => {
                decode_scaled_cpu_upload(decoder, pool, fmt, scale, external_live_bytes)
            }
            BackendRequest::Auto | BackendRequest::Metal => {
                let decision = choose_route(decoder, backend, fmt, op, packets);
                if let Some(err) = routing::decision_error(decision) {
                    return Err(err);
                }
                match decision {
                    routing::RouteDecision::CpuHost => {
                        decode_scaled_cpu_upload(decoder, pool, fmt, scale, external_live_bytes)
                    }
                    routing::RouteDecision::MetalKernel => {
                        #[cfg(target_os = "macos")]
                        {
                            reject_cpu_staged_metal_upload(compute::decode_scaled_to_surface(
                                decoder,
                                pool,
                                fmt,
                                scale,
                                packets,
                                external_live_bytes,
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
            decoder,
            pool,
            RegionScaledSurfaceRequest {
                fmt,
                roi,
                scale,
                backend,
                packets,
                external_live_bytes,
            },
        ),
    }
}

fn decode_full_cpu_upload(
    decoder: &CpuDecoder<'_>,
    _pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    let dims = decoder.info().dimensions;
    decode_cpu_request_upload(
        decoder,
        CpuDecodeRequest::full(fmt),
        dims,
        fmt,
        BackendRequest::Cpu,
        external_live_bytes,
    )
}

fn decode_region_cpu_upload(
    decoder: &CpuDecoder<'_>,
    _pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    roi: Rect,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    let dims = (roi.w, roi.h);
    decode_cpu_request_upload(
        decoder,
        CpuDecodeRequest::region_scaled(fmt, roi.into(), Downscale::None),
        dims,
        fmt,
        BackendRequest::Cpu,
        external_live_bytes,
    )
}

fn decode_scaled_cpu_upload(
    decoder: &CpuDecoder<'_>,
    _pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    scale: Downscale,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    let dims = scaled_dims(decoder.info().dimensions, scale);
    decode_cpu_request_upload(
        decoder,
        CpuDecodeRequest::scaled(fmt, scale),
        dims,
        fmt,
        BackendRequest::Cpu,
        external_live_bytes,
    )
}

#[derive(Clone, Copy)]
struct RegionScaledSurfaceRequest<'a> {
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    backend: BackendRequest,
    packets: JpegFastPackets<'a>,
    external_live_bytes: usize,
}

fn decode_region_scaled_surface_from_decoder(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    request: RegionScaledSurfaceRequest<'_>,
) -> Result<Surface, Error> {
    let RegionScaledSurfaceRequest {
        fmt,
        roi,
        scale,
        backend,
        packets,
        external_live_bytes,
    } = request;
    match backend {
        BackendRequest::Cpu => decode_region_scaled_cpu_upload(
            decoder,
            pool,
            fmt,
            roi,
            scale,
            BackendRequest::Cpu,
            external_live_bytes,
        ),
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
                    external_live_bytes,
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
                            external_live_bytes,
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
        match jpeg_route_profile_fields(backend, fmt, op, capabilities, labels) {
            Ok(fields) => j2k_profile::emit_gpu_route_fields("jpeg", "metal", &fields),
            Err(error) => {
                j2k_profile::emit_profile_error("jpeg_metal_gpu_route_fields", &error);
            }
        }
    }
    decision
}

fn jpeg_route_profile_fields(
    backend: BackendRequest,
    fmt: PixelFormat,
    op: batch::BatchOp,
    capabilities: routing::JpegMetalCapabilities,
    labels: j2k_metal_support::MetalRouteProfileLabels,
) -> j2k_profile::ProfileResult<[j2k_profile::ProfileField; 7]> {
    Ok([
        j2k_profile::ProfileField::label("request", format_args!("{backend:?}"))?,
        j2k_profile::ProfileField::label("fmt", format_args!("{fmt:?}"))?,
        j2k_profile::ProfileField::label("op", jpeg_batch_op_profile(op))?,
        j2k_profile::ProfileField::label("has_fast_packet", capabilities.has_fast_packet())?,
        j2k_profile::ProfileField::label(
            "supports_output_format",
            capabilities.supports_output_format(),
        )?,
        j2k_profile::ProfileField::label("decision", labels.decision)?,
        j2k_profile::ProfileField::label("reason", labels.reason)?,
    ])
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
    _pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    backend: BackendRequest,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    let scaled = roi.scaled_covering(scale);
    let dims = (scaled.w, scaled.h);
    decode_cpu_request_upload(
        decoder,
        CpuDecodeRequest::region_scaled(fmt, roi.into(), scale),
        dims,
        fmt,
        backend,
        external_live_bytes,
    )
}

fn decode_cpu_request_upload(
    decoder: &CpuDecoder<'_>,
    request: CpuDecodeRequest,
    dims: (u32, u32),
    fmt: PixelFormat,
    backend: BackendRequest,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    let decoder_retained_bytes = j2k_jpeg::adapter::decoder_retained_allocation_bytes(decoder)?;
    let external_live_bytes = external_live_bytes
        .checked_sub(decoder_retained_bytes)
        .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
            "JPEG Metal CPU fallback decoder baseline underflow",
        ))?;
    let (output, _) = decoder.decode_request_with_external_live(request, external_live_bytes)?;
    upload_surface(output, dims, fmt, backend)
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
            storage: Storage::Host(std::sync::Arc::new(bytes)),
        }),
        BackendRequest::Auto | BackendRequest::Metal => {
            #[cfg(target_os = "macos")]
            {
                let device = Device::system_default().ok_or(Error::MetalUnavailable)?;
                let buffer = buffers::new_shared_buffer_with_data(&device, &bytes)?;
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
                        storage: Storage::Host(std::sync::Arc::new(bytes)),
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
