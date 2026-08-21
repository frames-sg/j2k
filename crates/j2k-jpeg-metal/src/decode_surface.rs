// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified JPEG surface routing, CPU fallback, and Metal upload operations.

use crate::{
    batch, batch_allocation, plan_owner_ledger, routing, session, Error, JpegFastPackets,
    MetalBackendSession, SharedJpegFastPacket, SharedJpegInput, Storage, Surface,
};
#[cfg(target_os = "macos")]
use crate::{
    buffers,
    compute::{batch_entry, single_decode},
};
use j2k_core::{BackendKind, BackendRequest, Downscale, PixelFormat, Rect, SurfaceResidency};
use j2k_jpeg::{
    DecodeRequest as CpuDecodeRequest, Decoder as CpuDecoder, ScratchPool as CpuScratchPool,
};

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

pub(crate) fn decode_compatible_batch_with_session(
    requests: &[batch::QueuedRequest],
    session: &mut session::SessionState,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    #[cfg(target_os = "macos")]
    {
        batch_entry::decode_full_batch_to_surfaces_with_session_state(requests, session)
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

    batch_entry::decode_full_batch_to_surfaces_with_session(&requests, session)
}

#[expect(
    clippy::too_many_lines,
    reason = "the decoder dispatcher keeps the ordered fast-packet routes and CPU fallback together so backend selection stays deterministic"
)]
pub(crate) fn decode_surface_from_decoder(
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
                            reject_cpu_staged_metal_upload(single_decode::decode_to_surface(
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
                            reject_cpu_staged_metal_upload(single_decode::decode_region_to_surface(
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
                            reject_cpu_staged_metal_upload(single_decode::decode_scaled_to_surface(
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
                        reject_cpu_staged_metal_upload(
                            single_decode::decode_region_scaled_to_surface(
                                decoder,
                                pool,
                                fmt,
                                roi.into(),
                                scale,
                                packets,
                                external_live_bytes,
                            )?,
                        )
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
pub(crate) fn reject_cpu_staged_metal_upload(surface: Surface) -> Result<Surface, Error> {
    if surface.residency() == SurfaceResidency::CpuStagedMetalUpload {
        return Err(Error::capability_rejected(j2k_core::CapabilityRejection::unsupported_operation("JPEG Metal explicit device decode requires a direct resident Metal decode; use the CPU path for CPU-staged output")));
    }
    Ok(surface)
}

pub(crate) fn choose_route(
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

pub(crate) fn scaled_dims(full: (u32, u32), scale: Downscale) -> (u32, u32) {
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
                let device = j2k_metal_support::system_default_device()
                    .map_err(|_| Error::MetalUnavailable)?;
                let buffer = buffers::new_shared_buffer_with_data(&device, &bytes)?;
                Surface::from_cpu_staged_metal_buffer(buffer, dimensions, fmt)
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
