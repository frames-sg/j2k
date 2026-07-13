// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::BackendRequest;
#[cfg(target_os = "macos")]
use j2k_core::PixelFormat;
#[cfg(target_os = "macos")]
use j2k_jpeg::adapter::JpegPlanCache;
use j2k_jpeg::Decoder as CpuDecoder;

use super::{is_contiguous_viewport_workload, ViewportWorkload};
#[cfg(target_os = "macos")]
use super::{validate_viewport_workload_budget, viewport_source_bounds};
#[cfg(target_os = "macos")]
use crate::{batch, fast_packets::JpegFastPackets, routing};
use crate::{Error, SharedJpegFastPacket};

#[cfg(all(target_os = "macos", test))]
mod resident;
#[cfg(all(target_os = "macos", test))]
pub(crate) use resident::choose_resizable_metal_viewport_strategy;
#[cfg(all(target_os = "macos", test))]
pub(super) use resident::{
    choose_resizable_metal_viewport_strategy_for_decoder,
    validate_resident_viewport_composition_request,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Execution strategy selected for a viewport decode.
pub enum ViewportSurfaceStrategy {
    /// Decode each tile on CPU and composite into a host viewport.
    CpuComposite,
    /// Decode one contiguous source region on CPU.
    CpuContiguous,
    /// Decode or upload through Metal while compositing multiple source tiles.
    HybridComposite,
    /// Decode one contiguous source region through the Metal path.
    HybridContiguous,
}

pub(super) struct ResolvedViewportSurfacePlan {
    pub(super) strategy: ViewportSurfaceStrategy,
    pub(super) fast_packet: Option<SharedJpegFastPacket>,
    pub(super) external_live_bytes: usize,
}

#[cfg(all(target_os = "macos", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Resident Metal output strategy selected for a reusable viewport decode.
pub(crate) enum ViewportResidentOutputStrategy {
    /// Decode the contiguous source bounds through the direct resident batch path.
    DirectContiguous,
    /// Decode component rows into resident planes and pack the composed viewport.
    Composite,
}

/// Choose the backend strategy for a workload without inspecting JPEG capabilities.
pub fn choose_viewport_surface_strategy(
    workload: &ViewportWorkload,
    backend: BackendRequest,
) -> Result<ViewportSurfaceStrategy, Error> {
    let contiguous = is_contiguous_viewport_workload(workload);
    match backend {
        BackendRequest::Cpu => Ok(if contiguous {
            ViewportSurfaceStrategy::CpuContiguous
        } else {
            ViewportSurfaceStrategy::CpuComposite
        }),
        BackendRequest::Auto | BackendRequest::Metal => {
            #[cfg(target_os = "macos")]
            {
                Ok(if contiguous {
                    ViewportSurfaceStrategy::HybridContiguous
                } else {
                    ViewportSurfaceStrategy::HybridComposite
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                if matches!(backend, BackendRequest::Metal) {
                    Err(Error::MetalUnavailable)
                } else if contiguous {
                    Ok(ViewportSurfaceStrategy::CpuContiguous)
                } else {
                    Ok(ViewportSurfaceStrategy::CpuComposite)
                }
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    }
}

#[cfg(test)]
pub(super) fn choose_viewport_surface_strategy_for_decoder(
    decoder: &CpuDecoder<'_>,
    workload: &ViewportWorkload,
    backend: BackendRequest,
) -> Result<ViewportSurfaceStrategy, Error> {
    resolve_viewport_surface_plan(decoder, workload, backend).map(|plan| plan.strategy)
}

pub(super) fn resolve_viewport_surface_plan(
    decoder: &CpuDecoder<'_>,
    workload: &ViewportWorkload,
    backend: BackendRequest,
) -> Result<ResolvedViewportSurfacePlan, Error> {
    let decoder_live_bytes = j2k_jpeg::adapter::decoder_retained_allocation_bytes(decoder)?;
    if matches!(backend, BackendRequest::Metal) {
        #[cfg(target_os = "macos")]
        {
            let (fast_packet, external_live_bytes) =
                build_resolved_viewport_packet(decoder, decoder_live_bytes)?;
            validate_explicit_metal_viewport_request_with_packets(
                decoder,
                workload,
                JpegFastPackets::from_shared(fast_packet.as_ref()),
                external_live_bytes,
            )?;
            return Ok(ResolvedViewportSurfacePlan {
                strategy: choose_viewport_surface_strategy(workload, backend)?,
                fast_packet,
                external_live_bytes,
            });
        }
        #[cfg(not(target_os = "macos"))]
        {
            return choose_viewport_surface_strategy(workload, backend).map(|strategy| {
                ResolvedViewportSurfacePlan {
                    strategy,
                    fast_packet: None,
                    external_live_bytes: decoder_live_bytes,
                }
            });
        }
    }

    if !matches!(backend, BackendRequest::Auto) {
        return choose_viewport_surface_strategy(workload, backend).map(|strategy| {
            ResolvedViewportSurfacePlan {
                strategy,
                fast_packet: None,
                external_live_bytes: decoder_live_bytes,
            }
        });
    }

    #[cfg(not(target_os = "macos"))]
    let _ = decoder;

    #[cfg(target_os = "macos")]
    {
        let contiguous = is_contiguous_viewport_workload(workload);
        if !contiguous {
            return Ok(ResolvedViewportSurfacePlan {
                strategy: ViewportSurfaceStrategy::CpuComposite,
                fast_packet: None,
                external_live_bytes: decoder_live_bytes,
            });
        }

        let restart_coded = decoder.info().restart_interval.is_some();
        if !restart_coded {
            return Ok(ResolvedViewportSurfacePlan {
                strategy: ViewportSurfaceStrategy::CpuContiguous,
                fast_packet: None,
                external_live_bytes: decoder_live_bytes,
            });
        }

        let (fast_packet, external_live_bytes) =
            build_resolved_viewport_packet(decoder, decoder_live_bytes)?;
        Ok(ResolvedViewportSurfacePlan {
            strategy: if fast_packet.is_some() {
                ViewportSurfaceStrategy::HybridContiguous
            } else {
                ViewportSurfaceStrategy::CpuContiguous
            },
            fast_packet,
            external_live_bytes,
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        choose_viewport_surface_strategy(workload, backend).map(|strategy| {
            ResolvedViewportSurfacePlan {
                strategy,
                fast_packet: None,
                external_live_bytes: decoder_live_bytes,
            }
        })
    }
}

#[cfg(all(target_os = "macos", test))]
pub(super) fn has_direct_viewport_packet(decoder: &CpuDecoder<'_>) -> Result<bool, Error> {
    let decoder_live_bytes = j2k_jpeg::adapter::decoder_retained_allocation_bytes(decoder)?;
    build_resolved_viewport_packet(decoder, decoder_live_bytes).map(|(packet, _)| packet.is_some())
}

#[cfg(target_os = "macos")]
pub(super) fn validate_explicit_metal_viewport_request_with_packets(
    decoder: &CpuDecoder<'_>,
    workload: &ViewportWorkload,
    fast_packets: JpegFastPackets<'_>,
    external_live_bytes: usize,
) -> Result<(), Error> {
    validate_viewport_workload_budget(workload, external_live_bytes)?;
    let source = viewport_source_bounds(workload);
    let capabilities = routing::JpegMetalCapabilities::for_request(
        decoder,
        PixelFormat::Rgb8,
        batch::BatchOp::RegionScaled {
            roi: source,
            scale: workload.scale,
        },
        fast_packets.fast444,
        fast_packets.fast422,
        fast_packets.fast420,
    );
    let decision = routing::decide_route(BackendRequest::Metal, capabilities);
    if let Some(err) = routing::decision_error(decision) {
        return Err(err);
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn build_resolved_viewport_packet(
    decoder: &CpuDecoder<'_>,
    decoder_live_bytes: usize,
) -> Result<(Option<SharedJpegFastPacket>, usize), Error> {
    let mut plans = JpegPlanCache::default();
    let plan = plans.resolve_from_decoder_with_external_live(decoder, decoder_live_bytes)?;
    let fast_packet = plan.fast_packet().cloned();
    let packet_live_bytes = fast_packet
        .as_ref()
        .map_or(Ok(0), SharedJpegFastPacket::retained_cache_bytes)?;
    let external_live_bytes = decoder_live_bytes.checked_add(packet_live_bytes).ok_or(
        j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
            "JPEG Metal viewport plan owner baseline overflow",
        ),
    )?;
    Ok((fast_packet, external_live_bytes))
}
