// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, PixelFormat};
use j2k_jpeg::{
    adapter::{JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1},
    metal_hybrid_eligibility, Decoder as CpuDecoder, JpegBackendEligibility, JpegCapabilityRequest,
    JpegDecodeOp,
};
use j2k_metal_support::{
    cpu_host_route, metal_kernel_route, metal_unavailable_route, reject_explicit_metal_route,
    reject_unsupported_backend_route, MetalRouteProfileLabels,
};

use crate::{batch::BatchOp, Error};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteDecision {
    CpuHost,
    #[cfg_attr(
        not(target_os = "macos"),
        expect(dead_code, reason = "the Metal route is constructed only on macOS")
    )]
    MetalKernel,
    /// CPU entropy decode, then Metal IDCT, upsampling, color, and packing.
    #[cfg_attr(
        not(target_os = "macos"),
        expect(dead_code, reason = "the Metal route is constructed only on macOS")
    )]
    MetalHybrid,
    RejectExplicitMetal {
        reason: ExplicitMetalRejection,
    },
    RejectUnsupportedBackend {
        request: BackendRequest,
    },
    #[cfg_attr(
        target_os = "macos",
        expect(
            dead_code,
            reason = "Metal-unavailable routing is constructed only on non-macOS hosts"
        )
    )]
    MetalUnavailable,
}

impl RouteDecision {
    pub(crate) fn profile_labels(self) -> MetalRouteProfileLabels {
        match self {
            Self::CpuHost => cpu_host_route(),
            Self::MetalKernel => metal_kernel_route(),
            Self::MetalHybrid => {
                MetalRouteProfileLabels::new("metal_hybrid", "cpu_entropy_metal_reconstruction")
            }
            Self::RejectExplicitMetal { reason } => {
                reject_explicit_metal_route(reason.profile_reason())
            }
            Self::RejectUnsupportedBackend { .. } => reject_unsupported_backend_route(),
            Self::MetalUnavailable => metal_unavailable_route(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExplicitMetalRejection {
    UnsupportedOutputFormat,
    /// CPU decode, which hybrid Metal decode mirrors, rejects the request.
    UnsupportedJpeg {
        reason: &'static str,
    },
}

impl ExplicitMetalRejection {
    fn error_reason(self) -> &'static str {
        match self {
            Self::UnsupportedOutputFormat => {
                "JPEG Metal supports explicit requests only for Gray8, Rgb8, Rgba8, Gray16, Rgb16, or Rgba16 output formats"
            }
            Self::UnsupportedJpeg { reason } => reason,
        }
    }

    fn profile_reason(self) -> &'static str {
        match self {
            Self::UnsupportedOutputFormat => "unsupported_format",
            Self::UnsupportedJpeg { .. } => "unsupported_jpeg",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JpegMetalCapabilities {
    has_fast_packet: bool,
    /// Gray8, Rgb8, or Rgba8: the formats the fast-packet kernels emit.
    supports_fast_output_format: bool,
    /// Any of the six formats some Metal route can emit.
    supports_output_format: bool,
    hybrid: JpegBackendEligibility,
}

impl JpegMetalCapabilities {
    pub(crate) fn has_fast_packet(self) -> bool {
        self.has_fast_packet
    }

    pub(crate) fn supports_output_format(self) -> bool {
        self.supports_output_format
    }

    pub(crate) fn hybrid_eligible(self) -> bool {
        self.hybrid.eligible
    }

    pub(crate) fn for_request(
        decoder: &CpuDecoder<'_>,
        fmt: PixelFormat,
        op: BatchOp,
        fast444_packet: Option<&JpegFast444PacketV1>,
        fast422_packet: Option<&JpegFast422PacketV1>,
        fast420_packet: Option<&JpegFast420PacketV1>,
    ) -> Self {
        let has_fast_packet =
            fast444_packet.is_some() || fast422_packet.is_some() || fast420_packet.is_some();
        let supports_fast_output_format = matches!(
            fmt,
            PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8
        );
        let supports_output_format = supports_metal_output_format(fmt);
        let hybrid = metal_hybrid_eligibility(
            decoder.info(),
            JpegCapabilityRequest {
                op: capability_op(op),
                fmt,
            },
        );

        Self {
            has_fast_packet,
            supports_fast_output_format,
            supports_output_format,
            hybrid,
        }
    }
}

fn capability_op(op: BatchOp) -> JpegDecodeOp {
    match op {
        BatchOp::Full => JpegDecodeOp::Full,
        BatchOp::Region(roi) => JpegDecodeOp::Region(roi.into()),
        BatchOp::Scaled(scale) => JpegDecodeOp::Scaled(scale),
        BatchOp::RegionScaled { roi, scale } => JpegDecodeOp::RegionScaled {
            roi: roi.into(),
            scale,
        },
    }
}

pub(crate) fn decide_route(
    backend: BackendRequest,
    capabilities: JpegMetalCapabilities,
) -> RouteDecision {
    match backend {
        BackendRequest::Cpu => RouteDecision::CpuHost,
        BackendRequest::Auto => {
            let _ = capabilities;
            RouteDecision::CpuHost
        }
        BackendRequest::Metal => {
            let fast = capabilities.has_fast_packet && capabilities.supports_fast_output_format;
            if !fast && !capabilities.hybrid.eligible {
                return RouteDecision::RejectExplicitMetal {
                    reason: explicit_metal_rejection(capabilities),
                };
            }

            #[cfg(not(target_os = "macos"))]
            {
                RouteDecision::MetalUnavailable
            }
            #[cfg(target_os = "macos")]
            {
                if fast {
                    RouteDecision::MetalKernel
                } else {
                    RouteDecision::MetalHybrid
                }
            }
        }
        BackendRequest::Cuda => RouteDecision::RejectUnsupportedBackend {
            request: BackendRequest::Cuda,
        },
    }
}

/// Why an explicit Metal request neither fast nor hybrid decode accepts is
/// rejected.
fn explicit_metal_rejection(capabilities: JpegMetalCapabilities) -> ExplicitMetalRejection {
    if !capabilities.supports_output_format {
        return ExplicitMetalRejection::UnsupportedOutputFormat;
    }
    ExplicitMetalRejection::UnsupportedJpeg {
        reason: capabilities
            .hybrid
            .reason
            .unwrap_or("JPEG Metal decode rejected this request"),
    }
}

fn supports_metal_output_format(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::Gray8
            | PixelFormat::Rgb8
            | PixelFormat::Rgba8
            | PixelFormat::Gray16
            | PixelFormat::Rgb16
            | PixelFormat::Rgba16
    )
}

pub(crate) fn decision_error(decision: RouteDecision) -> Option<Error> {
    match decision {
        RouteDecision::RejectExplicitMetal { reason } => Some(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_operation(reason.error_reason()),
        )),
        RouteDecision::RejectUnsupportedBackend { request } => {
            Some(Error::UnsupportedBackend { request })
        }
        RouteDecision::MetalUnavailable => Some(Error::MetalUnavailable),
        RouteDecision::CpuHost | RouteDecision::MetalKernel | RouteDecision::MetalHybrid => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JpegFastPackets;

    const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
    const BASELINE_422: &[u8] = include_bytes!("../fixtures/jpeg/baseline_422_16x8.jpg");
    const BASELINE_444: &[u8] = include_bytes!("../fixtures/jpeg/baseline_444_8x8.jpg");
    const GRAYSCALE: &[u8] = include_bytes!("../fixtures/jpeg/grayscale_8x8.jpg");

    fn capabilities_for(bytes: &[u8], fmt: PixelFormat) -> JpegMetalCapabilities {
        let mut plans = j2k_jpeg::adapter::JpegPlanCache::default();
        let (plan, decoder) = plans
            .resolve_with_decoder_and_external_live(bytes, 0)
            .expect("cached decoder plan");
        let packets = JpegFastPackets::from_shared(plan.fast_packet());

        JpegMetalCapabilities::for_request(
            &decoder,
            fmt,
            BatchOp::Full,
            packets.fast444,
            packets.fast422,
            packets.fast420,
        )
    }

    fn eligible_everywhere() -> JpegMetalCapabilities {
        JpegMetalCapabilities {
            has_fast_packet: true,
            supports_fast_output_format: true,
            supports_output_format: true,
            hybrid: capabilities_for(BASELINE_420, PixelFormat::Rgb8).hybrid,
        }
    }

    #[test]
    fn cuda_route_reports_unsupported_backend() {
        let capabilities = eligible_everywhere();

        assert_eq!(
            decide_route(BackendRequest::Cuda, capabilities),
            RouteDecision::RejectUnsupportedBackend {
                request: BackendRequest::Cuda
            }
        );
        assert!(matches!(
            decision_error(decide_route(BackendRequest::Cuda, capabilities)),
            Some(Error::UnsupportedBackend {
                request: BackendRequest::Cuda
            })
        ));
    }

    #[test]
    fn explicit_metal_unsupported_output_format_is_rejected_before_launch() {
        let capabilities = capabilities_for(BASELINE_420, PixelFormat::RgbI16);

        assert!(!capabilities.supports_output_format());
        assert!(matches!(
            decide_route(BackendRequest::Metal, capabilities),
            RouteDecision::RejectExplicitMetal {
                reason: ExplicitMetalRejection::UnsupportedOutputFormat
            }
        ));
        let labels = decide_route(BackendRequest::Metal, capabilities).profile_labels();
        assert_eq!(labels.decision, "reject_explicit_metal");
        assert_eq!(labels.reason, "unsupported_format");
    }

    #[test]
    fn explicit_metal_accepts_fast_baseline_sampling_families() {
        for (name, bytes) in [
            ("fast420", BASELINE_420),
            ("fast422", BASELINE_422),
            ("fast444", BASELINE_444),
        ] {
            let capabilities = capabilities_for(bytes, PixelFormat::Rgb8);

            assert!(capabilities.has_fast_packet(), "{name}");
            assert!(capabilities.supports_output_format(), "{name}");
            assert!(
                matches!(
                    decide_route(BackendRequest::Metal, capabilities),
                    RouteDecision::MetalKernel | RouteDecision::MetalUnavailable
                ),
                "{name}"
            );
        }
    }

    #[test]
    fn explicit_metal_routes_jpeg_without_fast_packet_to_hybrid_decode() {
        let capabilities = capabilities_for(GRAYSCALE, PixelFormat::Gray8);

        assert!(!capabilities.has_fast_packet());
        assert!(capabilities.hybrid_eligible());
        let decision = decide_route(BackendRequest::Metal, capabilities);
        assert!(matches!(
            decision,
            RouteDecision::MetalHybrid | RouteDecision::MetalUnavailable
        ));
        #[cfg(target_os = "macos")]
        {
            let labels = decision.profile_labels();
            assert_eq!(labels.decision, "metal_hybrid");
        }
    }

    #[test]
    fn explicit_metal_rejects_what_cpu_decode_rejects_with_the_cpu_reason() {
        for bytes in [GRAYSCALE, BASELINE_420] {
            let capabilities = capabilities_for(bytes, PixelFormat::Rgb16);

            assert!(capabilities.supports_output_format());
            assert!(!capabilities.hybrid_eligible());
            let decision = decide_route(BackendRequest::Metal, capabilities);
            let RouteDecision::RejectExplicitMetal {
                reason: ExplicitMetalRejection::UnsupportedJpeg { reason },
            } = decision
            else {
                panic!("expected a CPU-mirrored rejection, got {decision:?}");
            };
            assert!(
                reason.starts_with("JPEG Metal decode supports Gray8, Rgb8, or Rgba8"),
                "{reason}"
            );
            let labels = decision.profile_labels();
            assert_eq!(labels.decision, "reject_explicit_metal");
            assert_eq!(labels.reason, "unsupported_jpeg");
            assert!(matches!(
                decision_error(decision),
                Some(Error::UnsupportedMetalRequest { reason: message }) if message == reason
            ));
        }
    }

    #[test]
    fn auto_routes_single_request_to_cpu_even_when_metal_capabilities_match() {
        assert_eq!(
            decide_route(BackendRequest::Auto, eligible_everywhere()),
            RouteDecision::CpuHost
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn explicit_metal_unsupported_shape_is_rejected_before_host_unavailability() {
        let capabilities = capabilities_for(GRAYSCALE, PixelFormat::Rgb16);

        assert!(matches!(
            decide_route(BackendRequest::Metal, capabilities),
            RouteDecision::RejectExplicitMetal {
                reason: ExplicitMetalRejection::UnsupportedJpeg { .. }
            }
        ));
    }
}
