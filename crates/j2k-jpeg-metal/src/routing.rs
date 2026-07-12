// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, PixelFormat};
use j2k_jpeg::{
    adapter::{JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1},
    Decoder as CpuDecoder,
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
    MissingFastPacket,
    UnsupportedOutputFormat,
}

impl ExplicitMetalRejection {
    fn error_reason(self) -> &'static str {
        match self {
            Self::MissingFastPacket => {
                "JPEG Metal supports explicit requests only for fast 4:2:0, 4:2:2, or 4:4:4 baseline packets"
            }
            Self::UnsupportedOutputFormat => {
                "JPEG Metal supports explicit requests only for Gray8, Rgb8, or Rgba8 output formats"
            }
        }
    }

    fn profile_reason(self) -> &'static str {
        match self {
            Self::MissingFastPacket => "no_fast_packet",
            Self::UnsupportedOutputFormat => "unsupported_format",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JpegMetalCapabilities {
    has_fast_packet: bool,
    supports_output_format: bool,
}

impl JpegMetalCapabilities {
    pub(crate) fn has_fast_packet(self) -> bool {
        self.has_fast_packet
    }

    pub(crate) fn supports_output_format(self) -> bool {
        self.supports_output_format
    }

    pub(crate) fn for_request(
        _decoder: &CpuDecoder<'_>,
        fmt: PixelFormat,
        _op: BatchOp,
        fast444_packet: Option<&JpegFast444PacketV1>,
        fast422_packet: Option<&JpegFast422PacketV1>,
        fast420_packet: Option<&JpegFast420PacketV1>,
    ) -> Self {
        let has_fast_packet =
            fast444_packet.is_some() || fast422_packet.is_some() || fast420_packet.is_some();
        let supports_output_format = supports_metal_output_format(fmt);

        Self {
            has_fast_packet,
            supports_output_format,
        }
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
            if !capabilities.has_fast_packet {
                return RouteDecision::RejectExplicitMetal {
                    reason: ExplicitMetalRejection::MissingFastPacket,
                };
            }
            if !capabilities.supports_output_format {
                return RouteDecision::RejectExplicitMetal {
                    reason: ExplicitMetalRejection::UnsupportedOutputFormat,
                };
            }

            #[cfg(not(target_os = "macos"))]
            {
                RouteDecision::MetalUnavailable
            }
            #[cfg(target_os = "macos")]
            {
                RouteDecision::MetalKernel
            }
        }
        BackendRequest::Cuda => RouteDecision::RejectUnsupportedBackend {
            request: BackendRequest::Cuda,
        },
    }
}

fn supports_metal_output_format(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8
    )
}

pub(crate) fn decision_error(decision: RouteDecision) -> Option<Error> {
    match decision {
        RouteDecision::RejectExplicitMetal { reason } => Some(Error::UnsupportedMetalRequest {
            reason: reason.error_reason(),
        }),
        RouteDecision::RejectUnsupportedBackend { request } => {
            Some(Error::UnsupportedBackend { request })
        }
        RouteDecision::MetalUnavailable => Some(Error::MetalUnavailable),
        RouteDecision::CpuHost | RouteDecision::MetalKernel => None,
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

    #[test]
    fn cuda_route_reports_unsupported_backend() {
        let capabilities = JpegMetalCapabilities {
            has_fast_packet: true,
            supports_output_format: true,
        };

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
        let capabilities = JpegMetalCapabilities {
            has_fast_packet: true,
            supports_output_format: false,
        };

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
    fn explicit_metal_rejects_supported_output_when_fast_packet_is_missing() {
        let capabilities = capabilities_for(GRAYSCALE, PixelFormat::Gray8);

        assert!(!capabilities.has_fast_packet());
        assert!(capabilities.supports_output_format());
        assert!(matches!(
            decide_route(BackendRequest::Metal, capabilities),
            RouteDecision::RejectExplicitMetal {
                reason: ExplicitMetalRejection::MissingFastPacket
            }
        ));
        let labels = decide_route(BackendRequest::Metal, capabilities).profile_labels();
        assert_eq!(labels.decision, "reject_explicit_metal");
        assert_eq!(labels.reason, "no_fast_packet");
    }

    #[test]
    fn auto_routes_single_request_to_cpu_even_when_metal_capabilities_match() {
        let capabilities = JpegMetalCapabilities {
            has_fast_packet: true,
            supports_output_format: true,
        };

        assert_eq!(
            decide_route(BackendRequest::Auto, capabilities),
            RouteDecision::CpuHost
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn auto_routes_to_cpu_host_on_non_macos_even_when_metal_would_be_preferred() {
        let capabilities = JpegMetalCapabilities {
            has_fast_packet: true,
            supports_output_format: true,
        };

        assert_eq!(
            decide_route(BackendRequest::Auto, capabilities),
            RouteDecision::CpuHost
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn explicit_metal_unsupported_shape_is_rejected_before_host_unavailability() {
        let capabilities = JpegMetalCapabilities {
            has_fast_packet: false,
            supports_output_format: true,
        };

        assert!(matches!(
            decide_route(BackendRequest::Metal, capabilities),
            RouteDecision::RejectExplicitMetal {
                reason: ExplicitMetalRejection::MissingFastPacket
            }
        ));
    }
}
