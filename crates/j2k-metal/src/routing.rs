// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, PixelFormat};
#[cfg(target_os = "macos")]
use j2k_metal_support::metal_kernel_route;
use j2k_metal_support::{
    cpu_host_route, reject_explicit_metal_route, reject_unsupported_backend_route,
    MetalRouteProfileLabels,
};

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteDecision {
    CpuHost,
    #[cfg(target_os = "macos")]
    MetalKernel,
    RejectExplicitMetal {
        reason: ExplicitMetalRejection,
    },
    RejectUnsupportedBackend {
        request: BackendRequest,
    },
    #[cfg(not(target_os = "macos"))]
    MetalUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExplicitMetalRejection {
    UnsupportedFormat { fmt: PixelFormat },
}

impl ExplicitMetalRejection {
    fn error_reason(self) -> &'static str {
        match self {
            Self::UnsupportedFormat {
                fmt: PixelFormat::Rgba16,
            } => "J2K Metal does not support PixelFormat::Rgba16",
            Self::UnsupportedFormat { .. } => {
                "J2K Metal does not support the requested PixelFormat"
            }
        }
    }

    fn profile_reason(self) -> &'static str {
        match self {
            Self::UnsupportedFormat { .. } => "unsupported_format",
        }
    }
}

pub(crate) fn supports_metal_format(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::Gray8
            | PixelFormat::Rgb8
            | PixelFormat::Rgba8
            | PixelFormat::Gray16
            | PixelFormat::Rgb16
    )
}

pub(crate) fn decide_route(backend: BackendRequest, fmt: PixelFormat) -> RouteDecision {
    let decision = match backend {
        BackendRequest::Cpu | BackendRequest::Auto => RouteDecision::CpuHost,
        BackendRequest::Metal => {
            if supports_metal_format(fmt) {
                #[cfg(not(target_os = "macos"))]
                {
                    RouteDecision::MetalUnavailable
                }
                #[cfg(target_os = "macos")]
                {
                    RouteDecision::MetalKernel
                }
            } else {
                RouteDecision::RejectExplicitMetal {
                    reason: unsupported_metal_format_reason(fmt),
                }
            }
        }
        BackendRequest::Cuda => RouteDecision::RejectUnsupportedBackend {
            request: BackendRequest::Cuda,
        },
    };
    if j2k_profile::gpu_route_profile_enabled() {
        let request_s = format!("{backend:?}");
        let fmt_s = format!("{fmt:?}");
        let labels = j2k_route_decision_profile(decision);
        j2k_profile::emit_gpu_route_profile(
            "j2k",
            "metal",
            &[
                ("request", request_s.as_str()),
                ("fmt", fmt_s.as_str()),
                ("decision", labels.decision),
                ("reason", labels.reason),
            ],
        );
    }
    decision
}

pub(crate) fn decision_error(decision: RouteDecision) -> Option<Error> {
    match decision {
        RouteDecision::RejectExplicitMetal { reason } => Some(Error::UnsupportedMetalRequest {
            reason: reason.error_reason(),
        }),
        RouteDecision::RejectUnsupportedBackend { request } => {
            Some(Error::UnsupportedBackend { request })
        }
        #[cfg(not(target_os = "macos"))]
        RouteDecision::MetalUnavailable => Some(Error::MetalUnavailable),
        #[cfg(target_os = "macos")]
        RouteDecision::CpuHost | RouteDecision::MetalKernel => None,
        #[cfg(not(target_os = "macos"))]
        RouteDecision::CpuHost => None,
    }
}

fn unsupported_metal_format_reason(fmt: PixelFormat) -> ExplicitMetalRejection {
    ExplicitMetalRejection::UnsupportedFormat { fmt }
}

fn j2k_route_decision_profile(decision: RouteDecision) -> MetalRouteProfileLabels {
    match decision {
        RouteDecision::CpuHost => cpu_host_route(),
        #[cfg(target_os = "macos")]
        RouteDecision::MetalKernel => metal_kernel_route(),
        RouteDecision::RejectExplicitMetal { reason } => {
            reject_explicit_metal_route(reason.profile_reason())
        }
        RouteDecision::RejectUnsupportedBackend { .. } => reject_unsupported_backend_route(),
        #[cfg(not(target_os = "macos"))]
        RouteDecision::MetalUnavailable => j2k_metal_support::metal_unavailable_route(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_route_reports_unsupported_backend() {
        assert_eq!(
            decide_route(BackendRequest::Cuda, PixelFormat::Rgba16),
            RouteDecision::RejectUnsupportedBackend {
                request: BackendRequest::Cuda
            }
        );
        assert!(matches!(
            decision_error(decide_route(BackendRequest::Cuda, PixelFormat::Rgba16)),
            Some(Error::UnsupportedBackend {
                request: BackendRequest::Cuda
            })
        ));
    }

    #[test]
    fn explicit_metal_unsupported_format_is_rejected_before_launch() {
        assert!(matches!(
            decide_route(BackendRequest::Metal, PixelFormat::Rgba16),
            RouteDecision::RejectExplicitMetal {
                reason: ExplicitMetalRejection::UnsupportedFormat {
                    fmt: PixelFormat::Rgba16
                }
            }
        ));
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn explicit_metal_unsupported_format_is_rejected_before_host_unavailability() {
        assert!(matches!(
            decide_route(BackendRequest::Metal, PixelFormat::Rgba16),
            RouteDecision::RejectExplicitMetal {
                reason: ExplicitMetalRejection::UnsupportedFormat {
                    fmt: PixelFormat::Rgba16
                }
            }
        ));
        assert!(matches!(
            decide_route(BackendRequest::Metal, PixelFormat::Rgb8),
            RouteDecision::MetalUnavailable
        ));
    }
}
