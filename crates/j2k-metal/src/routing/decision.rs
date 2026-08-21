// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, PixelFormat};

use crate::Error;

use super::{
    availability::metal_is_compiled,
    eligibility::supports_metal_format,
    rejection::{unsupported_metal_format, ExplicitMetalRejection},
    telemetry,
};

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

pub(crate) fn decide_route(backend: BackendRequest, fmt: PixelFormat) -> RouteDecision {
    let decision = match backend {
        BackendRequest::Cpu | BackendRequest::Auto => RouteDecision::CpuHost,
        BackendRequest::Metal if !supports_metal_format(fmt) => {
            RouteDecision::RejectExplicitMetal {
                reason: unsupported_metal_format(fmt),
            }
        }
        BackendRequest::Metal if metal_is_compiled() => {
            #[cfg(target_os = "macos")]
            {
                RouteDecision::MetalKernel
            }
            #[cfg(not(target_os = "macos"))]
            unreachable!("compile-time availability must match the target")
        }
        BackendRequest::Metal => {
            #[cfg(not(target_os = "macos"))]
            {
                RouteDecision::MetalUnavailable
            }
            #[cfg(target_os = "macos")]
            unreachable!("compile-time availability must match the target")
        }
        BackendRequest::Cuda => RouteDecision::RejectUnsupportedBackend {
            request: BackendRequest::Cuda,
        },
    };
    telemetry::observe(backend, fmt, decision);
    decision
}

pub(crate) fn decision_error(decision: RouteDecision) -> Option<Error> {
    match decision {
        RouteDecision::RejectExplicitMetal { reason } => Some(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_operation(reason.error_reason()),
        )),
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
