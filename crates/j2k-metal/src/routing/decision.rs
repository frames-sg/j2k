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

/// The widest component precision Metal decodes. The transforms and stores
/// run in f32, which holds every integer only below 2^24, and the 5/3 and
/// RCT sums of wider planes leave that range; the output scales also assume
/// samples of at most 16 bits.
pub(crate) const MAX_METAL_DECODE_PRECISION: u8 = 16;

/// Whether a codestream has a component wider than Metal decodes.
pub(crate) fn exceeds_metal_decode_precision(bytes: &[u8]) -> bool {
    j2k::J2kDecoder::inspect_support(bytes).is_ok_and(|support| {
        support
            .components
            .iter()
            .any(|component| component.bit_depth > MAX_METAL_DECODE_PRECISION)
    })
}

/// [`decide_route`] for an image whose widest component has
/// `max_precision` bits: Metal declines planes wider than
/// [`MAX_METAL_DECODE_PRECISION`], rejecting an explicit Metal request and
/// leaving Auto on the CPU.
pub(crate) fn decide_route_for_precision(
    backend: BackendRequest,
    fmt: PixelFormat,
    max_precision: u8,
) -> RouteDecision {
    if max_precision <= MAX_METAL_DECODE_PRECISION || backend != BackendRequest::Metal {
        return decide_route(backend, fmt);
    }
    let decision = RouteDecision::RejectExplicitMetal {
        reason: ExplicitMetalRejection::PrecisionAboveSixteen,
    };
    telemetry::observe(backend, fmt, decision);
    decision
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

/// The capability rejection for a component wider than
/// [`MAX_METAL_DECODE_PRECISION`].
#[cfg(target_os = "macos")]
pub(crate) fn precision_rejection() -> Error {
    ExplicitMetalRejection::PrecisionAboveSixteen.error()
}

pub(crate) fn decision_error(decision: RouteDecision) -> Option<Error> {
    match decision {
        RouteDecision::RejectExplicitMetal { reason } => Some(reason.error()),
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
