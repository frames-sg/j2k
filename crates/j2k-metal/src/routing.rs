// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(any(target_os = "macos", test))]
use j2k_core::CompressedTransferSyntax;
use j2k_core::{BackendRequest, PixelFormat};
#[cfg(target_os = "macos")]
use j2k_metal_support::metal_kernel_route;
use j2k_metal_support::{
    cpu_host_route, reject_explicit_metal_route, reject_unsupported_backend_route,
    MetalRouteProfileLabels,
};

use crate::Error;

pub(crate) const AUTO_DECODE_CPU_FALLBACK_REASON: &str =
    "J2K Metal Auto decode stays on CPU until decode benchmark evidence justifies Metal routing";

// Minimum qualified cells from verified Auto-routing artifact
// 162a47f7a96b2be88abebc100aab672513af04895532863fa1a293660546f879.
#[cfg(any(target_os = "macos", test))]
const AUTO_REPEATED_DECODE_MIN_COUNT: usize = 16;
#[cfg(any(target_os = "macos", test))]
const AUTO_REPEATED_GRAY8_MIN_PIXELS: u64 = 2_960_793;
#[cfg(any(target_os = "macos", test))]
const AUTO_REPEATED_RGB8_LOSSY_MIN_PIXELS: u64 = 307_200;
#[cfg(any(target_os = "macos", test))]
const AUTO_REPEATED_RGB8_LOSSLESS_MIN_PIXELS: u64 = 5_038_848;

#[cfg(any(target_os = "macos", test))]
pub(crate) fn auto_repeated_decode_uses_metal(
    dimensions: (u32, u32),
    fmt: PixelFormat,
    count: usize,
    transfer_syntax: CompressedTransferSyntax,
) -> bool {
    if count < AUTO_REPEATED_DECODE_MIN_COUNT {
        return false;
    }
    let pixels = u64::from(dimensions.0) * u64::from(dimensions.1);
    match (fmt, transfer_syntax) {
        (PixelFormat::Gray8, CompressedTransferSyntax::Jpeg2000Lossy) => {
            pixels >= AUTO_REPEATED_GRAY8_MIN_PIXELS
        }
        (PixelFormat::Rgb8, CompressedTransferSyntax::Jpeg2000Lossy) => {
            pixels >= AUTO_REPEATED_RGB8_LOSSY_MIN_PIXELS
        }
        (PixelFormat::Rgb8, CompressedTransferSyntax::Jpeg2000Lossless) => {
            pixels >= AUTO_REPEATED_RGB8_LOSSLESS_MIN_PIXELS
        }
        _ => false,
    }
}

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
        let labels = j2k_route_decision_profile(decision);
        match route_profile_fields(backend, fmt, labels) {
            Ok(fields) => j2k_profile::emit_gpu_route_fields("j2k", "metal", &fields),
            Err(error) => {
                j2k_profile::emit_profile_error("metal_gpu_route_fields", &error);
            }
        }
    }
    decision
}

fn route_profile_fields(
    backend: BackendRequest,
    fmt: PixelFormat,
    labels: MetalRouteProfileLabels,
) -> j2k_profile::ProfileResult<[j2k_profile::ProfileField; 5]> {
    Ok([
        j2k_profile::ProfileField::label("request", format_args!("{backend:?}"))?,
        j2k_profile::ProfileField::label("fmt", format_args!("{fmt:?}"))?,
        j2k_profile::ProfileField::label("op", "full")?,
        j2k_profile::ProfileField::label("decision", labels.decision)?,
        j2k_profile::ProfileField::label("reason", labels.reason)?,
    ])
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
    fn auto_repeated_decode_thresholds_match_verified_external_cells() {
        assert!(!auto_repeated_decode_uses_metal(
            (512, 512),
            PixelFormat::Gray8,
            16,
            CompressedTransferSyntax::Jpeg2000Lossy,
        ));
        assert!(auto_repeated_decode_uses_metal(
            (3323, 891),
            PixelFormat::Gray8,
            16,
            CompressedTransferSyntax::Jpeg2000Lossy,
        ));
        assert!(!auto_repeated_decode_uses_metal(
            (3323, 891),
            PixelFormat::Gray8,
            16,
            CompressedTransferSyntax::Jpeg2000Lossless,
        ));
        assert!(!auto_repeated_decode_uses_metal(
            (3323, 891),
            PixelFormat::Gray16,
            16,
            CompressedTransferSyntax::Jpeg2000Lossy,
        ));

        assert!(!auto_repeated_decode_uses_metal(
            (256, 149),
            PixelFormat::Rgb8,
            16,
            CompressedTransferSyntax::Jpeg2000Lossy,
        ));
        assert!(auto_repeated_decode_uses_metal(
            (640, 480),
            PixelFormat::Rgb8,
            16,
            CompressedTransferSyntax::Jpeg2000Lossy,
        ));
        assert!(!auto_repeated_decode_uses_metal(
            (640, 480),
            PixelFormat::Rgb8,
            16,
            CompressedTransferSyntax::Jpeg2000Lossless,
        ));
        assert!(auto_repeated_decode_uses_metal(
            (2592, 1944),
            PixelFormat::Rgb8,
            16,
            CompressedTransferSyntax::Jpeg2000Lossless,
        ));
        assert!(!auto_repeated_decode_uses_metal(
            (640, 480),
            PixelFormat::Rgba8,
            16,
            CompressedTransferSyntax::Jpeg2000Lossy,
        ));
        assert!(!auto_repeated_decode_uses_metal(
            (2592, 1944),
            PixelFormat::Rgb8,
            15,
            CompressedTransferSyntax::Jpeg2000Lossy,
        ));
        assert!(!auto_repeated_decode_uses_metal(
            (2592, 1944),
            PixelFormat::Rgb8,
            16,
            CompressedTransferSyntax::HtJpeg2000Lossless,
        ));
    }

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
