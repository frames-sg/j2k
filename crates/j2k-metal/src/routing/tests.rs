use j2k_core::{
    BackendRequest, CompressedPayloadKind, CompressedTransferSyntax, Downscale, PixelFormat,
};

use crate::Error;

use super::{
    auto_repeated_decode_uses_metal, auto_scaled_decode_uses_metal, decide_route, decision_error,
    rejection::ExplicitMetalRejection, RouteDecision,
};

#[test]
fn auto_repeated_decode_thresholds_match_verified_external_cells() {
    use CompressedPayloadKind::{Jpeg2000Codestream as Raw, JphFile as Jph};
    use CompressedTransferSyntax::{
        HtJpeg2000Lossless as HtLossless, HtJpeg2000Lossy as HtLossy,
        Jpeg2000Lossless as ClassicLossless, Jpeg2000Lossy as ClassicLossy,
    };

    for (dimensions, format, batch, transfer_syntax, payload_kind, expected) in [
        ((512, 512), PixelFormat::Gray8, 16, ClassicLossy, Raw, false),
        ((3323, 891), PixelFormat::Gray8, 16, ClassicLossy, Raw, true),
        (
            (3323, 891),
            PixelFormat::Gray8,
            16,
            ClassicLossless,
            Raw,
            false,
        ),
        (
            (3323, 891),
            PixelFormat::Gray16,
            16,
            ClassicLossy,
            Raw,
            false,
        ),
        ((256, 149), PixelFormat::Rgb8, 16, ClassicLossy, Raw, false),
        ((640, 480), PixelFormat::Rgb8, 16, ClassicLossy, Raw, true),
        (
            (640, 480),
            PixelFormat::Rgb8,
            16,
            ClassicLossless,
            Raw,
            false,
        ),
        (
            (2592, 1944),
            PixelFormat::Rgb8,
            16,
            ClassicLossless,
            Raw,
            true,
        ),
        ((640, 480), PixelFormat::Rgba8, 16, ClassicLossy, Raw, false),
        (
            (2592, 1944),
            PixelFormat::Rgb8,
            15,
            ClassicLossy,
            Raw,
            false,
        ),
        ((767, 512), PixelFormat::Rgb8, 16, HtLossless, Jph, false),
        ((768, 512), PixelFormat::Rgb8, 16, HtLossless, Jph, true),
        ((639, 480), PixelFormat::Rgb8, 16, HtLossy, Raw, false),
        ((640, 480), PixelFormat::Rgb8, 16, HtLossy, Raw, true),
        ((768, 512), PixelFormat::Gray8, 16, HtLossless, Jph, false),
    ] {
        assert_eq!(
                auto_repeated_decode_uses_metal(
                    dimensions,
                    format,
                    batch,
                    transfer_syntax,
                    payload_kind,
                ),
                expected,
                "unexpected repeated-decode route for {dimensions:?}/{format:?}/{transfer_syntax:?}/{payload_kind:?}",
            );
    }
}

#[test]
fn auto_repeated_decode_requires_the_measured_payload_kind() {
    use CompressedPayloadKind::{Jp2File as Jp2, Jpeg2000Codestream as Raw, JphFile as Jph};
    use CompressedTransferSyntax::{
        HtJpeg2000Lossless as HtLossless, HtJpeg2000Lossy as HtLossy, Jpeg2000Lossy as ClassicLossy,
    };

    assert!(auto_repeated_decode_uses_metal(
        (640, 480),
        PixelFormat::Rgb8,
        16,
        ClassicLossy,
        Raw,
    ));
    assert!(!auto_repeated_decode_uses_metal(
        (640, 480),
        PixelFormat::Rgb8,
        16,
        ClassicLossy,
        Jp2,
    ));
    assert!(auto_repeated_decode_uses_metal(
        (640, 480),
        PixelFormat::Rgb8,
        16,
        HtLossy,
        Raw,
    ));
    assert!(!auto_repeated_decode_uses_metal(
        (640, 480),
        PixelFormat::Rgb8,
        16,
        HtLossy,
        Jph,
    ));
    assert!(auto_repeated_decode_uses_metal(
        (768, 512),
        PixelFormat::Rgb8,
        16,
        HtLossless,
        Jph,
    ));
    assert!(!auto_repeated_decode_uses_metal(
        (768, 512),
        PixelFormat::Rgb8,
        16,
        HtLossless,
        Raw,
    ));
}

#[test]
fn auto_scaled_decode_threshold_matches_only_the_verified_ht_cell() {
    use CompressedPayloadKind::{Jpeg2000Codestream as Raw, JphFile as Jph};
    use CompressedTransferSyntax::{HtJpeg2000Lossless as Lossless, HtJpeg2000Lossy as Lossy};

    assert!(auto_scaled_decode_uses_metal(
        (320, 240),
        3,
        PixelFormat::Rgb8,
        Lossy,
        Raw,
        Downscale::Half,
    ));
    for (dimensions, components, fmt, transfer_syntax, payload_kind, scale) in [
        (
            (319, 240),
            3,
            PixelFormat::Rgb8,
            Lossy,
            Raw,
            Downscale::Half,
        ),
        (
            (320, 240),
            1,
            PixelFormat::Rgb8,
            Lossy,
            Raw,
            Downscale::Half,
        ),
        (
            (320, 240),
            3,
            PixelFormat::Gray8,
            Lossy,
            Raw,
            Downscale::Half,
        ),
        (
            (320, 240),
            3,
            PixelFormat::Rgb8,
            Lossless,
            Raw,
            Downscale::Half,
        ),
        (
            (320, 240),
            3,
            PixelFormat::Rgb8,
            Lossy,
            Jph,
            Downscale::Half,
        ),
        (
            (320, 240),
            3,
            PixelFormat::Rgb8,
            Lossy,
            Raw,
            Downscale::Quarter,
        ),
    ] {
        assert!(!auto_scaled_decode_uses_metal(
            dimensions,
            components,
            fmt,
            transfer_syntax,
            payload_kind,
            scale,
        ));
    }
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
