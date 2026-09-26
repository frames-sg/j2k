use j2k_core::{
    BackendRequest, CompressedPayloadKind, CompressedTransferSyntax, Downscale, PixelFormat,
};

use crate::Error;

use super::{
    auto_full_decode_uses_metal, auto_repeated_decode_uses_metal, auto_scaled_decode_uses_metal,
    decide_route, decision_error, rejection::ExplicitMetalRejection, RouteDecision,
};

#[test]
fn auto_repeated_decode_thresholds_match_verified_cells() {
    use CompressedPayloadKind::{Jpeg2000Codestream as Raw, JphFile as Jph};
    use CompressedTransferSyntax::{
        HtJpeg2000Lossless as HtLossless, HtJpeg2000Lossy as HtLossy,
        Jpeg2000Lossless as ClassicLossless, Jpeg2000Lossy as ClassicLossy,
    };
    use PixelFormat::{Gray16, Gray8, Rgb8, Rgba8};

    for (dimensions, format, batch, transfer_syntax, payload_kind, expected) in [
        ((255, 256), Rgb8, 16, HtLossless, Raw, false),
        ((256, 256), Rgb8, 16, HtLossless, Raw, true),
        ((639, 480), Rgb8, 16, HtLossless, Jph, false),
        ((640, 480), Rgb8, 16, HtLossless, Jph, true),
        ((640, 479), Gray8, 16, HtLossless, Raw, false),
        ((640, 480), Gray8, 16, HtLossless, Raw, true),
        ((639, 480), Rgb8, 16, ClassicLossless, Raw, false),
        ((640, 480), Rgb8, 16, ClassicLossless, Raw, true),
        ((2047, 2048), Gray8, 16, ClassicLossless, Raw, false),
        ((2048, 2048), Gray8, 16, ClassicLossless, Raw, true),
        ((640, 480), Rgb8, 15, ClassicLossless, Raw, false),
        ((640, 480), Rgba8, 16, ClassicLossless, Raw, false),
        ((2048, 2048), Gray16, 16, ClassicLossless, Raw, false),
        ((255, 256), Rgb8, 16, HtLossy, Raw, false),
        ((256, 256), Rgb8, 16, HtLossy, Raw, true),
        // Part 1 lossy RGB8 measured slower at 640x480 and faster from 1024x1024.
        ((640, 480), Rgb8, 16, ClassicLossy, Raw, false),
        ((1023, 1024), Rgb8, 16, ClassicLossy, Raw, false),
        ((1024, 1024), Rgb8, 16, ClassicLossy, Raw, true),
        // Lossy Gray8 qualifies only from the measured 2048x2048 cells.
        ((640, 480), Gray8, 16, HtLossy, Raw, false),
        ((2047, 2048), Gray8, 16, HtLossy, Raw, false),
        ((2048, 2048), Gray8, 16, HtLossy, Raw, true),
        ((3323, 891), Gray8, 16, ClassicLossy, Raw, false),
        ((2048, 2048), Gray8, 16, ClassicLossy, Raw, true),
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
        HtJpeg2000Lossless as HtLossless, Jpeg2000Lossless as ClassicLossless,
    };

    for (dimensions, transfer_syntax, payload_kind, expected) in [
        ((640, 480), ClassicLossless, Raw, true),
        ((640, 480), ClassicLossless, Jp2, false),
        ((256, 256), HtLossless, Raw, true),
        ((256, 256), HtLossless, Jph, false),
        ((640, 480), HtLossless, Jph, true),
        ((640, 480), HtLossless, Jp2, false),
    ] {
        assert_eq!(
            auto_repeated_decode_uses_metal(
                dimensions,
                PixelFormat::Rgb8,
                16,
                transfer_syntax,
                payload_kind,
            ),
            expected,
            "{dimensions:?}/{transfer_syntax:?}/{payload_kind:?}"
        );
    }
}

#[test]
fn auto_full_decode_thresholds_match_verified_ht_cells() {
    use CompressedPayloadKind::{Jp2File as Jp2, Jpeg2000Codestream as Raw, JphFile as Jph};
    use CompressedTransferSyntax::{
        HtJpeg2000Lossless as HtLossless, HtJpeg2000Lossy as HtLossy,
        Jpeg2000Lossless as ClassicLossless, Jpeg2000Lossy as ClassicLossy,
    };
    use PixelFormat::{Gray8, Rgb8};

    for (dimensions, components, format, transfer_syntax, payload_kind, expected) in [
        ((639, 480), 3, Rgb8, HtLossless, Raw, false),
        ((640, 480), 3, Rgb8, HtLossless, Raw, true),
        ((640, 480), 3, Rgb8, HtLossless, Jph, true),
        ((640, 480), 3, Rgb8, HtLossless, Jp2, false),
        ((640, 480), 1, Gray8, HtLossless, Raw, true),
        ((640, 479), 1, Gray8, HtLossless, Raw, false),
        // Only the measured source/output pairings qualify.
        ((640, 480), 3, Gray8, HtLossless, Raw, false),
        ((640, 480), 1, Rgb8, HtLossless, Raw, false),
        ((639, 480), 3, Rgb8, HtLossy, Raw, false),
        ((640, 480), 3, Rgb8, HtLossy, Raw, true),
        ((640, 479), 1, Gray8, HtLossy, Raw, false),
        ((640, 480), 1, Gray8, HtLossy, Raw, true),
        // Part 1 full decodes measured slower on Metal, lossless and lossy.
        ((2048, 2048), 3, Rgb8, ClassicLossless, Raw, false),
        ((2048, 2048), 3, Rgb8, ClassicLossy, Raw, false),
    ] {
        assert_eq!(
            auto_full_decode_uses_metal(
                dimensions,
                components,
                format,
                transfer_syntax,
                payload_kind
            ),
            expected,
            "{dimensions:?}/{components}/{format:?}/{transfer_syntax:?}/{payload_kind:?}"
        );
    }
}

#[test]
fn auto_scaled_decode_stays_on_cpu_without_a_verified_cell() {
    use CompressedPayloadKind::Jpeg2000Codestream as Raw;
    use CompressedTransferSyntax::{HtJpeg2000Lossless as Lossless, HtJpeg2000Lossy as Lossy};

    for (dimensions, transfer_syntax) in [
        ((320, 240), Lossy),
        ((1024, 1024), Lossy),
        ((1024, 1024), Lossless),
    ] {
        assert!(!auto_scaled_decode_uses_metal(
            dimensions,
            3,
            PixelFormat::Rgb8,
            transfer_syntax,
            Raw,
            Downscale::Half,
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
