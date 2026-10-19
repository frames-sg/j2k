// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hybrid Metal decode eligibility: explicit Metal accepts exactly what CPU
//! decode accepts, resolving to the fast-packet kernels where they apply and
//! to hybrid decode everywhere else.

use j2k_core::BackendRequest;
use j2k_jpeg::{
    metal_hybrid_eligibility, Decoder, JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp,
    JpegResolvedDecodePath,
};

use super::support::{Downscale, PixelFormat, Rect, BASELINE_420};

fn report(input: &[u8], op: JpegDecodeOp, fmt: PixelFormat) -> JpegCapabilityReport {
    JpegCapabilityReport::inspect(input, JpegCapabilityRequest { op, fmt })
        .expect("capability report")
}

#[test]
fn fast_packet_jpeg_resolves_to_the_metal_fast_path() {
    let report = report(BASELINE_420, JpegDecodeOp::Full, PixelFormat::Rgb8);

    assert!(report.metal_fast.eligible);
    assert!(report.metal_hybrid().eligible);
    assert_eq!(
        report.resolve_path(BackendRequest::Metal),
        JpegResolvedDecodePath::MetalFast
    );
}

#[test]
fn jpeg_classes_outside_the_fast_path_resolve_to_hybrid_metal_decode() {
    let lossless = j2k_test_support::lossless_matrix_cases()
        .into_iter()
        .find(|case| case.is_gray() && case.precision == 16 && case.restart_rows == 0)
        .expect("16-bit grayscale lossless case")
        .jpeg;
    let cases: [(&str, Vec<u8>, PixelFormat); 6] = [
        (
            "grayscale",
            j2k_test_support::baseline_grayscale_jpeg(37, 29),
            PixelFormat::Gray8,
        ),
        (
            "progressive",
            j2k_test_support::progressive_8x8_jpeg(),
            PixelFormat::Rgba8,
        ),
        (
            "cmyk",
            j2k_test_support::cmyk_16x16_420_jpeg(),
            PixelFormat::Rgb8,
        ),
        (
            "12-bit",
            j2k_test_support::extended_12bit_ycbcr_420_32x32_jpeg(),
            PixelFormat::Rgb16,
        ),
        ("lossless", lossless.to_vec(), PixelFormat::Gray16),
        (
            "rgb app14",
            j2k_test_support::rgb_app14_8x8_jpeg(),
            PixelFormat::Rgb8,
        ),
    ];
    let ops = [
        JpegDecodeOp::Full,
        JpegDecodeOp::Scaled(Downscale::Quarter),
        JpegDecodeOp::RegionScaled {
            roi: Rect {
                x: 1,
                y: 2,
                w: 5,
                h: 4,
            },
            scale: Downscale::Half,
        },
    ];
    for (name, input, fmt) in &cases {
        for op in ops {
            let report = report(input, op, *fmt);

            assert!(report.cpu.eligible, "{name} {op:?}");
            assert!(!report.metal_fast.eligible, "{name} {op:?}");
            assert!(report.metal_hybrid().eligible, "{name} {op:?}");
            assert_eq!(
                report.resolve_path(BackendRequest::Metal),
                JpegResolvedDecodePath::MetalHybrid,
                "{name} {op:?}"
            );
            // Auto stays on the CPU until measured workload evidence promotes it.
            assert_eq!(
                report.resolve_path(BackendRequest::Auto),
                JpegResolvedDecodePath::CpuHost,
                "{name} {op:?}"
            );
        }
    }
}

#[test]
fn hybrid_metal_rejects_what_cpu_decode_rejects_with_metal_wording() {
    let cases = [
        (
            j2k_test_support::progressive_8x8_jpeg(),
            PixelFormat::Rgb16,
            "for 8-bit JPEG",
        ),
        (
            j2k_test_support::extended_12bit_rgb_8x8_jpeg(),
            PixelFormat::Rgb8,
            "for 12-bit JPEG",
        ),
        (
            j2k_test_support::extended_12bit_ycbcr_8x8_jpeg(),
            PixelFormat::Gray16,
            "for 12-bit JPEG",
        ),
    ];
    for (input, fmt, wording) in cases {
        let report = report(&input, JpegDecodeOp::Full, fmt);

        assert!(!report.cpu.eligible, "{fmt:?}");
        let hybrid = report.metal_hybrid();
        assert!(!hybrid.eligible, "{fmt:?}");
        let reason = hybrid.reason.expect("rejection reason");
        assert!(
            reason.starts_with("JPEG Metal") && reason.contains(wording),
            "{reason}"
        );
        assert_eq!(
            report.resolve_path(BackendRequest::Metal),
            JpegResolvedDecodePath::Rejected {
                backend: BackendRequest::Metal,
                reason,
            }
        );
    }
}

#[test]
fn hybrid_metal_eligibility_matches_cpu_eligibility_across_the_matrices() {
    let formats = [
        PixelFormat::Gray8,
        PixelFormat::Rgb8,
        PixelFormat::Rgba8,
        PixelFormat::Gray16,
        PixelFormat::Rgb16,
        PixelFormat::Rgba16,
    ];
    let inputs = j2k_test_support::scaled_matrix_cases()
        .into_iter()
        .map(|case| case.jpeg)
        .chain(
            j2k_test_support::lossless_matrix_cases()
                .into_iter()
                .map(|case| case.jpeg),
        );
    for input in inputs {
        let Ok(decoder) = Decoder::new(input) else {
            continue;
        };
        for fmt in formats {
            let request = JpegCapabilityRequest {
                op: JpegDecodeOp::Full,
                fmt,
            };
            let report = JpegCapabilityReport::for_decoder(&decoder, request);
            assert_eq!(
                metal_hybrid_eligibility(decoder.info(), request).eligible,
                report.cpu.eligible,
                "{:?} {fmt:?}",
                decoder.info().sof_kind
            );
        }
    }
}
