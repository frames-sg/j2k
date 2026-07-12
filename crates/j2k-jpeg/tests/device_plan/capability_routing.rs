// SPDX-License-Identifier: MIT OR Apache-2.0

use super::support::{
    ColorSpace, Downscale, JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp, PixelFormat,
    Rect, BASELINE_420, BASELINE_422, BASELINE_444,
};

#[test]
fn capability_report_exposes_metadata_and_fast_backend_eligibility() {
    let report = JpegCapabilityReport::inspect(
        BASELINE_420,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Rgb8,
        },
    )
    .expect("capability report");

    assert_eq!(report.info.dimensions, (16, 16));
    assert_eq!(report.info.color_space, ColorSpace::YCbCr);
    assert_eq!(report.info.sof_kind, j2k_jpeg::SofKind::Baseline8);
    assert!(report.cpu.eligible);
    assert!(report.owned_cuda.eligible);
    assert!(report.metal_fast.eligible);
    assert!(report.device.matches_fast_420);
    assert!(!report.device.matches_fast_422);
    assert!(!report.device.matches_fast_444);
}

#[test]
fn capability_report_marks_owned_cuda_eligible_for_fast_422_and_444_rgb8() {
    for (input, expected_dimensions, expected_shape) in [
        (BASELINE_422, (16, 8), "4:2:2"),
        (BASELINE_444, (8, 8), "4:4:4"),
    ] {
        let report = JpegCapabilityReport::inspect(
            input,
            JpegCapabilityRequest {
                op: JpegDecodeOp::Full,
                fmt: PixelFormat::Rgb8,
            },
        )
        .expect("capability report");

        assert_eq!(report.info.dimensions, expected_dimensions);
        assert!(
            report.owned_cuda.eligible,
            "owned CUDA must be eligible for full-tile RGB8 fast {expected_shape}"
        );
        assert!(report.metal_fast.eligible);
    }
}

#[test]
fn capability_report_rejects_owned_cuda_for_scaled_or_non_rgb8_requests() {
    let scaled = JpegCapabilityReport::inspect(
        BASELINE_420,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Scaled(Downscale::Quarter),
            fmt: PixelFormat::Rgb8,
        },
    )
    .expect("scaled capability report");
    let gray = JpegCapabilityReport::inspect(
        BASELINE_420,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Gray8,
        },
    )
    .expect("gray capability report");

    assert!(!scaled.owned_cuda.eligible);
    assert!(scaled
        .owned_cuda
        .reason
        .expect("scaled cuda rejection")
        .contains("full-tile RGB8"));
    assert!(!gray.owned_cuda.eligible);
    assert!(gray
        .owned_cuda
        .reason
        .expect("gray cuda rejection")
        .contains("full-tile RGB8"));
}

#[test]
fn capability_report_keeps_roi_shape_visible_for_statumen_routing() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let report = JpegCapabilityReport::inspect(
        BASELINE_420,
        JpegCapabilityRequest {
            op: JpegDecodeOp::RegionScaled {
                roi,
                scale: Downscale::Quarter,
            },
            fmt: PixelFormat::Rgb8,
        },
    )
    .expect("roi capability report");

    assert_eq!(
        report.request.op,
        JpegDecodeOp::RegionScaled {
            roi,
            scale: Downscale::Quarter,
        }
    );
    assert!(report.cpu.eligible);
    assert!(!report.owned_cuda.eligible);
}

#[test]
fn capability_report_exposes_resident_metal_rgb8_batch_output_eligibility() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let report = JpegCapabilityReport::inspect(
        BASELINE_420,
        JpegCapabilityRequest {
            op: JpegDecodeOp::RegionScaled {
                roi,
                scale: Downscale::Quarter,
            },
            fmt: PixelFormat::Rgb8,
        },
    )
    .expect("region-scaled capability report");

    assert!(report.metal_fast.eligible);
    assert!(report.metal_resident_rgb8_batch_output().eligible);
}

#[test]
fn capability_report_distinguishes_metal_fast_shape_from_reusable_rgb8_output() {
    let gray = JpegCapabilityReport::inspect(
        BASELINE_420,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Gray8,
        },
    )
    .expect("gray capability report");
    let region = JpegCapabilityReport::inspect(
        BASELINE_420,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Region(Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            }),
            fmt: PixelFormat::Rgb8,
        },
    )
    .expect("region capability report");

    assert!(gray.metal_fast.eligible);
    assert!(!gray.metal_resident_rgb8_batch_output().eligible);
    assert!(gray
        .metal_resident_rgb8_batch_output()
        .reason
        .expect("gray rejection")
        .contains("RGB8"));

    assert!(region.metal_fast.eligible);
    assert!(!region.metal_resident_rgb8_batch_output().eligible);
    assert!(region
        .metal_resident_rgb8_batch_output()
        .reason
        .expect("region rejection")
        .contains("full, scaled, or region-scaled"));
}
