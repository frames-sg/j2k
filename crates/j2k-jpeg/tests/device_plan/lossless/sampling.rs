// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::support::{
    grayscale_sof_jpeg, lossless_3x3_region_scaled_op, lossless_predictor_grayscale_3x3_jpeg,
    lossless_rgb_16bit_420_4x4_jpeg, lossless_rgb_16bit_420_restart_4x4_jpeg,
    lossless_rgb_16bit_422_4x2_jpeg, lossless_rgb_16bit_422_restart_4x2_jpeg,
    lossless_rgb_8bit_420_4x4_jpeg, lossless_rgb_8bit_420_restart_4x4_jpeg,
    lossless_rgb_8bit_422_4x2_jpeg, lossless_rgb_8bit_422_restart_4x2_jpeg,
    lossless_ycbcr_16bit_420_4x4_jpeg, lossless_ycbcr_16bit_420_restart_4x4_jpeg,
    lossless_ycbcr_16bit_422_4x2_jpeg, lossless_ycbcr_16bit_422_restart_4x2_jpeg,
    lossless_ycbcr_8bit_420_4x4_jpeg, lossless_ycbcr_8bit_420_restart_4x4_jpeg,
    lossless_ycbcr_8bit_422_4x2_jpeg, lossless_ycbcr_8bit_422_restart_4x2_jpeg,
    progressive_12_bit_jpeg, ColorSpace, Downscale, JpegCapabilityReport, JpegCapabilityRequest,
    JpegDecodeOp, PixelFormat, Rect, SofKind,
};

#[test]
fn capability_report_inspects_12_bit_and_lossless_sof_without_building_decoder() {
    for (input, expected_sof, expected_bits, expected_dimensions, expected_reason) in [
        (
            grayscale_sof_jpeg(0xc1, 12),
            SofKind::Extended12,
            12,
            (8, 8),
            "12-bit",
        ),
        (
            progressive_12_bit_jpeg(),
            SofKind::Progressive12,
            12,
            (8, 8),
            "12-bit",
        ),
        (
            lossless_predictor_grayscale_3x3_jpeg(1),
            SofKind::Lossless,
            8,
            (3, 3),
            "lossless SOF3",
        ),
    ] {
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op: JpegDecodeOp::Full,
                fmt: PixelFormat::Rgba8,
            },
        )
        .expect("capability report should parse unsupported SOF metadata");

        assert_eq!(report.info.sof_kind, expected_sof);
        assert_eq!(report.info.bit_depth, expected_bits);
        assert_eq!(report.info.dimensions, expected_dimensions);
        assert!(!report.cpu.eligible);
        assert!(report
            .cpu
            .reason
            .expect("CPU rejection reason")
            .contains(expected_reason));
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the capability report test verifies one coherent sampled-lossless eligibility contract"
)]
fn capability_report_marks_lossless_8bit_sampled_color_cpu_eligible() {
    let requests = [
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Rgb8,
        },
        JpegCapabilityRequest {
            op: JpegDecodeOp::Region(Rect {
                x: 1,
                y: 0,
                w: 2,
                h: 2,
            }),
            fmt: PixelFormat::Rgb8,
        },
        JpegCapabilityRequest {
            op: JpegDecodeOp::Scaled(Downscale::Half),
            fmt: PixelFormat::Rgba8,
        },
        JpegCapabilityRequest {
            op: JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 0,
                    w: 2,
                    h: 2,
                },
                scale: Downscale::Half,
            },
            fmt: PixelFormat::Rgba8,
        },
    ];

    for (input, color_space, dimensions, sampling, label) in [
        (
            lossless_rgb_8bit_422_4x2_jpeg(4),
            ColorSpace::Rgb,
            (4, 2),
            [(2, 1), (1, 1), (1, 1)],
            "4:2:2 APP14 RGB",
        ),
        (
            lossless_rgb_8bit_422_restart_4x2_jpeg(4),
            ColorSpace::Rgb,
            (4, 2),
            [(2, 1), (1, 1), (1, 1)],
            "4:2:2 APP14 RGB restart",
        ),
        (
            lossless_ycbcr_8bit_422_4x2_jpeg(4),
            ColorSpace::YCbCr,
            (4, 2),
            [(2, 1), (1, 1), (1, 1)],
            "4:2:2 YCbCr",
        ),
        (
            lossless_ycbcr_8bit_422_restart_4x2_jpeg(4),
            ColorSpace::YCbCr,
            (4, 2),
            [(2, 1), (1, 1), (1, 1)],
            "4:2:2 YCbCr restart",
        ),
        (
            lossless_rgb_8bit_420_4x4_jpeg(4),
            ColorSpace::Rgb,
            (4, 4),
            [(2, 2), (1, 1), (1, 1)],
            "4:2:0 APP14 RGB",
        ),
        (
            lossless_rgb_8bit_420_restart_4x4_jpeg(4),
            ColorSpace::Rgb,
            (4, 4),
            [(2, 2), (1, 1), (1, 1)],
            "4:2:0 APP14 RGB restart",
        ),
        (
            lossless_ycbcr_8bit_420_4x4_jpeg(4),
            ColorSpace::YCbCr,
            (4, 4),
            [(2, 2), (1, 1), (1, 1)],
            "4:2:0 YCbCr",
        ),
        (
            lossless_ycbcr_8bit_420_restart_4x4_jpeg(4),
            ColorSpace::YCbCr,
            (4, 4),
            [(2, 2), (1, 1), (1, 1)],
            "4:2:0 YCbCr restart",
        ),
    ] {
        for request in requests {
            let report = JpegCapabilityReport::inspect(&input, request).unwrap_or_else(|err| {
                panic!(
                    "lossless SOF3 8-bit sampled {label} should report CPU-eligible capability metadata, got {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless, "{label}");
            assert_eq!(report.info.bit_depth, 8, "{label}");
            assert_eq!(report.info.dimensions, dimensions, "{label}");
            assert!(
                matches!(report.info.restart_interval, None | Some(2)),
                "{label}"
            );
            assert_eq!(report.info.color_space, color_space, "{label}");
            assert_eq!(report.info.sampling.components(), &sampling, "{label}");
            assert!(report.cpu.eligible, "{label} {request:?}");
            assert_eq!(report.cpu.reason, None, "{label} {request:?}");
            assert!(!report.owned_cuda.eligible, "{label}");
            assert!(!report.metal_fast.eligible, "{label}");
        }
    }
}

#[test]
fn capability_report_marks_lossless_16bit_422_color_cpu_eligible() {
    let requests = [
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Rgb16,
        },
        JpegCapabilityRequest {
            op: JpegDecodeOp::Region(Rect {
                x: 1,
                y: 0,
                w: 2,
                h: 2,
            }),
            fmt: PixelFormat::Rgb16,
        },
        JpegCapabilityRequest {
            op: JpegDecodeOp::Scaled(Downscale::Half),
            fmt: PixelFormat::Rgba16,
        },
        JpegCapabilityRequest {
            op: JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 0,
                    w: 2,
                    h: 2,
                },
                scale: Downscale::Half,
            },
            fmt: PixelFormat::Rgba16,
        },
    ];

    for (input, color_space, label) in [
        (
            lossless_rgb_16bit_422_4x2_jpeg(4),
            ColorSpace::Rgb,
            "APP14 RGB",
        ),
        (
            lossless_rgb_16bit_422_restart_4x2_jpeg(4),
            ColorSpace::Rgb,
            "APP14 RGB restart",
        ),
        (
            lossless_ycbcr_16bit_422_4x2_jpeg(4),
            ColorSpace::YCbCr,
            "YCbCr",
        ),
        (
            lossless_ycbcr_16bit_422_restart_4x2_jpeg(4),
            ColorSpace::YCbCr,
            "YCbCr restart",
        ),
    ] {
        for request in requests {
            let report = JpegCapabilityReport::inspect(&input, request).unwrap_or_else(|err| {
                panic!(
                    "lossless SOF3 16-bit 4:2:2 {label} should report CPU-eligible capability metadata, got {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless, "{label}");
            assert_eq!(report.info.bit_depth, 16, "{label}");
            assert_eq!(report.info.dimensions, (4, 2), "{label}");
            assert!(
                matches!(report.info.restart_interval, None | Some(2)),
                "{label}"
            );
            assert_eq!(report.info.color_space, color_space, "{label}");
            assert_eq!(report.info.sampling.max_h, 2, "{label}");
            assert_eq!(report.info.sampling.max_v, 1, "{label}");
            assert_eq!(
                report.info.sampling.components(),
                &[(2, 1), (1, 1), (1, 1)],
                "{label}"
            );
            assert!(report.cpu.eligible, "{label} {request:?}");
            assert_eq!(report.cpu.reason, None, "{label} {request:?}");
            assert!(!report.owned_cuda.eligible, "{label}");
            assert!(!report.metal_fast.eligible, "{label}");
        }
    }
}

#[test]
fn capability_report_marks_lossless_16bit_420_color_cpu_eligible() {
    let requests = [
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Rgb16,
        },
        JpegCapabilityRequest {
            op: JpegDecodeOp::Region(Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            }),
            fmt: PixelFormat::Rgb16,
        },
        JpegCapabilityRequest {
            op: JpegDecodeOp::Scaled(Downscale::Half),
            fmt: PixelFormat::Rgba16,
        },
        JpegCapabilityRequest {
            op: lossless_3x3_region_scaled_op(),
            fmt: PixelFormat::Rgba16,
        },
    ];

    for (input, color_space, label) in [
        (
            lossless_rgb_16bit_420_4x4_jpeg(4),
            ColorSpace::Rgb,
            "APP14 RGB",
        ),
        (
            lossless_rgb_16bit_420_restart_4x4_jpeg(4),
            ColorSpace::Rgb,
            "APP14 RGB restart",
        ),
        (
            lossless_ycbcr_16bit_420_4x4_jpeg(4),
            ColorSpace::YCbCr,
            "YCbCr",
        ),
        (
            lossless_ycbcr_16bit_420_restart_4x4_jpeg(4),
            ColorSpace::YCbCr,
            "YCbCr restart",
        ),
    ] {
        for request in requests {
            let report = JpegCapabilityReport::inspect(&input, request).unwrap_or_else(|err| {
                panic!(
                    "lossless SOF3 16-bit 4:2:0 {label} should report CPU-eligible capability metadata, got {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless, "{label}");
            assert_eq!(report.info.bit_depth, 16, "{label}");
            assert_eq!(report.info.dimensions, (4, 4), "{label}");
            assert!(
                matches!(report.info.restart_interval, None | Some(2)),
                "{label}"
            );
            assert_eq!(report.info.color_space, color_space, "{label}");
            assert_eq!(report.info.sampling.max_h, 2, "{label}");
            assert_eq!(report.info.sampling.max_v, 2, "{label}");
            assert_eq!(
                report.info.sampling.components(),
                &[(2, 2), (1, 1), (1, 1)],
                "{label}"
            );
            assert!(report.cpu.eligible, "{label} {request:?}");
            assert_eq!(report.cpu.reason, None, "{label} {request:?}");
            assert!(!report.owned_cuda.eligible, "{label}");
            assert!(!report.metal_fast.eligible, "{label}");
        }
    }
}
