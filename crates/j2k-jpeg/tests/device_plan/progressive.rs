// SPDX-License-Identifier: MIT OR Apache-2.0

use super::support::{
    progressive_12bit_grayscale_8x8_jpeg, progressive_12bit_rgb_8x8_jpeg,
    progressive_12bit_ycbcr_420_32x32_jpeg, progressive_12bit_ycbcr_422_32x8_jpeg,
    progressive_12bit_ycbcr_8x8_jpeg, progressive_8x8_jpeg, ColorSpace, Downscale,
    JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp, PixelFormat, Rect, SofKind,
};

#[test]
fn capability_report_marks_progressive_roi_and_scaled_cpu_eligible() {
    let input = progressive_8x8_jpeg();
    let full = JpegCapabilityReport::inspect(
        &input,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Rgb8,
        },
    )
    .expect("progressive full capability report");
    let roi_scaled = JpegCapabilityReport::inspect(
        &input,
        JpegCapabilityRequest {
            op: JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 0,
                    y: 0,
                    w: 4,
                    h: 4,
                },
                scale: Downscale::Half,
            },
            fmt: PixelFormat::Rgb8,
        },
    )
    .expect("progressive region-scaled capability report");

    assert_eq!(full.info.sof_kind, SofKind::Progressive8);
    assert!(full.cpu.eligible);
    assert!(roi_scaled.cpu.eligible);
    assert!(!roi_scaled.owned_cuda.eligible);
    assert!(!roi_scaled.metal_fast.eligible);
}

#[test]
fn capability_report_marks_progressive12_gray16_and_rgb16_cpu_eligible() {
    let input = progressive_12bit_grayscale_8x8_jpeg();
    for fmt in [PixelFormat::Gray16, PixelFormat::Rgb16] {
        for op in [
            JpegDecodeOp::Full,
            JpegDecodeOp::Region(Rect {
                x: 2,
                y: 1,
                w: 3,
                h: 4,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
                    w: 6,
                    h: 6,
                },
                scale: Downscale::Half,
            },
        ] {
            let report = JpegCapabilityReport::inspect(&input, JpegCapabilityRequest { op, fmt })
                .expect("capability report should parse 12-bit SOF2 grayscale metadata");

            assert_eq!(report.info.sof_kind, SofKind::Progressive12);
            assert_eq!(report.info.bit_depth, 12);
            assert_eq!(report.info.dimensions, (8, 8));
            assert!(report.cpu.eligible, "fmt {fmt:?} op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_progressive12_rgba16_cpu_eligible() {
    for (name, input, expected_color, expected_dimensions, expected_sampling) in [
        (
            "grayscale",
            progressive_12bit_grayscale_8x8_jpeg(),
            ColorSpace::Grayscale,
            (8, 8),
            (1, 1),
        ),
        (
            "APP14 RGB 4:4:4",
            progressive_12bit_rgb_8x8_jpeg(),
            ColorSpace::Rgb,
            (8, 8),
            (1, 1),
        ),
        (
            "YCbCr 4:4:4",
            progressive_12bit_ycbcr_8x8_jpeg(),
            ColorSpace::YCbCr,
            (8, 8),
            (1, 1),
        ),
        (
            "YCbCr 4:2:2",
            progressive_12bit_ycbcr_422_32x8_jpeg(),
            ColorSpace::YCbCr,
            (32, 8),
            (2, 1),
        ),
        (
            "YCbCr 4:2:0",
            progressive_12bit_ycbcr_420_32x32_jpeg(),
            ColorSpace::YCbCr,
            (32, 32),
            (2, 2),
        ),
    ] {
        for op in [
            JpegDecodeOp::Full,
            JpegDecodeOp::Region(Rect {
                x: 1,
                y: 1,
                w: expected_dimensions.0 / 2,
                h: expected_dimensions.1 / 2,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
                    w: expected_dimensions.0 / 2,
                    h: expected_dimensions.1 / 2,
                },
                scale: Downscale::Half,
            },
        ] {
            let report = JpegCapabilityReport::inspect(
                &input,
                JpegCapabilityRequest {
                    op,
                    fmt: PixelFormat::Rgba16,
                },
            )
            .unwrap_or_else(|err| {
                panic!("capability report should parse 12-bit progressive {name} metadata: {err}")
            });

            assert_eq!(report.info.sof_kind, SofKind::Progressive12, "{name}");
            assert_eq!(report.info.bit_depth, 12, "{name}");
            assert_eq!(report.info.dimensions, expected_dimensions, "{name}");
            assert_eq!(report.info.color_space, expected_color, "{name}");
            assert_eq!(report.info.sampling.max_h, expected_sampling.0, "{name}");
            assert_eq!(report.info.sampling.max_v, expected_sampling.1, "{name}");
            assert!(report.cpu.eligible, "{name} op {op:?}");
            assert!(!report.owned_cuda.eligible, "{name}");
            assert!(!report.metal_fast.eligible, "{name}");
        }
    }
}

#[test]
fn capability_report_marks_progressive12_app14_rgb_rgb16_cpu_eligible() {
    let input = progressive_12bit_rgb_8x8_jpeg();
    for op in [
        JpegDecodeOp::Full,
        JpegDecodeOp::Region(Rect {
            x: 2,
            y: 1,
            w: 3,
            h: 4,
        }),
        JpegDecodeOp::Scaled(Downscale::Half),
        JpegDecodeOp::RegionScaled {
            roi: Rect {
                x: 1,
                y: 1,
                w: 6,
                h: 6,
            },
            scale: Downscale::Half,
        },
    ] {
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op,
                fmt: PixelFormat::Rgb16,
            },
        )
        .expect("capability report should parse 12-bit SOF2 APP14 RGB metadata");

        assert_eq!(report.info.sof_kind, SofKind::Progressive12);
        assert_eq!(report.info.bit_depth, 12);
        assert_eq!(report.info.color_space, ColorSpace::Rgb);
        assert!(report.cpu.eligible, "op {op:?}");
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}

#[test]
fn capability_report_marks_progressive12_ycbcr444_rgb16_cpu_eligible() {
    let input = progressive_12bit_ycbcr_8x8_jpeg();
    for op in [
        JpegDecodeOp::Full,
        JpegDecodeOp::Region(Rect {
            x: 2,
            y: 1,
            w: 3,
            h: 4,
        }),
        JpegDecodeOp::Scaled(Downscale::Half),
        JpegDecodeOp::RegionScaled {
            roi: Rect {
                x: 1,
                y: 1,
                w: 6,
                h: 6,
            },
            scale: Downscale::Half,
        },
    ] {
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op,
                fmt: PixelFormat::Rgb16,
            },
        )
        .expect("capability report should parse 12-bit SOF2 YCbCr metadata");

        assert_eq!(report.info.sof_kind, SofKind::Progressive12);
        assert_eq!(report.info.bit_depth, 12);
        assert_eq!(report.info.color_space, ColorSpace::YCbCr);
        assert!(report.cpu.eligible, "op {op:?}");
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}

#[test]
fn capability_report_marks_progressive12_ycbcr422_rgb16_cpu_eligible() {
    let input = progressive_12bit_ycbcr_422_32x8_jpeg();
    for op in [
        JpegDecodeOp::Full,
        JpegDecodeOp::Region(Rect {
            x: 13,
            y: 0,
            w: 8,
            h: 4,
        }),
        JpegDecodeOp::Scaled(Downscale::Half),
        JpegDecodeOp::RegionScaled {
            roi: Rect {
                x: 13,
                y: 0,
                w: 8,
                h: 4,
            },
            scale: Downscale::Half,
        },
    ] {
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op,
                fmt: PixelFormat::Rgb16,
            },
        )
        .expect("capability report should parse 12-bit SOF2 YCbCr 4:2:2 metadata");

        assert_eq!(report.info.sof_kind, SofKind::Progressive12);
        assert_eq!(report.info.bit_depth, 12);
        assert_eq!(report.info.dimensions, (32, 8));
        assert_eq!(report.info.color_space, ColorSpace::YCbCr);
        assert_eq!(report.info.sampling.max_h, 2);
        assert_eq!(report.info.sampling.max_v, 1);
        assert!(report.cpu.eligible, "op {op:?}");
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}

#[test]
fn capability_report_marks_progressive12_ycbcr420_rgb16_cpu_eligible() {
    let input = progressive_12bit_ycbcr_420_32x32_jpeg();
    for op in [
        JpegDecodeOp::Full,
        JpegDecodeOp::Region(Rect {
            x: 13,
            y: 14,
            w: 10,
            h: 10,
        }),
        JpegDecodeOp::Scaled(Downscale::Half),
        JpegDecodeOp::RegionScaled {
            roi: Rect {
                x: 13,
                y: 14,
                w: 10,
                h: 10,
            },
            scale: Downscale::Half,
        },
    ] {
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op,
                fmt: PixelFormat::Rgb16,
            },
        )
        .expect("capability report should parse 12-bit SOF2 YCbCr 4:2:0 metadata");

        assert_eq!(report.info.sof_kind, SofKind::Progressive12);
        assert_eq!(report.info.bit_depth, 12);
        assert_eq!(report.info.dimensions, (32, 32));
        assert_eq!(report.info.color_space, ColorSpace::YCbCr);
        assert_eq!(report.info.sampling.max_h, 2);
        assert_eq!(report.info.sampling.max_v, 2);
        assert!(report.cpu.eligible, "op {op:?}");
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}
