// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::support::{
    extended_12bit_grayscale_restart_16x8_jpeg, extended_12bit_rgb_420_32x32_jpeg,
    extended_12bit_rgb_422_32x8_jpeg, extended_12bit_rgb_8x8_jpeg,
    extended_12bit_rgb_restart_16x8_jpeg, extended_12bit_ycbcr_420_32x32_jpeg,
    extended_12bit_ycbcr_420_restart_32x32_jpeg, extended_12bit_ycbcr_422_32x8_jpeg,
    extended_12bit_ycbcr_422_restart_32x8_jpeg, extended_12bit_ycbcr_8x8_jpeg,
    extended_12bit_ycbcr_restart_16x8_jpeg, grayscale_sof_jpeg,
    progressive_12bit_rgb_420_32x32_jpeg, progressive_12bit_rgb_422_32x8_jpeg, ColorSpace,
    Downscale, JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp, PixelFormat, Rect,
    SofKind,
};

#[test]
fn capability_report_marks_extended12_gray16_full_cpu_eligible() {
    let input = grayscale_sof_jpeg(0xc1, 12);
    let report = JpegCapabilityReport::inspect(
        &input,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Gray16,
        },
    )
    .expect("capability report should parse 12-bit SOF1 metadata");

    assert_eq!(report.info.sof_kind, SofKind::Extended12);
    assert_eq!(report.info.bit_depth, 12);
    assert!(report.cpu.eligible);
    assert!(!report.owned_cuda.eligible);
    assert!(!report.metal_fast.eligible);
}

#[test]
fn capability_report_marks_extended12_gray16_region_cpu_eligible() {
    let input = grayscale_sof_jpeg(0xc1, 12);
    let report = JpegCapabilityReport::inspect(
        &input,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Region(Rect {
                x: 2,
                y: 1,
                w: 3,
                h: 4,
            }),
            fmt: PixelFormat::Gray16,
        },
    )
    .expect("capability report should parse 12-bit SOF1 metadata");

    assert_eq!(report.info.sof_kind, SofKind::Extended12);
    assert!(report.cpu.eligible);
    assert!(!report.owned_cuda.eligible);
    assert!(!report.metal_fast.eligible);
}

#[test]
fn capability_report_marks_extended12_rgb16_full_and_region_cpu_eligible() {
    let input = grayscale_sof_jpeg(0xc1, 12);
    for op in [
        JpegDecodeOp::Full,
        JpegDecodeOp::Region(Rect {
            x: 1,
            y: 2,
            w: 4,
            h: 3,
        }),
    ] {
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op,
                fmt: PixelFormat::Rgb16,
            },
        )
        .expect("capability report should parse 12-bit SOF1 metadata");

        assert_eq!(report.info.sof_kind, SofKind::Extended12);
        assert!(report.cpu.eligible, "op {op:?}");
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}

#[test]
fn capability_report_marks_extended12_gray16_and_rgb16_scaled_cpu_eligible() {
    let input = grayscale_sof_jpeg(0xc1, 12);
    for fmt in [PixelFormat::Gray16, PixelFormat::Rgb16] {
        for op in [
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
                .expect("capability report should parse 12-bit SOF1 metadata");

            assert_eq!(report.info.sof_kind, SofKind::Extended12);
            assert!(report.cpu.eligible, "fmt {fmt:?} op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_extended12_app14_rgb_rgb16_cpu_eligible() {
    let input = extended_12bit_rgb_8x8_jpeg();
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
        .expect("capability report should parse 12-bit SOF1 APP14 RGB metadata");

        assert_eq!(report.info.sof_kind, SofKind::Extended12);
        assert_eq!(report.info.bit_depth, 12);
        assert_eq!(report.info.color_space, ColorSpace::Rgb);
        assert!(report.cpu.eligible, "op {op:?}");
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}

#[test]
fn capability_report_marks_12bit_subsampled_app14_rgb_cpu_eligible() {
    for (name, input, expected_sof, expected_dimensions, expected_sampling) in [
        (
            "12-bit extended APP14 RGB 4:2:2",
            extended_12bit_rgb_422_32x8_jpeg(),
            SofKind::Extended12,
            (32, 8),
            [(2, 1), (1, 1), (1, 1)],
        ),
        (
            "12-bit extended APP14 RGB 4:2:0",
            extended_12bit_rgb_420_32x32_jpeg(),
            SofKind::Extended12,
            (32, 32),
            [(2, 2), (1, 1), (1, 1)],
        ),
        (
            "12-bit progressive APP14 RGB 4:2:2",
            progressive_12bit_rgb_422_32x8_jpeg(),
            SofKind::Progressive12,
            (32, 8),
            [(2, 1), (1, 1), (1, 1)],
        ),
        (
            "12-bit progressive APP14 RGB 4:2:0",
            progressive_12bit_rgb_420_32x32_jpeg(),
            SofKind::Progressive12,
            (32, 32),
            [(2, 2), (1, 1), (1, 1)],
        ),
    ] {
        for fmt in [PixelFormat::Rgb16, PixelFormat::Rgba16] {
            for op in [
                JpegDecodeOp::Full,
                JpegDecodeOp::Region(Rect {
                    x: expected_dimensions.0 / 4,
                    y: expected_dimensions.1 / 4,
                    w: expected_dimensions.0 / 2,
                    h: expected_dimensions.1 / 2,
                }),
                JpegDecodeOp::Scaled(Downscale::Half),
                JpegDecodeOp::RegionScaled {
                    roi: Rect {
                        x: expected_dimensions.0 / 4,
                        y: expected_dimensions.1 / 4,
                        w: expected_dimensions.0 / 2,
                        h: expected_dimensions.1 / 2,
                    },
                    scale: Downscale::Half,
                },
            ] {
                let report =
                    JpegCapabilityReport::inspect(&input, JpegCapabilityRequest { op, fmt })
                        .unwrap_or_else(|err| {
                            panic!("capability report should parse {name} metadata: {err}")
                        });

                assert_eq!(report.info.sof_kind, expected_sof, "{name}");
                assert_eq!(report.info.bit_depth, 12, "{name}");
                assert_eq!(report.info.dimensions, expected_dimensions, "{name}");
                assert_eq!(report.info.color_space, ColorSpace::Rgb, "{name}");
                assert_eq!(
                    report.info.sampling.components(),
                    &expected_sampling,
                    "{name}"
                );
                assert!(report.cpu.eligible, "{name} {fmt:?} {op:?}");
                assert_eq!(report.cpu.reason, None, "{name} {fmt:?} {op:?}");
                assert!(!report.owned_cuda.eligible, "{name}");
                assert!(!report.metal_fast.eligible, "{name}");
            }
        }
    }
}

#[test]
fn capability_report_marks_extended12_ycbcr444_rgb16_cpu_eligible() {
    let input = extended_12bit_ycbcr_8x8_jpeg();
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
        .expect("capability report should parse 12-bit SOF1 YCbCr metadata");

        assert_eq!(report.info.sof_kind, SofKind::Extended12);
        assert_eq!(report.info.bit_depth, 12);
        assert_eq!(report.info.color_space, ColorSpace::YCbCr);
        assert!(report.cpu.eligible, "op {op:?}");
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}

#[test]
fn capability_report_marks_extended12_ycbcr422_rgb16_cpu_eligible() {
    let input = extended_12bit_ycbcr_422_32x8_jpeg();
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
        .expect("capability report should parse 12-bit SOF1 YCbCr 4:2:2 metadata");

        assert_eq!(report.info.sof_kind, SofKind::Extended12);
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
fn capability_report_marks_extended12_ycbcr420_rgb16_cpu_eligible() {
    let input = extended_12bit_ycbcr_420_32x32_jpeg();
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
        .expect("capability report should parse 12-bit SOF1 YCbCr 4:2:0 metadata");

        assert_eq!(report.info.sof_kind, SofKind::Extended12);
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

#[test]
fn capability_report_marks_extended12_color_restart_rgb16_cpu_eligible() {
    for (name, input, expected_color, expected_dimensions, expected_sampling) in [
        (
            "APP14 RGB 4:4:4",
            extended_12bit_rgb_restart_16x8_jpeg(),
            ColorSpace::Rgb,
            (16, 8),
            (1, 1),
        ),
        (
            "YCbCr 4:4:4",
            extended_12bit_ycbcr_restart_16x8_jpeg(),
            ColorSpace::YCbCr,
            (16, 8),
            (1, 1),
        ),
        (
            "YCbCr 4:2:2",
            extended_12bit_ycbcr_422_restart_32x8_jpeg(),
            ColorSpace::YCbCr,
            (32, 8),
            (2, 1),
        ),
        (
            "YCbCr 4:2:0",
            extended_12bit_ycbcr_420_restart_32x32_jpeg(),
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
                    fmt: PixelFormat::Rgb16,
                },
            )
            .unwrap_or_else(|err| {
                panic!("capability report should parse 12-bit restart {name} metadata: {err}")
            });

            assert_eq!(report.info.sof_kind, SofKind::Extended12, "{name}");
            assert_eq!(report.info.restart_interval, Some(1), "{name}");
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
fn capability_report_marks_extended12_rgba16_cpu_eligible() {
    for (name, input, expected_color, expected_dimensions, expected_sampling) in [
        (
            "grayscale",
            grayscale_sof_jpeg(0xc1, 12),
            ColorSpace::Grayscale,
            (8, 8),
            (1, 1),
        ),
        (
            "APP14 RGB 4:4:4",
            extended_12bit_rgb_8x8_jpeg(),
            ColorSpace::Rgb,
            (8, 8),
            (1, 1),
        ),
        (
            "YCbCr 4:4:4",
            extended_12bit_ycbcr_8x8_jpeg(),
            ColorSpace::YCbCr,
            (8, 8),
            (1, 1),
        ),
        (
            "YCbCr 4:2:2",
            extended_12bit_ycbcr_422_32x8_jpeg(),
            ColorSpace::YCbCr,
            (32, 8),
            (2, 1),
        ),
        (
            "YCbCr 4:2:0",
            extended_12bit_ycbcr_420_32x32_jpeg(),
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
                panic!("capability report should parse 12-bit extended {name} metadata: {err}")
            });

            assert_eq!(report.info.sof_kind, SofKind::Extended12, "{name}");
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
fn capability_report_marks_extended12_restart_grayscale_cpu_eligible() {
    let input = extended_12bit_grayscale_restart_16x8_jpeg();
    for fmt in [PixelFormat::Gray16, PixelFormat::Rgb16] {
        for op in [
            JpegDecodeOp::Full,
            JpegDecodeOp::Region(Rect {
                x: 2,
                y: 1,
                w: 12,
                h: 6,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 2,
                    y: 1,
                    w: 12,
                    h: 6,
                },
                scale: Downscale::Half,
            },
        ] {
            let report = JpegCapabilityReport::inspect(&input, JpegCapabilityRequest { op, fmt })
                .expect("capability report should parse 12-bit restart metadata");

            assert_eq!(report.info.sof_kind, SofKind::Extended12);
            assert_eq!(report.info.restart_interval, Some(1));
            assert_eq!(report.info.dimensions, (16, 8));
            assert_eq!(report.info.color_space, ColorSpace::Grayscale);
            assert!(report.cpu.eligible, "fmt {fmt:?}, op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}
