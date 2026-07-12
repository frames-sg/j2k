// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::support::{
    extended_12bit_cmyk_16x16_420_jpeg, extended_12bit_cmyk_16x8_422_jpeg,
    extended_12bit_cmyk_420_restart_32x16_jpeg, extended_12bit_cmyk_422_restart_32x8_jpeg,
    extended_12bit_cmyk_8x8_jpeg, extended_12bit_cmyk_restart_16x8_jpeg,
    extended_12bit_ycck_16x16_420_jpeg, extended_12bit_ycck_16x8_422_jpeg,
    extended_12bit_ycck_420_restart_32x16_jpeg, extended_12bit_ycck_422_restart_32x8_jpeg,
    extended_12bit_ycck_8x8_jpeg, extended_12bit_ycck_restart_16x8_jpeg,
    progressive_12bit_cmyk_16x16_420_jpeg, progressive_12bit_cmyk_16x8_422_jpeg,
    progressive_12bit_cmyk_420_restart_32x16_jpeg, progressive_12bit_cmyk_422_restart_32x8_jpeg,
    progressive_12bit_cmyk_8x8_jpeg, progressive_12bit_cmyk_restart_16x8_jpeg,
    progressive_12bit_ycck_16x16_420_jpeg, progressive_12bit_ycck_16x8_422_jpeg,
    progressive_12bit_ycck_420_restart_32x16_jpeg, progressive_12bit_ycck_422_restart_32x8_jpeg,
    progressive_12bit_ycck_8x8_jpeg, progressive_12bit_ycck_restart_16x8_jpeg, ColorSpace,
    Downscale, JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp, PixelFormat, Rect,
    SofKind,
};

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the capability report test verifies one coherent 12-bit four-component eligibility contract"
)]
fn capability_report_marks_12bit_four_component_cpu_eligible() {
    for (name, input, expected_sof, expected_color, expected_dimensions, expected_sampling) in [
        (
            "12-bit CMYK 4:4:4",
            extended_12bit_cmyk_8x8_jpeg(),
            SofKind::Extended12,
            ColorSpace::Cmyk,
            (8, 8),
            [(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "12-bit YCCK 4:4:4",
            extended_12bit_ycck_8x8_jpeg(),
            SofKind::Extended12,
            ColorSpace::Ycck,
            (8, 8),
            [(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "12-bit CMYK 4:2:2",
            extended_12bit_cmyk_16x8_422_jpeg(),
            SofKind::Extended12,
            ColorSpace::Cmyk,
            (16, 8),
            [(2, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "12-bit YCCK 4:2:2",
            extended_12bit_ycck_16x8_422_jpeg(),
            SofKind::Extended12,
            ColorSpace::Ycck,
            (16, 8),
            [(2, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "12-bit CMYK 4:2:0",
            extended_12bit_cmyk_16x16_420_jpeg(),
            SofKind::Extended12,
            ColorSpace::Cmyk,
            (16, 16),
            [(2, 2), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "12-bit YCCK 4:2:0",
            extended_12bit_ycck_16x16_420_jpeg(),
            SofKind::Extended12,
            ColorSpace::Ycck,
            (16, 16),
            [(2, 2), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "progressive 12-bit CMYK 4:4:4",
            progressive_12bit_cmyk_8x8_jpeg(),
            SofKind::Progressive12,
            ColorSpace::Cmyk,
            (8, 8),
            [(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "progressive 12-bit YCCK 4:4:4",
            progressive_12bit_ycck_8x8_jpeg(),
            SofKind::Progressive12,
            ColorSpace::Ycck,
            (8, 8),
            [(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "progressive 12-bit CMYK 4:2:2",
            progressive_12bit_cmyk_16x8_422_jpeg(),
            SofKind::Progressive12,
            ColorSpace::Cmyk,
            (16, 8),
            [(2, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "progressive 12-bit YCCK 4:2:2",
            progressive_12bit_ycck_16x8_422_jpeg(),
            SofKind::Progressive12,
            ColorSpace::Ycck,
            (16, 8),
            [(2, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "progressive 12-bit CMYK 4:2:0",
            progressive_12bit_cmyk_16x16_420_jpeg(),
            SofKind::Progressive12,
            ColorSpace::Cmyk,
            (16, 16),
            [(2, 2), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "progressive 12-bit YCCK 4:2:0",
            progressive_12bit_ycck_16x16_420_jpeg(),
            SofKind::Progressive12,
            ColorSpace::Ycck,
            (16, 16),
            [(2, 2), (1, 1), (1, 1), (1, 1)],
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
                assert_eq!(report.info.color_space, expected_color, "{name}");
                assert_eq!(report.info.sampling.components().len(), 4, "{name}");
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
#[expect(
    clippy::too_many_lines,
    reason = "the capability report test verifies one coherent restart-coded eligibility contract"
)]
fn capability_report_marks_restart_coded_12bit_four_component_cpu_eligible() {
    for (name, input, expected_sof, expected_color, expected_dimensions, expected_sampling) in [
        (
            "restart-coded 12-bit CMYK 4:4:4",
            extended_12bit_cmyk_restart_16x8_jpeg(),
            SofKind::Extended12,
            ColorSpace::Cmyk,
            (16, 8),
            [(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "restart-coded 12-bit YCCK 4:4:4",
            extended_12bit_ycck_restart_16x8_jpeg(),
            SofKind::Extended12,
            ColorSpace::Ycck,
            (16, 8),
            [(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "restart-coded 12-bit CMYK 4:2:2",
            extended_12bit_cmyk_422_restart_32x8_jpeg(),
            SofKind::Extended12,
            ColorSpace::Cmyk,
            (32, 8),
            [(2, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "restart-coded 12-bit YCCK 4:2:2",
            extended_12bit_ycck_422_restart_32x8_jpeg(),
            SofKind::Extended12,
            ColorSpace::Ycck,
            (32, 8),
            [(2, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "restart-coded 12-bit CMYK 4:2:0",
            extended_12bit_cmyk_420_restart_32x16_jpeg(),
            SofKind::Extended12,
            ColorSpace::Cmyk,
            (32, 16),
            [(2, 2), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "restart-coded 12-bit YCCK 4:2:0",
            extended_12bit_ycck_420_restart_32x16_jpeg(),
            SofKind::Extended12,
            ColorSpace::Ycck,
            (32, 16),
            [(2, 2), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "restart-coded progressive 12-bit CMYK 4:4:4",
            progressive_12bit_cmyk_restart_16x8_jpeg(),
            SofKind::Progressive12,
            ColorSpace::Cmyk,
            (16, 8),
            [(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "restart-coded progressive 12-bit YCCK 4:4:4",
            progressive_12bit_ycck_restart_16x8_jpeg(),
            SofKind::Progressive12,
            ColorSpace::Ycck,
            (16, 8),
            [(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "restart-coded progressive 12-bit CMYK 4:2:2",
            progressive_12bit_cmyk_422_restart_32x8_jpeg(),
            SofKind::Progressive12,
            ColorSpace::Cmyk,
            (32, 8),
            [(2, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "restart-coded progressive 12-bit YCCK 4:2:2",
            progressive_12bit_ycck_422_restart_32x8_jpeg(),
            SofKind::Progressive12,
            ColorSpace::Ycck,
            (32, 8),
            [(2, 1), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "restart-coded progressive 12-bit CMYK 4:2:0",
            progressive_12bit_cmyk_420_restart_32x16_jpeg(),
            SofKind::Progressive12,
            ColorSpace::Cmyk,
            (32, 16),
            [(2, 2), (1, 1), (1, 1), (1, 1)],
        ),
        (
            "restart-coded progressive 12-bit YCCK 4:2:0",
            progressive_12bit_ycck_420_restart_32x16_jpeg(),
            SofKind::Progressive12,
            ColorSpace::Ycck,
            (32, 16),
            [(2, 2), (1, 1), (1, 1), (1, 1)],
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
                assert_eq!(report.info.restart_interval, Some(1), "{name}");
                assert_eq!(report.info.dimensions, expected_dimensions, "{name}");
                assert_eq!(report.info.color_space, expected_color, "{name}");
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
