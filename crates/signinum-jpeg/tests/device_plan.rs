use std::borrow::Cow;

use signinum_jpeg::{
    ColorSpace, Decoder, Downscale, JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp,
    JpegError, PixelFormat, Rect, SofKind, UnsupportedReason, Warning,
};
use signinum_test_support::{
    baseline_grayscale_jpeg, restart_coded_grayscale_jpeg, JPEG_BASELINE_420_16X16,
    JPEG_BASELINE_422_16X8, JPEG_BASELINE_444_8X8,
};

use fixtures::{
    cmyk_16x16_420_jpeg, cmyk_16x8_422_jpeg, cmyk_16x8_nonleading_max_422_jpeg, cmyk_8x8_jpeg,
    extended_12bit_cmyk_16x16_420_jpeg, extended_12bit_cmyk_16x8_422_jpeg,
    extended_12bit_cmyk_420_restart_32x16_jpeg, extended_12bit_cmyk_422_restart_32x8_jpeg,
    extended_12bit_cmyk_8x8_jpeg, extended_12bit_cmyk_restart_16x8_jpeg,
    extended_12bit_grayscale_restart_16x8_jpeg, extended_12bit_rgb_420_32x32_jpeg,
    extended_12bit_rgb_422_32x8_jpeg, extended_12bit_rgb_8x8_jpeg,
    extended_12bit_rgb_restart_16x8_jpeg, extended_12bit_ycbcr_420_32x32_jpeg,
    extended_12bit_ycbcr_420_restart_32x32_jpeg, extended_12bit_ycbcr_422_32x8_jpeg,
    extended_12bit_ycbcr_422_restart_32x8_jpeg, extended_12bit_ycbcr_8x8_jpeg,
    extended_12bit_ycbcr_restart_16x8_jpeg, extended_12bit_ycck_16x16_420_jpeg,
    extended_12bit_ycck_16x8_422_jpeg, extended_12bit_ycck_420_restart_32x16_jpeg,
    extended_12bit_ycck_422_restart_32x8_jpeg, extended_12bit_ycck_8x8_jpeg,
    extended_12bit_ycck_restart_16x8_jpeg, lossless_predictor_grayscale_16bit_3x3_jpeg,
    lossless_predictor_grayscale_3x3_jpeg, lossless_predictor_rgb_16bit_3x3_jpeg,
    lossless_predictor_rgb_3x3_jpeg, lossless_predictor_ycbcr_16bit_3x3_jpeg,
    lossless_predictor_ycbcr_3x3_jpeg, lossless_restart_predictor_grayscale_16bit_3x3_jpeg,
    lossless_restart_predictor_grayscale_3x3_jpeg, lossless_restart_predictor_rgb_16bit_3x3_jpeg,
    lossless_restart_predictor_rgb_3x3_jpeg, lossless_restart_predictor_ycbcr_16bit_3x3_jpeg,
    lossless_restart_predictor_ycbcr_3x3_jpeg, lossless_rgb_16bit_420_4x4_jpeg,
    lossless_rgb_16bit_420_restart_4x4_jpeg, lossless_rgb_16bit_422_4x2_jpeg,
    lossless_rgb_16bit_422_restart_4x2_jpeg, lossless_rgb_8bit_420_4x4_jpeg,
    lossless_rgb_8bit_420_restart_4x4_jpeg, lossless_rgb_8bit_422_4x2_jpeg,
    lossless_rgb_8bit_422_restart_4x2_jpeg, lossless_ycbcr_16bit_420_4x4_jpeg,
    lossless_ycbcr_16bit_420_restart_4x4_jpeg, lossless_ycbcr_16bit_422_3x3_jpeg,
    lossless_ycbcr_16bit_422_4x2_jpeg, lossless_ycbcr_16bit_422_restart_4x2_jpeg,
    lossless_ycbcr_8bit_420_4x4_jpeg, lossless_ycbcr_8bit_420_restart_4x4_jpeg,
    lossless_ycbcr_8bit_422_4x2_jpeg, lossless_ycbcr_8bit_422_restart_4x2_jpeg,
    malformed_cmyk_nondivisible_sampling_jpeg, progressive_12bit_cmyk_16x16_420_jpeg,
    progressive_12bit_cmyk_16x8_422_jpeg, progressive_12bit_cmyk_420_restart_32x16_jpeg,
    progressive_12bit_cmyk_422_restart_32x8_jpeg, progressive_12bit_cmyk_8x8_jpeg,
    progressive_12bit_cmyk_restart_16x8_jpeg, progressive_12bit_grayscale_8x8_jpeg,
    progressive_12bit_rgb_420_32x32_jpeg, progressive_12bit_rgb_422_32x8_jpeg,
    progressive_12bit_rgb_8x8_jpeg, progressive_12bit_ycbcr_420_32x32_jpeg,
    progressive_12bit_ycbcr_422_32x8_jpeg, progressive_12bit_ycbcr_8x8_jpeg,
    progressive_12bit_ycck_16x16_420_jpeg, progressive_12bit_ycck_16x8_422_jpeg,
    progressive_12bit_ycck_420_restart_32x16_jpeg, progressive_12bit_ycck_422_restart_32x8_jpeg,
    progressive_12bit_ycck_8x8_jpeg, progressive_12bit_ycck_restart_16x8_jpeg,
    progressive_8x8_jpeg, ycck_16x16_420_jpeg, ycck_16x8_422_jpeg,
    ycck_16x8_nonleading_max_422_jpeg, ycck_8x8_jpeg,
};
use signinum_test_support as fixtures;

const BASELINE_420: &[u8] = JPEG_BASELINE_420_16X16;
const BASELINE_422: &[u8] = JPEG_BASELINE_422_16X8;
const BASELINE_444: &[u8] = JPEG_BASELINE_444_8X8;

fn baseline_420_with_sof_marker(marker: u8) -> Vec<u8> {
    let mut bytes = BASELINE_420.to_vec();
    let pos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xc0])
        .expect("baseline fixture has SOF0 marker");
    bytes[pos + 1] = marker;
    bytes
}

fn standard_ops(region_roi: Rect, scaled_roi: Rect) -> [JpegDecodeOp; 4] {
    [
        JpegDecodeOp::Full,
        JpegDecodeOp::Region(region_roi),
        JpegDecodeOp::Scaled(Downscale::Half),
        JpegDecodeOp::RegionScaled {
            roi: scaled_roi,
            scale: Downscale::Half,
        },
    ]
}

fn inspect_capability(
    input: &[u8],
    op: JpegDecodeOp,
    fmt: PixelFormat,
    context: &str,
) -> JpegCapabilityReport {
    JpegCapabilityReport::inspect(input, JpegCapabilityRequest { op, fmt })
        .unwrap_or_else(|err| panic!("{context}: {err}"))
}

fn assert_cpu_only(report: &JpegCapabilityReport, context: &str) {
    assert!(report.cpu.eligible, "{context}");
    assert!(!report.owned_cuda.eligible, "{context}");
    assert!(!report.metal_fast.eligible, "{context}");
}

#[test]
fn adapter_device_plan_exposes_scan_metadata() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 4).expect("device plan");

    assert_eq!(plan.dimensions, (16, 16));
    assert_eq!(plan.color_space, ColorSpace::YCbCr);
    assert_eq!(plan.components.len(), 3);
    assert_eq!(plan.checkpoints[0].mcu_index, 0);
    assert!(!plan.scan_bytes.is_empty());
}

#[test]
fn adapter_device_plan_borrows_scan_bytes_for_well_formed_streams() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 4).expect("device plan");

    assert!(matches!(plan.scan_bytes, Cow::Borrowed(_)));
}

#[test]
fn adapter_device_plan_keeps_fast_420_shape_information() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 4).expect("device plan");

    assert!(plan.matches_fast_420);
    assert!(!plan.matches_fast_422);
    assert!(!plan.matches_fast_444);
}

#[test]
fn adapter_device_plan_keeps_fast_422_shape_information() {
    let decoder = Decoder::new(BASELINE_422).expect("decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 4).expect("device plan");

    assert!(!plan.matches_fast_420);
    assert!(plan.matches_fast_422);
    assert!(!plan.matches_fast_444);
}

#[test]
fn adapter_device_plan_keeps_fast_444_shape_information() {
    let decoder = Decoder::new(BASELINE_444).expect("decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 4).expect("device plan");

    assert!(!plan.matches_fast_420);
    assert!(!plan.matches_fast_422);
    assert!(plan.matches_fast_444);
}

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
    assert_eq!(report.info.sof_kind, signinum_jpeg::SofKind::Baseline8);
    assert!(report.cpu.eligible);
    assert!(report.owned_cuda.eligible);
    assert!(report.metal_fast.eligible);
    assert!(report.device.matches_fast_420);
    assert!(!report.device.matches_fast_422);
    assert!(!report.device.matches_fast_444);
}

#[test]
fn capability_report_marks_cmyk_and_ycck_cpu_rgb8_rgba8_eligible() {
    for (input, expected_color) in [
        (cmyk_8x8_jpeg(), ColorSpace::Cmyk),
        (ycck_8x8_jpeg(), ColorSpace::Ycck),
    ] {
        for op in standard_ops(
            Rect {
                x: 2,
                y: 1,
                w: 5,
                h: 4,
            },
            Rect {
                x: 1,
                y: 1,
                w: 6,
                h: 6,
            },
        ) {
            let report = inspect_capability(
                &input,
                op,
                PixelFormat::Rgb8,
                "capability report should parse four-component color metadata",
            );

            assert_eq!(report.info.sof_kind, SofKind::Baseline8);
            assert_eq!(report.info.color_space, expected_color);
            assert_cpu_only(&report, &format!("{expected_color:?} {op:?}"));
            assert!(report
                .metal_fast
                .reason
                .expect("Metal rejection reason")
                .contains("YCbCr"));
        }

        for op in standard_ops(
            Rect {
                x: 3,
                y: 2,
                w: 3,
                h: 4,
            },
            Rect {
                x: 1,
                y: 1,
                w: 6,
                h: 6,
            },
        ) {
            let report = inspect_capability(
                &input,
                op,
                PixelFormat::Rgba8,
                "capability report should parse four-component color metadata",
            );

            assert_eq!(report.info.sof_kind, SofKind::Baseline8);
            assert_eq!(report.info.color_space, expected_color);
            assert_cpu_only(&report, &format!("{expected_color:?} {op:?}"));
        }
    }
}

#[test]
fn capability_report_marks_subsampled_cmyk_and_ycck_cpu_rgb8_rgba8_eligible() {
    for (name, input, expected_color, expected_dimensions, expected_sampling) in [
        (
            "CMYK 4:2:2",
            cmyk_16x8_422_jpeg(),
            ColorSpace::Cmyk,
            (16, 8),
            (2, 1),
        ),
        (
            "YCCK 4:2:2",
            ycck_16x8_422_jpeg(),
            ColorSpace::Ycck,
            (16, 8),
            (2, 1),
        ),
        (
            "CMYK 4:2:0",
            cmyk_16x16_420_jpeg(),
            ColorSpace::Cmyk,
            (16, 16),
            (2, 2),
        ),
        (
            "YCCK 4:2:0",
            ycck_16x16_420_jpeg(),
            ColorSpace::Ycck,
            (16, 16),
            (2, 2),
        ),
    ] {
        for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8] {
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

                assert_eq!(report.info.sof_kind, SofKind::Baseline8, "{name}");
                assert_eq!(report.info.dimensions, expected_dimensions, "{name}");
                assert_eq!(report.info.color_space, expected_color, "{name}");
                assert_eq!(report.info.sampling.max_h, expected_sampling.0, "{name}");
                assert_eq!(report.info.sampling.max_v, expected_sampling.1, "{name}");
                assert_eq!(report.info.sampling.components().len(), 4, "{name}");
                assert!(report.cpu.eligible, "{name} {fmt:?} {op:?}");
                assert!(!report.owned_cuda.eligible, "{name}");
                let owned_cuda_reason = report
                    .owned_cuda
                    .reason
                    .expect("owned CUDA rejection reason");
                if op == JpegDecodeOp::Full && fmt == PixelFormat::Rgb8 {
                    assert!(owned_cuda_reason.contains("YCbCr"), "{name}");
                } else {
                    assert!(owned_cuda_reason.contains("full-tile RGB8"), "{name}");
                }
                assert!(!report.metal_fast.eligible, "{name}");
                assert!(report
                    .metal_fast
                    .reason
                    .expect("Metal rejection reason")
                    .contains("YCbCr"));
            }
        }
    }
}

#[test]
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

#[test]
fn capability_report_marks_nonleading_max_four_component_sampling_cpu_eligible() {
    for (input, color_space, label) in [
        (
            cmyk_16x8_nonleading_max_422_jpeg(),
            ColorSpace::Cmyk,
            "CMYK",
        ),
        (
            ycck_16x8_nonleading_max_422_jpeg(),
            ColorSpace::Ycck,
            "YCCK",
        ),
    ] {
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op: JpegDecodeOp::Full,
                fmt: PixelFormat::Rgb8,
            },
        )
        .unwrap_or_else(|err| {
            panic!("{label} capability report should parse legal non-leading-max metadata: {err}")
        });

        assert_eq!(report.info.sof_kind, SofKind::Baseline8, "{label}");
        assert_eq!(report.info.color_space, color_space, "{label}");
        assert_eq!(
            report.info.sampling.components(),
            &[(1, 1), (2, 1), (1, 1), (1, 1)],
            "{label}"
        );
        assert!(report.cpu.eligible, "{label}");
        assert_eq!(report.cpu.reason, None, "{label}");
        assert!(!report.owned_cuda.eligible, "{label}");
        assert!(!report.metal_fast.eligible, "{label}");
    }
}

#[test]
fn capability_report_rejects_malformed_four_component_sampling_shape() {
    let input = malformed_cmyk_nondivisible_sampling_jpeg();
    let report = JpegCapabilityReport::inspect(
        &input,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Rgb8,
        },
    )
    .expect("capability report should parse malformed four-component metadata");

    assert_eq!(report.info.sof_kind, SofKind::Baseline8);
    assert_eq!(report.info.color_space, ColorSpace::Cmyk);
    assert_eq!(report.info.sampling.max_h, 3);
    assert_eq!(report.info.sampling.max_v, 1);
    assert_eq!(report.info.sampling.component(0), Some((3, 1)));
    assert!(!report.cpu.eligible);
    assert!(report
        .cpu
        .reason
        .expect("CPU rejection reason")
        .contains("planner rejected"));
    assert!(!report.owned_cuda.eligible);
    assert!(!report.metal_fast.eligible);
}

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
fn capability_report_rejects_future_sof_classes_with_typed_errors() {
    for (marker, expected_reason) in [
        (0xc9, UnsupportedReason::ArithmeticCoding),
        (0xc5, UnsupportedReason::DifferentialBaseline),
        (0xc6, UnsupportedReason::Hierarchical),
        (0xcd, UnsupportedReason::ArithmeticAndHierarchical),
    ] {
        let input = baseline_420_with_sof_marker(marker);
        let err = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op: JpegDecodeOp::Full,
                fmt: PixelFormat::Rgb8,
            },
        )
        .expect_err("future SOF classes should stay explicit unsupported errors");

        assert!(matches!(
            err,
            JpegError::UnsupportedSof {
                marker: got_marker,
                reason,
            } if got_marker == marker && reason == expected_reason
        ));
        assert!(err.is_unsupported());
    }
}

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

#[test]
fn capability_report_marks_lossless_common_predictor_gray8_full_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_predictor_grayscale_3x3_jpeg(predictor);
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op: JpegDecodeOp::Full,
                fmt: PixelFormat::Gray8,
            },
        )
        .unwrap_or_else(|err| {
            panic!("capability report should parse SOF3 predictor-{predictor} metadata: {err}")
        });

        assert_eq!(report.info.sof_kind, SofKind::Lossless);
        assert_eq!(report.info.bit_depth, 8);
        assert!(report.cpu.eligible, "predictor {predictor}");
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}

#[test]
fn capability_report_marks_lossless_common_predictor_gray8_roi_and_scaled_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_predictor_grayscale_3x3_jpeg(predictor);
        for op in [
            JpegDecodeOp::Region(Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
                    w: 2,
                    h: 2,
                },
                scale: Downscale::Half,
            },
        ] {
            let report = JpegCapabilityReport::inspect(
                &input,
                JpegCapabilityRequest {
                    op,
                    fmt: PixelFormat::Gray8,
                },
            )
            .unwrap_or_else(|err| {
                panic!("capability report should parse SOF3 predictor-{predictor} metadata: {err}")
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless);
            assert_eq!(report.info.bit_depth, 8);
            assert!(report.cpu.eligible, "predictor {predictor} op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_lossless_16bit_gray16_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_predictor_grayscale_16bit_3x3_jpeg(predictor);
        for op in [
            JpegDecodeOp::Full,
            JpegDecodeOp::Region(Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
                    w: 2,
                    h: 2,
                },
                scale: Downscale::Half,
            },
        ] {
            let report = JpegCapabilityReport::inspect(
                &input,
                JpegCapabilityRequest {
                    op,
                    fmt: PixelFormat::Gray16,
                },
            )
            .unwrap_or_else(|err| {
                panic!(
                    "capability report should parse 16-bit SOF3 predictor-{predictor} metadata: {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless);
            assert_eq!(report.info.bit_depth, 16);
            assert!(report.cpu.eligible, "predictor {predictor} op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_restart_coded_lossless_grayscale_cpu_eligible() {
    for predictor in 1..=7 {
        let cases = [
            (
                lossless_restart_predictor_grayscale_3x3_jpeg(predictor),
                PixelFormat::Gray8,
                8,
            ),
            (
                lossless_restart_predictor_grayscale_16bit_3x3_jpeg(predictor),
                PixelFormat::Gray16,
                16,
            ),
        ];

        for (input, fmt, bit_depth) in cases {
            let report = JpegCapabilityReport::inspect(
                &input,
                JpegCapabilityRequest {
                    op: JpegDecodeOp::RegionScaled {
                        roi: Rect {
                            x: 1,
                            y: 1,
                            w: 2,
                            h: 2,
                        },
                        scale: Downscale::Half,
                    },
                    fmt,
                },
            )
            .unwrap_or_else(|err| {
                panic!(
                    "capability report should parse restart-coded SOF3 predictor-{predictor} grayscale metadata: {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless);
            assert_eq!(report.info.bit_depth, bit_depth);
            assert_eq!(report.info.restart_interval, Some(3));
            assert!(report.cpu.eligible, "predictor {predictor}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_lossless_app14_rgb8_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_predictor_rgb_3x3_jpeg(predictor);
        for op in [
            JpegDecodeOp::Full,
            JpegDecodeOp::Region(Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
                    w: 2,
                    h: 2,
                },
                scale: Downscale::Half,
            },
        ] {
            let report = JpegCapabilityReport::inspect(
                &input,
                JpegCapabilityRequest {
                    op,
                    fmt: PixelFormat::Rgb8,
                },
            )
            .unwrap_or_else(|err| {
                panic!(
                    "capability report should parse SOF3 APP14 RGB predictor-{predictor} metadata: {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless);
            assert_eq!(report.info.bit_depth, 8);
            assert_eq!(report.info.color_space, ColorSpace::Rgb);
            assert!(report.cpu.eligible, "predictor {predictor} op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_lossless_app14_rgb8_rgba8_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_predictor_rgb_3x3_jpeg(predictor);
        for op in [
            JpegDecodeOp::Full,
            JpegDecodeOp::Region(Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
                    w: 2,
                    h: 2,
                },
                scale: Downscale::Half,
            },
        ] {
            let report = JpegCapabilityReport::inspect(
                &input,
                JpegCapabilityRequest {
                    op,
                    fmt: PixelFormat::Rgba8,
                },
            )
            .unwrap_or_else(|err| {
                panic!(
                    "capability report should parse SOF3 APP14 RGB RGBA8 predictor-{predictor} metadata: {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless);
            assert_eq!(report.info.bit_depth, 8);
            assert_eq!(report.info.color_space, ColorSpace::Rgb);
            assert!(report.cpu.eligible, "predictor {predictor} op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_lossless_app14_rgb16_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_predictor_rgb_16bit_3x3_jpeg(predictor);
        for op in [
            JpegDecodeOp::Full,
            JpegDecodeOp::Region(Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
                    w: 2,
                    h: 2,
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
                panic!(
                    "capability report should parse 16-bit SOF3 APP14 RGB predictor-{predictor} metadata: {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless);
            assert_eq!(report.info.bit_depth, 16);
            assert_eq!(report.info.color_space, ColorSpace::Rgb);
            assert!(report.cpu.eligible, "predictor {predictor} op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_lossless_app14_rgb16_rgba16_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_predictor_rgb_16bit_3x3_jpeg(predictor);
        for op in [
            JpegDecodeOp::Full,
            JpegDecodeOp::Region(Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
                    w: 2,
                    h: 2,
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
                panic!(
                    "capability report should parse 16-bit SOF3 APP14 RGB RGBA16 predictor-{predictor} metadata: {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless);
            assert_eq!(report.info.bit_depth, 16);
            assert_eq!(report.info.color_space, ColorSpace::Rgb);
            assert!(report.cpu.eligible, "predictor {predictor} op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_restart_coded_lossless_app14_rgb8_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_restart_predictor_rgb_3x3_jpeg(predictor);
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op: JpegDecodeOp::RegionScaled {
                    roi: Rect {
                        x: 1,
                        y: 1,
                        w: 2,
                        h: 2,
                    },
                    scale: Downscale::Half,
                },
                fmt: PixelFormat::Rgb8,
            },
        )
        .unwrap_or_else(|err| {
            panic!(
                "capability report should parse restart-coded SOF3 APP14 RGB predictor-{predictor} metadata: {err}"
            )
        });

        assert_eq!(report.info.sof_kind, SofKind::Lossless);
        assert_eq!(report.info.bit_depth, 8);
        assert_eq!(report.info.color_space, ColorSpace::Rgb);
        assert_eq!(report.info.restart_interval, Some(3));
        assert!(report.cpu.eligible, "predictor {predictor}");
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}

#[test]
fn capability_report_marks_lossless_ycbcr_rgb8_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_predictor_ycbcr_3x3_jpeg(predictor);
        for op in [
            JpegDecodeOp::Full,
            JpegDecodeOp::Region(Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
                    w: 2,
                    h: 2,
                },
                scale: Downscale::Half,
            },
        ] {
            let report = JpegCapabilityReport::inspect(
                &input,
                JpegCapabilityRequest {
                    op,
                    fmt: PixelFormat::Rgb8,
                },
            )
            .unwrap_or_else(|err| {
                panic!(
                    "capability report should parse SOF3 YCbCr predictor-{predictor} metadata: {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless);
            assert_eq!(report.info.bit_depth, 8);
            assert_eq!(report.info.color_space, ColorSpace::YCbCr);
            assert!(report.cpu.eligible, "predictor {predictor} op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_lossless_ycbcr_rgb8_rgba8_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_predictor_ycbcr_3x3_jpeg(predictor);
        for op in [
            JpegDecodeOp::Full,
            JpegDecodeOp::Region(Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
                    w: 2,
                    h: 2,
                },
                scale: Downscale::Half,
            },
        ] {
            let report = JpegCapabilityReport::inspect(
                &input,
                JpegCapabilityRequest {
                    op,
                    fmt: PixelFormat::Rgba8,
                },
            )
            .unwrap_or_else(|err| {
                panic!(
                    "capability report should parse SOF3 YCbCr RGBA8 predictor-{predictor} metadata: {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless);
            assert_eq!(report.info.bit_depth, 8);
            assert_eq!(report.info.color_space, ColorSpace::YCbCr);
            assert!(report.cpu.eligible, "predictor {predictor} op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_restart_coded_lossless_ycbcr_rgb8_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_restart_predictor_ycbcr_3x3_jpeg(predictor);
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op: JpegDecodeOp::RegionScaled {
                    roi: Rect {
                        x: 1,
                        y: 1,
                        w: 2,
                        h: 2,
                    },
                    scale: Downscale::Half,
                },
                fmt: PixelFormat::Rgb8,
            },
        )
        .unwrap_or_else(|err| {
            panic!(
                "capability report should parse restart-coded SOF3 YCbCr predictor-{predictor} metadata: {err}"
            )
        });

        assert_eq!(report.info.sof_kind, SofKind::Lossless);
        assert_eq!(report.info.bit_depth, 8);
        assert_eq!(report.info.color_space, ColorSpace::YCbCr);
        assert_eq!(report.info.restart_interval, Some(3));
        assert!(report.cpu.eligible, "predictor {predictor}");
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}

#[test]
fn capability_report_marks_lossless_ycbcr16_rgb16_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_predictor_ycbcr_16bit_3x3_jpeg(predictor);
        for op in [
            JpegDecodeOp::Full,
            JpegDecodeOp::Region(Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
                    w: 2,
                    h: 2,
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
                panic!(
                    "capability report should parse SOF3 16-bit YCbCr predictor-{predictor} metadata: {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless);
            assert_eq!(report.info.bit_depth, 16);
            assert_eq!(report.info.color_space, ColorSpace::YCbCr);
            assert!(report.cpu.eligible, "predictor {predictor} op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_lossless_ycbcr16_rgba16_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_predictor_ycbcr_16bit_3x3_jpeg(predictor);
        for op in [
            JpegDecodeOp::Full,
            JpegDecodeOp::Region(Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            }),
            JpegDecodeOp::Scaled(Downscale::Half),
            JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
                    w: 2,
                    h: 2,
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
                panic!(
                    "capability report should parse SOF3 16-bit YCbCr RGBA16 predictor-{predictor} metadata: {err}"
                )
            });

            assert_eq!(report.info.sof_kind, SofKind::Lossless);
            assert_eq!(report.info.bit_depth, 16);
            assert_eq!(report.info.color_space, ColorSpace::YCbCr);
            assert!(report.cpu.eligible, "predictor {predictor} op {op:?}");
            assert!(!report.owned_cuda.eligible);
            assert!(!report.metal_fast.eligible);
        }
    }
}

#[test]
fn capability_report_marks_restart_coded_lossless_ycbcr16_rgb16_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_restart_predictor_ycbcr_16bit_3x3_jpeg(predictor);
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op: JpegDecodeOp::RegionScaled {
                    roi: Rect {
                        x: 1,
                        y: 1,
                        w: 2,
                        h: 2,
                    },
                    scale: Downscale::Half,
                },
                fmt: PixelFormat::Rgb16,
            },
        )
        .unwrap_or_else(|err| {
            panic!(
                "capability report should parse restart-coded SOF3 16-bit YCbCr predictor-{predictor} metadata: {err}"
            )
        });

        assert_eq!(report.info.sof_kind, SofKind::Lossless);
        assert_eq!(report.info.bit_depth, 16);
        assert_eq!(report.info.color_space, ColorSpace::YCbCr);
        assert_eq!(report.info.restart_interval, Some(3));
        assert!(report.cpu.eligible, "predictor {predictor}");
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}

#[test]
fn capability_report_marks_restart_coded_lossless_app14_rgb16_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_restart_predictor_rgb_16bit_3x3_jpeg(predictor);
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op: JpegDecodeOp::RegionScaled {
                    roi: Rect {
                        x: 1,
                        y: 1,
                        w: 2,
                        h: 2,
                    },
                    scale: Downscale::Half,
                },
                fmt: PixelFormat::Rgb16,
            },
        )
        .unwrap_or_else(|err| {
            panic!(
                "capability report should parse restart-coded 16-bit SOF3 APP14 RGB predictor-{predictor} metadata: {err}"
            )
        });

        assert_eq!(report.info.sof_kind, SofKind::Lossless);
        assert_eq!(report.info.bit_depth, 16);
        assert_eq!(report.info.color_space, ColorSpace::Rgb);
        assert_eq!(report.info.restart_interval, Some(3));
        assert!(report.cpu.eligible, "predictor {predictor}");
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
    }
}

#[test]
fn capability_report_rejects_unsupported_lossless_predictor_explicitly() {
    let input = lossless_predictor_grayscale_3x3_jpeg(8);
    let err = JpegCapabilityReport::inspect(
        &input,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Gray8,
        },
    )
    .expect_err("unsupported SOF3 predictor should not infer CPU eligibility from parsed info");

    assert!(matches!(
        err,
        JpegError::UnsupportedPredictor { predictor: 8 }
    ));
}

#[test]
fn capability_report_rejects_unsupported_lossless_scan_shapes_without_info_fallback() {
    let mut invalid_scan_params = lossless_predictor_grayscale_3x3_jpeg(1);
    let sos = invalid_scan_params
        .windows(2)
        .position(|w| w == [0xff, 0xda])
        .expect("fixture has SOS");
    invalid_scan_params[sos + 8] = 1;

    let err = JpegCapabilityReport::inspect(
        &invalid_scan_params,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Gray8,
        },
    )
    .expect_err("unsupported SOF3 scan shape should not infer eligibility from parsed info");

    assert!(matches!(
        err,
        JpegError::NotImplemented {
            sof: SofKind::Lossless
        }
    ));
}

#[test]
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
            op: JpegDecodeOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 1,
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

#[test]
fn capability_report_rejects_malformed_subsampled_lossless_scan_params_without_info_fallback() {
    let mut invalid_scan_params = lossless_ycbcr_16bit_422_3x3_jpeg();
    let sos = invalid_scan_params
        .windows(2)
        .position(|w| w == [0xff, 0xda])
        .expect("fixture has SOS");
    let scan_component_count = usize::from(invalid_scan_params[sos + 4]);
    let se_offset = sos + 6 + scan_component_count * 2;
    invalid_scan_params[se_offset] = 1;

    let err = JpegCapabilityReport::inspect(
        &invalid_scan_params,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Rgb16,
        },
    )
    .expect_err(
        "malformed subsampled SOF3 scan shape should not infer eligibility from parsed info",
    );

    assert!(matches!(
        err,
        JpegError::NotImplemented {
            sof: SofKind::Lossless
        }
    ));
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

#[test]
fn adapter_device_plan_scan_bytes_keep_terminal_eoi() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 4).expect("device plan");

    assert!(plan.scan_bytes.ends_with(&[0xff, 0xd9]));
}

#[test]
fn adapter_device_plan_checkpoint_cadence_handles_multi_mcu_inputs() {
    let bytes = baseline_grayscale_jpeg(24, 24);
    let decoder = Decoder::new(&bytes).expect("grayscale decoder");

    let cadence_zero =
        signinum_jpeg::adapter::build_device_plan(&decoder, 0).expect("zero-cadence plan");
    let cadence_two =
        signinum_jpeg::adapter::build_device_plan(&decoder, 2).expect("cadence-two plan");

    assert_eq!(
        cadence_zero
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.mcu_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8]
    );
    let zero_offsets = cadence_zero
        .checkpoints
        .iter()
        .map(|checkpoint| checkpoint.scan_offset)
        .collect::<Vec<_>>();
    assert_eq!(zero_offsets.first(), Some(&0));
    assert!(zero_offsets.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(
        cadence_two
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.mcu_index)
            .collect::<Vec<_>>(),
        vec![0, 2, 4, 6, 8]
    );
    let cadence_two_offsets = cadence_two
        .checkpoints
        .iter()
        .map(|checkpoint| checkpoint.scan_offset)
        .collect::<Vec<_>>();
    assert_eq!(cadence_two_offsets.first(), Some(&0));
    assert!(cadence_two_offsets
        .windows(2)
        .all(|pair| pair[0] <= pair[1]));
    assert!(cadence_two
        .checkpoints
        .iter()
        .all(|checkpoint| checkpoint.bits_buffered <= 64 && checkpoint.expected_rst == 0));
}

#[test]
fn adapter_device_plan_restart_checkpoints_capture_resume_state() {
    let bytes = restart_coded_grayscale_jpeg(24, 24);
    let decoder = Decoder::new(&bytes).expect("restart-coded decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 2).expect("device plan");

    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.mcu_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.scan_offset)
            .collect::<Vec<_>>(),
        vec![0, 3, 6, 9, 12, 15, 18, 21, 24]
    );
    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.expected_rst)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 0]
    );
    assert!(plan
        .checkpoints
        .iter()
        .all(|checkpoint| checkpoint.bits_buffered == 0 && checkpoint.prev_dc == [0; 4]));
}

#[test]
fn adapter_device_plan_treats_dri_zero_as_non_restart_fast_path() {
    let bytes = insert_restart_interval(BASELINE_420.to_vec(), 0);
    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 2).expect("device plan");

    assert_eq!(plan.restart_interval, None);
    assert!(plan.matches_fast_420);
    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.expected_rst)
            .collect::<Vec<_>>(),
        vec![0; plan.checkpoints.len()]
    );
}

#[test]
fn adapter_device_plan_handles_restart_after_partial_entropy_byte() {
    let bytes = restart_coded_grayscale_jpeg(16, 8);
    let decoder = Decoder::new(&bytes).expect("restart-coded decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 2).expect("device plan");

    assert_eq!(plan.checkpoints.len(), 2);
    assert_eq!(plan.checkpoints[1].mcu_index, 1);
    assert_eq!(plan.checkpoints[1].scan_offset, 3);
    assert_eq!(plan.checkpoints[1].expected_rst, 1);
}

#[test]
fn adapter_device_plan_surfaces_missing_eoi_warning() {
    let mut bytes = baseline_grayscale_jpeg(24, 24);
    bytes.truncate(bytes.len() - 2);

    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 2)
        .expect("missing EOI should remain decodable");

    assert!(plan.warnings.contains(&Warning::MissingEoi));
}

#[test]
fn adapter_device_plan_treats_trailing_ff_as_missing_eoi() {
    let mut bytes = baseline_grayscale_jpeg(24, 24);
    bytes.truncate(bytes.len() - 1);

    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 2)
        .expect("trailing FF should remain decodable");

    assert!(plan.warnings.contains(&Warning::MissingEoi));
    assert_eq!(plan.scan_bytes.last(), Some(&0xff));
}

#[test]
fn adapter_device_plan_rejects_non_eoi_marker_after_entropy() {
    let mut bytes = restart_coded_grayscale_jpeg(24, 24);
    let marker = bytes
        .windows(2)
        .position(|window| matches!(window, [0xff, 0xd0..=0xd7]))
        .expect("restart marker");
    bytes[marker + 1] = 0xe0;

    let decoder = Decoder::new(&bytes).expect("restart-coded decoder");
    let err = signinum_jpeg::adapter::build_device_plan(&decoder, 2)
        .expect_err("unexpected marker should fail");

    assert!(matches!(
        err,
        signinum_jpeg::JpegError::UnexpectedMarker {
            expected: signinum_jpeg::MarkerKind::Eoi,
            found: 0xe0,
            ..
        }
    ));
}

#[test]
fn adapter_device_plan_rejects_restart_marker_without_dri() {
    let bytes = insert_entropy_marker(BASELINE_420.to_vec(), 0xd0);
    let decoder = Decoder::new(&bytes).expect("decoder");
    let err = signinum_jpeg::adapter::build_device_plan(&decoder, 2)
        .expect_err("restart marker without DRI must fail");

    assert!(matches!(
        err,
        signinum_jpeg::JpegError::UnexpectedMarker {
            expected: signinum_jpeg::MarkerKind::Eoi,
            found: 0xd0,
            ..
        }
    ));
}

#[test]
fn adapter_device_plan_rejects_doubled_ff_before_terminal_eoi() {
    let mut bytes = baseline_grayscale_jpeg(24, 24);
    bytes.insert(bytes.len() - 1, 0xff);

    let decoder = Decoder::new(&bytes).expect("decoder");
    let err = signinum_jpeg::adapter::build_device_plan(&decoder, 2)
        .expect_err("double-FF terminal marker should fail");

    assert!(matches!(
        err,
        signinum_jpeg::JpegError::UnexpectedMarker {
            expected: signinum_jpeg::MarkerKind::Eoi,
            found: 0xff,
            ..
        }
    ));
}

fn grayscale_sof_jpeg(marker: u8, precision: u8) -> Vec<u8> {
    let mut bytes = baseline_grayscale_jpeg(8, 8);
    let sof = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xc0])
        .expect("SOF0 marker");
    bytes[sof + 1] = marker;
    bytes[sof + 4] = precision;
    bytes
}

fn progressive_12_bit_jpeg() -> Vec<u8> {
    let mut bytes = progressive_8x8_jpeg();
    let sof = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xc2])
        .expect("SOF2 marker");
    bytes[sof + 4] = 12;
    bytes
}

fn insert_restart_interval(mut bytes: Vec<u8>, interval: u16) -> Vec<u8> {
    let sos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("SOS marker");
    bytes.splice(
        sos..sos,
        [
            0xff,
            0xdd,
            0x00,
            0x04,
            (interval >> 8) as u8,
            interval as u8,
        ],
    );
    bytes
}

fn insert_entropy_marker(mut bytes: Vec<u8>, marker: u8) -> Vec<u8> {
    bytes.splice(bytes.len() - 2..bytes.len() - 2, [0xff, marker]);
    bytes
}
