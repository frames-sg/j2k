// SPDX-License-Identifier: MIT OR Apache-2.0

use super::support::{
    assert_cpu_only, cmyk_16x16_420_jpeg, cmyk_16x8_422_jpeg, cmyk_16x8_nonleading_max_422_jpeg,
    cmyk_8x8_jpeg, inspect_capability, standard_ops, ycck_16x16_420_jpeg, ycck_16x8_422_jpeg,
    ycck_16x8_nonleading_max_422_jpeg, ycck_8x8_jpeg, ColorSpace, Downscale, JpegCapabilityReport,
    JpegCapabilityRequest, JpegDecodeOp, PixelFormat, Rect, SofKind,
};

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
