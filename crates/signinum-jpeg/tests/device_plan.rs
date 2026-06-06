use std::borrow::Cow;

use signinum_jpeg::{
    ColorSpace, Decoder, Downscale, JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp,
    JpegError, PixelFormat, Rect, SofKind, Warning,
};

mod fixtures;
use fixtures::{
    cmyk_8x8_jpeg, extended_12bit_rgb_8x8_jpeg, extended_12bit_ycbcr_8x8_jpeg,
    lossless_predictor_grayscale_16bit_3x3_jpeg, lossless_predictor_grayscale_3x3_jpeg,
    progressive_12bit_grayscale_8x8_jpeg, progressive_12bit_rgb_8x8_jpeg,
    progressive_12bit_ycbcr_8x8_jpeg, progressive_8x8_jpeg, ycck_8x8_jpeg,
};

const BASELINE_420: &[u8] = include_bytes!("../fixtures/conformance/baseline_420_16x16.jpg");
const BASELINE_422: &[u8] = include_bytes!("../fixtures/conformance/baseline_422_16x8.jpg");
const BASELINE_444: &[u8] = include_bytes!("../fixtures/conformance/baseline_444_8x8.jpg");

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
fn capability_report_marks_cmyk_and_ycck_cpu_rgb8_eligible() {
    for (input, expected_color) in [
        (cmyk_8x8_jpeg(), ColorSpace::Cmyk),
        (ycck_8x8_jpeg(), ColorSpace::Ycck),
    ] {
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op: JpegDecodeOp::Full,
                fmt: PixelFormat::Rgb8,
            },
        )
        .expect("capability report should parse unsupported color metadata");

        assert_eq!(report.info.sof_kind, SofKind::Baseline8);
        assert_eq!(report.info.color_space, expected_color);
        assert!(report.cpu.eligible);
        assert!(!report.owned_cuda.eligible);
        assert!(!report.metal_fast.eligible);
        assert!(report
            .metal_fast
            .reason
            .expect("Metal rejection reason")
            .contains("YCbCr"));
    }
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
                fmt: PixelFormat::Rgba16,
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
fn capability_report_rejects_extended12_restart_interval_cpu_decode() {
    let input = grayscale_restart_sof_jpeg(0xc1, 12);
    let report = JpegCapabilityReport::inspect(
        &input,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Scaled(Downscale::Half),
            fmt: PixelFormat::Gray16,
        },
    )
    .expect("capability report should parse 12-bit restart metadata");

    assert_eq!(report.info.sof_kind, SofKind::Extended12);
    assert_eq!(report.info.restart_interval, Some(1));
    assert!(!report.cpu.eligible);
    assert!(report
        .cpu
        .reason
        .expect("12-bit restart rejection")
        .contains("restart intervals"));
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
    let restart_coded = insert_restart_interval(lossless_predictor_grayscale_3x3_jpeg(1), 1);
    let mut invalid_scan_params = lossless_predictor_grayscale_3x3_jpeg(1);
    let sos = invalid_scan_params
        .windows(2)
        .position(|w| w == [0xff, 0xda])
        .expect("fixture has SOS");
    invalid_scan_params[sos + 8] = 1;

    for input in [restart_coded, invalid_scan_params] {
        let err = JpegCapabilityReport::inspect(
            &input,
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
    let bytes = grayscale_jpeg(24, 24);
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
    let bytes = grayscale_restart_jpeg();
    let decoder = Decoder::new(&bytes).expect("restart-coded decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 2).expect("device plan");

    assert_eq!(plan.checkpoints.len(), 2);
    assert_eq!(plan.checkpoints[1].mcu_index, 1);
    assert_eq!(plan.checkpoints[1].scan_offset, 3);
    assert_eq!(plan.checkpoints[1].expected_rst, 1);
}

#[test]
fn adapter_device_plan_surfaces_missing_eoi_warning() {
    let mut bytes = grayscale_jpeg(24, 24);
    bytes.truncate(bytes.len() - 2);

    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = signinum_jpeg::adapter::build_device_plan(&decoder, 2)
        .expect("missing EOI should remain decodable");

    assert!(plan.warnings.contains(&Warning::MissingEoi));
}

#[test]
fn adapter_device_plan_treats_trailing_ff_as_missing_eoi() {
    let mut bytes = grayscale_jpeg(24, 24);
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
    let mut bytes = grayscale_jpeg(24, 24);
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

fn restart_coded_grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff,
        0xc0,
        0x00,
        11,
        8,
        (height >> 8) as u8,
        height as u8,
        (width >> 8) as u8,
        width as u8,
        1,
        1,
        0x11,
        0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);

    let mcu_cols = u32::from(width).div_ceil(8);
    let mcu_rows = u32::from(height).div_ceil(8);
    let mcu_count = (mcu_cols * mcu_rows) as usize;
    for mcu in 0..mcu_count {
        bytes.push(0x00);
        if mcu + 1 != mcu_count {
            bytes.extend_from_slice(&[0xff, 0xd0 | ((mcu as u8) & 0x07)]);
        }
    }

    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff,
        0xc0,
        0x00,
        11,
        8,
        (height >> 8) as u8,
        height as u8,
        (width >> 8) as u8,
        width as u8,
        1,
        1,
        0x11,
        0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);

    let mcu_cols = u32::from(width).div_ceil(8);
    let mcu_rows = u32::from(height).div_ceil(8);
    let mcu_count = (mcu_cols * mcu_rows) as usize;
    bytes.extend(std::iter::repeat_n(0x00, mcu_count));

    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn grayscale_restart_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[0xff, 0xc0, 0x00, 11, 8, 0, 8, 0, 16, 1, 1, 0x11, 0]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);
    bytes.extend_from_slice(&[0x00, 0xff, 0xd0, 0x00, 0xff, 0xd9]);
    bytes
}

fn grayscale_sof_jpeg(marker: u8, precision: u8) -> Vec<u8> {
    let mut bytes = grayscale_jpeg(8, 8);
    let sof = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xc0])
        .expect("SOF0 marker");
    bytes[sof + 1] = marker;
    bytes[sof + 4] = precision;
    bytes
}

fn grayscale_restart_sof_jpeg(marker: u8, precision: u8) -> Vec<u8> {
    let mut bytes = grayscale_restart_jpeg();
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
