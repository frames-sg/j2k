// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::support::{
    lossless_3x3_ops, lossless_3x3_region_scaled_op, lossless_3x3_roi_and_scaled_ops,
    lossless_predictor_grayscale_16bit_3x3_jpeg, lossless_predictor_grayscale_3x3_jpeg,
    lossless_restart_predictor_grayscale_16bit_3x3_jpeg,
    lossless_restart_predictor_grayscale_3x3_jpeg, JpegCapabilityReport, JpegCapabilityRequest,
    JpegDecodeOp, PixelFormat, SofKind,
};

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
        for op in lossless_3x3_roi_and_scaled_ops() {
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
        for op in lossless_3x3_ops() {
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
                    op: lossless_3x3_region_scaled_op(),
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
