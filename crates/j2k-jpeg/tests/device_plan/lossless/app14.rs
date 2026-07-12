// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::support::{
    lossless_3x3_ops, lossless_3x3_region_scaled_op, lossless_predictor_rgb_16bit_3x3_jpeg,
    lossless_predictor_rgb_3x3_jpeg, lossless_restart_predictor_rgb_16bit_3x3_jpeg,
    lossless_restart_predictor_rgb_3x3_jpeg, ColorSpace, JpegCapabilityReport,
    JpegCapabilityRequest, PixelFormat, SofKind,
};

#[test]
fn capability_report_marks_lossless_app14_rgb8_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_predictor_rgb_3x3_jpeg(predictor);
        for op in lossless_3x3_ops() {
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
        for op in lossless_3x3_ops() {
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
        for op in lossless_3x3_ops() {
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
        for op in lossless_3x3_ops() {
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
                op: lossless_3x3_region_scaled_op(),
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
fn capability_report_marks_restart_coded_lossless_app14_rgb16_cpu_eligible() {
    for predictor in 1..=7 {
        let input = lossless_restart_predictor_rgb_16bit_3x3_jpeg(predictor);
        let report = JpegCapabilityReport::inspect(
            &input,
            JpegCapabilityRequest {
                op: lossless_3x3_region_scaled_op(),
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
