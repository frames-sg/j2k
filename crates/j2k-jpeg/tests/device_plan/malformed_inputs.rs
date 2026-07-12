// SPDX-License-Identifier: MIT OR Apache-2.0

use super::support::{
    baseline_420_with_sof_marker, baseline_grayscale_jpeg, insert_entropy_marker,
    lossless_predictor_grayscale_3x3_jpeg, lossless_ycbcr_16bit_422_3x3_jpeg,
    malformed_cmyk_nondivisible_sampling_jpeg, restart_coded_grayscale_jpeg, ColorSpace, Decoder,
    JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp, JpegError, PixelFormat, SofKind,
    UnsupportedReason, Warning, BASELINE_420,
};

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
fn adapter_device_plan_surfaces_missing_eoi_warning() {
    let mut bytes = baseline_grayscale_jpeg(24, 24);
    bytes.truncate(bytes.len() - 2);

    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = j2k_jpeg::adapter::build_device_plan(&decoder, 2)
        .expect("missing EOI should remain decodable");

    assert!(plan.warnings.contains(&Warning::MissingEoi));
}

#[test]
fn adapter_device_plan_treats_trailing_ff_as_missing_eoi() {
    let mut bytes = baseline_grayscale_jpeg(24, 24);
    bytes.truncate(bytes.len() - 1);

    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = j2k_jpeg::adapter::build_device_plan(&decoder, 2)
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
    let err = j2k_jpeg::adapter::build_device_plan(&decoder, 2)
        .expect_err("unexpected marker should fail");

    assert!(matches!(
        err,
        j2k_jpeg::JpegError::UnexpectedMarker {
            expected: j2k_jpeg::MarkerKind::Eoi,
            found: 0xe0,
            ..
        }
    ));
}

#[test]
fn adapter_device_plan_rejects_restart_marker_without_dri() {
    let bytes = insert_entropy_marker(BASELINE_420.to_vec(), 0xd0);
    let decoder = Decoder::new(&bytes).expect("decoder");
    let err = j2k_jpeg::adapter::build_device_plan(&decoder, 2)
        .expect_err("restart marker without DRI must fail");

    assert!(matches!(
        err,
        j2k_jpeg::JpegError::UnexpectedMarker {
            expected: j2k_jpeg::MarkerKind::Eoi,
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
    let err = j2k_jpeg::adapter::build_device_plan(&decoder, 2)
        .expect_err("double-FF terminal marker should fail");

    assert!(matches!(
        err,
        j2k_jpeg::JpegError::UnexpectedMarker {
            expected: j2k_jpeg::MarkerKind::Eoi,
            found: 0xff,
            ..
        }
    ));
}
