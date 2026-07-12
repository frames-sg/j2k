// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) use std::borrow::Cow;

pub(crate) use j2k_jpeg::{
    ColorSpace, Decoder, Downscale, JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp,
    JpegError, PixelFormat, Rect, SofKind, UnsupportedReason, Warning,
};
pub(crate) use j2k_test_support::{
    baseline_grayscale_jpeg, restart_coded_grayscale_jpeg, JPEG_BASELINE_420_16X16,
    JPEG_BASELINE_422_16X8, JPEG_BASELINE_444_8X8,
};

pub(crate) use fixtures::{
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
use j2k_test_support as fixtures;

pub(crate) const BASELINE_420: &[u8] = JPEG_BASELINE_420_16X16;
pub(crate) const BASELINE_422: &[u8] = JPEG_BASELINE_422_16X8;
pub(crate) const BASELINE_444: &[u8] = JPEG_BASELINE_444_8X8;

pub(crate) fn baseline_420_with_sof_marker(marker: u8) -> Vec<u8> {
    let mut bytes = BASELINE_420.to_vec();
    let pos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xc0])
        .expect("baseline fixture has SOF0 marker");
    bytes[pos + 1] = marker;
    bytes
}

pub(crate) fn standard_ops(region_roi: Rect, scaled_roi: Rect) -> [JpegDecodeOp; 4] {
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

pub(crate) fn lossless_3x3_roi() -> Rect {
    Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    }
}

pub(crate) fn lossless_3x3_ops() -> [JpegDecodeOp; 4] {
    standard_ops(lossless_3x3_roi(), lossless_3x3_roi())
}

pub(crate) fn lossless_3x3_roi_and_scaled_ops() -> [JpegDecodeOp; 3] {
    [
        JpegDecodeOp::Region(lossless_3x3_roi()),
        JpegDecodeOp::Scaled(Downscale::Half),
        lossless_3x3_region_scaled_op(),
    ]
}

pub(crate) fn lossless_3x3_region_scaled_op() -> JpegDecodeOp {
    JpegDecodeOp::RegionScaled {
        roi: lossless_3x3_roi(),
        scale: Downscale::Half,
    }
}

pub(crate) fn inspect_capability(
    input: &[u8],
    op: JpegDecodeOp,
    fmt: PixelFormat,
    context: &str,
) -> JpegCapabilityReport {
    JpegCapabilityReport::inspect(input, JpegCapabilityRequest { op, fmt })
        .unwrap_or_else(|err| panic!("{context}: {err}"))
}

pub(crate) fn assert_cpu_only(report: &JpegCapabilityReport, context: &str) {
    assert!(report.cpu.eligible, "{context}");
    assert!(!report.owned_cuda.eligible, "{context}");
    assert!(!report.metal_fast.eligible, "{context}");
}

pub(crate) fn grayscale_sof_jpeg(marker: u8, precision: u8) -> Vec<u8> {
    let mut bytes = baseline_grayscale_jpeg(8, 8);
    let sof = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xc0])
        .expect("SOF0 marker");
    bytes[sof + 1] = marker;
    bytes[sof + 4] = precision;
    bytes
}

pub(crate) fn progressive_12_bit_jpeg() -> Vec<u8> {
    let mut bytes = progressive_8x8_jpeg();
    let sof = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xc2])
        .expect("SOF2 marker");
    bytes[sof + 4] = 12;
    bytes
}

pub(crate) fn insert_restart_interval(mut bytes: Vec<u8>, interval: u16) -> Vec<u8> {
    let [interval_hi, interval_lo] = interval.to_be_bytes();
    let sos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("SOS marker");
    bytes.splice(sos..sos, [0xff, 0xdd, 0x00, 0x04, interval_hi, interval_lo]);
    bytes
}

pub(crate) fn insert_entropy_marker(mut bytes: Vec<u8>, marker: u8) -> Vec<u8> {
    bytes.splice(bytes.len() - 2..bytes.len() - 2, [0xff, marker]);
    bytes
}
