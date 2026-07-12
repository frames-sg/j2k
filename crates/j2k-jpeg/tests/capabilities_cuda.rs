// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_jpeg::{
    rewrite_sof_dimensions, JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp, PixelFormat,
};
use j2k_test_support::JPEG_BASELINE_420_16X16;

fn full_rgb8_report(input: &[u8]) -> JpegCapabilityReport {
    JpegCapabilityReport::inspect(
        input,
        JpegCapabilityRequest {
            op: JpegDecodeOp::Full,
            fmt: PixelFormat::Rgb8,
        },
    )
    .expect("valid baseline JPEG capability report")
}

#[test]
fn owned_cuda_accepts_u32_addressable_rgb8_output() {
    let report = full_rgb8_report(JPEG_BASELINE_420_16X16);

    assert!(report.owned_cuda.eligible);
    assert_eq!(report.owned_cuda.reason, None);
}

#[test]
fn owned_cuda_rejects_rgb8_output_beyond_u32_byte_addressing() {
    let oversized = rewrite_sof_dimensions(JPEG_BASELINE_420_16X16, (65_500, 65_500))
        .expect("valid maximum JPEG dimensions");
    let report = full_rgb8_report(&oversized);

    assert!(!report.owned_cuda.eligible);
    assert_eq!(
        report.owned_cuda.reason,
        Some("J2K-owned CUDA JPEG decode requires RGB8 output addressable by u32 byte offsets")
    );
}
