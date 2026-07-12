// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::adapter::fast_packet::{
    build_fast420_packet, build_gray_packet, classify_color_fast_packet_family,
    JpegFastPacketFamily,
};
use crate::Decoder;

#[test]
fn color_family_classification_includes_restart_coded_420() {
    for (bytes, expected) in [
        (
            j2k_test_support::JPEG_BASELINE_420_16X16,
            JpegFastPacketFamily::Fast420,
        ),
        (
            j2k_test_support::JPEG_BASELINE_420_RESTART_32X16,
            JpegFastPacketFamily::Fast420,
        ),
        (
            j2k_test_support::JPEG_BASELINE_422_16X8,
            JpegFastPacketFamily::Fast422,
        ),
        (
            j2k_test_support::JPEG_BASELINE_444_8X8,
            JpegFastPacketFamily::Fast444,
        ),
    ] {
        let decoder = Decoder::new(bytes).expect("fixture decoder");
        assert_eq!(classify_color_fast_packet_family(&decoder), Some(expected));
    }
}

#[test]
fn packets_accept_missing_eoi_without_changing_materialized_entropy() {
    let color = j2k_test_support::minimal_baseline_420_jpeg();
    let expected_color = build_fast420_packet(&color).expect("terminated color packet");
    let color_without_eoi = &color[..color.len() - 2];
    let actual_color = build_fast420_packet(color_without_eoi).expect("missing-EOI color packet");
    assert_eq!(actual_color.entropy_bytes, expected_color.entropy_bytes);
    assert_eq!(
        actual_color.entropy_checkpoints,
        expected_color.entropy_checkpoints
    );
    let color_without_eoi_code = &color[..color.len() - 1];
    let actual_color =
        build_fast420_packet(color_without_eoi_code).expect("missing EOI code color packet");
    assert_eq!(actual_color.entropy_bytes, expected_color.entropy_bytes);
    assert_eq!(
        actual_color.entropy_checkpoints,
        expected_color.entropy_checkpoints
    );

    let gray = j2k_test_support::grayscale_8x8_jpeg();
    let expected_gray = build_gray_packet(&gray).expect("terminated gray packet");
    let gray_without_eoi = &gray[..gray.len() - 2];
    let actual_gray = build_gray_packet(gray_without_eoi).expect("missing-EOI gray packet");
    assert_eq!(actual_gray.entropy_bytes, expected_gray.entropy_bytes);
    assert_eq!(actual_gray.restart_offsets, expected_gray.restart_offsets);
    let gray_without_eoi_code = &gray[..gray.len() - 1];
    let actual_gray =
        build_gray_packet(gray_without_eoi_code).expect("missing EOI code gray packet");
    assert_eq!(actual_gray.entropy_bytes, expected_gray.entropy_bytes);
    assert_eq!(actual_gray.restart_offsets, expected_gray.restart_offsets);
}

#[test]
fn malformed_packet_input_returns_an_error() {
    let malformed = [0xff, 0xd8, 0xff];
    assert!(build_fast420_packet(&malformed).is_err());
    assert!(build_gray_packet(&malformed).is_err());
}
