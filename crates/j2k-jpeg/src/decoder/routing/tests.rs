// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;

use j2k_core::{Downscale, PixelFormat};

use crate::{Decoder, JpegError, Rect, ScratchPool};

use super::{DownscaleFactor, OutputFormat};

#[test]
fn non_lossless_decoder_declines_lossless_routing() {
    let decoder =
        Decoder::new(j2k_test_support::JPEG_BASELINE_420_16X16).expect("baseline fixture decoder");
    let mut out = vec![0; 16 * 16 * 3];

    assert!(decoder
        .decode_lossless_output_format_region_scaled(
            &mut out,
            16 * 3,
            OutputFormat::Rgb8,
            Rect::full((16, 16)),
            DownscaleFactor::Full,
            0,
        )
        .is_none());
}

#[test]
fn region_routing_rejects_out_of_bounds_before_output_validation() {
    let decoder =
        Decoder::new(j2k_test_support::JPEG_BASELINE_420_16X16).expect("baseline fixture decoder");
    let roi = Rect {
        x: 15,
        y: 15,
        w: 2,
        h: 2,
    };
    let mut out = [];

    assert_eq!(
        decoder
            .decode_region_scaled_into(&mut out, 0, PixelFormat::Rgb8, roi, Downscale::None)
            .expect_err("out-of-bounds ROI must fail first"),
        JpegError::RectOutOfBounds {
            rect: roi,
            width: 16,
            height: 16
        }
    );
}

#[test]
fn full_region_routing_matches_full_decode_with_caller_scratch() {
    let decoder =
        Decoder::new(j2k_test_support::JPEG_BASELINE_420_16X16).expect("baseline fixture decoder");
    let mut full = vec![0; 16 * 16 * 3];
    let mut region = vec![0; 16 * 16 * 3];
    let mut full_pool = ScratchPool::new();
    let mut region_pool = ScratchPool::new();

    let full_outcome = decoder
        .decode_scaled_into_with_scratch(
            &mut full_pool,
            &mut full,
            16 * 3,
            PixelFormat::Rgb8,
            Downscale::None,
        )
        .expect("full decode");
    let region_outcome = decoder
        .decode_region_scaled_into_with_scratch(
            &mut region_pool,
            &mut region,
            16 * 3,
            PixelFormat::Rgb8,
            Rect::full((16, 16)),
            Downscale::None,
        )
        .expect("full-sized region decode");

    assert_eq!(region, full);
    assert_eq!(region_outcome, full_outcome);
}

#[test]
fn scaled_routing_reports_required_output_geometry() {
    let decoder =
        Decoder::new(j2k_test_support::JPEG_BASELINE_420_16X16).expect("baseline fixture decoder");
    let mut out = vec![0; 7 * 8 * 3];

    assert!(matches!(
        decoder.decode_scaled_into(&mut out, 7 * 3, PixelFormat::Rgb8, Downscale::Half),
        Err(JpegError::InvalidStride {
            stride,
            row
        }) if stride == 7 * 3 && row == 8 * 3
    ));
}
