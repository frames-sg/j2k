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

/// Rectangles with no pixels, placed inside a `width`×`height` image.
fn empty_rects(width: u32, height: u32) -> [Rect; 4] {
    [
        Rect {
            x: 0,
            y: 0,
            w: width,
            h: 0,
        },
        Rect {
            x: 0,
            y: 0,
            w: 0,
            h: height,
        },
        Rect {
            x: width / 2,
            y: height / 2,
            w: 0,
            h: 0,
        },
        Rect {
            x: width,
            y: height,
            w: 0,
            h: 0,
        },
    ]
}

#[test]
fn region_routing_rejects_empty_regions_for_every_frame_kind() {
    let extended12_gray = j2k_test_support::extended_12bit_grayscale_8x8_jpeg();
    let extended12_rgb = j2k_test_support::extended_12bit_rgb_420_32x32_jpeg();
    let progressive12_gray = j2k_test_support::progressive_12bit_grayscale_8x8_jpeg();
    let cases: [(&str, &[u8], PixelFormat); 7] = [
        (
            "baseline 4:2:0",
            j2k_test_support::JPEG_BASELINE_420_16X16,
            PixelFormat::Rgb8,
        ),
        (
            "baseline 4:2:2",
            j2k_test_support::JPEG_BASELINE_422_16X8,
            PixelFormat::Rgb8,
        ),
        (
            "baseline 4:4:4",
            j2k_test_support::JPEG_BASELINE_444_8X8,
            PixelFormat::Rgb8,
        ),
        (
            "baseline grayscale",
            j2k_test_support::JPEG_GRAYSCALE_8X8,
            PixelFormat::Gray8,
        ),
        (
            "extended 12-bit gray",
            &extended12_gray,
            PixelFormat::Gray16,
        ),
        ("extended 12-bit 4:2:0", &extended12_rgb, PixelFormat::Rgb16),
        (
            "progressive 12-bit gray",
            &progressive12_gray,
            PixelFormat::Gray16,
        ),
    ];

    for (name, jpeg, fmt) in cases {
        let decoder = Decoder::new(jpeg).expect("fixture decoder");
        let (width, height) = decoder.info().dimensions;
        let stride = width as usize * fmt.bytes_per_pixel();
        let mut out = vec![0; stride * height as usize];
        for roi in empty_rects(width, height) {
            for scale in [Downscale::None, Downscale::Half] {
                assert_eq!(
                    decoder
                        .decode_region_scaled_into(&mut out, stride, fmt, roi, scale)
                        .expect_err("an empty region has nothing to decode"),
                    JpegError::RectOutOfBounds {
                        rect: roi,
                        width,
                        height
                    },
                    "{name}, {roi:?}, {scale:?}"
                );
            }
        }
    }
}

#[test]
fn owned_region_requests_reject_empty_regions() {
    let decoder =
        Decoder::new(j2k_test_support::JPEG_BASELINE_420_16X16).expect("baseline fixture decoder");
    let mut pool = ScratchPool::new();

    for roi in empty_rects(16, 16) {
        let request = crate::DecodeRequest {
            fmt: PixelFormat::Rgb8,
            region: Some(roi),
            scale: Downscale::None,
        };

        assert_eq!(
            decoder
                .decode_request_with_scratch_and_external_live(&mut pool, request, 0)
                .expect_err("an empty region has nothing to decode"),
            JpegError::RectOutOfBounds {
                rect: roi,
                width: 16,
                height: 16
            },
            "{roi:?}"
        );
    }
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
