// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for `Decoder::decode_into`.

use j2k_jpeg::{DecodeRequest, Decoder, Downscale, JpegError, PixelFormat, Rect, SofKind};

use fixtures::{
    cmyk_16x16_420_jpeg, cmyk_16x8_422_jpeg, cmyk_16x8_nonleading_max_422_jpeg, cmyk_8x8_jpeg,
    extended_12bit_cmyk_8x8_jpeg, extended_12bit_cmyk_nonconstant_8x8_jpeg,
    extended_12bit_grayscale_8x8_jpeg, extended_12bit_grayscale_restart_16x8_jpeg,
    extended_12bit_ycck_8x8_jpeg, extended_12bit_ycck_nonconstant_8x8_jpeg,
    four_component_12bit_16x16_rgb16, four_component_12bit_16x8_rgb16,
    four_component_12bit_32x16_rgb16, four_component_12bit_32x8_rgb16,
    four_component_12bit_8x8_cmyk_nonconstant_rgb16, four_component_12bit_8x8_rgb16,
    four_component_12bit_8x8_ycck_nonconstant_rgb16, four_component_16x16_rgb,
    four_component_16x8_rgb, four_component_8x8_rgb, grayscale_8x8_jpeg,
    lossless_predictor_grayscale_16bit_3x3_jpeg, lossless_predictor_grayscale_3x3_jpeg,
    lossless_predictor_rgb_16bit_3x3_jpeg, lossless_predictor_rgb_3x3_jpeg,
    lossless_predictor_ycbcr_16bit_3x3_jpeg, lossless_predictor_ycbcr_3x3_jpeg,
    lossless_restart_predictor_grayscale_16bit_3x3_jpeg,
    lossless_restart_predictor_grayscale_3x3_jpeg, lossless_restart_predictor_rgb_16bit_3x3_jpeg,
    lossless_restart_predictor_rgb_3x3_jpeg, lossless_restart_predictor_ycbcr_16bit_3x3_jpeg,
    lossless_restart_predictor_ycbcr_3x3_jpeg, lossless_rgb_16bit_420_4x4_jpeg,
    lossless_rgb_16bit_420_4x4_rgb16, lossless_rgb_16bit_420_restart_4x4_jpeg,
    lossless_rgb_16bit_422_4x2_jpeg, lossless_rgb_16bit_422_4x2_rgb16,
    lossless_rgb_16bit_422_restart_4x2_jpeg, lossless_rgb_8bit_420_4x4_jpeg,
    lossless_rgb_8bit_420_4x4_rgb8, lossless_rgb_8bit_420_restart_4x4_jpeg,
    lossless_rgb_8bit_422_4x2_jpeg, lossless_rgb_8bit_422_4x2_rgb8,
    lossless_rgb_8bit_422_restart_4x2_jpeg, lossless_ycbcr_16bit_3x3_rgb16,
    lossless_ycbcr_16bit_420_4x4_jpeg, lossless_ycbcr_16bit_420_4x4_rgb16,
    lossless_ycbcr_16bit_420_restart_4x4_jpeg, lossless_ycbcr_16bit_422_4x2_jpeg,
    lossless_ycbcr_16bit_422_4x2_rgb16, lossless_ycbcr_16bit_422_restart_4x2_jpeg,
    lossless_ycbcr_3x3_rgb8, lossless_ycbcr_8bit_420_4x4_jpeg, lossless_ycbcr_8bit_420_4x4_rgb8,
    lossless_ycbcr_8bit_420_restart_4x4_jpeg, lossless_ycbcr_8bit_422_4x2_jpeg,
    lossless_ycbcr_8bit_422_4x2_rgb8, lossless_ycbcr_8bit_422_restart_4x2_jpeg,
    malformed_cmyk_nondivisible_sampling_jpeg, minimal_baseline_420_jpeg,
    progressive_12bit_cmyk_nonconstant_8x8_jpeg, progressive_12bit_grayscale_8x8_jpeg,
    progressive_12bit_rgb_8x8_jpeg, progressive_12bit_ycck_nonconstant_8x8_jpeg,
    progressive_8x8_jpeg, rgb_app14_8x8_jpeg, rgb_app14_8x8_rgb, ycck_16x16_420_jpeg,
    ycck_16x8_422_jpeg, ycck_16x8_nonleading_max_422_jpeg, ycck_8x8_jpeg,
    LOSSLESS_GRAYSCALE_16BIT_3X3_PIXELS, LOSSLESS_GRAYSCALE_3X3_PIXELS,
    LOSSLESS_RGB_16BIT_3X3_PIXELS, LOSSLESS_RGB_3X3_PIXELS,
};
use fixtures::{
    extended_12bit_cmyk_16x16_420_jpeg, extended_12bit_cmyk_16x8_422_jpeg,
    extended_12bit_cmyk_420_restart_32x16_jpeg, extended_12bit_cmyk_422_restart_32x8_jpeg,
    extended_12bit_cmyk_restart_16x8_jpeg, extended_12bit_rgb_32x32_rgb16,
    extended_12bit_rgb_32x8_rgb16, extended_12bit_rgb_420_32x32_jpeg,
    extended_12bit_rgb_422_32x8_jpeg, extended_12bit_rgb_8x8_jpeg, extended_12bit_rgb_8x8_rgb16,
    extended_12bit_rgb_restart_16x8_jpeg, extended_12bit_rgb_restart_16x8_rgb16,
    extended_12bit_ycbcr_420_32x32_jpeg, extended_12bit_ycbcr_420_32x32_rgb16,
    extended_12bit_ycbcr_420_restart_32x32_jpeg, extended_12bit_ycbcr_420_restart_32x32_rgb16,
    extended_12bit_ycbcr_422_32x8_jpeg, extended_12bit_ycbcr_422_32x8_rgb16,
    extended_12bit_ycbcr_422_restart_32x8_jpeg, extended_12bit_ycbcr_422_restart_32x8_rgb16,
    extended_12bit_ycbcr_8x8_jpeg, extended_12bit_ycbcr_8x8_rgb16,
    extended_12bit_ycbcr_restart_16x8_jpeg, extended_12bit_ycbcr_restart_16x8_rgb16,
    extended_12bit_ycck_16x16_420_jpeg, extended_12bit_ycck_16x8_422_jpeg,
    extended_12bit_ycck_420_restart_32x16_jpeg, extended_12bit_ycck_422_restart_32x8_jpeg,
    extended_12bit_ycck_restart_16x8_jpeg, progressive_12bit_cmyk_16x16_420_jpeg,
    progressive_12bit_cmyk_16x8_422_jpeg, progressive_12bit_cmyk_420_restart_32x16_jpeg,
    progressive_12bit_cmyk_422_restart_32x8_jpeg, progressive_12bit_cmyk_8x8_jpeg,
    progressive_12bit_cmyk_restart_16x8_jpeg, progressive_12bit_rgb_420_32x32_jpeg,
    progressive_12bit_rgb_422_32x8_jpeg, progressive_12bit_ycbcr_420_32x32_jpeg,
    progressive_12bit_ycbcr_422_32x8_jpeg, progressive_12bit_ycbcr_8x8_jpeg,
    progressive_12bit_ycck_16x16_420_jpeg, progressive_12bit_ycck_16x8_422_jpeg,
    progressive_12bit_ycck_420_restart_32x16_jpeg, progressive_12bit_ycck_422_restart_32x8_jpeg,
    progressive_12bit_ycck_8x8_jpeg, progressive_12bit_ycck_restart_16x8_jpeg,
};
use j2k_test_support as fixtures;
use j2k_test_support::{
    crop_interleaved_u16, crop_interleaved_u8, project_scaled_interleaved_u16,
    project_scaled_interleaved_u8, rgb8_to_rgba8, scaled_rect_covering, PixelRect,
};

#[path = "decode_into/color.rs"]
mod color;
#[path = "decode_into/lossless.rs"]
mod lossless;

#[test]
fn decode_into_rgb8_returns_decoded_rect_full_image() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).expect("baseline 4:2:0 must construct");
    let (w, h) = dec.info().dimensions;
    let mut buf = vec![0u8; (w * h * 3) as usize];
    let outcome = dec
        .decode_into(&mut buf, (w * 3) as usize, PixelFormat::Rgb8)
        .expect("baseline 4:2:0 decode must succeed");
    assert_eq!(outcome.decoded.w, w);
    assert_eq!(outcome.decoded.h, h);
}

#[test]
fn decode_owned_rgb8_matches_decode_into() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).expect("baseline 4:2:0 must construct");
    let (w, h) = dec.info().dimensions;
    let mut expected = vec![0u8; (w * h * 3) as usize];
    let expected_outcome = dec
        .decode_into(&mut expected, (w * 3) as usize, PixelFormat::Rgb8)
        .expect("baseline 4:2:0 decode must succeed");

    let (owned, outcome) = dec
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .unwrap();
    assert_eq!(owned, expected);
    assert_eq!(outcome, expected_outcome);
}

#[test]
fn decode_into_rgba8_defaults_alpha_to_255() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let (w, h) = dec.info().dimensions;
    let mut buf = vec![0u8; (w * h * 4) as usize];
    dec.decode_into(&mut buf, (w * 4) as usize, PixelFormat::Rgba8)
        .unwrap();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * 4;
            assert_eq!(buf[idx + 3], 255, "pixel ({x},{y}) alpha");
        }
    }
}

#[test]
fn decode_owned_region_scaled_matches_decode_region_into() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let roi = Rect {
        x: 2,
        y: 2,
        w: 4,
        h: 4,
    };
    let mut expected = vec![0u8; 2 * 2 * 3];
    let expected_outcome = dec
        .decode_region_scaled_into(
            &mut expected,
            2 * 3,
            PixelFormat::Rgb8,
            roi,
            Downscale::Half,
        )
        .unwrap();

    let (owned, outcome) = dec
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            roi,
            Downscale::Half,
        ))
        .unwrap();
    assert_eq!(owned, expected);
    assert_eq!(outcome, expected_outcome);
}

#[test]
fn decode_request_region_scaled_is_repeatable() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let roi = Rect {
        x: 2,
        y: 2,
        w: 4,
        h: 4,
    };

    let first = dec
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            roi,
            Downscale::Half,
        ))
        .unwrap();
    let second = dec
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            roi,
            Downscale::Half,
        ))
        .unwrap();

    assert_eq!(second, first);
}

#[test]
fn decode_owned_scaled_matches_decode_scaled_into() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let mut expected = vec![0u8; 4 * 4 * 3];
    let expected_outcome = dec
        .decode_scaled_into(&mut expected, 4 * 3, PixelFormat::Rgb8, Downscale::Half)
        .unwrap();

    let (owned, outcome) = dec
        .decode_request(DecodeRequest::scaled(PixelFormat::Rgb8, Downscale::Half))
        .unwrap();
    assert_eq!(owned, expected);
    assert_eq!(outcome, expected_outcome);
}

fn request_output_dimensions(request: DecodeRequest, full: (u32, u32)) -> (u32, u32) {
    let rect = request.region.unwrap_or(Rect::full(full));
    let denominator = request.scale.denominator();
    let x_end = rect.x.checked_add(rect.w).expect("fixture x extent");
    let y_end = rect.y.checked_add(rect.h).expect("fixture y extent");
    (
        x_end.div_ceil(denominator) - rect.x / denominator,
        y_end.div_ceil(denominator) - rect.y / denominator,
    )
}

fn assert_owned_request_matches_caller_owned(decoder: &Decoder<'_>, request: DecodeRequest) {
    let output_dimensions = request_output_dimensions(request, decoder.info().dimensions);
    let stride = usize::try_from(output_dimensions.0)
        .expect("fixture width fits usize")
        .checked_mul(request.fmt.bytes_per_pixel())
        .expect("fixture stride");
    let len = stride
        .checked_mul(usize::try_from(output_dimensions.1).expect("fixture height fits usize"))
        .expect("fixture output length");
    let mut expected = vec![0xA5; len];
    let expected_outcome = match (request.region, request.scale) {
        (None, Downscale::None) => decoder.decode_into(&mut expected, stride, request.fmt),
        (None, scale) => decoder.decode_scaled_into(&mut expected, stride, request.fmt, scale),
        (Some(roi), Downscale::None) => {
            decoder.decode_region_into(&mut expected, stride, request.fmt, roi)
        }
        (Some(roi), scale) => {
            decoder.decode_region_scaled_into(&mut expected, stride, request.fmt, roi, scale)
        }
    }
    .expect("caller-owned decode succeeds");

    let (owned, outcome) = decoder
        .decode_request(request)
        .expect("owned decode succeeds");
    assert_eq!(owned.len(), len, "request={request:?}");
    assert_eq!(owned, expected, "request={request:?}");
    assert_eq!(outcome, expected_outcome, "request={request:?}");
}

fn assert_owned_request_shapes(bytes: &[u8], formats: &[PixelFormat], roi: Rect) {
    let decoder = Decoder::new(bytes).expect("owned-output fixture decoder");
    for &format in formats {
        for request in [
            DecodeRequest::full(format),
            DecodeRequest::scaled(format, Downscale::Half),
            DecodeRequest::region(format, roi),
            DecodeRequest::region_scaled(format, roi, Downscale::Half),
        ] {
            assert_owned_request_matches_caller_owned(&decoder, request);
        }
    }
}

#[test]
fn owned_output_matches_prefilled_caller_owned_across_codec_routes() {
    assert_owned_request_shapes(
        &minimal_baseline_420_jpeg(),
        &[PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8],
        Rect {
            x: 4,
            y: 4,
            w: 8,
            h: 8,
        },
    );
    assert_owned_request_shapes(
        &extended_12bit_grayscale_8x8_jpeg(),
        &[PixelFormat::Gray16, PixelFormat::Rgb16, PixelFormat::Rgba16],
        Rect {
            x: 2,
            y: 2,
            w: 4,
            h: 4,
        },
    );
    assert_owned_request_shapes(
        &progressive_12bit_grayscale_8x8_jpeg(),
        &[PixelFormat::Gray16, PixelFormat::Rgb16, PixelFormat::Rgba16],
        Rect {
            x: 2,
            y: 2,
            w: 4,
            h: 4,
        },
    );

    let lossless_roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    assert_owned_request_shapes(
        &lossless_predictor_grayscale_3x3_jpeg(4),
        &[PixelFormat::Gray8],
        lossless_roi,
    );
    for bytes in [
        lossless_predictor_rgb_3x3_jpeg(4),
        lossless_predictor_ycbcr_3x3_jpeg(4),
    ] {
        assert_owned_request_shapes(
            &bytes,
            &[PixelFormat::Rgb8, PixelFormat::Rgba8],
            lossless_roi,
        );
    }
    assert_owned_request_shapes(
        &lossless_predictor_grayscale_16bit_3x3_jpeg(4),
        &[PixelFormat::Gray16],
        lossless_roi,
    );
    for bytes in [
        lossless_predictor_rgb_16bit_3x3_jpeg(4),
        lossless_predictor_ycbcr_16bit_3x3_jpeg(4),
    ] {
        assert_owned_request_shapes(
            &bytes,
            &[PixelFormat::Rgb16, PixelFormat::Rgba16],
            lossless_roi,
        );
    }
}

#[test]
fn full_tile_region_scaled_matches_full_scaled_decode() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let (w, h) = dec.info().dimensions;
    let roi = Rect { x: 0, y: 0, w, h };
    let stride = w.div_ceil(4) as usize * 3;
    let mut expected = vec![0u8; stride * h.div_ceil(4) as usize];
    let expected_outcome = dec
        .decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, Downscale::Quarter)
        .unwrap();
    let mut actual = vec![0u8; expected.len()];

    let actual_outcome = dec
        .decode_region_scaled_into(
            &mut actual,
            stride,
            PixelFormat::Rgb8,
            roi,
            Downscale::Quarter,
        )
        .unwrap();

    assert_eq!(actual, expected);
    assert_eq!(actual_outcome, expected_outcome);
    assert_eq!(actual_outcome.decoded, roi);
}

#[test]
fn decode_into_gray8_produces_single_byte_per_pixel() {
    let bytes = grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let (w, h) = dec.info().dimensions;
    assert_eq!((w, h), (8, 8));
    let mut buf = vec![0u8; (w * h) as usize];
    let outcome = dec
        .decode_into(&mut buf, w as usize, PixelFormat::Gray8)
        .unwrap();
    assert_eq!(outcome.decoded.w, 8);
    assert!(buf.iter().any(|&b| b != 0), "expected non-zero pixels");
}

#[test]
fn decode_into_gray16_accepts_extended12_grayscale_samples() {
    let bytes = extended_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended grayscale JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Gray16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Gray16)
        .expect("12-bit grayscale decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    for sample in buf.chunks_exact(2) {
        assert_eq!(u16::from_le_bytes([sample[0], sample[1]]), 2048);
    }
}

#[test]
fn decode_region_into_gray16_crops_extended12_grayscale_samples() {
    let bytes = extended_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended grayscale JPEG must construct");
    let roi = Rect {
        x: 2,
        y: 1,
        w: 3,
        h: 4,
    };
    let stride = roi.w as usize * PixelFormat::Gray16.bytes_per_pixel() + 4;
    let mut buf = vec![0xaau8; stride * roi.h as usize];

    let outcome = dec
        .decode_region_into(&mut buf, stride, PixelFormat::Gray16, roi)
        .expect("12-bit grayscale ROI decode must succeed");

    assert_eq!(outcome.decoded, roi);
    for row in buf.chunks_exact(stride) {
        for sample in row[..roi.w as usize * 2].chunks_exact(2) {
            assert_eq!(u16::from_le_bytes([sample[0], sample[1]]), 2048);
        }
        assert_eq!(&row[roi.w as usize * 2..], &[0xaa; 4]);
    }
}

#[test]
fn decode_scaled_into_gray16_projects_extended12_grayscale_samples() {
    let bytes = extended_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended grayscale JPEG must construct");
    let scale = Downscale::Half;
    let scaled_w = 4usize;
    let scaled_h = 4usize;
    let stride = scaled_w * PixelFormat::Gray16.bytes_per_pixel() + 4;
    let mut buf = vec![0xaau8; stride * scaled_h];

    let outcome = dec
        .decode_scaled_into(&mut buf, stride, PixelFormat::Gray16, scale)
        .expect("12-bit grayscale scaled decode must succeed");

    assert_eq!(outcome.decoded, Rect::full(dec.info().dimensions));
    for row in buf.chunks_exact(stride) {
        for sample in row[..scaled_w * 2].chunks_exact(2) {
            assert_eq!(u16::from_le_bytes([sample[0], sample[1]]), 2048);
        }
        assert_eq!(&row[scaled_w * 2..], &[0xaa; 4]);
    }
}

#[test]
fn decode_into_gray16_accepts_extended12_restart_grayscale_samples() {
    let bytes = extended_12bit_grayscale_restart_16x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended restart grayscale JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Gray16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Gray16)
        .expect("12-bit restart grayscale decode must succeed");

    assert_eq!(dec.info().restart_interval, Some(1));
    assert_eq!(outcome.decoded, Rect::full((w, h)));
    for sample in buf.chunks_exact(2) {
        assert_eq!(u16::from_le_bytes([sample[0], sample[1]]), 2048);
    }
}

#[test]
fn decode_region_scaled_into_gray16_accepts_extended12_restart_grayscale_samples() {
    let bytes = extended_12bit_grayscale_restart_16x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended restart grayscale JPEG must construct");
    let roi = Rect {
        x: 2,
        y: 1,
        w: 12,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Gray16.bytes_per_pixel() + 4;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Gray16, roi, Downscale::Half)
        .expect("12-bit restart grayscale region-scaled decode must succeed");

    assert_eq!(dec.info().restart_interval, Some(1));
    assert_eq!(outcome.decoded, roi);
    for row in buf.chunks_exact(stride) {
        for sample in row[..scaled_roi.w as usize * 2].chunks_exact(2) {
            assert_eq!(u16::from_le_bytes([sample[0], sample[1]]), 2048);
        }
        assert_eq!(&row[scaled_roi.w as usize * 2..], &[0xaa; 4]);
    }
}

#[test]
fn decode_region_scaled_into_rgb16_accepts_extended12_restart_grayscale_samples() {
    let bytes = extended_12bit_grayscale_restart_16x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended restart grayscale JPEG must construct");
    let roi = Rect {
        x: 2,
        y: 1,
        w: 12,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit restart grayscale region-scaled Rgb16 decode must succeed");

    assert_eq!(dec.info().restart_interval, Some(1));
    assert_eq!(outcome.decoded, roi);
    for row in buf.chunks_exact(stride) {
        for pixel in row[..scaled_roi.w as usize * 6].chunks_exact(6) {
            let channels = [
                u16::from_le_bytes([pixel[0], pixel[1]]),
                u16::from_le_bytes([pixel[2], pixel[3]]),
                u16::from_le_bytes([pixel[4], pixel[5]]),
            ];
            assert_eq!(channels, [2048; 3]);
        }
        assert_eq!(&row[scaled_roi.w as usize * 6..], &[0xaa; 6]);
    }
}

#[test]
fn decode_into_gray16_accepts_progressive12_grayscale_samples() {
    let bytes = progressive_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive grayscale JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Gray16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Gray16)
        .expect("12-bit progressive Gray16 decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    for sample in buf.chunks_exact(2) {
        assert_eq!(u16::from_le_bytes([sample[0], sample[1]]), 2048);
    }
}

#[test]
fn decode_region_into_gray16_crops_progressive12_grayscale_samples() {
    let bytes = progressive_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive grayscale JPEG must construct");
    let roi = Rect {
        x: 2,
        y: 1,
        w: 3,
        h: 4,
    };
    let stride = roi.w as usize * PixelFormat::Gray16.bytes_per_pixel() + 4;
    let mut buf = vec![0xaau8; stride * roi.h as usize];

    let outcome = dec
        .decode_region_into(&mut buf, stride, PixelFormat::Gray16, roi)
        .expect("12-bit progressive Gray16 ROI decode must succeed");

    assert_eq!(outcome.decoded, roi);
    for row in buf.chunks_exact(stride) {
        for sample in row[..roi.w as usize * 2].chunks_exact(2) {
            assert_eq!(u16::from_le_bytes([sample[0], sample[1]]), 2048);
        }
        assert_eq!(&row[roi.w as usize * 2..], &[0xaa; 4]);
    }
}

#[test]
fn decode_scaled_into_gray16_projects_progressive12_grayscale_samples() {
    let bytes = progressive_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive grayscale JPEG must construct");
    let scaled_w = 4usize;
    let scaled_h = 4usize;
    let stride = scaled_w * PixelFormat::Gray16.bytes_per_pixel() + 4;
    let mut buf = vec![0xaau8; stride * scaled_h];

    let outcome = dec
        .decode_scaled_into(&mut buf, stride, PixelFormat::Gray16, Downscale::Half)
        .expect("12-bit progressive Gray16 scaled decode must succeed");

    assert_eq!(outcome.decoded, Rect::full(dec.info().dimensions));
    for row in buf.chunks_exact(stride) {
        for sample in row[..scaled_w * 2].chunks_exact(2) {
            assert_eq!(u16::from_le_bytes([sample[0], sample[1]]), 2048);
        }
        assert_eq!(&row[scaled_w * 2..], &[0xaa; 4]);
    }
}

#[test]
fn decode_region_scaled_into_gray16_projects_progressive12_grayscale_samples() {
    let bytes = progressive_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive grayscale JPEG must construct");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Gray16.bytes_per_pixel() + 4;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Gray16, roi, Downscale::Half)
        .expect("12-bit progressive Gray16 region-scaled decode must succeed");

    assert_eq!(outcome.decoded, roi);
    for row in buf.chunks_exact(stride) {
        for sample in row[..scaled_roi.w as usize * 2].chunks_exact(2) {
            assert_eq!(u16::from_le_bytes([sample[0], sample[1]]), 2048);
        }
        assert_eq!(&row[scaled_roi.w as usize * 2..], &[0xaa; 4]);
    }
}

#[test]
fn decode_into_rgb16_expands_progressive12_grayscale_samples() {
    let bytes = progressive_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive grayscale JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit progressive Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    for pixel in buf.chunks_exact(6) {
        let channels = [
            u16::from_le_bytes([pixel[0], pixel[1]]),
            u16::from_le_bytes([pixel[2], pixel[3]]),
            u16::from_le_bytes([pixel[4], pixel[5]]),
        ];
        assert_eq!(channels, [2048; 3]);
    }
}

#[test]
fn decode_region_scaled_into_rgb16_projects_progressive12_grayscale_samples() {
    let bytes = progressive_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive grayscale JPEG must construct");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit progressive region-scaled Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, roi);
    for row in buf.chunks_exact(stride) {
        for pixel in row[..scaled_roi.w as usize * 6].chunks_exact(6) {
            let channels = [
                u16::from_le_bytes([pixel[0], pixel[1]]),
                u16::from_le_bytes([pixel[2], pixel[3]]),
                u16::from_le_bytes([pixel[4], pixel[5]]),
            ];
            assert_eq!(channels, [2048; 3]);
        }
        assert_eq!(&row[scaled_roi.w as usize * 6..], &[0xaa; 6]);
    }
}

#[test]
fn decode_region_scaled_into_rgba16_projects_12bit_grayscale_samples() {
    for (bytes, roi, label) in [
        (
            extended_12bit_grayscale_restart_16x8_jpeg(),
            Rect {
                x: 2,
                y: 1,
                w: 12,
                h: 6,
            },
            "extended restart",
        ),
        (
            progressive_12bit_grayscale_8x8_jpeg(),
            Rect {
                x: 1,
                y: 1,
                w: 6,
                h: 6,
            },
            "progressive",
        ),
    ] {
        let dec = Decoder::new(&bytes)
            .unwrap_or_else(|err| panic!("12-bit {label} grayscale JPEG must construct: {err}"));
        let scaled_roi = scaled_rect_covering_for_test(roi, 2);
        let row_bytes = scaled_roi.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
        let stride = row_bytes + 8;
        let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgba16, roi, Downscale::Half)
            .unwrap_or_else(|err| {
                panic!("12-bit {label} grayscale RGBA16 region-scaled decode must succeed: {err}")
            });

        assert_eq!(outcome.decoded, roi, "{label}");
        for row in buf.chunks_exact(stride) {
            for pixel in row[..row_bytes].chunks_exact(8) {
                let channels = [
                    u16::from_le_bytes([pixel[0], pixel[1]]),
                    u16::from_le_bytes([pixel[2], pixel[3]]),
                    u16::from_le_bytes([pixel[4], pixel[5]]),
                    u16::from_le_bytes([pixel[6], pixel[7]]),
                ];
                assert_eq!(channels, [2048, 2048, 2048, u16::MAX], "{label}");
            }
            assert_eq!(&row[row_bytes..], &[0xaa; 8], "{label}");
        }
    }
}

#[test]
fn decode_into_rgb16_accepts_progressive12_app14_rgb_samples() {
    let bytes = progressive_12bit_rgb_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive APP14 RGB JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit progressive APP14 RGB decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_eq!(buf, extended_12bit_rgb_8x8_rgb16());
}

#[test]
fn decode_region_scaled_into_rgb16_projects_progressive12_app14_rgb_samples() {
    let bytes = progressive_12bit_rgb_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive APP14 RGB JPEG must construct");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];
    let expected_pixel = [2064u16, 2072, 2032]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit progressive APP14 RGB region-scaled decode must succeed");

    assert_eq!(outcome.decoded, roi);
    for row in buf.chunks_exact(stride) {
        for pixel in row[..scaled_roi.w as usize * 6].chunks_exact(6) {
            assert_eq!(pixel, expected_pixel.as_slice());
        }
        assert_eq!(&row[scaled_roi.w as usize * 6..], &[0xaa; 6]);
    }
}

#[test]
fn decode_into_rgb16_converts_progressive12_ycbcr444_samples() {
    let bytes = progressive_12bit_ycbcr_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive YCbCr JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit progressive YCbCr Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_eq!(buf, extended_12bit_ycbcr_8x8_rgb16());
}

#[test]
fn decode_region_scaled_into_rgb16_converts_progressive12_ycbcr444_samples() {
    let bytes = progressive_12bit_ycbcr_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive YCbCr JPEG must construct");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];
    let expected_pixel = [2042u16, 2067, 2107]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit progressive YCbCr region-scaled Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, roi);
    for row in buf.chunks_exact(stride) {
        for pixel in row[..scaled_roi.w as usize * 6].chunks_exact(6) {
            assert_eq!(pixel, expected_pixel.as_slice());
        }
        assert_eq!(&row[scaled_roi.w as usize * 6..], &[0xaa; 6]);
    }
}

#[test]
fn decode_into_rgb16_converts_progressive12_ycbcr422_samples() {
    let bytes = progressive_12bit_ycbcr_422_32x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive YCbCr 4:2:2 JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit progressive YCbCr 4:2:2 Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_eq!(buf, extended_12bit_ycbcr_422_32x8_rgb16());
}

#[test]
fn decode_region_scaled_into_rgb16_converts_progressive12_ycbcr422_samples() {
    let bytes = progressive_12bit_ycbcr_422_32x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive YCbCr 4:2:2 JPEG must construct");
    let roi = Rect {
        x: 13,
        y: 0,
        w: 8,
        h: 4,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];
    let full = extended_12bit_ycbcr_422_32x8_rgb16();
    let expected_pixels = expected_scaled_rgb16_pixels(&full, 32, roi, 2);

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit progressive YCbCr 4:2:2 region-scaled Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, roi);
    assert_padded_rgb16_rows(&buf, stride, scaled_roi.w as usize, &expected_pixels);
}

#[test]
fn decode_into_rgb16_converts_progressive12_ycbcr420_samples() {
    let bytes = progressive_12bit_ycbcr_420_32x32_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive YCbCr 4:2:0 JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit progressive YCbCr 4:2:0 Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_rgb16_image_eq(&buf, &extended_12bit_ycbcr_420_32x32_rgb16(), w as usize);
}

#[test]
fn decode_region_scaled_into_rgb16_converts_progressive12_ycbcr420_samples() {
    let bytes = progressive_12bit_ycbcr_420_32x32_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive YCbCr 4:2:0 JPEG must construct");
    let roi = Rect {
        x: 13,
        y: 14,
        w: 10,
        h: 10,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];
    let full = extended_12bit_ycbcr_420_32x32_rgb16();
    let expected_pixels = expected_scaled_rgb16_pixels(&full, 32, roi, 2);

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit progressive YCbCr 4:2:0 region-scaled Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, roi);
    assert_padded_rgb16_rows(&buf, stride, scaled_roi.w as usize, &expected_pixels);
}

#[test]
fn decode_region_scaled_into_rgba16_projects_progressive12_color_samples() {
    for (bytes, full, full_width, roi, label) in [
        (
            progressive_12bit_rgb_8x8_jpeg(),
            extended_12bit_rgb_8x8_rgb16(),
            8,
            Rect {
                x: 1,
                y: 1,
                w: 6,
                h: 6,
            },
            "APP14 RGB",
        ),
        (
            progressive_12bit_ycbcr_420_32x32_jpeg(),
            extended_12bit_ycbcr_420_32x32_rgb16(),
            32,
            Rect {
                x: 13,
                y: 14,
                w: 10,
                h: 10,
            },
            "YCbCr 4:2:0",
        ),
    ] {
        let dec = Decoder::new(&bytes)
            .unwrap_or_else(|err| panic!("12-bit progressive {label} JPEG must construct: {err}"));
        let scaled_roi = scaled_rect_covering_for_test(roi, 2);
        let row_bytes = scaled_roi.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
        let stride = row_bytes + 8;
        let expected_rgb = expected_scaled_rgb16_pixels(&full, full_width, roi, 2);
        let expected = rgb16_to_rgba16(&expected_rgb, u16::MAX);
        let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgba16, roi, Downscale::Half)
            .unwrap_or_else(|err| {
                panic!("12-bit progressive {label} RGBA16 region-scaled decode must succeed: {err}")
            });

        assert_eq!(outcome.decoded, roi, "{label}");
        assert_padded_rgba16_rows(&buf, stride, scaled_roi.w as usize, &expected);
    }
}

#[test]
fn decode_into_rgb16_expands_extended12_grayscale_samples() {
    let bytes = extended_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended grayscale JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit grayscale Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    for pixel in buf.chunks_exact(6) {
        let channels = [
            u16::from_le_bytes([pixel[0], pixel[1]]),
            u16::from_le_bytes([pixel[2], pixel[3]]),
            u16::from_le_bytes([pixel[4], pixel[5]]),
        ];
        assert_eq!(channels, [2048; 3]);
    }
}

#[test]
fn decode_region_into_rgb16_crops_extended12_grayscale_samples() {
    let bytes = extended_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended grayscale JPEG must construct");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 4,
        h: 3,
    };
    let stride = roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * roi.h as usize];

    let outcome = dec
        .decode_region_into(&mut buf, stride, PixelFormat::Rgb16, roi)
        .expect("12-bit grayscale Rgb16 ROI decode must succeed");

    assert_eq!(outcome.decoded, roi);
    for row in buf.chunks_exact(stride) {
        for pixel in row[..roi.w as usize * 6].chunks_exact(6) {
            let channels = [
                u16::from_le_bytes([pixel[0], pixel[1]]),
                u16::from_le_bytes([pixel[2], pixel[3]]),
                u16::from_le_bytes([pixel[4], pixel[5]]),
            ];
            assert_eq!(channels, [2048; 3]);
        }
        assert_eq!(&row[roi.w as usize * 6..], &[0xaa; 6]);
    }
}

#[test]
fn decode_region_scaled_into_rgb16_projects_extended12_grayscale_samples() {
    let bytes = extended_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended grayscale JPEG must construct");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit grayscale region-scaled Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, roi);
    for row in buf.chunks_exact(stride) {
        for pixel in row[..scaled_roi.w as usize * 6].chunks_exact(6) {
            let channels = [
                u16::from_le_bytes([pixel[0], pixel[1]]),
                u16::from_le_bytes([pixel[2], pixel[3]]),
                u16::from_le_bytes([pixel[4], pixel[5]]),
            ];
            assert_eq!(channels, [2048; 3]);
        }
        assert_eq!(&row[scaled_roi.w as usize * 6..], &[0xaa; 6]);
    }
}

#[test]
fn decode_into_rgba16_accepts_extended12_color_samples() {
    for (bytes, expected_rgb, label) in [
        (
            extended_12bit_rgb_8x8_jpeg(),
            extended_12bit_rgb_8x8_rgb16(),
            "APP14 RGB",
        ),
        (
            extended_12bit_ycbcr_420_32x32_jpeg(),
            extended_12bit_ycbcr_420_32x32_rgb16(),
            "YCbCr 4:2:0",
        ),
    ] {
        let dec = Decoder::new(&bytes)
            .unwrap_or_else(|err| panic!("12-bit extended {label} JPEG must construct: {err}"));
        let (w, h) = dec.info().dimensions;
        let stride = w as usize * PixelFormat::Rgba16.bytes_per_pixel();
        let mut buf = vec![0u8; stride * h as usize];

        let outcome = dec
            .decode_into(&mut buf, stride, PixelFormat::Rgba16)
            .unwrap_or_else(|err| {
                panic!("12-bit extended {label} RGBA16 decode must succeed: {err}")
            });

        assert_eq!(outcome.decoded, Rect::full((w, h)), "{label}");
        assert_eq!(buf, rgb16_to_rgba16(&expected_rgb, u16::MAX), "{label}");
    }
}

#[test]
fn decode_into_rgb16_accepts_extended12_app14_rgb_samples() {
    let bytes = extended_12bit_rgb_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended APP14 RGB JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit APP14 RGB decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_eq!(buf, extended_12bit_rgb_8x8_rgb16());
}

#[test]
fn decode_into_rgb16_accepts_extended12_restart_app14_rgb_samples() {
    let bytes = extended_12bit_rgb_restart_16x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended restart APP14 RGB JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit restart APP14 RGB decode must succeed");

    assert_eq!(dec.info().restart_interval, Some(1));
    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_eq!(buf, extended_12bit_rgb_restart_16x8_rgb16());
}

#[test]
fn decode_region_scaled_into_rgb16_projects_extended12_app14_rgb_samples() {
    let bytes = extended_12bit_rgb_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended APP14 RGB JPEG must construct");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];
    let expected_pixel = [2064u16, 2072, 2032]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit APP14 RGB region-scaled decode must succeed");

    assert_eq!(outcome.decoded, roi);
    for row in buf.chunks_exact(stride) {
        for pixel in row[..scaled_roi.w as usize * 6].chunks_exact(6) {
            assert_eq!(pixel, expected_pixel.as_slice());
        }
        assert_eq!(&row[scaled_roi.w as usize * 6..], &[0xaa; 6]);
    }
}

#[test]
fn decode_region_scaled_into_rgba16_projects_extended12_restart_color_samples() {
    for (bytes, full, full_width, roi, label) in [
        (
            extended_12bit_rgb_restart_16x8_jpeg(),
            extended_12bit_rgb_restart_16x8_rgb16(),
            16,
            Rect {
                x: 2,
                y: 1,
                w: 12,
                h: 6,
            },
            "APP14 RGB",
        ),
        (
            extended_12bit_ycbcr_420_restart_32x32_jpeg(),
            extended_12bit_ycbcr_420_restart_32x32_rgb16(),
            32,
            Rect {
                x: 13,
                y: 14,
                w: 10,
                h: 10,
            },
            "YCbCr 4:2:0",
        ),
    ] {
        let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
            panic!("12-bit extended restart {label} JPEG must construct: {err}")
        });
        let scaled_roi = scaled_rect_covering_for_test(roi, 2);
        let row_bytes = scaled_roi.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
        let stride = row_bytes + 8;
        let expected_rgb = expected_scaled_rgb16_pixels(&full, full_width, roi, 2);
        let expected = rgb16_to_rgba16(&expected_rgb, u16::MAX);
        let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];

        let outcome = dec
            .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgba16, roi, Downscale::Half)
            .unwrap_or_else(|err| {
                panic!(
                    "12-bit extended restart {label} RGBA16 region-scaled decode must succeed: {err}"
                )
            });

        assert_eq!(dec.info().restart_interval, Some(1), "{label}");
        assert_eq!(outcome.decoded, roi, "{label}");
        assert_padded_rgba16_rows(&buf, stride, scaled_roi.w as usize, &expected);
    }
}

#[test]
fn decode_region_scaled_into_rgb16_projects_extended12_restart_app14_rgb_samples() {
    let bytes = extended_12bit_rgb_restart_16x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended restart APP14 RGB JPEG must construct");
    let roi = Rect {
        x: 2,
        y: 1,
        w: 12,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];
    let full = extended_12bit_rgb_restart_16x8_rgb16();
    let expected_pixels = expected_scaled_rgb16_pixels(&full, 16, roi, 2);

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit restart APP14 RGB region-scaled decode must succeed");

    assert_eq!(dec.info().restart_interval, Some(1));
    assert_eq!(outcome.decoded, roi);
    assert_padded_rgb16_rows(&buf, stride, scaled_roi.w as usize, &expected_pixels);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the end-to-end APP14 matrix keeps full, ROI, scaled, and region-scaled assertions correlated"
)]
fn decode_12bit_app14_rgb_subsampled_full_roi_scaled_and_region_scaled_outputs() {
    for (bytes, expected_full, width, height, label) in [
        (
            extended_12bit_rgb_422_32x8_jpeg(),
            extended_12bit_rgb_32x8_rgb16(),
            32,
            8,
            "12-bit extended APP14 RGB 4:2:2",
        ),
        (
            extended_12bit_rgb_420_32x32_jpeg(),
            extended_12bit_rgb_32x32_rgb16(),
            32,
            32,
            "12-bit extended APP14 RGB 4:2:0",
        ),
        (
            progressive_12bit_rgb_422_32x8_jpeg(),
            extended_12bit_rgb_32x8_rgb16(),
            32,
            8,
            "12-bit progressive APP14 RGB 4:2:2",
        ),
        (
            progressive_12bit_rgb_420_32x32_jpeg(),
            extended_12bit_rgb_32x32_rgb16(),
            32,
            32,
            "12-bit progressive APP14 RGB 4:2:0",
        ),
    ] {
        let dec = Decoder::new(&bytes)
            .unwrap_or_else(|err| panic!("{label} decoder should construct: {err}"));
        let full_rect = Rect::full((width, height));

        let mut full =
            vec![0u8; width as usize * height as usize * PixelFormat::Rgb16.bytes_per_pixel()];
        let outcome = dec
            .decode_into(
                &mut full,
                width as usize * PixelFormat::Rgb16.bytes_per_pixel(),
                PixelFormat::Rgb16,
            )
            .unwrap_or_else(|err| panic!("{label} RGB16 full decode should succeed: {err}"));
        assert_eq!(outcome.decoded, full_rect, "{label}");
        assert_eq!(full, expected_full, "{label}");

        let roi = Rect {
            x: 3,
            y: 1,
            w: 11,
            h: 6,
        };
        let expected_roi = crop_rgb16_bytes(&expected_full, width as usize, roi);
        let mut roi_buf = vec![0u8; roi.w as usize * roi.h as usize * 6];
        let outcome = dec
            .decode_region_into(
                &mut roi_buf,
                roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel(),
                PixelFormat::Rgb16,
                roi,
            )
            .unwrap_or_else(|err| panic!("{label} RGB16 ROI decode should succeed: {err}"));
        assert_eq!(outcome.decoded, roi, "{label}");
        assert_eq!(roi_buf, expected_roi, "{label}");

        let scaled_rect = scaled_rect_covering_for_test(full_rect, 2);
        let scaled_row_bytes = scaled_rect.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
        let scaled_stride = scaled_row_bytes + 8;
        let expected_scaled = rgb16_to_rgba16(
            &expected_scaled_rgb16_pixels(&expected_full, width as usize, full_rect, 2),
            u16::MAX,
        );
        let mut scaled = vec![0xaau8; scaled_stride * scaled_rect.h as usize];
        let outcome = dec
            .decode_scaled_into(
                &mut scaled,
                scaled_stride,
                PixelFormat::Rgba16,
                Downscale::Half,
            )
            .unwrap_or_else(|err| panic!("{label} RGBA16 scaled decode should succeed: {err}"));
        assert_eq!(outcome.decoded, full_rect, "{label}");
        assert_padded_rgba16_rows(
            &scaled,
            scaled_stride,
            scaled_rect.w as usize,
            &expected_scaled,
        );

        let region_scaled = scaled_rect_covering_for_test(roi, 2);
        let region_scaled_row_bytes =
            region_scaled.w as usize * PixelFormat::Rgba16.bytes_per_pixel();
        let region_scaled_stride = region_scaled_row_bytes + 8;
        let expected_region_scaled = rgb16_to_rgba16(
            &expected_scaled_rgb16_pixels(&expected_full, width as usize, roi, 2),
            u16::MAX,
        );
        let mut region_scaled_buf = vec![0xaau8; region_scaled_stride * region_scaled.h as usize];
        let outcome = dec
            .decode_region_scaled_into(
                &mut region_scaled_buf,
                region_scaled_stride,
                PixelFormat::Rgba16,
                roi,
                Downscale::Half,
            )
            .unwrap_or_else(|err| {
                panic!("{label} RGBA16 region-scaled decode should succeed: {err}")
            });
        assert_eq!(outcome.decoded, roi, "{label}");
        assert_padded_rgba16_rows(
            &region_scaled_buf,
            region_scaled_stride,
            region_scaled.w as usize,
            &expected_region_scaled,
        );
    }
}

#[test]
fn decode_into_rgb16_converts_extended12_ycbcr444_samples() {
    let bytes = extended_12bit_ycbcr_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended YCbCr JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit YCbCr Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_eq!(buf, extended_12bit_ycbcr_8x8_rgb16());
}

#[test]
fn decode_into_rgb16_converts_extended12_restart_ycbcr444_samples() {
    let bytes = extended_12bit_ycbcr_restart_16x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended restart YCbCr JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit restart YCbCr Rgb16 decode must succeed");

    assert_eq!(dec.info().restart_interval, Some(1));
    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_eq!(buf, extended_12bit_ycbcr_restart_16x8_rgb16());
}

#[test]
fn decode_region_scaled_into_rgb16_converts_extended12_ycbcr444_samples() {
    let bytes = extended_12bit_ycbcr_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended YCbCr JPEG must construct");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];
    let expected_pixel = [2042u16, 2067, 2107]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit YCbCr region-scaled Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, roi);
    for row in buf.chunks_exact(stride) {
        for pixel in row[..scaled_roi.w as usize * 6].chunks_exact(6) {
            assert_eq!(pixel, expected_pixel.as_slice());
        }
        assert_eq!(&row[scaled_roi.w as usize * 6..], &[0xaa; 6]);
    }
}

#[test]
fn decode_into_rgb16_converts_extended12_ycbcr422_samples() {
    let bytes = extended_12bit_ycbcr_422_32x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended YCbCr 4:2:2 JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit YCbCr 4:2:2 Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_eq!(buf, extended_12bit_ycbcr_422_32x8_rgb16());
}

#[test]
fn decode_into_rgb16_converts_extended12_restart_ycbcr422_samples() {
    let bytes = extended_12bit_ycbcr_422_restart_32x8_jpeg();
    let dec =
        Decoder::new(&bytes).expect("12-bit extended restart YCbCr 4:2:2 JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit restart YCbCr 4:2:2 Rgb16 decode must succeed");

    assert_eq!(dec.info().restart_interval, Some(1));
    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_eq!(buf, extended_12bit_ycbcr_422_restart_32x8_rgb16());
}

#[test]
fn decode_region_scaled_into_rgb16_converts_extended12_ycbcr422_samples() {
    let bytes = extended_12bit_ycbcr_422_32x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended YCbCr 4:2:2 JPEG must construct");
    let roi = Rect {
        x: 13,
        y: 0,
        w: 8,
        h: 4,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];
    let full = extended_12bit_ycbcr_422_32x8_rgb16();
    let expected_pixels = expected_scaled_rgb16_pixels(&full, 32, roi, 2);

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit YCbCr 4:2:2 region-scaled Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, roi);
    assert_padded_rgb16_rows(&buf, stride, scaled_roi.w as usize, &expected_pixels);
}

#[test]
fn decode_into_rgb16_converts_extended12_ycbcr420_samples() {
    let bytes = extended_12bit_ycbcr_420_32x32_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended YCbCr 4:2:0 JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit YCbCr 4:2:0 Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_rgb16_image_eq(&buf, &extended_12bit_ycbcr_420_32x32_rgb16(), w as usize);
}

#[test]
fn decode_into_rgb16_converts_extended12_restart_ycbcr420_samples() {
    let bytes = extended_12bit_ycbcr_420_restart_32x32_jpeg();
    let dec =
        Decoder::new(&bytes).expect("12-bit extended restart YCbCr 4:2:0 JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut buf = vec![0u8; stride * h as usize];

    let outcome = dec
        .decode_into(&mut buf, stride, PixelFormat::Rgb16)
        .expect("12-bit restart YCbCr 4:2:0 Rgb16 decode must succeed");

    assert_eq!(dec.info().restart_interval, Some(1));
    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_rgb16_image_eq(
        &buf,
        &extended_12bit_ycbcr_420_restart_32x32_rgb16(),
        w as usize,
    );
}

#[test]
fn decode_region_scaled_into_rgb16_converts_extended12_ycbcr420_samples() {
    let bytes = extended_12bit_ycbcr_420_32x32_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit extended YCbCr 4:2:0 JPEG must construct");
    let roi = Rect {
        x: 13,
        y: 14,
        w: 10,
        h: 10,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];
    let full = extended_12bit_ycbcr_420_32x32_rgb16();
    let expected_pixels = expected_scaled_rgb16_pixels(&full, 32, roi, 2);

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit YCbCr 4:2:0 region-scaled Rgb16 decode must succeed");

    assert_eq!(outcome.decoded, roi);
    assert_padded_rgb16_rows(&buf, stride, scaled_roi.w as usize, &expected_pixels);
}

#[test]
fn decode_region_scaled_into_rgb16_converts_extended12_restart_ycbcr420_samples() {
    let bytes = extended_12bit_ycbcr_420_restart_32x32_jpeg();
    let dec =
        Decoder::new(&bytes).expect("12-bit extended restart YCbCr 4:2:0 JPEG must construct");
    let roi = Rect {
        x: 13,
        y: 14,
        w: 10,
        h: 10,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel() + 6;
    let mut buf = vec![0xaau8; stride * scaled_roi.h as usize];
    let full = extended_12bit_ycbcr_420_restart_32x32_rgb16();
    let expected_pixels = expected_scaled_rgb16_pixels(&full, 32, roi, 2);

    let outcome = dec
        .decode_region_scaled_into(&mut buf, stride, PixelFormat::Rgb16, roi, Downscale::Half)
        .expect("12-bit restart YCbCr 4:2:0 region-scaled Rgb16 decode must succeed");

    assert_eq!(dec.info().restart_interval, Some(1));
    assert_eq!(outcome.decoded, roi);
    assert_padded_rgb16_rows(&buf, stride, scaled_roi.w as usize, &expected_pixels);
}

#[test]
fn decode_into_rejects_undersized_buffer_with_api_misuse_error() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let mut buf = vec![0u8; 4];
    let err = dec
        .decode_into(&mut buf, 48, PixelFormat::Rgb8)
        .unwrap_err();
    assert!(err.is_api_misuse());
    assert!(matches!(err, JpegError::OutputBufferTooSmall { .. }));
}

#[test]
fn decode_into_rejects_stride_narrower_than_row_width() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let mut buf = vec![0u8; 16 * 16 * 3];
    let err = dec
        .decode_into(&mut buf, 10, PixelFormat::Rgb8)
        .unwrap_err();
    assert!(err.is_api_misuse());
    assert!(matches!(err, JpegError::InvalidStride { .. }));
}

#[test]
fn decode_into_tolerates_padded_stride() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let (w, h) = dec.info().dimensions;
    let padded_stride = (w as usize * 3) + 32;
    let mut buf = vec![0xAAu8; padded_stride * h as usize];
    dec.decode_into(&mut buf, padded_stride, PixelFormat::Rgb8)
        .unwrap();
    let last_row_start = (h as usize - 1) * padded_stride;
    let last_row_end = last_row_start + w as usize * 3;
    assert_eq!(
        &buf[last_row_end..last_row_end + 16],
        &[0xAA; 16],
        "stride padding must not be overwritten"
    );
}

#[test]
fn decode_into_rgb8_preserves_app14_rgb_pixels() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("APP14 RGB fixture must construct");
    let (w, h) = dec.info().dimensions;
    assert_eq!((w, h), (8, 8));
    let mut buf = vec![0u8; (w * h * 3) as usize];
    dec.decode_into(&mut buf, (w * 3) as usize, PixelFormat::Rgb8)
        .expect("APP14 RGB decode must succeed");
    assert_eq!(buf, rgb_app14_8x8_rgb());
}

#[test]
fn decode_into_rgb8_scaled_preserves_constant_app14_rgb_pixels() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();

    for (factor, dims) in [
        (Downscale::Half, (4u32, 4u32)),
        (Downscale::Quarter, (2u32, 2u32)),
        (Downscale::Eighth, (1u32, 1u32)),
    ] {
        let mut buf = vec![0u8; dims.0 as usize * dims.1 as usize * 3];
        dec.decode_scaled_into(&mut buf, dims.0 as usize * 3, PixelFormat::Rgb8, factor)
            .unwrap();
        let mut expected = Vec::with_capacity(buf.len());
        for _ in 0..(dims.0 * dims.1) {
            expected.extend_from_slice(&[200, 20, 10]);
        }
        assert_eq!(buf, expected, "factor={factor:?}");
    }
}

#[test]
fn decode_into_gray8_scaled_projects_constant_app14_rgb_pixels() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let expected = u8::try_from((77 * 200 + 150 * 20 + 29 * 10 + 128) >> 8)
        .expect("weighted grayscale fixture fits in u8");

    for (factor, dims) in [
        (Downscale::Half, (4u32, 4u32)),
        (Downscale::Quarter, (2u32, 2u32)),
        (Downscale::Eighth, (1u32, 1u32)),
    ] {
        let mut buf = vec![0u8; dims.0 as usize * dims.1 as usize];
        dec.decode_scaled_into(&mut buf, dims.0 as usize, PixelFormat::Gray8, factor)
            .unwrap();
        assert!(buf.iter().all(|&px| px == expected), "factor={factor:?}");
    }
}

#[test]
fn decoder_new_accepts_progressive8() {
    let bytes = progressive_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("progressive 8-bit JPEG must construct");

    assert_eq!(dec.info().dimensions, (8, 8));
}

#[test]
fn decode_progressive8_rgb8_matches_jpeg_decoder_reference() {
    let bytes = progressive_8x8_jpeg();
    let mut reference_decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&bytes));
    let reference = reference_decoder
        .decode()
        .expect("jpeg-decoder reference decode");
    let info = reference_decoder.info().expect("jpeg-decoder info");
    assert_eq!((info.width, info.height), (8, 8));

    let dec = Decoder::new(&bytes).expect("progressive 8-bit JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let mut actual = vec![0u8; (w * h * 3) as usize];
    dec.decode_into(&mut actual, (w * 3) as usize, PixelFormat::Rgb8)
        .expect("progressive decode must succeed");

    assert_eq!(actual.len(), reference.len());
    let max_delta = actual
        .iter()
        .zip(reference.iter())
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap_or(0);
    assert!(
        max_delta <= 3,
        "progressive RGB max channel delta {max_delta} exceeds tolerance"
    );
}

#[test]
fn decode_region_into_rgb8_crops_progressive8_pixels() {
    let bytes = progressive_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("progressive 8-bit JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * 3;
    let mut full = vec![0u8; stride * h as usize];
    dec.decode_into(&mut full, stride, PixelFormat::Rgb8)
        .expect("full progressive decode must succeed");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let mut actual = vec![0u8; roi.w as usize * roi.h as usize * 3];

    let outcome = dec
        .decode_region_into(&mut actual, roi.w as usize * 3, PixelFormat::Rgb8, roi)
        .expect("progressive ROI decode must succeed");

    assert_eq!(outcome.decoded, roi);
    assert_eq!(actual, crop_rgb(&full, w, roi));
}

#[test]
fn decode_scaled_into_rgb8_projects_progressive8_pixels() {
    let bytes = progressive_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("progressive 8-bit JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * 3;
    let mut full = vec![0u8; stride * h as usize];
    dec.decode_into(&mut full, stride, PixelFormat::Rgb8)
        .expect("full progressive decode must succeed");
    let scale = Downscale::Half;
    let scaled_w = w.div_ceil(2);
    let scaled_h = h.div_ceil(2);
    let scaled_rect = Rect {
        x: 0,
        y: 0,
        w: scaled_w,
        h: scaled_h,
    };
    let mut actual = vec![0u8; scaled_w as usize * scaled_h as usize * 3];

    let outcome = dec
        .decode_scaled_into(&mut actual, scaled_w as usize * 3, PixelFormat::Rgb8, scale)
        .expect("progressive scaled decode must succeed");

    assert_eq!(outcome.decoded, Rect::full((w, h)));
    assert_eq!(actual, project_scaled_rgb(&full, w, h, scaled_rect, 2));
}

#[test]
fn decode_region_scaled_into_rgb8_projects_progressive8_pixels() {
    let bytes = progressive_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("progressive 8-bit JPEG must construct");
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * 3;
    let mut full = vec![0u8; stride * h as usize];
    dec.decode_into(&mut full, stride, PixelFormat::Rgb8)
        .expect("full progressive decode must succeed");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_roi = scaled_rect_covering_for_test(roi, 2);
    let mut actual = vec![0u8; scaled_roi.w as usize * scaled_roi.h as usize * 3];

    let outcome = dec
        .decode_region_scaled_into(
            &mut actual,
            scaled_roi.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
            Downscale::Half,
        )
        .expect("progressive region-scaled decode must succeed");

    assert_eq!(outcome.decoded, roi);
    assert_eq!(actual, project_scaled_rgb(&full, w, h, scaled_roi, 2));
}

#[test]
fn decode_region_into_rgb8_crops_constant_app14_rgb_pixels() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let roi = Rect {
        x: 2,
        y: 1,
        w: 3,
        h: 4,
    };
    let mut buf = vec![0u8; roi.w as usize * roi.h as usize * 3];
    let outcome = dec
        .decode_region_into(&mut buf, roi.w as usize * 3, PixelFormat::Rgb8, roi)
        .unwrap();
    assert_eq!(outcome.decoded, roi);
    let mut expected = Vec::with_capacity(buf.len());
    for _ in 0..(roi.w * roi.h) {
        expected.extend_from_slice(&[200, 20, 10]);
    }
    assert_eq!(buf, expected);
}

fn crop_rgb(full: &[u8], width: u32, roi: Rect) -> Vec<u8> {
    crop_interleaved_u8(full, width as usize, 3, pixel_rect(roi))
}

fn crop_gray(full: &[u8], width: u32, roi: Rect) -> Vec<u8> {
    crop_interleaved_u8(full, width as usize, 1, pixel_rect(roi))
}

fn crop_gray16(full: &[u16], width: u32, roi: Rect) -> Vec<u16> {
    crop_interleaved_u16(full, width as usize, 1, pixel_rect(roi))
}

fn project_scaled_rgb(
    full: &[u8],
    width: u32,
    height: u32,
    output_rect: Rect,
    denom: u32,
) -> Vec<u8> {
    project_scaled_interleaved_u8(full, width, height, 3, pixel_rect(output_rect), denom)
}

fn project_scaled_gray(
    full: &[u8],
    width: u32,
    height: u32,
    output_rect: Rect,
    denom: u32,
) -> Vec<u8> {
    project_scaled_interleaved_u8(full, width, height, 1, pixel_rect(output_rect), denom)
}

fn project_scaled_gray16(
    full: &[u16],
    width: u32,
    height: u32,
    output_rect: Rect,
    denom: u32,
) -> Vec<u16> {
    project_scaled_interleaved_u16(full, width, height, 1, pixel_rect(output_rect), denom)
}

fn assert_gray16_samples(buf: &[u8], stride: usize, width: u32, expected: &[u16], context: u8) {
    for (row, expected_row) in buf
        .chunks_exact(stride)
        .zip(expected.chunks_exact(width as usize))
    {
        for (sample, expected) in row[..width as usize * 2]
            .chunks_exact(2)
            .zip(expected_row.iter().copied())
        {
            assert_eq!(
                u16::from_le_bytes([sample[0], sample[1]]),
                expected,
                "predictor {context}"
            );
        }
    }
}

fn assert_gray16_rows_with_padding(
    buf: &[u8],
    stride: usize,
    width: u32,
    expected: &[u16],
    pad: usize,
    context: u8,
) {
    for (row, expected_row) in buf
        .chunks_exact(stride)
        .zip(expected.chunks_exact(width as usize))
    {
        for (sample, expected) in row[..width as usize * 2]
            .chunks_exact(2)
            .zip(expected_row.iter().copied())
        {
            assert_eq!(
                u16::from_le_bytes([sample[0], sample[1]]),
                expected,
                "predictor {context}"
            );
        }
        assert_eq!(&row[width as usize * 2..], vec![0xaa; pad].as_slice());
    }
}

fn scaled_rect_covering_for_test(rect: Rect, denom: u32) -> Rect {
    jpeg_rect(scaled_rect_covering(pixel_rect(rect), denom))
}

fn pixel_rect(rect: Rect) -> PixelRect {
    PixelRect::new(rect.x, rect.y, rect.w, rect.h)
}

fn jpeg_rect(rect: PixelRect) -> Rect {
    Rect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}

fn expected_scaled_rgb16_pixels(full: &[u8], full_width: usize, roi: Rect, denom: u32) -> Vec<u8> {
    let scaled = scaled_rect_covering_for_test(roi, denom);
    let mut expected = Vec::with_capacity(scaled.w as usize * scaled.h as usize * 6);
    for out_y in scaled.y..scaled.y + scaled.h {
        let src_y = out_y.saturating_mul(denom).min(roi.y + roi.h - 1);
        for out_x in scaled.x..scaled.x + scaled.w {
            let src_x = out_x.saturating_mul(denom).min(roi.x + roi.w - 1);
            let start = (src_y as usize * full_width + src_x as usize) * 6;
            expected.extend_from_slice(&full[start..start + 6]);
        }
    }
    expected
}

fn assert_padded_rgb16_rows(buf: &[u8], stride: usize, width: usize, expected: &[u8]) {
    let row_bytes = width * PixelFormat::Rgb16.bytes_per_pixel();
    for (row_index, row) in buf.chunks_exact(stride).enumerate() {
        let start = row_index * row_bytes;
        assert_rgb16_image_eq(
            &row[..row_bytes],
            &expected[start..start + row_bytes],
            width,
        );
        assert_eq!(&row[row_bytes..], &[0xaa; 6]);
    }
}

fn assert_padded_rgba16_rows(buf: &[u8], stride: usize, width: usize, expected: &[u8]) {
    let row_bytes = width * PixelFormat::Rgba16.bytes_per_pixel();
    for (row_index, row) in buf.chunks_exact(stride).enumerate() {
        let start = row_index * row_bytes;
        assert_eq!(&row[..row_bytes], &expected[start..start + row_bytes]);
        assert_eq!(&row[row_bytes..], &[0xaa; 8]);
    }
}

fn rgb16_samples_to_le_bytes(samples: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

fn rgb16_to_rgba16(rgb: &[u8], alpha: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 6 * 8);
    let alpha = alpha.to_le_bytes();
    for pixel in rgb.chunks_exact(6) {
        out.extend_from_slice(pixel);
        out.extend_from_slice(&alpha);
    }
    out
}

fn crop_rgb16_bytes(full: &[u8], width: usize, roi: Rect) -> Vec<u8> {
    let mut out = Vec::with_capacity(roi.w as usize * roi.h as usize * 6);
    for y in roi.y..roi.y + roi.h {
        let row = y as usize * width * 6;
        let start = row + roi.x as usize * 6;
        let end = start + roi.w as usize * 6;
        out.extend_from_slice(&full[start..end]);
    }
    out
}

fn assert_rgb16_image_eq(actual: &[u8], expected: &[u8], width: usize) {
    assert_eq!(actual.len(), expected.len());
    for (pixel_index, (actual_pixel, expected_pixel)) in actual
        .chunks_exact(6)
        .zip(expected.chunks_exact(6))
        .enumerate()
    {
        if actual_pixel != expected_pixel {
            let actual_rgb = rgb16_pixel(actual_pixel);
            let expected_rgb = rgb16_pixel(expected_pixel);
            panic!(
                "RGB16 pixel mismatch at ({}, {}): actual {:?}, expected {:?}",
                pixel_index % width,
                pixel_index / width,
                actual_rgb,
                expected_rgb
            );
        }
    }
}

fn rgb16_pixel(pixel: &[u8]) -> [u16; 3] {
    [
        u16::from_le_bytes([pixel[0], pixel[1]]),
        u16::from_le_bytes([pixel[2], pixel[3]]),
        u16::from_le_bytes([pixel[4], pixel[5]]),
    ]
}
