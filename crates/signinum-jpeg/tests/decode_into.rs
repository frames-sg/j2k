// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `Decoder::decode_into`.

use signinum_jpeg::{Decoder, Downscale, JpegError, PixelFormat, Rect};

mod fixtures;
use fixtures::{
    cmyk_8x8_jpeg, four_component_8x8_rgb, grayscale_8x8_jpeg, minimal_baseline_420_jpeg,
    progressive_8x8_jpeg, rgb_app14_8x8_jpeg, rgb_app14_8x8_rgb, ycck_8x8_jpeg,
};

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

    let (owned, outcome) = dec.decode(PixelFormat::Rgb8).unwrap();
    assert_eq!(owned, expected);
    assert_eq!(outcome, expected_outcome);
}

#[test]
fn decode_into_rgba8_writes_alpha_byte() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let (w, h) = dec.info().dimensions;
    let mut buf = vec![0u8; (w * h * 4) as usize];
    dec.decode_rgba8_into_with_alpha(&mut buf, (w * 4) as usize, 200)
        .unwrap();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * 4;
            assert_eq!(buf[idx + 3], 200, "pixel ({x},{y}) alpha");
        }
    }
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
        .decode_region_scaled(PixelFormat::Rgb8, roi, Downscale::Half)
        .unwrap();
    assert_eq!(owned, expected);
    assert_eq!(outcome, expected_outcome);
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
        .decode_scaled(PixelFormat::Rgb8, Downscale::Half)
        .unwrap();
    assert_eq!(owned, expected);
    assert_eq!(outcome, expected_outcome);
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
    let expected = ((77 * 200 + 150 * 20 + 29 * 10 + 128) >> 8) as u8;

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
    let mut out = Vec::with_capacity(roi.w as usize * roi.h as usize * 3);
    for y in roi.y..roi.y + roi.h {
        let row = y as usize * width as usize * 3;
        let start = row + roi.x as usize * 3;
        let end = start + roi.w as usize * 3;
        out.extend_from_slice(&full[start..end]);
    }
    out
}

fn project_scaled_rgb(
    full: &[u8],
    width: u32,
    height: u32,
    output_rect: Rect,
    denom: u32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(output_rect.w as usize * output_rect.h as usize * 3);
    for sy in output_rect.y..output_rect.y + output_rect.h {
        let src_y = (sy * denom).min(height - 1);
        for sx in output_rect.x..output_rect.x + output_rect.w {
            let src_x = (sx * denom).min(width - 1);
            let offset = (src_y as usize * width as usize + src_x as usize) * 3;
            out.extend_from_slice(&full[offset..offset + 3]);
        }
    }
    out
}

fn scaled_rect_covering_for_test(rect: Rect, denom: u32) -> Rect {
    let x1 = (rect.x + rect.w).div_ceil(denom);
    let y1 = (rect.y + rect.h).div_ceil(denom);
    Rect {
        x: rect.x / denom,
        y: rect.y / denom,
        w: x1 - rect.x / denom,
        h: y1 - rect.y / denom,
    }
}

#[test]
fn decode_region_into_rgb8_scaled_crops_constant_app14_rgb_pixels() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let roi = Rect {
        x: 2,
        y: 2,
        w: 4,
        h: 4,
    };
    let mut buf = vec![0u8; 2 * 2 * 3];
    let outcome = dec
        .decode_region_scaled_into(&mut buf, 2 * 3, PixelFormat::Rgb8, roi, Downscale::Half)
        .unwrap();
    assert_eq!(outcome.decoded, roi);
    let mut expected = Vec::with_capacity(buf.len());
    for _ in 0..4 {
        expected.extend_from_slice(&[200, 20, 10]);
    }
    assert_eq!(buf, expected);
}

#[test]
fn decode_into_rgb8_converts_cmyk_and_ycck() {
    for bytes in [cmyk_8x8_jpeg(), ycck_8x8_jpeg()] {
        let dec = Decoder::new(&bytes).expect("four-component baseline JPEG should construct");
        let (w, h) = dec.info().dimensions;
        let mut buf = vec![0u8; (w * h * 3) as usize];

        dec.decode_into(&mut buf, (w * 3) as usize, PixelFormat::Rgb8)
            .expect("CMYK/YCCK to RGB8 decode should succeed");

        assert_eq!(buf, four_component_8x8_rgb());
    }
}

#[test]
fn decode_into_rgba8_converts_cmyk_and_ycck_with_alpha() {
    for bytes in [cmyk_8x8_jpeg(), ycck_8x8_jpeg()] {
        let dec = Decoder::new(&bytes).expect("four-component baseline JPEG should construct");
        let (w, h) = dec.info().dimensions;
        let mut buf = vec![0u8; (w * h * 4) as usize];

        dec.decode_into(&mut buf, (w * 4) as usize, PixelFormat::Rgba8)
            .expect("CMYK/YCCK to RGBA8 decode should succeed");

        for pixel in buf.chunks_exact(4) {
            assert_eq!(pixel, &[64, 64, 64, 255]);
        }
    }
}

#[test]
fn decoder_new_rejects_invalid_sequential_scan_parameters() {
    let mut bytes = minimal_baseline_420_jpeg();
    let sos = bytes
        .windows(2)
        .position(|w| w == [0xff, 0xda])
        .expect("fixture SOS");
    bytes[sos + 2 + 2 + 1 + 3 * 2] = 1;

    let err = Decoder::new(&bytes).expect_err("baseline Ss=1 must be rejected");
    assert!(matches!(
        err,
        JpegError::InvalidScanParameters {
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
            ..
        }
    ));
}
