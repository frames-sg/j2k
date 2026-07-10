// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bit-exact parity against libjpeg-turbo's ISLOW path.

use j2k_jpeg::{DecodeRequest, Decoder, Downscale, PixelFormat, Rect};
use j2k_test_support::{
    crop_interleaved_bytes, crop_interleaved_u8, restart_coded_grayscale_jpeg,
    scaled_rect_covering, PixelRect, JPEG_BASELINE_420_16X16, JPEG_BASELINE_420_16X16_RGB,
    JPEG_BASELINE_420_RESTART_32X16, JPEG_BASELINE_420_RESTART_32X16_RGB, JPEG_BASELINE_422_16X8,
    JPEG_BASELINE_422_16X8_RGB, JPEG_BASELINE_444_8X8, JPEG_BASELINE_444_8X8_RGB,
    JPEG_GRAYSCALE_8X8, JPEG_GRAYSCALE_8X8_GRAY,
};

const BASELINE_420_JPG: &[u8] = JPEG_BASELINE_420_16X16;
const BASELINE_420_RGB: &[u8] = JPEG_BASELINE_420_16X16_RGB;

const GRAYSCALE_8X8_JPG: &[u8] = JPEG_GRAYSCALE_8X8;
const GRAYSCALE_8X8_GRAY: &[u8] = JPEG_GRAYSCALE_8X8_GRAY;

#[test]
fn baseline_420_16x16_matches_libjpeg_turbo_bit_exact() {
    let dec = Decoder::new(BASELINE_420_JPG).expect("fixture must parse");
    let (w, h) = dec.info().dimensions;
    assert_eq!((w, h), (16, 16));
    let mut out = vec![0u8; 16 * 16 * 3];
    let outcome = dec
        .decode_scaled_into(&mut out, 16 * 3, PixelFormat::Rgb8, Downscale::None)
        .expect("decode must succeed");
    assert_eq!(outcome.decoded.w, 16);
    assert_eq!(outcome.decoded.h, 16);

    if out != BASELINE_420_RGB {
        let first_diff = out
            .iter()
            .zip(BASELINE_420_RGB.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(usize::MAX);
        panic!(
            "parity mismatch at byte {first_diff}: got {:?} want {:?}\nfull decoded: {:?}\nreference:    {:?}",
            out.get(first_diff),
            BASELINE_420_RGB.get(first_diff),
            &out[..first_diff.min(out.len())],
            &BASELINE_420_RGB[..first_diff.min(BASELINE_420_RGB.len())],
        );
    }
}

#[test]
fn baseline_fast444_and_restart_routes_match_stored_rgb_fixtures() {
    let cases = [
        (
            "fast 4:4:4",
            JPEG_BASELINE_444_8X8,
            JPEG_BASELINE_444_8X8_RGB,
        ),
        (
            "restart-coded 4:2:0",
            JPEG_BASELINE_420_RESTART_32X16,
            JPEG_BASELINE_420_RESTART_32X16_RGB,
        ),
    ];

    for (route, encoded, expected) in cases {
        let decoder = Decoder::new(encoded).expect("fixture must parse");
        let (width, height) = decoder.info().dimensions;
        let stride = width as usize * 3;
        let mut actual = vec![0u8; stride * height as usize];
        decoder
            .decode_scaled_into(&mut actual, stride, PixelFormat::Rgb8, Downscale::None)
            .expect("route must decode");
        assert_eq!(actual, expected, "{route} output changed");
    }
}

#[test]
fn generic_422_current_output_regression_baseline() {
    let decoder = Decoder::new(JPEG_BASELINE_422_16X8).expect("4:2:2 fixture must parse");
    let mut actual_422 = vec![0u8; JPEG_BASELINE_422_16X8_RGB.len()];
    decoder
        .decode_scaled_into(&mut actual_422, 16 * 3, PixelFormat::Rgb8, Downscale::None)
        .expect("generic 4:2:2 route must decode");

    // Preserve the existing j2k output while the separately tracked libjpeg-turbo
    // 4:2:2 interpolation-parity gap is investigated.
    let output_hash = actual_422
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    assert_eq!(
        output_hash, 0x4a9b_e9f5_ec1f_80df,
        "generic 4:2:2 output changed"
    );

    let mut differing_bytes = 0usize;
    let mut max_delta = 0u8;
    let mut differing_by_channel = [0usize; 3];
    for (index, (&actual, &reference)) in actual_422
        .iter()
        .zip(JPEG_BASELINE_422_16X8_RGB)
        .enumerate()
    {
        let delta = actual.abs_diff(reference);
        if delta != 0 {
            differing_bytes += 1;
            max_delta = max_delta.max(delta);
            differing_by_channel[index % 3] += 1;
        }
    }
    assert_eq!(
        (differing_bytes, max_delta, differing_by_channel),
        (16, 2, [0, 4, 12]),
        "update only after resolving the tracked 4:2:2 interpolation gap"
    );
}

#[test]
fn grayscale_8x8_matches_libjpeg_turbo_bit_exact() {
    let dec = Decoder::new(GRAYSCALE_8X8_JPG).expect("grayscale fixture must parse");
    let (w, h) = dec.info().dimensions;
    assert_eq!((w, h), (8, 8));
    let mut out = vec![0u8; 8 * 8];
    dec.decode_scaled_into(&mut out, 8, PixelFormat::Gray8, Downscale::None)
        .expect("grayscale decode must succeed");
    assert_eq!(
        out, GRAYSCALE_8X8_GRAY,
        "grayscale parity must be bit-exact against djpeg -grayscale"
    );
}

#[test]
fn baseline_420_wsi_shaped_region_matches_full_decode_crop() {
    let dec = Decoder::new(BASELINE_420_JPG).expect("fixture must parse");
    let roi = Rect {
        x: 3,
        y: 2,
        w: 10,
        h: 11,
    };

    let full = decode_full_rgb(&dec);
    let region = decode_region_rgb(&dec, roi);
    assert_eq!(region, crop_rgb8(&full, 16, roi));
}

#[test]
fn baseline_420_wsi_shaped_scaled_region_matches_full_decode_crop() {
    let dec = Decoder::new(BASELINE_420_JPG).expect("fixture must parse");
    let roi = Rect {
        x: 3,
        y: 2,
        w: 10,
        h: 11,
    };

    let mut full = vec![0u8; 8 * 8 * 3];
    dec.decode_scaled_into(&mut full, 8 * 3, PixelFormat::Rgb8, Downscale::Half)
        .expect("full scaled decode must succeed");

    let region = dec
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            roi,
            Downscale::Half,
        ))
        .expect("scaled region decode must succeed")
        .0;

    let scaled_roi = scaled_rect_covering_half(roi);
    assert_eq!(region, crop_rgb8(&full, 8, scaled_roi));
}

#[test]
fn restart_coded_grayscale_wsi_shaped_region_matches_full_decode_crop() {
    let bytes = restart_coded_grayscale_jpeg(24, 24);
    let dec = Decoder::new(&bytes).expect("restart-coded fixture must parse");
    let roi = Rect {
        x: 5,
        y: 6,
        w: 11,
        h: 10,
    };

    let mut full = vec![0u8; 24 * 24];
    dec.decode_scaled_into(&mut full, 24, PixelFormat::Gray8, Downscale::None)
        .expect("full grayscale decode must succeed");
    let region = dec
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Gray8,
            roi,
            Downscale::None,
        ))
        .expect("restart-coded region decode must succeed")
        .0;

    assert_eq!(region, crop_gray8(&full, 24, roi));
}

fn decode_region_rgb(dec: &Decoder<'_>, roi: Rect) -> Vec<u8> {
    dec.decode_request(DecodeRequest::region_scaled(
        PixelFormat::Rgb8,
        roi,
        Downscale::None,
    ))
    .expect("region decode must succeed")
    .0
}

fn decode_full_rgb(dec: &Decoder<'_>) -> Vec<u8> {
    let (w, h) = dec.info().dimensions;
    let mut out = vec![0u8; (w * h * 3) as usize];
    dec.decode_scaled_into(
        &mut out,
        (w * 3) as usize,
        PixelFormat::Rgb8,
        Downscale::None,
    )
    .expect("full decode must succeed");
    out
}

fn crop_rgb8(full: &[u8], width: usize, roi: Rect) -> Vec<u8> {
    crop_interleaved_bytes(full, width, 3, pixel_rect(roi))
}

fn crop_gray8(full: &[u8], width: usize, roi: Rect) -> Vec<u8> {
    crop_interleaved_u8(full, width, 1, pixel_rect(roi))
}

fn scaled_rect_covering_half(roi: Rect) -> Rect {
    let scaled = scaled_rect_covering(pixel_rect(roi), 2);
    Rect {
        x: scaled.x,
        y: scaled.y,
        w: scaled.w,
        h: scaled.h,
    }
}

fn pixel_rect(roi: Rect) -> PixelRect {
    PixelRect::new(roi.x, roi.y, roi.w, roi.h)
}
