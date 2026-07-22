// SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../benches/common/libjpeg_turbo.rs"]
mod libjpeg_turbo;
#[path = "../benches/common/libjpeg_turbo_extended.rs"]
mod libjpeg_turbo_extended;
#[cfg(all(has_libjpeg_turbo, has_libjpeg_turbo_v3))]
#[path = "../benches/common/libjpeg_turbo_v2.rs"]
mod libjpeg_turbo_v2;

use j2k_jpeg::{DecodeRequest, Decoder, Downscale, PixelFormat, Rect};
use j2k_test_support::{
    JPEG_BASELINE_420_16X16, JPEG_BASELINE_422_16X8, JPEG_BASELINE_422_16X8_RGB, JPEG_GRAYSCALE_8X8,
};

#[test]
fn turbojpeg_rgb_and_region_match_j2k_fixtures() {
    let require_turbo = std::env::var_os("J2K_REQUIRE_LIBJPEG_TURBO").is_some();
    let turbo_available = libjpeg_turbo::is_available();
    assert!(
        !require_turbo || turbo_available,
        "J2K_REQUIRE_LIBJPEG_TURBO is set but libjpeg-turbo is unavailable"
    );
    if !turbo_available {
        return;
    }

    let bytes = JPEG_BASELINE_420_16X16;
    let dec = Decoder::new(bytes).expect("j2k decoder");
    let mut turbo = libjpeg_turbo::TurboJpegDecoder::new().expect("turbojpeg decoder");

    let info = turbo.inspect(bytes).expect("turbojpeg inspect");
    assert_eq!((info.width, info.height), (16, 16));
    assert_eq!(info.subsamp, 2, "fixture should report 4:2:0 sampling");

    let (rgb, _) = dec
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("j2k rgb");
    let turbo_rgb = turbo.decode_rgb(bytes).expect("turbojpeg rgb");
    assert_eq!(turbo_rgb, rgb);

    let prepared = turbo.prepare_rgb(bytes).expect("turbojpeg prepare RGB");
    assert_eq!((prepared.width, prepared.height), (16, 16));
    assert_eq!(prepared.subsamp, 2);
    let mut prepared_rgb = vec![0_u8; 16 * 16 * 3];
    turbo
        .decode_prepared_rgb_into(bytes, &mut prepared_rgb, 16 * 3, 16, 16)
        .expect("turbojpeg prepared RGB");
    assert_eq!(prepared_rgb, turbo_rgb);

    let gray_decoder = Decoder::new(JPEG_GRAYSCALE_8X8).expect("j2k grayscale decoder");
    let (gray, _) = gray_decoder
        .decode_request(DecodeRequest::full(PixelFormat::Gray8))
        .expect("j2k grayscale");
    let turbo_gray = turbo
        .decode_gray(JPEG_GRAYSCALE_8X8)
        .expect("turbojpeg grayscale");
    assert_eq!(turbo_gray, gray);

    let (scaled, _) = dec
        .decode_request(DecodeRequest::scaled(PixelFormat::Rgb8, Downscale::Quarter))
        .expect("j2k scaled");
    let turbo_scaled = turbo
        .decode_scaled_rgb(bytes, Downscale::Quarter)
        .expect("turbojpeg scaled");
    assert_eq!(turbo_scaled.len(), scaled.len());
    assert!(!turbo_scaled.is_empty());

    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let turbo_region_a = turbo
        .decode_region_rgb(bytes, roi)
        .expect("turbojpeg region");
    let turbo_region_b = turbo
        .decode_region_rgb(bytes, roi)
        .expect("turbojpeg region");
    assert_eq!(turbo_region_a, turbo_region_b);
    assert_eq!(turbo_region_a.len(), crop_rgb(&turbo_rgb, 16, roi).len());

    let turbo_region_scaled_a = turbo
        .decode_region_scaled_rgb(bytes, roi, Downscale::Quarter)
        .expect("turbojpeg scaled region");
    let turbo_region_scaled_b = turbo
        .decode_region_scaled_rgb(bytes, roi, Downscale::Quarter)
        .expect("turbojpeg scaled region");
    assert_eq!(turbo_region_scaled_a, turbo_region_scaled_b);
    assert_eq!(turbo_region_scaled_a.len(), 2 * 2 * 3);

    let bytes_422 = JPEG_BASELINE_422_16X8;
    let decoder_422 = Decoder::new(bytes_422).expect("j2k 4:2:2 decoder");
    let info_422 = turbo.inspect(bytes_422).expect("turbojpeg 4:2:2 inspect");
    assert_eq!((info_422.width, info_422.height), (16, 8));
    assert_eq!(info_422.subsamp, 1, "fixture should report 4:2:2 sampling");
    let (rgb_422, _) = decoder_422
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("j2k 4:2:2 RGB");
    let turbo_rgb_422 = turbo.decode_rgb(bytes_422).expect("turbojpeg 4:2:2 RGB");
    assert_eq!(turbo_rgb_422, JPEG_BASELINE_422_16X8_RGB);
    assert_eq!(rgb_422, turbo_rgb_422);
}

#[cfg(all(has_libjpeg_turbo, has_libjpeg_turbo_v3))]
#[test]
fn legacy_turbojpeg_abi_matches_the_v3_adapter() {
    let bytes = JPEG_BASELINE_420_16X16;
    let mut current = libjpeg_turbo::TurboJpegDecoder::new().expect("v3 turbojpeg decoder");
    let mut legacy = libjpeg_turbo_v2::TurboJpegDecoder::new().expect("legacy turbojpeg decoder");

    assert_eq!(
        legacy.decode_rgb(bytes).expect("legacy full RGB"),
        current.decode_rgb(bytes).expect("v3 full RGB")
    );
    assert_eq!(
        legacy
            .decode(JPEG_GRAYSCALE_8X8, 6, None, Downscale::None)
            .expect("legacy grayscale"),
        current
            .decode(JPEG_GRAYSCALE_8X8, 6, None, Downscale::None)
            .expect("v3 grayscale")
    );
    assert_eq!(
        legacy
            .decode(bytes, 0, None, Downscale::Quarter)
            .expect("legacy scaled RGB"),
        current
            .decode(bytes, 0, None, Downscale::Quarter)
            .expect("v3 scaled RGB")
    );

    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    assert_eq!(
        legacy
            .decode(bytes, 0, Some(roi), Downscale::None)
            .expect("legacy region RGB"),
        current
            .decode(bytes, 0, Some(roi), Downscale::None)
            .expect("v3 region RGB")
    );
    assert_eq!(
        legacy
            .decode(bytes, 0, Some(roi), Downscale::Quarter)
            .expect("legacy scaled region RGB"),
        current
            .decode(bytes, 0, Some(roi), Downscale::Quarter)
            .expect("v3 scaled region RGB")
    );

    let (width, height, _) = legacy
        .prepare_full_frame(bytes)
        .expect("legacy prepare RGB");
    let pitch = width as usize * 3;
    let mut prepared = vec![0_u8; pitch * height as usize];
    legacy
        .decompress(bytes, &mut prepared, pitch, 0)
        .expect("legacy prepared RGB");
    assert_eq!(
        prepared,
        current.decode_rgb(bytes).expect("v3 prepared reference")
    );
}

fn crop_rgb(full: &[u8], width: usize, roi: Rect) -> Vec<u8> {
    let stride = width * 3;
    let mut out = vec![0u8; roi.w as usize * roi.h as usize * 3];
    for row in 0..roi.h as usize {
        let src_start = (roi.y as usize + row) * stride + roi.x as usize * 3;
        let src_end = src_start + roi.w as usize * 3;
        let dst_start = row * roi.w as usize * 3;
        out[dst_start..dst_start + roi.w as usize * 3].copy_from_slice(&full[src_start..src_end]);
    }
    out
}
