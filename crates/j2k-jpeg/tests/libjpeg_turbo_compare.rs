// SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../benches/common/libjpeg_turbo.rs"]
mod libjpeg_turbo;

use j2k_jpeg::{DecodeRequest, Decoder, Downscale, PixelFormat, Rect};
use j2k_test_support::JPEG_BASELINE_420_16X16;

#[test]
fn turbojpeg_rgb_and_region_match_j2k_fixture() {
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

    let (rgb, _) = dec
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("j2k rgb");
    let turbo_rgb = turbo.decode_rgb(bytes).expect("turbojpeg rgb");
    assert_eq!(turbo_rgb, rgb);

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
