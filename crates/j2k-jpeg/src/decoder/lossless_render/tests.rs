// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_core::{Downscale, PixelFormat};

fn valid_lossless_gray_jpeg(width: u16, height: u16, precision: u8) -> Vec<u8> {
    let [height_hi, height_lo] = height.to_be_bytes();
    let [width_hi, width_lo] = width.to_be_bytes();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xFF, 0xD8]);
    bytes.extend_from_slice(&[
        0xFF, 0xC3, 0x00, 11, precision, height_hi, height_lo, width_hi, width_lo, 1, 1, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xFF, 0xC4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00,
    ]);
    bytes.extend_from_slice(&[0xFF, 0xDA, 0x00, 8, 1, 1, 0x00, 1, 0, 0]);
    let entropy_bytes = (usize::from(width) * usize::from(height)).div_ceil(8);
    bytes.extend(core::iter::repeat_n(0u8, entropy_bytes));
    bytes.extend_from_slice(&[0xFF, 0xD9]);
    bytes
}

fn valid_lossless_color_jpeg(width: u16, height: u16, precision: u8) -> Vec<u8> {
    let [height_hi, height_lo] = height.to_be_bytes();
    let [width_hi, width_lo] = width.to_be_bytes();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xFF, 0xD8]);
    bytes.extend_from_slice(&[
        0xFF, 0xC3, 0x00, 17, precision, height_hi, height_lo, width_hi, width_lo, 3, 1, 0x11, 0,
        2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xFF, 0xC4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00,
    ]);
    bytes.extend_from_slice(&[0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 1, 0, 0]);
    let entropy_bits = usize::from(width) * usize::from(height) * 3;
    bytes.extend(core::iter::repeat_n(0u8, entropy_bits.div_ceil(8)));
    bytes.extend_from_slice(&[0xFF, 0xD9]);
    bytes
}

fn source_roi() -> Rect {
    Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    }
}

#[test]
fn lossless_gray8_full_and_scaled_region_decode_constant_samples() {
    let bytes = valid_lossless_gray_jpeg(16, 16, 8);
    let decoder = Decoder::new(&bytes).expect("valid lossless Gray8 fixture");

    let mut full = vec![0; 16 * 16];
    let full_outcome = decoder
        .decode_into(&mut full, 16, PixelFormat::Gray8)
        .expect("lossless Gray8 full decode");
    assert_eq!(full, vec![128; 16 * 16]);
    assert_eq!(full_outcome.decoded, Rect::full((16, 16)));

    let mut region = vec![0; 4 * 4];
    let region_outcome = decoder
        .decode_region_scaled_into(
            &mut region,
            4,
            PixelFormat::Gray8,
            source_roi(),
            Downscale::Half,
        )
        .expect("lossless Gray8 scaled region decode");
    assert_eq!(region, vec![128; 4 * 4]);
    assert_eq!(region_outcome.decoded, source_roi());
}

#[test]
fn lossless_gray16_full_and_scaled_region_decode_constant_samples() {
    let bytes = valid_lossless_gray_jpeg(16, 16, 16);
    let decoder = Decoder::new(&bytes).expect("valid lossless Gray16 fixture");

    let mut full = vec![0; 16 * 16 * 2];
    let full_outcome = decoder
        .decode_into(&mut full, 16 * 2, PixelFormat::Gray16)
        .expect("lossless Gray16 full decode");
    assert!(full.chunks_exact(2).all(|sample| sample == [0x00, 0x80]));
    assert_eq!(full_outcome.decoded, Rect::full((16, 16)));

    let mut region = vec![0; 4 * 4 * 2];
    let region_outcome = decoder
        .decode_region_scaled_into(
            &mut region,
            4 * 2,
            PixelFormat::Gray16,
            source_roi(),
            Downscale::Half,
        )
        .expect("lossless Gray16 scaled region decode");
    assert!(region.chunks_exact(2).all(|sample| sample == [0x00, 0x80]));
    assert_eq!(region_outcome.decoded, source_roi());
}

#[test]
fn lossless_color8_scaled_region_preserves_source_roi_for_rgb_and_rgba() {
    let bytes = valid_lossless_color_jpeg(16, 16, 8);
    let decoder = Decoder::new(&bytes).expect("valid lossless color8 fixture");

    let mut rgb = vec![0; 4 * 4 * 3];
    let rgb_outcome = decoder
        .decode_region_scaled_into(
            &mut rgb,
            4 * 3,
            PixelFormat::Rgb8,
            source_roi(),
            Downscale::Half,
        )
        .expect("lossless RGB8 scaled region decode");
    assert!(rgb.chunks_exact(3).all(|pixel| pixel == [128, 128, 128]));
    assert_eq!(rgb_outcome.decoded, source_roi());

    let mut rgba = vec![0; 4 * 4 * 4];
    let rgba_outcome = decoder
        .decode_region_scaled_into(
            &mut rgba,
            4 * 4,
            PixelFormat::Rgba8,
            source_roi(),
            Downscale::Half,
        )
        .expect("lossless RGBA8 scaled region decode");
    assert!(rgba
        .chunks_exact(4)
        .all(|pixel| pixel == [128, 128, 128, 255]));
    assert_eq!(rgba_outcome.decoded, source_roi());
}

#[test]
fn lossless_color16_scaled_region_preserves_source_roi_for_rgb_and_rgba() {
    let bytes = valid_lossless_color_jpeg(16, 16, 16);
    let decoder = Decoder::new(&bytes).expect("valid lossless color16 fixture");

    let mut rgb = vec![0; 4 * 4 * 6];
    let rgb_outcome = decoder
        .decode_region_scaled_into(
            &mut rgb,
            4 * 6,
            PixelFormat::Rgb16,
            source_roi(),
            Downscale::Half,
        )
        .expect("lossless RGB16 scaled region decode");
    assert!(rgb.chunks_exact(2).all(|sample| sample == [0x00, 0x80]));
    assert_eq!(rgb_outcome.decoded, source_roi());

    let mut rgba = vec![0; 4 * 4 * 8];
    let rgba_outcome = decoder
        .decode_region_scaled_into(
            &mut rgba,
            4 * 8,
            PixelFormat::Rgba16,
            source_roi(),
            Downscale::Half,
        )
        .expect("lossless RGBA16 scaled region decode");
    assert!(rgba
        .chunks_exact(8)
        .all(|pixel| { pixel == [0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0xFF, 0xFF] }));
    assert_eq!(rgba_outcome.decoded, source_roi());
}
