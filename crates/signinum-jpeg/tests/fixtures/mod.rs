// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

//! Fixture JPEGs for decode integration tests. Inputs are committed under
//! `corpus/conformance/` and embedded via `include_bytes!` so tests remain
//! hermetic (no filesystem dependency at run time).

/// A 16×16 baseline JPEG with 4:2:0 sampling.
pub(crate) fn minimal_baseline_420_jpeg() -> Vec<u8> {
    include_bytes!("../../fixtures/conformance/baseline_420_16x16.jpg").to_vec()
}

/// An 8×8 grayscale (single-component) baseline JPEG.
pub(crate) fn grayscale_8x8_jpeg() -> Vec<u8> {
    include_bytes!("../../fixtures/conformance/grayscale_8x8.jpg").to_vec()
}

/// An 8x8 12-bit extended sequential grayscale JPEG with all-zero DCT blocks.
pub(crate) fn extended_12bit_grayscale_8x8_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[0xff, 0xc1, 0x00, 11, 12, 0, 8, 0, 8, 1, 1, 0x11, 0]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);
    bytes.push(0x00);
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// A 16x8 12-bit extended sequential grayscale JPEG with DRI=1 and RST0.
pub(crate) fn extended_12bit_grayscale_restart_16x8_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[0xff, 0xc1, 0x00, 11, 12, 0, 8, 0, 16, 1, 1, 0x11, 0]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);
    bytes.extend_from_slice(&[0x00, 0xff, 0xd0, 0x00]);
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// An 8x8 12-bit extended sequential APP14 RGB JPEG.
pub(crate) fn extended_12bit_rgb_8x8_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[
        0xff, 0xee, 0x00, 0x0e, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc1, 0x00, 17, 12, 0, 8, 0, 8, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0,
    ]);
    bytes.extend(dc_category4_rgb_entropy());
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// A 16x8 12-bit extended sequential APP14 RGB JPEG with DRI=1 and RST0.
pub(crate) fn extended_12bit_rgb_restart_16x8_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[
        0xff, 0xee, 0x00, 0x0e, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc1, 0x00, 17, 12, 0, 8, 0, 16, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0,
    ]);
    bytes.extend(restart_segmented_entropy(
        2,
        dc_category4_rgb_mcu_bits(false),
    ));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// Reference Rgb16 pixels for [`extended_12bit_rgb_8x8_jpeg`].
pub(crate) fn extended_12bit_rgb_8x8_rgb16() -> Vec<u8> {
    repeat_rgb16_pixels(8, 8, [2064, 2072, 2032])
}

/// Reference Rgb16 pixels for [`extended_12bit_rgb_restart_16x8_jpeg`].
pub(crate) fn extended_12bit_rgb_restart_16x8_rgb16() -> Vec<u8> {
    repeat_rgb16_pixels(16, 8, [2064, 2072, 2032])
}

fn repeat_rgb16_pixels(width: usize, height: usize, rgb: [u16; 3]) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * height * 6);
    for _ in 0..width * height {
        for sample in rgb {
            out.extend_from_slice(&sample.to_le_bytes());
        }
    }
    out
}

/// An 8x8 12-bit extended sequential YCbCr 4:4:4 JPEG.
pub(crate) fn extended_12bit_ycbcr_8x8_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc1, 0x00, 17, 12, 0, 8, 0, 8, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0,
    ]);
    bytes.extend(dc_category4_rgb_entropy());
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// A 16x8 12-bit extended sequential YCbCr 4:4:4 JPEG with DRI=1 and RST0.
pub(crate) fn extended_12bit_ycbcr_restart_16x8_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc1, 0x00, 17, 12, 0, 8, 0, 16, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0,
    ]);
    bytes.extend(restart_segmented_entropy(
        2,
        dc_category4_rgb_mcu_bits(false),
    ));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// Reference Rgb16 pixels for [`extended_12bit_ycbcr_8x8_jpeg`].
pub(crate) fn extended_12bit_ycbcr_8x8_rgb16() -> Vec<u8> {
    repeat_rgb16_pixels(8, 8, [2042, 2067, 2107])
}

/// Reference Rgb16 pixels for [`extended_12bit_ycbcr_restart_16x8_jpeg`].
pub(crate) fn extended_12bit_ycbcr_restart_16x8_rgb16() -> Vec<u8> {
    repeat_rgb16_pixels(16, 8, [2042, 2067, 2107])
}

/// A 32x8 12-bit extended sequential YCbCr 4:2:2 JPEG.
pub(crate) fn extended_12bit_ycbcr_422_32x8_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc1, 0x00, 17, 12, 0, 8, 0, 32, 3, 1, 0x21, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 21, 0x00, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0,
    ]);
    bytes.extend(dc_ycbcr422_entropy(false));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// A 32x8 12-bit extended sequential YCbCr 4:2:2 JPEG with DRI=1.
pub(crate) fn extended_12bit_ycbcr_422_restart_32x8_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc1, 0x00, 17, 12, 0, 8, 0, 32, 3, 1, 0x21, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 21, 0x00, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0,
    ]);
    bytes.extend(restart_segmented_entropy(
        2,
        dc_ycbcr422_uniform_mcu_bits(false),
    ));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// Reference Rgb16 pixels for [`extended_12bit_ycbcr_422_32x8_jpeg`].
pub(crate) fn extended_12bit_ycbcr_422_32x8_rgb16() -> Vec<u8> {
    let mut out = Vec::with_capacity(32 * 8 * 6);
    for _ in 0..8 {
        append_rgb16_run(&mut out, 15, [2042, 2067, 2107]);
        append_rgb16_run(&mut out, 1, [2047, 2066, 2099]);
        append_rgb16_run(&mut out, 1, [2058, 2063, 2085]);
        append_rgb16_run(&mut out, 15, [2064, 2061, 2078]);
    }
    out
}

/// Reference Rgb16 pixels for [`extended_12bit_ycbcr_422_restart_32x8_jpeg`].
pub(crate) fn extended_12bit_ycbcr_422_restart_32x8_rgb16() -> Vec<u8> {
    repeat_rgb16_pixels(32, 8, [2042, 2067, 2107])
}

/// A 32x32 12-bit extended sequential YCbCr 4:2:0 JPEG.
pub(crate) fn extended_12bit_ycbcr_420_32x32_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc1, 0x00, 17, 12, 0, 32, 0, 32, 3, 1, 0x22, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 21, 0x00, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0,
    ]);
    bytes.extend(dc_ycbcr420_entropy(false));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// A 32x32 12-bit extended sequential YCbCr 4:2:0 JPEG with DRI=1.
pub(crate) fn extended_12bit_ycbcr_420_restart_32x32_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc1, 0x00, 17, 12, 0, 32, 0, 32, 3, 1, 0x22, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 21, 0x00, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0,
    ]);
    bytes.extend(restart_segmented_entropy(
        4,
        dc_ycbcr420_uniform_mcu_bits(false),
    ));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// Reference Rgb16 pixels for [`extended_12bit_ycbcr_420_32x32_jpeg`].
pub(crate) fn extended_12bit_ycbcr_420_32x32_rgb16() -> Vec<u8> {
    let mut out = Vec::with_capacity(32 * 32 * 6);
    let cb_plane = ycbcr420_chroma_plane_for_fixture(2072, 2056, 2040, 2064);
    let cr_plane = ycbcr420_chroma_plane_for_fixture(2032, 2048, 2072, 2056);
    for y in 0..32 {
        for x in 0..32 {
            let cb = upsample_h2v2_12bit_for_fixture(&cb_plane, x, y);
            let cr = upsample_h2v2_12bit_for_fixture(&cr_plane, x, y);
            let (r, g, b) = ycbcr12_to_rgb16_for_fixture(2064, cb, cr);
            append_rgb16_pixel(&mut out, [r, g, b]);
        }
    }
    out
}

/// Reference Rgb16 pixels for [`extended_12bit_ycbcr_420_restart_32x32_jpeg`].
pub(crate) fn extended_12bit_ycbcr_420_restart_32x32_rgb16() -> Vec<u8> {
    repeat_rgb16_pixels(32, 32, [2042, 2067, 2107])
}

/// An 8x8 12-bit progressive grayscale JPEG with one DC-only scan.
pub(crate) fn progressive_12bit_grayscale_8x8_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[0xff, 0xc2, 0x00, 11, 12, 0, 8, 0, 8, 1, 1, 0x11, 0]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 0, 0]);
    bytes.push(0x00);
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// An 8x8 12-bit progressive APP14 RGB JPEG with one DC-only scan.
pub(crate) fn progressive_12bit_rgb_8x8_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[
        0xff, 0xee, 0x00, 0x0e, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc2, 0x00, 17, 12, 0, 8, 0, 8, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 0, 0,
    ]);
    bytes.extend(dc_category4_rgb_progressive_entropy());
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// An 8x8 12-bit progressive YCbCr 4:4:4 JPEG with one DC-only scan.
pub(crate) fn progressive_12bit_ycbcr_8x8_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc2, 0x00, 17, 12, 0, 8, 0, 8, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 0, 0,
    ]);
    bytes.extend(dc_category4_rgb_progressive_entropy());
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// A 32x8 12-bit progressive YCbCr 4:2:2 JPEG with one DC-only scan.
pub(crate) fn progressive_12bit_ycbcr_422_32x8_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc2, 0x00, 17, 12, 0, 8, 0, 32, 3, 1, 0x21, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 21, 0x00, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 0, 0,
    ]);
    bytes.extend(dc_ycbcr422_entropy(true));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// A 32x32 12-bit progressive YCbCr 4:2:0 JPEG with one DC-only scan.
pub(crate) fn progressive_12bit_ycbcr_420_32x32_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc2, 0x00, 17, 12, 0, 32, 0, 32, 3, 1, 0x22, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 21, 0x00, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 0, 0,
    ]);
    bytes.extend(dc_ycbcr420_entropy(true));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn dc_category4_rgb_entropy() -> Vec<u8> {
    pack_entropy_bits(dc_category4_rgb_mcu_bits(false))
}

fn dc_category4_rgb_mcu_bits(progressive: bool) -> Vec<bool> {
    let mut bits = Vec::new();
    for magnitude in [0b1000, 0b1100, 0b0111] {
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, magnitude, 4);
        if !progressive {
            push_bits(&mut bits, 0, 1);
        }
    }
    bits
}

fn dc_category4_rgb_progressive_entropy() -> Vec<u8> {
    pack_entropy_bits(dc_category4_rgb_mcu_bits(true))
}

fn dc_ycbcr422_entropy(progressive: bool) -> Vec<u8> {
    let mut bits = Vec::new();
    for magnitude in [
        Some(0b1000),
        None,
        Some(0b1100),
        Some(0b0111),
        None,
        None,
        Some(0b0111),
        Some(0b1000),
    ] {
        match magnitude {
            Some(value) => {
                push_bits(&mut bits, 1, 1);
                push_bits(&mut bits, value, 4);
            }
            None => push_bits(&mut bits, 0, 1),
        }
        if !progressive {
            push_bits(&mut bits, 0, 1);
        }
    }
    pack_entropy_bits(bits)
}

fn dc_ycbcr422_uniform_mcu_bits(progressive: bool) -> Vec<bool> {
    let mut bits = Vec::new();
    for magnitude in [Some(0b1000), None, Some(0b1100), Some(0b0111)] {
        match magnitude {
            Some(value) => {
                push_bits(&mut bits, 1, 1);
                push_bits(&mut bits, value, 4);
            }
            None => push_bits(&mut bits, 0, 1),
        }
        if !progressive {
            push_bits(&mut bits, 0, 1);
        }
    }
    bits
}

fn dc_ycbcr420_entropy(progressive: bool) -> Vec<u8> {
    let mut bits = Vec::new();
    let mcu_blocks = [
        [Some(0b1000), None, None, None, Some(0b1100), Some(0b0111)],
        [None, None, None, None, Some(0b0111), Some(0b1000)],
        [None, None, None, None, Some(0b0111), Some(0b1100)],
        [None, None, None, None, Some(0b1100), Some(0b0111)],
    ];
    for mcu in mcu_blocks {
        for magnitude in mcu {
            match magnitude {
                Some(value) => {
                    push_bits(&mut bits, 1, 1);
                    push_bits(&mut bits, value, 4);
                }
                None => push_bits(&mut bits, 0, 1),
            }
            if !progressive {
                push_bits(&mut bits, 0, 1);
            }
        }
    }
    pack_entropy_bits(bits)
}

fn dc_ycbcr420_uniform_mcu_bits(progressive: bool) -> Vec<bool> {
    let mut bits = Vec::new();
    for magnitude in [Some(0b1000), None, None, None, Some(0b1100), Some(0b0111)] {
        match magnitude {
            Some(value) => {
                push_bits(&mut bits, 1, 1);
                push_bits(&mut bits, value, 4);
            }
            None => push_bits(&mut bits, 0, 1),
        }
        if !progressive {
            push_bits(&mut bits, 0, 1);
        }
    }
    bits
}

fn restart_segmented_entropy(mcus: usize, mcu_bits: Vec<bool>) -> Vec<u8> {
    let mut entropy = Vec::new();
    for mcu in 0..mcus {
        entropy.extend(pack_entropy_bits(mcu_bits.clone()));
        if mcu + 1 != mcus {
            entropy.extend_from_slice(&[0xff, 0xd0 | ((mcu as u8) & 0x07)]);
        }
    }
    entropy
}

fn append_rgb16_run(out: &mut Vec<u8>, len: usize, rgb: [u16; 3]) {
    for _ in 0..len {
        append_rgb16_pixel(out, rgb);
    }
}

fn append_rgb16_pixel(out: &mut Vec<u8>, rgb: [u16; 3]) {
    for sample in rgb {
        out.extend_from_slice(&sample.to_le_bytes());
    }
}

fn ycbcr420_chroma_row_for_fixture(left: u16, right: u16) -> [u16; 16] {
    let mut row = [0u16; 16];
    row[..8].fill(left);
    row[8..].fill(right);
    row
}

fn ycbcr420_chroma_plane_for_fixture(
    top_left: u16,
    top_right: u16,
    bottom_left: u16,
    bottom_right: u16,
) -> [[u16; 16]; 16] {
    let top = ycbcr420_chroma_row_for_fixture(top_left, top_right);
    let bottom = ycbcr420_chroma_row_for_fixture(bottom_left, bottom_right);
    core::array::from_fn(|y| if y < 8 { top } else { bottom })
}

fn upsample_h2v2_12bit_for_fixture(
    plane: &[[u16; 16]; 16],
    output_x: usize,
    output_y: usize,
) -> u16 {
    let chroma_y = output_y / 2;
    let current = &plane[chroma_y];
    let near_y = if output_y.is_multiple_of(2) {
        chroma_y.saturating_sub(1)
    } else {
        (chroma_y + 1).min(15)
    };
    let near = &plane[near_y];
    let sample = output_x / 2;
    let colsum =
        |row: &[u16; 16], index: usize| 3 * u32::from(current[index]) + u32::from(row[index]);
    let this = colsum(near, sample);
    match output_x {
        0 => ((this * 4 + 8) >> 4) as u16,
        31 => ((this * 4 + 7) >> 4) as u16,
        _ if output_x.is_multiple_of(2) => {
            let last = colsum(near, sample - 1);
            ((this * 3 + last + 8) >> 4) as u16
        }
        _ => {
            let next = colsum(near, sample + 1);
            ((this * 3 + next + 7) >> 4) as u16
        }
    }
}

fn ycbcr12_to_rgb16_for_fixture(y: u16, cb: u16, cr: u16) -> (u16, u16, u16) {
    const FIX_1_40200: i32 = 91_881;
    const FIX_0_34414: i32 = 22_554;
    const FIX_0_71414: i32 = 46_802;
    const FIX_1_77200: i32 = 116_130;
    const ROUND: i32 = 1 << 15;

    let y = i32::from(y);
    let cb_centered = i32::from(cb) - 2048;
    let cr_centered = i32::from(cr) - 2048;
    let r = y + ((FIX_1_40200 * cr_centered + ROUND) >> 16);
    let g = y - ((FIX_0_34414 * cb_centered + FIX_0_71414 * cr_centered + ROUND) >> 16);
    let b = y + ((FIX_1_77200 * cb_centered + ROUND) >> 16);

    (
        r.clamp(0, 4095) as u16,
        g.clamp(0, 4095) as u16,
        b.clamp(0, 4095) as u16,
    )
}

pub(crate) const LOSSLESS_GRAYSCALE_3X3_PIXELS: [u8; 9] =
    [130, 132, 136, 128, 135, 142, 125, 137, 150];

pub(crate) const LOSSLESS_GRAYSCALE_16BIT_3X3_PIXELS: [u16; 9] = [
    33000, 33012, 33025, 32990, 33020, 33044, 32970, 33030, 33080,
];

pub(crate) const LOSSLESS_RGB_3X3_PIXELS: [u8; 27] = [
    130, 50, 200, 132, 53, 198, 136, 55, 195, 128, 54, 202, 135, 56, 199, 142, 59, 196, 125, 57,
    204, 137, 60, 201, 150, 64, 198,
];

/// A 3x3 SOF3 lossless grayscale JPEG using predictor 1..=7.
pub(crate) fn lossless_predictor_grayscale_3x3_jpeg(predictor: u8) -> Vec<u8> {
    lossless_grayscale_jpeg(3, 3, predictor, &LOSSLESS_GRAYSCALE_3X3_PIXELS)
}

/// A 3x3 SOF3 lossless APP14 RGB JPEG using predictor 1..=7.
pub(crate) fn lossless_predictor_rgb_3x3_jpeg(predictor: u8) -> Vec<u8> {
    lossless_rgb_jpeg(3, 3, predictor, &LOSSLESS_RGB_3X3_PIXELS)
}

/// A 3x3 SOF3 lossless APP14 RGB JPEG with row-boundary restart markers.
pub(crate) fn lossless_restart_predictor_rgb_3x3_jpeg(predictor: u8) -> Vec<u8> {
    lossless_rgb_restart_jpeg(3, 3, predictor, 3, &LOSSLESS_RGB_3X3_PIXELS)
}

/// A 3x3 SOF3 lossless grayscale JPEG with row-boundary restart markers.
pub(crate) fn lossless_restart_predictor_grayscale_3x3_jpeg(predictor: u8) -> Vec<u8> {
    lossless_grayscale_restart_jpeg(3, 3, predictor, 3, &LOSSLESS_GRAYSCALE_3X3_PIXELS)
}

/// A 3x3 16-bit SOF3 lossless grayscale JPEG using predictor 1..=7.
pub(crate) fn lossless_predictor_grayscale_16bit_3x3_jpeg(predictor: u8) -> Vec<u8> {
    lossless_grayscale_16bit_jpeg(3, 3, predictor, &LOSSLESS_GRAYSCALE_16BIT_3X3_PIXELS)
}

/// A 3x3 16-bit SOF3 lossless grayscale JPEG with row-boundary restart markers.
pub(crate) fn lossless_restart_predictor_grayscale_16bit_3x3_jpeg(predictor: u8) -> Vec<u8> {
    lossless_grayscale_16bit_restart_jpeg(3, 3, predictor, 3, &LOSSLESS_GRAYSCALE_16BIT_3X3_PIXELS)
}

fn lossless_grayscale_jpeg(width: u16, height: u16, predictor: u8, samples: &[u8]) -> Vec<u8> {
    assert_eq!(samples.len(), usize::from(width) * usize::from(height));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xc3, 0x00, 11, 8]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&[1, 1, 0x11, 0]);
    let mut dht = Vec::new();
    dht.push(0x00);
    dht.extend_from_slice(&[0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    dht.extend(0..=8);
    bytes.extend_from_slice(&[0xff, 0xc4]);
    bytes.extend_from_slice(&(dht.len() as u16 + 2).to_be_bytes());
    bytes.extend(dht);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, predictor, 0, 0]);
    bytes.extend(lossless_entropy(width, predictor, samples));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn lossless_rgb_jpeg(width: u16, height: u16, predictor: u8, samples: &[u8]) -> Vec<u8> {
    assert_eq!(samples.len(), usize::from(width) * usize::from(height) * 3);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[
        0xff, 0xee, 0x00, 0x0e, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ]);
    bytes.extend_from_slice(&[0xff, 0xc3, 0x00, 17, 8]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&[3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0]);
    let mut dht = Vec::new();
    dht.push(0x00);
    dht.extend_from_slice(&[0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    dht.extend(0..=8);
    bytes.extend_from_slice(&[0xff, 0xc4]);
    bytes.extend_from_slice(&(dht.len() as u16 + 2).to_be_bytes());
    bytes.extend(dht);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, predictor, 0, 0,
    ]);
    bytes.extend(lossless_rgb_entropy(width, predictor, samples));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn lossless_rgb_restart_jpeg(
    width: u16,
    height: u16,
    predictor: u8,
    restart_interval: u16,
    samples: &[u8],
) -> Vec<u8> {
    assert_eq!(samples.len(), usize::from(width) * usize::from(height) * 3);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[
        0xff, 0xee, 0x00, 0x0e, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ]);
    bytes.extend_from_slice(&[0xff, 0xc3, 0x00, 17, 8]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&[3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0]);
    let mut dht = Vec::new();
    dht.push(0x00);
    dht.extend_from_slice(&[0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    dht.extend(0..=8);
    bytes.extend_from_slice(&[0xff, 0xc4]);
    bytes.extend_from_slice(&(dht.len() as u16 + 2).to_be_bytes());
    bytes.extend(dht);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04]);
    bytes.extend_from_slice(&restart_interval.to_be_bytes());
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, predictor, 0, 0,
    ]);
    bytes.extend(lossless_rgb_entropy_with_restarts(
        width,
        predictor,
        samples,
        restart_interval,
    ));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn lossless_grayscale_restart_jpeg(
    width: u16,
    height: u16,
    predictor: u8,
    restart_interval: u16,
    samples: &[u8],
) -> Vec<u8> {
    assert_eq!(samples.len(), usize::from(width) * usize::from(height));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xc3, 0x00, 11, 8]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&[1, 1, 0x11, 0]);
    let mut dht = Vec::new();
    dht.push(0x00);
    dht.extend_from_slice(&[0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    dht.extend(0..=8);
    bytes.extend_from_slice(&[0xff, 0xc4]);
    bytes.extend_from_slice(&(dht.len() as u16 + 2).to_be_bytes());
    bytes.extend(dht);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04]);
    bytes.extend_from_slice(&restart_interval.to_be_bytes());
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, predictor, 0, 0]);
    bytes.extend(lossless_entropy_with_restarts(
        width,
        predictor,
        samples,
        restart_interval,
    ));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn lossless_grayscale_16bit_jpeg(
    width: u16,
    height: u16,
    predictor: u8,
    samples: &[u16],
) -> Vec<u8> {
    assert_eq!(samples.len(), usize::from(width) * usize::from(height));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xc3, 0x00, 11, 16]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&[1, 1, 0x11, 0]);
    let mut dht = Vec::new();
    dht.push(0x00);
    dht.extend_from_slice(&[0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    dht.extend(0..=15);
    bytes.extend_from_slice(&[0xff, 0xc4]);
    bytes.extend_from_slice(&(dht.len() as u16 + 2).to_be_bytes());
    bytes.extend(dht);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, predictor, 0, 0]);
    bytes.extend(lossless_entropy_16bit(width, predictor, samples));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn lossless_grayscale_16bit_restart_jpeg(
    width: u16,
    height: u16,
    predictor: u8,
    restart_interval: u16,
    samples: &[u16],
) -> Vec<u8> {
    assert_eq!(samples.len(), usize::from(width) * usize::from(height));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xc3, 0x00, 11, 16]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&[1, 1, 0x11, 0]);
    let mut dht = Vec::new();
    dht.push(0x00);
    dht.extend_from_slice(&[0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    dht.extend(0..=15);
    bytes.extend_from_slice(&[0xff, 0xc4]);
    bytes.extend_from_slice(&(dht.len() as u16 + 2).to_be_bytes());
    bytes.extend(dht);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04]);
    bytes.extend_from_slice(&restart_interval.to_be_bytes());
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, predictor, 0, 0]);
    bytes.extend(lossless_entropy_16bit_with_restarts(
        width,
        predictor,
        samples,
        restart_interval,
    ));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn lossless_entropy(width: u16, predictor: u8, samples: &[u8]) -> Vec<u8> {
    let width = usize::from(width);
    let mut bits = Vec::new();
    for (idx, &sample) in samples.iter().enumerate() {
        let x = idx % width;
        let y = idx / width;
        let predicted = lossless_predicted_value(samples, width, x, y, predictor);
        let diff = i32::from(sample) - predicted;
        let category = lossless_diff_category(diff);
        push_bits(&mut bits, u32::from(category), 4);
        if category != 0 {
            push_bits(&mut bits, lossless_magnitude_bits(diff, category), category);
        }
    }
    pack_entropy_bits(bits)
}

fn lossless_rgb_entropy(width: u16, predictor: u8, samples: &[u8]) -> Vec<u8> {
    let width = usize::from(width);
    let mut bits = Vec::new();
    for pixel in 0..samples.len() / 3 {
        let x = pixel % width;
        let y = pixel / width;
        for component in 0..3 {
            let sample = samples[pixel * 3 + component];
            let predicted =
                lossless_predicted_rgb_value(samples, width, x, y, component, predictor);
            let diff = i32::from(sample) - predicted;
            let category = lossless_diff_category(diff);
            push_bits(&mut bits, u32::from(category), 4);
            if category != 0 {
                push_bits(&mut bits, lossless_magnitude_bits(diff, category), category);
            }
        }
    }
    pack_entropy_bits(bits)
}

fn lossless_rgb_entropy_with_restarts(
    width: u16,
    predictor: u8,
    samples: &[u8],
    restart_interval: u16,
) -> Vec<u8> {
    assert!(restart_interval > 0);
    let width = usize::from(width);
    let restart_interval = usize::from(restart_interval);
    let pixel_count = samples.len() / 3;
    let mut out = Vec::new();
    let mut expected_rst = 0u8;
    for segment_start in (0..pixel_count).step_by(restart_interval) {
        let segment_end = (segment_start + restart_interval).min(pixel_count);
        let mut bits = Vec::new();
        for (segment_offset, pixel) in (segment_start..segment_end).enumerate() {
            let x = pixel % width;
            let y = pixel / width;
            for component in 0..3 {
                let sample = samples[pixel * 3 + component];
                let predicted = if segment_offset == 0 {
                    128
                } else {
                    lossless_predicted_rgb_value(samples, width, x, y, component, predictor)
                };
                let diff = i32::from(sample) - predicted;
                let category = lossless_diff_category(diff);
                push_bits(&mut bits, u32::from(category), 4);
                if category != 0 {
                    push_bits(&mut bits, lossless_magnitude_bits(diff, category), category);
                }
            }
        }
        out.extend(pack_entropy_bits(bits));
        if segment_end < pixel_count {
            out.extend_from_slice(&[0xff, 0xd0 + expected_rst]);
            expected_rst = (expected_rst + 1) & 0x07;
        }
    }
    out
}

fn lossless_entropy_with_restarts(
    width: u16,
    predictor: u8,
    samples: &[u8],
    restart_interval: u16,
) -> Vec<u8> {
    assert!(restart_interval > 0);
    let width = usize::from(width);
    let restart_interval = usize::from(restart_interval);
    let mut out = Vec::new();
    let mut expected_rst = 0u8;
    for segment_start in (0..samples.len()).step_by(restart_interval) {
        let segment_end = (segment_start + restart_interval).min(samples.len());
        let mut bits = Vec::new();
        for (segment_offset, idx) in (segment_start..segment_end).enumerate() {
            let sample = samples[idx];
            let x = idx % width;
            let y = idx / width;
            let predicted = if segment_offset == 0 {
                128
            } else {
                lossless_predicted_value(samples, width, x, y, predictor)
            };
            let diff = i32::from(sample) - predicted;
            let category = lossless_diff_category(diff);
            push_bits(&mut bits, u32::from(category), 4);
            if category != 0 {
                push_bits(&mut bits, lossless_magnitude_bits(diff, category), category);
            }
        }
        out.extend(pack_entropy_bits(bits));
        if segment_end < samples.len() {
            out.extend_from_slice(&[0xff, 0xd0 + expected_rst]);
            expected_rst = (expected_rst + 1) & 0x07;
        }
    }
    out
}

fn lossless_entropy_16bit(width: u16, predictor: u8, samples: &[u16]) -> Vec<u8> {
    let width = usize::from(width);
    let mut bits = Vec::new();
    for (idx, &sample) in samples.iter().enumerate() {
        let x = idx % width;
        let y = idx / width;
        let predicted = lossless_predicted_value_16bit(samples, width, x, y, predictor);
        let diff = i32::from(sample) - predicted;
        let category = lossless_diff_category(diff);
        push_bits(&mut bits, u32::from(category), 4);
        if category != 0 {
            push_bits(&mut bits, lossless_magnitude_bits(diff, category), category);
        }
    }
    pack_entropy_bits(bits)
}

fn lossless_entropy_16bit_with_restarts(
    width: u16,
    predictor: u8,
    samples: &[u16],
    restart_interval: u16,
) -> Vec<u8> {
    assert!(restart_interval > 0);
    let width = usize::from(width);
    let restart_interval = usize::from(restart_interval);
    let mut out = Vec::new();
    let mut expected_rst = 0u8;
    for segment_start in (0..samples.len()).step_by(restart_interval) {
        let segment_end = (segment_start + restart_interval).min(samples.len());
        let mut bits = Vec::new();
        for (segment_offset, idx) in (segment_start..segment_end).enumerate() {
            let sample = samples[idx];
            let x = idx % width;
            let y = idx / width;
            let predicted = if segment_offset == 0 {
                32768
            } else {
                lossless_predicted_value_16bit(samples, width, x, y, predictor)
            };
            let diff = i32::from(sample) - predicted;
            let category = lossless_diff_category(diff);
            push_bits(&mut bits, u32::from(category), 4);
            if category != 0 {
                push_bits(&mut bits, lossless_magnitude_bits(diff, category), category);
            }
        }
        out.extend(pack_entropy_bits(bits));
        if segment_end < samples.len() {
            out.extend_from_slice(&[0xff, 0xd0 + expected_rst]);
            expected_rst = (expected_rst + 1) & 0x07;
        }
    }
    out
}

fn lossless_predicted_value(
    samples: &[u8],
    width: usize,
    x: usize,
    y: usize,
    predictor: u8,
) -> i32 {
    let idx = y * width + x;
    if x == 0 && y == 0 {
        return 128;
    }
    if y == 0 {
        return i32::from(samples[idx - 1]);
    }
    if x == 0 {
        return i32::from(samples[idx - width]);
    }

    let ra = i32::from(samples[idx - 1]);
    let rb = i32::from(samples[idx - width]);
    let rc = i32::from(samples[idx - width - 1]);
    match predictor {
        1 => ra,
        2 => rb,
        3 => rc,
        4 => ra + rb - rc,
        5 => ra + ((rb - rc) >> 1),
        6 => rb + ((ra - rc) >> 1),
        7 => (ra + rb) >> 1,
        _ => 128,
    }
}

fn lossless_predicted_value_16bit(
    samples: &[u16],
    width: usize,
    x: usize,
    y: usize,
    predictor: u8,
) -> i32 {
    let idx = y * width + x;
    if x == 0 && y == 0 {
        return 32768;
    }
    if y == 0 {
        return i32::from(samples[idx - 1]);
    }
    if x == 0 {
        return i32::from(samples[idx - width]);
    }

    let ra = i32::from(samples[idx - 1]);
    let rb = i32::from(samples[idx - width]);
    let rc = i32::from(samples[idx - width - 1]);
    match predictor {
        1 => ra,
        2 => rb,
        3 => rc,
        4 => ra + rb - rc,
        5 => ra + ((rb - rc) >> 1),
        6 => rb + ((ra - rc) >> 1),
        7 => (ra + rb) >> 1,
        _ => 32768,
    }
}

fn lossless_predicted_rgb_value(
    samples: &[u8],
    width: usize,
    x: usize,
    y: usize,
    component: usize,
    predictor: u8,
) -> i32 {
    let idx = (y * width + x) * 3 + component;
    if x == 0 && y == 0 {
        return 128;
    }
    if y == 0 {
        return i32::from(samples[idx - 3]);
    }
    if x == 0 {
        return i32::from(samples[idx - width * 3]);
    }

    let ra = i32::from(samples[idx - 3]);
    let rb = i32::from(samples[idx - width * 3]);
    let rc = i32::from(samples[idx - (width + 1) * 3]);
    match predictor {
        1 => ra,
        2 => rb,
        3 => rc,
        4 => ra + rb - rc,
        5 => ra + ((rb - rc) >> 1),
        6 => rb + ((ra - rc) >> 1),
        7 => (ra + rb) >> 1,
        _ => 128,
    }
}

fn lossless_diff_category(diff: i32) -> u8 {
    if diff == 0 {
        return 0;
    }
    let magnitude = diff.unsigned_abs();
    (32 - magnitude.leading_zeros()) as u8
}

fn lossless_magnitude_bits(diff: i32, category: u8) -> u32 {
    if diff >= 0 {
        return diff as u32;
    }
    (diff + ((1i32 << category) - 1)) as u32
}

fn push_bits(bits: &mut Vec<bool>, value: u32, count: u8) {
    for bit in (0..count).rev() {
        bits.push(((value >> bit) & 1) != 0);
    }
}

fn pack_entropy_bits(mut bits: Vec<bool>) -> Vec<u8> {
    while bits.len() % 8 != 0 {
        bits.push(true);
    }
    let mut out = Vec::new();
    for chunk in bits.chunks_exact(8) {
        let mut byte = 0u8;
        for &bit in chunk {
            byte = (byte << 1) | u8::from(bit);
        }
        out.push(byte);
        if byte == 0xff {
            out.push(0x00);
        }
    }
    out
}

/// An 8x8 Adobe APP14 CMYK JPEG whose four decoded channels are all 128.
pub(crate) fn cmyk_8x8_jpeg() -> Vec<u8> {
    four_component_8x8_jpeg(Some(0))
}

/// An 8x8 Adobe APP14 YCCK JPEG whose four decoded channels are all 128.
pub(crate) fn ycck_8x8_jpeg() -> Vec<u8> {
    four_component_8x8_jpeg(Some(2))
}

/// Reference RGB pixels for [`cmyk_8x8_jpeg`] and [`ycck_8x8_jpeg`].
pub(crate) fn four_component_8x8_rgb() -> Vec<u8> {
    vec![64; 8 * 8 * 3]
}

/// An 8×8 baseline JPEG with 4:4:4 sampling.
pub(crate) fn baseline_444_8x8_jpeg() -> Vec<u8> {
    include_bytes!("../../fixtures/conformance/baseline_444_8x8.jpg").to_vec()
}

/// Reference pixels for [`baseline_444_8x8_jpeg`].
pub(crate) fn baseline_444_8x8_rgb() -> Vec<u8> {
    include_bytes!("../../fixtures/conformance/baseline_444_8x8.rgb").to_vec()
}

/// A 16×8 baseline JPEG with 4:2:2 sampling.
pub(crate) fn baseline_422_16x8_jpeg() -> Vec<u8> {
    include_bytes!("../../fixtures/conformance/baseline_422_16x8.jpg").to_vec()
}

/// Reference pixels for [`baseline_422_16x8_jpeg`].
pub(crate) fn baseline_422_16x8_rgb() -> Vec<u8> {
    include_bytes!("../../fixtures/conformance/baseline_422_16x8.rgb").to_vec()
}

/// A 32×16 baseline JPEG with 4:2:0 sampling and restart coding.
pub(crate) fn baseline_420_restart_32x16_jpeg() -> Vec<u8> {
    include_bytes!("../../fixtures/conformance/baseline_420_restart_32x16.jpg").to_vec()
}

/// Reference pixels for [`baseline_420_restart_32x16_jpeg`].
pub(crate) fn baseline_420_restart_32x16_rgb() -> Vec<u8> {
    include_bytes!("../../fixtures/conformance/baseline_420_restart_32x16.rgb").to_vec()
}

/// An 8×8 APP14 RGB JPEG with constant pixel value `(200, 20, 10)`.
pub(crate) fn rgb_app14_8x8_jpeg() -> Vec<u8> {
    vec![
        0xff, 0xd8, 0xff, 0xee, 0x00, 0x0e, 0x41, 0x64, 0x6f, 0x62, 0x65, 0x00, 0x64, 0x00, 0x00,
        0x00, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0xff, 0xc0, 0x00,
        0x11, 0x08, 0x00, 0x08, 0x00, 0x08, 0x03, 0x52, 0x11, 0x00, 0x47, 0x11, 0x00, 0x42, 0x11,
        0x00, 0xff, 0xc4, 0x00, 0x1f, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0xff, 0xc4, 0x00, 0xb5, 0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02,
        0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7d, 0x01, 0x02, 0x03, 0x00, 0x04,
        0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
        0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62,
        0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a,
        0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a,
        0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
        0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2,
        0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,
        0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3,
        0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xff, 0xda, 0x00, 0x0c, 0x03, 0x52, 0x00, 0x47,
        0x00, 0x42, 0x00, 0x00, 0x3f, 0x00, 0xfe, 0x90, 0x2b, 0xf8, 0x9f, 0xaf, 0xe1, 0x3e, 0xbf,
        0xff, 0xd9,
    ]
}

/// Reference pixels for [`rgb_app14_8x8_jpeg`].
pub(crate) fn rgb_app14_8x8_rgb() -> Vec<u8> {
    let mut out = Vec::with_capacity(8 * 8 * 3);
    for _ in 0..64 {
        out.extend_from_slice(&[200, 20, 10]);
    }
    out
}

/// A progressive 8×8 JPEG with 10 SOS markers.
pub(crate) fn progressive_8x8_jpeg() -> Vec<u8> {
    vec![
        0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00,
        0x01, 0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x03, 0x02,
        0x02, 0x03, 0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05, 0x08, 0x05, 0x05, 0x04, 0x04,
        0x05, 0x0a, 0x07, 0x07, 0x06, 0x08, 0x0c, 0x0a, 0x0c, 0x0c, 0x0b, 0x0a, 0x0b, 0x0b, 0x0d,
        0x0e, 0x12, 0x10, 0x0d, 0x0e, 0x11, 0x0e, 0x0b, 0x0b, 0x10, 0x16, 0x10, 0x11, 0x13, 0x14,
        0x15, 0x15, 0x15, 0x0c, 0x0f, 0x17, 0x18, 0x16, 0x14, 0x18, 0x12, 0x14, 0x15, 0x14, 0xff,
        0xdb, 0x00, 0x43, 0x01, 0x03, 0x04, 0x04, 0x05, 0x04, 0x05, 0x09, 0x05, 0x05, 0x09, 0x14,
        0x0d, 0x0b, 0x0d, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0xff, 0xc2, 0x00, 0x11, 0x08, 0x00, 0x08,
        0x00, 0x08, 0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xff, 0xc4, 0x00,
        0x15, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x06, 0xff, 0xc4, 0x00, 0x15, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x06, 0xff, 0xda,
        0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x10, 0x03, 0x10, 0x00, 0x00, 0x01, 0x88, 0x13, 0x6f,
        0x7f, 0xff, 0xc4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x01, 0x05, 0x02, 0x7f, 0xff, 0xc4, 0x00, 0x14, 0x11, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xda, 0x00,
        0x08, 0x01, 0x03, 0x01, 0x01, 0x3f, 0x01, 0x7f, 0xff, 0xc4, 0x00, 0x14, 0x11, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xff, 0xda, 0x00, 0x08, 0x01, 0x02, 0x01, 0x01, 0x3f, 0x01, 0x7f, 0xff, 0xc4, 0x00, 0x14,
        0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x06, 0x3f, 0x02, 0x7f, 0xff,
        0xc4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x3f,
        0x21, 0x7f, 0xff, 0xda, 0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x10, 0xf7, 0xff, 0xc4, 0x00, 0x14, 0x11, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xda, 0x00, 0x08, 0x01, 0x03,
        0x01, 0x01, 0x3f, 0x10, 0x7f, 0xff, 0xc4, 0x00, 0x14, 0x11, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xda, 0x00,
        0x08, 0x01, 0x02, 0x01, 0x01, 0x3f, 0x10, 0x7f, 0xff, 0xc4, 0x00, 0x14, 0x10, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x3f, 0x10, 0x7f, 0xff, 0xd9,
    ]
}

fn four_component_8x8_jpeg(app14_transform: Option<u8>) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    if let Some(transform) = app14_transform {
        bytes.extend_from_slice(&[
            0xff, 0xee, 0x00, 0x0e, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00, 0x00,
            0x00, transform,
        ]);
    }
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(1u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc0, 0x00, 20, 8, 0, 8, 0, 8, 4, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0, 4, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0e, 4, 1, 0x00, 2, 0x00, 3, 0x00, 4, 0x00, 0, 63, 0, 0x00, 0xff, 0xd9,
    ]);
    bytes
}
