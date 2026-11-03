// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

pub(super) fn grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
    let [height_hi, height_lo] = height.to_be_bytes();
    let [width_hi, width_lo] = width.to_be_bytes();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(core::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc0, 0x00, 11, 8, height_hi, height_lo, width_hi, width_lo, 1, 1, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);

    let mcu_cols = u32::from(width).div_ceil(8);
    let mcu_rows = u32::from(height).div_ceil(8);
    let mcu_count = (mcu_cols * mcu_rows) as usize;
    bytes.extend(core::iter::repeat_n(0x00, mcu_count));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// Baseline JPEG of deterministic noise, so every block carries real AC
/// symbols and `FF 00` stuffing occurs throughout the scan.
pub(super) fn noisy_jpeg(
    width: u32,
    height: u32,
    subsampling: crate::JpegSubsampling,
    restart_interval: Option<u16>,
) -> Vec<u8> {
    let rgb = crate::entropy::sequential::stream_tests::noisy_rgb8(width, height, width ^ height);
    crate::encode_jpeg_baseline(
        crate::JpegSamples::Rgb8 {
            width,
            height,
            data: &rgb,
        },
        crate::JpegEncodeOptions {
            quality: 92,
            subsampling,
            restart_interval,
            backend: crate::JpegBackend::Cpu,
        },
    )
    .expect("encode noisy fixture")
    .data
}
