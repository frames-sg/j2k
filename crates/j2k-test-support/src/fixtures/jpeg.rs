// SPDX-License-Identifier: MIT OR Apache-2.0

/// Minimal grayscale JPEG with caller-provided dimensions and one entropy byte.
///
/// This is intentionally not a complete image for large dimensions; tests use
/// it to exercise header validation and row-streaming paths without allocating
/// full-image entropy payloads.
pub fn minimal_grayscale_jpeg_with_dimensions(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = grayscale_jpeg_header(width, height);
    bytes.extend_from_slice(&[0x00, 0xff, 0xd9]);
    bytes
}

/// Baseline grayscale JPEG with one zero-DC entropy byte per MCU.
pub fn baseline_grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = grayscale_jpeg_header(width, height);
    let mcu_cols = u32::from(width).div_ceil(8);
    let mcu_rows = u32::from(height).div_ceil(8);
    let mcu_count = (mcu_cols * mcu_rows) as usize;
    bytes.extend(core::iter::repeat_n(0x00, mcu_count));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// Minimal 16x16 baseline JPEG with 4:2:0 sampling.
pub fn minimal_baseline_jpeg() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&[0xff, 0xd8]);
    out.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    out.extend(core::iter::repeat_n(1u8, 64));
    out.extend_from_slice(&[
        0xff, 0xc0, 0x00, 17, 8, 0, 16, 0, 16, 3, 1, 0x22, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    out.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xaa,
    ]);
    out.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xbb,
    ]);
    out.extend_from_slice(&[0xff, 0xda, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);
    out.extend_from_slice(&[0x00, 0xff, 0xd9]);
    out
}

/// Minimal baseline JPEG with a DRI marker inserted before SOS.
///
/// # Panics
///
/// Panics if the generated minimal fixture no longer contains an SOS marker.
pub fn minimal_baseline_jpeg_with_restart_interval(interval: u16) -> Vec<u8> {
    let mut bytes = minimal_baseline_jpeg();
    let sos_pos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("minimal fixture includes SOS marker");
    let [interval_high, interval_low] = interval.to_be_bytes();
    bytes.splice(
        sos_pos..sos_pos,
        [0xff, 0xdd, 0x00, 0x04, interval_high, interval_low],
    );
    bytes
}

/// Restart-coded grayscale JPEG with one zero-DC block per MCU.
pub fn restart_coded_grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = grayscale_jpeg_prefix(width, height);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    append_grayscale_huffman_and_scan_header(&mut bytes);

    let mcu_cols = u32::from(width).div_ceil(8);
    let mcu_rows = u32::from(height).div_ceil(8);
    let mcu_count = (mcu_cols * mcu_rows) as usize;
    for mcu in 0..mcu_count {
        bytes.push(0x00);
        if mcu + 1 != mcu_count {
            bytes.extend_from_slice(&[0xff, 0xd0 | restart_index(mcu)]);
        }
    }

    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn grayscale_jpeg_header(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = grayscale_jpeg_prefix(width, height);
    append_grayscale_huffman_and_scan_header(&mut bytes);
    bytes
}

fn restart_index(mcu: usize) -> u8 {
    u8::try_from(mcu & 0x07).expect("restart index is three bits")
}

fn grayscale_jpeg_prefix(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = Vec::new();
    let [height_high, height_low] = height.to_be_bytes();
    let [width_high, width_low] = width.to_be_bytes();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(core::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff,
        0xc0,
        0x00,
        11,
        8,
        height_high,
        height_low,
        width_high,
        width_low,
        1,
        1,
        0x11,
        0,
    ]);
    bytes
}

fn append_grayscale_huffman_and_scan_header(bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);
}
