// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

/// Generates deterministic RGB8 pixels for tests and benches.
pub fn patterned_rgb8(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 17 + y * 3) & 0xFF) as u8);
            pixels.push(((x * 5 + y * 11 + 40) & 0xFF) as u8);
            pixels.push(((x * 13 + y * 7 + 90) & 0xFF) as u8);
        }
    }
    pixels
}

/// Generates deterministic grayscale pixels for tests and benches.
pub fn patterned_gray8(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 19 + y * 23) & 0xFF) as u8);
        }
    }
    pixels
}

/// Generates a simple deterministic gradient for reference codec parity cases.
pub fn gradient_u8(width: u32, height: u32, channels: usize) -> Vec<u8> {
    gradient_variant_u8(width, height, channels, 0)
}

/// Generates a deterministic gradient variant keyed by `seed`.
pub fn gradient_variant_u8(width: u32, height: u32, channels: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(width as usize * height as usize * channels);
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                out.push(((x + y + seed * 13 + (c as u32 * 17)) & 0xFF) as u8);
            }
        }
    }
    out
}

/// Generates deterministic RGB8 pixels used by GPU upload/decode benches.
pub fn gpu_bench_rgb8(width: u32, height: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.push(((x * 13 + y * 3) & 0xff) as u8);
            rgb.push(((x * 5 + y * 11 + (x ^ y)) & 0xff) as u8);
            rgb.push(((x * 7 + y * 17 + (x.wrapping_mul(y) >> 5)) & 0xff) as u8);
        }
    }
    rgb
}

/// Generates contiguous RGB8 tiles for JPEG baseline encode benches.
pub fn patterned_rgb8_tiles(width: u32, height: u32, tile_count: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3 * tile_count);
    for tile in 0..tile_count as u32 {
        for y in 0..height {
            for x in 0..width {
                rgb.push(((x * 13 + y * 3 + tile * 29) & 0xff) as u8);
                rgb.push(((x * 5 + y * 11 + (x ^ y) + tile * 17) & 0xff) as u8);
                rgb.push(((x * 7 + y * 17 + (x.wrapping_mul(y) >> 5) + tile * 23) & 0xff) as u8);
            }
        }
    }
    rgb
}

/// Builds a minimal JPEG 2000 codestream header for inspect/parser tests.
pub fn minimal_j2k_codestream() -> Vec<u8> {
    let mut bytes = vec![0xFF, 0x4F];
    let mut siz = Vec::new();
    push_u16(&mut siz, 0);
    push_u32(&mut siz, 128);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 0);
    push_u16(&mut siz, 3);
    for _ in 0..3 {
        siz.extend_from_slice(&[0x07, 0x01, 0x01]);
    }
    bytes.extend_from_slice(&[0xFF, 0x51]);
    push_u16(&mut bytes, (siz.len() + 2) as u16);
    bytes.extend_from_slice(&siz);

    let cod = [0x00, 0x00, 0x00, 0x01, 0x01, 0x05, 0x04, 0x04, 0x00, 0x01];
    bytes.extend_from_slice(&[0xFF, 0x52]);
    push_u16(&mut bytes, (cod.len() + 2) as u16);
    bytes.extend_from_slice(&cod);
    bytes.extend_from_slice(&[0xFF, 0x90, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    bytes
}

/// Wraps [`minimal_j2k_codestream`] in a minimal JP2 container.
pub fn minimal_jp2() -> Vec<u8> {
    let codestream = minimal_j2k_codestream();
    wrap_codestream_jp2(&codestream, 128, 64, 3, 8, 16)
}

/// Wraps a JPEG 2000 codestream in a minimal JP2 container.
pub fn wrap_codestream_jp2(
    codestream: &[u8],
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    colorspace_enum: u32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
    bytes.extend_from_slice(&[
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);

    let bpc = bit_depth.saturating_sub(1);
    bytes.extend_from_slice(&[
        0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r',
    ]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&components.to_be_bytes());
    bytes.extend_from_slice(&[bpc, 7, 0, 0]);
    bytes.extend_from_slice(&[0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0]);
    bytes.extend_from_slice(&colorspace_enum.to_be_bytes());

    let len = (8 + codestream.len()) as u32;
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(b"jp2c");
    bytes.extend_from_slice(codestream);
    bytes
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::{minimal_j2k_codestream, minimal_jp2, wrap_codestream_jp2};

    #[test]
    fn minimal_j2k_codestream_has_j2k_magic_and_siz_marker() {
        let codestream = minimal_j2k_codestream();

        assert!(codestream.starts_with(&[0xFF, 0x4F]));
        assert!(codestream.windows(2).any(|marker| marker == [0xFF, 0x51]));
    }

    #[test]
    fn minimal_jp2_wraps_the_minimal_codestream() {
        let jp2 = minimal_jp2();

        assert!(jp2.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']));
        assert!(jp2.windows(4).any(|box_type| box_type == b"jp2c"));
        assert!(jp2.windows(2).any(|marker| marker == [0xFF, 0x4F]));
    }

    #[test]
    fn jp2_wrapper_writes_image_header_dimensions_and_colorspace() {
        let jp2 = wrap_codestream_jp2(&[0xFF, 0x4F], 320, 240, 3, 8, 16);

        assert!(jp2.windows(4).any(|box_type| box_type == b"jp2h"));
        assert!(jp2.windows(4).any(|value| value == 240u32.to_be_bytes()));
        assert!(jp2.windows(4).any(|value| value == 320u32.to_be_bytes()));
        assert!(jp2.windows(4).any(|value| value == 16u32.to_be_bytes()));
    }
}
