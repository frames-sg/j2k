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
