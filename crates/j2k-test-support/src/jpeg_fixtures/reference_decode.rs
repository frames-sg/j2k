// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) fn append_rgb16_run(out: &mut Vec<u8>, len: usize, rgb: [u16; 3]) {
    for _ in 0..len {
        append_rgb16_pixel(out, rgb);
    }
}

pub(super) fn append_rgb16_pixel(out: &mut Vec<u8>, rgb: [u16; 3]) {
    for sample in rgb {
        out.extend_from_slice(&sample.to_le_bytes());
    }
}

pub(super) fn ycbcr420_chroma_row_for_fixture(left: u16, right: u16) -> [u16; 16] {
    let mut row = [0u16; 16];
    row[..8].fill(left);
    row[8..].fill(right);
    row
}

pub(super) fn ycbcr420_chroma_plane_for_fixture(
    top_left: u16,
    top_right: u16,
    bottom_left: u16,
    bottom_right: u16,
) -> [[u16; 16]; 16] {
    let top = ycbcr420_chroma_row_for_fixture(top_left, top_right);
    let bottom = ycbcr420_chroma_row_for_fixture(bottom_left, bottom_right);
    core::array::from_fn(|y| if y < 8 { top } else { bottom })
}

pub(super) fn upsample_h2v2_12bit_for_fixture(
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

pub(super) fn ycbcr12_to_rgb16_for_fixture(y: u16, cb: u16, cr: u16) -> (u16, u16, u16) {
    const FIX_1_40200: i32 = 91_881;
    const FIX_0_34414: i32 = 22_554;
    const FIX_0_71414: i32 = 46_802;
    const FIX_1_77200: i32 = 116_130;
    const ROUND: i32 = 1 << 15;

    let y = i32::from(y);
    let blue_delta = i32::from(cb) - 2048;
    let red_delta = i32::from(cr) - 2048;
    let r = y + ((FIX_1_40200 * red_delta + ROUND) >> 16);
    let g = y - ((FIX_0_34414 * blue_delta + FIX_0_71414 * red_delta + ROUND) >> 16);
    let b = y + ((FIX_1_77200 * blue_delta + ROUND) >> 16);

    (
        r.clamp(0, 4095) as u16,
        g.clamp(0, 4095) as u16,
        b.clamp(0, 4095) as u16,
    )
}

pub(super) fn ycbcr8_pixels_to_rgb8(samples: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len());
    for pixel in samples.chunks_exact(3) {
        let (r, g, b) = ycbcr8_to_rgb8_for_fixture(pixel[0], pixel[1], pixel[2]);
        out.extend_from_slice(&[r, g, b]);
    }
    out
}

pub(super) fn ycbcr8_to_rgb8_for_fixture(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    const FIX_1_40200: i32 = 91_881;
    const FIX_0_34414: i32 = 22_554;
    const FIX_0_71414: i32 = 46_802;
    const FIX_1_77200: i32 = 116_130;
    const ROUND: i32 = 1 << 15;

    let y = i32::from(y);
    let blue_delta = i32::from(cb) - 128;
    let red_delta = i32::from(cr) - 128;
    let r = y + ((FIX_1_40200 * red_delta + ROUND) >> 16);
    let g = y - ((FIX_0_34414 * blue_delta + FIX_0_71414 * red_delta + ROUND) >> 16);
    let b = y + ((FIX_1_77200 * blue_delta + ROUND) >> 16);

    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

pub(super) fn ycbcr16_pixels_to_rgb16(samples: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for pixel in samples.chunks_exact(3) {
        let (r, g, b) = ycbcr16_to_rgb16_for_fixture(pixel[0], pixel[1], pixel[2]);
        out.extend_from_slice(&r.to_le_bytes());
        out.extend_from_slice(&g.to_le_bytes());
        out.extend_from_slice(&b.to_le_bytes());
    }
    out
}

pub(super) fn ycbcr16_to_rgb16_for_fixture(y: u16, cb: u16, cr: u16) -> (u16, u16, u16) {
    const FIX_1_40200: i64 = 91_881;
    const FIX_0_34414: i64 = 22_554;
    const FIX_0_71414: i64 = 46_802;
    const FIX_1_77200: i64 = 116_130;
    const ROUND: i64 = 1 << 15;

    let y = i64::from(y);
    let blue_delta = i64::from(cb) - 32768;
    let red_delta = i64::from(cr) - 32768;
    let r = y + ((FIX_1_40200 * red_delta + ROUND) >> 16);
    let g = y - ((FIX_0_34414 * blue_delta + FIX_0_71414 * red_delta + ROUND) >> 16);
    let b = y + ((FIX_1_77200 * blue_delta + ROUND) >> 16);

    (
        r.clamp(0, i64::from(u16::MAX)) as u16,
        g.clamp(0, i64::from(u16::MAX)) as u16,
        b.clamp(0, i64::from(u16::MAX)) as u16,
    )
}

#[derive(Clone, Copy)]
pub(super) enum ColorSpaceFixture {
    Rgb,
    YCbCr,
}

pub(super) fn lossless_422_planes_to_rgb8(
    color_space: ColorSpaceFixture,
    width: usize,
    height: usize,
    c0: &[u8],
    c1: &[u8],
    c2: &[u8],
) -> Vec<u8> {
    let chroma_width = width.div_ceil(2);
    let mut out = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let c1_row = &c1[y * chroma_width..(y + 1) * chroma_width];
        let c2_row = &c2[y * chroma_width..(y + 1) * chroma_width];
        for x in 0..width {
            let c0_sample = c0[y * width + x];
            let c1_sample = upsample_h2v1_8bit_for_fixture(c1_row, x);
            let c2_sample = upsample_h2v1_8bit_for_fixture(c2_row, x);
            let (r, g, b) = match color_space {
                ColorSpaceFixture::Rgb => (c0_sample, c1_sample, c2_sample),
                ColorSpaceFixture::YCbCr => {
                    ycbcr8_to_rgb8_for_fixture(c0_sample, c1_sample, c2_sample)
                }
            };
            out.extend_from_slice(&[r, g, b]);
        }
    }
    out
}

pub(super) fn lossless_420_planes_to_rgb8(
    color_space: ColorSpaceFixture,
    width: usize,
    height: usize,
    c0: &[u8],
    c1: &[u8],
    c2: &[u8],
) -> Vec<u8> {
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let mut out = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let c0_sample = c0[y * width + x];
            let c1_sample =
                upsample_h2v2_8bit_for_fixture(c1, chroma_width, chroma_height, width, x, y);
            let c2_sample =
                upsample_h2v2_8bit_for_fixture(c2, chroma_width, chroma_height, width, x, y);
            let (r, g, b) = match color_space {
                ColorSpaceFixture::Rgb => (c0_sample, c1_sample, c2_sample),
                ColorSpaceFixture::YCbCr => {
                    ycbcr8_to_rgb8_for_fixture(c0_sample, c1_sample, c2_sample)
                }
            };
            out.extend_from_slice(&[r, g, b]);
        }
    }
    out
}

pub(super) fn lossless_422_planes_to_rgb16(
    color_space: ColorSpaceFixture,
    width: usize,
    height: usize,
    c0: &[u16],
    c1: &[u16],
    c2: &[u16],
) -> Vec<u8> {
    let chroma_width = width.div_ceil(2);
    let mut out = Vec::with_capacity(width * height * 6);
    for y in 0..height {
        let c1_row = &c1[y * chroma_width..(y + 1) * chroma_width];
        let c2_row = &c2[y * chroma_width..(y + 1) * chroma_width];
        for x in 0..width {
            let c0_sample = c0[y * width + x];
            let c1_sample = upsample_h2v1_16bit_for_fixture(c1_row, x);
            let c2_sample = upsample_h2v1_16bit_for_fixture(c2_row, x);
            let (r, g, b) = match color_space {
                ColorSpaceFixture::Rgb => (c0_sample, c1_sample, c2_sample),
                ColorSpaceFixture::YCbCr => {
                    ycbcr16_to_rgb16_for_fixture(c0_sample, c1_sample, c2_sample)
                }
            };
            append_rgb16_pixel(&mut out, [r, g, b]);
        }
    }
    out
}

pub(super) fn lossless_420_planes_to_rgb16(
    color_space: ColorSpaceFixture,
    width: usize,
    height: usize,
    c0: &[u16],
    c1: &[u16],
    c2: &[u16],
) -> Vec<u8> {
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let mut out = Vec::with_capacity(width * height * 6);
    for y in 0..height {
        for x in 0..width {
            let c0_sample = c0[y * width + x];
            let c1_sample =
                upsample_h2v2_16bit_for_fixture(c1, chroma_width, chroma_height, width, x, y);
            let c2_sample =
                upsample_h2v2_16bit_for_fixture(c2, chroma_width, chroma_height, width, x, y);
            let (r, g, b) = match color_space {
                ColorSpaceFixture::Rgb => (c0_sample, c1_sample, c2_sample),
                ColorSpaceFixture::YCbCr => {
                    ycbcr16_to_rgb16_for_fixture(c0_sample, c1_sample, c2_sample)
                }
            };
            append_rgb16_pixel(&mut out, [r, g, b]);
        }
    }
    out
}

pub(super) fn upsample_h2v1_8bit_for_fixture(row: &[u8], output_x: usize) -> u8 {
    if row.len() == 1 {
        return row[0];
    }
    let sample = output_x / 2;
    if output_x == 0 {
        row[0]
    } else if output_x == row.len() * 2 - 1 {
        row[row.len() - 1]
    } else if output_x.is_multiple_of(2) {
        ((3 * u32::from(row[sample]) + u32::from(row[sample - 1]) + 2) / 4) as u8
    } else {
        ((3 * u32::from(row[sample]) + u32::from(row[sample + 1]) + 2) / 4) as u8
    }
}

pub(super) fn upsample_h2v2_8bit_for_fixture(
    plane: &[u8],
    chroma_width: usize,
    chroma_height: usize,
    output_width: usize,
    output_x: usize,
    output_y: usize,
) -> u8 {
    let chroma_y = output_y / 2;
    let current = &plane[chroma_y * chroma_width..(chroma_y + 1) * chroma_width];
    let near_y = if output_y.is_multiple_of(2) {
        chroma_y.saturating_sub(1)
    } else {
        (chroma_y + 1).min(chroma_height - 1)
    };
    let near = &plane[near_y * chroma_width..(near_y + 1) * chroma_width];
    let colsum = |index: usize| 3 * u32::from(current[index]) + u32::from(near[index]);
    if chroma_width == 1 {
        return ((4 * colsum(0) + 8) >> 4) as u8;
    }

    let sample = output_x / 2;
    let this = colsum(sample);
    match output_x {
        0 => ((this * 4 + 8) >> 4) as u8,
        _ if output_x == output_width - 1 => ((this * 4 + 7) >> 4) as u8,
        _ if output_x.is_multiple_of(2) => {
            let last = colsum(sample - 1);
            ((this * 3 + last + 8) >> 4) as u8
        }
        _ => {
            let next = colsum(sample + 1);
            ((this * 3 + next + 7) >> 4) as u8
        }
    }
}

pub(super) fn upsample_h2v1_16bit_for_fixture(row: &[u16], output_x: usize) -> u16 {
    if row.len() == 1 {
        return row[0];
    }
    let sample = output_x / 2;
    if output_x == 0 {
        row[0]
    } else if output_x == row.len() * 2 - 1 {
        row[row.len() - 1]
    } else if output_x.is_multiple_of(2) {
        ((3 * u32::from(row[sample]) + u32::from(row[sample - 1]) + 2) / 4) as u16
    } else {
        ((3 * u32::from(row[sample]) + u32::from(row[sample + 1]) + 2) / 4) as u16
    }
}

pub(super) fn upsample_h2v2_16bit_for_fixture(
    plane: &[u16],
    chroma_width: usize,
    chroma_height: usize,
    output_width: usize,
    output_x: usize,
    output_y: usize,
) -> u16 {
    let chroma_y = output_y / 2;
    let current = &plane[chroma_y * chroma_width..(chroma_y + 1) * chroma_width];
    let near_y = if output_y.is_multiple_of(2) {
        chroma_y.saturating_sub(1)
    } else {
        (chroma_y + 1).min(chroma_height - 1)
    };
    let near = &plane[near_y * chroma_width..(near_y + 1) * chroma_width];
    let colsum = |index: usize| 3 * u32::from(current[index]) + u32::from(near[index]);
    if chroma_width == 1 {
        return ((4 * colsum(0) + 8) >> 4) as u16;
    }

    let sample = output_x / 2;
    let this = colsum(sample);
    match output_x {
        0 => ((this * 4 + 8) >> 4) as u16,
        _ if output_x == output_width - 1 => ((this * 4 + 7) >> 4) as u16,
        _ if output_x.is_multiple_of(2) => {
            let last = colsum(sample - 1);
            ((this * 3 + last + 8) >> 4) as u16
        }
        _ => {
            let next = colsum(sample + 1);
            ((this * 3 + next + 7) >> 4) as u16
        }
    }
}
