// SPDX-License-Identifier: Apache-2.0

//! Pixel conversion and region projection helpers for tests.

/// Rectangle type used by test helper APIs without depending on `j2k-core`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl PixelRect {
    /// Creates a pixel rectangle.
    pub const fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }
}

/// Returns a centered rectangle capped to the provided dimensions.
pub fn centered_rect((width, height): (u32, u32), side: u32) -> PixelRect {
    let w = side.min(width);
    let h = side.min(height);
    PixelRect {
        x: (width - w) / 2,
        y: (height - h) / 2,
        w,
        h,
    }
}

/// Returns the downscaled rectangle that fully covers `rect`.
pub fn scaled_rect_covering(rect: PixelRect, denom: u32) -> PixelRect {
    let x1 = (rect.x + rect.w).div_ceil(denom);
    let y1 = (rect.y + rect.h).div_ceil(denom);
    PixelRect {
        x: rect.x / denom,
        y: rect.y / denom,
        w: x1 - rect.x / denom,
        h: y1 - rect.y / denom,
    }
}

/// Crops interleaved `u8` samples from a full image.
pub fn crop_interleaved_u8(
    full: &[u8],
    full_width: usize,
    channels: usize,
    roi: PixelRect,
) -> Vec<u8> {
    crop_interleaved_bytes(full, full_width, channels, roi)
}

/// Crops interleaved bytes from a full image.
pub fn crop_interleaved_bytes(
    full: &[u8],
    full_width: usize,
    bytes_per_pixel: usize,
    roi: PixelRect,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(roi.w as usize * roi.h as usize * bytes_per_pixel);
    let row_bytes = full_width * bytes_per_pixel;
    let roi_row_bytes = roi.w as usize * bytes_per_pixel;
    for y in roi.y as usize..(roi.y + roi.h) as usize {
        let start = y * row_bytes + roi.x as usize * bytes_per_pixel;
        out.extend_from_slice(&full[start..start + roi_row_bytes]);
    }
    out
}

/// Crops interleaved `u16` samples from a full image.
pub fn crop_interleaved_u16(
    full: &[u16],
    full_width: usize,
    channels: usize,
    roi: PixelRect,
) -> Vec<u16> {
    let mut out = Vec::with_capacity(roi.w as usize * roi.h as usize * channels);
    let row_samples = full_width * channels;
    let roi_row_samples = roi.w as usize * channels;
    for y in roi.y as usize..(roi.y + roi.h) as usize {
        let start = y * row_samples + roi.x as usize * channels;
        out.extend_from_slice(&full[start..start + roi_row_samples]);
    }
    out
}

/// Projects a downscaled interleaved `u8` output rectangle from a full image.
pub fn project_scaled_interleaved_u8(
    full: &[u8],
    width: u32,
    height: u32,
    channels: usize,
    output_rect: PixelRect,
    denom: u32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(output_rect.w as usize * output_rect.h as usize * channels);
    for sy in output_rect.y..output_rect.y + output_rect.h {
        let src_y = sy.saturating_mul(denom).min(height - 1);
        for sx in output_rect.x..output_rect.x + output_rect.w {
            let src_x = sx.saturating_mul(denom).min(width - 1);
            let offset = (src_y as usize * width as usize + src_x as usize) * channels;
            out.extend_from_slice(&full[offset..offset + channels]);
        }
    }
    out
}

/// Projects a downscaled interleaved `u16` output rectangle from a full image.
pub fn project_scaled_interleaved_u16(
    full: &[u16],
    width: u32,
    height: u32,
    channels: usize,
    output_rect: PixelRect,
    denom: u32,
) -> Vec<u16> {
    let mut out = Vec::with_capacity(output_rect.w as usize * output_rect.h as usize * channels);
    for sy in output_rect.y..output_rect.y + output_rect.h {
        let src_y = sy.saturating_mul(denom).min(height - 1);
        for sx in output_rect.x..output_rect.x + output_rect.w {
            let src_x = sx.saturating_mul(denom).min(width - 1);
            let offset = (src_y as usize * width as usize + src_x as usize) * channels;
            out.extend_from_slice(&full[offset..offset + channels]);
        }
    }
    out
}

/// Converts RGB8 bytes to RGBA8 bytes with constant alpha.
pub fn rgb8_to_rgba8(rgb: &[u8], alpha: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
    for pixel in rgb.chunks_exact(3) {
        out.extend_from_slice(&[pixel[0], pixel[1], pixel[2], alpha]);
    }
    out
}

/// Converts `u16` samples to little-endian bytes.
pub fn u16_samples_to_le_bytes(samples: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

/// Converts little-endian RGB16 bytes to RGBA16 bytes with constant alpha.
pub fn rgb16le_to_rgba16le(rgb: &[u8], alpha: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 6 * 8);
    let alpha = alpha.to_le_bytes();
    for pixel in rgb.chunks_exact(6) {
        out.extend_from_slice(pixel);
        out.extend_from_slice(&alpha);
    }
    out
}

/// Converts native-endian RGB16 bytes to opaque RGBA16 bytes.
pub fn rgb16ne_to_opaque_rgba16ne(rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 6 * 8);
    for pixel in rgb.chunks_exact(6) {
        out.extend_from_slice(pixel);
        out.extend_from_slice(&u16::MAX.to_ne_bytes());
    }
    out
}
