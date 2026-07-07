// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Rect, Vec, Warning};

pub(super) fn merged_warnings(
    header_warnings: &[Warning],
    scan_warnings: Vec<Warning>,
) -> Vec<Warning> {
    if header_warnings.is_empty() {
        return scan_warnings;
    }
    if scan_warnings.is_empty() {
        return header_warnings.to_vec();
    }
    let mut warnings = Vec::with_capacity(header_warnings.len() + scan_warnings.len());
    warnings.extend_from_slice(header_warnings);
    warnings.extend(scan_warnings);
    warnings
}

pub(super) fn copy_gray8_scaled_rect(
    full: &[u8],
    dimensions: (u32, u32),
    output_rect: Rect,
    denom: u32,
    out: &mut [u8],
    stride: usize,
) {
    let (width, height) = dimensions;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1);
        let dst_row = (output_y - output_rect.y) as usize;
        let dst_start = dst_row * stride;
        let dst = &mut out[dst_start..dst_start + output_rect.w as usize];
        for (dst_px, output_x) in (output_rect.x..output_rect.x + output_rect.w).enumerate() {
            let source_x = output_x.saturating_mul(denom).min(width - 1);
            let src = source_y as usize * width as usize + source_x as usize;
            dst[dst_px] = full[src];
        }
    }
}

pub(super) fn copy_rgb8_scaled_rect(
    full: &[u8],
    dimensions: (u32, u32),
    output_rect: Rect,
    denom: u32,
    out: &mut [u8],
    stride: usize,
) {
    let (width, height) = dimensions;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1);
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1);
            let src = (source_y as usize * width as usize + source_x as usize) * 3;
            let dst = dst_row * stride + (output_x - output_rect.x) as usize * 3;
            out[dst..dst + 3].copy_from_slice(&full[src..src + 3]);
        }
    }
}

pub(super) fn convert_ycbcr8_to_rgb8_in_place(
    out: &mut [u8],
    stride: usize,
    dimensions: (u32, u32),
) {
    let (width, height) = dimensions;
    let row_bytes = width as usize * 3;
    for y in 0..height as usize {
        let row = &mut out[y * stride..y * stride + row_bytes];
        for pixel in row.chunks_exact_mut(3) {
            let (r, g, b) = crate::color::ycbcr::ycbcr_to_rgb(pixel[0], pixel[1], pixel[2]);
            pixel.copy_from_slice(&[r, g, b]);
        }
    }
}

pub(super) fn copy_ycbcr8_row_to_rgb8(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (source, target) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        let (r, g, b) = crate::color::ycbcr::ycbcr_to_rgb(source[0], source[1], source[2]);
        target.copy_from_slice(&[r, g, b]);
    }
}

pub(super) fn copy_rgb8_to_rgba8(
    src: &[u8],
    src_stride: usize,
    width: u32,
    height: u32,
    dst: &mut [u8],
    dst_stride: usize,
    alpha: u8,
) {
    let src_row_bytes = width as usize * 3;
    let dst_row_bytes = width as usize * 4;
    for y in 0..height as usize {
        let src_row = &src[y * src_stride..y * src_stride + src_row_bytes];
        let dst_row = &mut dst[y * dst_stride..y * dst_stride + dst_row_bytes];
        for (source, target) in src_row.chunks_exact(3).zip(dst_row.chunks_exact_mut(4)) {
            target.copy_from_slice(&[source[0], source[1], source[2], alpha]);
        }
    }
}

pub(super) fn copy_rgb16_to_rgba16(
    src: &[u8],
    src_stride: usize,
    width: u32,
    height: u32,
    dst: &mut [u8],
    dst_stride: usize,
    alpha: u16,
) {
    let src_row_bytes = width as usize * 6;
    let dst_row_bytes = width as usize * 8;
    let alpha = alpha.to_le_bytes();
    for y in 0..height as usize {
        let src_row = &src[y * src_stride..y * src_stride + src_row_bytes];
        let dst_row = &mut dst[y * dst_stride..y * dst_stride + dst_row_bytes];
        for (source, target) in src_row.chunks_exact(6).zip(dst_row.chunks_exact_mut(8)) {
            target[..6].copy_from_slice(source);
            target[6..8].copy_from_slice(&alpha);
        }
    }
}

pub(super) fn convert_ycbcr16_to_rgb16_in_place(
    out: &mut [u8],
    stride: usize,
    dimensions: (u32, u32),
) {
    let (width, height) = dimensions;
    let row_bytes = width as usize * 6;
    for y in 0..height as usize {
        let row = &mut out[y * stride..y * stride + row_bytes];
        for pixel in row.chunks_exact_mut(6) {
            let y = u16::from_le_bytes([pixel[0], pixel[1]]);
            let cb = u16::from_le_bytes([pixel[2], pixel[3]]);
            let cr = u16::from_le_bytes([pixel[4], pixel[5]]);
            let (r, g, b) = crate::color::ycbcr::ycbcr16_to_rgb16(y, cb, cr);
            pixel[0..2].copy_from_slice(&r.to_le_bytes());
            pixel[2..4].copy_from_slice(&g.to_le_bytes());
            pixel[4..6].copy_from_slice(&b.to_le_bytes());
        }
    }
}

pub(super) fn copy_ycbcr16_row_to_rgb16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (source, target) in src.chunks_exact(6).zip(dst.chunks_exact_mut(6)) {
        let y = u16::from_le_bytes([source[0], source[1]]);
        let cb = u16::from_le_bytes([source[2], source[3]]);
        let cr = u16::from_le_bytes([source[4], source[5]]);
        let (r, g, b) = crate::color::ycbcr::ycbcr16_to_rgb16(y, cb, cr);
        target[0..2].copy_from_slice(&r.to_le_bytes());
        target[2..4].copy_from_slice(&g.to_le_bytes());
        target[4..6].copy_from_slice(&b.to_le_bytes());
    }
}

pub(super) fn copy_rgb16_scaled_rect(
    full: &[u8],
    dimensions: (u32, u32),
    output_rect: Rect,
    denom: u32,
    out: &mut [u8],
    stride: usize,
) {
    let (width, height) = dimensions;
    let full_stride = width as usize * 6;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1);
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1);
            let src = source_y as usize * full_stride + source_x as usize * 6;
            let dst = dst_row * stride + (output_x - output_rect.x) as usize * 6;
            out[dst..dst + 6].copy_from_slice(&full[src..src + 6]);
        }
    }
}

pub(super) fn copy_gray16_scaled_rect(
    full: &[u8],
    dimensions: (u32, u32),
    output_rect: Rect,
    denom: u32,
    out: &mut [u8],
    stride: usize,
) {
    let (width, height) = dimensions;
    let full_stride = width as usize * 2;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1);
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1);
            let src = source_y as usize * full_stride + source_x as usize * 2;
            let dst = dst_row * stride + (output_x - output_rect.x) as usize * 2;
            out[dst..dst + 2].copy_from_slice(&full[src..src + 2]);
        }
    }
}
