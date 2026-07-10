// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scaling- and ROI-aware extended-precision output writers.

use super::super::lossless_helpers::upsample_h2v1_u16_at;
use super::super::{ColorSpace, DownscaleFactor, Rect};
use super::planes::Extended12Plane;
use super::sampling::Extended12ColorSampling;
use super::upsample::{
    extended12_plane_row, sample_extended12_plane_at, upsample_extended12_plane_h2v1_at,
    upsample_extended12_plane_h2v2_at, upsample_h2v2_u16_rows_at,
};

#[derive(Debug, Clone, Copy)]
pub(super) enum Extended12Output {
    Gray16,
    Rgb16,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Extended12RgbProjection {
    Identity,
    YCbCr,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Extended12WriteRegion {
    pub(super) output_rect: Rect,
    pub(super) dimensions: (u32, u32),
    pub(super) downscale: DownscaleFactor,
    pub(super) output: Extended12Output,
}

pub(super) fn write_extended12_rgb_block_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    projection: Extended12RgbProjection,
    block_origin: (u32, u32),
    pixels: &[[u16; 64]; 3],
) {
    let (width, height) = region.dimensions;
    let (x0, y0) = block_origin;
    let block_x1 = (x0 + 8).min(width);
    let block_y1 = (y0 + 8).min(height);
    let denom = region.downscale.denominator();
    let output_rect = region.output_rect;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1);
        if source_y < y0 || source_y >= block_y1 {
            continue;
        }
        let src_row = (source_y - y0) as usize;
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1);
            if source_x < x0 || source_x >= block_x1 {
                continue;
            }
            let src_col = (source_x - x0) as usize;
            let src_index = src_row * 8 + src_col;
            let dst_col = (output_x - output_rect.x) as usize;
            let dst_start = dst_row * stride + dst_col * 6;
            let dst = &mut out[dst_start..dst_start + 6];
            let (r, g, b) = match projection {
                Extended12RgbProjection::Identity => (
                    pixels[0][src_index],
                    pixels[1][src_index],
                    pixels[2][src_index],
                ),
                Extended12RgbProjection::YCbCr => crate::color::ycbcr::ycbcr12_to_rgb16(
                    pixels[0][src_index],
                    pixels[1][src_index],
                    pixels[2][src_index],
                ),
            };
            dst[0..2].copy_from_slice(&r.to_le_bytes());
            dst[2..4].copy_from_slice(&g.to_le_bytes());
            dst[4..6].copy_from_slice(&b.to_le_bytes());
        }
    }
}

pub(super) fn write_extended12_four_component_block_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    color_space: ColorSpace,
    block_origin: (u32, u32),
    pixels: &[[u16; 64]; 4],
) {
    let (width, height) = region.dimensions;
    let (x0, y0) = block_origin;
    let block_x1 = (x0 + 8).min(width);
    let block_y1 = (y0 + 8).min(height);
    let denom = region.downscale.denominator();
    let output_rect = region.output_rect;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1);
        if source_y < y0 || source_y >= block_y1 {
            continue;
        }
        let src_row = (source_y - y0) as usize;
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1);
            if source_x < x0 || source_x >= block_x1 {
                continue;
            }
            let src_col = (source_x - x0) as usize;
            let src_index = src_row * 8 + src_col;
            let (r, g, b) = match color_space {
                ColorSpace::Cmyk => crate::color::cmyk::inverted_cmyk12_to_rgb16(
                    pixels[0][src_index],
                    pixels[1][src_index],
                    pixels[2][src_index],
                    pixels[3][src_index],
                ),
                ColorSpace::Ycck => crate::color::cmyk::ycck12_to_rgb16(
                    pixels[0][src_index],
                    pixels[1][src_index],
                    pixels[2][src_index],
                    pixels[3][src_index],
                ),
                _ => unreachable!("12-bit four-component path only accepts CMYK/YCCK"),
            };
            let dst_col = (output_x - output_rect.x) as usize;
            let dst_start = dst_row * stride + dst_col * 6;
            let dst = &mut out[dst_start..dst_start + 6];
            dst[0..2].copy_from_slice(&r.to_le_bytes());
            dst[2..4].copy_from_slice(&g.to_le_bytes());
            dst[4..6].copy_from_slice(&b.to_le_bytes());
        }
    }
}

pub(super) fn write_extended12_color422_planes_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    projection: Extended12RgbProjection,
    planes: &[Extended12Plane; 3],
) {
    let (width, height) = region.dimensions;
    let denom = region.downscale.denominator();
    let output_rect = region.output_rect;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1) as usize;
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1) as usize;
            let y = planes[0].pixels[source_y * planes[0].stride + source_x];
            let chroma_y = source_y.min(planes[1].pixels.len() / planes[1].stride - 1);
            let cb_row = &planes[1].pixels
                [chroma_y * planes[1].stride..chroma_y * planes[1].stride + planes[1].width];
            let cr_row = &planes[2].pixels
                [chroma_y * planes[2].stride..chroma_y * planes[2].stride + planes[2].width];
            let c1 = upsample_h2v1_u16_at(cb_row, source_x);
            let c2 = upsample_h2v1_u16_at(cr_row, source_x);
            let (r, g, b) = match projection {
                Extended12RgbProjection::Identity => (y, c1, c2),
                Extended12RgbProjection::YCbCr => crate::color::ycbcr::ycbcr12_to_rgb16(y, c1, c2),
            };
            let dst_col = (output_x - output_rect.x) as usize;
            let dst_start = dst_row * stride + dst_col * 6;
            let dst = &mut out[dst_start..dst_start + 6];
            dst[0..2].copy_from_slice(&r.to_le_bytes());
            dst[2..4].copy_from_slice(&g.to_le_bytes());
            dst[4..6].copy_from_slice(&b.to_le_bytes());
        }
    }
}

pub(super) fn write_extended12_color420_planes_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    projection: Extended12RgbProjection,
    planes: &[Extended12Plane; 3],
) {
    let (width, height) = region.dimensions;
    let denom = region.downscale.denominator();
    let output_rect = region.output_rect;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1) as usize;
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1) as usize;
            let y = planes[0].pixels[source_y * planes[0].stride + source_x];
            let chroma_height = planes[1].pixels.len() / planes[1].stride;
            let chroma_y = (source_y / 2).min(chroma_height - 1);
            let prev_y = chroma_y.saturating_sub(1);
            let next_y = (chroma_y + 1).min(chroma_height - 1);
            let c1 = upsample_h2v2_u16_rows_at(
                extended12_plane_row(&planes[1], prev_y),
                extended12_plane_row(&planes[1], chroma_y),
                extended12_plane_row(&planes[1], next_y),
                source_x,
                !source_y.is_multiple_of(2),
            );
            let c2 = upsample_h2v2_u16_rows_at(
                extended12_plane_row(&planes[2], prev_y),
                extended12_plane_row(&planes[2], chroma_y),
                extended12_plane_row(&planes[2], next_y),
                source_x,
                !source_y.is_multiple_of(2),
            );
            let (r, g, b) = match projection {
                Extended12RgbProjection::Identity => (y, c1, c2),
                Extended12RgbProjection::YCbCr => crate::color::ycbcr::ycbcr12_to_rgb16(y, c1, c2),
            };
            let dst_col = (output_x - output_rect.x) as usize;
            let dst_start = dst_row * stride + dst_col * 6;
            let dst = &mut out[dst_start..dst_start + 6];
            dst[0..2].copy_from_slice(&r.to_le_bytes());
            dst[2..4].copy_from_slice(&g.to_le_bytes());
            dst[4..6].copy_from_slice(&b.to_le_bytes());
        }
    }
}

pub(super) fn write_extended12_four_component_planes_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    color_space: ColorSpace,
    sampling: Extended12ColorSampling,
    planes: &[Extended12Plane; 4],
) {
    let (width, height) = region.dimensions;
    let denom = region.downscale.denominator();
    let output_rect = region.output_rect;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1) as usize;
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1) as usize;
            let c0 = planes[0].pixels[source_y * planes[0].stride + source_x];
            let (c1, c2, c3) = match sampling {
                Extended12ColorSampling::S444 => (
                    sample_extended12_plane_at(&planes[1], source_x, source_y),
                    sample_extended12_plane_at(&planes[2], source_x, source_y),
                    sample_extended12_plane_at(&planes[3], source_x, source_y),
                ),
                Extended12ColorSampling::S422 => (
                    upsample_extended12_plane_h2v1_at(&planes[1], source_x, source_y),
                    upsample_extended12_plane_h2v1_at(&planes[2], source_x, source_y),
                    upsample_extended12_plane_h2v1_at(&planes[3], source_x, source_y),
                ),
                Extended12ColorSampling::S420 => (
                    upsample_extended12_plane_h2v2_at(&planes[1], source_x, source_y),
                    upsample_extended12_plane_h2v2_at(&planes[2], source_x, source_y),
                    upsample_extended12_plane_h2v2_at(&planes[3], source_x, source_y),
                ),
            };
            let (r, g, b) = match color_space {
                ColorSpace::Cmyk => crate::color::cmyk::inverted_cmyk12_to_rgb16(c0, c1, c2, c3),
                ColorSpace::Ycck => crate::color::cmyk::ycck12_to_rgb16(c0, c1, c2, c3),
                _ => unreachable!("12-bit four-component plane path only accepts CMYK/YCCK"),
            };
            let dst_col = (output_x - output_rect.x) as usize;
            let dst_start = dst_row * stride + dst_col * 6;
            let dst = &mut out[dst_start..dst_start + 6];
            dst[0..2].copy_from_slice(&r.to_le_bytes());
            dst[2..4].copy_from_slice(&g.to_le_bytes());
            dst[4..6].copy_from_slice(&b.to_le_bytes());
        }
    }
}

pub(super) fn write_extended12_block_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    block_origin: (u32, u32),
    pixels: &[u16; 64],
) {
    let (width, height) = region.dimensions;
    let (x0, y0) = block_origin;
    let block_x1 = (x0 + 8).min(width);
    let block_y1 = (y0 + 8).min(height);
    let denom = region.downscale.denominator();
    let bytes_per_pixel = match region.output {
        Extended12Output::Gray16 => 2,
        Extended12Output::Rgb16 => 6,
    };
    let output_rect = region.output_rect;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1);
        if source_y < y0 || source_y >= block_y1 {
            continue;
        }
        let src_row = (source_y - y0) as usize;
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1);
            if source_x < x0 || source_x >= block_x1 {
                continue;
            }
            let src_col = (source_x - x0) as usize;
            let sample = pixels[src_row * 8 + src_col].to_le_bytes();
            let dst_col = (output_x - output_rect.x) as usize;
            let dst_start = dst_row * stride + dst_col * bytes_per_pixel;
            let dst = &mut out[dst_start..dst_start + bytes_per_pixel];
            match region.output {
                Extended12Output::Gray16 => {
                    dst.copy_from_slice(&sample);
                }
                Extended12Output::Rgb16 => {
                    dst[0..2].copy_from_slice(&sample);
                    dst[2..4].copy_from_slice(&sample);
                    dst[4..6].copy_from_slice(&sample);
                }
            }
        }
    }
}
