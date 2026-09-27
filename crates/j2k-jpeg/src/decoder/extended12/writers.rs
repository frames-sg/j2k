// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scaling- and ROI-aware extended-precision output writers.

use core::ops::Range;

use super::super::{ColorSpace, DownscaleFactor, Rect};
use super::planes::Extended12Plane;
use super::sampling::Extended12ColorSampling;
use super::upsample::{
    extended12_plane_row, sample_extended12_plane_at, upsample_extended12_h2v1_at,
    upsample_extended12_plane_h2v1_at, upsample_extended12_plane_h2v2_at, upsample_h2v2_rows_at,
};

#[cfg(test)]
mod tests;

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

/// Output coordinates in `start..start + len` whose clamped source
/// coordinate `min(o * denom, extent - 1)` lies in the block span `lo..hi`.
/// The mapping is monotonic, so the matches form one contiguous range.
fn block_output_span(
    start: u32,
    len: u32,
    denom: u32,
    extent: u32,
    lo: u32,
    hi: u32,
) -> Range<u32> {
    let end = start + len;
    let first = lo.div_ceil(denom).clamp(start, end);
    let last = if hi >= extent {
        end
    } else {
        hi.div_ceil(denom).clamp(first, end)
    };
    first..last
}

/// Visit each output pixel that samples the 8x8 block at `block_origin`, as
/// `(dst_row, dst_col, src_index)`.
fn for_each_block_output(
    region: Extended12WriteRegion,
    block_origin: (u32, u32),
    mut visit: impl FnMut(usize, usize, usize),
) {
    let (width, height) = region.dimensions;
    let (x0, y0) = block_origin;
    let denom = region.downscale.denominator();
    let rect = region.output_rect;
    let rows = block_output_span(rect.y, rect.h, denom, height, y0, (y0 + 8).min(height));
    let cols = block_output_span(rect.x, rect.w, denom, width, x0, (x0 + 8).min(width));
    for output_y in rows {
        let src_row = (output_y.saturating_mul(denom).min(height - 1) - y0) as usize;
        let dst_row = (output_y - rect.y) as usize;
        for output_x in cols.clone() {
            let src_col = (output_x.saturating_mul(denom).min(width - 1) - x0) as usize;
            visit(dst_row, (output_x - rect.x) as usize, src_row * 8 + src_col);
        }
    }
}

pub(super) fn write_extended12_rgb_block_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    projection: Extended12RgbProjection,
    block_origin: (u32, u32),
    pixels: &[[u16; 64]; 3],
) {
    if region.downscale.denominator() == 1 {
        write_rgb16_block_full(out, stride, region, projection, block_origin, pixels);
        return;
    }
    for_each_block_output(region, block_origin, |dst_row, dst_col, src_index| {
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
        write_rgb16(out, dst_row * stride + dst_col * 6, (r, g, b));
    });
}

pub(super) fn write_extended12_four_component_block_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    color_space: ColorSpace,
    block_origin: (u32, u32),
    pixels: &[[u16; 64]; 4],
) {
    for_each_block_output(region, block_origin, |dst_row, dst_col, src_index| {
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
        write_rgb16(out, dst_row * stride + dst_col * 6, (r, g, b));
    });
}

fn write_rgb16(out: &mut [u8], offset: usize, (r, g, b): (u16, u16, u16)) {
    let dst = &mut out[offset..offset + 6];
    dst[0..2].copy_from_slice(&r.to_le_bytes());
    dst[2..4].copy_from_slice(&g.to_le_bytes());
    dst[4..6].copy_from_slice(&b.to_le_bytes());
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
    let chroma_rows = planes[1].pixels.len() / planes[1].stride;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1) as usize;
        let luma = extended12_plane_row(&planes[0], source_y);
        let chroma_y = source_y.min(chroma_rows - 1);
        let cb_row = extended12_plane_row(&planes[1], chroma_y);
        let cr_row = extended12_plane_row(&planes[2], chroma_y);
        let dst_row = &mut out[(output_y - output_rect.y) as usize * stride..];
        if denom == 1 {
            let span = output_rect.x as usize..(output_rect.x + output_rect.w) as usize;
            let rows = [cb_row, cr_row];
            write_fancy_full_row(
                dst_row,
                projection,
                luma,
                span,
                (FANCY_H2V1, cb_row.len() - 1),
                |channel, sample| u32::from(rows[channel][sample]),
            );
            continue;
        }
        for (dst, output_x) in dst_row
            .chunks_exact_mut(6)
            .zip(output_rect.x..output_rect.x + output_rect.w)
        {
            let source_x = output_x.saturating_mul(denom).min(width - 1) as usize;
            let c1 = upsample_extended12_h2v1_at(cb_row, source_x);
            let c2 = upsample_extended12_h2v1_at(cr_row, source_x);
            write_projected_rgb16(dst, projection, luma[source_x], c1, c2);
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
    // Real chroma rows; the planes below them hold MCU padding.
    let chroma_height = (height as usize).div_ceil(2);
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1) as usize;
        let luma = extended12_plane_row(&planes[0], source_y);
        let chroma_y = (source_y / 2).min(chroma_height - 1);
        // Odd rows blend toward the chroma row below, even rows toward the
        // one above.
        let near_y = if source_y.is_multiple_of(2) {
            chroma_y.saturating_sub(1)
        } else {
            (chroma_y + 1).min(chroma_height - 1)
        };
        let (cb, cb_near) = (
            extended12_plane_row(&planes[1], chroma_y),
            extended12_plane_row(&planes[1], near_y),
        );
        let (cr, cr_near) = (
            extended12_plane_row(&planes[2], chroma_y),
            extended12_plane_row(&planes[2], near_y),
        );
        let dst_row = &mut out[(output_y - output_rect.y) as usize * stride..];
        if denom == 1 && cb.len() > 1 {
            let span = output_rect.x as usize..(output_rect.x + output_rect.w) as usize;
            let rows = [(cb, cb_near), (cr, cr_near)];
            write_fancy_full_row(
                dst_row,
                projection,
                luma,
                span,
                (FANCY_H2V2, cb.len() - 1),
                |channel, sample| {
                    let (curr, near) = rows[channel];
                    3 * u32::from(curr[sample]) + u32::from(near[sample])
                },
            );
            continue;
        }
        for (dst, output_x) in dst_row
            .chunks_exact_mut(6)
            .zip(output_rect.x..output_rect.x + output_rect.w)
        {
            let source_x = output_x.saturating_mul(denom).min(width - 1) as usize;
            let c1 = upsample_h2v2_rows_at(cb, cb_near, cb.len() * 2, source_x);
            let c2 = upsample_h2v2_rows_at(cr, cr_near, cr.len() * 2, source_x);
            write_projected_rgb16(dst, projection, luma[source_x], c1, c2);
        }
    }
}

/// Output pixels a [`write_fancy_full_row`] chunk covers.
const FANCY_CHUNK: usize = 64;

/// Fancy horizontal upsampling rule: each chroma sample `c` yields an even
/// output `(3 * sum(c) + sum(left) + even_round) >> shift` and an odd output
/// `(3 * sum(c) + sum(right) + odd_round) >> shift`, with the column sums
/// replicated past the row ends. This reproduces libjpeg-turbo's fancy
/// upsamplers, including their end-of-row cases.
#[derive(Clone, Copy)]
struct FancyRule {
    even_round: u32,
    odd_round: u32,
    shift: u32,
}

/// `h2v1_fancy_upsample` for DCT-coded samples ([`upsample_extended12_h2v1_at`]).
const FANCY_H2V1: FancyRule = FancyRule {
    even_round: 1,
    odd_round: 2,
    shift: 2,
};

/// `h2v2_fancy_upsample` over `3 * curr + near` column sums
/// ([`upsample_h2v2_rows_at`]).
const FANCY_H2V2: FancyRule = FancyRule {
    even_round: 8,
    odd_round: 7,
    shift: 4,
};

/// Full-resolution color row over source columns `span`, upsampling chroma a
/// chunk at a time. `column_sum(channel, sample)` gives the chroma column sum
/// of channel 0 (Cb) or 1 (Cr).
fn write_fancy_full_row(
    dst_row: &mut [u8],
    projection: Extended12RgbProjection,
    luma: &[u16],
    span: Range<usize>,
    (rule, last_sample): (FancyRule, usize),
    column_sum: impl Fn(usize, usize) -> u32,
) {
    let mut up = [[0u16; FANCY_CHUNK + 2]; 2];
    let mut start = span.start;
    let mut dst = dst_row.chunks_exact_mut(6);
    while start < span.end {
        let end = (start + FANCY_CHUNK).min(span.end);
        let first_sample = start / 2;
        let pairs = end.div_ceil(2) - first_sample;
        for (channel, up) in up.iter_mut().enumerate() {
            fancy_pairs(
                rule,
                first_sample,
                last_sample,
                &mut up[..2 * pairs],
                |sample| column_sum(channel, sample),
            );
        }
        let offset = start - 2 * first_sample;
        let chunk = luma[start..end]
            .iter()
            .zip(&up[0][offset..])
            .zip(&up[1][offset..]);
        for (((&y, &c1), &c2), dst) in chunk.zip(&mut dst) {
            write_projected_rgb16(dst, projection, y, c1, c2);
        }
        start = end;
    }
}

/// Upsample chroma samples `first..first + out.len() / 2` into interleaved
/// even/odd output pairs under `rule`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a weighted mean of 12-bit samples fits u16"
)]
fn fancy_pairs(
    rule: FancyRule,
    first: usize,
    last: usize,
    out: &mut [u16],
    column_sum: impl Fn(usize) -> u32,
) {
    // Column sums for samples `first - 1 ..= first + pairs`, replicated at
    // the row ends.
    let mut sums = [0u32; FANCY_CHUNK / 2 + 3];
    let pairs = out.len() / 2;
    for (offset, sum) in sums[..pairs + 2].iter_mut().enumerate() {
        *sum = column_sum((first + offset).saturating_sub(1).min(last));
    }
    for (pair, window) in out.chunks_exact_mut(2).zip(sums.windows(3)) {
        let this = 3 * window[1];
        pair[0] = ((this + window[0] + rule.even_round) >> rule.shift) as u16;
        pair[1] = ((this + window[2] + rule.odd_round) >> rule.shift) as u16;
    }
}

fn write_projected_rgb16(
    dst: &mut [u8],
    projection: Extended12RgbProjection,
    c0: u16,
    c1: u16,
    c2: u16,
) {
    let (r, g, b) = match projection {
        Extended12RgbProjection::Identity => (c0, c1, c2),
        Extended12RgbProjection::YCbCr => crate::color::ycbcr::ycbcr12_to_rgb16(c0, c1, c2),
    };
    dst[0..2].copy_from_slice(&r.to_le_bytes());
    dst[2..4].copy_from_slice(&g.to_le_bytes());
    dst[4..6].copy_from_slice(&b.to_le_bytes());
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
    // Real 4:2:0 chroma rows; the planes below them hold MCU padding.
    let chroma_height = (height as usize).div_ceil(2);
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
                    upsample_extended12_plane_h2v2_at(
                        &planes[1],
                        chroma_height,
                        source_x,
                        source_y,
                    ),
                    upsample_extended12_plane_h2v2_at(
                        &planes[2],
                        chroma_height,
                        source_x,
                        source_y,
                    ),
                    upsample_extended12_plane_h2v2_at(
                        &planes[3],
                        chroma_height,
                        source_x,
                        source_y,
                    ),
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
    if matches!(region.output, Extended12Output::Gray16) && region.downscale.denominator() == 1 {
        write_gray16_block_full(out, stride, region, block_origin, pixels);
        return;
    }
    for_each_block_output(region, block_origin, |dst_row, dst_col, src_index| {
        let sample = pixels[src_index];
        match region.output {
            Extended12Output::Gray16 => {
                let offset = dst_row * stride + dst_col * 2;
                out[offset..offset + 2].copy_from_slice(&sample.to_le_bytes());
            }
            Extended12Output::Rgb16 => {
                write_rgb16(
                    out,
                    dst_row * stride + dst_col * 6,
                    (sample, sample, sample),
                );
            }
        }
    });
}

/// Full-resolution block rows, clipped to the image and the output
/// rectangle: `(block row, source column, output row, output column, length)`
/// for each visible row of the block at `(x0, y0)`.
fn full_block_rows(
    region: Extended12WriteRegion,
    (x0, y0): (u32, u32),
) -> impl Iterator<Item = (usize, usize, usize, usize, usize)> {
    let (width, height) = region.dimensions;
    let rect = region.output_rect;
    let rows = block_output_span(rect.y, rect.h, 1, height, y0, (y0 + 8).min(height));
    let cols = block_output_span(rect.x, rect.w, 1, width, x0, (x0 + 8).min(width));
    let (src_col, dst_col, len) = (
        (cols.start - x0) as usize,
        (cols.start - rect.x) as usize,
        cols.len(),
    );
    rows.map(move |output_y| {
        (
            (output_y - y0) as usize,
            src_col,
            (output_y - rect.y) as usize,
            dst_col,
            len,
        )
    })
}

fn write_gray16_block_full(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    block_origin: (u32, u32),
    pixels: &[u16; 64],
) {
    for (row, src_col, dst_row, dst_col, len) in full_block_rows(region, block_origin) {
        let src = &pixels[row * 8 + src_col..][..len];
        let dst = &mut out[dst_row * stride + dst_col * 2..][..len * 2];
        for (dst, &sample) in dst.chunks_exact_mut(2).zip(src) {
            dst.copy_from_slice(&sample.to_le_bytes());
        }
    }
}

fn write_rgb16_block_full(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    projection: Extended12RgbProjection,
    block_origin: (u32, u32),
    pixels: &[[u16; 64]; 3],
) {
    for (row, src_col, dst_row, dst_col, len) in full_block_rows(region, block_origin) {
        let start = row * 8 + src_col;
        let [c0, c1, c2] = pixels.each_ref().map(|plane| &plane[start..start + len]);
        let dst = &mut out[dst_row * stride + dst_col * 6..][..len * 6];
        for (dst, ((&c0, &c1), &c2)) in dst.chunks_exact_mut(6).zip(c0.iter().zip(c1).zip(c2)) {
            write_projected_rgb16(dst, projection, c0, c1, c2);
        }
    }
}
