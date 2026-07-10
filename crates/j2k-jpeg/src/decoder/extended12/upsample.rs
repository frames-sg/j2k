// SPDX-License-Identifier: MIT OR Apache-2.0

//! Point sampling and fancy chroma upsampling for extended-precision planes.

use super::super::lossless_helpers::upsample_h2v1_u16_at;
use super::planes::Extended12Plane;

pub(super) fn sample_extended12_plane_at(
    plane: &Extended12Plane,
    source_x: usize,
    source_y: usize,
) -> u16 {
    let height = plane.pixels.len() / plane.stride;
    let y = source_y.min(height - 1);
    let x = source_x.min(plane.width - 1);
    plane.pixels[y * plane.stride + x]
}

pub(super) fn upsample_extended12_plane_h2v1_at(
    plane: &Extended12Plane,
    source_x: usize,
    source_y: usize,
) -> u16 {
    let height = plane.pixels.len() / plane.stride;
    let y = source_y.min(height - 1);
    upsample_h2v1_u16_at(extended12_plane_row(plane, y), source_x)
}

pub(super) fn upsample_extended12_plane_h2v2_at(
    plane: &Extended12Plane,
    source_x: usize,
    source_y: usize,
) -> u16 {
    let height = plane.pixels.len() / plane.stride;
    let chroma_y = (source_y / 2).min(height - 1);
    let prev_y = chroma_y.saturating_sub(1);
    let next_y = (chroma_y + 1).min(height - 1);
    upsample_h2v2_u16_rows_at(
        extended12_plane_row(plane, prev_y),
        extended12_plane_row(plane, chroma_y),
        extended12_plane_row(plane, next_y),
        source_x,
        !source_y.is_multiple_of(2),
    )
}

pub(super) fn extended12_plane_row(plane: &Extended12Plane, y: usize) -> &[u16] {
    let row_start = y * plane.stride;
    &plane.pixels[row_start..row_start + plane.width]
}

pub(in crate::decoder) trait UpsampleSample: Copy {
    fn to_u32(self) -> u32;
    fn from_u32(value: u32) -> Self;
}

impl UpsampleSample for u8 {
    fn to_u32(self) -> u32 {
        u32::from(self)
    }

    fn from_u32(value: u32) -> Self {
        value as u8
    }
}

impl UpsampleSample for u16 {
    fn to_u32(self) -> u32 {
        u32::from(self)
    }

    fn from_u32(value: u32) -> Self {
        value as u16
    }
}

pub(in crate::decoder) fn upsample_h2v1_sample_at<S: UpsampleSample>(
    row: &[S],
    output_x: usize,
) -> S {
    debug_assert!(!row.is_empty());
    if row.len() == 1 {
        return row[0];
    }
    let sample = output_x / 2;
    if output_x == 0 {
        row[0]
    } else if output_x == row.len() * 2 - 1 {
        row[row.len() - 1]
    } else if output_x.is_multiple_of(2) {
        S::from_u32((3 * row[sample].to_u32() + row[sample - 1].to_u32() + 2) / 4)
    } else {
        S::from_u32((3 * row[sample].to_u32() + row[sample + 1].to_u32() + 2) / 4)
    }
}

pub(in crate::decoder) fn upsample_h2v2_rows_at<S: UpsampleSample>(
    curr: &[S],
    near: &[S],
    output_width: usize,
    output_x: usize,
) -> S {
    debug_assert!(!curr.is_empty());
    debug_assert_eq!(near.len(), curr.len());
    let colsum = |index: usize| 3 * curr[index].to_u32() + near[index].to_u32();
    if curr.len() == 1 {
        return S::from_u32((4 * colsum(0) + 8) >> 4);
    }

    let sample = output_x / 2;
    let this = colsum(sample);
    // Match IJG/libjpeg fancy h2v2 upsampling: left/even samples round with
    // +8, right/odd samples with +7 before >> 4 to preserve bit-identical
    // interpolation at mirrored sample positions.
    match output_x {
        0 => S::from_u32((this * 4 + 8) >> 4),
        _ if output_x == output_width - 1 => S::from_u32((this * 4 + 7) >> 4),
        _ if output_x.is_multiple_of(2) => {
            let last = colsum(sample - 1);
            S::from_u32((this * 3 + last + 8) >> 4)
        }
        _ => {
            let next = colsum(sample + 1);
            S::from_u32((this * 3 + next + 7) >> 4)
        }
    }
}

pub(super) fn upsample_h2v2_u16_rows_at(
    prev: &[u16],
    curr: &[u16],
    next: &[u16],
    output_x: usize,
    output_is_bottom: bool,
) -> u16 {
    debug_assert_eq!(prev.len(), curr.len());
    debug_assert_eq!(next.len(), curr.len());
    let near = if output_is_bottom { next } else { prev };
    upsample_h2v2_rows_at(curr, near, curr.len() * 2, output_x)
}
