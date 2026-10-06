// SPDX-License-Identifier: MIT OR Apache-2.0

//! Full-resolution placement of a subsampled component: every sample is
//! replicated over its `scale_x` x `scale_y` footprint on the reference grid,
//! clipped to the image. Footprints tile the grid without overlap, so each
//! output row of a footprint row is a copy of the first.

use core::ops::Range;

pub(super) struct SubsampledPlacement<'a> {
    /// IDWT output, indexed relative to `component_origin` with row stride
    /// `input_stride`.
    pub(super) input: &'a [f32],
    pub(super) input_stride: usize,
    pub(super) component_origin: (u32, u32),
    /// Component-grid columns and rows to place.
    pub(super) columns: Range<u32>,
    pub(super) rows: Range<u32>,
    pub(super) scale: (u32, u32),
    pub(super) image_size: (u32, u32),
    /// Reference-grid position of the image origin.
    pub(super) image_offset: (u32, u32),
}

/// Place the samples into `output`, an image-sized row-major plane.
pub(super) fn place_subsampled(placement: &SubsampledPlacement<'_>, output: &mut [f32]) {
    let (scale_x, scale_y) = placement.scale;
    let (image_width, image_height) = placement.image_size;
    let (x_offset, y_offset) = placement.image_offset;
    let (x0, x1) = (placement.columns.start, placement.columns.end);
    let first_column = (scale_x * x0).max(x_offset);
    let end_column = (scale_x * x1).min(image_width + x_offset);
    if first_column >= end_column {
        return;
    }
    let width = image_width as usize;
    let written = (first_column - x_offset) as usize..(end_column - x_offset) as usize;
    let input_columns =
        (x0 - placement.component_origin.0) as usize..(x1 - placement.component_origin.0) as usize;
    for y in placement.rows.clone() {
        let first_row = (scale_y * y).max(y_offset);
        let end_row = (scale_y * y + scale_y).min(image_height + y_offset);
        if first_row >= end_row {
            continue;
        }
        let input_row = &placement.input
            [(y - placement.component_origin.1) as usize * placement.input_stride..];
        let out_row = (first_row - y_offset) as usize;
        fill_row(
            &input_row[input_columns.clone()],
            &mut output[out_row * width..][..width],
            placement,
        );
        for row in out_row + 1..(end_row - y_offset) as usize {
            output.copy_within(
                out_row * width + written.start..out_row * width + written.end,
                row * width + written.start,
            );
        }
    }
}

fn fill_row(samples: &[f32], out_row: &mut [f32], placement: &SubsampledPlacement<'_>) {
    let scale_x = placement.scale.0;
    let x_offset = placement.image_offset.0;
    let (x0, x1) = (placement.columns.start, placement.columns.end);
    if scale_x * x0 < x_offset || scale_x * x1 > placement.image_size.0 + x_offset {
        // A footprint is clipped at the image edge.
        for (x, &sample) in (x0..x1).zip(samples) {
            let first = (scale_x * x).max(x_offset);
            let end = (scale_x * x + scale_x).min(placement.image_size.0 + x_offset);
            if first < end {
                out_row[(first - x_offset) as usize..(end - x_offset) as usize].fill(sample);
            }
        }
        return;
    }
    let out = &mut out_row[(scale_x * x0 - x_offset) as usize..(scale_x * x1 - x_offset) as usize];
    match scale_x {
        1 => out.copy_from_slice(samples),
        2 => {
            for (pair, &sample) in out.as_chunks_mut::<2>().0.iter_mut().zip(samples) {
                *pair = [sample; 2];
            }
        }
        _ => {
            for (footprint, &sample) in out.chunks_exact_mut(scale_x as usize).zip(samples) {
                footprint.fill(sample);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{place_subsampled, SubsampledPlacement};
    use alloc::vec;
    use alloc::vec::Vec;

    /// The per-sample placement loop this module replaces at full resolution.
    fn place_reference(placement: &SubsampledPlacement<'_>, output: &mut [f32]) {
        let (scale_x, scale_y) = placement.scale;
        let (image_width, image_height) = placement.image_size;
        let (x_offset, y_offset) = placement.image_offset;
        for y in placement.rows.clone() {
            let relative_y = (y - placement.component_origin.1) as usize;
            for x in placement.columns.clone() {
                let relative_x = (x - placement.component_origin.0) as usize;
                let sample = placement.input[relative_y * placement.input_stride + relative_x];
                for x_position in
                    (scale_x * x).max(x_offset)..(scale_x * x + scale_x).min(image_width + x_offset)
                {
                    for y_position in (scale_y * y).max(y_offset)
                        ..(scale_y * y + scale_y).min(image_height + y_offset)
                    {
                        output[(y_position - y_offset) as usize * image_width as usize
                            + (x_position - x_offset) as usize] = sample;
                    }
                }
            }
        }
    }

    #[test]
    fn placement_matches_the_per_sample_reference() {
        for scale in [(1_u32, 1_u32), (2, 1), (1, 2), (2, 2), (3, 2)] {
            // Offsets and component origins that clip the first and last
            // footprints, alongside an aligned case.
            for (image_offset, origin, columns, rows) in [
                ((0_u32, 0_u32), (0_u32, 0_u32), 0_u32..5, 0_u32..4),
                ((1, 1), (0, 0), 0..5, 0..4),
                ((0, 0), (2, 1), 2..6, 1..4),
                ((3, 2), (1, 1), 1..6, 1..5),
            ] {
                let (scale_x, scale_y) = scale;
                let image_size = (
                    (scale_x * columns.end)
                        .saturating_sub(image_offset.0)
                        .max(1)
                        - 1,
                    (scale_y * rows.end).saturating_sub(image_offset.1).max(1),
                );
                let input_stride = (columns.end - origin.0) as usize;
                let input = core::iter::successors(Some(0.5_f32), |value| Some(value + 1.0))
                    .take(input_stride * (rows.end - origin.1) as usize)
                    .collect::<Vec<_>>();
                let placement = SubsampledPlacement {
                    input: &input,
                    input_stride,
                    component_origin: origin,
                    columns: columns.clone(),
                    rows: rows.clone(),
                    scale,
                    image_size,
                    image_offset,
                };
                let len = (image_size.0 * image_size.1) as usize;
                let (mut fast, mut reference) = (vec![-1.0; len], vec![-1.0; len]);
                place_subsampled(&placement, &mut fast);
                place_reference(&placement, &mut reference);
                assert_eq!(fast, reference, "scale {scale:?} offset {image_offset:?}");
            }
        }
    }
}
