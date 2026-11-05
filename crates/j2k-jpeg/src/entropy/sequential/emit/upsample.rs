// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{super::StripePlane, types::StripeNeighbors};
use crate::backend::Rgb420ChromaRows;
use crate::color::scaled_sampling::{ScaledComponent, Upsample};
use crate::color::upsample::{
    upsample_1x1, upsample_h1v2_fancy_row, upsample_h2v1_fancy_row, upsample_h2v2_fancy_row,
    upsample_h2v2_fancy_rows,
};

#[derive(Clone, Copy)]
pub(super) struct StripeComponentUpsampleSpec {
    pub(super) plane_idx: usize,
    pub(super) component: ScaledComponent,
    pub(super) local_y_out: u32,
    pub(super) stripe_rows: usize,
    pub(super) width: usize,
}

pub(super) struct StripeComponentUpsample<'a, 'b> {
    pub(super) neighbors: StripeNeighbors<'a>,
    pub(super) spec: StripeComponentUpsampleSpec,
    pub(super) out: &'b mut [u8],
}

#[derive(Clone, Copy)]
pub(super) struct Stripe420PairSpec {
    pub(super) plane_idx: usize,
    pub(super) local_y_out: u32,
    pub(super) stripe_rows: usize,
    pub(super) width: usize,
}

pub(super) struct Stripe420PairUpsample<'a, 'b> {
    pub(super) neighbors: StripeNeighbors<'a>,
    pub(super) spec: Stripe420PairSpec,
    pub(super) top: &'b mut [u8],
    pub(super) bot: &'b mut [u8],
}

/// Rows of a component plane that hold image samples when its stripe emits
/// `stripe_rows` output rows. Only the image's final stripe is short; the rest
/// of that stripe's plane is MCU padding.
pub(in crate::entropy::sequential) fn valid_component_rows(
    stripe_rows: usize,
    v_ratio: usize,
) -> usize {
    stripe_rows.div_ceil(v_ratio)
}

/// Returns the rows above, at, and below `local_row` for vertical upsampling.
///
/// `valid_rows` counts the rows of `curr` that hold image samples. Below the
/// last of them the current row is replicated, as libjpeg-turbo does
/// (`set_bottom_pointers` in `jdmainct.c`), so decoded MCU padding never
/// reaches the output.
pub(in crate::entropy::sequential) fn component_row_triplet<'a>(
    prev: Option<StripePlane<'a>>,
    curr: StripePlane<'a>,
    next: Option<StripePlane<'a>>,
    local_row: usize,
    valid_rows: usize,
) -> (&'a [u8], &'a [u8], &'a [u8]) {
    fn plane_row(plane: StripePlane<'_>, row: usize) -> &[u8] {
        let start = row * plane.stride;
        &plane.data[start..start + plane.stride]
    }

    let curr_rows = curr.rows.min(valid_rows);
    debug_assert!(
        local_row < curr_rows,
        "row {local_row} is outside the image"
    );
    debug_assert!(
        next.is_none() || curr_rows == curr.rows,
        "only the final stripe may end before its padded height"
    );
    let prev_row = if local_row == 0 {
        match prev {
            Some(plane) => plane_row(plane, plane.rows - 1),
            None => plane_row(curr, 0),
        }
    } else {
        plane_row(curr, local_row - 1)
    };
    let curr_row = plane_row(curr, local_row);
    let next_row = if local_row + 1 < curr_rows {
        plane_row(curr, local_row + 1)
    } else {
        match next {
            Some(plane) => plane_row(plane, 0),
            None => plane_row(curr, curr_rows - 1),
        }
    };
    (prev_row, curr_row, next_row)
}

/// The Cb and Cr rows above, at and below chroma row `local_row` (see
/// [`component_row_triplet`]), cut to `chroma_width` samples: the chroma a
/// 4:2:0 row-pair kernel reads. `planes(plane_idx)` gives the previous,
/// current and next stripe's view of chroma plane 1 or 2.
#[expect(
    clippy::inline_always,
    reason = "runs once per output row pair in both 4:2:0 emitters, next to the row kernels"
)]
#[inline(always)]
pub(in crate::entropy::sequential) fn rgb420_chroma_rows<'a>(
    planes: impl Fn(
        usize,
    ) -> (
        Option<StripePlane<'a>>,
        StripePlane<'a>,
        Option<StripePlane<'a>>,
    ),
    local_row: usize,
    valid_rows: usize,
    chroma_width: usize,
) -> Rgb420ChromaRows<'a> {
    let rows = |plane_idx| {
        let (prev, curr, next) = planes(plane_idx);
        <[&[u8]; 3]>::from(component_row_triplet(
            prev, curr, next, local_row, valid_rows,
        ))
        .map(|row| &row[..chroma_width])
    };
    let ([prev_cb, curr_cb, next_cb], [prev_cr, curr_cr, next_cr]) = (rows(1), rows(2));
    Rgb420ChromaRows::new(prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "component row indices are bounded by validated u32 stripe geometry"
)]
pub(super) fn upsample_component_row_stripe(request: StripeComponentUpsample<'_, '_>) {
    let StripeComponentUpsample {
        neighbors,
        spec,
        out,
    } = request;
    let StripeNeighbors { prev, curr, next } = neighbors;
    let StripeComponentUpsampleSpec {
        plane_idx,
        component,
        local_y_out,
        stripe_rows,
        width,
    } = spec;
    let ScaledComponent {
        h_ratio,
        v_ratio,
        upsample,
        ..
    } = component;
    let curr_plane = curr.plane(plane_idx);
    let chroma_rows = curr_plane.rows as u32;
    let chroma_y = (local_y_out / v_ratio).min(chroma_rows.saturating_sub(1));
    let (prev_row, curr_row, next_row) = match upsample {
        Upsample::FancyH1V2 | Upsample::FancyH2V2 => component_row_triplet(
            prev.map(|stripe| stripe.plane(plane_idx)),
            curr_plane,
            next.map(|stripe| stripe.plane(plane_idx)),
            chroma_y as usize,
            valid_component_rows(stripe_rows, v_ratio as usize),
        ),
        _ => {
            // Horizontal filters and replication use no neighboring rows.
            let start = chroma_y as usize * curr_plane.stride;
            let row = &curr_plane.data[start..start + curr_plane.stride];
            (row, row, row)
        }
    };

    match upsample {
        Upsample::None => {
            upsample_1x1(&curr_row[..width], out);
        }
        Upsample::FancyH2V1 => {
            let chroma_cols = width.div_ceil(2);
            upsample_h2v1_fancy_row(&curr_row[..chroma_cols], width, out);
        }
        Upsample::FancyH1V2 => {
            upsample_h1v2_fancy_row(
                &prev_row[..width],
                &curr_row[..width],
                &next_row[..width],
                width,
                !local_y_out.is_multiple_of(2),
                out,
            );
        }
        Upsample::FancyH2V2 => {
            let chroma_cols = width.div_ceil(2);
            upsample_h2v2_fancy_row(
                &prev_row[..chroma_cols],
                &curr_row[..chroma_cols],
                &next_row[..chroma_cols],
                width,
                !local_y_out.is_multiple_of(2),
                out,
            );
        }
        Upsample::Replicate => {
            for (x, slot) in out.iter_mut().enumerate().take(width) {
                let cx = ((x as u32) / h_ratio).min(curr_row.len() as u32 - 1);
                *slot = curr_row[cx as usize];
            }
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "4:2:0 row indices are bounded by validated u32 stripe geometry"
)]
pub(super) fn upsample_420_pair(request: Stripe420PairUpsample<'_, '_>) {
    let Stripe420PairUpsample {
        neighbors,
        spec,
        top,
        bot,
    } = request;
    let StripeNeighbors { prev, curr, next } = neighbors;
    let Stripe420PairSpec {
        plane_idx,
        local_y_out,
        stripe_rows,
        width,
    } = spec;
    let curr_plane = curr.plane(plane_idx);
    let chroma_rows = curr_plane.rows as u32;
    let chroma_y = (local_y_out / 2).min(chroma_rows.saturating_sub(1));
    let (prev_row, curr_row, next_row) = component_row_triplet(
        prev.map(|stripe| stripe.plane(plane_idx)),
        curr_plane,
        next.map(|stripe| stripe.plane(plane_idx)),
        chroma_y as usize,
        valid_component_rows(stripe_rows, 2),
    );
    let chroma_cols = width.div_ceil(2);

    upsample_h2v2_fancy_rows(
        &prev_row[..chroma_cols],
        &curr_row[..chroma_cols],
        &next_row[..chroma_cols],
        width,
        top,
        bot,
    );
}
