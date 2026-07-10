// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{super::StripePlane, types::StripeNeighbors};
use crate::color::upsample::{
    upsample_1x1, upsample_h2v1_fancy_row, upsample_h2v2_fancy_row, upsample_h2v2_fancy_rows,
};

#[derive(Clone, Copy)]
pub(super) struct StripeComponentUpsampleSpec {
    pub(super) plane_idx: usize,
    pub(super) comp_h: u32,
    pub(super) comp_v: u32,
    pub(super) max_h: u32,
    pub(super) max_v: u32,
    pub(super) local_y_out: u32,
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
    pub(super) width: usize,
}

pub(super) struct Stripe420PairUpsample<'a, 'b> {
    pub(super) neighbors: StripeNeighbors<'a>,
    pub(super) spec: Stripe420PairSpec,
    pub(super) top: &'b mut [u8],
    pub(super) bot: &'b mut [u8],
}

pub(in crate::entropy::sequential) fn component_row_triplet<'a>(
    prev: Option<StripePlane<'a>>,
    curr: StripePlane<'a>,
    next: Option<StripePlane<'a>>,
    local_row: usize,
) -> (&'a [u8], &'a [u8], &'a [u8]) {
    fn plane_row(plane: StripePlane<'_>, row: usize) -> &[u8] {
        let start = row * plane.stride;
        &plane.data[start..start + plane.stride]
    }

    let curr_rows = curr.rows;
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
        comp_h,
        comp_v,
        max_h,
        max_v,
        local_y_out,
        width,
    } = spec;
    let v_ratio = max_v / comp_v;
    let h_ratio = max_h / comp_h;
    let curr_plane = curr.plane(plane_idx);
    let chroma_rows = curr_plane.rows as u32;
    let chroma_y = (local_y_out / v_ratio).min(chroma_rows.saturating_sub(1));
    let (prev_row, curr_row, next_row) = component_row_triplet(
        prev.map(|stripe| stripe.plane(plane_idx)),
        curr_plane,
        next.map(|stripe| stripe.plane(plane_idx)),
        chroma_y as usize,
    );

    match (h_ratio, v_ratio) {
        (1, 1) => {
            upsample_1x1(&curr_row[..width], out);
        }
        (2, 1) => {
            let chroma_cols = width.div_ceil(2);
            upsample_h2v1_fancy_row(&curr_row[..chroma_cols], width, out);
        }
        (2, 2) => {
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
        _ => {
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
    );

    upsample_h2v2_fancy_rows(prev_row, curr_row, next_row, width, top, bot);
}
