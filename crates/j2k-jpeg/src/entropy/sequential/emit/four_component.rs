// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    super::PreparedDecodePlan,
    types::StripeNeighbors,
    upsample::{
        upsample_component_row_stripe, StripeComponentUpsample, StripeComponentUpsampleSpec,
    },
};
use crate::{
    color::cmyk::{inverted_cmyk_to_rgb, ycck_to_rgb},
    color::scaled_sampling::ScaledSampling,
    info::ColorSpace,
    internal::scratch::RgbGenericRows,
};

/// One output row of a stripe: its stripe-local index, the rows the stripe
/// emits, and the output width.
#[derive(Clone, Copy)]
pub(super) struct FourComponentRow {
    pub(super) local_y: u32,
    pub(super) stripe_rows: usize,
    pub(super) width: usize,
}

pub(super) fn fill_four_component_rgb_row(
    plan: &PreparedDecodePlan,
    scaled: &ScaledSampling,
    neighbors: StripeNeighbors<'_>,
    row: FourComponentRow,
    scratch: &mut RgbGenericRows,
) {
    let FourComponentRow {
        local_y,
        stripe_rows,
        width,
    } = row;

    upsample_component_row_stripe(StripeComponentUpsample {
        neighbors,
        spec: StripeComponentUpsampleSpec {
            plane_idx: 0,
            component: scaled.component(0),
            local_y_out: local_y,
            stripe_rows,
            width,
        },
        out: &mut scratch.r,
    });
    upsample_component_row_stripe(StripeComponentUpsample {
        neighbors,
        spec: StripeComponentUpsampleSpec {
            plane_idx: 1,
            component: scaled.component(1),
            local_y_out: local_y,
            stripe_rows,
            width,
        },
        out: &mut scratch.g,
    });
    upsample_component_row_stripe(StripeComponentUpsample {
        neighbors,
        spec: StripeComponentUpsampleSpec {
            plane_idx: 2,
            component: scaled.component(2),
            local_y_out: local_y,
            stripe_rows,
            width,
        },
        out: &mut scratch.b,
    });
    upsample_component_row_stripe(StripeComponentUpsample {
        neighbors,
        spec: StripeComponentUpsampleSpec {
            plane_idx: 3,
            component: scaled.component(3),
            local_y_out: local_y,
            stripe_rows,
            width,
        },
        out: &mut scratch.k,
    });

    for x in 0..width {
        let (r, g, b) = match plan.color_space {
            ColorSpace::Cmyk => {
                inverted_cmyk_to_rgb(scratch.r[x], scratch.g[x], scratch.b[x], scratch.k[x])
            }
            ColorSpace::Ycck => ycck_to_rgb(scratch.r[x], scratch.g[x], scratch.b[x], scratch.k[x]),
            _ => unreachable!("four-component conversion requires CMYK/YCCK input"),
        };
        scratch.r[x] = r;
        scratch.g[x] = g;
        scratch.b[x] = b;
    }
}
