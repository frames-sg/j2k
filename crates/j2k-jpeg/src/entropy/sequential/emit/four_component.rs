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
    error::JpegError,
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
    neighbors: StripeNeighbors<'_>,
    row: FourComponentRow,
    scratch: &mut RgbGenericRows,
) -> Result<(), JpegError> {
    let FourComponentRow {
        local_y,
        stripe_rows,
        width,
    } = row;
    let (c0_h, c0_v) = plan
        .sampling
        .component(0)
        .map(|(h, v)| (u32::from(h), u32::from(v)))
        .ok_or(JpegError::UnsupportedComponentCount { count: 0 })?;
    let (c1_h, c1_v) = plan
        .sampling
        .component(1)
        .map(|(h, v)| (u32::from(h), u32::from(v)))
        .ok_or(JpegError::UnsupportedComponentCount { count: 1 })?;
    let (c2_h, c2_v) = plan
        .sampling
        .component(2)
        .map(|(h, v)| (u32::from(h), u32::from(v)))
        .ok_or(JpegError::UnsupportedComponentCount { count: 2 })?;
    let (k_h, k_v) = plan
        .sampling
        .component(3)
        .map(|(h, v)| (u32::from(h), u32::from(v)))
        .ok_or(JpegError::UnsupportedComponentCount { count: 3 })?;
    let max_h = u32::from(plan.sampling.max_h);
    let max_v = u32::from(plan.sampling.max_v);

    upsample_component_row_stripe(StripeComponentUpsample {
        neighbors,
        spec: StripeComponentUpsampleSpec {
            plane_idx: 0,
            comp_h: c0_h,
            comp_v: c0_v,
            max_h,
            max_v,
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
            comp_h: c1_h,
            comp_v: c1_v,
            max_h,
            max_v,
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
            comp_h: c2_h,
            comp_v: c2_v,
            max_h,
            max_v,
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
            comp_h: k_h,
            comp_v: k_v,
            max_h,
            max_v,
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

    Ok(())
}
