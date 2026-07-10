// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{PreparedDecodePlan, StripeBuffer};
use crate::{
    backend::Backend,
    error::JpegError,
    output::{InterleavedRgbWriter, OutputWriter},
};

pub(in crate::entropy::sequential) fn emit_stripe_rgb_444<
    W: OutputWriter + InterleavedRgbWriter,
>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    stripe: &StripeBuffer,
    stripe_index: u32,
    writer: &mut W,
) -> Result<(), JpegError> {
    let (width, height) = plan.dimensions;
    let y_start = stripe_index * 8;
    let stripe_rows = (height.saturating_sub(y_start)).min(8) as usize;
    let width = width as usize;
    let y_stride = stripe.plane_strides[0];
    let cb_stride = stripe.plane_strides[1];
    let cr_stride = stripe.plane_strides[2];

    let mut local_y = 0usize;
    while local_y + 1 < stripe_rows {
        let y_top_start = local_y * y_stride;
        let y_bottom_start = y_top_start + y_stride;
        let cb_top_start = local_y * cb_stride;
        let cb_bottom_start = cb_top_start + cb_stride;
        let cr_top_start = local_y * cr_stride;
        let cr_bottom_start = cr_top_start + cr_stride;
        writer.with_rgb_rows(y_start + local_y as u32, 2, |dst_top, dst_bottom| {
            let dst_bottom = dst_bottom.ok_or(JpegError::InternalInvariant {
                reason: "row_count=2 supplies bottom row",
            })?;
            backend.fill_rgb_row_from_ycbcr(
                &stripe.planes[0][y_top_start..y_top_start + width],
                &stripe.planes[1][cb_top_start..cb_top_start + width],
                &stripe.planes[2][cr_top_start..cr_top_start + width],
                dst_top,
            );
            backend.fill_rgb_row_from_ycbcr(
                &stripe.planes[0][y_bottom_start..y_bottom_start + width],
                &stripe.planes[1][cb_bottom_start..cb_bottom_start + width],
                &stripe.planes[2][cr_bottom_start..cr_bottom_start + width],
                dst_bottom,
            );
            Ok(())
        })?;
        local_y += 2;
    }

    if local_y < stripe_rows {
        let y_row_start = local_y * y_stride;
        let cb_row_start = local_y * cb_stride;
        let cr_row_start = local_y * cr_stride;
        writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
            backend.fill_rgb_row_from_ycbcr(
                &stripe.planes[0][y_row_start..y_row_start + width],
                &stripe.planes[1][cb_row_start..cb_row_start + width],
                &stripe.planes[2][cr_row_start..cr_row_start + width],
                dst,
            );
            Ok(())
        })?;
    }

    Ok(())
}
