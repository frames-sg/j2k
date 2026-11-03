// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{PreparedDecodePlan, StripeBuffer};
use crate::{
    backend::Backend,
    error::JpegError,
    info::{ColorSpace, Rect},
    output::{InterleavedRgbWriter, OutputWriter},
};

/// Convert MCU row `stripe_index` of a 4:4:4 stripe to `writer`: output rows
/// of `roi` (the writer's row 0 is `roi.y`) and window columns
/// `crop_start..crop_start + roi.w`, as YCbCr or identity RGB.
pub(in crate::entropy::sequential) fn emit_stripe_rgb_444<
    W: OutputWriter + InterleavedRgbWriter,
>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    stripe: &StripeBuffer,
    stripe_index: u32,
    roi: Rect,
    crop_start: usize,
    writer: &mut W,
) -> Result<(), JpegError> {
    let y_start = stripe_index * 8;
    // `roi` lies inside the image, so it also clips the final MCU row's padding.
    let y_end = (y_start + 8).min(roi.y + roi.h);
    let columns = crop_start..crop_start + roi.w as usize;
    let identity = plan.color_space == ColorSpace::Rgb;
    let fill = |y: u32, dst: &mut [u8]| {
        let local_y = (y - y_start) as usize;
        let [c0, c1, c2] =
            [0, 1, 2].map(|plane_idx| &stripe.row(plane_idx, local_y)[columns.clone()]);
        if identity {
            backend.fill_rgb_row_from_rgb(c0, c1, c2, dst);
        } else {
            backend.fill_rgb_row_from_ycbcr(c0, c1, c2, dst);
        }
    };

    let mut y = y_start.max(roi.y);
    while y + 1 < y_end {
        writer.with_rgb_rows(y - roi.y, 2, |dst_top, dst_bottom| {
            let dst_bottom = dst_bottom.ok_or(JpegError::InternalInvariant {
                reason: "row_count=2 supplies bottom row",
            })?;
            fill(y, dst_top);
            fill(y + 1, dst_bottom);
            Ok(())
        })?;
        y += 2;
    }
    if y < y_end {
        writer.with_rgb_rows(y - roi.y, 1, |dst, _| {
            fill(y, dst);
            Ok(())
        })?;
    }
    Ok(())
}
