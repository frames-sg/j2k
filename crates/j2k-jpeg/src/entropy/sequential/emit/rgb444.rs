// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{PreparedDecodePlan, StripeBuffer};
use crate::{
    backend::{Backend, Rgb422Row},
    color::upsample::upsample_h2v1_fancy_row,
    error::JpegError,
    info::{ColorSpace, Rect},
    internal::scratch::YCbCrGenericRows,
    output::{InterleavedRgbWriter, OutputWriter},
};

/// The part of a decoded window of horizontal MCUs that reaches the writer.
#[derive(Clone, Copy)]
pub(in crate::entropy::sequential) struct StripeOutputWindow {
    /// Requested output rectangle; the writer's row 0 is `roi.y`.
    pub(in crate::entropy::sequential) roi: Rect,
    /// Window column of `roi.x`.
    pub(in crate::entropy::sequential) crop_start: usize,
    /// Visible luma width of the decoded window.
    pub(in crate::entropy::sequential) row_width: usize,
}

/// Convert MCU row `stripe_index` of a full-size 4:4:4 (YCbCr or identity
/// RGB) or 4:2:2 stripe to `writer`, for the rows and columns of `window`.
/// 4:2:2 chroma is fancy-upsampled across the whole decoded window into
/// `upsampled`, so its edge columns see the same neighbours as a full-width
/// decode, then converted for the requested columns.
pub(in crate::entropy::sequential) fn emit_stripe_rgb_v1<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    stripe: &StripeBuffer,
    stripe_index: u32,
    window: StripeOutputWindow,
    upsampled: &mut YCbCrGenericRows,
    writer: &mut W,
) -> Result<(), JpegError> {
    let StripeOutputWindow {
        roi,
        crop_start,
        row_width,
    } = window;
    let y_start = stripe_index * 8;
    // `roi` lies inside the image, so it also clips the final MCU row's padding.
    let y_end = (y_start + 8).min(roi.y + roi.h);
    let columns = crop_start..crop_start + roi.w as usize;
    let h2v1 = plan.sampling.max_h == 2;
    let chroma_width = if h2v1 {
        row_width.div_ceil(2)
    } else {
        row_width
    };
    let fused_422 =
        h2v1 && columns == (0..row_width) && row_width >= 33 && backend.prefers_fused_422();
    let identity = plan.color_space == ColorSpace::Rgb;

    for y in y_start.max(roi.y)..y_end {
        let local_y = (y - y_start) as usize;
        let [c0, c1, c2] = [0, 1, 2].map(|plane_idx| stripe.row(plane_idx, local_y));
        writer.with_rgb_rows(y - roi.y, 1, |dst, _| {
            if fused_422 {
                backend.fill_rgb_row_from_422(Rgb422Row::new(
                    &c0[..row_width],
                    &c1[..chroma_width],
                    &c2[..chroma_width],
                    dst,
                ));
                return Ok(());
            }
            let (c1, c2) = if h2v1 {
                upsample_h2v1_fancy_row(&c1[..chroma_width], row_width, &mut upsampled.cb_up);
                upsample_h2v1_fancy_row(&c2[..chroma_width], row_width, &mut upsampled.cr_up);
                (&upsampled.cb_up[..], &upsampled.cr_up[..])
            } else {
                (c1, c2)
            };
            let [c0, c1, c2] = [c0, c1, c2].map(|row| &row[columns.clone()]);
            if identity {
                backend.fill_rgb_row_from_rgb(c0, c1, c2, dst);
            } else {
                backend.fill_rgb_row_from_ycbcr(c0, c1, c2, dst);
            }
            Ok(())
        })?;
    }
    Ok(())
}
