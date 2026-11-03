// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::layout::CropWindow;
use super::super::{PreparedDecodePlan, StripeBuffer};
use super::rgb::uses_fused_422_row;
use crate::{
    backend::{Backend, Rgb422Row},
    color::upsample::upsample_h2v1_fancy_row,
    error::JpegError,
    info::{ColorSpace, Rect},
    internal::scratch::YCbCrGenericRows,
    output::{InterleavedRgbWriter, OutputWriter},
};

/// Converts MCU rows of a full-size 4:4:4 (YCbCr or identity RGB) or 4:2:2
/// stripe to the writer, for the rows of `roi` (the writer's row 0 is
/// `roi.y`) and the `crop` columns of the decoded window.
#[derive(Clone, Copy)]
pub(in crate::entropy::sequential) struct FastRgbRowsEmitter<'p> {
    pub(in crate::entropy::sequential) plan: &'p PreparedDecodePlan,
    pub(in crate::entropy::sequential) backend: Backend,
    pub(in crate::entropy::sequential) roi: Rect,
    pub(in crate::entropy::sequential) crop: CropWindow,
}

impl FastRgbRowsEmitter<'_> {
    /// Convert MCU row `stripe_index` of `stripe`. 4:2:2 chroma is
    /// fancy-upsampled across the whole decoded window into `upsampled`, so
    /// its edge columns see the same neighbours as a full-width decode, then
    /// converted for the requested columns.
    pub(in crate::entropy::sequential) fn emit_stripe_rgb_rows<W>(
        self,
        stripe: &StripeBuffer,
        stripe_index: u32,
        upsampled: &mut YCbCrGenericRows,
        writer: &mut W,
    ) -> Result<(), JpegError>
    where
        W: OutputWriter + InterleavedRgbWriter,
    {
        let Self {
            plan,
            backend,
            roi,
            crop,
        } = self;
        let row_width = crop.row_width;
        let y_start = stripe_index * 8;
        // `roi` lies inside the image, so it also clips the final MCU row's padding.
        let y_end = (y_start + 8).min(roi.y + roi.h);
        let columns = crop.columns();
        let h2v1 = plan.sampling.max_h == 2;
        let chroma_width = if h2v1 { crop.chroma_width() } else { row_width };
        let fused_422 = h2v1 && crop.is_full_width() && uses_fused_422_row(backend, row_width);
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
}
