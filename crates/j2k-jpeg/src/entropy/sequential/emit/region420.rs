// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    super::{scaled_dimensions, PreparedDecodePlan},
    types::{Fast420RegionStripe, StripeNeighbors},
    upsample::component_row_triplet,
};
use crate::{
    backend::{Backend, Rgb420ChromaRows, Rgb420Crop, Rgb420CroppedRowPair, Rgb420RowPair},
    error::JpegError,
    info::DownscaleFactor,
    output::{InterleavedRgbWriter, OutputWriter},
};

pub(in crate::entropy::sequential) fn emit_stripe_rgb_420_region<
    W: OutputWriter + InterleavedRgbWriter,
>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    writer: &mut W,
    stripe: Fast420RegionStripe<'_>,
) -> Result<(), JpegError> {
    let Fast420RegionStripe {
        neighbors,
        stripe_index,
        roi,
        region_layout,
        crop_rows,
        downscale,
    } = stripe;
    let StripeNeighbors { prev, curr, next } = neighbors;
    let max_v = plan.sampling.max_v as u32;
    let mcu_height_px = downscale.output_block_size() * max_v;
    let y_start = stripe_index * mcu_height_px;
    let (_, scaled_height) = scaled_dimensions(plan.dimensions, downscale);
    let y_end = (y_start + mcu_height_px).min(scaled_height);
    let stripe_rows = (y_end - y_start) as usize;

    if stripe_rows == 0 {
        return Ok(());
    }

    let row_width = region_layout.row_width();
    let chroma_width = row_width.div_ceil(2);
    let row_len = row_width * 3;
    let crop_width = region_layout.crop_end - region_layout.crop_start;
    let crop_len = crop_width * 3;
    let use_direct_crop = should_use_direct_420_crop(backend, downscale, row_width, crop_width);
    let mut local_y = 0usize;
    while local_y < stripe_rows {
        let next_local_y = local_y + 1;
        let global_y = y_start + local_y as u32;
        let top_in = global_y >= roi.y && global_y < roi.y + roi.h;
        let bottom_in =
            next_local_y < stripe_rows && global_y + 1 >= roi.y && global_y + 1 < roi.y + roi.h;
        if !top_in && !bottom_in {
            local_y += 2;
            continue;
        }

        let y_top = &curr.row(0, local_y)[..row_width];
        let y_bottom =
            (next_local_y < stripe_rows).then(|| &curr.row(0, next_local_y)[..row_width]);
        let chroma_y = (local_y / 2).min(curr.row_count(1).saturating_sub(1));
        let (prev_cb, curr_cb, next_cb) = component_row_triplet(
            prev.map(|stripe| stripe.plane(1)),
            curr.plane(1),
            next.map(|stripe| stripe.plane(1)),
            chroma_y,
        );
        let (prev_cr, curr_cr, next_cr) = component_row_triplet(
            prev.map(|stripe| stripe.plane(2)),
            curr.plane(2),
            next.map(|stripe| stripe.plane(2)),
            chroma_y,
        );
        let chroma = Rgb420ChromaRows::new(
            &prev_cb[..chroma_width],
            &curr_cb[..chroma_width],
            &next_cb[..chroma_width],
            &prev_cr[..chroma_width],
            &curr_cr[..chroma_width],
            &next_cr[..chroma_width],
        );

        if use_direct_crop {
            match (top_in, bottom_in) {
                (true, true) => {
                    writer.with_rgb_rows(global_y - roi.y, 2, |dst_top, dst_bottom| {
                        backend.fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
                            Rgb420RowPair::new(y_top, y_bottom, chroma, dst_top, dst_bottom),
                            Rgb420Crop::new(region_layout.crop_start, crop_width),
                        ));
                        Ok(())
                    })?;
                }
                (true, false) => {
                    writer.with_rgb_rows(global_y - roi.y, 1, |dst, _| {
                        backend.fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
                            Rgb420RowPair::new(y_top, None, chroma, dst, None),
                            Rgb420Crop::new(region_layout.crop_start, crop_width),
                        ));
                        Ok(())
                    })?;
                }
                (false, true) => {
                    let y_bottom = y_bottom.ok_or(JpegError::InternalInvariant {
                        reason: "bottom ROI row requires a decoded bottom row",
                    })?;
                    writer.with_rgb_rows(global_y + 1 - roi.y, 1, |dst, _| {
                        backend.fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
                            Rgb420RowPair::new(
                                y_top,
                                Some(y_bottom),
                                chroma,
                                &mut crop_rows.top_row[..crop_len],
                                Some(dst),
                            ),
                            Rgb420Crop::new(region_layout.crop_start, crop_width),
                        ));
                        Ok(())
                    })?;
                }
                (false, false) => unreachable!("ROI row pair must intersect at least one row"),
            }
            local_y += 2;
            continue;
        }

        backend.fill_rgb_row_pair_from_420(Rgb420RowPair::new(
            y_top,
            y_bottom,
            chroma,
            &mut crop_rows.top_row[..row_len],
            y_bottom
                .as_ref()
                .map(|_| &mut crop_rows.bottom_row[..row_len]),
        ));

        let x0 = region_layout.crop_start * 3;
        let x1 = region_layout.crop_end * 3;
        match (top_in, bottom_in) {
            (true, true) => {
                writer.with_rgb_rows(global_y - roi.y, 2, |dst_top, dst_bottom| {
                    dst_top.copy_from_slice(&crop_rows.top_row[x0..x1]);
                    let dst_bottom = dst_bottom.ok_or(JpegError::InternalInvariant {
                        reason: "row_count=2 supplies bottom row",
                    })?;
                    dst_bottom.copy_from_slice(&crop_rows.bottom_row[x0..x1]);
                    Ok(())
                })?;
            }
            (true, false) => {
                writer.with_rgb_rows(global_y - roi.y, 1, |dst, _| {
                    dst.copy_from_slice(&crop_rows.top_row[x0..x1]);
                    Ok(())
                })?;
            }
            (false, true) => {
                writer.with_rgb_rows(global_y + 1 - roi.y, 1, |dst, _| {
                    dst.copy_from_slice(&crop_rows.bottom_row[x0..x1]);
                    Ok(())
                })?;
            }
            (false, false) => unreachable!("ROI row pair must intersect at least one row"),
        }

        local_y += 2;
    }

    Ok(())
}

#[inline]
pub(in crate::entropy::sequential) fn should_use_direct_420_crop(
    backend: Backend,
    _downscale: DownscaleFactor,
    row_width: usize,
    crop_width: usize,
) -> bool {
    backend.prefers_cropped_420_region(row_width, crop_width)
}
