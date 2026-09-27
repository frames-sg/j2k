// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    super::{scaled_dimensions, uses_fancy_420_emit, PreparedDecodePlan, RgbOutputScratch},
    four_component::{fill_four_component_rgb_row, FourComponentRow},
    types::{ensure_color_components, StripeEmit, StripeNeighbors},
    upsample::{
        component_row_triplet, upsample_component_row_stripe, valid_component_rows,
        StripeComponentUpsample, StripeComponentUpsampleSpec,
    },
};
use crate::{
    backend::{Backend, Rgb420ChromaRows, Rgb420RowPair},
    error::JpegError,
    info::ColorSpace,
    output::{InterleavedRgbWriter, OutputWriter},
};

#[expect(
    clippy::too_many_lines,
    reason = "RGB stripe emission keeps sampling-family dispatch, upsampling, and writer calls in output order"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "stripe and component row indices are bounded by validated u32 JPEG dimensions"
)]
pub(in crate::entropy::sequential) fn emit_stripe_rgb<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    writer: &mut W,
    output_scratch: &mut RgbOutputScratch<'_>,
    emit: StripeEmit<'_>,
) -> Result<(), JpegError> {
    let StripeEmit {
        prev,
        curr,
        next,
        stripe_index,
        source_width,
        downscale,
        scaled,
    } = emit;
    let max_v = u32::from(plan.sampling.max_v);
    let mcu_height_px = downscale.output_block_size() * max_v;
    let y_start = stripe_index * mcu_height_px;
    let (_, scaled_height) = scaled_dimensions(plan.dimensions, downscale);
    let y_end = (y_start + mcu_height_px).min(scaled_height);
    let stripe_rows = (y_end - y_start) as usize;

    if stripe_rows == 0 {
        return Ok(());
    }
    ensure_color_components(plan)?;

    let width = source_width;
    let neighbors = StripeNeighbors { prev, curr, next };
    match plan.color_space {
        ColorSpace::Grayscale => {
            for local_y in 0..stripe_rows {
                let y_row = &curr.row(0, local_y)[..width];
                writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
                    backend.fill_rgb_row_from_gray(y_row, dst);
                    Ok(())
                })?;
            }
        }
        ColorSpace::YCbCr if uses_fancy_420_emit(plan, scaled) => {
            let RgbOutputScratch::YCbCr420 = output_scratch else {
                unreachable!("4:2:0 YCbCr RGB output requires dedicated scratch");
            };
            let chroma_valid_rows = valid_component_rows(stripe_rows, 2);
            let mut local_y = 0usize;
            while local_y < stripe_rows {
                let y_top = &curr.row(0, local_y)[..width];
                let next_local_y = local_y + 1;
                let y_bottom =
                    (next_local_y < stripe_rows).then(|| &curr.row(0, next_local_y)[..width]);
                let row_count = if y_bottom.is_some() { 2 } else { 1 };
                let chroma_y = (local_y / 2).min(curr.row_count(1).saturating_sub(1));
                let chroma_cols = width.div_ceil(2);
                let (prev_cb, curr_cb, next_cb) = component_row_triplet(
                    prev.map(|stripe| stripe.plane(1)),
                    curr.plane(1),
                    next.map(|stripe| stripe.plane(1)),
                    chroma_y,
                    chroma_valid_rows,
                );
                let (prev_cr, curr_cr, next_cr) = component_row_triplet(
                    prev.map(|stripe| stripe.plane(2)),
                    curr.plane(2),
                    next.map(|stripe| stripe.plane(2)),
                    chroma_y,
                    chroma_valid_rows,
                );

                writer.with_rgb_rows(
                    y_start + local_y as u32,
                    row_count,
                    |dst_top, dst_bottom| {
                        backend.fill_rgb_row_pair_from_420(Rgb420RowPair::new(
                            y_top,
                            y_bottom,
                            Rgb420ChromaRows::new(
                                &prev_cb[..chroma_cols],
                                &curr_cb[..chroma_cols],
                                &next_cb[..chroma_cols],
                                &prev_cr[..chroma_cols],
                                &curr_cr[..chroma_cols],
                                &next_cr[..chroma_cols],
                            ),
                            dst_top,
                            dst_bottom,
                        ));
                        Ok(())
                    },
                )?;
                local_y += 2;
            }
        }
        ColorSpace::YCbCr => {
            let RgbOutputScratch::YCbCrGeneric(scratch) = output_scratch else {
                unreachable!("generic YCbCr RGB output requires reusable row scratch");
            };

            // Every plane is at output resolution, including 4:2:0 chroma
            // that a reduced scale decodes with a larger IDCT.
            if scaled.is_unsampled() {
                for local_y in 0..stripe_rows {
                    let y_row = &curr.row(0, local_y)[..width];
                    let cb_row = &curr.row(1, local_y)[..width];
                    let cr_row = &curr.row(2, local_y)[..width];
                    writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
                        backend.fill_rgb_row_from_ycbcr(y_row, cb_row, cr_row, dst);
                        Ok(())
                    })?;
                }
                return Ok(());
            }

            for local_y in 0..stripe_rows {
                let y_row = &curr.row(0, local_y)[..width];
                upsample_component_row_stripe(StripeComponentUpsample {
                    neighbors,
                    spec: StripeComponentUpsampleSpec {
                        plane_idx: 1,
                        component: scaled.component(1),
                        local_y_out: local_y as u32,
                        stripe_rows,
                        width,
                    },
                    out: &mut scratch.cb_up,
                });
                upsample_component_row_stripe(StripeComponentUpsample {
                    neighbors,
                    spec: StripeComponentUpsampleSpec {
                        plane_idx: 2,
                        component: scaled.component(2),
                        local_y_out: local_y as u32,
                        stripe_rows,
                        width,
                    },
                    out: &mut scratch.cr_up,
                });
                writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
                    backend.fill_rgb_row_from_ycbcr(y_row, &scratch.cb_up, &scratch.cr_up, dst);
                    Ok(())
                })?;
            }
        }
        ColorSpace::Rgb => {
            let RgbOutputScratch::RgbGeneric(scratch) = output_scratch else {
                unreachable!("RGB output requires reusable row scratch");
            };

            for local_y in 0..stripe_rows {
                upsample_component_row_stripe(StripeComponentUpsample {
                    neighbors,
                    spec: StripeComponentUpsampleSpec {
                        plane_idx: 0,
                        component: scaled.component(0),
                        local_y_out: local_y as u32,
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
                        local_y_out: local_y as u32,
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
                        local_y_out: local_y as u32,
                        stripe_rows,
                        width,
                    },
                    out: &mut scratch.b,
                });
                writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
                    backend.fill_rgb_row_from_rgb(&scratch.r, &scratch.g, &scratch.b, dst);
                    Ok(())
                })?;
            }
        }
        ColorSpace::Cmyk | ColorSpace::Ycck => {
            let RgbOutputScratch::RgbGeneric(scratch) = output_scratch else {
                unreachable!("CMYK/YCCK RGB output requires reusable row scratch");
            };
            for local_y in 0..stripe_rows {
                fill_four_component_rgb_row(
                    plan,
                    scaled,
                    neighbors,
                    FourComponentRow {
                        local_y: local_y as u32,
                        stripe_rows,
                        width,
                    },
                    scratch,
                );
                writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
                    backend.fill_rgb_row_from_rgb(
                        &scratch.r[..width],
                        &scratch.g[..width],
                        &scratch.b[..width],
                        dst,
                    );
                    Ok(())
                })?;
            }
        }
    }

    Ok(())
}
