// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    super::{scaled_dimensions, uses_fancy_420_emit, OutputScratch, PreparedDecodePlan},
    four_component::{fill_four_component_rgb_row, FourComponentRow},
    types::{ensure_color_components, StripeEmit, StripeNeighbors},
    upsample::{
        upsample_420_pair, upsample_component_row_stripe, Stripe420PairSpec, Stripe420PairUpsample,
        StripeComponentUpsample, StripeComponentUpsampleSpec,
    },
};
use crate::{error::JpegError, info::ColorSpace, output::OutputWriter};

#[expect(
    clippy::too_many_lines,
    reason = "stripe emission keeps format-specific row construction and writer calls in output order"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "stripe row indices are bounded by validated u32 JPEG dimensions before writer dispatch"
)]
pub(in crate::entropy::sequential) fn emit_stripe<W: OutputWriter>(
    plan: &PreparedDecodePlan,
    writer: &mut W,
    output_scratch: &mut OutputScratch<'_>,
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
                writer.write_gray_row(y_start + local_y as u32, y_row)?;
            }
        }
        ColorSpace::YCbCr => {
            if uses_fancy_420_emit(plan, scaled) {
                let OutputScratch::YCbCr420(scratch) = output_scratch else {
                    unreachable!("4:2:0 YCbCr requires dedicated scratch");
                };

                let mut local_y = 0usize;
                while local_y < stripe_rows {
                    let y_top = &curr.row(0, local_y)[..width];
                    let next_local_y = local_y + 1;
                    let y_bottom =
                        (next_local_y < stripe_rows).then(|| &curr.row(0, next_local_y)[..width]);

                    upsample_420_pair(Stripe420PairUpsample {
                        neighbors,
                        spec: Stripe420PairSpec {
                            plane_idx: 1,
                            local_y_out: local_y as u32,
                            stripe_rows,
                            width,
                        },
                        top: &mut scratch.cb_top,
                        bot: &mut scratch.cb_bot,
                    });
                    upsample_420_pair(Stripe420PairUpsample {
                        neighbors,
                        spec: Stripe420PairSpec {
                            plane_idx: 2,
                            local_y_out: local_y as u32,
                            stripe_rows,
                            width,
                        },
                        top: &mut scratch.cr_top,
                        bot: &mut scratch.cr_bot,
                    });

                    writer.write_ycbcr_row(
                        y_start + local_y as u32,
                        y_top,
                        &scratch.cb_top,
                        &scratch.cr_top,
                    )?;
                    if let Some(y_bottom) = y_bottom {
                        writer.write_ycbcr_row(
                            y_start + next_local_y as u32,
                            y_bottom,
                            &scratch.cb_bot,
                            &scratch.cr_bot,
                        )?;
                    }
                    local_y += 2;
                }
            } else {
                let OutputScratch::YCbCrGeneric(scratch) = output_scratch else {
                    unreachable!("generic YCbCr requires reusable row scratch");
                };

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
                    writer.write_ycbcr_row(
                        y_start + local_y as u32,
                        y_row,
                        &scratch.cb_up,
                        &scratch.cr_up,
                    )?;
                }
            }
        }
        ColorSpace::Rgb => {
            let OutputScratch::RgbGeneric(scratch) = output_scratch else {
                unreachable!("RGB decode requires reusable row scratch");
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
                writer.write_rgb_row(
                    y_start + local_y as u32,
                    &scratch.r,
                    &scratch.g,
                    &scratch.b,
                )?;
            }
        }
        ColorSpace::Cmyk | ColorSpace::Ycck => {
            let OutputScratch::RgbGeneric(scratch) = output_scratch else {
                unreachable!("CMYK/YCCK decode requires reusable row scratch");
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
                writer.write_rgb_row(
                    y_start + local_y as u32,
                    &scratch.r[..width],
                    &scratch.g[..width],
                    &scratch.b[..width],
                )?;
            }
        }
    }
    Ok(())
}
