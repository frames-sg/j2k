// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    super::{is_ycbcr_420, scaled_dimensions, OutputScratch, PreparedDecodePlan},
    four_component::fill_four_component_rgb_row,
    types::{StripeEmit, StripeNeighbors},
    upsample::{
        upsample_420_pair, upsample_component_row_stripe, Stripe420PairSpec, Stripe420PairUpsample,
        StripeComponentUpsample, StripeComponentUpsampleSpec,
    },
};
use crate::{error::JpegError, info::ColorSpace, output::OutputWriter};

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
    } = emit;
    let max_v = plan.sampling.max_v as u32;
    let mcu_height_px = downscale.output_block_size() * max_v;
    let y_start = stripe_index * mcu_height_px;
    let (_, scaled_height) = scaled_dimensions(plan.dimensions, downscale);
    let y_end = (y_start + mcu_height_px).min(scaled_height);
    let stripe_rows = (y_end - y_start) as usize;

    if stripe_rows == 0 {
        return Ok(());
    }

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
            let (cb_h, cb_v) = plan
                .sampling
                .component(1)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 1 })?;
            let (cr_h, cr_v) = plan
                .sampling
                .component(2)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 2 })?;

            let max_h = plan.sampling.max_h as u32;
            let max_v = plan.sampling.max_v as u32;

            if is_ycbcr_420(plan) {
                let OutputScratch::YCbCr420(scratch) = output_scratch else {
                    unreachable!("4:2:0 YCbCr requires dedicated scratch");
                };
                debug_assert!(
                    max_h == 2 && max_v == 2 && cb_h == 1 && cb_v == 1 && cr_h == 1 && cr_v == 1
                );

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
                            comp_h: cb_h,
                            comp_v: cb_v,
                            max_h,
                            max_v,
                            local_y_out: local_y as u32,
                            width,
                        },
                        out: &mut scratch.cb_up,
                    });
                    upsample_component_row_stripe(StripeComponentUpsample {
                        neighbors,
                        spec: StripeComponentUpsampleSpec {
                            plane_idx: 2,
                            comp_h: cr_h,
                            comp_v: cr_v,
                            max_h,
                            max_v,
                            local_y_out: local_y as u32,
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
            let (r_h, r_v) = plan
                .sampling
                .component(0)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 0 })?;
            let (g_h, g_v) = plan
                .sampling
                .component(1)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 1 })?;
            let (b_h, b_v) = plan
                .sampling
                .component(2)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 2 })?;

            let max_h = plan.sampling.max_h as u32;
            let max_v = plan.sampling.max_v as u32;

            let OutputScratch::RgbGeneric(scratch) = output_scratch else {
                unreachable!("RGB decode requires reusable row scratch");
            };

            for local_y in 0..stripe_rows {
                upsample_component_row_stripe(StripeComponentUpsample {
                    neighbors,
                    spec: StripeComponentUpsampleSpec {
                        plane_idx: 0,
                        comp_h: r_h,
                        comp_v: r_v,
                        max_h,
                        max_v,
                        local_y_out: local_y as u32,
                        width,
                    },
                    out: &mut scratch.r,
                });
                upsample_component_row_stripe(StripeComponentUpsample {
                    neighbors,
                    spec: StripeComponentUpsampleSpec {
                        plane_idx: 1,
                        comp_h: g_h,
                        comp_v: g_v,
                        max_h,
                        max_v,
                        local_y_out: local_y as u32,
                        width,
                    },
                    out: &mut scratch.g,
                });
                upsample_component_row_stripe(StripeComponentUpsample {
                    neighbors,
                    spec: StripeComponentUpsampleSpec {
                        plane_idx: 2,
                        comp_h: b_h,
                        comp_v: b_v,
                        max_h,
                        max_v,
                        local_y_out: local_y as u32,
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
                    prev,
                    curr,
                    next,
                    local_y as u32,
                    width,
                    scratch,
                )?;
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
