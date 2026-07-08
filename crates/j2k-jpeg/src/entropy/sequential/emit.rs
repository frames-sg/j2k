// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    is_ycbcr_420, scaled_dimensions, Fast420RegionLayout, OutputScratch, PreparedDecodePlan,
    RgbOutputScratch, StripeBuffer, StripePlane,
};
use crate::backend::{Backend, Rgb420ChromaRows, Rgb420Crop, Rgb420CroppedRowPair, Rgb420RowPair};
use crate::color::cmyk::{inverted_cmyk_to_rgb, ycck_to_rgb};
use crate::color::upsample::{
    upsample_1x1, upsample_h2v1_fancy_row, upsample_h2v2_fancy_row, upsample_h2v2_fancy_rows,
};
use crate::error::JpegError;
use crate::info::{ColorSpace, DownscaleFactor, Rect};
use crate::internal::scratch::{RgbGenericRows, SinkRows};
use crate::output::{InterleavedRgbWriter, OutputWriter};

pub(super) struct Fast420RegionStripe<'a> {
    pub(super) neighbors: StripeNeighbors<'a>,
    pub(super) stripe_index: u32,
    pub(super) roi: Rect,
    pub(super) region_layout: Fast420RegionLayout,
    pub(super) crop_rows: &'a mut SinkRows,
    pub(super) downscale: DownscaleFactor,
}

pub(super) fn emit_stripe_rgb_420_region<W: OutputWriter + InterleavedRgbWriter>(
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
pub(super) fn should_use_direct_420_crop(
    backend: Backend,
    _downscale: DownscaleFactor,
    row_width: usize,
    crop_width: usize,
) -> bool {
    backend.prefers_cropped_420_region(row_width, crop_width)
}

pub(super) fn emit_stripe_rgb_444<W: OutputWriter + InterleavedRgbWriter>(
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

#[derive(Clone, Copy)]
pub(super) struct StripeEmit<'a> {
    pub(super) prev: Option<&'a StripeBuffer>,
    pub(super) curr: &'a StripeBuffer,
    pub(super) next: Option<&'a StripeBuffer>,
    pub(super) stripe_index: u32,
    pub(super) source_width: usize,
    pub(super) downscale: DownscaleFactor,
}

pub(super) fn emit_stripe<W: OutputWriter>(
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

pub(super) fn emit_stripe_rgb<W: OutputWriter + InterleavedRgbWriter>(
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
                writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
                    backend.fill_rgb_row_from_gray(y_row, dst);
                    Ok(())
                })?;
            }
        }
        ColorSpace::YCbCr if is_ycbcr_420(plan) => {
            let RgbOutputScratch::YCbCr420 = output_scratch else {
                unreachable!("4:2:0 YCbCr RGB output requires dedicated scratch");
            };
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
                );
                let (prev_cr, curr_cr, next_cr) = component_row_triplet(
                    prev.map(|stripe| stripe.plane(2)),
                    curr.plane(2),
                    next.map(|stripe| stripe.plane(2)),
                    chroma_y,
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

            if cb_h == 1 && cb_v == 1 && cr_h == 1 && cr_v == 1 && max_h == 1 && max_v == 1 {
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
                    prev,
                    curr,
                    next,
                    local_y as u32,
                    width,
                    scratch,
                )?;
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

fn fill_four_component_rgb_row(
    plan: &PreparedDecodePlan,
    prev: Option<&StripeBuffer>,
    curr: &StripeBuffer,
    next: Option<&StripeBuffer>,
    local_y: u32,
    width: usize,
    scratch: &mut RgbGenericRows,
) -> Result<(), JpegError> {
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
    let neighbors = StripeNeighbors { prev, curr, next };

    upsample_component_row_stripe(StripeComponentUpsample {
        neighbors,
        spec: StripeComponentUpsampleSpec {
            plane_idx: 0,
            comp_h: c0_h,
            comp_v: c0_v,
            max_h,
            max_v,
            local_y_out: local_y,
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

#[derive(Clone, Copy)]
pub(super) struct StripeNeighbors<'a> {
    pub(super) prev: Option<&'a StripeBuffer>,
    pub(super) curr: &'a StripeBuffer,
    pub(super) next: Option<&'a StripeBuffer>,
}

#[derive(Clone, Copy)]
struct StripeComponentUpsampleSpec {
    plane_idx: usize,
    comp_h: u32,
    comp_v: u32,
    max_h: u32,
    max_v: u32,
    local_y_out: u32,
    width: usize,
}

struct StripeComponentUpsample<'a, 'b> {
    neighbors: StripeNeighbors<'a>,
    spec: StripeComponentUpsampleSpec,
    out: &'b mut [u8],
}

#[derive(Clone, Copy)]
struct Stripe420PairSpec {
    plane_idx: usize,
    local_y_out: u32,
    width: usize,
}

struct Stripe420PairUpsample<'a, 'b> {
    neighbors: StripeNeighbors<'a>,
    spec: Stripe420PairSpec,
    top: &'b mut [u8],
    bot: &'b mut [u8],
}

pub(super) fn component_row_triplet<'a>(
    prev: Option<StripePlane<'a>>,
    curr: StripePlane<'a>,
    next: Option<StripePlane<'a>>,
    local_row: usize,
) -> (&'a [u8], &'a [u8], &'a [u8]) {
    fn plane_row(plane: StripePlane<'_>, row: usize) -> &[u8] {
        let start = row * plane.stride;
        &plane.data[start..start + plane.stride]
    }

    let curr_rows = curr.rows;
    let prev_row = if local_row == 0 {
        match prev {
            Some(plane) => plane_row(plane, plane.rows - 1),
            None => plane_row(curr, 0),
        }
    } else {
        plane_row(curr, local_row - 1)
    };
    let curr_row = plane_row(curr, local_row);
    let next_row = if local_row + 1 < curr_rows {
        plane_row(curr, local_row + 1)
    } else {
        match next {
            Some(plane) => plane_row(plane, 0),
            None => plane_row(curr, curr_rows - 1),
        }
    };
    (prev_row, curr_row, next_row)
}

fn upsample_component_row_stripe(request: StripeComponentUpsample<'_, '_>) {
    let StripeComponentUpsample {
        neighbors,
        spec,
        out,
    } = request;
    let StripeNeighbors { prev, curr, next } = neighbors;
    let StripeComponentUpsampleSpec {
        plane_idx,
        comp_h,
        comp_v,
        max_h,
        max_v,
        local_y_out,
        width,
    } = spec;
    let v_ratio = max_v / comp_v;
    let h_ratio = max_h / comp_h;
    let curr_plane = curr.plane(plane_idx);
    let chroma_rows = curr_plane.rows as u32;
    let chroma_y = (local_y_out / v_ratio).min(chroma_rows.saturating_sub(1));
    let (prev_row, curr_row, next_row) = component_row_triplet(
        prev.map(|stripe| stripe.plane(plane_idx)),
        curr_plane,
        next.map(|stripe| stripe.plane(plane_idx)),
        chroma_y as usize,
    );

    match (h_ratio, v_ratio) {
        (1, 1) => {
            upsample_1x1(&curr_row[..width], out);
        }
        (2, 1) => {
            let chroma_cols = width.div_ceil(2);
            upsample_h2v1_fancy_row(&curr_row[..chroma_cols], width, out);
        }
        (2, 2) => {
            let chroma_cols = width.div_ceil(2);
            upsample_h2v2_fancy_row(
                &prev_row[..chroma_cols],
                &curr_row[..chroma_cols],
                &next_row[..chroma_cols],
                width,
                !local_y_out.is_multiple_of(2),
                out,
            );
        }
        _ => {
            for (x, slot) in out.iter_mut().enumerate().take(width) {
                let cx = ((x as u32) / h_ratio).min(curr_row.len() as u32 - 1);
                *slot = curr_row[cx as usize];
            }
        }
    }
}

fn upsample_420_pair(request: Stripe420PairUpsample<'_, '_>) {
    let Stripe420PairUpsample {
        neighbors,
        spec,
        top,
        bot,
    } = request;
    let StripeNeighbors { prev, curr, next } = neighbors;
    let Stripe420PairSpec {
        plane_idx,
        local_y_out,
        width,
    } = spec;
    let curr_plane = curr.plane(plane_idx);
    let chroma_rows = curr_plane.rows as u32;
    let chroma_y = (local_y_out / 2).min(chroma_rows.saturating_sub(1));
    let (prev_row, curr_row, next_row) = component_row_triplet(
        prev.map(|stripe| stripe.plane(plane_idx)),
        curr_plane,
        next.map(|stripe| stripe.plane(plane_idx)),
        chroma_y as usize,
    );

    upsample_h2v2_fancy_rows(prev_row, curr_row, next_row, width, top, bot);
}
