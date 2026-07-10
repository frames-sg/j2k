// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{PreparedComponentPlan, PreparedDecodePlan};
use crate::info::{ColorSpace, DownscaleFactor, Rect};

#[derive(Clone, Copy)]
pub(super) struct RgbCropWindow {
    scratch_x0: usize,
    scratch_x1: usize,
}

impl RgbCropWindow {
    pub(super) fn new(width: usize, roi: Rect) -> Self {
        let roi_x0 = roi.x as usize;
        let roi_x1 = roi_x0 + roi.w as usize;
        let chroma_width = width.div_ceil(2);
        let sample_start = (roi_x0 / 2).saturating_sub(1);
        let sample_end = (roi_x1.div_ceil(2) + 1).min(chroma_width);
        let scratch_x0 = sample_start * 2;
        let scratch_x1 = (sample_end * 2).min(width);
        Self {
            scratch_x0,
            scratch_x1,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StripeRegionLayout {
    pub(crate) stripe_mcu_start: u32,
    pub(crate) stripe_mcus_per_row: u32,
    pub(crate) source_x0: u32,
    pub(crate) source_width: u32,
}

impl StripeRegionLayout {
    fn new(full_width: u32, mcu_width_px: u32, output_rect: Rect) -> Self {
        let source_x0 = (output_rect.x / mcu_width_px) * mcu_width_px;
        let source_x1 = output_rect
            .x
            .saturating_add(output_rect.w)
            .div_ceil(mcu_width_px)
            .saturating_mul(mcu_width_px)
            .min(full_width);
        let stripe_mcu_start = source_x0 / mcu_width_px;
        let stripe_mcu_end = source_x1.div_ceil(mcu_width_px);
        Self {
            stripe_mcu_start,
            stripe_mcus_per_row: stripe_mcu_end.saturating_sub(stripe_mcu_start),
            source_x0,
            source_width: source_x1.saturating_sub(source_x0),
        }
    }

    pub(super) fn source_width_usize(self) -> usize {
        self.source_width as usize
    }
}

pub(crate) fn stripe_region_layout(
    plan: &PreparedDecodePlan,
    downscale: DownscaleFactor,
    output_rect: Rect,
) -> StripeRegionLayout {
    let (scaled_width, scaled_height) = scaled_dimensions(plan.dimensions, downscale);
    let expanded_rect = expanded_output_rect(output_rect, scaled_width, scaled_height);
    let mcu_width_px = downscale.output_block_size() * u32::from(plan.sampling.max_h);
    StripeRegionLayout::new(scaled_width, mcu_width_px, expanded_rect)
}

#[inline]
pub(super) fn last_mcu_row_for_rect(rect: Rect, mcu_height_px: u32, mcu_rows: u32) -> u32 {
    let last_y = rect.y.saturating_add(rect.h).saturating_sub(1);
    (last_y / mcu_height_px).min(mcu_rows.saturating_sub(1))
}

#[inline]
pub(super) fn first_mcu_row_for_rect(rect: Rect, mcu_height_px: u32) -> u32 {
    rect.y / mcu_height_px
}

#[inline]
pub(super) fn first_decode_mcu_row_for_rect(
    full_output_rect: bool,
    rect: Rect,
    mcu_height_px: u32,
) -> u32 {
    if full_output_rect {
        0
    } else {
        first_mcu_row_for_rect(rect, mcu_height_px).saturating_sub(1)
    }
}

#[inline]
pub(super) fn decode_mcu_row_end_for_rect(
    full_output_rect: bool,
    rect: Rect,
    mcu_height_px: u32,
    mcu_rows: u32,
) -> u32 {
    if full_output_rect {
        return mcu_rows;
    }
    let last_output_mcu_row = last_mcu_row_for_rect(rect, mcu_height_px, mcu_rows);
    if last_output_mcu_row + 1 < mcu_rows {
        last_output_mcu_row + 2
    } else {
        mcu_rows
    }
}

#[inline]
pub(super) fn fast420_first_decode_mcu_row(roi: Rect, mcu_height_px: u32) -> u32 {
    let first_row = first_mcu_row_for_rect(roi, mcu_height_px);
    if roi.y.is_multiple_of(mcu_height_px) {
        first_row.saturating_sub(1)
    } else {
        first_row
    }
}

#[inline]
pub(super) fn fast420_decode_mcu_row_end(roi: Rect, mcu_height_px: u32, mcu_rows: u32) -> u32 {
    let last_row = last_mcu_row_for_rect(roi, mcu_height_px, mcu_rows);
    let last_local_y = (roi.y + roi.h - 1) % mcu_height_px;
    let needs_next_row = last_local_y == mcu_height_px.saturating_sub(1);
    if needs_next_row && last_row + 1 < mcu_rows {
        last_row + 2
    } else {
        last_row + 1
    }
}

pub(crate) fn fast_tile_region_first_decode_mcu(
    plan: &PreparedDecodePlan,
    roi: Rect,
    downscale: DownscaleFactor,
) -> u32 {
    let (width, _) = scaled_dimensions(plan.dimensions, downscale);
    let block_size = downscale.output_block_size();
    let mcu_width_px = block_size * u32::from(plan.sampling.max_h);
    let mcu_height_px = block_size * u32::from(plan.sampling.max_v);
    let mcus_per_row = width.div_ceil(mcu_width_px);
    fast420_first_decode_mcu_row(roi, mcu_height_px) * mcus_per_row
}

#[inline]
pub(super) fn mcu_row_intersects_rect(stripe_index: u32, mcu_height_px: u32, rect: Rect) -> bool {
    let y0 = stripe_index * mcu_height_px;
    let y1 = y0 + mcu_height_px;
    let rect_y1 = rect.y + rect.h;
    y0 < rect_y1 && y1 > rect.y
}

#[derive(Clone, Copy)]
pub(super) struct Fast420RegionLayout {
    pub(super) stripe_mcu_start: u32,
    pub(super) stripe_mcus_per_row: u32,
    pub(super) y_decode_start: usize,
    pub(super) y_decode_end: usize,
    pub(super) crop_start: usize,
    pub(super) crop_end: usize,
}

impl Fast420RegionLayout {
    pub(super) fn new(width: usize, roi: Rect) -> Self {
        Self::new_for_mcu_width(width, roi, 16)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "layout widths originate from validated u32 JPEG dimensions before allocation"
    )]
    pub(super) fn new_for_mcu_width(width: usize, roi: Rect, mcu_width_px: u32) -> Self {
        let crop_window = RgbCropWindow::new(width, roi);
        let stripe = StripeRegionLayout::new(
            width as u32,
            mcu_width_px,
            Rect {
                x: crop_window.scratch_x0 as u32,
                y: 0,
                w: (crop_window.scratch_x1 - crop_window.scratch_x0) as u32,
                h: 1,
            },
        );
        let y_decode_start = stripe.source_x0 as usize;
        let y_decode_end = y_decode_start + stripe.source_width as usize;
        let crop_start = roi.x as usize - y_decode_start;
        let crop_end = crop_start + roi.w as usize;

        Self {
            stripe_mcu_start: stripe.stripe_mcu_start,
            stripe_mcus_per_row: stripe.stripe_mcus_per_row,
            y_decode_start,
            y_decode_end,
            crop_start,
            crop_end,
        }
    }

    pub(super) fn row_width(self) -> usize {
        self.y_decode_end - self.y_decode_start
    }

    #[cfg(test)]
    pub(super) fn chroma_width(self) -> usize {
        self.row_width().div_ceil(2)
    }
}

pub(super) fn is_ycbcr_420(plan: &PreparedDecodePlan) -> bool {
    plan.color_space == ColorSpace::YCbCr
        && plan.sampling.max_h == 2
        && plan.sampling.max_v == 2
        && plan.sampling.components() == [(2, 2), (1, 1), (1, 1)]
}

pub(super) fn scaled_dimensions(dims: (u32, u32), downscale: DownscaleFactor) -> (u32, u32) {
    let denom = downscale.denominator();
    (dims.0.div_ceil(denom), dims.1.div_ceil(denom))
}

pub(super) fn expanded_output_rect(rect: Rect, width: u32, height: u32) -> Rect {
    let x = rect.x.saturating_sub(2);
    let y = rect.y.saturating_sub(2);
    let x_end = rect.x.saturating_add(rect.w).saturating_add(2).min(width);
    let y_end = rect.y.saturating_add(rect.h).saturating_add(2).min(height);
    Rect {
        x,
        y,
        w: x_end.saturating_sub(x),
        h: y_end.saturating_sub(y),
    }
}

#[derive(Clone, Copy)]
pub(super) struct ComponentBlockPosition {
    pub(super) mcu_x: u32,
    pub(super) mcu_y: u32,
    pub(super) block_x: u32,
    pub(super) block_y: u32,
}

pub(super) fn component_block_intersects_rect(
    plan: &PreparedDecodePlan,
    comp: &PreparedComponentPlan,
    downscale: DownscaleFactor,
    block: ComponentBlockPosition,
    rect: Rect,
) -> bool {
    let block_size = downscale.output_block_size();
    let h_ratio = u32::from(plan.sampling.max_h / comp.h);
    let v_ratio = u32::from(plan.sampling.max_v / comp.v);
    let x0 = block.mcu_x * u32::from(plan.sampling.max_h) * block_size
        + block.block_x * h_ratio * block_size;
    let y0 = block.mcu_y * u32::from(plan.sampling.max_v) * block_size
        + block.block_y * v_ratio * block_size;
    let w = h_ratio * block_size;
    let h = v_ratio * block_size;
    let x1 = x0 + w;
    let y1 = y0 + h;
    let rect_x1 = rect.x + rect.w;
    let rect_y1 = rect.y + rect.h;
    x0 < rect_x1 && x1 > rect.x && y0 < rect_y1 && y1 > rect.y
}
