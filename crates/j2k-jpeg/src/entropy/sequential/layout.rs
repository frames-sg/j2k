// SPDX-License-Identifier: MIT OR Apache-2.0

use super::restart::McuPosition;
use super::{PreparedComponentPlan, PreparedDecodePlan};
use crate::color::scaled_sampling::ScaledSampling;
use crate::info::{ColorSpace, DownscaleFactor, Rect};

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

    /// The window a fancy 4:2:0 region of `roi` decodes in an image `width`
    /// pixels wide: the MCU columns covering the chroma samples of `roi`
    /// plus one on either side, which the edge columns' upsampling reads.
    pub(super) fn for_fancy_420(width: u32, roi: Rect) -> Self {
        let chroma_width = width.div_ceil(2);
        let sample_start = (roi.x / 2).saturating_sub(1);
        let sample_end = ((roi.x + roi.w).div_ceil(2) + 1).min(chroma_width);
        let x0 = sample_start * 2;
        let x1 = (sample_end * 2).min(width);
        Self::new(
            width,
            16,
            Rect {
                x: x0,
                y: 0,
                w: x1 - x0,
                h: 1,
            },
        )
    }

    pub(super) fn source_width_usize(self) -> usize {
        self.source_width as usize
    }
}

/// The MCU grid of a scan and the MCU columns a decode deposits into its
/// stripe; MCUs in the other columns are entropy-skipped.
#[derive(Clone, Copy)]
pub(super) struct McuColumns {
    pub(super) mcus_per_row: u32,
    pub(super) mcu_rows: u32,
    pub(super) stripe_mcu_start: u32,
    pub(super) stripe_mcus_per_row: u32,
}

impl McuColumns {
    /// The columns of `region` in a scan of `(mcus_per_row, mcu_rows)` MCUs.
    pub(super) const fn new(
        (mcus_per_row, mcu_rows): (u32, u32),
        region: StripeRegionLayout,
    ) -> Self {
        Self {
            mcus_per_row,
            mcu_rows,
            stripe_mcu_start: region.stripe_mcu_start,
            stripe_mcus_per_row: region.stripe_mcus_per_row,
        }
    }

    #[inline]
    pub(super) const fn contains(self, mcu_x: u32) -> bool {
        mcu_x >= self.stripe_mcu_start && mcu_x < self.stripe_mcu_start + self.stripe_mcus_per_row
    }

    /// Stripe column of MCU column `mcu_x`, which [`Self::contains`].
    #[inline]
    pub(super) const fn local(self, mcu_x: u32) -> u32 {
        mcu_x - self.stripe_mcu_start
    }

    /// Scan position of the MCU at column `mcu_x` of row `mcu_y`.
    #[inline]
    pub(super) const fn mcu_position(self, mcu_x: u32, mcu_y: u32) -> McuPosition {
        McuPosition {
            current: mcu_y * self.mcus_per_row + mcu_x,
            total: self.mcu_rows * self.mcus_per_row,
        }
    }

    /// Whether every MCU column is decoded.
    pub(super) const fn is_whole_row(self) -> bool {
        self.stripe_mcu_start == 0 && self.stripe_mcus_per_row == self.mcus_per_row
    }
}

/// The columns of a decoded window that reach the writer.
#[derive(Clone, Copy)]
pub(super) struct CropWindow {
    /// Visible luma width of the decoded window.
    pub(super) row_width: usize,
    /// Window column of the first output column.
    pub(super) crop_start: usize,
    /// Output columns.
    pub(super) crop_width: usize,
}

impl CropWindow {
    /// The columns of `roi` in the decoded `region`.
    pub(super) const fn new(region: StripeRegionLayout, roi: Rect) -> Self {
        Self {
            row_width: region.source_width as usize,
            crop_start: (roi.x - region.source_x0) as usize,
            crop_width: roi.w as usize,
        }
    }

    pub(super) const fn columns(self) -> core::ops::Range<usize> {
        self.crop_start..self.crop_start + self.crop_width
    }

    /// Whether the writer takes the decoded window's whole rows.
    pub(super) const fn is_full_width(self) -> bool {
        self.crop_start == 0 && self.crop_width == self.row_width
    }

    /// Visible 2:1 chroma width of the decoded window.
    pub(super) const fn chroma_width(self) -> usize {
        self.row_width.div_ceil(2)
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

/// End of the MCU rows a fancy 4:2:0 region decode reads when an MCU row's
/// last `trailing_rows` output rows are emitted with the next MCU row (one
/// for stripe-at-a-time emission, which needs the next chroma row only for
/// the very last row).
#[inline]
pub(super) fn fast420_decode_mcu_row_end(
    roi: Rect,
    mcu_height_px: u32,
    mcu_rows: u32,
    trailing_rows: u32,
) -> u32 {
    let last_row = last_mcu_row_for_rect(roi, mcu_height_px, mcu_rows);
    let last_local_y = (roi.y + roi.h - 1) % mcu_height_px;
    let needs_next_row = last_local_y >= mcu_height_px.saturating_sub(trailing_rows);
    if needs_next_row && last_row + 1 < mcu_rows {
        last_row + 2
    } else {
        last_row + 1
    }
}

#[inline]
pub(super) fn mcu_row_intersects_rect(stripe_index: u32, mcu_height_px: u32, rect: Rect) -> bool {
    let y0 = stripe_index * mcu_height_px;
    let y1 = y0 + mcu_height_px;
    let rect_y1 = rect.y + rect.h;
    y0 < rect_y1 && y1 > rect.y
}

/// Whether the dedicated 4:2:0 smoothing emitters produce this decode: coded
/// 4:2:0 YCbCr at a scale where libjpeg-turbo still upsamples its chroma with
/// `h2v2_fancy_upsample`.
pub(super) fn uses_fancy_420_emit(plan: &PreparedDecodePlan, scaled: &ScaledSampling) -> bool {
    is_ycbcr_420(plan) && scaled.is_fancy_420()
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
