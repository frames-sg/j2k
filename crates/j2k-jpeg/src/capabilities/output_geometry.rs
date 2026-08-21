// SPDX-License-Identifier: MIT OR Apache-2.0

//! Output geometry derived from normalized decode operations.

use super::JpegDecodeOp;
use crate::{Info, Rect};
use j2k_core::Downscale;

pub(super) fn output_rect_for_request(info: &Info, op: JpegDecodeOp) -> Rect {
    match op {
        JpegDecodeOp::Full => Rect::full(info.dimensions),
        JpegDecodeOp::Region(roi) => roi,
        JpegDecodeOp::Scaled(scale) => scaled_rect(Rect::full(info.dimensions), scale),
        JpegDecodeOp::RegionScaled { roi, scale } => scaled_rect(roi, scale),
    }
}

fn scaled_rect(rect: Rect, scale: Downscale) -> Rect {
    let denom = scale.denominator();
    let x_end = rect.x.saturating_add(rect.w);
    let y_end = rect.y.saturating_add(rect.h);
    let x0 = rect.x / denom;
    let y0 = rect.y / denom;
    let x1 = x_end.div_ceil(denom);
    let y1 = y_end.div_ceil(denom);
    Rect {
        x: x0,
        y: y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
    }
}
