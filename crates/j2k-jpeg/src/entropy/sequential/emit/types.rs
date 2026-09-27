// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{Fast420RegionLayout, PreparedDecodePlan, StripeBuffer};
use crate::{
    color::scaled_sampling::ScaledSampling,
    error::JpegError,
    info::{ColorSpace, DownscaleFactor, Rect},
    internal::scratch::SinkRows,
};

pub(in crate::entropy::sequential) struct Fast420RegionStripe<'a> {
    pub(in crate::entropy::sequential) neighbors: StripeNeighbors<'a>,
    pub(in crate::entropy::sequential) stripe_index: u32,
    pub(in crate::entropy::sequential) roi: Rect,
    pub(in crate::entropy::sequential) region_layout: Fast420RegionLayout,
    pub(in crate::entropy::sequential) crop_rows: &'a mut SinkRows,
    pub(in crate::entropy::sequential) downscale: DownscaleFactor,
}

#[derive(Clone, Copy)]
pub(in crate::entropy::sequential) struct StripeEmit<'a> {
    pub(in crate::entropy::sequential) prev: Option<&'a StripeBuffer>,
    pub(in crate::entropy::sequential) curr: &'a StripeBuffer,
    pub(in crate::entropy::sequential) next: Option<&'a StripeBuffer>,
    pub(in crate::entropy::sequential) stripe_index: u32,
    pub(in crate::entropy::sequential) source_width: usize,
    pub(in crate::entropy::sequential) downscale: DownscaleFactor,
    pub(in crate::entropy::sequential) scaled: &'a ScaledSampling,
}

#[derive(Clone, Copy)]
pub(in crate::entropy::sequential) struct StripeNeighbors<'a> {
    pub(in crate::entropy::sequential) prev: Option<&'a StripeBuffer>,
    pub(in crate::entropy::sequential) curr: &'a StripeBuffer,
    pub(in crate::entropy::sequential) next: Option<&'a StripeBuffer>,
}

/// Rejects a plan with fewer components than its color space's emitters read
/// (three for YCbCr and RGB, four for CMYK and YCCK).
pub(in crate::entropy::sequential) fn ensure_color_components(
    plan: &PreparedDecodePlan,
) -> Result<(), JpegError> {
    let needed = match plan.color_space {
        ColorSpace::Grayscale => 1,
        ColorSpace::YCbCr | ColorSpace::Rgb => 3,
        ColorSpace::Cmyk | ColorSpace::Ycck => 4,
    };
    if plan.sampling.len() < needed {
        return Err(JpegError::UnsupportedComponentCount {
            count: u8::try_from(plan.sampling.len()).unwrap_or(u8::MAX),
        });
    }
    Ok(())
}
