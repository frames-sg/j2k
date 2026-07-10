// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{Fast420RegionLayout, StripeBuffer};
use crate::{
    info::{DownscaleFactor, Rect},
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
}

#[derive(Clone, Copy)]
pub(in crate::entropy::sequential) struct StripeNeighbors<'a> {
    pub(in crate::entropy::sequential) prev: Option<&'a StripeBuffer>,
    pub(in crate::entropy::sequential) curr: &'a StripeBuffer,
    pub(in crate::entropy::sequential) next: Option<&'a StripeBuffer>,
}
