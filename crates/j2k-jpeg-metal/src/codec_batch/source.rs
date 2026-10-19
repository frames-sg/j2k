// SPDX-License-Identifier: MIT OR Apache-2.0

//! Normalized raw/prepared RGB8 batch source values.

use crate::batch;

pub(super) struct ResolvedRgb8BatchSource {
    pub(super) request: batch::QueuedRequest,
    pub(super) output_dimensions: (u32, u32),
    pub(super) cache_retained_bytes: usize,
}
