// SPDX-License-Identifier: Apache-2.0

use slidecodec_core::{CacheStats, CodecContext};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct J2kContext;

impl J2kContext {
    pub const fn new() -> Self {
        Self
    }
}

impl CodecContext for J2kContext {
    fn clear(&mut self) {}

    fn cache_stats(&self) -> CacheStats {
        CacheStats::default()
    }
}
