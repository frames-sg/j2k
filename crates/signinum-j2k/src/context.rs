// SPDX-License-Identifier: Apache-2.0

use signinum_core::{CacheStats, CodecContext};
use signinum_j2k_native::CpuDecodeParallelism;

/// Reusable JPEG 2000 decode context and cache state.
#[derive(Debug, Default, Clone)]
pub struct J2kContext {
    hits: u64,
    misses: u64,
    cpu_decode_parallelism: CpuDecodeParallelism,
}

impl J2kContext {
    /// Create an empty JPEG 2000 context.
    pub const fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            cpu_decode_parallelism: CpuDecodeParallelism::Auto,
        }
    }

    pub(crate) fn record_tile_decode(&mut self) {
        self.misses = self.misses.saturating_add(1);
    }

    /// Return the CPU decode parallelism policy.
    pub fn cpu_decode_parallelism(&self) -> CpuDecodeParallelism {
        self.cpu_decode_parallelism
    }

    /// Set the CPU decode parallelism policy.
    pub fn set_cpu_decode_parallelism(&mut self, parallelism: CpuDecodeParallelism) {
        self.cpu_decode_parallelism = parallelism;
    }
}

impl CodecContext for J2kContext {
    fn clear(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.cpu_decode_parallelism = CpuDecodeParallelism::Auto;
    }

    fn cache_stats(&self) -> CacheStats {
        CacheStats::new(self.hits, self.misses)
    }
}
