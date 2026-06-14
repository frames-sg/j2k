// SPDX-License-Identifier: Apache-2.0

use signinum_core::{CacheStats, CodecContext};

use crate::CpuDecodeParallelism;

/// Reusable JPEG 2000 decode context and cache state.
#[derive(Debug, Default, Clone)]
pub struct J2kContext {
    cpu_decode_parallelism: CpuDecodeParallelism,
}

impl J2kContext {
    /// Create an empty JPEG 2000 context.
    pub const fn new() -> Self {
        Self {
            cpu_decode_parallelism: CpuDecodeParallelism::Auto,
        }
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
        self.cpu_decode_parallelism = CpuDecodeParallelism::Auto;
    }

    fn cache_stats(&self) -> CacheStats {
        CacheStats::default()
    }
}
