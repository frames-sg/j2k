// SPDX-License-Identifier: Apache-2.0

use slidecodec_core::ScratchPool;

/// Caller-owned scratch for `slidecodec-j2k`.
///
/// The M1 backend does not expose reusable internal buffers through
/// `slidecodec-core` yet, so this pool is currently a thin no-op carrier that
/// keeps the trait surface stable for later milestones.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct J2kScratchPool;

impl J2kScratchPool {
    pub const fn new() -> Self {
        Self
    }
}

impl ScratchPool for J2kScratchPool {
    fn bytes_allocated(&self) -> usize {
        0
    }

    fn reset(&mut self) {}
}
