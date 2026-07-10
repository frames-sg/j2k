// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::build::CodeBlock;
use crate::profile;

#[derive(Default)]
pub(crate) struct HtBlockDecodeContext {
    pub(super) coefficients: Vec<u32>,
    pub(super) scratch: HtBlockDecodeScratch,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl HtBlockDecodeContext {
    pub(super) fn reset(&mut self, code_block: &CodeBlock) {
        self.width = code_block.rect.width();
        self.height = code_block.rect.height();
        self.coefficients.clear();
        self.coefficients
            .resize((self.width * self.height) as usize, 0);
    }

    pub(crate) fn coefficient_rows(&self) -> impl Iterator<Item = &[u32]> {
        self.coefficients.chunks_exact(self.width as usize)
    }
}

#[derive(Default)]
pub(crate) struct HtBlockDecodeScratch {
    pub(super) cleanup: Vec<u16>,
    pub(super) v_n: Vec<u32>,
    pub(super) sigma: Vec<u16>,
    pub(super) prev_row_sig: Vec<u16>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HtBlockDecodeStats {
    pub(crate) blocks: u128,
    pub(crate) refinement_blocks: u128,
    pub(crate) cleanup_bytes: u128,
    pub(crate) refinement_bytes: u128,
    pub(crate) ht_cleanup_us: u128,
    pub(crate) ht_mag_sgn_us: u128,
    pub(crate) ht_sigma_us: u128,
    pub(crate) ht_sigprop_us: u128,
    pub(crate) ht_magref_us: u128,
}

impl HtBlockDecodeStats {
    fn record_block(&mut self, cleanup_bytes: usize, refinement_bytes: usize) {
        self.blocks += 1;
        self.cleanup_bytes += cleanup_bytes as u128;
        if refinement_bytes > 0 {
            self.refinement_blocks += 1;
            self.refinement_bytes += refinement_bytes as u128;
        }
    }
}

pub(super) trait HtDecodeObserver {
    #[inline(always)]
    fn record_block(&mut self, _cleanup_bytes: usize, _refinement_bytes: usize) {}

    #[expect(
        clippy::inline_always,
        reason = "erase the unprofiled observer clock hook"
    )]
    #[inline(always)]
    fn phase_start(&self) -> Option<profile::ProfileInstant> {
        None
    }

    #[inline(always)]
    fn add_cleanup_us(&mut self, _start: Option<profile::ProfileInstant>) {}

    #[inline(always)]
    fn add_mag_sgn_us(&mut self, _start: Option<profile::ProfileInstant>) {}

    #[inline(always)]
    fn add_sigma_us(&mut self, _start: Option<profile::ProfileInstant>) {}

    #[inline(always)]
    fn add_sigprop_us(&mut self, _start: Option<profile::ProfileInstant>) {}

    #[inline(always)]
    fn add_magref_us(&mut self, _start: Option<profile::ProfileInstant>) {}
}

pub(super) struct NoHtDecodeStats;

impl HtDecodeObserver for NoHtDecodeStats {}

pub(super) struct RecordingHtDecodeStats<'a> {
    pub(super) stats: &'a mut HtBlockDecodeStats,
    pub(super) profile_enabled: bool,
}

impl HtDecodeObserver for RecordingHtDecodeStats<'_> {
    #[expect(clippy::inline_always, reason = "fuse observer accounting into decode")]
    #[inline(always)]
    fn record_block(&mut self, cleanup_bytes: usize, refinement_bytes: usize) {
        self.stats.record_block(cleanup_bytes, refinement_bytes);
    }

    #[expect(clippy::inline_always, reason = "fuse observer timing into decode")]
    #[inline(always)]
    fn phase_start(&self) -> Option<profile::ProfileInstant> {
        if self.profile_enabled {
            profile::profile_now(true)
        } else {
            None
        }
    }

    #[expect(clippy::inline_always, reason = "fuse observer timing into decode")]
    #[inline(always)]
    fn add_cleanup_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.ht_cleanup_us += profile::elapsed_us(start);
        }
    }

    #[expect(clippy::inline_always, reason = "fuse observer timing into decode")]
    #[inline(always)]
    fn add_mag_sgn_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.ht_mag_sgn_us += profile::elapsed_us(start);
        }
    }

    #[expect(clippy::inline_always, reason = "fuse observer timing into decode")]
    #[inline(always)]
    fn add_sigma_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.ht_sigma_us += profile::elapsed_us(start);
        }
    }

    #[expect(clippy::inline_always, reason = "fuse observer timing into decode")]
    #[inline(always)]
    fn add_sigprop_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.ht_sigprop_us += profile::elapsed_us(start);
        }
    }

    #[expect(clippy::inline_always, reason = "fuse observer timing into decode")]
    #[inline(always)]
    fn add_magref_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.ht_magref_us += profile::elapsed_us(start);
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HtBlockDecodeScratchCapacities {
    pub(super) cleanup: usize,
    pub(super) v_n: usize,
    pub(super) sigma: usize,
    pub(super) prev_row_sig: usize,
}

#[cfg(test)]
impl HtBlockDecodeScratch {
    pub(super) fn capacities_for_test(&self) -> HtBlockDecodeScratchCapacities {
        HtBlockDecodeScratchCapacities {
            cleanup: self.cleanup.capacity(),
            v_n: self.v_n.capacity(),
            sigma: self.sigma.capacity(),
            prev_row_sig: self.prev_row_sig.capacity(),
        }
    }

    pub(super) fn poison_for_test(&mut self) {
        self.cleanup.fill(u16::MAX);
        self.v_n.fill(u32::MAX);
        self.sigma.fill(u16::MAX);
        self.prev_row_sig.fill(u16::MAX);
    }
}

#[expect(
    clippy::inline_always,
    reason = "fuse scratch clearing into the phase loop"
)]
#[inline(always)]
pub(super) fn zeroed_u16_scratch(buffer: &mut Vec<u16>, len: usize) -> &mut [u16] {
    if buffer.len() < len {
        buffer.resize(len, 0);
    }
    buffer[..len].fill(0);

    &mut buffer[..len]
}

#[cfg(test)]
pub(super) fn zeroed_u32_scratch(buffer: &mut Vec<u32>, len: usize) -> &mut [u32] {
    if buffer.len() < len {
        buffer.resize(len, 0);
    }
    buffer[..len].fill(0);

    &mut buffer[..len]
}

#[expect(
    clippy::inline_always,
    reason = "fuse scratch resizing into the phase loop"
)]
#[inline(always)]
pub(super) fn resized_u16_scratch(buffer: &mut Vec<u16>, len: usize) -> &mut [u16] {
    if buffer.len() < len {
        buffer.resize(len, 0);
    }

    &mut buffer[..len]
}

#[expect(
    clippy::inline_always,
    reason = "fuse scratch resizing into the phase loop"
)]
#[inline(always)]
pub(super) fn resized_u32_scratch(buffer: &mut Vec<u32>, len: usize) -> &mut [u32] {
    if buffer.len() < len {
        buffer.resize(len, 0);
    }

    &mut buffer[..len]
}
