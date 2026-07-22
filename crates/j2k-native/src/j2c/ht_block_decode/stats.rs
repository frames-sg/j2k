// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::profile;

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
