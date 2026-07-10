// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::profile;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct J2kBlockDecodeStats {
    pub(crate) sigprop_us: u128,
    pub(crate) magref_us: u128,
    pub(crate) cleanup_us: u128,
    pub(crate) bypass_us: u128,
}

pub(super) trait J2kDecodeObserver {
    #[inline(always)]
    fn phase_start(&self) -> Option<profile::ProfileInstant> {
        None
    }

    #[inline(always)]
    fn add_sigprop_us(&mut self, _start: Option<profile::ProfileInstant>) {}

    #[inline(always)]
    fn add_magref_us(&mut self, _start: Option<profile::ProfileInstant>) {}

    #[inline(always)]
    fn add_cleanup_us(&mut self, _start: Option<profile::ProfileInstant>) {}

    #[inline(always)]
    fn add_bypass_us(&mut self, _start: Option<profile::ProfileInstant>) {}
}

pub(super) struct NoJ2kDecodeStats;

impl J2kDecodeObserver for NoJ2kDecodeStats {}

pub(super) struct RecordingJ2kDecodeStats<'a> {
    pub(super) stats: &'a mut J2kBlockDecodeStats,
    pub(super) profile_enabled: bool,
}

impl J2kDecodeObserver for RecordingJ2kDecodeStats<'_> {
    #[inline(always)]
    fn phase_start(&self) -> Option<profile::ProfileInstant> {
        if self.profile_enabled {
            profile::profile_now(true)
        } else {
            None
        }
    }

    #[inline(always)]
    fn add_sigprop_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.sigprop_us += profile::elapsed_us(start);
        }
    }

    #[inline(always)]
    fn add_magref_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.magref_us += profile::elapsed_us(start);
        }
    }

    #[inline(always)]
    fn add_cleanup_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.cleanup_us += profile::elapsed_us(start);
        }
    }

    #[inline(always)]
    fn add_bypass_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.bypass_us += profile::elapsed_us(start);
        }
    }
}
