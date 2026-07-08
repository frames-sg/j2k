// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "bench-internals")]
use crate::bench_support::{BenchBlockActivityCounts, BenchFast420Profile};
use crate::entropy::block::BlockActivity;
#[cfg(feature = "bench-internals")]
use std::time::Instant;

pub(super) trait Fast420Profiler {
    fn record_activity(&mut self, activity: BlockActivity);
}

#[derive(Default)]
pub(super) struct NoopFast420Profiler;

impl Fast420Profiler for NoopFast420Profiler {
    #[inline]
    fn record_activity(&mut self, _activity: BlockActivity) {}
}

#[cfg(feature = "bench-internals")]
impl Fast420Profiler for BenchBlockActivityCounts {
    #[inline]
    fn record_activity(&mut self, activity: BlockActivity) {
        match activity {
            BlockActivity::DcOnly => self.record_dc_only(),
            BlockActivity::BottomHalfZero => self.record_bottom_half_zero(),
            BlockActivity::General => self.record_general(),
        }
    }
}

#[cfg(feature = "bench-internals")]
pub(super) enum Fast420StageTimer {
    Active(Instant),
    Disabled,
}

#[cfg(not(feature = "bench-internals"))]
pub(super) struct Fast420StageTimer;

impl Fast420StageTimer {
    #[inline]
    fn disabled() -> Self {
        #[cfg(feature = "bench-internals")]
        {
            Self::Disabled
        }
        #[cfg(not(feature = "bench-internals"))]
        {
            Self
        }
    }

    #[cfg(feature = "bench-internals")]
    #[inline]
    fn active() -> Self {
        Self::Active(Instant::now())
    }

    #[cfg(feature = "bench-internals")]
    #[inline]
    fn elapsed_ns(self) -> u128 {
        match self {
            Self::Active(start) => start.elapsed().as_nanos(),
            Self::Disabled => 0,
        }
    }
}

pub(super) trait Fast420ScanProfiler {
    type ActivityProfiler: Fast420Profiler;

    fn activity_profiler(&mut self) -> &mut Self::ActivityProfiler;

    #[inline]
    fn begin_mcu_decode(&mut self) -> Fast420StageTimer {
        Fast420StageTimer::disabled()
    }

    #[inline]
    fn finish_mcu_decode(&mut self, _timer: Fast420StageTimer) {}

    #[inline]
    fn begin_rgb_emit(&mut self) -> Fast420StageTimer {
        Fast420StageTimer::disabled()
    }

    #[inline]
    fn finish_rgb_emit(&mut self, _timer: Fast420StageTimer) {}

    #[inline]
    fn begin_finish_scan(&mut self) -> Fast420StageTimer {
        Fast420StageTimer::disabled()
    }

    #[inline]
    fn finish_finish_scan(&mut self, _timer: Fast420StageTimer) {}
}

#[derive(Default)]
pub(super) struct NoopFast420ScanProfile {
    activity: NoopFast420Profiler,
}

impl Fast420ScanProfiler for NoopFast420ScanProfile {
    type ActivityProfiler = NoopFast420Profiler;

    #[inline]
    fn activity_profiler(&mut self) -> &mut Self::ActivityProfiler {
        &mut self.activity
    }
}

#[cfg(feature = "bench-internals")]
impl Fast420ScanProfiler for BenchFast420Profile {
    type ActivityProfiler = BenchBlockActivityCounts;

    #[inline]
    fn activity_profiler(&mut self) -> &mut Self::ActivityProfiler {
        self.block_activity_counts_mut()
    }

    #[inline]
    fn begin_mcu_decode(&mut self) -> Fast420StageTimer {
        Fast420StageTimer::active()
    }

    #[inline]
    fn finish_mcu_decode(&mut self, timer: Fast420StageTimer) {
        self.add_mcu_decode_ns(timer.elapsed_ns());
    }

    #[inline]
    fn begin_rgb_emit(&mut self) -> Fast420StageTimer {
        Fast420StageTimer::active()
    }

    #[inline]
    fn finish_rgb_emit(&mut self, timer: Fast420StageTimer) {
        self.add_rgb_emit_ns(timer.elapsed_ns());
    }

    #[inline]
    fn begin_finish_scan(&mut self) -> Fast420StageTimer {
        Fast420StageTimer::active()
    }

    #[inline]
    fn finish_finish_scan(&mut self, timer: Fast420StageTimer) {
        self.add_finish_ns(timer.elapsed_ns());
    }
}
