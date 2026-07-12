// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public façade for the bounded backend-neutral JPEG plan cache.

use std::collections::hash_map::RandomState;

use super::{JpegCachedPlan, JpegPlanCacheError};

mod diagnostics;
mod resolve;
mod state;

pub use diagnostics::JpegPlanCacheDiagnostics;
#[cfg(test)]
pub(super) use state::metadata_entry_size_for_test;
pub(super) use state::PlanCache;

/// Default number of complete JPEG inputs retained by an accelerator cache.
pub const DEFAULT_JPEG_PLAN_CACHE_ENTRIES: usize = 8;
/// Default retained host-memory limit for an accelerator cache.
pub const DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
/// Result of optional JPEG plan-cache admission.
pub enum JpegPlanCacheInsert {
    /// The plan was inserted or replaced an entry with the same full input.
    Cached,
    /// Admission was skipped because the entry limit is zero.
    SkippedDisabled,
    /// Admission was skipped because the plan cannot fit by itself.
    SkippedOversized,
}

#[doc(hidden)]
/// Randomized-digest, full-input-validated, byte-bounded flat LRU cache.
pub struct JpegPlanCache {
    inner: PlanCache<RandomState>,
}

impl JpegPlanCache {
    /// Construct a cache with the production entry and host-memory limits.
    #[must_use]
    pub fn new() -> Self {
        Self::with_limits(
            DEFAULT_JPEG_PLAN_CACHE_ENTRIES,
            DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES,
        )
    }

    /// Construct a cache with explicit entry and retained-host-memory limits.
    #[must_use]
    pub fn with_limits(entry_limit: usize, host_byte_limit: usize) -> Self {
        Self {
            inner: PlanCache::with_limits_and_digest_builder(
                entry_limit,
                host_byte_limit,
                RandomState::new(),
            ),
        }
    }

    /// Find a plan only after randomized digest and full-input equality checks.
    ///
    /// The returned plan clone only clones small shared handles; it does not
    /// copy the input or any packet vector.
    pub fn get(&mut self, input: &[u8]) -> Option<JpegCachedPlan> {
        self.inner.get(input)
    }

    /// Admit one fully built plan, evicting deterministic least-recent entries.
    ///
    /// Disabled and individually oversized plans are ordinary non-errors and
    /// do not evict or replace any existing entry.
    ///
    /// # Errors
    ///
    /// Returns a typed metadata-allocation or retained-byte invariant failure.
    pub fn insert(
        &mut self,
        plan: JpegCachedPlan,
    ) -> Result<JpegPlanCacheInsert, JpegPlanCacheError> {
        self.inner.insert(plan)
    }

    /// Current and high-water cache retention and admission diagnostics.
    #[must_use]
    pub const fn diagnostics(&self) -> JpegPlanCacheDiagnostics {
        self.inner.diagnostics()
    }
}

impl Default for JpegPlanCache {
    fn default() -> Self {
        Self::new()
    }
}
