// SPDX-License-Identifier: MIT OR Apache-2.0

mod admission;

use alloc::vec::Vec;
use core::mem::size_of;
use std::hash::BuildHasher;

use super::diagnostics::JpegPlanCacheDiagnostics;
use super::JpegPlanCacheInsert;
use crate::adapter::fast_packet::{JpegCachedPlan, JpegPlanCacheError};

struct JpegPlanCacheEntry {
    digest: u64,
    plan: JpegCachedPlan,
    retained_bytes: usize,
    last_used: u64,
}

pub(in crate::adapter::fast_packet::cache) struct PlanCache<S> {
    entries: Vec<JpegPlanCacheEntry>,
    digest_builder: S,
    entry_limit: usize,
    host_byte_limit: usize,
    entry_bytes: usize,
    access_clock: u64,
    diagnostics: JpegPlanCacheDiagnostics,
}

impl<S: BuildHasher> PlanCache<S> {
    pub(in crate::adapter::fast_packet::cache) fn with_limits_and_digest_builder(
        entry_limit: usize,
        host_byte_limit: usize,
        digest_builder: S,
    ) -> Self {
        Self {
            entries: Vec::new(),
            digest_builder,
            entry_limit,
            host_byte_limit,
            entry_bytes: 0,
            access_clock: 0,
            diagnostics: JpegPlanCacheDiagnostics::new(entry_limit, host_byte_limit),
        }
    }

    pub(in crate::adapter::fast_packet::cache) fn get(
        &mut self,
        input: &[u8],
    ) -> Option<JpegCachedPlan> {
        let digest = self.digest_builder.hash_one(input);
        let Some(index) = self.find_index(digest, input) else {
            self.diagnostics.misses = self.diagnostics.misses.saturating_add(1);
            return None;
        };
        let stamp = self.next_access_stamp();
        self.entries[index].last_used = stamp;
        self.diagnostics.hits = self.diagnostics.hits.saturating_add(1);
        Some(self.entries[index].plan.clone())
    }

    pub(in crate::adapter::fast_packet::cache) fn prepare_for_miss(
        &mut self,
        external_live_bytes: usize,
        operation_cap: usize,
    ) -> Result<(), JpegPlanCacheError> {
        if self.entry_limit != 0 {
            let _admission_metadata_fits = self
                .ensure_metadata_capacity_with_external_live(external_live_bytes, operation_cap)?;
        }
        Ok(())
    }

    pub(in crate::adapter::fast_packet::cache) fn insert(
        &mut self,
        plan: JpegCachedPlan,
    ) -> Result<JpegPlanCacheInsert, JpegPlanCacheError> {
        if self.entry_limit == 0 {
            self.diagnostics.disabled_rejections =
                self.diagnostics.disabled_rejections.saturating_add(1);
            return Ok(JpegPlanCacheInsert::SkippedDisabled);
        }

        let retained_bytes = plan.retained_cache_bytes()?;
        let digest = self.digest_builder.hash_one(plan.input().as_bytes());
        if let Some(index) = self.find_index(digest, plan.input().as_bytes()) {
            return self.replace(index, plan, retained_bytes);
        }

        let requested_metadata = self.requested_metadata_bytes()?;
        if !fits_within(requested_metadata, retained_bytes, self.host_byte_limit) {
            return Ok(self.reject_oversized());
        }
        if !self.ensure_metadata_capacity()? {
            return Ok(self.reject_oversized());
        }
        if !fits_within(
            self.diagnostics.metadata_capacity_bytes,
            retained_bytes,
            self.host_byte_limit,
        ) {
            return Ok(self.reject_oversized());
        }

        self.evict_until_fits(retained_bytes, None)?;
        let stamp = self.next_access_stamp();
        self.entries.push(JpegPlanCacheEntry {
            digest,
            plan,
            retained_bytes,
            last_used: stamp,
        });
        self.entry_bytes =
            self.entry_bytes
                .checked_add(retained_bytes)
                .ok_or(JpegPlanCacheError::Invariant(
                    "JPEG plan cache retained entry bytes overflow",
                ))?;
        self.refresh_current()?;
        Ok(JpegPlanCacheInsert::Cached)
    }

    pub(in crate::adapter::fast_packet::cache) const fn diagnostics(
        &self,
    ) -> JpegPlanCacheDiagnostics {
        self.diagnostics
    }

    fn find_index(&self, digest: u64, input: &[u8]) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| entry.digest == digest && entry.plan.input().as_bytes() == input)
    }

    fn requested_metadata_bytes(&self) -> Result<usize, JpegPlanCacheError> {
        metadata_bytes(self.entry_limit)
    }

    fn refresh_current(&mut self) -> Result<(), JpegPlanCacheError> {
        let retained_bytes = self
            .diagnostics
            .metadata_capacity_bytes
            .checked_add(self.entry_bytes)
            .ok_or(JpegPlanCacheError::Invariant(
                "JPEG plan cache total retained bytes overflow",
            ))?;
        self.diagnostics.entries = self.entries.len();
        self.diagnostics.retained_bytes = retained_bytes;
        self.diagnostics.peak_entries = self.diagnostics.peak_entries.max(self.entries.len());
        self.diagnostics.peak_bytes = self.diagnostics.peak_bytes.max(retained_bytes);
        Ok(())
    }

    fn reject_oversized(&mut self) -> JpegPlanCacheInsert {
        self.diagnostics.oversized_rejections =
            self.diagnostics.oversized_rejections.saturating_add(1);
        JpegPlanCacheInsert::SkippedOversized
    }

    fn next_access_stamp(&mut self) -> u64 {
        if self.access_clock == u64::MAX {
            self.entries.sort_unstable_by_key(|entry| entry.last_used);
            for (index, entry) in self.entries.iter_mut().enumerate() {
                entry.last_used = index as u64;
            }
            self.access_clock = self.entries.len() as u64;
        }
        self.access_clock += 1;
        self.access_clock
    }
}

fn fits_within(base: usize, extra: usize, limit: usize) -> bool {
    base.checked_add(extra).is_some_and(|bytes| bytes <= limit)
}

fn metadata_bytes(capacity: usize) -> Result<usize, JpegPlanCacheError> {
    capacity
        .checked_mul(size_of::<JpegPlanCacheEntry>())
        .ok_or(JpegPlanCacheError::Invariant(
            "JPEG plan cache metadata byte count overflow",
        ))
}

#[cfg(test)]
pub(in crate::adapter::fast_packet::cache) const fn metadata_entry_size_for_test() -> usize {
    size_of::<JpegPlanCacheEntry>()
}
