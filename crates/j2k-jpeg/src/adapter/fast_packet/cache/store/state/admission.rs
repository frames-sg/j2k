// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible metadata reservation, replacement, and deterministic eviction.

use alloc::vec::Vec;
use std::hash::BuildHasher;

use super::{fits_within, metadata_bytes, PlanCache};
use crate::adapter::fast_packet::cache::shared_allocation::checked_live_bytes;
use crate::adapter::fast_packet::JpegPlanCacheInsert;
use crate::adapter::fast_packet::{JpegCachedPlan, JpegPlanCacheError};

impl<S: BuildHasher> PlanCache<S> {
    pub(super) fn replace(
        &mut self,
        mut index: usize,
        plan: JpegCachedPlan,
        retained_bytes: usize,
    ) -> Result<JpegPlanCacheInsert, JpegPlanCacheError> {
        if !fits_within(
            self.diagnostics.metadata_capacity_bytes,
            retained_bytes,
            self.host_byte_limit,
        ) {
            return Ok(self.reject_oversized());
        }
        while !self.replacement_fits(index, retained_bytes)? {
            let evicted = self
                .oldest_index(Some(index))
                .ok_or(JpegPlanCacheError::Invariant(
                    "JPEG plan replacement that fits alone has no evictable entry",
                ))?;
            self.evict_index(evicted)?;
            if evicted < index {
                index -= 1;
            }
        }
        let old_bytes = self.entries[index].retained_bytes;
        self.entry_bytes = self
            .entry_bytes
            .checked_sub(old_bytes)
            .and_then(|bytes| bytes.checked_add(retained_bytes))
            .ok_or(JpegPlanCacheError::Invariant(
                "JPEG plan cache replacement byte accounting overflow",
            ))?;
        self.entries[index].plan = plan;
        self.entries[index].retained_bytes = retained_bytes;
        self.entries[index].last_used = self.next_access_stamp();
        self.refresh_current()?;
        Ok(JpegPlanCacheInsert::Cached)
    }

    pub(super) fn ensure_metadata_capacity(&mut self) -> Result<bool, JpegPlanCacheError> {
        self.ensure_metadata_capacity_with_external_live(0, usize::MAX)
    }

    pub(super) fn ensure_metadata_capacity_with_external_live(
        &mut self,
        external_live_bytes: usize,
        operation_cap: usize,
    ) -> Result<bool, JpegPlanCacheError> {
        let current_live_bytes = checked_live_bytes(
            "JPEG plan cache and external owners before metadata reserve",
            self.diagnostics.retained_bytes,
            external_live_bytes,
            operation_cap,
        )?;
        if self.entries.capacity() >= self.entry_limit {
            return Ok(true);
        }
        let requested_bytes = self.requested_metadata_bytes()?;
        if requested_bytes > self.host_byte_limit {
            return Ok(false);
        }
        checked_live_bytes(
            "JPEG plan cache metadata reserve owner graph",
            current_live_bytes,
            requested_bytes,
            operation_cap,
        )?;
        if !self.entries.is_empty() {
            return Err(JpegPlanCacheError::Invariant(
                "JPEG plan cache metadata capacity changed after insertion",
            ));
        }
        let mut entries = Vec::new();
        if let Err(source) = entries.try_reserve_exact(self.entry_limit) {
            self.diagnostics.metadata_allocation_failures = self
                .diagnostics
                .metadata_allocation_failures
                .saturating_add(1);
            return Err(JpegPlanCacheError::allocation(
                "JPEG plan cache entry metadata",
                requested_bytes,
                source,
            ));
        }
        let actual_bytes = metadata_bytes(entries.capacity())?;
        if actual_bytes > self.host_byte_limit {
            return Ok(false);
        }
        checked_live_bytes(
            "JPEG plan cache allocated metadata owner graph",
            current_live_bytes,
            actual_bytes,
            operation_cap,
        )?;
        self.entries = entries;
        self.diagnostics.metadata_capacity_bytes = actual_bytes;
        self.refresh_current()?;
        Ok(true)
    }

    pub(super) fn evict_until_fits(
        &mut self,
        retained_bytes: usize,
        excluded: Option<usize>,
    ) -> Result<(), JpegPlanCacheError> {
        while self.entries.len() >= self.entry_limit
            || !fits_within(
                self.diagnostics.retained_bytes,
                retained_bytes,
                self.host_byte_limit,
            )
        {
            let index = self
                .oldest_index(excluded)
                .ok_or(JpegPlanCacheError::Invariant(
                    "JPEG plan cache has no entry available for required eviction",
                ))?;
            self.evict_index(index)?;
        }
        Ok(())
    }

    fn replacement_fits(
        &self,
        index: usize,
        retained_bytes: usize,
    ) -> Result<bool, JpegPlanCacheError> {
        let remaining = self
            .entry_bytes
            .checked_sub(self.entries[index].retained_bytes)
            .ok_or(JpegPlanCacheError::Invariant(
                "JPEG plan cache replacement byte accounting underflow",
            ))?;
        Ok(self
            .diagnostics
            .metadata_capacity_bytes
            .checked_add(remaining)
            .and_then(|bytes| bytes.checked_add(retained_bytes))
            .is_some_and(|bytes| bytes <= self.host_byte_limit))
    }

    fn oldest_index(&self, excluded: Option<usize>) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != excluded)
            .min_by_key(|(index, entry)| (entry.last_used, *index))
            .map(|(index, _)| index)
    }

    fn evict_index(&mut self, index: usize) -> Result<(), JpegPlanCacheError> {
        let entry = self.entries.remove(index);
        self.entry_bytes = self.entry_bytes.checked_sub(entry.retained_bytes).ok_or(
            JpegPlanCacheError::Invariant("JPEG plan cache retained bytes underflow on eviction"),
        )?;
        self.diagnostics.evictions = self.diagnostics.evictions.saturating_add(1);
        self.refresh_current()
    }
}
