// SPDX-License-Identifier: MIT OR Apache-2.0

//! Session cache policy for immutable prepared HT executions.

use core::mem::size_of;
use std::sync::Arc;

use j2k_core::HtGpuJobChunkLimits;
use metal::Device;

use super::super::HtBatchInput;
use super::PreparedMetalHtExecution;
use crate::compute::{Error, MetalRuntime};
use crate::session::{PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES, PREPARED_PLAN_CACHE_MAX_HOST_BYTES};

mod entry;

use self::entry::PreparedMetalHtExecutionCacheEntry;

const PREPARED_METAL_HT_EXECUTION_CACHE_CAP: usize = 128;

pub(in crate::compute) struct PreparedMetalHtExecutionCache {
    entries: Vec<PreparedMetalHtExecutionCacheEntry>,
    retained_host_bytes: usize,
    retained_device_bytes: usize,
    access_clock: u64,
}

impl PreparedMetalHtExecutionCache {
    pub(in crate::compute) const fn new() -> Self {
        Self {
            entries: Vec::new(),
            retained_host_bytes: 0,
            retained_device_bytes: 0,
            access_clock: 0,
        }
    }

    fn get_or_prepare(
        &mut self,
        device: &Device,
        batches: &[HtBatchInput<'_>],
        limits: HtGpuJobChunkLimits,
    ) -> Result<Arc<PreparedMetalHtExecution>, Error> {
        if let Some(index) = self.find(batches, limits) {
            self.access_clock = self.access_clock.saturating_add(1);
            self.entries[index].mark_used(self.access_clock);
            return Ok(self.entries[index].execution());
        }

        let (execution, entry) = self.prepare_entry(device, batches, limits)?;
        let Some(entry) = entry else {
            return Ok(execution);
        };
        self.evict_until_fits(entry.host_bytes(), entry.device_bytes())?;
        self.insert_entry(entry)?;
        Ok(execution)
    }

    fn prepare_entry(
        &mut self,
        device: &Device,
        batches: &[HtBatchInput<'_>],
        limits: HtGpuJobChunkLimits,
    ) -> Result<
        (
            Arc<PreparedMetalHtExecution>,
            Option<PreparedMetalHtExecutionCacheEntry>,
        ),
        Error,
    > {
        self.ensure_metadata_capacity()?;
        let metadata_host_bytes = self
            .entries
            .capacity()
            .checked_mul(size_of::<PreparedMetalHtExecutionCacheEntry>())
            .ok_or(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "entry metadata byte count overflow",
            })?;
        PreparedMetalHtExecutionCacheEntry::prepare(device, batches, limits, metadata_host_bytes)
    }

    fn insert_entry(&mut self, mut entry: PreparedMetalHtExecutionCacheEntry) -> Result<(), Error> {
        let retained_host_bytes = self
            .retained_host_bytes
            .checked_add(entry.host_bytes())
            .ok_or(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "retained host byte count overflow",
            })?;
        let retained_device_bytes = self
            .retained_device_bytes
            .checked_add(entry.device_bytes())
            .ok_or(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "retained device byte count overflow",
            })?;
        self.entries
            .try_reserve(1)
            .map_err(|source| Error::PreparedPlanCacheAllocation {
                context: "J2K Metal prepared HT execution cache entry",
                source,
            })?;
        self.access_clock = self.access_clock.saturating_add(1);
        entry.mark_used(self.access_clock);
        self.entries.push(entry);
        self.retained_host_bytes = retained_host_bytes;
        self.retained_device_bytes = retained_device_bytes;
        Ok(())
    }

    fn find(&self, batches: &[HtBatchInput<'_>], limits: HtGpuJobChunkLimits) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| entry.matches(batches, limits))
    }

    fn ensure_metadata_capacity(&mut self) -> Result<(), Error> {
        if self.entries.capacity() >= PREPARED_METAL_HT_EXECUTION_CACHE_CAP {
            return Ok(());
        }
        if !self.entries.is_empty() {
            return Err(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "entry metadata capacity changed after cache insertion",
            });
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(PREPARED_METAL_HT_EXECUTION_CACHE_CAP)
            .map_err(|source| Error::PreparedPlanCacheAllocation {
                context: "J2K Metal prepared HT execution cache metadata",
                source,
            })?;
        let metadata_bytes = entries
            .capacity()
            .checked_mul(size_of::<PreparedMetalHtExecutionCacheEntry>())
            .ok_or(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "entry metadata byte count overflow",
            })?;
        if metadata_bytes > PREPARED_PLAN_CACHE_MAX_HOST_BYTES {
            return Err(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "entry metadata exceeds the shared host cache limit",
            });
        }
        self.entries = entries;
        self.retained_host_bytes = metadata_bytes;
        Ok(())
    }

    fn evict_until_fits(
        &mut self,
        incoming_host_bytes: usize,
        incoming_device_bytes: usize,
    ) -> Result<(), Error> {
        while self.entries.len() >= PREPARED_METAL_HT_EXECUTION_CACHE_CAP
            || self
                .retained_host_bytes
                .checked_add(incoming_host_bytes)
                .is_none_or(|bytes| bytes > PREPARED_PLAN_CACHE_MAX_HOST_BYTES)
            || self
                .retained_device_bytes
                .checked_add(incoming_device_bytes)
                .is_none_or(|bytes| bytes > PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES)
        {
            let index = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(index, entry)| (entry.last_used(), *index))
                .map(|(index, _)| index)
                .ok_or(Error::MetalStateInvariant {
                    state: "J2K Metal prepared HT execution cache",
                    reason: "cache limits require eviction but no entry is retained",
                })?;
            let entry = self.entries.remove(index);
            self.retained_host_bytes = self
                .retained_host_bytes
                .checked_sub(entry.host_bytes())
                .ok_or(Error::MetalStateInvariant {
                    state: "J2K Metal prepared HT execution cache",
                    reason: "retained host byte count underflow",
                })?;
            self.retained_device_bytes = self
                .retained_device_bytes
                .checked_sub(entry.device_bytes())
                .ok_or(Error::MetalStateInvariant {
                    state: "J2K Metal prepared HT execution cache",
                    reason: "retained device byte count underflow",
                })?;
        }
        Ok(())
    }
}

pub(in crate::compute) fn prepared_metal_ht_execution(
    runtime: &MetalRuntime,
    batches: &[HtBatchInput<'_>],
    limits: HtGpuJobChunkLimits,
) -> Result<Arc<PreparedMetalHtExecution>, Error> {
    runtime
        .prepared_ht_execution_cache
        .lock()
        .map_err(|_| Error::MetalStatePoisoned {
            state: "J2K Metal prepared HT execution cache",
        })?
        .get_or_prepare(&runtime.device, batches, limits)
}

#[cfg(test)]
mod tests;
