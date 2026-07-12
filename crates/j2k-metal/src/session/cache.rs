// SPDX-License-Identifier: MIT OR Apache-2.0

//! Collision-safe, byte-bounded prepared-plan cache.

use std::{collections::hash_map::RandomState, hash::BuildHasher, mem::size_of};

mod key;
use key::OwnedPreparedPlanCacheKey;
pub(crate) use key::PreparedPlanCacheKey;

pub(crate) const PREPARED_PLAN_CACHE_MAX_HOST_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug)]
pub(crate) enum PreparedPlanCacheError {
    Allocation(std::collections::TryReserveError),
    Invariant(&'static str),
}

impl core::fmt::Display for PreparedPlanCacheError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Allocation(error) => write!(formatter, "allocation failed: {error}"),
            Self::Invariant(message) => write!(formatter, "cache invariant failed: {message}"),
        }
    }
}

impl std::error::Error for PreparedPlanCacheError {}

impl From<std::collections::TryReserveError> for PreparedPlanCacheError {
    fn from(error: std::collections::TryReserveError) -> Self {
        Self::Allocation(error)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PreparedPlanCacheWeight {
    pub(crate) host_bytes: usize,
    pub(crate) device_bytes: usize,
}

impl PreparedPlanCacheWeight {
    pub(crate) const fn new(host_bytes: usize, device_bytes: usize) -> Self {
        Self {
            host_bytes,
            device_bytes,
        }
    }
}

pub(crate) trait PreparedPlanCacheValue {
    fn retained_cache_weight(&self) -> Result<PreparedPlanCacheWeight, PreparedPlanCacheError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PreparedPlanCacheInsert {
    Cached,
    SkippedDisabled,
    SkippedOversized,
}

struct PreparedPlanCacheEntry<V> {
    digest: u64,
    key: OwnedPreparedPlanCacheKey,
    value: V,
    value_weight: PreparedPlanCacheWeight,
    last_used: u64,
}

/// Flat LRU prepared-plan cache with randomized digests and full-key validation.
///
/// Digests are only lookup accelerators. Every hit compares the owned input and
/// all semantic key fields. Host accounting includes allocator-returned key and
/// entry-vector capacities plus each value's nested owners; device accounting
/// uses each retained Metal buffer's reported length.
pub(crate) struct PreparedPlanCache<V, S = RandomState> {
    entries: Vec<PreparedPlanCacheEntry<V>>,
    digest_builder: S,
    entry_limit: usize,
    host_limit: usize,
    device_limit: usize,
    entry_host_bytes: usize,
    device_bytes: usize,
    access_clock: u64,
}

impl<V: PreparedPlanCacheValue> PreparedPlanCache<V> {
    pub(crate) fn new(entry_limit: usize) -> Self {
        Self::with_limits_and_digest_builder(
            entry_limit,
            PREPARED_PLAN_CACHE_MAX_HOST_BYTES,
            PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES,
            RandomState::new(),
        )
    }
}

impl<V, S> PreparedPlanCache<V, S>
where
    V: PreparedPlanCacheValue,
    S: BuildHasher,
{
    #[cfg(test)]
    fn with_digest_builder(entry_limit: usize, digest_builder: S) -> Self {
        Self::with_limits_and_digest_builder(
            entry_limit,
            PREPARED_PLAN_CACHE_MAX_HOST_BYTES,
            PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES,
            digest_builder,
        )
    }

    fn with_limits_and_digest_builder(
        entry_limit: usize,
        host_limit: usize,
        device_limit: usize,
        digest_builder: S,
    ) -> Self {
        Self {
            entries: Vec::new(),
            digest_builder,
            entry_limit,
            host_limit,
            device_limit,
            entry_host_bytes: 0,
            device_bytes: 0,
            access_clock: 0,
        }
    }

    pub(crate) fn get(&mut self, key: PreparedPlanCacheKey<'_>) -> Option<&V> {
        let digest = self.digest_builder.hash_one(key);
        let index = self.find_index(digest, key)?;
        let stamp = self.next_access_stamp();
        self.entries[index].last_used = stamp;
        Some(&self.entries[index].value)
    }

    pub(crate) fn insert(
        &mut self,
        key: PreparedPlanCacheKey<'_>,
        value: V,
    ) -> Result<PreparedPlanCacheInsert, PreparedPlanCacheError> {
        if self.entry_limit == 0 {
            return Ok(PreparedPlanCacheInsert::SkippedDisabled);
        }

        let value_weight = value.retained_cache_weight()?;
        if value_weight.device_bytes > self.device_limit {
            return Ok(PreparedPlanCacheInsert::SkippedOversized);
        }

        let digest = self.digest_builder.hash_one(key);
        if let Some(index) = self.find_index(digest, key) {
            return self.replace(index, value, value_weight);
        }

        let requested_metadata = self.requested_metadata_bytes()?;
        if requested_metadata
            .checked_add(key.input_len())
            .and_then(|bytes| bytes.checked_add(value_weight.host_bytes))
            .is_none_or(|bytes| bytes > self.host_limit)
        {
            return Ok(PreparedPlanCacheInsert::SkippedOversized);
        }

        let owned_key = OwnedPreparedPlanCacheKey::try_from_borrowed(key)?;
        if !self.ensure_metadata_capacity()? {
            return Ok(PreparedPlanCacheInsert::SkippedOversized);
        }
        if self
            .metadata_host_bytes()?
            .checked_add(owned_key.input_capacity())
            .and_then(|bytes| bytes.checked_add(value_weight.host_bytes))
            .is_none_or(|bytes| bytes > self.host_limit)
        {
            return Ok(PreparedPlanCacheInsert::SkippedOversized);
        }

        let new_entry_host = owned_key
            .input_capacity()
            .checked_add(value_weight.host_bytes)
            .ok_or(PreparedPlanCacheError::Invariant(
                "prepared-plan entry host byte count overflow",
            ))?;
        self.evict_until_fits(new_entry_host, value_weight.device_bytes, None)?;

        let stamp = self.next_access_stamp();
        self.entries.push(PreparedPlanCacheEntry {
            digest,
            key: owned_key,
            value,
            value_weight,
            last_used: stamp,
        });
        self.entry_host_bytes = self.entry_host_bytes.checked_add(new_entry_host).ok_or(
            PreparedPlanCacheError::Invariant("prepared-plan retained host byte count overflow"),
        )?;
        self.device_bytes = self
            .device_bytes
            .checked_add(value_weight.device_bytes)
            .ok_or(PreparedPlanCacheError::Invariant(
                "prepared-plan retained device byte count overflow",
            ))?;
        Ok(PreparedPlanCacheInsert::Cached)
    }

    fn replace(
        &mut self,
        mut index: usize,
        value: V,
        value_weight: PreparedPlanCacheWeight,
    ) -> Result<PreparedPlanCacheInsert, PreparedPlanCacheError> {
        let key_bytes = self.entries[index].key.input_capacity();
        let metadata = self.metadata_host_bytes()?;
        if metadata
            .checked_add(key_bytes)
            .and_then(|bytes| bytes.checked_add(value_weight.host_bytes))
            .is_none_or(|bytes| bytes > self.host_limit)
            || value_weight.device_bytes > self.device_limit
        {
            return Ok(PreparedPlanCacheInsert::SkippedOversized);
        }

        loop {
            let old_weight = self.entries[index].value_weight;
            let projected_host = self
                .retained_host_bytes()?
                .checked_sub(old_weight.host_bytes)
                .and_then(|bytes| bytes.checked_add(value_weight.host_bytes));
            let projected_device = self
                .device_bytes
                .checked_sub(old_weight.device_bytes)
                .and_then(|bytes| bytes.checked_add(value_weight.device_bytes));
            if projected_host.is_some_and(|bytes| bytes <= self.host_limit)
                && projected_device.is_some_and(|bytes| bytes <= self.device_limit)
            {
                break;
            }
            let evicted =
                self.oldest_index(Some(index))
                    .ok_or(PreparedPlanCacheError::Invariant(
                        "replacement that fits alone has no evictable cache entry",
                    ))?;
            self.evict_index(evicted)?;
            if evicted < index {
                index -= 1;
            }
        }

        let old_weight = self.entries[index].value_weight;
        self.entry_host_bytes = self
            .entry_host_bytes
            .checked_sub(old_weight.host_bytes)
            .and_then(|bytes| bytes.checked_add(value_weight.host_bytes))
            .ok_or(PreparedPlanCacheError::Invariant(
                "prepared-plan replacement host accounting overflow",
            ))?;
        self.device_bytes = self
            .device_bytes
            .checked_sub(old_weight.device_bytes)
            .and_then(|bytes| bytes.checked_add(value_weight.device_bytes))
            .ok_or(PreparedPlanCacheError::Invariant(
                "prepared-plan replacement device accounting overflow",
            ))?;
        self.entries[index].value = value;
        self.entries[index].value_weight = value_weight;
        self.entries[index].last_used = self.next_access_stamp();
        Ok(PreparedPlanCacheInsert::Cached)
    }

    fn ensure_metadata_capacity(&mut self) -> Result<bool, PreparedPlanCacheError> {
        if self.entries.capacity() >= self.entry_limit {
            return Ok(true);
        }
        if self.requested_metadata_bytes()? > self.host_limit {
            return Ok(false);
        }

        let mut entries = Vec::new();
        entries.try_reserve_exact(self.entry_limit)?;
        let actual = entries
            .capacity()
            .checked_mul(size_of::<PreparedPlanCacheEntry<V>>())
            .ok_or(PreparedPlanCacheError::Invariant(
                "prepared-plan cache metadata capacity overflow",
            ))?;
        if actual > self.host_limit {
            return Ok(false);
        }
        if !self.entries.is_empty() {
            return Err(PreparedPlanCacheError::Invariant(
                "prepared-plan metadata capacity changed after insertion",
            ));
        }
        self.entries = entries;
        Ok(true)
    }

    fn evict_until_fits(
        &mut self,
        new_host_bytes: usize,
        new_device_bytes: usize,
        retained_index: Option<usize>,
    ) -> Result<(), PreparedPlanCacheError> {
        while self.entries.len() >= self.entry_limit
            || self
                .retained_host_bytes()?
                .checked_add(new_host_bytes)
                .is_none_or(|bytes| bytes > self.host_limit)
            || self
                .device_bytes
                .checked_add(new_device_bytes)
                .is_none_or(|bytes| bytes > self.device_limit)
        {
            let index =
                self.oldest_index(retained_index)
                    .ok_or(PreparedPlanCacheError::Invariant(
                        "prepared-plan cache has no entry available for required eviction",
                    ))?;
            self.evict_index(index)?;
        }
        Ok(())
    }

    fn oldest_index(&self, excluded: Option<usize>) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != excluded)
            .min_by_key(|(index, entry)| (entry.last_used, *index))
            .map(|(index, _)| index)
    }

    fn evict_index(&mut self, index: usize) -> Result<(), PreparedPlanCacheError> {
        let entry = self.entries.remove(index);
        let entry_host = entry
            .key
            .input_capacity()
            .checked_add(entry.value_weight.host_bytes)
            .ok_or(PreparedPlanCacheError::Invariant(
                "evicted prepared-plan entry host byte count overflow",
            ))?;
        self.entry_host_bytes = self.entry_host_bytes.checked_sub(entry_host).ok_or(
            PreparedPlanCacheError::Invariant(
                "prepared-plan retained host byte count underflow on eviction",
            ),
        )?;
        self.device_bytes = self
            .device_bytes
            .checked_sub(entry.value_weight.device_bytes)
            .ok_or(PreparedPlanCacheError::Invariant(
                "prepared-plan retained device byte count underflow on eviction",
            ))?;
        Ok(())
    }

    fn find_index(&self, digest: u64, key: PreparedPlanCacheKey<'_>) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| entry.digest == digest && entry.key.matches(key))
    }

    fn requested_metadata_bytes(&self) -> Result<usize, PreparedPlanCacheError> {
        self.entry_limit
            .checked_mul(size_of::<PreparedPlanCacheEntry<V>>())
            .ok_or(PreparedPlanCacheError::Invariant(
                "prepared-plan requested metadata byte count overflow",
            ))
    }

    fn metadata_host_bytes(&self) -> Result<usize, PreparedPlanCacheError> {
        self.entries
            .capacity()
            .checked_mul(size_of::<PreparedPlanCacheEntry<V>>())
            .ok_or(PreparedPlanCacheError::Invariant(
                "prepared-plan actual metadata byte count overflow",
            ))
    }

    fn retained_host_bytes(&self) -> Result<usize, PreparedPlanCacheError> {
        self.metadata_host_bytes()?
            .checked_add(self.entry_host_bytes)
            .ok_or(PreparedPlanCacheError::Invariant(
                "prepared-plan total retained host byte count overflow",
            ))
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

    #[cfg(test)]
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.entry_host_bytes = 0;
        self.device_bytes = 0;
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    fn retained_device_bytes(&self) -> usize {
        self.device_bytes
    }
}

#[cfg(test)]
mod tests;
