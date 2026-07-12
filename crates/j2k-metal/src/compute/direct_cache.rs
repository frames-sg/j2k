// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    mem::size_of,
    sync::{Arc, Mutex},
};

use super::PreparedDirectGrayscalePlan;
use crate::Error;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CpuTier1CoefficientCacheKey {
    step_idx: usize,
    output_len: usize,
}

pub(super) struct CpuTier1CoefficientCache {
    state: Mutex<CpuTier1CoefficientCacheState>,
}

struct CpuTier1CoefficientCacheState {
    retention_enabled: bool,
    entries: Vec<CpuTier1CoefficientCacheEntry>,
}

struct CpuTier1CoefficientCacheEntry {
    key: CpuTier1CoefficientCacheKey,
    coefficients: Arc<[f32]>,
}

impl Default for CpuTier1CoefficientCache {
    fn default() -> Self {
        Self {
            state: Mutex::new(CpuTier1CoefficientCacheState {
                retention_enabled: true,
                entries: Vec::new(),
            }),
        }
    }
}

impl PreparedDirectGrayscalePlan {
    pub(super) fn cached_cpu_tier1_coefficients(
        &self,
        step_idx: usize,
        output_len: usize,
    ) -> Result<Option<Vec<f32>>, Error> {
        let key = CpuTier1CoefficientCacheKey {
            step_idx,
            output_len,
        };
        let state = self
            .cpu_tier1_cache
            .state
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "hybrid CPU Tier-1 coefficient cache",
            })?;
        if !state.retention_enabled {
            return Ok(None);
        }
        Ok(state
            .entries
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| entry.coefficients.to_vec()))
    }

    pub(super) fn store_cpu_tier1_coefficients(
        &self,
        step_idx: usize,
        output_len: usize,
        coefficients: Vec<f32>,
    ) -> Result<Vec<f32>, Error> {
        let key = CpuTier1CoefficientCacheKey {
            step_idx,
            output_len,
        };
        let mut state =
            self.cpu_tier1_cache
                .state
                .lock()
                .map_err(|_| Error::MetalStatePoisoned {
                    state: "hybrid CPU Tier-1 coefficient cache",
                })?;
        if !state.retention_enabled {
            return Ok(coefficients);
        }
        let existing = state.entries.iter().position(|entry| entry.key == key);
        if existing.is_none() {
            state
                .entries
                .try_reserve(1)
                .map_err(|source| Error::PreparedPlanCacheAllocation {
                    context: "J2K Metal hybrid CPU Tier-1 cache metadata growth failed",
                    source,
                })?;
        }
        let mut cached = Vec::new();
        cached
            .try_reserve_exact(coefficients.len())
            .map_err(|source| Error::PreparedPlanCacheAllocation {
                context: "J2K Metal hybrid CPU Tier-1 coefficient retention failed",
                source,
            })?;
        cached.extend_from_slice(&coefficients);
        let cached = Arc::<[f32]>::from(cached);
        if let Some(index) = existing {
            state.entries[index].coefficients = cached;
        } else {
            state.entries.push(CpuTier1CoefficientCacheEntry {
                key,
                coefficients: cached,
            });
        }
        Ok(coefficients)
    }

    pub(super) fn clear_cpu_tier1_cache(&self) -> Result<(), Error> {
        let mut state =
            self.cpu_tier1_cache
                .state
                .lock()
                .map_err(|_| Error::MetalStatePoisoned {
                    state: "hybrid CPU Tier-1 coefficient cache",
                })?;
        state.entries.clear();
        Ok(())
    }

    pub(crate) fn disable_cpu_tier1_retention(&self) -> Result<(), Error> {
        let mut state =
            self.cpu_tier1_cache
                .state
                .lock()
                .map_err(|_| Error::MetalStatePoisoned {
                    state: "hybrid CPU Tier-1 coefficient cache",
                })?;
        state.retention_enabled = false;
        state.entries.clear();
        Ok(())
    }
}

impl CpuTier1CoefficientCache {
    pub(super) fn retained_cache_bytes(&self) -> Result<usize, &'static str> {
        let state = self
            .state
            .lock()
            .map_err(|_| "prepared-plan CPU Tier-1 cache lock is poisoned")?;
        let metadata = state
            .entries
            .capacity()
            .checked_mul(size_of::<CpuTier1CoefficientCacheEntry>())
            .ok_or("prepared-plan CPU Tier-1 cache metadata overflow")?;
        state.entries.iter().try_fold(metadata, |bytes, entry| {
            let coefficient_bytes = entry
                .coefficients
                .len()
                .checked_mul(size_of::<f32>())
                .ok_or("prepared-plan CPU Tier-1 coefficient byte overflow")?;
            bytes
                .checked_add(coefficient_bytes)
                .and_then(|bytes| bytes.checked_add(2 * size_of::<usize>()))
                .ok_or("prepared-plan CPU Tier-1 aggregate byte overflow")
        })
    }
}
