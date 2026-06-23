// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use super::PreparedDirectGrayscalePlan;
use crate::Error;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CpuTier1CoefficientCacheKey {
    step_idx: usize,
    output_len: usize,
}

#[derive(Default)]
pub(super) struct CpuTier1CoefficientCache {
    entries: Mutex<HashMap<CpuTier1CoefficientCacheKey, Arc<[f32]>>>,
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
        let entries = self
            .cpu_tier1_cache
            .entries
            .lock()
            .map_err(|_| Error::MetalKernel {
                message: "J2K MetalDirect hybrid CPU Tier-1 cache lock is poisoned".to_string(),
            })?;
        Ok(entries.get(&key).map(|coefficients| coefficients.to_vec()))
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
        let cached = Arc::<[f32]>::from(coefficients.clone());
        let mut entries = self
            .cpu_tier1_cache
            .entries
            .lock()
            .map_err(|_| Error::MetalKernel {
                message: "J2K MetalDirect hybrid CPU Tier-1 cache lock is poisoned".to_string(),
            })?;
        entries.insert(key, cached);
        Ok(coefficients)
    }

    pub(super) fn clear_cpu_tier1_cache(&self) -> Result<(), Error> {
        let mut entries = self
            .cpu_tier1_cache
            .entries
            .lock()
            .map_err(|_| Error::MetalKernel {
                message: "J2K MetalDirect hybrid CPU Tier-1 cache lock is poisoned".to_string(),
            })?;
        entries.clear();
        Ok(())
    }
}
