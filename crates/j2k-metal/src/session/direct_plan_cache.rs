// SPDX-License-Identifier: MIT OR Apache-2.0

//! Session-owned native and prepared direct-plan caches.

use std::sync::{Arc, Mutex};

use j2k_core::PixelFormat;

use super::{cache, MetalBackendSession};
use crate::Error;

const DIRECT_PLAN_CACHE_CAP: usize = 128;

#[derive(Clone)]
struct DirectGrayPlanCacheEntry {
    prepared: Arc<crate::engine::PreparedDirectGrayscalePlan>,
}

#[derive(Clone)]
struct DirectColorPlanCacheEntry {
    prepared: Arc<crate::engine::PreparedDirectColorPlan>,
}

#[derive(Clone)]
pub(super) struct DirectPlanCaches {
    gray: Arc<Mutex<cache::PreparedPlanCache<DirectGrayPlanCacheEntry>>>,
    color: Arc<Mutex<cache::PreparedPlanCache<DirectColorPlanCacheEntry>>>,
    region_scaled_color:
        Arc<Mutex<cache::PreparedPlanCache<Arc<crate::engine::PreparedDirectColorPlan>>>>,
}

impl DirectPlanCaches {
    pub(super) fn new() -> Self {
        Self {
            gray: Arc::new(Mutex::new(cache::PreparedPlanCache::new(
                DIRECT_PLAN_CACHE_CAP,
            ))),
            color: Arc::new(Mutex::new(cache::PreparedPlanCache::new(
                DIRECT_PLAN_CACHE_CAP,
            ))),
            region_scaled_color: Arc::new(Mutex::new(cache::PreparedPlanCache::new(
                crate::hybrid::REGION_SCALED_COLOR_PLAN_CACHE_CAP,
            ))),
        }
    }

    #[cfg(test)]
    fn ids(&self) -> (usize, usize, usize) {
        (
            Arc::as_ptr(&self.gray) as usize,
            Arc::as_ptr(&self.color) as usize,
            Arc::as_ptr(&self.region_scaled_color) as usize,
        )
    }
}

// Plans built by `J2kDecoder` hold the only reference to the codestream copy
// their HT payload ranges point into, so the cache weight counts it.
impl cache::PreparedPlanCacheValue for DirectGrayPlanCacheEntry {
    fn retained_cache_weight(
        &self,
    ) -> Result<cache::PreparedPlanCacheWeight, cache::PreparedPlanCacheError> {
        let retained = self
            .prepared
            .retained_cache_bytes()
            .map_err(cache::PreparedPlanCacheError::Invariant)?;
        let input = self
            .prepared
            .referenced_input_bytes()
            .map_err(cache::PreparedPlanCacheError::Invariant)?;
        owning_plan_cache_weight(
            core::mem::size_of::<crate::engine::PreparedDirectGrayscalePlan>(),
            retained.host,
            retained.device,
            input,
        )
    }
}

impl cache::PreparedPlanCacheValue for DirectColorPlanCacheEntry {
    fn retained_cache_weight(
        &self,
    ) -> Result<cache::PreparedPlanCacheWeight, cache::PreparedPlanCacheError> {
        let retained = self
            .prepared
            .retained_cache_bytes()
            .map_err(cache::PreparedPlanCacheError::Invariant)?;
        let input = self
            .prepared
            .referenced_input_bytes()
            .map_err(cache::PreparedPlanCacheError::Invariant)?;
        owning_plan_cache_weight(
            core::mem::size_of::<crate::engine::PreparedDirectColorPlan>(),
            retained.host,
            retained.device,
            input,
        )
    }
}

fn owning_plan_cache_weight(
    root_bytes: usize,
    retained_host_bytes: usize,
    retained_device_bytes: usize,
    referenced_input_bytes: usize,
) -> Result<cache::PreparedPlanCacheWeight, cache::PreparedPlanCacheError> {
    let nested_host = retained_host_bytes
        .checked_add(referenced_input_bytes)
        .ok_or(cache::PreparedPlanCacheError::Invariant(
            "direct-plan cache weight overflow",
        ))?;
    Ok(cache::PreparedPlanCacheWeight::new(
        arc_owner_bytes(root_bytes, nested_host)?,
        retained_device_bytes,
    ))
}

impl cache::PreparedPlanCacheValue for Arc<crate::engine::PreparedDirectColorPlan> {
    fn retained_cache_weight(
        &self,
    ) -> Result<cache::PreparedPlanCacheWeight, cache::PreparedPlanCacheError> {
        let prepared = self
            .retained_cache_bytes()
            .map_err(cache::PreparedPlanCacheError::Invariant)?;
        let host_bytes = arc_owner_bytes(
            core::mem::size_of::<crate::engine::PreparedDirectColorPlan>(),
            prepared.host,
        )?;
        Ok(cache::PreparedPlanCacheWeight::new(
            host_bytes,
            prepared.device,
        ))
    }
}

impl cache::PreparedPlanCacheValue for Arc<crate::engine::PreparedDirectGrayscalePlan> {
    fn retained_cache_weight(
        &self,
    ) -> Result<cache::PreparedPlanCacheWeight, cache::PreparedPlanCacheError> {
        let prepared = self
            .retained_cache_bytes()
            .map_err(cache::PreparedPlanCacheError::Invariant)?;
        let host_bytes = arc_owner_bytes(
            core::mem::size_of::<crate::engine::PreparedDirectGrayscalePlan>(),
            prepared.host,
        )?;
        Ok(cache::PreparedPlanCacheWeight::new(
            host_bytes,
            prepared.device,
        ))
    }
}

fn arc_owner_bytes(
    root_bytes: usize,
    nested_bytes: usize,
) -> Result<usize, cache::PreparedPlanCacheError> {
    root_bytes
        .checked_add(2 * core::mem::size_of::<usize>())
        .and_then(|bytes| bytes.checked_add(nested_bytes))
        .ok_or(cache::PreparedPlanCacheError::Invariant(
            "prepared-plan Arc owner byte count overflow",
        ))
}

pub(crate) fn direct_gray_plan_cache_key(
    bytes: &[u8],
    format: PixelFormat,
) -> cache::PreparedPlanCacheKey<'_> {
    cache::PreparedPlanCacheKey::direct_gray(bytes, format)
}

pub(crate) fn direct_plan_cache_key(
    bytes: &[u8],
    format: PixelFormat,
) -> cache::PreparedPlanCacheKey<'_> {
    cache::PreparedPlanCacheKey::direct_color(bytes, format)
}

pub(crate) fn cached_session_direct_gray_plan(
    session: &MetalBackendSession,
    key: cache::PreparedPlanCacheKey<'_>,
) -> Result<Option<Arc<crate::engine::PreparedDirectGrayscalePlan>>, Error> {
    let mut guard =
        session
            .direct_plan_caches
            .gray
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "direct grayscale prepared-plan cache",
            })?;
    Ok(guard.get(key).map(|entry| entry.prepared.clone()))
}

pub(crate) fn store_session_direct_gray_plan(
    session: &MetalBackendSession,
    key: cache::PreparedPlanCacheKey<'_>,
    prepared: Arc<crate::engine::PreparedDirectGrayscalePlan>,
) -> Result<(), Error> {
    prepared.disable_cpu_tier1_retention()?;
    let mut guard =
        session
            .direct_plan_caches
            .gray
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "direct grayscale prepared-plan cache",
            })?;
    evict_one_direct_plan_if_needed(&mut guard, key, DirectGrayPlanCacheEntry { prepared })
}

pub(crate) fn cached_session_direct_color_plan(
    session: &MetalBackendSession,
    key: cache::PreparedPlanCacheKey<'_>,
) -> Result<Option<Arc<crate::engine::PreparedDirectColorPlan>>, Error> {
    let mut guard =
        session
            .direct_plan_caches
            .color
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "direct color prepared-plan cache",
            })?;
    Ok(guard.get(key).map(|entry| entry.prepared.clone()))
}

pub(crate) fn store_session_direct_color_plan(
    session: &MetalBackendSession,
    key: cache::PreparedPlanCacheKey<'_>,
    prepared: Arc<crate::engine::PreparedDirectColorPlan>,
) -> Result<(), Error> {
    prepared.disable_dynamic_cpu_tier1_retention()?;
    let mut guard =
        session
            .direct_plan_caches
            .color
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "direct color prepared-plan cache",
            })?;
    evict_one_direct_plan_if_needed(&mut guard, key, DirectColorPlanCacheEntry { prepared })
}

pub(crate) fn cached_session_region_scaled_color_plan(
    session: &MetalBackendSession,
    key: cache::PreparedPlanCacheKey<'_>,
) -> Result<Option<Arc<crate::engine::PreparedDirectColorPlan>>, Error> {
    let mut guard = session
        .direct_plan_caches
        .region_scaled_color
        .lock()
        .map_err(|_| Error::MetalStatePoisoned {
            state: "session region-scaled color prepared-plan cache",
        })?;
    Ok(guard.get(key).cloned())
}

pub(crate) fn store_session_region_scaled_color_plan(
    session: &MetalBackendSession,
    key: cache::PreparedPlanCacheKey<'_>,
    plan: Arc<crate::engine::PreparedDirectColorPlan>,
) -> Result<(), Error> {
    let mut guard = session
        .direct_plan_caches
        .region_scaled_color
        .lock()
        .map_err(|_| Error::MetalStatePoisoned {
            state: "session region-scaled color prepared-plan cache",
        })?;
    guard.insert(key, plan).map(|_| ()).map_err(|error| {
        prepared_plan_cache_error(
            "Metal region-scaled prepared-plan cache update failed",
            error,
        )
    })
}

fn evict_one_direct_plan_if_needed<T: cache::PreparedPlanCacheValue>(
    cache: &mut cache::PreparedPlanCache<T>,
    key: cache::PreparedPlanCacheKey<'_>,
    value: T,
) -> Result<(), Error> {
    cache.insert(key, value).map(|_| ()).map_err(|error| {
        prepared_plan_cache_error("Metal prepared-plan cache update failed", error)
    })
}

pub(crate) fn prepared_plan_cache_error(
    context: &'static str,
    error: cache::PreparedPlanCacheError,
) -> Error {
    match error {
        cache::PreparedPlanCacheError::Allocation(source) => {
            Error::PreparedPlanCacheAllocation { context, source }
        }
        cache::PreparedPlanCacheError::Invariant(reason) => {
            Error::PreparedPlanCacheInvariant { context, reason }
        }
    }
}

#[cfg(test)]
pub(crate) fn direct_cache_ids_for_test(session: &MetalBackendSession) -> (usize, usize, usize) {
    session.direct_plan_caches.ids()
}

#[cfg(test)]
mod tests;
