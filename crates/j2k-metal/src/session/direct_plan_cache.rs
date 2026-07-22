// SPDX-License-Identifier: MIT OR Apache-2.0

//! Session-owned native and prepared direct-plan caches.

use std::sync::{Arc, Mutex};

use j2k_core::PixelFormat;
use j2k_native::{J2kDirectColorPlan, J2kDirectGrayscalePlan};

use super::{cache, MetalBackendSession};
use crate::Error;

const DIRECT_PLAN_CACHE_CAP: usize = 128;

#[derive(Clone)]
struct DirectGrayPlanCacheEntry {
    plan: Arc<J2kDirectGrayscalePlan>,
    prepared: Arc<crate::compute::PreparedDirectGrayscalePlan>,
}

#[derive(Clone)]
struct DirectColorPlanCacheEntry {
    plan: Arc<J2kDirectColorPlan>,
    prepared: Arc<crate::compute::PreparedDirectColorPlan>,
}

type CachedDirectGrayPlan = (
    Arc<J2kDirectGrayscalePlan>,
    Arc<crate::compute::PreparedDirectGrayscalePlan>,
);

type CachedDirectColorPlan = (
    Arc<J2kDirectColorPlan>,
    Arc<crate::compute::PreparedDirectColorPlan>,
);

#[derive(Clone)]
pub(super) struct DirectPlanCaches {
    gray: Arc<Mutex<cache::PreparedPlanCache<DirectGrayPlanCacheEntry>>>,
    color: Arc<Mutex<cache::PreparedPlanCache<DirectColorPlanCacheEntry>>>,
    region_scaled_color:
        Arc<Mutex<cache::PreparedPlanCache<Arc<crate::compute::PreparedDirectColorPlan>>>>,
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

impl cache::PreparedPlanCacheValue for DirectGrayPlanCacheEntry {
    fn retained_cache_weight(
        &self,
    ) -> Result<cache::PreparedPlanCacheWeight, cache::PreparedPlanCacheError> {
        let prepared = self
            .prepared
            .retained_cache_bytes()
            .map_err(cache::PreparedPlanCacheError::Invariant)?;
        direct_plan_cache_weight(
            self.plan.retained_allocation_bytes().map_err(|_| {
                cache::PreparedPlanCacheError::Invariant(
                    "native grayscale direct-plan retained-byte accounting failed",
                )
            })?,
            core::mem::size_of::<J2kDirectGrayscalePlan>(),
            prepared.host,
            prepared.device,
            core::mem::size_of::<crate::compute::PreparedDirectGrayscalePlan>(),
        )
    }
}

impl cache::PreparedPlanCacheValue for DirectColorPlanCacheEntry {
    fn retained_cache_weight(
        &self,
    ) -> Result<cache::PreparedPlanCacheWeight, cache::PreparedPlanCacheError> {
        let prepared = self
            .prepared
            .retained_cache_bytes()
            .map_err(cache::PreparedPlanCacheError::Invariant)?;
        direct_plan_cache_weight(
            self.plan.retained_allocation_bytes().map_err(|_| {
                cache::PreparedPlanCacheError::Invariant(
                    "native color direct-plan retained-byte accounting failed",
                )
            })?,
            core::mem::size_of::<J2kDirectColorPlan>(),
            prepared.host,
            prepared.device,
            core::mem::size_of::<crate::compute::PreparedDirectColorPlan>(),
        )
    }
}

impl cache::PreparedPlanCacheValue for Arc<crate::compute::PreparedDirectColorPlan> {
    fn retained_cache_weight(
        &self,
    ) -> Result<cache::PreparedPlanCacheWeight, cache::PreparedPlanCacheError> {
        let prepared = self
            .retained_cache_bytes()
            .map_err(cache::PreparedPlanCacheError::Invariant)?;
        let host_bytes = arc_owner_bytes(
            core::mem::size_of::<crate::compute::PreparedDirectColorPlan>(),
            prepared.host,
        )?;
        Ok(cache::PreparedPlanCacheWeight::new(
            host_bytes,
            prepared.device,
        ))
    }
}

impl cache::PreparedPlanCacheValue for Arc<crate::compute::PreparedDirectGrayscalePlan> {
    fn retained_cache_weight(
        &self,
    ) -> Result<cache::PreparedPlanCacheWeight, cache::PreparedPlanCacheError> {
        let prepared = self
            .retained_cache_bytes()
            .map_err(cache::PreparedPlanCacheError::Invariant)?;
        let host_bytes = arc_owner_bytes(
            core::mem::size_of::<crate::compute::PreparedDirectGrayscalePlan>(),
            prepared.host,
        )?;
        Ok(cache::PreparedPlanCacheWeight::new(
            host_bytes,
            prepared.device,
        ))
    }
}

fn direct_plan_cache_weight(
    native_nested_host: usize,
    native_root_bytes: usize,
    prepared_host_bytes: usize,
    prepared_device_bytes: usize,
    prepared_root_bytes: usize,
) -> Result<cache::PreparedPlanCacheWeight, cache::PreparedPlanCacheError> {
    let native_host = arc_owner_bytes(native_root_bytes, native_nested_host)?;
    let prepared_host = arc_owner_bytes(prepared_root_bytes, prepared_host_bytes)?;
    let host_bytes =
        native_host
            .checked_add(prepared_host)
            .ok_or(cache::PreparedPlanCacheError::Invariant(
                "native and prepared direct-plan cache weight overflow",
            ))?;
    Ok(cache::PreparedPlanCacheWeight::new(
        host_bytes,
        prepared_device_bytes,
    ))
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
) -> Result<Option<CachedDirectGrayPlan>, Error> {
    let mut guard =
        session
            .direct_plan_caches
            .gray
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "direct grayscale prepared-plan cache",
            })?;
    Ok(guard
        .get(key)
        .map(|entry| (entry.plan.clone(), entry.prepared.clone())))
}

pub(crate) fn store_session_direct_gray_plan(
    session: &MetalBackendSession,
    key: cache::PreparedPlanCacheKey<'_>,
    plan: Arc<J2kDirectGrayscalePlan>,
    prepared: Arc<crate::compute::PreparedDirectGrayscalePlan>,
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
    evict_one_direct_plan_if_needed(&mut guard, key, DirectGrayPlanCacheEntry { plan, prepared })
}

pub(crate) fn cached_session_direct_color_plan(
    session: &MetalBackendSession,
    key: cache::PreparedPlanCacheKey<'_>,
) -> Result<Option<CachedDirectColorPlan>, Error> {
    let mut guard =
        session
            .direct_plan_caches
            .color
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "direct color prepared-plan cache",
            })?;
    Ok(guard
        .get(key)
        .map(|entry| (entry.plan.clone(), entry.prepared.clone())))
}

pub(crate) fn store_session_direct_color_plan(
    session: &MetalBackendSession,
    key: cache::PreparedPlanCacheKey<'_>,
    plan: Arc<J2kDirectColorPlan>,
    prepared: Arc<crate::compute::PreparedDirectColorPlan>,
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
    evict_one_direct_plan_if_needed(
        &mut guard,
        key,
        DirectColorPlanCacheEntry { plan, prepared },
    )
}

pub(crate) fn cached_session_region_scaled_color_plan(
    session: &MetalBackendSession,
    key: cache::PreparedPlanCacheKey<'_>,
) -> Result<Option<Arc<crate::compute::PreparedDirectColorPlan>>, Error> {
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
    plan: Arc<crate::compute::PreparedDirectColorPlan>,
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
