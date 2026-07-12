// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::sync::{Arc, Mutex};

#[cfg(target_os = "macos")]
use j2k_core::{BackendKind, PixelFormat};
#[cfg(target_os = "macos")]
use j2k_metal_support::{MetalRuntimeSession, MetalSupportError};
#[cfg(target_os = "macos")]
use j2k_native::{J2kDirectColorPlan, J2kDirectGrayscalePlan};
#[cfg(target_os = "macos")]
use metal::Device;

use crate::{batch, Error};

#[cfg(any(test, target_os = "macos"))]
mod cache;

#[cfg(target_os = "macos")]
pub(crate) use cache::{
    PreparedPlanCache, PreparedPlanCacheError, PreparedPlanCacheKey, PreparedPlanCacheValue,
};

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct DirectGrayPlanCacheEntry {
    plan: Arc<J2kDirectGrayscalePlan>,
    prepared: Arc<crate::compute::PreparedDirectGrayscalePlan>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct DirectColorPlanCacheEntry {
    plan: Arc<J2kDirectColorPlan>,
    prepared: Arc<crate::compute::PreparedDirectColorPlan>,
}

#[cfg(target_os = "macos")]
type CachedDirectGrayPlan = (
    Arc<J2kDirectGrayscalePlan>,
    Arc<crate::compute::PreparedDirectGrayscalePlan>,
);

#[cfg(target_os = "macos")]
type CachedDirectColorPlan = (
    Arc<J2kDirectColorPlan>,
    Arc<crate::compute::PreparedDirectColorPlan>,
);

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
const DIRECT_PLAN_CACHE_CAP: usize = 128;

#[cfg(target_os = "macos")]
#[derive(Clone)]
/// Reusable Metal device session for J2K decode and encode submissions.
pub struct MetalBackendSession {
    runtime_session: MetalRuntimeSession<Arc<crate::compute::MetalRuntime>, MetalSupportError>,
    direct_gray_plan_cache: Arc<Mutex<PreparedPlanCache<DirectGrayPlanCacheEntry>>>,
    direct_color_plan_cache: Arc<Mutex<PreparedPlanCache<DirectColorPlanCacheEntry>>>,
    pub(crate) region_scaled_color_plan_cache:
        Arc<Mutex<PreparedPlanCache<Arc<crate::compute::PreparedDirectColorPlan>>>>,
}

#[cfg(target_os = "macos")]
impl MetalBackendSession {
    /// Create a session bound to an existing Metal device.
    pub fn new(device: Device) -> Self {
        Self::with_runtime_session(MetalRuntimeSession::new(device))
    }

    fn with_runtime_session(
        runtime_session: MetalRuntimeSession<Arc<crate::compute::MetalRuntime>, MetalSupportError>,
    ) -> Self {
        Self {
            runtime_session,
            direct_gray_plan_cache: Arc::new(Mutex::new(PreparedPlanCache::new(
                DIRECT_PLAN_CACHE_CAP,
            ))),
            direct_color_plan_cache: Arc::new(Mutex::new(PreparedPlanCache::new(
                DIRECT_PLAN_CACHE_CAP,
            ))),
            region_scaled_color_plan_cache: Arc::new(Mutex::new(PreparedPlanCache::new(
                crate::hybrid::REGION_SCALED_COLOR_PLAN_CACHE_CAP,
            ))),
        }
    }

    /// Create a session from the system default Metal device.
    pub fn system_default() -> Result<Self, Error> {
        MetalRuntimeSession::system_default()
            .map(Self::with_runtime_session)
            .map_err(|error| crate::compute::runtime_initialization_error(&error))
    }

    /// Metal device used by this session.
    pub fn device(&self) -> &metal::DeviceRef {
        self.runtime_session.device()
    }

    pub(crate) fn device_handle(&self) -> &Device {
        self.runtime_session.device_handle()
    }

    pub(crate) fn runtime(&self) -> Result<Arc<crate::compute::MetalRuntime>, Error> {
        match self.runtime_session.get_or_init_runtime(|device| {
            crate::compute::MetalRuntime::new_with_device(device).map(Arc::new)
        }) {
            Ok(runtime) => Ok(runtime.clone()),
            Err(error) => Err(crate::compute::runtime_initialization_error(error)),
        }
    }

    /// Return private/shared scratch-pool retention and high-water counters.
    ///
    /// # Errors
    ///
    /// Returns a typed Metal initialization or poisoned-state error when the
    /// runtime or its pool ledger cannot be inspected safely.
    pub fn buffer_pool_diagnostics(&self) -> Result<crate::MetalBufferPoolsDiagnostics, Error> {
        self.runtime()?.buffer_pool_diagnostics()
    }

    #[cfg(test)]
    pub(crate) fn direct_cache_ids_for_test(&self) -> (usize, usize, usize) {
        (
            Arc::as_ptr(&self.direct_gray_plan_cache) as usize,
            Arc::as_ptr(&self.direct_color_plan_cache) as usize,
            Arc::as_ptr(&self.region_scaled_color_plan_cache) as usize,
        )
    }
}

#[cfg(target_os = "macos")]
#[doc(hidden)]
impl j2k_core::AcceleratorSession for MetalBackendSession {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Metal
    }
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for MetalBackendSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalBackendSession")
            .field("device", &self.runtime_session.device_handle().name())
            .finish_non_exhaustive()
    }
}

#[cfg(not(target_os = "macos"))]
#[derive(Clone, Copy, Debug, Default)]
/// Placeholder Metal session for non-macOS builds.
pub struct MetalBackendSession {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
impl MetalBackendSession {
    /// Return `Error::MetalUnavailable` on hosts without Metal support.
    pub fn system_default() -> Result<Self, Error> {
        Err(Error::MetalUnavailable)
    }
}

#[derive(Clone, Default)]
/// Shared batching session used by J2K Metal submit APIs.
pub struct MetalSession {
    pub(crate) shared: batch::SharedSession,
    #[cfg(target_os = "macos")]
    backend: Option<MetalBackendSession>,
}

impl MetalSession {
    /// Create a batching session backed by an explicit Metal backend session.
    #[cfg(target_os = "macos")]
    pub fn with_backend_session(backend: MetalBackendSession) -> Self {
        Self {
            shared: batch::SharedSession::default(),
            backend: Some(backend),
        }
    }

    /// Metal backend session owned by this batching session, if any.
    #[cfg(target_os = "macos")]
    pub fn backend_session(&self) -> Option<&MetalBackendSession> {
        self.backend.as_ref()
    }

    /// Number of Metal or emulated submissions flushed through this session.
    pub fn submissions(&self) -> Result<u64, Error> {
        Ok(self.shared.lock()?.submissions)
    }

    pub(crate) fn record_submit(&mut self) -> Result<(), Error> {
        let mut session = self.shared.lock()?;
        session.submissions = session.submissions.saturating_add(1);
        Ok(())
    }
}

impl core::fmt::Debug for MetalSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalSession")
            .field("submissions", &self.submissions())
            .finish()
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn direct_gray_plan_cache_key(
    bytes: &[u8],
    format: PixelFormat,
) -> PreparedPlanCacheKey<'_> {
    PreparedPlanCacheKey::direct_gray(bytes, format)
}

#[cfg(target_os = "macos")]
pub(crate) fn cached_session_direct_gray_plan(
    session: &MetalBackendSession,
    key: PreparedPlanCacheKey<'_>,
) -> Result<Option<CachedDirectGrayPlan>, Error> {
    let mut guard =
        session
            .direct_gray_plan_cache
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "direct grayscale prepared-plan cache",
            })?;
    Ok(guard
        .get(key)
        .map(|entry| (entry.plan.clone(), entry.prepared.clone())))
}

#[cfg(target_os = "macos")]
pub(crate) fn store_session_direct_gray_plan(
    session: &MetalBackendSession,
    key: PreparedPlanCacheKey<'_>,
    plan: Arc<J2kDirectGrayscalePlan>,
    prepared: Arc<crate::compute::PreparedDirectGrayscalePlan>,
) -> Result<(), Error> {
    prepared.disable_cpu_tier1_retention()?;
    let mut guard =
        session
            .direct_gray_plan_cache
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "direct grayscale prepared-plan cache",
            })?;
    evict_one_direct_plan_if_needed(&mut guard, key, DirectGrayPlanCacheEntry { plan, prepared })
}

#[cfg(target_os = "macos")]
pub(crate) fn cached_session_direct_color_plan(
    session: &MetalBackendSession,
    key: PreparedPlanCacheKey<'_>,
) -> Result<Option<CachedDirectColorPlan>, Error> {
    let mut guard =
        session
            .direct_color_plan_cache
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "direct color prepared-plan cache",
            })?;
    Ok(guard
        .get(key)
        .map(|entry| (entry.plan.clone(), entry.prepared.clone())))
}

#[cfg(target_os = "macos")]
pub(crate) fn store_session_direct_color_plan(
    session: &MetalBackendSession,
    key: PreparedPlanCacheKey<'_>,
    plan: Arc<J2kDirectColorPlan>,
    prepared: Arc<crate::compute::PreparedDirectColorPlan>,
) -> Result<(), Error> {
    prepared.disable_dynamic_cpu_tier1_retention()?;
    let mut guard =
        session
            .direct_color_plan_cache
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

#[cfg(target_os = "macos")]
pub(crate) fn direct_plan_cache_key(bytes: &[u8], format: PixelFormat) -> PreparedPlanCacheKey<'_> {
    PreparedPlanCacheKey::direct_color(bytes, format)
}

#[cfg(target_os = "macos")]
fn evict_one_direct_plan_if_needed<T: PreparedPlanCacheValue>(
    cache: &mut PreparedPlanCache<T>,
    key: PreparedPlanCacheKey<'_>,
    value: T,
) -> Result<(), Error> {
    cache.insert(key, value).map(|_| ()).map_err(|error| {
        prepared_plan_cache_error("Metal prepared-plan cache update failed", error)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepared_plan_cache_error(
    context: &'static str,
    error: PreparedPlanCacheError,
) -> Error {
    match error {
        PreparedPlanCacheError::Allocation(source) => {
            Error::PreparedPlanCacheAllocation { context, source }
        }
        PreparedPlanCacheError::Invariant(reason) => {
            Error::PreparedPlanCacheInvariant { context, reason }
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod error_tests {
    use j2k_core::CodecError;

    use super::{prepared_plan_cache_error, PreparedPlanCacheError};
    use crate::Error;

    #[test]
    fn prepared_plan_cache_allocation_keeps_its_source_and_classification() {
        let source = Vec::<u8>::new()
            .try_reserve(usize::MAX)
            .expect_err("capacity overflow must fail before allocation");
        let source_message = source.to_string();
        let error = prepared_plan_cache_error(
            "Metal prepared-plan cache update failed",
            PreparedPlanCacheError::Allocation(source),
        );

        assert!(matches!(
            &error,
            Error::PreparedPlanCacheAllocation {
                context: "Metal prepared-plan cache update failed",
                ..
            }
        ));
        let chained = std::error::Error::source(&error).expect("cache allocation source");
        assert_eq!(chained.to_string(), source_message);
        assert!(!error.is_unsupported());
        assert!(!error.is_buffer_error());
    }

    #[test]
    fn prepared_plan_cache_invariant_keeps_static_reason_without_source() {
        let error = prepared_plan_cache_error(
            "Metal region-scaled prepared-plan cache update failed",
            PreparedPlanCacheError::Invariant("test cache invariant"),
        );

        assert_eq!(
            error.to_string(),
            "Metal kernel error: Metal region-scaled prepared-plan cache update failed: cache invariant failed: test cache invariant"
        );
        assert!(matches!(
            &error,
            Error::PreparedPlanCacheInvariant {
                context: "Metal region-scaled prepared-plan cache update failed",
                reason: "test cache invariant",
            }
        ));
        assert!(std::error::Error::source(&error).is_none());
        assert!(!error.is_unsupported());
        assert!(!error.is_buffer_error());
    }
}
