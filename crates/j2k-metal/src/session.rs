// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
};

#[cfg(target_os = "macos")]
use j2k_core::BackendKind;
#[cfg(target_os = "macos")]
use j2k_metal_support::{MetalRuntimeSession, MetalSupportError};
#[cfg(target_os = "macos")]
use j2k_native::{J2kDirectColorPlan, J2kDirectGrayscalePlan};
#[cfg(target_os = "macos")]
use metal::Device;

use crate::{batch, Error};

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct DirectGrayPlanCacheEntry {
    plan: J2kDirectGrayscalePlan,
    prepared: Arc<crate::compute::PreparedDirectGrayscalePlan>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct DirectColorPlanCacheEntry {
    plan: J2kDirectColorPlan,
    prepared: Arc<crate::compute::PreparedDirectColorPlan>,
}

#[cfg(target_os = "macos")]
const DIRECT_PLAN_CACHE_CAP: usize = 128;

#[cfg(target_os = "macos")]
#[derive(Clone)]
/// Reusable Metal device session for J2K decode and encode submissions.
pub struct MetalBackendSession {
    runtime_session: MetalRuntimeSession<Arc<crate::compute::MetalRuntime>, MetalSupportError>,
    direct_gray_plan_cache: Arc<Mutex<HashMap<u64, DirectGrayPlanCacheEntry>>>,
    direct_color_plan_cache: Arc<Mutex<HashMap<u64, DirectColorPlanCacheEntry>>>,
    pub(crate) region_scaled_color_plan_cache:
        Arc<Mutex<HashMap<u64, Arc<crate::compute::PreparedDirectColorPlan>>>>,
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
            direct_gray_plan_cache: Arc::new(Mutex::new(HashMap::new())),
            direct_color_plan_cache: Arc::new(Mutex::new(HashMap::new())),
            region_scaled_color_plan_cache: Arc::new(Mutex::new(HashMap::new())),
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
pub(crate) fn direct_gray_plan_cache_key(bytes: &[u8]) -> u64 {
    direct_plan_cache_key(bytes)
}

#[cfg(target_os = "macos")]
pub(crate) fn cached_session_direct_gray_plan(
    session: &MetalBackendSession,
    key: u64,
) -> Option<(
    J2kDirectGrayscalePlan,
    Arc<crate::compute::PreparedDirectGrayscalePlan>,
)> {
    let guard = session.direct_gray_plan_cache.lock().ok()?;
    guard
        .get(&key)
        .map(|entry| (entry.plan.clone(), entry.prepared.clone()))
}

#[cfg(target_os = "macos")]
pub(crate) fn store_session_direct_gray_plan(
    session: &MetalBackendSession,
    key: u64,
    plan: &J2kDirectGrayscalePlan,
    prepared: Arc<crate::compute::PreparedDirectGrayscalePlan>,
) {
    if let Ok(mut guard) = session.direct_gray_plan_cache.lock() {
        evict_one_direct_plan_if_needed(&mut guard);
        guard.insert(
            key,
            DirectGrayPlanCacheEntry {
                plan: plan.clone(),
                prepared,
            },
        );
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn cached_session_direct_color_plan(
    session: &MetalBackendSession,
    key: u64,
) -> Option<(
    J2kDirectColorPlan,
    Arc<crate::compute::PreparedDirectColorPlan>,
)> {
    let guard = session.direct_color_plan_cache.lock().ok()?;
    guard
        .get(&key)
        .map(|entry| (entry.plan.clone(), entry.prepared.clone()))
}

#[cfg(target_os = "macos")]
pub(crate) fn store_session_direct_color_plan(
    session: &MetalBackendSession,
    key: u64,
    plan: &J2kDirectColorPlan,
    prepared: Arc<crate::compute::PreparedDirectColorPlan>,
) {
    if let Ok(mut guard) = session.direct_color_plan_cache.lock() {
        evict_one_direct_plan_if_needed(&mut guard);
        guard.insert(
            key,
            DirectColorPlanCacheEntry {
                plan: plan.clone(),
                prepared,
            },
        );
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn direct_plan_cache_key(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

#[cfg(target_os = "macos")]
fn evict_one_direct_plan_if_needed<T>(cache: &mut HashMap<u64, T>) {
    if cache.len() < DIRECT_PLAN_CACHE_CAP {
        return;
    }
    if let Some(key) = cache.keys().next().copied() {
        cache.remove(&key);
    }
}
