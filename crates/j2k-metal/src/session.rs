// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::sync::{Arc, Mutex};

#[cfg(target_os = "macos")]
use j2k_core::BackendKind;
#[cfg(target_os = "macos")]
use j2k_metal_support::{MetalRuntimeSession, MetalSupportError};
#[cfg(target_os = "macos")]
use metal::{
    foreign_types::{ForeignType, ForeignTypeRef},
    Device,
};

use crate::{batch, Error};

#[cfg(any(test, target_os = "macos"))]
mod cache;
#[cfg(target_os = "macos")]
pub(crate) mod direct_plan_cache;

#[cfg(target_os = "macos")]
pub(crate) use cache::{
    PreparedPlanCache, PreparedPlanCacheKey, PreparedPlanCacheValue,
    PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES, PREPARED_PLAN_CACHE_MAX_HOST_BYTES,
};

#[cfg(target_os = "macos")]
pub(crate) struct MetalConsumerEventTimeline {
    pub(crate) event: Option<metal::Event>,
    pub(crate) next_value: u64,
}

#[cfg(target_os = "macos")]
impl MetalConsumerEventTimeline {
    const fn new() -> Self {
        Self {
            event: None,
            next_value: 0,
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
/// Reusable Metal device session for J2K decode and encode submissions.
pub struct MetalBackendSession {
    runtime_session: MetalRuntimeSession<Arc<crate::compute::MetalRuntime>, MetalSupportError>,
    command_queue: Option<metal::CommandQueue>,
    direct_plan_caches: direct_plan_cache::DirectPlanCaches,
    consumer_event_timeline: Arc<Mutex<MetalConsumerEventTimeline>>,
}

#[cfg(target_os = "macos")]
impl MetalBackendSession {
    /// Create a session bound to an existing Metal device.
    pub fn new(device: Device) -> Self {
        Self::with_runtime_session(MetalRuntimeSession::new(device), None)
    }

    /// Create a session that submits on an existing queue from the same device.
    ///
    /// Sharing the framework's exact queue provides producer and consumer
    /// ordering without host waits or an additional event bridge.
    pub fn with_command_queue(
        device: Device,
        command_queue: metal::CommandQueue,
    ) -> Result<Self, Error> {
        if command_queue.device().as_ptr() != device.as_ptr() {
            return Err(Error::UnsupportedMetalRequest {
                reason: "command queue belongs to a different Metal device",
            });
        }
        Ok(Self::with_runtime_session(
            MetalRuntimeSession::new(device),
            Some(command_queue),
        ))
    }

    fn with_runtime_session(
        runtime_session: MetalRuntimeSession<Arc<crate::compute::MetalRuntime>, MetalSupportError>,
        command_queue: Option<metal::CommandQueue>,
    ) -> Self {
        Self {
            runtime_session,
            command_queue,
            direct_plan_caches: direct_plan_cache::DirectPlanCaches::new(),
            consumer_event_timeline: Arc::new(Mutex::new(MetalConsumerEventTimeline::new())),
        }
    }

    /// Create a session from the system default Metal device.
    pub fn system_default() -> Result<Self, Error> {
        MetalRuntimeSession::system_default()
            .map(|runtime_session| Self::with_runtime_session(runtime_session, None))
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
        let command_queue = self.command_queue.clone();
        match self.runtime_session.get_or_init_runtime(move |device| {
            match command_queue {
                Some(queue) => {
                    crate::compute::MetalRuntime::new_with_device_and_queue(device, queue)
                }
                None => crate::compute::MetalRuntime::new_with_device(device),
            }
            .map(Arc::new)
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

    /// Return whether this session submits on the exact supplied command queue.
    #[doc(hidden)]
    pub fn uses_command_queue(
        &self,
        command_queue: &metal::CommandQueueRef,
    ) -> Result<bool, Error> {
        Ok(self.runtime()?.queue.as_ptr() == command_queue.as_ptr())
    }

    pub(crate) fn consumer_event_timeline(&self) -> Arc<Mutex<MetalConsumerEventTimeline>> {
        self.consumer_event_timeline.clone()
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
}

impl MetalSession {
    /// Create a batching session backed by an explicit Metal backend session.
    #[cfg(target_os = "macos")]
    pub fn with_backend_session(backend: MetalBackendSession) -> Self {
        Self {
            shared: batch::SharedSession::with_backend_session(backend),
        }
    }

    /// Metal backend session owned by this batching session, if any.
    #[cfg(target_os = "macos")]
    pub fn backend_session(&self) -> Option<&MetalBackendSession> {
        self.shared.backend_session()
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
