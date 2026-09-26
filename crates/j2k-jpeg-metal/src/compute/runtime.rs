// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal device/session lifecycle and mutable runtime cache ownership.

use std::{
    cell::RefCell,
    sync::{Arc, Mutex, MutexGuard},
};

use super::pipeline_registry::JpegPipelineRegistry;
use super::scratch_pool::{BatchScratchLease, BatchScratchPool};
use super::surface_output_pool::SurfaceOutputPool;
use super::viewport_cache::{
    CachedViewportPlanes, ViewportPlaneCacheGate, ViewportPlaneCacheLease,
};
use crate::error::{metal_runtime_support_error, Error};
use crate::metal_types::{Buffer, CommandBuffer, CommandQueue, Device};
use j2k_core::PixelFormat;
use j2k_metal_support::{checked_command_queue, system_default_device, MetalSupportError};

thread_local! {
    static DEFAULT_METAL_SESSION: RefCell<Option<Result<crate::MetalBackendSession, MetalSupportError>>> = const { RefCell::new(None) };
}

/// Backend session shared by every `MetalSession` created without one, so
/// one-shot tile batches reuse one command queue and warm buffer pools instead
/// of building a runtime per session. Its pooled buffers are released under
/// memory pressure and by `release_default_session_buffers`.
static DEFAULT_TILE_SESSION: Mutex<Option<crate::MetalBackendSession>> = Mutex::new(None);

fn default_tile_session_slot(
) -> Result<MutexGuard<'static, Option<crate::MetalBackendSession>>, Error> {
    DEFAULT_TILE_SESSION
        .lock()
        .map_err(|_| Error::MetalStatePoisoned {
            state: "JPEG Metal default tile session",
        })
}

/// The process-wide backend session for `MetalSession`s created without one.
pub(crate) fn default_tile_backend_session() -> Result<crate::MetalBackendSession, Error> {
    let mut slot = default_tile_session_slot()?;
    if let Some(session) = slot.as_ref() {
        return Ok(session.clone());
    }
    let session = crate::MetalBackendSession::system_default()?;
    *slot = Some(session.clone());
    install_memory_pressure_release();
    Ok(session)
}

/// Releases the buffers that the default session of `MetalSession`s created
/// without a backend session keeps pooled between calls.
///
/// Buffers still referenced by surfaces or in-flight work stay alive until
/// those references drop. The session itself, its queue and compiled
/// pipelines are kept. This also runs automatically when macOS reports
/// memory pressure.
///
/// # Errors
///
/// Returns an error if a pool lock was poisoned by a panic.
pub fn release_default_session_buffers() -> Result<(), Error> {
    let session = default_tile_session_slot()?.clone();
    let Some(session) = session else {
        return Ok(());
    };
    match session.initialized_runtime() {
        Some(Ok(runtime)) => runtime.release_pooled_buffers(),
        Some(Err(_)) | None => Ok(()),
    }
}

/// Registers a libdispatch memory-pressure source (warning and critical
/// levels) that calls `release_default_session_buffers`, once per process.
fn install_memory_pressure_release() {
    use dispatch2::{
        dispatch_source_memorypressure_flags_t as PressureFlags, DispatchObject, DispatchQoS,
        DispatchQueue, DispatchSource, GlobalQueueIdentifier,
    };

    extern "C" fn release_on_memory_pressure(_context: *mut core::ffi::c_void) {
        // The handler has no caller to report to; a poisoned pool keeps its
        // buffers and the next explicit release returns the error.
        let _ = release_default_session_buffers();
    }

    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        let queue = DispatchQueue::global_queue(GlobalQueueIdentifier::QualityOfService(
            DispatchQoS::Utility,
        ));
        let levels = PressureFlags::DISPATCH_MEMORYPRESSURE_WARN.0
            | PressureFlags::DISPATCH_MEMORYPRESSURE_CRITICAL.0;
        // SAFETY: a memory-pressure source takes no handle and a mask of
        // pressure levels; the source type is libdispatch's static descriptor.
        let source = unsafe {
            DispatchSource::new(
                (&raw const dispatch2::_dispatch_source_type_memorypressure).cast_mut(),
                0,
                usize::try_from(levels).expect("pressure mask fits usize"),
                Some(&queue),
            )
        };
        source.set_event_handler_f(release_on_memory_pressure);
        source.activate();
        // The source lives for the rest of the process.
        core::mem::forget(source);
    });
}

pub(crate) struct MetalRuntime {
    pub(in crate::compute) device: Device,
    pub(in crate::compute) queue: CommandQueue,
    pub(in crate::compute) pipelines: Arc<JpegPipelineRegistry>,
    batch_scratch: BatchScratchPool,
    surface_outputs: SurfaceOutputPool,
    viewport_plane_cache: Mutex<Option<CachedViewportPlanes>>,
    viewport_plane_cache_gate: Arc<ViewportPlaneCacheGate>,
}

// SAFETY: Metal devices, queues, and immutable pipeline states support
// cross-thread use. All mutable host-side caches are protected by mutexes, and
// each command encoder remains confined to the submission that creates it.
unsafe impl Send for MetalRuntime {}
// SAFETY: Shared runtime operations allocate independent command buffers;
// shared scratch/cache mutation is serialized by the corresponding mutex.
unsafe impl Sync for MetalRuntime {}

impl MetalRuntime {
    #[cfg(test)]
    pub(in crate::compute) fn new() -> Result<Self, MetalSupportError> {
        let device = system_default_device()?;
        Self::new_with_device(device)
    }

    /// Builds an uncached runtime whose pipelines are compiled from
    /// `shader_source`, so the kernel harness can A/B shader variants.
    #[cfg(test)]
    pub(in crate::compute) fn new_with_shader_source(
        shader_source: &str,
    ) -> Result<Self, MetalSupportError> {
        let device = system_default_device()?;
        let pipelines = Arc::new(JpegPipelineRegistry::load_from_source(
            &device,
            shader_source,
        )?);
        Self::from_pipelines(device, pipelines)
    }

    pub(crate) fn new_with_device(device: Device) -> Result<Self, MetalSupportError> {
        let pipelines = JpegPipelineRegistry::shared(&device)?;
        Self::from_pipelines(device, pipelines)
    }

    fn from_pipelines(
        device: Device,
        pipelines: Arc<JpegPipelineRegistry>,
    ) -> Result<Self, MetalSupportError> {
        let queue = checked_command_queue(&device)?;
        Ok(Self {
            device,
            queue,
            pipelines,
            batch_scratch: BatchScratchPool::default(),
            surface_outputs: SurfaceOutputPool::default(),
            viewport_plane_cache: Mutex::new(None),
            viewport_plane_cache_gate: ViewportPlaneCacheGate::new(),
        })
    }

    pub(in crate::compute) fn batch_scratch(&self) -> Result<BatchScratchLease<'_>, Error> {
        self.batch_scratch.acquire()
    }

    pub(in crate::compute) fn try_batch_scratch(
        &self,
    ) -> Result<Option<BatchScratchLease<'_>>, Error> {
        self.batch_scratch.try_acquire()
    }

    /// Shared output buffer of at least `bytes` for batch surfaces that the
    /// caller keeps after this call; see `surface_output_pool`.
    pub(in crate::compute) fn surface_output_buffer(&self, bytes: usize) -> Result<Buffer, Error> {
        self.surface_outputs.acquire(&self.device, bytes)
    }

    /// Drops pooled surface outputs and idle batch scratch; see
    /// `release_default_session_buffers`.
    pub(in crate::compute) fn release_pooled_buffers(&self) -> Result<(), Error> {
        self.surface_outputs.release()?;
        self.batch_scratch.release_idle()
    }

    #[cfg(test)]
    pub(in crate::compute) fn pooled_surface_outputs_for_test(&self) -> usize {
        self.surface_outputs.pooled_count_for_test()
    }

    #[cfg(test)]
    pub(in crate::compute) fn batch_scratch_in_use_for_test(&self) -> bool {
        self.batch_scratch.in_use()
    }

    pub(in crate::compute) fn viewport_plane_cache(
        &self,
    ) -> Result<MutexGuard<'_, Option<CachedViewportPlanes>>, Error> {
        self.viewport_plane_cache
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "JPEG Metal viewport plane cache",
            })
    }

    pub(in crate::compute) fn viewport_plane_cache_lease(
        &self,
    ) -> Result<ViewportPlaneCacheLease, Error> {
        self.viewport_plane_cache_gate.acquire()
    }

    #[cfg(test)]
    pub(in crate::compute) fn viewport_plane_cache_id_for_test(
        &self,
    ) -> Result<Option<usize>, Error> {
        Ok(self
            .viewport_plane_cache()?
            .as_ref()
            .map(|cached| objc2::rc::Retained::as_ptr(&cached.plane0).cast::<()>() as usize))
    }
}

pub(in crate::compute) fn with_runtime<R>(
    operation: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    DEFAULT_METAL_SESSION.with(|session| {
        let mut session = session.borrow_mut();
        if session.is_none() {
            *session = Some(system_default_device().map(crate::MetalBackendSession::new));
        }
        let Some(session) = session.as_ref() else {
            return Err(Error::MetalRuntime {
                message: "JPEG Metal default session was not initialized".to_string(),
            });
        };
        match session {
            Ok(session) => with_runtime_for_session(session, operation),
            Err(error) => Err(runtime_initialization_error(error)),
        }
    })
}

pub(in crate::compute) fn with_runtime_for_session<R>(
    session: &crate::MetalBackendSession,
    operation: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    match session.runtime_result() {
        Ok(runtime) => operation(runtime),
        Err(error) => Err(runtime_initialization_error(error)),
    }
}

pub(crate) fn runtime_initialization_error(error: &MetalSupportError) -> Error {
    metal_runtime_support_error(error)
}

pub(in crate::compute) struct FastRgbDecodeBuffer {
    pub(in crate::compute) buffer: Buffer,
    pub(in crate::compute) dimensions: (u32, u32),
    pub(in crate::compute) status_buffer: Buffer,
    pub(in crate::compute) command_buffer: CommandBuffer,
}

pub(in crate::compute) fn private_jpeg_tile_from_fast_rgb_buffer(
    decoded: FastRgbDecodeBuffer,
) -> Result<crate::ResidentPrivateJpegTile, Error> {
    crate::ResidentPrivateJpegTile::new(
        decoded.buffer,
        0,
        decoded.dimensions,
        PixelFormat::Rgb8,
        decoded.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel(),
        decoded.status_buffer,
        decoded.command_buffer,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_forgets_pooled_outputs_and_keeps_leased_scratch() {
        use objc2_metal::MTLBuffer as _;

        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let runtime = MetalRuntime::new().expect("runtime");
        let output = runtime
            .surface_output_buffer(2 * 1024 * 1024)
            .expect("pooled output");
        let kept_output = output.clone();
        drop(output);
        assert_eq!(runtime.pooled_surface_outputs_for_test(), 1);

        let mut lease = runtime.batch_scratch().expect("scratch lease");
        let status = lease
            .shared_buffer_with_bytes(&runtime.device, "release status", &[7, 9])
            .expect("leased status");
        runtime.release_pooled_buffers().expect("release");
        assert_eq!(runtime.pooled_surface_outputs_for_test(), 0);
        // A buffer the caller still holds survives the release.
        assert!(kept_output.length() >= 2 * 1024 * 1024);
        // Leased scratch is untouched and returns to its slot on drop.
        assert!(runtime.batch_scratch_in_use_for_test());
        let bytes = crate::buffers::checked_buffer_slice::<u8>(&status, 2, "leased status")
            .expect("read leased status");
        assert_eq!(bytes, [7, 9]);
        drop(lease);
        runtime
            .release_pooled_buffers()
            .expect("release idle scratch");
        assert!(!runtime.batch_scratch_in_use_for_test());
    }

    #[test]
    fn scratch_slots_allow_two_independent_batches() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let runtime = Arc::new(MetalRuntime::new().expect("runtime"));
        let mut first = runtime.batch_scratch().expect("first lease");
        let first_buffer = first
            .shared_buffer_with_bytes(&runtime.device, "concurrent status", &[1, 2])
            .expect("first status");
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let other_runtime = Arc::clone(&runtime);
        let worker = std::thread::spawn(move || {
            let mut second = other_runtime.batch_scratch().expect("second lease");
            let buffer = second
                .shared_buffer_with_bytes(&other_runtime.device, "concurrent status", &[3, 4])
                .expect("second status");
            ready_tx
                .send(objc2::rc::Retained::as_ptr(&buffer).cast::<()>() as usize)
                .expect("ready");
            let _ = release_rx.recv();
        });
        let concurrent_buffer = ready_rx.recv_timeout(std::time::Duration::from_secs(1));
        let first_bytes =
            crate::buffers::checked_buffer_slice::<u8>(&first_buffer, 2, "first status")
                .expect("read own status");
        drop(first);
        let _ = release_tx.send(());
        worker.join().expect("worker");
        assert_ne!(
            concurrent_buffer.expect(
                "second batch must acquire independent scratch while the first is in flight"
            ),
            objc2::rc::Retained::as_ptr(&first_buffer).cast::<()>() as usize
        );
        assert_eq!(
            first_bytes,
            [1, 2],
            "concurrent staging must not overwrite the first batch"
        );
    }

    #[test]
    fn scratch_slots_apply_backpressure_when_both_are_leased() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let runtime = Arc::new(MetalRuntime::new().expect("runtime"));
        let first = runtime.batch_scratch().expect("first lease");
        let second = runtime.batch_scratch().expect("second lease");
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let other = Arc::clone(&runtime);
        let worker = std::thread::spawn(move || {
            started_tx.send(()).expect("started");
            let _third = other.batch_scratch().expect("third lease");
            ready_tx.send(()).expect("ready");
        });
        started_rx.recv().expect("worker started");
        let blocked = ready_rx.recv_timeout(std::time::Duration::from_millis(100));
        drop(second);
        let progressed = ready_rx.recv_timeout(std::time::Duration::from_secs(1));
        drop(first);
        worker.join().expect("worker");
        progressed
            .expect("a waiter must reuse either released slot, even while slot zero stays busy");
        assert!(matches!(
            blocked,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ));
    }
}
