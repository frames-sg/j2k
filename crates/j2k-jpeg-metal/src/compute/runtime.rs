// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal device/session lifecycle and mutable runtime cache ownership.

use std::{
    cell::RefCell,
    sync::{Arc, Mutex, MutexGuard},
};

use super::pipeline_registry::JpegPipelineRegistry;
use super::viewport_cache::{
    CachedViewportPlanes, ViewportPlaneCacheGate, ViewportPlaneCacheLease,
};
use crate::buffers::MetalBatchScratch;
use crate::error::{metal_runtime_support_error, Error};
use crate::metal_types::{Buffer, CommandBuffer, CommandQueue, Device};
use j2k_core::PixelFormat;
use j2k_metal_support::{checked_command_queue, system_default_device, MetalSupportError};

thread_local! {
    static DEFAULT_METAL_SESSION: RefCell<Option<Result<crate::MetalBackendSession, MetalSupportError>>> = const { RefCell::new(None) };
}

pub(crate) struct MetalRuntime {
    pub(in crate::compute) device: Device,
    pub(in crate::compute) queue: CommandQueue,
    pub(in crate::compute) pipelines: JpegPipelineRegistry,
    batch_scratch: Mutex<MetalBatchScratch>,
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

    pub(crate) fn new_with_device(device: Device) -> Result<Self, MetalSupportError> {
        let pipelines = JpegPipelineRegistry::load(&device)?;
        let queue = checked_command_queue(&device)?;
        Ok(Self {
            device,
            queue,
            pipelines,
            batch_scratch: Mutex::new(MetalBatchScratch::default()),
            viewport_plane_cache: Mutex::new(None),
            viewport_plane_cache_gate: ViewportPlaneCacheGate::new(),
        })
    }

    pub(in crate::compute) fn batch_scratch(
        &self,
    ) -> Result<MutexGuard<'_, MetalBatchScratch>, Error> {
        self.batch_scratch
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "JPEG Metal batch scratch",
            })
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
