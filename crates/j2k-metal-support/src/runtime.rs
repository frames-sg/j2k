// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, OnceLock};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent, MTLSharedEvent,
};

use crate::{MetalCommandEncoderKind, MetalSupportError};

type DeviceHandle = Retained<ProtocolObject<dyn MTLDevice>>;

/// Return the system default Metal device, or a stable error message.
///
/// # Errors
///
/// Returns [`MetalSupportError::MetalUnavailable`] when the host exposes no
/// default Metal device.
pub fn system_default_device() -> Result<DeviceHandle, MetalSupportError> {
    MTLCreateSystemDefaultDevice().ok_or(MetalSupportError::MetalUnavailable)
}

/// Shared lazy Metal runtime session used by backend adapter crates.
pub struct MetalRuntimeSession<R, E> {
    device: DeviceHandle,
    runtime: Arc<OnceLock<Result<R, E>>>,
}

// SAFETY: Apple Metal device objects support cross-thread use, and this type
// exposes the lazily initialized runtime only through `OnceLock`. The generic
// runtime and error must themselves be safe to share and move.
unsafe impl<R: Send + Sync, E: Send + Sync> Send for MetalRuntimeSession<R, E> {}
// SAFETY: The same device/`OnceLock` argument applies to shared references;
// initialization is serialized by `OnceLock` and `R`/`E` are `Sync`.
unsafe impl<R: Send + Sync, E: Send + Sync> Sync for MetalRuntimeSession<R, E> {}

impl<R, E> Clone for MetalRuntimeSession<R, E> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            runtime: Arc::clone(&self.runtime),
        }
    }
}

impl<R, E> MetalRuntimeSession<R, E> {
    /// Create a session bound to an existing Metal device.
    #[must_use]
    pub fn new(device: DeviceHandle) -> Self {
        Self {
            device,
            runtime: Arc::new(OnceLock::new()),
        }
    }

    /// Create a session bound to the system default Metal device.
    ///
    /// # Errors
    ///
    /// Returns [`MetalSupportError::MetalUnavailable`] when the host exposes no
    /// default Metal device.
    pub fn system_default() -> Result<Self, MetalSupportError> {
        system_default_device().map(Self::new)
    }

    /// Metal device used by this session.
    #[must_use]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Metal device handle used when constructing a crate-specific runtime.
    #[must_use]
    pub fn device_handle(&self) -> &DeviceHandle {
        &self.device
    }

    /// Return whether the lazy runtime has been initialized.
    #[must_use]
    pub fn runtime_initialized(&self) -> bool {
        self.runtime.get().is_some()
    }

    /// Return the initialized runtime result, if runtime construction has run.
    #[must_use]
    pub fn runtime_result(&self) -> Option<&Result<R, E>> {
        self.runtime.get()
    }

    /// Initialize or reuse the crate-specific runtime for this Metal device.
    pub fn get_or_init_runtime(
        &self,
        init: impl FnOnce(&DeviceHandle) -> Result<R, E>,
    ) -> &Result<R, E> {
        self.runtime.get_or_init(|| init(&self.device))
    }
}

/// Create a command queue and surface null-queue failures explicitly.
///
/// # Errors
///
/// Returns [`MetalSupportError::CommandQueueUnavailable`] when Metal returns
/// no command queue for the selected device.
pub fn checked_command_queue(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLCommandQueue>>, MetalSupportError> {
    device
        .newCommandQueue()
        .ok_or(MetalSupportError::CommandQueueUnavailable)
}

/// Create a command buffer and reject a null Objective-C result.
///
/// # Errors
///
/// Returns a typed construction error when Objective-C dispatch fails or Metal
/// returns nil.
pub fn checked_command_buffer(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, MetalSupportError> {
    // `commandBuffer` is the retaining variant. objc2 retains the autoreleased
    // result before returning it, so it remains valid after a pool drains.
    queue
        .commandBuffer()
        .ok_or(MetalSupportError::CommandBufferUnavailable)
}

/// Create a compute command encoder and reject a null Objective-C result.
///
/// # Errors
///
/// Returns a typed construction error when Objective-C dispatch fails or Metal
/// returns nil.
pub fn checked_compute_command_encoder(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, MetalSupportError> {
    command_buffer
        .computeCommandEncoder()
        .ok_or(MetalSupportError::CommandEncoderUnavailable {
            kind: MetalCommandEncoderKind::Compute,
        })
}

/// Create a blit command encoder and reject a null Objective-C result.
///
/// # Errors
///
/// Returns a typed construction error when Objective-C dispatch fails or Metal
/// returns nil.
pub fn checked_blit_command_encoder(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>, MetalSupportError> {
    command_buffer
        .blitCommandEncoder()
        .ok_or(MetalSupportError::CommandEncoderUnavailable {
            kind: MetalCommandEncoderKind::Blit,
        })
}

/// Create a single-device event and reject a nil Objective-C result.
///
/// # Errors
///
/// Returns [`MetalSupportError::EventUnavailable`] when Metal cannot create
/// the event.
pub fn checked_event(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLEvent>>, MetalSupportError> {
    device.newEvent().ok_or(MetalSupportError::EventUnavailable)
}

/// Create a shared event and reject a nil Objective-C result.
///
/// # Errors
///
/// Returns [`MetalSupportError::SharedEventUnavailable`] when Metal cannot
/// create the event.
pub fn checked_shared_event(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLSharedEvent>>, MetalSupportError> {
    device
        .newSharedEvent()
        .ok_or(MetalSupportError::SharedEventUnavailable)
}

/// Commit a command buffer, wait for completion, and surface failed completion.
///
/// # Errors
///
/// Returns [`MetalSupportError::CommandBuffer`] when Metal does not report a
/// successful final status.
pub fn commit_and_wait(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<(), MetalSupportError> {
    command_buffer.commit();
    wait_for_completion(command_buffer)
}

/// Wait for an already committed command buffer and surface failed completion.
///
/// # Errors
///
/// Returns [`MetalSupportError::CommandBuffer`] when Metal does not report a
/// successful final status.
pub fn wait_for_completion(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<(), MetalSupportError> {
    command_buffer.waitUntilCompleted();
    ensure_completed(command_buffer)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandBufferCompletion {
    Completed,
    Incomplete,
    Failed,
}

pub(crate) const fn classify_command_buffer_status(
    status: MTLCommandBufferStatus,
) -> CommandBufferCompletion {
    match status {
        MTLCommandBufferStatus::Completed => CommandBufferCompletion::Completed,
        MTLCommandBufferStatus::NotEnqueued
        | MTLCommandBufferStatus::Enqueued
        | MTLCommandBufferStatus::Committed
        | MTLCommandBufferStatus::Scheduled => CommandBufferCompletion::Incomplete,
        // `Error` and any future status value fail closed.
        _ => CommandBufferCompletion::Failed,
    }
}

/// Surface a failed command buffer after the caller has already synchronized it.
///
/// # Errors
///
/// Returns [`MetalSupportError::CommandBuffer`] unless the final status is
/// [`MTLCommandBufferStatus::Completed`].
pub fn ensure_completed(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<(), MetalSupportError> {
    let status = command_buffer.status();
    match classify_command_buffer_status(status) {
        CommandBufferCompletion::Completed => Ok(()),
        CommandBufferCompletion::Incomplete | CommandBufferCompletion::Failed => {
            Err(MetalSupportError::CommandBuffer {
                label: "unlabeled".to_string(),
                status: format!("{status:?}"),
            })
        }
    }
}
