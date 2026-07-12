// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, OnceLock};

use metal::{
    foreign_types::{ForeignType, ForeignTypeRef},
    objc::{runtime::Sel, Message},
    BlitCommandEncoder, BlitCommandEncoderRef, CommandBuffer, CommandBufferRef, CommandQueue,
    CommandQueueRef, ComputeCommandEncoder, ComputeCommandEncoderRef, Device, DeviceRef,
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue,
    MTLComputeCommandEncoder,
};

use crate::{MetalCommandEncoderKind, MetalSupportError};

/// Return the system default Metal device, or a stable error message.
///
/// # Errors
///
/// Returns [`MetalSupportError::MetalUnavailable`] when the host exposes no
/// default Metal device.
pub fn system_default_device() -> Result<Device, MetalSupportError> {
    Device::system_default().ok_or(MetalSupportError::MetalUnavailable)
}

/// Shared lazy Metal runtime session used by backend adapter crates.
pub struct MetalRuntimeSession<R, E> {
    device: Device,
    runtime: Arc<OnceLock<Result<R, E>>>,
}

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
    pub fn new(device: Device) -> Self {
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
    pub fn device(&self) -> &DeviceRef {
        self.device.as_ref()
    }

    /// Metal device handle used when constructing a crate-specific runtime.
    #[must_use]
    pub fn device_handle(&self) -> &Device {
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
    pub fn get_or_init_runtime(&self, init: impl FnOnce(&Device) -> Result<R, E>) -> &Result<R, E> {
        self.runtime.get_or_init(|| init(&self.device))
    }
}

pub(crate) unsafe fn checked_command_queue_from_retained_ptr(
    raw: *mut MTLCommandQueue,
) -> Result<CommandQueue, MetalSupportError> {
    if raw.is_null() {
        Err(MetalSupportError::CommandQueueUnavailable)
    } else {
        // SAFETY: The caller guarantees that a non-null pointer is a retained
        // MTLCommandQueue result whose ownership transfers here.
        Ok(unsafe { CommandQueue::from_ptr(raw) })
    }
}

/// Create a command queue and surface null-queue failures explicitly.
///
/// # Errors
///
/// Returns [`MetalSupportError::CommandQueue`] if Objective-C queue creation
/// fails, or [`MetalSupportError::CommandQueueUnavailable`] for a null queue.
pub fn checked_command_queue(device: &DeviceRef) -> Result<CommandQueue, MetalSupportError> {
    // SAFETY: The selector takes no arguments and the retained result is
    // checked before ownership is transferred to a Rust handle.
    let queue: *mut MTLCommandQueue = unsafe {
        device
            .send_message(Sel::register("newCommandQueue"), ())
            .map_err(|error| MetalSupportError::CommandQueue {
                message: error.to_string(),
            })?
    };
    // SAFETY: `newCommandQueue` returns either nil or a retained queue pointer.
    unsafe { checked_command_queue_from_retained_ptr(queue) }
}

pub(crate) unsafe fn checked_command_buffer_from_autoreleased_ptr(
    raw: *mut MTLCommandBuffer,
) -> Result<CommandBuffer, MetalSupportError> {
    if raw.is_null() {
        Err(MetalSupportError::CommandBufferUnavailable)
    } else {
        // SAFETY: The caller guarantees that a non-null pointer is a valid
        // autoreleased MTLCommandBuffer. `to_owned` retains it before the
        // surrounding autorelease pool can drain.
        Ok(unsafe { CommandBufferRef::from_ptr(raw) }.to_owned())
    }
}

/// Create a command buffer and reject a null Objective-C result.
///
/// # Errors
///
/// Returns a typed construction error when Objective-C dispatch fails or Metal
/// returns nil.
pub fn checked_command_buffer(queue: &CommandQueueRef) -> Result<CommandBuffer, MetalSupportError> {
    // SAFETY: The selector takes no arguments and returns an autoreleased
    // MTLCommandBuffer. The raw result is checked before creating a reference.
    let raw: *mut MTLCommandBuffer = unsafe {
        queue
            .send_message(Sel::register("commandBuffer"), ())
            .map_err(|error| MetalSupportError::CommandBufferCreation {
                message: error.to_string(),
            })?
    };
    // SAFETY: `commandBuffer` returns either nil or a valid autoreleased object;
    // the helper retains it before returning an owned Rust handle.
    unsafe { checked_command_buffer_from_autoreleased_ptr(raw) }
}

pub(crate) unsafe fn checked_compute_encoder_from_autoreleased_ptr(
    raw: *mut MTLComputeCommandEncoder,
) -> Result<ComputeCommandEncoder, MetalSupportError> {
    if raw.is_null() {
        Err(MetalSupportError::CommandEncoderUnavailable {
            kind: MetalCommandEncoderKind::Compute,
        })
    } else {
        // SAFETY: The caller guarantees that a non-null pointer is a valid
        // autoreleased MTLComputeCommandEncoder. `to_owned` retains it.
        Ok(unsafe { ComputeCommandEncoderRef::from_ptr(raw) }.to_owned())
    }
}

/// Create a compute command encoder and reject a null Objective-C result.
///
/// # Errors
///
/// Returns a typed construction error when Objective-C dispatch fails or Metal
/// returns nil.
pub fn checked_compute_command_encoder(
    command_buffer: &CommandBufferRef,
) -> Result<ComputeCommandEncoder, MetalSupportError> {
    // SAFETY: The selector takes no arguments and returns an autoreleased
    // MTLComputeCommandEncoder. The raw result is checked before borrowing it.
    let raw: *mut MTLComputeCommandEncoder = unsafe {
        command_buffer
            .send_message(Sel::register("computeCommandEncoder"), ())
            .map_err(|error| MetalSupportError::CommandEncoderCreation {
                kind: MetalCommandEncoderKind::Compute,
                message: error.to_string(),
            })?
    };
    // SAFETY: `computeCommandEncoder` returns either nil or a valid
    // autoreleased object; the helper retains it before returning ownership.
    unsafe { checked_compute_encoder_from_autoreleased_ptr(raw) }
}

pub(crate) unsafe fn checked_blit_encoder_from_autoreleased_ptr(
    raw: *mut MTLBlitCommandEncoder,
) -> Result<BlitCommandEncoder, MetalSupportError> {
    if raw.is_null() {
        Err(MetalSupportError::CommandEncoderUnavailable {
            kind: MetalCommandEncoderKind::Blit,
        })
    } else {
        // SAFETY: The caller guarantees that a non-null pointer is a valid
        // autoreleased MTLBlitCommandEncoder. `to_owned` retains it.
        Ok(unsafe { BlitCommandEncoderRef::from_ptr(raw) }.to_owned())
    }
}

/// Create a blit command encoder and reject a null Objective-C result.
///
/// # Errors
///
/// Returns a typed construction error when Objective-C dispatch fails or Metal
/// returns nil.
pub fn checked_blit_command_encoder(
    command_buffer: &CommandBufferRef,
) -> Result<BlitCommandEncoder, MetalSupportError> {
    // SAFETY: The selector takes no arguments and returns an autoreleased
    // MTLBlitCommandEncoder. The raw result is checked before borrowing it.
    let raw: *mut MTLBlitCommandEncoder = unsafe {
        command_buffer
            .send_message(Sel::register("blitCommandEncoder"), ())
            .map_err(|error| MetalSupportError::CommandEncoderCreation {
                kind: MetalCommandEncoderKind::Blit,
                message: error.to_string(),
            })?
    };
    // SAFETY: `blitCommandEncoder` returns either nil or a valid autoreleased
    // object; the helper retains it before returning ownership.
    unsafe { checked_blit_encoder_from_autoreleased_ptr(raw) }
}

/// Commit a command buffer, wait for completion, and surface failed completion.
///
/// # Errors
///
/// Returns [`MetalSupportError::CommandBuffer`] when Metal does not report a
/// successful final status.
pub fn commit_and_wait(command_buffer: &CommandBufferRef) -> Result<(), MetalSupportError> {
    command_buffer.commit();
    wait_for_completion(command_buffer)
}

/// Wait for an already committed command buffer and surface failed completion.
///
/// # Errors
///
/// Returns [`MetalSupportError::CommandBuffer`] when Metal does not report a
/// successful final status.
pub fn wait_for_completion(command_buffer: &CommandBufferRef) -> Result<(), MetalSupportError> {
    command_buffer.wait_until_completed();
    ensure_completed(command_buffer)
}

/// Surface a failed command buffer after the caller has already synchronized it.
///
/// # Errors
///
/// Returns [`MetalSupportError::CommandBuffer`] unless the final status is
/// [`MTLCommandBufferStatus::Completed`].
pub fn ensure_completed(command_buffer: &CommandBufferRef) -> Result<(), MetalSupportError> {
    let status = command_buffer.status();
    if status == MTLCommandBufferStatus::Completed {
        Ok(())
    } else {
        Err(MetalSupportError::CommandBuffer {
            label: "unlabeled".to_string(),
            status: format!("{status:?}"),
        })
    }
}
