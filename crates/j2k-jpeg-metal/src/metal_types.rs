// SPDX-License-Identifier: MIT OR Apache-2.0

//! Crate-private spellings for owned and borrowed objc2 Metal objects.
//!
//! These aliases keep the host implementation readable; public expert APIs
//! spell out the corresponding objc2 protocol-object types directly.

use core::ptr::NonNull;

use j2k_core::accelerator::GpuAbi;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLTexture,
};

pub(crate) mod prelude {
    pub(crate) use super::JpegComputeEncoderExt;
    pub(crate) use objc2_metal::{MTLCommandEncoder, MTLComputeCommandEncoder};
}

pub(crate) type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub(crate) type BufferRef = ProtocolObject<dyn MTLBuffer>;
pub(crate) type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
pub(crate) type CommandBufferRef = ProtocolObject<dyn MTLCommandBuffer>;
pub(crate) type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub(crate) type CommandQueueRef = ProtocolObject<dyn MTLCommandQueue>;
pub(crate) type ComputeCommandEncoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
pub(crate) type ComputeCommandEncoderRef = ProtocolObject<dyn MTLComputeCommandEncoder>;
pub(crate) type BlitCommandEncoder = Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>;
pub(crate) type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub(crate) type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub(crate) type DeviceRef = ProtocolObject<dyn MTLDevice>;
pub(crate) type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
pub(crate) type TextureRef = ProtocolObject<dyn MTLTexture>;

/// Checked host-side binding vocabulary for J2K's fixed JPEG shader ABI.
///
/// objc2 correctly marks resource binding calls unsafe because arbitrary
/// buffer offsets, binding slots, pointers, and unretained command buffers are
/// not generally sound. J2K creates only retaining command buffers and limits
/// this trait to its private, statically matched shader layouts.
pub(crate) trait JpegComputeEncoderExt {
    fn bind_buffer(&self, index: u64, buffer: Option<&ProtocolObject<dyn MTLBuffer>>, offset: u64);

    fn bind_bytes<T: GpuAbi>(&self, index: u64, value: &T);

    fn bind_texture(&self, index: u64, texture: Option<&ProtocolObject<dyn MTLTexture>>);
}

impl JpegComputeEncoderExt for ProtocolObject<dyn MTLComputeCommandEncoder> {
    fn bind_buffer(&self, index: u64, buffer: Option<&ProtocolObject<dyn MTLBuffer>>, offset: u64) {
        let index = usize::try_from(index).expect("Metal buffer index fits usize");
        (index < 31)
            .then_some(())
            .expect("Metal buffer index exceeds the API binding table");
        let offset = usize::try_from(offset).expect("Metal buffer offset fits usize");
        if let Some(buffer) = buffer {
            (offset <= buffer.length())
                .then_some(())
                .expect("Metal buffer offset is out of bounds");
        }
        // SAFETY: The slot is checked against Metal's 31-entry buffer table,
        // the byte offset is within the allocation, and every JPEG encoder is
        // created from support's retaining command-buffer path. That path
        // keeps bound resources alive through completion; shader ABI types and
        // access ordering are fixed by the private call sites in this crate.
        unsafe { self.setBuffer_offset_atIndex(buffer, offset, index) };
    }

    fn bind_bytes<T: GpuAbi>(&self, index: u64, value: &T) {
        let index = usize::try_from(index).expect("Metal byte-binding index fits usize");
        (index < 31)
            .then_some(())
            .expect("Metal byte-binding index exceeds the API binding table");
        let bytes = T::as_bytes(value);
        (!bytes.is_empty())
            .then_some(())
            .expect("Metal byte binding requires a nonempty ABI value");
        (bytes.len() == core::mem::size_of::<T>())
            .then_some(())
            .expect("Metal byte-binding length must match its ABI value");
        let pointer = NonNull::from(bytes).cast::<core::ffi::c_void>();
        // SAFETY: `GpuAbi::as_bytes` guarantees a valid, initialized,
        // padding-free object representation for exactly `bytes.len()` bytes.
        // Metal copies those bytes synchronously, and the slot was checked.
        unsafe { self.setBytes_length_atIndex(pointer, bytes.len(), index) };
    }

    fn bind_texture(&self, index: u64, texture: Option<&ProtocolObject<dyn MTLTexture>>) {
        let index = usize::try_from(index).expect("Metal texture index fits usize");
        (index < 31)
            .then_some(())
            .expect("Metal texture index exceeds the API binding table");
        // SAFETY: The slot is checked, and JPEG uses only support-created
        // retaining command buffers, which retain the bound texture through
        // completion. Texture access ordering is owned by the submission.
        unsafe { self.setTexture_atIndex(texture, index) };
    }
}
