// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

/// Metal command-encoder kind used in typed construction errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetalCommandEncoderKind {
    /// Compute command encoder.
    Compute,
    /// Blit command encoder.
    Blit,
}

impl fmt::Display for MetalCommandEncoderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compute => f.write_str("compute"),
            Self::Blit => f.write_str("blit"),
        }
    }
}

/// Errors returned by shared Metal runtime setup helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetalSupportError {
    /// The host does not expose a system default Metal device.
    MetalUnavailable,
    /// Metal returned a null command queue for the selected device.
    CommandQueueUnavailable,
    /// Metal command queue creation failed before returning a queue.
    CommandQueue {
        /// Error reported by the Objective-C message send.
        message: String,
    },
    /// Metal returned a null command buffer for an available command queue.
    CommandBufferUnavailable,
    /// Objective-C dispatch failed while requesting a Metal command buffer.
    CommandBufferCreation {
        /// Error reported by the Objective-C message send.
        message: String,
    },
    /// Metal returned a null command encoder for an available command buffer.
    CommandEncoderUnavailable {
        /// Kind of encoder requested.
        kind: MetalCommandEncoderKind,
    },
    /// Objective-C dispatch failed while requesting a Metal command encoder.
    CommandEncoderCreation {
        /// Kind of encoder requested.
        kind: MetalCommandEncoderKind,
        /// Error reported by the Objective-C message send.
        message: String,
    },
    /// A committed Metal command buffer did not complete successfully.
    CommandBuffer {
        /// Best-effort label assigned to the command buffer.
        label: String,
        /// Final command-buffer status reported by Metal.
        status: String,
    },
    /// Metal shader source compilation failed.
    ShaderLibrary {
        /// Compiler error reported by Metal.
        message: String,
    },
    /// A named function was not present in a compiled Metal library.
    PipelineFunction {
        /// Requested Metal function name.
        function_name: String,
        /// Error reported by Metal.
        message: String,
    },
    /// Compute pipeline creation failed for a Metal function.
    PipelineState {
        /// Requested Metal function name.
        function_name: String,
        /// Error reported by Metal.
        message: String,
    },
    /// A requested CPU-visible buffer range does not fit in the Metal buffer.
    BufferBounds {
        /// Byte offset requested by the caller.
        offset_bytes: usize,
        /// Number of bytes requested by the caller.
        byte_len: usize,
        /// Metal buffer length in bytes.
        buffer_len: usize,
    },
    /// Metal image layout metadata is internally inconsistent or overflows.
    MetalImageLayout {
        /// Stable description of the invalid layout property.
        reason: &'static str,
    },
    /// A resident image was used with a different Metal device.
    MetalImageDeviceMismatch {
        /// Registry identifier recorded by the resident image.
        image_registry_id: u64,
        /// Registry identifier of the requested Metal device.
        requested_registry_id: u64,
    },
    /// A requested typed buffer view is not aligned for the element type.
    BufferAlignment {
        /// Byte offset requested by the caller.
        offset_bytes: usize,
        /// Required alignment in bytes.
        align: usize,
    },
    /// A zero-sized Rust type cannot describe a Metal buffer element ABI.
    BufferZeroSizedType {
        /// Human-readable ABI name supplied by [`j2k_core::accelerator::GpuAbi`].
        abi_name: &'static str,
    },
    /// Host allocation for an owned buffer readback failed.
    BufferReadbackAllocation {
        /// Human-readable ABI name supplied by [`j2k_core::accelerator::GpuAbi`].
        abi_name: &'static str,
        /// Number of elements requested by the caller.
        element_count: usize,
    },
    /// A requested Metal buffer allocation exceeds the configured/device limit.
    BufferAllocationTooLarge {
        /// Requested Metal buffer length in bytes.
        requested: usize,
        /// Maximum permitted Metal buffer length in bytes.
        cap: usize,
    },
    /// Metal rejected an otherwise valid buffer allocation by returning nil.
    BufferAllocationFailed {
        /// Requested Metal buffer length in bytes.
        requested: usize,
    },
    /// Objective-C dispatch failed while requesting a Metal buffer.
    BufferAllocation {
        /// Error reported by the Objective-C message send.
        message: String,
    },
    /// A texture descriptor has zero or otherwise unaccountable allocation geometry.
    TextureDescriptorInvalid {
        /// Stable descriptor property that failed validation.
        reason: &'static str,
    },
    /// A planned Metal texture allocation exceeds the repository resource cap.
    TextureAllocationTooLarge {
        /// Planned texture allocation bytes.
        requested: usize,
        /// Maximum permitted bytes for one texture allocation.
        cap: usize,
    },
    /// Metal rejected a texture allocation by returning nil.
    TextureAllocationFailed {
        /// Requested texture width.
        width: u64,
        /// Requested texture height.
        height: u64,
        /// Requested texture depth.
        depth: u64,
        /// Requested texture array length.
        array_length: u64,
    },
    /// Objective-C dispatch failed while requesting a Metal texture.
    TextureAllocation {
        /// Error reported by the Objective-C message send.
        message: String,
    },
    /// Metal returned no texture descriptor object.
    TextureDescriptorUnavailable,
    /// The Metal buffer is not CPU-visible through `contents()`.
    BufferContentsUnavailable,
}

impl MetalSupportError {
    /// Return true when the error represents an unavailable Metal backend.
    #[must_use]
    pub const fn is_unavailable(&self) -> bool {
        matches!(self, Self::MetalUnavailable | Self::CommandQueueUnavailable)
    }
}

impl fmt::Display for MetalSupportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetalUnavailable => f.write_str("Metal is unavailable on this host"),
            Self::CommandQueueUnavailable => f.write_str("Metal command queue is unavailable"),
            Self::CommandQueue { message } => {
                write!(f, "Metal command queue creation failed: {message}")
            }
            Self::CommandBufferUnavailable => f.write_str("Metal command buffer is unavailable"),
            Self::CommandBufferCreation { message } => {
                write!(f, "Metal command buffer creation failed: {message}")
            }
            Self::CommandEncoderUnavailable { kind } => {
                write!(f, "Metal {kind} command encoder is unavailable")
            }
            Self::CommandEncoderCreation { kind, message } => {
                write!(f, "Metal {kind} command encoder creation failed: {message}")
            }
            Self::CommandBuffer { label, status } => {
                write!(f, "Metal command buffer `{label}` completed with status {status}")
            }
            Self::ShaderLibrary { message } => {
                write!(f, "Metal shader library compilation failed: {message}")
            }
            Self::PipelineFunction {
                function_name,
                message,
            } => write!(
                f,
                "Metal pipeline function `{function_name}` lookup failed: {message}"
            ),
            Self::PipelineState {
                function_name,
                message,
            } => write!(
                f,
                "Metal compute pipeline `{function_name}` creation failed: {message}"
            ),
            Self::BufferBounds {
                offset_bytes,
                byte_len,
                buffer_len,
            } => write!(
                f,
                "Metal buffer range offset {offset_bytes} length {byte_len} exceeds buffer length {buffer_len}"
            ),
            Self::MetalImageLayout { reason } => write!(f, "invalid Metal image layout: {reason}"),
            Self::MetalImageDeviceMismatch {
                image_registry_id,
                requested_registry_id,
            } => write!(
                f,
                "Metal image device {image_registry_id} does not match requested device {requested_registry_id}"
            ),
            Self::BufferAlignment {
                offset_bytes,
                align,
            } => write!(
                f,
                "Metal buffer range offset {offset_bytes} is not aligned to {align} bytes"
            ),
            Self::BufferZeroSizedType { abi_name } => {
                write!(f, "Metal buffer ABI type `{abi_name}` is zero-sized")
            }
            Self::BufferReadbackAllocation {
                abi_name,
                element_count,
            } => write!(
                f,
                "Metal buffer readback allocation failed for {element_count} `{abi_name}` values"
            ),
            Self::BufferAllocationTooLarge { requested, cap } => write!(
                f,
                "Metal buffer allocation of {requested} bytes exceeds limit {cap}"
            ),
            Self::BufferAllocationFailed { requested } => {
                write!(f, "Metal buffer allocation failed for {requested} bytes")
            }
            Self::BufferAllocation { message } => {
                write!(f, "Metal buffer allocation dispatch failed: {message}")
            }
            Self::TextureDescriptorInvalid { reason } => write!(f, "invalid texture: {reason}"),
            Self::TextureAllocationTooLarge { requested, cap } => write!(
                f,
                "Metal texture allocation of {requested} bytes exceeds limit {cap}"
            ),
            Self::TextureAllocationFailed {
                width,
                height,
                depth,
                array_length,
            } => write!(
                f,
                "Metal texture allocation failed for {width}x{height}x{depth}, array length {array_length}"
            ),
            Self::TextureAllocation { message } => {
                write!(f, "Metal texture allocation dispatch failed: {message}")
            }
            Self::TextureDescriptorUnavailable => f.write_str("Metal texture is unavailable"),
            Self::BufferContentsUnavailable => f.write_str("Metal buffer is not CPU-visible"),
        }
    }
}

impl std::error::Error for MetalSupportError {}
