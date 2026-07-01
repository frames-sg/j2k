// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared Metal runtime setup helpers for J2K adapter crates.

#![warn(unreachable_pub)]

use core::fmt;

/// Stable profile labels for a Metal backend route decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalRouteProfileLabels {
    /// Route decision label emitted in GPU route profiles.
    pub decision: &'static str,
    /// Route reason label emitted in GPU route profiles.
    pub reason: &'static str,
}

impl MetalRouteProfileLabels {
    /// Construct route profile labels from stable string values.
    #[must_use]
    pub const fn new(decision: &'static str, reason: &'static str) -> Self {
        Self { decision, reason }
    }
}

/// Route profile labels for CPU host execution.
#[must_use]
pub const fn cpu_host_route() -> MetalRouteProfileLabels {
    MetalRouteProfileLabels::new("cpu_host", "none")
}

/// Route profile labels for Metal kernel execution.
#[must_use]
pub const fn metal_kernel_route() -> MetalRouteProfileLabels {
    MetalRouteProfileLabels::new("metal_kernel", "none")
}

/// Route profile labels for an explicit Metal request rejected by codec policy.
#[must_use]
pub const fn reject_explicit_metal_route(reason: &'static str) -> MetalRouteProfileLabels {
    MetalRouteProfileLabels::new("reject_explicit_metal", reason)
}

/// Route profile labels for backend requests unsupported by the Metal adapter.
#[must_use]
pub const fn reject_unsupported_backend_route() -> MetalRouteProfileLabels {
    MetalRouteProfileLabels::new("reject_unsupported_backend", "unsupported_backend")
}

/// Route profile labels for hosts without an available Metal runtime.
#[must_use]
pub const fn metal_unavailable_route() -> MetalRouteProfileLabels {
    MetalRouteProfileLabels::new("metal_unavailable", "metal_unavailable")
}

/// Errors returned by shared Metal runtime setup helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// A requested typed buffer view is not aligned for the element type.
    BufferAlignment {
        /// Byte offset requested by the caller.
        offset_bytes: usize,
        /// Required alignment in bytes.
        align: usize,
    },
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
            Self::CommandQueueUnavailable => {
                f.write_str("Metal command queue is unavailable on this host")
            }
            Self::CommandQueue { message } => {
                write!(f, "Metal command queue creation failed: {message}")
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
            Self::BufferAlignment {
                offset_bytes,
                align,
            } => write!(
                f,
                "Metal buffer range offset {offset_bytes} is not aligned to {align} bytes"
            ),
            Self::BufferContentsUnavailable => {
                f.write_str("Metal buffer contents are not CPU-visible")
            }
        }
    }
}

impl std::error::Error for MetalSupportError {}

#[cfg(target_os = "macos")]
use j2k_core::GpuAbi;
#[cfg(target_os = "macos")]
use metal::{
    foreign_types::ForeignType,
    objc::{runtime::Sel, Message},
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
    ComputePipelineState, Device, Library, MTLCommandBufferStatus, MTLCommandQueue,
    MTLResourceOptions, MTLSize,
};

#[cfg(target_os = "macos")]
/// Return the system default Metal device, or a stable error message.
pub fn system_default_device() -> Result<Device, MetalSupportError> {
    Device::system_default().ok_or(MetalSupportError::MetalUnavailable)
}

#[cfg(target_os = "macos")]
/// Create a command queue and surface null-queue failures explicitly.
pub fn checked_command_queue(device: &Device) -> Result<CommandQueue, MetalSupportError> {
    // SAFETY: Objective-C/Metal pointers are null-checked or range-validated before wrapping.
    let queue: *mut MTLCommandQueue = unsafe {
        device
            .as_ref()
            .send_message(Sel::register("newCommandQueue"), ())
            .map_err(|error| MetalSupportError::CommandQueue {
                message: error.to_string(),
            })?
    };
    if queue.is_null() {
        Err(MetalSupportError::CommandQueueUnavailable)
    } else {
        // SAFETY: Objective-C/Metal pointers are null-checked or range-validated before wrapping.
        Ok(unsafe { CommandQueue::from_ptr(queue) })
    }
}

#[cfg(target_os = "macos")]
/// Commit a command buffer, wait for completion, and surface failed completion.
pub fn commit_and_wait(command_buffer: &CommandBufferRef) -> Result<(), MetalSupportError> {
    command_buffer.commit();
    wait_for_completion(command_buffer)
}

#[cfg(target_os = "macos")]
/// Wait for an already committed command buffer and surface failed completion.
pub fn wait_for_completion(command_buffer: &CommandBufferRef) -> Result<(), MetalSupportError> {
    command_buffer.wait_until_completed();
    ensure_completed(command_buffer)
}

#[cfg(target_os = "macos")]
/// Surface a failed command buffer after the caller has already synchronized it.
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

#[cfg(target_os = "macos")]
/// Compile a Metal shader source string with default compile options.
pub fn shader_library(device: &Device, source: &str) -> Result<Library, MetalSupportError> {
    let options = CompileOptions::new();
    device
        .new_library_with_source(source, &options)
        .map_err(|message| MetalSupportError::ShaderLibrary { message })
}

#[cfg(target_os = "macos")]
/// Load a named compute pipeline from an already compiled shader library.
pub fn named_pipeline(
    device: &Device,
    library: &Library,
    function_name: &str,
) -> Result<ComputePipelineState, MetalSupportError> {
    let function = library
        .get_function(function_name, None)
        .map_err(|message| MetalSupportError::PipelineFunction {
            function_name: function_name.to_string(),
            message,
        })?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| MetalSupportError::PipelineState {
            function_name: function_name.to_string(),
            message,
        })
}

#[cfg(target_os = "macos")]
/// Caller-owned Metal device session shared by adapter crates.
#[derive(Clone)]
pub struct MetalDeviceSession {
    device: Device,
}

#[cfg(target_os = "macos")]
impl MetalDeviceSession {
    /// Create a session bound to an existing Metal device.
    #[must_use]
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Create a session from the system default Metal device.
    pub fn system_default() -> Result<Self, MetalSupportError> {
        system_default_device().map(Self::new)
    }

    /// Metal device used by this session.
    #[must_use]
    pub fn device(&self) -> &metal::DeviceRef {
        self.device.as_ref()
    }
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for MetalDeviceSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalDeviceSession")
            .field("device", &self.device.name())
            .finish()
    }
}

#[cfg(target_os = "macos")]
/// Allocate a shared Metal buffer, clamping zero-length requests to one byte.
#[must_use]
pub fn shared_buffer(device: &Device, bytes: usize) -> Buffer {
    device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared)
}

#[cfg(target_os = "macos")]
/// Allocate a private Metal buffer, clamping zero-length requests to one byte.
#[must_use]
pub fn private_buffer(device: &Device, bytes: usize) -> Buffer {
    device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModePrivate)
}

#[cfg(target_os = "macos")]
/// Allocate a shared Metal buffer initialized from bytes.
#[must_use]
pub fn shared_buffer_with_bytes(device: &Device, bytes: &[u8]) -> Buffer {
    if bytes.is_empty() {
        return shared_buffer(device, 1);
    }
    device.new_buffer_with_data(
        bytes.as_ptr().cast(),
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

#[cfg(target_os = "macos")]
/// Allocate a shared Metal buffer initialized from GPU ABI values.
#[must_use]
pub fn shared_buffer_with_slice<T: GpuAbi>(device: &Device, values: &[T]) -> Buffer {
    shared_buffer_with_bytes(device, T::slice_as_bytes(values))
}

#[cfg(target_os = "macos")]
/// Allocate a shared Metal buffer large enough for `len` GPU ABI values.
#[must_use]
pub fn shared_buffer_for_len<T: GpuAbi>(device: &Device, len: usize) -> Buffer {
    shared_buffer(device, len.saturating_mul(core::mem::size_of::<T>()))
}

#[cfg(target_os = "macos")]
fn checked_buffer_contents_ptr<T: GpuAbi>(
    buffer: &Buffer,
    offset_bytes: usize,
    len: usize,
) -> Result<*mut T, MetalSupportError> {
    let buffer_len = usize::try_from(buffer.length()).unwrap_or(usize::MAX);
    let byte_len =
        len.checked_mul(core::mem::size_of::<T>())
            .ok_or(MetalSupportError::BufferBounds {
                offset_bytes,
                byte_len: usize::MAX,
                buffer_len,
            })?;
    let end = offset_bytes
        .checked_add(byte_len)
        .ok_or(MetalSupportError::BufferBounds {
            offset_bytes,
            byte_len,
            buffer_len,
        })?;
    if end > buffer_len {
        return Err(MetalSupportError::BufferBounds {
            offset_bytes,
            byte_len,
            buffer_len,
        });
    }

    let base = buffer.contents().cast::<u8>();
    if base.is_null() {
        return Err(MetalSupportError::BufferContentsUnavailable);
    }
    let address =
        (base as usize)
            .checked_add(offset_bytes)
            .ok_or(MetalSupportError::BufferBounds {
                offset_bytes,
                byte_len,
                buffer_len,
            })?;
    let align = core::mem::align_of::<T>();
    if align > 1 && !address.is_multiple_of(align) {
        return Err(MetalSupportError::BufferAlignment {
            offset_bytes,
            align,
        });
    }

    // SAFETY: bounds and alignment were validated above.
    Ok(unsafe { base.add(offset_bytes).cast::<T>() })
}

#[cfg(target_os = "macos")]
/// Checked typed borrow from a CPU-visible shared Metal buffer.
///
/// Validates offset arithmetic, typed alignment, requested byte length, and the
/// Metal buffer length before constructing the returned slice.
pub fn checked_buffer_contents_slice<T: GpuAbi>(
    buffer: &Buffer,
    offset_bytes: usize,
    len: usize,
) -> Result<&[T], MetalSupportError> {
    let ptr = checked_buffer_contents_ptr::<T>(buffer, offset_bytes, len)?;
    // SAFETY: `checked_buffer_contents_ptr` validated the byte range and typed
    // alignment for `len` elements in this CPU-visible buffer.
    // SAFETY: Objective-C/Metal pointers are null-checked or range-validated before wrapping.
    Ok(unsafe { core::slice::from_raw_parts(ptr.cast_const(), len) })
}

#[cfg(target_os = "macos")]
/// Checked mutable typed borrow from a CPU-visible shared Metal buffer.
///
/// Validates offset arithmetic, typed alignment, requested byte length, and the
/// Metal buffer length before constructing the returned slice.
pub fn checked_buffer_contents_slice_mut<T: GpuAbi>(
    buffer: &mut Buffer,
    offset_bytes: usize,
    len: usize,
) -> Result<&mut [T], MetalSupportError> {
    let ptr = checked_buffer_contents_ptr::<T>(buffer, offset_bytes, len)?;
    // SAFETY: `checked_buffer_contents_ptr` validated the byte range and typed
    // alignment for `len` elements in this CPU-visible buffer. The mutable
    // borrow of `buffer` prevents another safe mutable slice from this helper.
    // SAFETY: Objective-C/Metal pointers are null-checked or range-validated before wrapping.
    Ok(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
}

#[cfg(target_os = "macos")]
/// Borrow typed contents from a shared Metal buffer.
///
/// # Safety
/// The caller must ensure the buffer is CPU-visible, contains at least
/// `offset_bytes + len * size_of::<T>()` initialized bytes, and is not mutably
/// aliased for the returned lifetime.
#[must_use]
pub unsafe fn buffer_contents_slice<T: GpuAbi>(
    buffer: &Buffer,
    offset_bytes: usize,
    len: usize,
) -> &[T] {
    checked_buffer_contents_slice(buffer, offset_bytes, len)
        .expect("Metal buffer contents slice violates unsafe API contract")
}

#[cfg(target_os = "macos")]
/// Mutably borrow typed contents from a shared Metal buffer.
///
/// # Safety
/// The caller must ensure the buffer is CPU-visible, contains at least
/// `offset_bytes + len * size_of::<T>()` initialized bytes, and no other alias
/// can read or write the same memory for the returned lifetime.
#[must_use]
pub unsafe fn buffer_contents_slice_mut<T: GpuAbi>(
    buffer: &mut Buffer,
    offset_bytes: usize,
    len: usize,
) -> &mut [T] {
    checked_buffer_contents_slice_mut(buffer, offset_bytes, len)
        .expect("Metal mutable buffer contents slice violates unsafe API contract")
}

#[cfg(target_os = "macos")]
/// Construct a Metal dispatch size.
#[must_use]
pub const fn mtl_size(width: u64, height: u64, depth: u64) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

#[cfg(target_os = "macos")]
/// One-dimensional thread-group size with empty SIMD widths clamped to one.
#[must_use]
pub const fn one_d_threads_per_group(simd_width: u64) -> MTLSize {
    mtl_size(if simd_width == 0 { 1 } else { simd_width }, 1, 1)
}

#[cfg(target_os = "macos")]
/// Two-dimensional thread-group size preserving SIMD width and filling height.
#[must_use]
pub const fn two_d_threads_per_group(simd_width: u64, max_threads: u64) -> MTLSize {
    let width = if simd_width == 0 { 1 } else { simd_width };
    let max_threads = if max_threads < width {
        width
    } else {
        max_threads
    };
    mtl_size(width, max_threads / width, 1)
}

#[cfg(target_os = "macos")]
/// Dispatch a one-dimensional compute workload with one SIMD group per threadgroup.
pub fn dispatch_1d_pipeline(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    width: u64,
) {
    encoder.dispatch_threads(
        mtl_size(width, 1, 1),
        one_d_threads_per_group(pipeline.thread_execution_width()),
    );
}

#[cfg(target_os = "macos")]
/// Dispatch a single compute thread.
pub fn dispatch_single_thread(encoder: &ComputeCommandEncoderRef) {
    encoder.dispatch_threads(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
}

#[cfg(target_os = "macos")]
/// Dispatch a two-dimensional compute workload using the pipeline's SIMD width.
pub fn dispatch_2d_pipeline(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32),
) {
    encoder.dispatch_threads(
        mtl_size(u64::from(dims.0), u64::from(dims.1), 1),
        two_d_threads_per_group(
            pipeline.thread_execution_width(),
            pipeline.max_total_threads_per_threadgroup(),
        ),
    );
}

#[cfg(target_os = "macos")]
/// Dispatch a three-dimensional compute workload using a 2D threadgroup shape.
pub fn dispatch_3d_pipeline(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32, u32),
) {
    encoder.dispatch_threads(
        mtl_size(u64::from(dims.0), u64::from(dims.1), u64::from(dims.2)),
        two_d_threads_per_group(
            pipeline.thread_execution_width(),
            pipeline.max_total_threads_per_threadgroup(),
        ),
    );
}

#[cfg(target_os = "macos")]
/// Convenience loader for many pipelines from one Metal shader library.
pub struct MetalPipelineLoader {
    device: Device,
    library: Library,
}

#[cfg(target_os = "macos")]
impl MetalPipelineLoader {
    /// Compile `source` and keep the resulting library for named pipeline loads.
    pub fn new(device: &Device, source: &str) -> Result<Self, MetalSupportError> {
        Ok(Self {
            device: device.clone(),
            library: shader_library(device, source)?,
        })
    }

    /// Load one named compute pipeline from the cached shader library.
    pub fn pipeline(&self, function_name: &str) -> Result<ComputePipelineState, MetalSupportError> {
        named_pipeline(&self.device, &self.library, function_name)
    }

    /// Borrow the compiled shader library.
    #[must_use]
    pub fn library(&self) -> &Library {
        &self.library
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{
        checked_buffer_contents_slice, checked_buffer_contents_slice_mut, checked_command_queue,
        commit_and_wait, one_d_threads_per_group, shared_buffer_for_len, shared_buffer_with_slice,
        system_default_device, two_d_threads_per_group, MetalSupportError,
    };

    #[test]
    fn two_d_threads_per_group_clamps_empty_pipeline_limits() {
        let threads = two_d_threads_per_group(0, 0);

        assert_eq!((threads.width, threads.height, threads.depth), (1, 1, 1));
    }

    #[test]
    fn one_d_threads_per_group_clamps_empty_pipeline_width() {
        let threads = one_d_threads_per_group(0);

        assert_eq!((threads.width, threads.height, threads.depth), (1, 1, 1));
    }

    #[test]
    fn two_d_threads_per_group_preserves_simd_width_and_derives_height() {
        let threads = two_d_threads_per_group(32, 1024);

        assert_eq!((threads.width, threads.height, threads.depth), (32, 32, 1));
    }

    #[test]
    fn commit_and_wait_accepts_unlabeled_command_buffer() {
        let Ok(device) = system_default_device() else {
            eprintln!("skipping command buffer completion test: no Metal device");
            return;
        };
        let queue = checked_command_queue(&device).expect("Metal command queue");
        let command_buffer = queue.new_command_buffer();

        commit_and_wait(command_buffer).expect("unlabeled command buffer completion");
    }

    #[test]
    fn buffer_contents_slice_reads_typed_shared_buffer_values() {
        let Ok(device) = system_default_device() else {
            eprintln!("skipping shared buffer slice test: no Metal device");
            return;
        };
        let buffer = shared_buffer_with_slice(&device, &[3_u32, 5, 8, 13]);

        let values = checked_buffer_contents_slice::<u32>(&buffer, 0, 4).expect("checked slice");

        assert_eq!(values, &[3, 5, 8, 13]);
    }

    #[test]
    fn buffer_contents_slice_mut_writes_typed_shared_buffer_values() {
        let Ok(device) = system_default_device() else {
            eprintln!("skipping mutable shared buffer slice test: no Metal device");
            return;
        };
        let mut buffer = shared_buffer_for_len::<u32>(&device, 3);

        let values =
            checked_buffer_contents_slice_mut::<u32>(&mut buffer, 0, 3).expect("checked slice");
        values.copy_from_slice(&[21, 34, 55]);
        let values = checked_buffer_contents_slice::<u32>(&buffer, 0, 3).expect("checked slice");

        assert_eq!(values, &[21, 34, 55]);
    }

    #[test]
    fn checked_buffer_contents_slice_rejects_out_of_bounds_range() {
        let Ok(device) = system_default_device() else {
            eprintln!("skipping shared buffer bounds test: no Metal device");
            return;
        };
        let buffer = shared_buffer_with_slice(&device, &[1_u32]);

        let err = checked_buffer_contents_slice::<u32>(&buffer, 0, 2).expect_err("bounds error");

        assert!(matches!(
            err,
            MetalSupportError::BufferBounds {
                offset_bytes: 0,
                byte_len: 8,
                buffer_len: 4,
            }
        ));
    }

    #[test]
    fn checked_buffer_contents_slice_rejects_unaligned_range() {
        let Ok(device) = system_default_device() else {
            eprintln!("skipping shared buffer alignment test: no Metal device");
            return;
        };
        let buffer = shared_buffer_with_slice(&device, &[1_u32, 2]);

        let err = checked_buffer_contents_slice::<u32>(&buffer, 1, 1).expect_err("alignment error");

        assert!(matches!(
            err,
            MetalSupportError::BufferAlignment {
                offset_bytes: 1,
                align: 4,
            }
        ));
    }
}
