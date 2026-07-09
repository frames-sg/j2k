// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared Metal runtime setup helpers for J2K adapter crates.

#![warn(unreachable_pub)]

use core::fmt;

#[cfg(target_os = "macos")]
use std::sync::{Arc, OnceLock};

/// Stable profile labels for a Metal backend route decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
            Self::BufferContentsUnavailable => {
                f.write_str("Metal buffer contents are not CPU-visible")
            }
        }
    }
}

impl std::error::Error for MetalSupportError {}

#[cfg(target_os = "macos")]
use j2k_core::accelerator::GpuAbi;
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
/// Shared lazy Metal runtime session used by backend adapter crates.
pub struct MetalRuntimeSession<R, E> {
    device: Device,
    runtime: Arc<OnceLock<Result<R, E>>>,
}

#[cfg(target_os = "macos")]
impl<R, E> Clone for MetalRuntimeSession<R, E> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            runtime: Arc::clone(&self.runtime),
        }
    }
}

#[cfg(target_os = "macos")]
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
    pub fn system_default() -> Result<Self, MetalSupportError> {
        system_default_device().map(Self::new)
    }

    /// Metal device used by this session.
    #[must_use]
    pub fn device(&self) -> &metal::DeviceRef {
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
fn checked_buffer_typed_range<T: GpuAbi>(
    buffer_len: usize,
    offset_bytes: usize,
    len: usize,
) -> Result<usize, MetalSupportError> {
    let element_size = core::mem::size_of::<T>();
    if element_size == 0 {
        return Err(MetalSupportError::BufferZeroSizedType { abi_name: T::NAME });
    }
    let byte_len = len
        .checked_mul(element_size)
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

    let align = core::mem::align_of::<T>();
    if align > 1 && !offset_bytes.is_multiple_of(align) {
        return Err(MetalSupportError::BufferAlignment {
            offset_bytes,
            align,
        });
    }

    Ok(byte_len)
}

#[cfg(target_os = "macos")]
fn checked_buffer_contents_ptr<T: GpuAbi>(
    buffer: &Buffer,
    offset_bytes: usize,
    len: usize,
) -> Result<*mut T, MetalSupportError> {
    let buffer_len = usize::try_from(buffer.length()).unwrap_or(usize::MAX);
    let byte_len = checked_buffer_typed_range::<T>(buffer_len, offset_bytes, len)?;

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
/// Copy one GPU ABI value out of a CPU-visible Metal buffer.
///
/// The returned value is owned and therefore cannot alias later Metal writes.
/// Bounds, offset arithmetic, alignment, CPU visibility, and the [`GpuAbi`]
/// element contract are validated before the copy.
///
/// # Safety
///
/// The caller must ensure that all Metal commands which can write this range
/// have completed and that neither the CPU nor GPU mutates it during the copy.
pub unsafe fn checked_buffer_read<T: GpuAbi>(
    buffer: &Buffer,
    offset_bytes: usize,
) -> Result<T, MetalSupportError> {
    let ptr = checked_buffer_contents_ptr::<T>(buffer, offset_bytes, 1)?;
    // SAFETY: The pointer is aligned and in bounds for one `T`. The caller
    // guarantees that the GPU cannot mutate the range during this copy.
    Ok(unsafe { ptr.cast_const().read() })
}

#[cfg(target_os = "macos")]
/// Copy GPU ABI values out of a CPU-visible Metal buffer.
///
/// The returned vector is owned and therefore cannot alias later Metal writes.
/// A zero-element request succeeds without dereferencing `contents()`.
///
/// # Safety
///
/// The caller must ensure that all Metal commands which can write this range
/// have completed and that neither the CPU nor GPU mutates it during the copy.
pub unsafe fn checked_buffer_read_vec<T: GpuAbi>(
    buffer: &Buffer,
    offset_bytes: usize,
    len: usize,
) -> Result<Vec<T>, MetalSupportError> {
    let buffer_len = usize::try_from(buffer.length()).unwrap_or(usize::MAX);
    checked_buffer_typed_range::<T>(buffer_len, offset_bytes, len)?;
    if len == 0 {
        return Ok(Vec::new());
    }

    let ptr = checked_buffer_contents_ptr::<T>(buffer, offset_bytes, len)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| MetalSupportError::BufferReadbackAllocation {
            abi_name: T::NAME,
            element_count: len,
        })?;
    // SAFETY: `values` has capacity for `len` values, and `ptr` is aligned and
    // in bounds for `len` values. `GpuAbi` guarantees every copied bit pattern
    // is a valid `T`; the caller guarantees synchronization with Metal.
    unsafe {
        core::ptr::copy_nonoverlapping(ptr.cast_const(), values.as_mut_ptr(), len);
        values.set_len(len);
    }
    Ok(values)
}

#[cfg(target_os = "macos")]
/// Copy GPU ABI values into a CPU-visible Metal buffer.
///
/// A zero-element write succeeds without dereferencing `contents()`.
///
/// # Safety
///
/// The caller must ensure that no Metal command or other CPU access reads or
/// writes this range during the copy. In normal use, initialize the range
/// before submitting a command buffer or after all prior uses have completed.
pub unsafe fn checked_buffer_write<T: GpuAbi>(
    buffer: &Buffer,
    offset_bytes: usize,
    values: &[T],
) -> Result<(), MetalSupportError> {
    let buffer_len = usize::try_from(buffer.length()).unwrap_or(usize::MAX);
    checked_buffer_typed_range::<T>(buffer_len, offset_bytes, values.len())?;
    if values.is_empty() {
        return Ok(());
    }

    let ptr = checked_buffer_contents_ptr::<T>(buffer, offset_bytes, values.len())?;
    // SAFETY: The destination is aligned and in bounds for `values`; the
    // caller guarantees exclusive CPU/GPU access for the duration of the copy.
    unsafe {
        core::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
/// Fill a checked byte range in a CPU-visible Metal buffer.
///
/// A zero-byte fill succeeds without dereferencing `contents()`.
///
/// # Safety
///
/// The caller must ensure that no Metal command or other CPU access reads or
/// writes this range during the fill.
pub unsafe fn checked_buffer_fill_bytes(
    buffer: &Buffer,
    offset_bytes: usize,
    len: usize,
    value: u8,
) -> Result<(), MetalSupportError> {
    let buffer_len = usize::try_from(buffer.length()).unwrap_or(usize::MAX);
    checked_buffer_typed_range::<u8>(buffer_len, offset_bytes, len)?;
    if len == 0 {
        return Ok(());
    }

    let ptr = checked_buffer_contents_ptr::<u8>(buffer, offset_bytes, len)?;
    // SAFETY: The destination is in bounds for `len` bytes; the caller
    // guarantees exclusive CPU/GPU access for the duration of the fill.
    unsafe {
        core::ptr::write_bytes(ptr, value, len);
    }
    Ok(())
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
        checked_buffer_fill_bytes, checked_buffer_read_vec, checked_buffer_typed_range,
        checked_buffer_write, checked_command_queue, commit_and_wait, one_d_threads_per_group,
        private_buffer, shared_buffer_for_len, shared_buffer_with_slice, system_default_device,
        two_d_threads_per_group, MetalSupportError,
    };

    #[derive(Clone, Copy)]
    struct ZeroSizedAbi;

    // SAFETY: This intentionally invalid zero-sized ABI implementation exists
    // only to prove that the Metal range validator rejects zero-sized types.
    unsafe impl j2k_core::accelerator::GpuAbi for ZeroSizedAbi {
        const NAME: &'static str = "ZeroSizedAbi";
    }

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
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let Ok(device) = system_default_device() else {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        };
        let queue = checked_command_queue(&device).expect("Metal command queue");
        let command_buffer = queue.new_command_buffer();

        commit_and_wait(command_buffer).expect("unlabeled command buffer completion");
    }

    #[test]
    fn buffer_readback_copies_typed_shared_buffer_values() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let Ok(device) = system_default_device() else {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        };
        let buffer = shared_buffer_with_slice(&device, &[3_u32, 5, 8, 13]);

        // SAFETY: The buffer was initialized by the CPU and has not been
        // submitted to Metal, so no GPU access can race this readback.
        let values =
            unsafe { checked_buffer_read_vec::<u32>(&buffer, 0, 4) }.expect("checked readback");

        assert_eq!(values, [3, 5, 8, 13]);
    }

    #[test]
    fn buffer_write_and_fill_copy_into_shared_buffer() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let Ok(device) = system_default_device() else {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        };
        let buffer = shared_buffer_for_len::<u32>(&device, 3);

        // SAFETY: The buffer has not been submitted to Metal, and these CPU
        // accesses are sequential and non-overlapping in time.
        unsafe {
            checked_buffer_fill_bytes(&buffer, 0, 12, 0).expect("checked fill");
            checked_buffer_write::<u32>(&buffer, 0, &[21, 34, 55]).expect("checked write");
        }
        // SAFETY: The preceding CPU write is complete and no GPU command can
        // access this never-submitted buffer.
        let values =
            unsafe { checked_buffer_read_vec::<u32>(&buffer, 0, 3) }.expect("checked readback");

        assert_eq!(values, [21, 34, 55]);
    }

    #[test]
    fn checked_buffer_readback_rejects_out_of_bounds_range() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let Ok(device) = system_default_device() else {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        };
        let buffer = shared_buffer_with_slice(&device, &[1_u32]);

        // SAFETY: No bytes are copied because validation rejects the range.
        let err =
            unsafe { checked_buffer_read_vec::<u32>(&buffer, 0, 2) }.expect_err("bounds error");

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
    fn checked_buffer_readback_rejects_unaligned_range() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let Ok(device) = system_default_device() else {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        };
        let buffer = shared_buffer_with_slice(&device, &[1_u32, 2]);

        // SAFETY: No bytes are copied because validation rejects the range.
        let err =
            unsafe { checked_buffer_read_vec::<u32>(&buffer, 1, 1) }.expect_err("alignment error");

        assert!(matches!(
            err,
            MetalSupportError::BufferAlignment {
                offset_bytes: 1,
                align: 4,
            }
        ));
    }

    #[test]
    fn typed_range_rejects_overflow_and_zero_sized_abi() {
        let overflow = checked_buffer_typed_range::<u32>(usize::MAX, 0, usize::MAX)
            .expect_err("element byte length overflow");
        assert!(matches!(
            overflow,
            MetalSupportError::BufferBounds {
                offset_bytes: 0,
                byte_len: usize::MAX,
                buffer_len: usize::MAX,
            }
        ));

        let range_overflow = checked_buffer_typed_range::<u8>(usize::MAX, usize::MAX, 1)
            .expect_err("range end overflow");
        assert!(matches!(
            range_overflow,
            MetalSupportError::BufferBounds {
                offset_bytes: usize::MAX,
                byte_len: 1,
                buffer_len: usize::MAX,
            }
        ));

        assert!(matches!(
            checked_buffer_typed_range::<ZeroSizedAbi>(8, 0, 1),
            Err(MetalSupportError::BufferZeroSizedType {
                abi_name: "ZeroSizedAbi"
            })
        ));
    }

    #[test]
    fn zero_length_readback_does_not_require_cpu_visible_contents() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let Ok(device) = system_default_device() else {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        };
        let buffer = private_buffer(&device, 4);

        // SAFETY: A zero-element request performs no memory access.
        let values =
            unsafe { checked_buffer_read_vec::<u32>(&buffer, 4, 0) }.expect("empty readback");

        assert!(values.is_empty());
    }
}
