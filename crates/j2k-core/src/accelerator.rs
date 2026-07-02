// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::{size_of, size_of_val};
use core::slice;

use crate::backend::BackendKind;

/// Residency of an accelerator-visible surface or buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SurfaceResidency {
    /// Host memory owned by CPU code.
    Host,
    /// Pixels were produced directly by a CUDA decode path.
    CudaResidentDecode,
    /// Pixels were decoded on CPU and uploaded into CUDA memory.
    CpuStagedCudaUpload,
    /// Pixels were produced directly by a Metal decode path.
    MetalResidentDecode,
    /// Pixels were decoded on CPU and uploaded into Metal memory.
    CpuStagedMetalUpload,
    /// Device-local memory owned by a backend.
    Device(BackendKind),
    /// Host/device shared memory for the backend.
    Shared(BackendKind),
}

impl SurfaceResidency {
    /// Generic residency for a backend-produced surface.
    #[must_use]
    pub const fn for_backend(backend: BackendKind) -> Self {
        match backend {
            BackendKind::Cpu => Self::Host,
            BackendKind::Metal => Self::Device(BackendKind::Metal),
            BackendKind::Cuda => Self::Device(BackendKind::Cuda),
        }
    }
}

/// Execution counters reported by accelerator sessions and surfaces.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionStats {
    /// Number of submitted backend command buffers, streams, or equivalent jobs.
    pub submissions: u64,
    /// Number of kernel or shader dispatches.
    pub kernel_dispatches: u64,
    /// Bytes uploaded from host to device.
    pub upload_bytes: u64,
    /// Bytes read back from device to host.
    pub readback_bytes: u64,
    /// Backend-reported execution time in microseconds, when available.
    pub device_us: u128,
}

impl ExecutionStats {
    /// Construct empty execution statistics.
    pub const fn new() -> Self {
        Self {
            submissions: 0,
            kernel_dispatches: 0,
            upload_bytes: 0,
            readback_bytes: 0,
            device_us: 0,
        }
    }

    /// Saturating sum of two execution-stat blocks.
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self {
            submissions: self.submissions.saturating_add(other.submissions),
            kernel_dispatches: self
                .kernel_dispatches
                .saturating_add(other.kernel_dispatches),
            upload_bytes: self.upload_bytes.saturating_add(other.upload_bytes),
            readback_bytes: self.readback_bytes.saturating_add(other.readback_bytes),
            device_us: self.device_us.saturating_add(other.device_us),
        }
    }
}

/// Opaque byte range in accelerator-visible memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceMemoryRange {
    /// Backend that owns the range.
    pub backend: BackendKind,
    /// Backend-local allocation identifier. Backends define its meaning.
    pub allocation: u64,
    /// Byte offset inside the allocation.
    pub offset: usize,
    /// Byte length of the range.
    pub len: usize,
}

impl DeviceMemoryRange {
    /// Construct a backend-local memory range.
    pub const fn new(backend: BackendKind, allocation: u64, offset: usize, len: usize) -> Self {
        Self {
            backend,
            allocation,
            offset,
            len,
        }
    }
}

/// Shared session contract for caller-owned accelerator runtime state.
pub trait AcceleratorSession {
    /// Backend owned by this session.
    fn backend_kind(&self) -> BackendKind;

    /// Execution statistics accumulated by this session.
    fn execution_stats(&self) -> ExecutionStats {
        ExecutionStats::default()
    }
}

/// Marker trait for host-side values whose memory layout is part of a GPU ABI.
///
/// # Safety
/// Implementers must guarantee that `Self` has a stable host/shader layout for
/// every backend that consumes it. In practice this means using `#[repr(C)]` or
/// an equivalent explicit layout, avoiding padding-sensitive interpretation
/// unless tests cover the exact byte layout, and keeping all fields plain data.
pub unsafe trait GpuAbi: Copy + 'static {
    /// Human-readable ABI name used in layout-test failures.
    const NAME: &'static str;

    /// View one value as bytes.
    fn as_bytes(value: &Self) -> &[u8] {
        // SAFETY: The trait contract requires `Self` to be plain GPU ABI data.
        unsafe { slice::from_raw_parts(core::ptr::from_ref(value).cast::<u8>(), size_of::<Self>()) }
    }

    /// View a slice of values as bytes.
    fn slice_as_bytes(values: &[Self]) -> &[u8] {
        // SAFETY: The trait contract requires `Self` to be plain GPU ABI data.
        unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values)) }
    }

    /// Mutably view a slice of values as bytes.
    fn slice_as_bytes_mut(values: &mut [Self]) -> &mut [u8] {
        // SAFETY: The trait contract requires `Self` to be plain GPU ABI data.
        unsafe { slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), size_of_val(values)) }
    }
}

macro_rules! impl_gpu_abi_primitive {
    ($($ty:ty),* $(,)?) => {
        $(
            // SAFETY: Primitive numeric types are plain data with stable Rust layouts
            // accepted by the shader ABI helpers as scalar values.
            unsafe impl GpuAbi for $ty {
                const NAME: &'static str = stringify!($ty);
            }
        )*
    };
}

impl_gpu_abi_primitive!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

// SAFETY: Arrays preserve element order and contain no extra non-element state;
// the element type supplies the GPU ABI layout contract.
unsafe impl<T, const N: usize> GpuAbi for [T; N]
where
    T: GpuAbi,
{
    const NAME: &'static str = "array";
}
