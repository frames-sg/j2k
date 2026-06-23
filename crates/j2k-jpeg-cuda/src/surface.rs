// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    copy_tight_pixels_to_strided_output, BackendKind, DeviceMemoryRange, DeviceSurface,
    ExecutionStats, PixelFormat,
};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::CudaDeviceBuffer;

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
use crate::Error;

#[derive(Debug)]
pub(crate) enum Storage {
    Host(Vec<u8>),
    #[cfg(feature = "cuda-runtime")]
    Cuda(CudaDeviceBuffer),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// CUDA JPEG decode path used to produce a surface.
pub enum CudaJpegDecodePath {
    /// Surface did not use a CUDA JPEG decode kernel or library path.
    #[default]
    None,
    /// Surface was produced by J2K-owned CUDA JPEG kernels.
    OwnedCuda,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Dispatch counters and residency metadata for a CUDA JPEG surface.
pub struct CudaSurfaceStats {
    pub(crate) kernel_dispatches: usize,
    pub(crate) copy_kernel_dispatches: usize,
    pub(crate) decode_kernel_dispatches: usize,
    pub(crate) hardware_decode: bool,
    pub(crate) decode_path: CudaJpegDecodePath,
}

impl CudaSurfaceStats {
    /// Total CUDA kernel or library dispatches used to produce the surface.
    pub fn kernel_dispatches(self) -> usize {
        self.kernel_dispatches
    }

    /// Number of copy-kernel dispatches used for the surface.
    pub fn copy_kernel_dispatches(self) -> usize {
        self.copy_kernel_dispatches
    }

    /// Number of decode kernel or library dispatches used for the surface.
    pub fn decode_kernel_dispatches(self) -> usize {
        self.decode_kernel_dispatches
    }

    /// CUDA JPEG decode path used for the surface.
    pub fn decode_path(self) -> CudaJpegDecodePath {
        self.decode_path
    }

    /// Whether the J2K-owned CUDA JPEG decode path was used.
    pub fn used_owned_cuda_decode(self) -> bool {
        self.decode_path == CudaJpegDecodePath::OwnedCuda
    }

    /// Whether hardware JPEG decode was used.
    pub fn used_hardware_decode(self) -> bool {
        self.hardware_decode
    }
}

#[derive(Clone, Copy, Debug)]
/// Borrowed CUDA-resident JPEG decode surface.
pub struct CudaSurface<'a> {
    #[cfg(feature = "cuda-runtime")]
    buffer: &'a CudaDeviceBuffer,
    #[cfg(not(feature = "cuda-runtime"))]
    _marker: core::marker::PhantomData<&'a ()>,
    pub(crate) stats: CudaSurfaceStats,
}

impl CudaSurface<'_> {
    /// Return the CUDA device pointer for the backing buffer.
    pub fn device_ptr(&self) -> u64 {
        #[cfg(feature = "cuda-runtime")]
        {
            self.buffer.device_ptr()
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            unreachable!("CudaSurface cannot be constructed without cuda-runtime support")
        }
    }

    /// Return dispatch statistics for the surface.
    pub fn stats(&self) -> CudaSurfaceStats {
        self.stats
    }
}

#[derive(Debug)]
/// Decoded JPEG surface returned by the CUDA adapter.
pub struct Surface {
    pub(crate) backend: BackendKind,
    pub(crate) dimensions: (u32, u32),
    pub(crate) fmt: PixelFormat,
    pub(crate) pitch_bytes: usize,
    pub(crate) stats: CudaSurfaceStats,
    pub(crate) storage: Storage,
}

impl Surface {
    /// Number of bytes between consecutive rows.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    /// Borrow host-resident bytes when this surface is not CUDA-backed.
    pub fn as_host_bytes(&self) -> Option<&[u8]> {
        match &self.storage {
            Storage::Host(bytes) => Some(bytes),
            #[cfg(feature = "cuda-runtime")]
            Storage::Cuda(_) => None,
        }
    }

    /// Copy surface bytes into a caller-provided strided output buffer.
    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        match &self.storage {
            Storage::Host(bytes) => {
                copy_tight_pixels_to_strided_output(bytes, self.dimensions, self.fmt, out, stride)
                    .map_err(Error::from)
            }
            #[cfg(feature = "cuda-runtime")]
            Storage::Cuda(buffer) => {
                let mut tight = vec![0u8; self.byte_len()];
                buffer.copy_to_host(&mut tight).map_err(cuda_error)?;
                copy_tight_pixels_to_strided_output(&tight, self.dimensions, self.fmt, out, stride)
                    .map_err(Error::from)
            }
        }
    }

    /// Borrow CUDA surface metadata when this surface is CUDA-backed.
    pub fn cuda_surface(&self) -> Option<CudaSurface<'_>> {
        #[cfg(feature = "cuda-runtime")]
        match &self.storage {
            Storage::Cuda(buffer) => Some(CudaSurface {
                buffer,
                stats: self.stats,
            }),
            Storage::Host(_) => None,
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            let _ = self.stats;
            None
        }
    }
}

impl DeviceSurface for Surface {
    fn backend_kind(&self) -> BackendKind {
        self.backend
    }

    fn residency(&self) -> j2k_core::SurfaceResidency {
        match &self.storage {
            Storage::Host(_) => j2k_core::SurfaceResidency::Host,
            #[cfg(feature = "cuda-runtime")]
            Storage::Cuda(_) => j2k_core::SurfaceResidency::CudaResidentDecode,
        }
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    fn pixel_format(&self) -> PixelFormat {
        self.fmt
    }

    fn byte_len(&self) -> usize {
        self.pitch_bytes * self.dimensions.1 as usize
    }

    fn execution_stats(&self) -> ExecutionStats {
        ExecutionStats {
            kernel_dispatches: self.stats.kernel_dispatches as u64,
            ..ExecutionStats::default()
        }
    }

    fn memory_range(&self) -> Option<DeviceMemoryRange> {
        match &self.storage {
            Storage::Host(_) => None,
            #[cfg(feature = "cuda-runtime")]
            Storage::Cuda(buffer) => Some(DeviceMemoryRange::new(
                BackendKind::Cuda,
                buffer.device_ptr(),
                0,
                self.byte_len(),
            )),
        }
    }
}
