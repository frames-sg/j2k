// SPDX-License-Identifier: Apache-2.0

use signinum_core::{copy_tight_pixels_to_strided_output, BackendKind, DeviceSurface, PixelFormat};
#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::CudaDeviceBuffer;

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
use crate::Error;

#[derive(Debug)]
pub(crate) enum Storage {
    Host(Vec<u8>),
    #[cfg(feature = "cuda-runtime")]
    Cuda(CudaDeviceBuffer),
}

/// CUDA surface execution counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaSurfaceStats {
    pub(crate) total: usize,
    pub(crate) copy: usize,
    pub(crate) decode: usize,
}

impl CudaSurfaceStats {
    /// Total CUDA kernel dispatches associated with the surface.
    pub fn kernel_dispatches(self) -> usize {
        self.total
    }

    /// CUDA copy/upload kernel dispatches associated with the surface.
    pub fn copy_kernel_dispatches(self) -> usize {
        self.copy
    }

    /// CUDA codestream decode kernel dispatches associated with the surface.
    pub fn decode_kernel_dispatches(self) -> usize {
        self.decode
    }
}

/// Borrowed view of a CUDA-resident surface.
#[derive(Clone, Copy, Debug)]
pub struct CudaSurface<'a> {
    #[cfg(feature = "cuda-runtime")]
    buffer: &'a CudaDeviceBuffer,
    #[cfg(not(feature = "cuda-runtime"))]
    _marker: core::marker::PhantomData<&'a ()>,
    pub(crate) stats: CudaSurfaceStats,
}

impl CudaSurface<'_> {
    /// Raw CUDA device pointer value.
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

    /// Execution counters for this surface.
    pub fn stats(&self) -> CudaSurfaceStats {
        self.stats
    }
}

/// Residency of a decoded J2K CUDA adapter surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurfaceResidency {
    /// Pixels are stored in host memory.
    Host,
    /// Pixels were produced directly by a CUDA codestream decode path.
    CudaResidentDecode,
    /// Pixels were decoded on CPU and uploaded into a CUDA buffer.
    CpuStagedCudaUpload,
}

impl Default for SurfaceResidency {
    fn default() -> Self {
        Self::Host
    }
}

/// Host- or CUDA-backed decoded surface.
#[derive(Debug)]
pub struct Surface {
    pub(crate) backend: BackendKind,
    pub(crate) residency: SurfaceResidency,
    pub(crate) dimensions: (u32, u32),
    pub(crate) fmt: PixelFormat,
    pub(crate) pitch_bytes: usize,
    pub(crate) stats: CudaSurfaceStats,
    pub(crate) storage: Storage,
}

impl Surface {
    /// Return where the surface's pixels currently reside.
    pub fn residency(&self) -> SurfaceResidency {
        self.residency
    }

    /// Row pitch in bytes.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    /// Borrow host bytes when the surface is host-backed.
    pub fn as_host_bytes(&self) -> Option<&[u8]> {
        match &self.storage {
            Storage::Host(bytes) => Some(bytes),
            #[cfg(feature = "cuda-runtime")]
            Storage::Cuda(_) => None,
        }
    }

    /// Download or copy the surface into caller-owned strided output.
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

    /// Borrow CUDA metadata when the surface is CUDA-backed.
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

    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    fn pixel_format(&self) -> PixelFormat {
        self.fmt
    }

    fn byte_len(&self) -> usize {
        self.pitch_bytes * self.dimensions.1 as usize
    }
}
