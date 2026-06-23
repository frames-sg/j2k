// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

use j2k_core::{
    copy_tight_pixels_to_strided_output, BackendKind, BufferError, DeviceMemoryRange,
    DeviceSurface, ExecutionStats, PixelFormat,
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
    #[cfg(feature = "cuda-runtime")]
    CudaRange {
        buffer: Arc<CudaDeviceBuffer>,
        offset: usize,
        len: usize,
    },
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
    #[cfg(feature = "cuda-runtime")]
    offset: usize,
    #[cfg(not(feature = "cuda-runtime"))]
    _marker: core::marker::PhantomData<&'a ()>,
    pub(crate) stats: CudaSurfaceStats,
}

impl CudaSurface<'_> {
    /// Raw CUDA device pointer value.
    pub fn device_ptr(&self) -> u64 {
        #[cfg(feature = "cuda-runtime")]
        {
            self.buffer.device_ptr().saturating_add(self.offset as u64)
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurfaceResidency {
    /// Pixels are stored in host memory.
    #[default]
    Host,
    /// Pixels were produced directly by a CUDA codestream decode path.
    CudaResidentDecode,
    /// Pixels were decoded on CPU and uploaded into a CUDA buffer.
    CpuStagedCudaUpload,
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
            Storage::Cuda(_) | Storage::CudaRange { .. } => None,
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
                let byte_len = self.byte_len();
                if let Some(len) =
                    tight_cuda_download_len(byte_len, self.pitch_bytes, stride, out.len())
                {
                    return buffer.copy_to_host(&mut out[..len]).map_err(cuda_error);
                }
                let mut tight = vec![0u8; byte_len];
                buffer.copy_to_host(&mut tight).map_err(cuda_error)?;
                copy_tight_pixels_to_strided_output(&tight, self.dimensions, self.fmt, out, stride)
                    .map_err(Error::from)
            }
            #[cfg(feature = "cuda-runtime")]
            Storage::CudaRange {
                buffer,
                offset,
                len,
            } => {
                let byte_len = self.byte_len();
                debug_assert_eq!(*len, byte_len);
                if let Some(len) =
                    tight_cuda_download_len(byte_len, self.pitch_bytes, stride, out.len())
                {
                    return buffer
                        .copy_range_to_host(*offset, &mut out[..len])
                        .map_err(cuda_error);
                }
                let mut tight = vec![0u8; byte_len];
                buffer
                    .copy_range_to_host(*offset, &mut tight)
                    .map_err(cuda_error)?;
                copy_tight_pixels_to_strided_output(&tight, self.dimensions, self.fmt, out, stride)
                    .map_err(Error::from)
            }
        }
    }

    /// Download the surface and return elapsed host copy time in microseconds.
    pub fn download_into_profiled(&self, out: &mut [u8], stride: usize) -> Result<u128, Error> {
        let started = std::time::Instant::now();
        self.download_into(out, stride)?;
        Ok(started.elapsed().as_micros())
    }

    /// Borrow CUDA metadata when the surface is CUDA-backed.
    pub fn cuda_surface(&self) -> Option<CudaSurface<'_>> {
        #[cfg(feature = "cuda-runtime")]
        match &self.storage {
            Storage::Cuda(buffer) => Some(CudaSurface {
                buffer,
                offset: 0,
                stats: self.stats,
            }),
            Storage::CudaRange { buffer, offset, .. } => Some(CudaSurface {
                buffer,
                offset: *offset,
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

    /// Download a sequence of surfaces into a tightly concatenated output buffer.
    ///
    /// CUDA surfaces produced from one contiguous batch allocation are copied
    /// with one device-to-host transfer. Other layouts fall back to downloading
    /// each surface tightly in order.
    pub fn download_batch_tight(surfaces: &[Self]) -> Result<Vec<u8>, Error> {
        let required = batch_tight_required_len(surfaces)?;
        if required == 0 {
            return Ok(Vec::new());
        }

        #[cfg(feature = "cuda-runtime")]
        if let Some((buffer, offset)) = contiguous_cuda_batch_range(surfaces) {
            let mut out = Vec::with_capacity(required);
            buffer
                .copy_range_to_host_uninit(offset, out.spare_capacity_mut())
                .map_err(cuda_error)?;
            // SAFETY: the CUDA copy above initialized exactly `required`
            // bytes in this Vec's spare capacity and returned success.
            unsafe {
                out.set_len(required);
            }
            return Ok(out);
        }

        let mut out = vec![0u8; required];
        Self::download_batch_tight_into(surfaces, &mut out)?;
        Ok(out)
    }

    /// Download a sequence of surfaces into a tightly concatenated output buffer.
    ///
    /// CUDA surfaces produced from one contiguous batch allocation are copied
    /// with one device-to-host transfer. Other layouts fall back to downloading
    /// each surface tightly in order.
    pub fn download_batch_tight_into(surfaces: &[Self], out: &mut [u8]) -> Result<(), Error> {
        let required = batch_tight_required_len(surfaces)?;
        if out.len() < required {
            return Err(BufferError::OutputTooSmall {
                required,
                have: out.len(),
            }
            .into());
        }
        if required == 0 {
            return Ok(());
        }

        #[cfg(feature = "cuda-runtime")]
        if let Some((buffer, offset)) = contiguous_cuda_batch_range(surfaces) {
            return buffer
                .copy_range_to_host(offset, &mut out[..required])
                .map_err(cuda_error);
        }

        let mut cursor = 0usize;
        for surface in surfaces {
            let len = surface.byte_len();
            surface.download_into(&mut out[cursor..cursor + len], surface.pitch_bytes)?;
            cursor += len;
        }
        Ok(())
    }
}

fn batch_tight_required_len(surfaces: &[Surface]) -> Result<usize, Error> {
    surfaces
        .iter()
        .try_fold(0usize, |sum, surface| sum.checked_add(surface.byte_len()))
        .ok_or(BufferError::SizeOverflow {
            what: "tight batch surface output",
        })
        .map_err(Error::from)
}

#[cfg(feature = "cuda-runtime")]
pub(crate) fn cuda_range_storage(
    buffer: Arc<CudaDeviceBuffer>,
    offset: usize,
    len: usize,
) -> Storage {
    Storage::CudaRange {
        buffer,
        offset,
        len,
    }
}

#[cfg(feature = "cuda-runtime")]
fn contiguous_cuda_batch_range(surfaces: &[Surface]) -> Option<(&CudaDeviceBuffer, usize)> {
    let first = surfaces.first()?;
    let Storage::CudaRange {
        buffer,
        offset,
        len,
    } = &first.storage
    else {
        return None;
    };
    let first_buffer = buffer;
    let first_offset = *offset;
    let mut expected_offset = first_offset.checked_add(*len)?;
    for surface in &surfaces[1..] {
        let Storage::CudaRange {
            buffer,
            offset,
            len,
        } = &surface.storage
        else {
            return None;
        };
        if !Arc::ptr_eq(first_buffer, buffer) || *offset != expected_offset {
            return None;
        }
        expected_offset = expected_offset.checked_add(*len)?;
    }
    Some((first_buffer.as_ref(), first_offset))
}

#[cfg(any(feature = "cuda-runtime", test))]
fn tight_cuda_download_len(
    byte_len: usize,
    pitch_bytes: usize,
    stride: usize,
    out_len: usize,
) -> Option<usize> {
    (stride == pitch_bytes && out_len >= byte_len).then_some(byte_len)
}

impl DeviceSurface for Surface {
    fn backend_kind(&self) -> BackendKind {
        self.backend
    }

    fn residency(&self) -> j2k_core::SurfaceResidency {
        match self.residency {
            SurfaceResidency::Host => j2k_core::SurfaceResidency::Host,
            SurfaceResidency::CudaResidentDecode => j2k_core::SurfaceResidency::CudaResidentDecode,
            SurfaceResidency::CpuStagedCudaUpload => {
                j2k_core::SurfaceResidency::CpuStagedCudaUpload
            }
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
            kernel_dispatches: self.stats.total as u64,
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
            #[cfg(feature = "cuda-runtime")]
            Storage::CudaRange {
                buffer,
                offset,
                len,
            } => Some(DeviceMemoryRange::new(
                BackendKind::Cuda,
                buffer.device_ptr(),
                *offset,
                *len,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{tight_cuda_download_len, CudaSurfaceStats, Storage, Surface, SurfaceResidency};
    use j2k_core::{BackendKind, PixelFormat};

    #[test]
    fn tight_cuda_download_len_accepts_exact_tight_output() {
        assert_eq!(tight_cuda_download_len(32, 8, 8, 32), Some(32));
    }

    #[test]
    fn download_batch_tight_returns_tightly_concatenated_host_surfaces() {
        let surfaces = [
            Surface {
                backend: BackendKind::Cpu,
                residency: SurfaceResidency::Host,
                dimensions: (2, 1),
                fmt: PixelFormat::Gray8,
                pitch_bytes: 2,
                stats: CudaSurfaceStats::default(),
                storage: Storage::Host(vec![1, 2]),
            },
            Surface {
                backend: BackendKind::Cpu,
                residency: SurfaceResidency::Host,
                dimensions: (1, 1),
                fmt: PixelFormat::Rgb8,
                pitch_bytes: 3,
                stats: CudaSurfaceStats::default(),
                storage: Storage::Host(vec![3, 4, 5]),
            },
        ];

        let tight = Surface::download_batch_tight(&surfaces).expect("batch download");

        assert_eq!(tight, vec![1, 2, 3, 4, 5]);
    }
}
