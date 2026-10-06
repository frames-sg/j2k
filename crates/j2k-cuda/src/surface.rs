// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

use j2k_core::{
    copy_tight_pixels_to_strided_output, BackendKind, BufferError, DeviceMemoryRange,
    DeviceSurface, ExecutionStats, PixelFormat, SurfaceMetadata,
};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{CudaDeviceBuffer, CudaPooledDeviceBuffer};

use crate::allocation::try_vec_filled;
#[cfg(feature = "cuda-runtime")]
use crate::allocation::try_vec_with_capacity;
#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
use crate::Error;

pub use j2k_core::SurfaceResidency;

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
    #[cfg(feature = "cuda-runtime")]
    CudaPooledRange {
        buffer: Arc<CudaPooledDeviceBuffer>,
        offset: usize,
        len: usize,
    },
}

#[cfg(feature = "cuda-runtime")]
impl Storage {
    fn cuda_buffer_range(&self) -> Option<(&CudaDeviceBuffer, usize, usize)> {
        match self {
            Self::Cuda(buffer) => Some((buffer, 0, buffer.byte_len())),
            Self::CudaRange {
                buffer,
                offset,
                len,
            } => Some((buffer, *offset, *len)),
            Self::CudaPooledRange {
                buffer,
                offset,
                len,
            } => Some((buffer.as_device_buffer()?, *offset, *len)),
            Self::Host(_) => None,
        }
    }
}

/// CUDA surface execution counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaSurfaceStats {
    pub(crate) total: usize,
    pub(crate) copy: usize,
    pub(crate) decode: usize,
}

#[doc(hidden)]
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
    device_ptr: u64,
    _marker: core::marker::PhantomData<&'a ()>,
    pub(crate) stats: CudaSurfaceStats,
}

impl CudaSurface<'_> {
    /// Raw CUDA device pointer value.
    ///
    /// Keep the owning [`Surface`] alive until every operation that reads
    /// this pointer on another CUDA stream has completed. Dropping the
    /// surface may return its memory to the session's buffer pool without
    /// synchronizing, and the next decode can then overwrite it.
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    /// Execution counters for this surface.
    #[doc(hidden)]
    pub fn stats(&self) -> CudaSurfaceStats {
        self.stats
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
    fn metadata(&self) -> SurfaceMetadata {
        SurfaceMetadata::new(
            self.backend,
            self.residency,
            self.dimensions,
            self.fmt,
            self.pitch_bytes,
        )
    }

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
            Storage::Cuda(_) | Storage::CudaRange { .. } | Storage::CudaPooledRange { .. } => None,
        }
    }

    /// Download or copy the surface into caller-owned strided output.
    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        if let Some(bytes) = self.as_host_bytes() {
            return copy_tight_pixels_to_strided_output(
                bytes,
                self.dimensions,
                self.fmt,
                out,
                stride,
            )
            .map_err(Error::from);
        }
        #[cfg(feature = "cuda-runtime")]
        if let Some((buffer, offset, len)) = self.storage.cuda_buffer_range() {
            let byte_len = self.byte_len();
            if len < byte_len {
                return Err(BufferError::InputTooSmall {
                    required: byte_len,
                    have: len,
                }
                .into());
            }
            if let Some(len) =
                tight_cuda_download_len(byte_len, self.pitch_bytes, stride, out.len())
            {
                return buffer
                    .copy_range_to_host(offset, &mut out[..len])
                    .map_err(cuda_error);
            }
            let mut tight = try_vec_filled(byte_len, 0u8, "j2k CUDA surface download staging")?;
            buffer
                .copy_range_to_host(offset, &mut tight)
                .map_err(cuda_error)?;
            return copy_tight_pixels_to_strided_output(
                &tight,
                self.dimensions,
                self.fmt,
                out,
                stride,
            )
            .map_err(Error::from);
        }
        Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::contract_violation(
                "CUDA surface has no device buffer to download",
            ),
        ))
    }

    /// Borrow CUDA metadata when the surface is CUDA-backed.
    pub fn cuda_surface(&self) -> Option<CudaSurface<'_>> {
        #[cfg(feature = "cuda-runtime")]
        {
            let (buffer, offset, _) = self.storage.cuda_buffer_range()?;
            let device_ptr = buffer
                .device_ptr()
                .checked_add(u64::try_from(offset).ok()?)?;
            Some(CudaSurface {
                device_ptr,
                _marker: core::marker::PhantomData,
                stats: self.stats,
            })
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
            let mut out = try_vec_with_capacity(required, "j2k CUDA contiguous batch download")?;
            buffer
                .copy_range_to_host_uninit(offset, &mut out.spare_capacity_mut()[..required])
                .map_err(cuda_error)?;
            // SAFETY: the CUDA copy above initialized exactly `required`
            // bytes in this Vec's spare capacity and returned success.
            unsafe {
                out.set_len(required);
            }
            return Ok(out);
        }

        let mut out = try_vec_filled(required, 0u8, "j2k CUDA batch download")?;
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
pub(crate) fn cuda_pooled_range_storage(
    buffer: Arc<CudaPooledDeviceBuffer>,
    offset: usize,
    len: usize,
) -> Storage {
    Storage::CudaPooledRange {
        buffer,
        offset,
        len,
    }
}

#[cfg(feature = "cuda-runtime")]
fn contiguous_cuda_batch_range(surfaces: &[Surface]) -> Option<(&CudaDeviceBuffer, usize)> {
    let (first_buffer, first_offset, first_len) = surfaces.first()?.storage.cuda_buffer_range()?;
    let mut expected_offset = first_offset.checked_add(first_len)?;
    for surface in &surfaces[1..] {
        let (buffer, offset, len) = surface.storage.cuda_buffer_range()?;
        if !core::ptr::eq(first_buffer, buffer) || offset != expected_offset {
            return None;
        }
        expected_offset = expected_offset.checked_add(len)?;
    }
    Some((first_buffer, first_offset))
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

#[doc(hidden)]
impl DeviceSurface for Surface {
    fn backend_kind(&self) -> BackendKind {
        self.metadata().backend
    }

    fn residency(&self) -> j2k_core::SurfaceResidency {
        self.metadata().residency
    }

    fn dimensions(&self) -> (u32, u32) {
        self.metadata().dimensions
    }

    fn pixel_format(&self) -> PixelFormat {
        self.metadata().pixel_format
    }

    fn byte_len(&self) -> usize {
        self.metadata().byte_len()
    }

    fn execution_stats(&self) -> ExecutionStats {
        ExecutionStats {
            kernel_dispatches: self.stats.total as u64,
            ..ExecutionStats::default()
        }
    }

    fn memory_range(&self) -> Option<DeviceMemoryRange> {
        #[cfg(feature = "cuda-runtime")]
        {
            let (buffer, offset, _) = self.storage.cuda_buffer_range()?;
            Some(DeviceMemoryRange::new(
                BackendKind::Cuda,
                buffer.device_ptr(),
                offset,
                self.byte_len(),
            ))
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            None
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
