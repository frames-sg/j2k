// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    copy_tight_pixels_to_strided_output, BackendKind, DeviceMemoryRange, DeviceSurface,
    ExecutionStats, PixelFormat, SurfaceMetadata, SurfaceResidency,
};
#[cfg(any(feature = "cuda-runtime", test))]
use j2k_core::{strided_output_len, validate_strided_output_buffer};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::CudaDeviceBuffer;

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
#[cfg(feature = "cuda-runtime")]
use crate::session::HostOwnerLease;
use crate::Error;

#[derive(Debug)]
pub(crate) enum Storage {
    Host(HostStorage),
    #[cfg(feature = "cuda-runtime")]
    Cuda(CudaDeviceBuffer),
}

#[derive(Debug)]
pub(crate) struct HostStorage {
    bytes: Vec<u8>,
    #[cfg(feature = "cuda-runtime")]
    _lease: HostOwnerLease,
}

impl HostStorage {
    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn new(bytes: Vec<u8>, lease: HostOwnerLease) -> Self {
        Self {
            bytes,
            _lease: lease,
        }
    }

    #[cfg(not(feature = "cuda-runtime"))]
    pub(crate) fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// CUDA JPEG decode path used to produce a surface.
#[doc(hidden)]
pub enum CudaJpegDecodePath {
    /// Surface did not use a CUDA JPEG decode kernel or library path.
    #[default]
    None,
    /// Surface was produced by J2K-owned CUDA JPEG kernels.
    OwnedCuda,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Dispatch counters and residency metadata for a CUDA JPEG surface.
#[doc(hidden)]
pub struct CudaSurfaceStats {
    pub(crate) kernel_dispatches: usize,
    pub(crate) copy_kernel_dispatches: usize,
    pub(crate) decode_kernel_dispatches: usize,
    pub(crate) hardware_decode: bool,
    pub(crate) decode_path: CudaJpegDecodePath,
}

#[doc(hidden)]
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
    #[doc(hidden)]
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
    fn metadata(&self) -> SurfaceMetadata {
        let residency = match &self.storage {
            Storage::Host(_) => SurfaceResidency::Host,
            #[cfg(feature = "cuda-runtime")]
            Storage::Cuda(_) => SurfaceResidency::CudaResidentDecode,
        };
        SurfaceMetadata::new(
            self.backend,
            residency,
            self.dimensions,
            self.fmt,
            self.pitch_bytes,
        )
    }

    /// Number of bytes between consecutive rows.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    /// Borrow host-resident bytes when this surface is not CUDA-backed.
    pub fn as_host_bytes(&self) -> Option<&[u8]> {
        match &self.storage {
            Storage::Host(storage) => Some(storage.as_slice()),
            #[cfg(feature = "cuda-runtime")]
            Storage::Cuda(_) => None,
        }
    }

    /// Copy surface bytes into a caller-provided strided output buffer.
    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        match &self.storage {
            Storage::Host(storage) => copy_tight_pixels_to_strided_output(
                storage.as_slice(),
                self.dimensions,
                self.fmt,
                out,
                stride,
            )
            .map_err(Error::from),
            #[cfg(feature = "cuda-runtime")]
            Storage::Cuda(buffer) => copy_cuda_surface_to_strided_output(
                self.dimensions,
                self.fmt,
                out,
                stride,
                |offset, destination| {
                    buffer
                        .copy_range_to_host(offset, destination)
                        .map_err(cuda_error)
                },
            ),
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

#[cfg(any(feature = "cuda-runtime", test))]
fn copy_cuda_surface_to_strided_output(
    dimensions: (u32, u32),
    fmt: PixelFormat,
    out: &mut [u8],
    stride: usize,
    mut copy_range: impl FnMut(usize, &mut [u8]) -> Result<(), Error>,
) -> Result<(), Error> {
    validate_strided_output_buffer(dimensions, out.len(), stride, fmt)?;
    if dimensions.0 == 0 || dimensions.1 == 0 {
        return Ok(());
    }

    let row_bytes = usize::try_from(dimensions.0)
        .ok()
        .and_then(|width| width.checked_mul(fmt.bytes_per_pixel()))
        .ok_or(j2k_core::BufferError::SizeOverflow {
            what: "CUDA JPEG surface row byte count",
        })?;
    let tight_len = strided_output_len(dimensions, row_bytes, fmt)?;
    if stride == row_bytes {
        return copy_range(0, &mut out[..tight_len]);
    }

    for row in 0..dimensions.1 as usize {
        let source_offset = row * row_bytes;
        let destination_offset = row * stride;
        copy_range(
            source_offset,
            &mut out[destination_offset..destination_offset + row_bytes],
        )?;
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::copy_cuda_surface_to_strided_output;
    use j2k_core::PixelFormat;

    #[test]
    fn tight_cuda_download_uses_one_direct_copy() {
        let source = (0_u8..12).collect::<Vec<_>>();
        let mut out = [0_u8; 12];
        let mut copies = Vec::new();

        copy_cuda_surface_to_strided_output(
            (2, 2),
            PixelFormat::Rgb8,
            &mut out,
            6,
            |offset, destination| {
                copies.push((offset, destination.len()));
                destination.copy_from_slice(&source[offset..offset + destination.len()]);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(out, source.as_slice());
        assert_eq!(copies, [(0, 12)]);
    }

    #[test]
    fn strided_cuda_download_copies_rows_without_frame_staging() {
        let source = (0_u8..12).collect::<Vec<_>>();
        let mut out = [0xee_u8; 16];
        let mut copies = Vec::new();

        copy_cuda_surface_to_strided_output(
            (2, 2),
            PixelFormat::Rgb8,
            &mut out,
            8,
            |offset, destination| {
                copies.push((offset, destination.len()));
                destination.copy_from_slice(&source[offset..offset + destination.len()]);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(&out[..6], &source[..6]);
        assert_eq!(&out[6..8], &[0xee, 0xee]);
        assert_eq!(&out[8..14], &source[6..12]);
        assert_eq!(&out[14..], &[0xee, 0xee]);
        assert_eq!(copies, [(0, 6), (6, 6)]);
    }

    #[test]
    fn invalid_cuda_download_layout_fails_before_copying() {
        let mut out = [0_u8; 12];
        let mut copied = false;

        let error =
            copy_cuda_surface_to_strided_output((2, 2), PixelFormat::Rgb8, &mut out, 5, |_, _| {
                copied = true;
                Ok(())
            })
            .expect_err("short stride must fail");

        assert!(matches!(
            error,
            crate::Error::Buffer(j2k_core::BufferError::StrideTooSmall {
                row_bytes: 6,
                stride: 5,
            })
        ));
        assert!(!copied);
    }
}
