// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{borrow::Cow, ops::Range, sync::Arc};

use j2k_core::{
    copy_tight_pixels_to_strided_output, BackendKind, DeviceMemoryRange, DeviceSurface,
    PixelFormat, SurfaceMetadata, SurfaceResidency,
};
#[cfg(target_os = "macos")]
use metal::foreign_types::ForeignType;
#[cfg(target_os = "macos")]
use metal::Buffer;

#[cfg(target_os = "macos")]
use crate::error::metal_kernel_support_error;
use crate::Error;

#[derive(Clone)]
pub(crate) enum Storage {
    Host(Arc<Vec<u8>>),
    #[cfg(target_os = "macos")]
    Metal(Buffer),
}

impl Storage {
    pub(crate) fn from_host(bytes: Vec<u8>) -> Self {
        Self::Host(Arc::new(bytes))
    }
}

#[derive(Clone)]
/// Decoded J2K image surface returned by the Metal backend.
pub struct Surface {
    pub(crate) backend: BackendKind,
    pub(crate) residency: SurfaceResidency,
    pub(crate) dimensions: (u32, u32),
    pub(crate) fmt: PixelFormat,
    pub(crate) pitch_bytes: usize,
    pub(crate) byte_offset: usize,
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
        .with_byte_offset(self.byte_offset)
    }

    /// Current residency for the surface bytes.
    pub fn residency(&self) -> SurfaceResidency {
        self.residency
    }

    /// Number of bytes between consecutive rows.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    fn checked_storage_range(&self, storage_len: usize) -> Result<Range<usize>, Error> {
        let len = self.byte_len();
        let end = self
            .byte_offset
            .checked_add(len)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal surface byte range overflows usize".to_string(),
            })?;
        if end > storage_len {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal surface byte range {start}..{end} exceeds storage length {storage_len}",
                    start = self.byte_offset
                ),
            });
        }
        Ok(self.byte_offset..end)
    }

    fn storage_bytes(&self) -> Result<Cow<'_, [u8]>, Error> {
        match &self.storage {
            Storage::Host(bytes) => {
                let range = self.checked_storage_range(bytes.len())?;
                Ok(Cow::Borrowed(&bytes[range]))
            }
            #[cfg(target_os = "macos")]
            Storage::Metal(buffer) => {
                // SAFETY: A returned `Surface` represents a completed decode.
                // External access to the handle is unsafe and requires callers
                // to exclude overlapping mutation during this owned readback.
                match unsafe {
                    j2k_metal_support::checked_buffer_read_vec::<u8>(
                        buffer,
                        self.byte_offset,
                        self.byte_len(),
                    )
                } {
                    Ok(bytes) => Ok(Cow::Owned(bytes)),
                    Err(
                        error @ j2k_metal_support::MetalSupportError::BufferContentsUnavailable,
                    ) => Err(metal_kernel_support_error(
                        "J2K Metal surface buffer is not host-addressable",
                        error,
                    )),
                    Err(error) => Err(metal_kernel_support_error(
                        format!("J2K Metal surface byte range invalid: {error}"),
                        error,
                    )),
                }
            }
        }
    }

    /// Return the tightly packed surface bytes.
    ///
    /// Host-backed surfaces are borrowed without copying. Metal-backed surfaces
    /// are copied into owned storage. Synchronization, readback, and validated
    /// range failures are returned through the backend's typed error contract.
    pub fn as_bytes(&self) -> Result<Cow<'_, [u8]>, Error> {
        self.storage_bytes()
    }

    /// Copy the tightly packed surface into a caller-provided strided buffer.
    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        let storage = self.storage_bytes()?;
        copy_tight_pixels_to_strided_output(
            storage.as_ref(),
            self.dimensions,
            self.fmt,
            out,
            stride,
        )
        .map_err(Error::from)
    }

    #[cfg(target_os = "macos")]
    /// Return the Metal buffer and byte offset when the surface is Metal-backed.
    ///
    /// # Safety
    ///
    /// All prior writers must have completed before this call. The caller must
    /// ensure that no CPU or GPU access through the returned handle (or a clone
    /// of it) mutates the surface range while this surface or any clone sharing
    /// the allocation remains alive.
    pub unsafe fn metal_buffer(&self) -> Option<(&Buffer, usize)> {
        self.metal_buffer_trusted()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn metal_buffer_trusted(&self) -> Option<(&Buffer, usize)> {
        match &self.storage {
            Storage::Metal(buffer) => Some((buffer, self.byte_offset)),
            Storage::Host(_) => None,
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
    ) -> Self {
        Self {
            backend: BackendKind::Metal,
            residency: SurfaceResidency::MetalResidentDecode,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            byte_offset: 0,
            storage: Storage::Metal(buffer),
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer_with_offset(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        byte_offset: usize,
    ) -> Self {
        Self {
            backend: BackendKind::Metal,
            residency: SurfaceResidency::MetalResidentDecode,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            byte_offset,
            storage: Storage::Metal(buffer),
        }
    }
}

#[doc(hidden)]
impl DeviceSurface for Surface {
    fn backend_kind(&self) -> BackendKind {
        self.metadata().backend
    }

    fn residency(&self) -> SurfaceResidency {
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

    fn memory_range(&self) -> Option<DeviceMemoryRange> {
        match &self.storage {
            Storage::Host(_) => None,
            #[cfg(target_os = "macos")]
            Storage::Metal(buffer) => Some(DeviceMemoryRange::new(
                BackendKind::Metal,
                u64::try_from(buffer.as_ptr() as usize).ok()?,
                self.byte_offset,
                self.byte_len(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Storage, Surface};
    use crate::Error;
    use j2k_core::{BackendKind, PixelFormat, SurfaceResidency};

    fn host_surface(bytes: Vec<u8>, byte_offset: usize) -> Surface {
        Surface {
            backend: BackendKind::Cpu,
            residency: SurfaceResidency::Host,
            dimensions: (2, 1),
            fmt: PixelFormat::Gray8,
            pitch_bytes: 2,
            byte_offset,
            storage: Storage::from_host(bytes),
        }
    }

    #[test]
    fn host_backed_byte_access_borrows_the_validated_range() {
        let surface = host_surface(vec![9, 1, 2, 8], 1);
        let bytes = surface.as_bytes().expect("valid host surface bytes");

        assert!(matches!(bytes, std::borrow::Cow::Borrowed(_)));
        assert_eq!(bytes.as_ref(), [1, 2]);
    }

    #[test]
    fn invalid_host_backed_range_returns_an_error_without_panicking() {
        let surface = host_surface(vec![1, 2], 1);
        let error = surface
            .as_bytes()
            .expect_err("out-of-range host surface must fail");

        assert!(matches!(error, Error::MetalKernel { .. }));
    }

    #[test]
    fn cloning_a_host_surface_shares_the_pixel_owner() {
        let surface = host_surface(vec![1, 2], 0);
        let cloned = surface.clone();

        let (Storage::Host(original), Storage::Host(clone)) = (&surface.storage, &cloned.storage)
        else {
            panic!("host surface clone must preserve host storage");
        };
        assert!(std::sync::Arc::ptr_eq(original, clone));
    }
}
