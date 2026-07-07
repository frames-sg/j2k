// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::Range;

use j2k_core::{
    copy_tight_pixels_to_strided_output, BackendKind, DeviceMemoryRange, DeviceSurface,
    PixelFormat, SurfaceMetadata, SurfaceResidency,
};
#[cfg(target_os = "macos")]
use metal::foreign_types::ForeignType;
#[cfg(target_os = "macos")]
use metal::Buffer;

use crate::Error;

#[derive(Clone)]
pub(crate) enum Storage {
    Host(Vec<u8>),
    #[cfg(target_os = "macos")]
    Metal(Buffer),
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

    fn storage_bytes(&self) -> Result<&[u8], Error> {
        match &self.storage {
            Storage::Host(bytes) => {
                let range = self.checked_storage_range(bytes.len())?;
                Ok(&bytes[range])
            }
            #[cfg(target_os = "macos")]
            Storage::Metal(buffer) => {
                match j2k_metal_support::checked_buffer_contents_slice::<u8>(
                    buffer,
                    self.byte_offset,
                    self.byte_len(),
                ) {
                    Ok(bytes) => Ok(bytes),
                    Err(j2k_metal_support::MetalSupportError::BufferContentsUnavailable) => {
                        Err(Error::MetalKernel {
                            message: "J2K Metal surface buffer is not host-addressable".to_string(),
                        })
                    }
                    Err(error) => Err(Error::MetalKernel {
                        message: format!("J2K Metal surface byte range invalid: {error}"),
                    }),
                }
            }
        }
    }

    /// Return the tightly packed surface bytes.
    ///
    /// Metal-backed surfaces are expected to use host-addressable buffers. This
    /// method panics only if the surface metadata is internally inconsistent;
    /// fallible operations such as [`Self::download_into`] return those errors.
    pub fn as_bytes(&self) -> &[u8] {
        self.storage_bytes()
            .expect("validated J2K Metal surface byte range")
    }

    /// Copy the tightly packed surface into a caller-provided strided buffer.
    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        copy_tight_pixels_to_strided_output(
            self.storage_bytes()?,
            self.dimensions,
            self.fmt,
            out,
            stride,
        )
        .map_err(Error::from)
    }

    #[cfg(target_os = "macos")]
    /// Return the Metal buffer and byte offset when the surface is Metal-backed.
    pub fn metal_buffer(&self) -> Option<(&Buffer, usize)> {
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
