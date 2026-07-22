// SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;
use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::sync::Mutex;

use j2k_core::{
    copy_tight_pixels_to_strided_output, BackendKind, DeviceMemoryRange, DeviceSurface,
    PixelFormat, SurfaceMetadata, SurfaceResidency,
};
#[cfg(target_os = "macos")]
use j2k_metal_support::{MetalImageLayout, ResidentMetalImage};

#[cfg(target_os = "macos")]
use crate::buffers::checked_buffer_slice_at;
#[cfg(target_os = "macos")]
use crate::error::metal_kernel_support_error;
use crate::Error;

#[cfg(target_os = "macos")]
use metal::foreign_types::ForeignType;
#[cfg(target_os = "macos")]
use metal::Buffer;

#[cfg(target_os = "macos")]
mod batch_buffer;
#[cfg(target_os = "macos")]
mod batch_texture;
#[cfg(target_os = "macos")]
mod resident_tile;
#[cfg(target_os = "macos")]
mod texture_tile;
#[cfg(target_os = "macos")]
pub use batch_buffer::MetalBatchOutputBuffer;
#[cfg(target_os = "macos")]
pub use batch_texture::MetalBatchTextureOutput;
#[cfg(target_os = "macos")]
pub use resident_tile::ResidentPrivateJpegTile;
#[cfg(target_os = "macos")]
pub use texture_tile::MetalTextureTile;

#[derive(Clone)]
pub(crate) enum Storage {
    Host(Arc<Vec<u8>>),
    #[cfg(target_os = "macos")]
    Metal {
        resident: Option<ResidentMetalImage>,
        reusable_buffer: Option<Buffer>,
        offset: usize,
        access_gate: Option<Arc<Mutex<()>>>,
    },
}

#[derive(Clone)]
/// Decoded image surface returned by the JPEG Metal backend.
pub struct Surface {
    pub(crate) backend: BackendKind,
    pub(crate) residency: SurfaceResidency,
    pub(crate) dimensions: (u32, u32),
    pub(crate) fmt: PixelFormat,
    pub(crate) pitch_bytes: usize,
    pub(crate) storage: Storage,
}

impl Surface {
    pub(crate) fn retained_host_capacity_bytes(&self) -> usize {
        match &self.storage {
            Storage::Host(bytes) => bytes.capacity(),
            #[cfg(target_os = "macos")]
            Storage::Metal { .. } => 0,
        }
    }

    fn metadata(&self) -> SurfaceMetadata {
        SurfaceMetadata::new(
            self.backend,
            self.residency,
            self.dimensions,
            self.fmt,
            self.pitch_bytes,
        )
    }

    /// Number of bytes between consecutive rows.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    /// Current residency for the surface bytes.
    pub fn residency(&self) -> SurfaceResidency {
        self.residency
    }

    /// Return the tightly packed surface bytes.
    ///
    /// Host storage is borrowed. Metal storage is copied into an owned snapshot
    /// so safe Rust never exposes a slice that aliases later GPU access.
    /// Synchronization, access-gate, and checked readback failures are returned
    /// through the backend's typed error contract.
    pub fn as_bytes(&self) -> Result<Cow<'_, [u8]>, Error> {
        self.storage_bytes()
    }

    #[cfg_attr(
        not(target_os = "macos"),
        expect(
            clippy::unnecessary_wraps,
            reason = "the host-only branch preserves the fallible Metal readback contract"
        )
    )]
    fn storage_bytes(&self) -> Result<Cow<'_, [u8]>, Error> {
        match &self.storage {
            Storage::Host(bytes) => Ok(Cow::Borrowed(bytes.as_slice())),
            #[cfg(target_os = "macos")]
            Storage::Metal {
                resident,
                reusable_buffer,
                offset,
                access_gate,
            } => {
                let _access = match access_gate {
                    Some(gate) => Some(gate.lock().map_err(|_| Error::MetalStatePoisoned {
                        state: "surface access gate",
                    })?),
                    None => None,
                };
                let len = self.byte_len();
                let buffer = match (resident, reusable_buffer) {
                    (Some(image), None) => {
                        // SAFETY: completed resident surfaces are read only.
                        unsafe { image.raw_buffer() }
                    }
                    (None, Some(buffer)) => buffer,
                    _ => {
                        return Err(Error::MetalKernel {
                            message: "JPEG Metal surface storage invariant failed".to_string(),
                        })
                    }
                };
                checked_buffer_slice_at::<u8>(buffer, *offset, len, "surface bytes").map(Cow::Owned)
            }
        }
    }

    /// Copy the tightly packed surface into a caller-provided strided buffer.
    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        let bytes = self.storage_bytes()?;
        copy_tight_pixels_to_strided_output(bytes.as_ref(), self.dimensions, self.fmt, out, stride)
            .map_err(Error::from)
    }

    #[cfg(target_os = "macos")]
    /// Return the raw Metal buffer and byte offset when the surface is Metal-backed.
    ///
    /// # Safety
    ///
    /// The caller must synchronize every CPU and GPU access made through the
    /// returned buffer or any handle cloned from it. The internal safe-access
    /// gate cannot observe work submitted through raw handles. In particular,
    /// no command may write the surface range while [`Surface::as_bytes`] or
    /// [`Surface::download_into`] reads it, and no raw access may overlap a safe
    /// decoder write through an aliasing [`MetalBatchOutputBuffer`].
    pub unsafe fn metal_buffer(&self) -> Option<(&Buffer, usize)> {
        self.metal_buffer_trusted()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn metal_buffer_trusted(&self) -> Option<(&Buffer, usize)> {
        match &self.storage {
            Storage::Metal {
                resident: Some(image),
                reusable_buffer: None,
                offset,
                ..
            } => {
                // SAFETY: backend code binds this private handle under the
                // immutable resident-image contract.
                Some((unsafe { image.raw_buffer() }, *offset))
            }
            Storage::Metal {
                resident: None,
                reusable_buffer: Some(buffer),
                offset,
                ..
            } => Some((buffer, *offset)),
            Storage::Metal { .. } | Storage::Host(_) => None,
        }
    }

    #[cfg(target_os = "macos")]
    /// Borrow the immutable resident image for a completed, non-reusable surface.
    pub fn resident_metal_image(&self) -> Option<&ResidentMetalImage> {
        match &self.storage {
            Storage::Metal {
                resident: Some(image),
                ..
            } => Some(image),
            Storage::Host(_) | Storage::Metal { .. } => None,
        }
    }

    #[cfg(target_os = "macos")]
    /// Consume this surface and return its immutable resident image.
    ///
    /// Reusable batch-output surfaces return `None` because their backing
    /// allocation remains intentionally mutable behind an access gate.
    pub fn into_resident_metal_image(self) -> Option<ResidentMetalImage> {
        match self.storage {
            Storage::Metal {
                resident: Some(image),
                ..
            } => Some(image),
            Storage::Host(_) | Storage::Metal { .. } => None,
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
    ) -> Result<Self, Error> {
        Self::from_metal_buffer_offset(buffer, dimensions, fmt, 0)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer_offset(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        offset: usize,
    ) -> Result<Self, Error> {
        Self::from_owned_metal_buffer_offset(
            buffer,
            dimensions,
            fmt,
            offset,
            SurfaceResidency::MetalResidentDecode,
        )
    }

    #[cfg(target_os = "macos")]
    fn from_owned_metal_buffer_offset(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        offset: usize,
        residency: SurfaceResidency,
    ) -> Result<Self, Error> {
        let pitch_bytes = usize::try_from(dimensions.0)
            .ok()
            .and_then(|width| width.checked_mul(fmt.bytes_per_pixel()))
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal surface pitch overflows usize".to_string(),
            })?;
        let layout = MetalImageLayout::new(offset, dimensions, pitch_bytes, fmt)
            .map_err(|source| metal_kernel_support_error("JPEG Metal surface layout", source))?;
        // SAFETY: these crate-private constructors are used by producer paths
        // that retain and complete the sole writer before exposing a surface.
        let image = unsafe { ResidentMetalImage::from_exclusive_pending_buffer(buffer, layout) }
            .map_err(|source| metal_kernel_support_error("JPEG Metal resident surface", source))?;
        Ok(Self {
            backend: BackendKind::Metal,
            residency,
            dimensions,
            fmt,
            pitch_bytes,
            storage: Storage::Metal {
                resident: Some(image),
                reusable_buffer: None,
                offset,
                access_gate: None,
            },
        })
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_cpu_staged_metal_buffer(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
    ) -> Result<Self, Error> {
        Self::from_cpu_staged_metal_buffer_offset(buffer, dimensions, fmt, 0)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_cpu_staged_metal_buffer_offset(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        offset: usize,
    ) -> Result<Self, Error> {
        Self::from_owned_metal_buffer_offset(
            buffer,
            dimensions,
            fmt,
            offset,
            SurfaceResidency::CpuStagedMetalUpload,
        )
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_batch_output_buffer_offset(
        output: &MetalBatchOutputBuffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        offset: usize,
    ) -> Self {
        Self {
            backend: BackendKind::Metal,
            residency: SurfaceResidency::MetalResidentDecode,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            storage: Storage::Metal {
                resident: None,
                reusable_buffer: Some(output.buffer.clone()),
                offset,
                access_gate: Some(Arc::clone(&output.access_gate)),
            },
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
            Storage::Metal { .. } => {
                let (buffer, offset) = self.metal_buffer_trusted()?;
                Some(DeviceMemoryRange::new(
                    BackendKind::Metal,
                    u64::try_from(buffer.as_ptr() as usize).ok()?,
                    offset,
                    self.byte_len(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod surface_access_tests {
    use std::sync::Arc;

    use super::{Storage, Surface};
    use j2k_core::{BackendKind, PixelFormat, SurfaceResidency};

    #[test]
    fn host_backed_byte_access_remains_borrowed_and_fallible() {
        let surface = Surface {
            backend: BackendKind::Cpu,
            residency: SurfaceResidency::Host,
            dimensions: (2, 1),
            fmt: PixelFormat::Gray8,
            pitch_bytes: 2,
            storage: Storage::Host(Arc::new(vec![1, 2])),
        };
        let bytes = surface.as_bytes().expect("valid host surface bytes");

        assert!(matches!(bytes, std::borrow::Cow::Borrowed(_)));
        assert_eq!(bytes.as_ref(), [1, 2]);
    }

    #[test]
    fn cloning_host_surface_shares_immutable_payload_allocation() {
        let surface = Surface {
            backend: BackendKind::Cpu,
            residency: SurfaceResidency::Host,
            dimensions: (4, 1),
            fmt: PixelFormat::Gray8,
            pitch_bytes: 4,
            storage: Storage::Host(Arc::new(vec![1, 2, 3, 4])),
        };
        let cloned = surface.clone();

        #[cfg(target_os = "macos")]
        let (Storage::Host(original), Storage::Host(shared)) = (&surface.storage, &cloned.storage) else {
            panic!("host surfaces must remain host-backed after clone");
        };
        #[cfg(not(target_os = "macos"))]
        let (Storage::Host(original), Storage::Host(shared)) = (&surface.storage, &cloned.storage);
        assert!(Arc::ptr_eq(original, shared));
        assert_eq!(original.capacity(), shared.capacity());
    }
}
