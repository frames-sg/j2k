// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use j2k::EncodedJ2k;
#[cfg(target_os = "macos")]
use j2k_core::{BackendKind, DeviceMemoryRange};
#[cfg(target_os = "macos")]
use metal::{foreign_types::ForeignType, Buffer};
#[cfg(target_os = "macos")]
use std::ops::Range;
#[cfg(target_os = "macos")]
use std::time::Duration;
#[cfg(target_os = "macos")]
use std::time::Instant;

#[cfg(target_os = "macos")]
/// JPEG 2000 codestream bytes owned by a Metal buffer.
///
/// The buffer is CPU-readable for the current padded resident encode API.
/// `codestream_bytes()` returns an owned snapshot. Access to the backing Metal
/// handle is unsafe because Metal synchronization cannot be represented by a
/// Rust borrow.
pub struct MetalEncodedJ2k {
    pub(crate) codestream_buffer: Buffer,
    pub(crate) byte_offset: usize,
    pub(crate) byte_len: usize,
    pub(crate) capacity: usize,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) components: u8,
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
}

#[cfg(target_os = "macos")]
impl MetalEncodedJ2k {
    fn try_from_parts(
        codestream_buffer: Buffer,
        codestream_range: Range<usize>,
        capacity: usize,
        dimensions: (u32, u32),
        components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, crate::Error> {
        let byte_len = codestream_range
            .end
            .checked_sub(codestream_range.start)
            .ok_or_else(|| crate::Error::MetalKernel {
                message: "J2K Metal codestream range ends before it starts".to_string(),
            })?;
        if byte_len > capacity {
            return Err(crate::Error::MetalKernel {
                message: format!(
                    "J2K Metal codestream length {byte_len} exceeds capacity {capacity}"
                ),
            });
        }
        let capacity_end = codestream_range
            .start
            .checked_add(capacity)
            .ok_or_else(|| crate::Error::MetalKernel {
                message: "J2K Metal codestream capacity range overflows usize".to_string(),
            })?;
        let allocation_len =
            usize::try_from(codestream_buffer.length()).map_err(|_| crate::Error::MetalKernel {
                message: "J2K Metal codestream allocation length exceeds usize".to_string(),
            })?;
        if capacity_end > allocation_len {
            return Err(crate::Error::MetalKernel {
                message: format!(
                    "J2K Metal codestream capacity range {}..{capacity_end} exceeds allocation length {allocation_len}",
                    codestream_range.start
                ),
            });
        }

        Ok(Self {
            codestream_buffer,
            byte_offset: codestream_range.start,
            byte_len,
            capacity,
            width: dimensions.0,
            height: dimensions.1,
            components,
            bit_depth,
            signed,
        })
    }

    /// Construct an encoded codestream from a caller-owned Metal allocation.
    ///
    /// `codestream_range` identifies the valid bytes and `capacity` starts at
    /// the same offset. Both ranges are validated against the allocation.
    ///
    /// # Safety
    ///
    /// All CPU and Metal commands that can write the codestream capacity range
    /// must have completed before this call. No CPU or GPU access may mutate
    /// that range until the returned object is dropped. These obligations also
    /// apply to every handle cloned from `codestream_buffer` before this call.
    pub unsafe fn from_raw_parts(
        codestream_buffer: Buffer,
        codestream_range: Range<usize>,
        capacity: usize,
        dimensions: (u32, u32),
        components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, crate::Error> {
        Self::try_from_parts(
            codestream_buffer,
            codestream_range,
            capacity,
            dimensions,
            components,
            bit_depth,
            signed,
        )
    }

    pub(crate) fn from_completed_buffer(
        codestream_buffer: Buffer,
        codestream_range: Range<usize>,
        capacity: usize,
        dimensions: (u32, u32),
        components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, crate::Error> {
        Self::try_from_parts(
            codestream_buffer,
            codestream_range,
            capacity,
            dimensions,
            components,
            bit_depth,
            signed,
        )
    }

    /// Consume this output and return its backing Metal allocation.
    ///
    /// Read metadata such as [`Self::byte_offset`] and [`Self::capacity`]
    /// before calling this method when it is needed for the handoff.
    ///
    /// # Safety
    ///
    /// Other encoded outputs, including sibling tiles in a batch, may share
    /// this allocation even though this value is consumed. The caller must
    /// ensure that no CPU or GPU access through the returned handle (or a clone)
    /// mutates any range while a sharing `MetalEncodedJ2k` remains alive. All
    /// prior writers must complete before any sharing output is read back.
    pub unsafe fn into_codestream_buffer(self) -> Buffer {
        self.codestream_buffer
    }

    pub(crate) fn codestream_buffer_trusted(&self) -> &Buffer {
        &self.codestream_buffer
    }

    /// Byte offset of the first valid codestream byte.
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Number of valid codestream bytes.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Codestream capacity in bytes, beginning at [`Self::byte_offset`].
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Encoded image dimensions in pixels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Encoded image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Encoded image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Number of encoded components.
    pub fn components(&self) -> u8 {
        self.components
    }

    /// Component bit depth.
    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    /// Whether component samples are signed.
    pub fn is_signed(&self) -> bool {
        self.signed
    }

    /// Backend-visible memory range for the valid codestream capacity.
    pub fn codestream_memory_range(&self) -> Option<DeviceMemoryRange> {
        Some(DeviceMemoryRange::new(
            BackendKind::Metal,
            u64::try_from(self.codestream_buffer_trusted().as_ptr() as usize).ok()?,
            self.byte_offset,
            self.capacity,
        ))
    }

    /// Backing Metal allocation length in bytes.
    pub fn codestream_allocation_len(&self) -> Option<usize> {
        usize::try_from(self.codestream_buffer_trusted().length()).ok()
    }

    /// Materialize the finished codestream bytes from the backing Metal buffer.
    pub fn codestream_bytes(&self) -> Result<Vec<u8>, crate::Error> {
        // SAFETY: Resident encode construction waits for the producing command
        // buffer before returning this value. Any external handle can only be
        // obtained through an unsafe API whose contract excludes overlapping
        // mutation.
        match unsafe {
            j2k_metal_support::checked_buffer_read_vec::<u8>(
                self.codestream_buffer_trusted(),
                self.byte_offset,
                self.byte_len,
            )
        } {
            Ok(bytes) => Ok(bytes),
            Err(j2k_metal_support::MetalSupportError::BufferContentsUnavailable) => {
                Err(crate::Error::MetalKernel {
                    message: "J2K Metal codestream buffer is not CPU-readable".to_string(),
                })
            }
            Err(error) => Err(crate::Error::MetalKernel {
                message: format!("J2K Metal codestream byte range invalid: {error}"),
            }),
        }
    }

    /// Materialize the buffer-backed codestream into the compatibility `Vec` API shape.
    pub fn to_encoded_j2k(&self) -> Result<EncodedJ2k, crate::Error> {
        let (encoded, _host_readback_duration) = self.to_encoded_j2k_with_readback_duration()?;
        Ok(encoded)
    }

    pub(super) fn to_encoded_j2k_with_readback_duration(
        &self,
    ) -> Result<(EncodedJ2k, Duration), crate::Error> {
        let readback_started = Instant::now();
        let codestream = self.codestream_bytes()?;
        let host_readback_duration = readback_started.elapsed();
        Ok((
            EncodedJ2k {
                codestream,
                backend: BackendKind::Metal,
                dispatch_report: j2k::J2kEncodeDispatchReport::default(),
                width: self.width,
                height: self.height,
                components: u16::from(self.components),
                bit_depth: self.bit_depth,
                signed: self.signed,
            },
            host_readback_duration,
        ))
    }
}

#[cfg(not(target_os = "macos"))]
/// Placeholder Metal codestream type for non-macOS builds.
pub struct MetalEncodedJ2k {
    _private: (),
}
