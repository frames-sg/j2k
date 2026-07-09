// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use j2k::EncodedJ2k;
#[cfg(target_os = "macos")]
use j2k_core::{BackendKind, DeviceMemoryRange};
#[cfg(target_os = "macos")]
use metal::{foreign_types::ForeignType, Buffer};
#[cfg(target_os = "macos")]
use std::time::Duration;
#[cfg(target_os = "macos")]
use std::time::Instant;

#[cfg(target_os = "macos")]
/// JPEG 2000 codestream bytes owned by a Metal buffer.
///
/// The buffer is CPU-readable for the current padded resident encode API.
/// `codestream_bytes()` returns an owned snapshot so safe Rust never exposes a
/// slice that aliases the publicly accessible Metal buffer.
pub struct MetalEncodedJ2k {
    /// Metal buffer containing the codestream bytes.
    pub codestream_buffer: Buffer,
    /// Byte offset of the first codestream byte in `codestream_buffer`.
    pub byte_offset: usize,
    /// Number of valid codestream bytes.
    pub byte_len: usize,
    /// Allocated codestream capacity in bytes.
    pub capacity: usize,
    /// Encoded image width in pixels.
    pub width: u32,
    /// Encoded image height in pixels.
    pub height: u32,
    /// Number of encoded components.
    pub components: u8,
    /// Component bit depth.
    pub bit_depth: u8,
    /// Whether components are signed.
    pub signed: bool,
}

#[cfg(target_os = "macos")]
impl MetalEncodedJ2k {
    /// Backend-visible memory range for the valid codestream capacity.
    pub fn codestream_memory_range(&self) -> Option<DeviceMemoryRange> {
        Some(DeviceMemoryRange::new(
            BackendKind::Metal,
            u64::try_from(self.codestream_buffer.as_ptr() as usize).ok()?,
            self.byte_offset,
            self.capacity,
        ))
    }

    /// Backing Metal allocation length in bytes.
    pub fn codestream_allocation_len(&self) -> Option<usize> {
        usize::try_from(self.codestream_buffer.length()).ok()
    }

    /// Materialize the finished codestream bytes from the backing Metal buffer.
    pub fn codestream_bytes(&self) -> Result<Vec<u8>, crate::Error> {
        // SAFETY: Resident encode construction waits for the producing command
        // buffer before returning this value. Owned readback avoids aliasing
        // later access through the public `codestream_buffer` handle.
        match unsafe {
            j2k_metal_support::checked_buffer_read_vec::<u8>(
                &self.codestream_buffer,
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
