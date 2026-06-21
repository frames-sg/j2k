// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use j2k::EncodedJ2k;
#[cfg(target_os = "macos")]
use j2k_core::BackendKind;
#[cfg(target_os = "macos")]
use metal::Buffer;
#[cfg(target_os = "macos")]
use std::time::Duration;
#[cfg(target_os = "macos")]
use std::time::Instant;

#[cfg(target_os = "macos")]
/// JPEG 2000 codestream bytes owned by a Metal buffer.
///
/// The buffer is CPU-readable for the current padded resident encode API, so
/// callers can stream `codestream_bytes()` into file or network writers without
/// first materializing an owned `Vec<u8>`.
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
    /// Borrow the finished codestream bytes from the backing Metal buffer.
    pub fn codestream_bytes(&self) -> Result<&[u8], crate::Error> {
        let end = self.byte_offset.checked_add(self.byte_len).ok_or_else(|| {
            crate::Error::MetalKernel {
                message: "J2K Metal codestream byte range overflow".to_string(),
            }
        })?;
        let buffer_len = usize::try_from(self.codestream_buffer.length()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal codestream buffer length exceeds usize".to_string(),
            }
        })?;
        if end > buffer_len {
            return Err(crate::Error::MetalKernel {
                message: "J2K Metal codestream byte range exceeds buffer length".to_string(),
            });
        }
        let ptr = self.codestream_buffer.contents().cast::<u8>();
        if ptr.is_null() {
            return Err(crate::Error::MetalKernel {
                message: "J2K Metal codestream buffer is not CPU-readable".to_string(),
            });
        }
        // SAFETY: Encoded Metal buffer views are bounds-checked before slice construction.
        Ok(unsafe { core::slice::from_raw_parts(ptr.add(self.byte_offset), self.byte_len) })
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
        let codestream = self.codestream_bytes()?.to_vec();
        let host_readback_duration = readback_started.elapsed();
        Ok((
            EncodedJ2k {
                codestream,
                backend: BackendKind::Metal,
                dispatch_report: j2k::adapter::encode_stage::J2kEncodeDispatchReport::default(),
                width: self.width,
                height: self.height,
                components: self.components,
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
