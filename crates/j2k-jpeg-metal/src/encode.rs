// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use j2k_core::PixelFormat;
#[cfg(target_os = "macos")]
use j2k_jpeg::adapter::{encode_jpeg_baseline_gpu_batch, encode_jpeg_baseline_gpu_tile};
use j2k_jpeg::{EncodedJpeg, JpegEncodeOptions};
#[cfg(target_os = "macos")]
use metal::{Buffer, BufferRef};

#[cfg(target_os = "macos")]
mod adapter;
#[cfg(any(target_os = "macos", test))]
pub(crate) mod allocation;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
/// Metal buffer and layout metadata for one baseline JPEG encode tile.
pub struct JpegBaselineMetalEncodeTile<'a> {
    buffer: &'a Buffer,
    byte_offset: usize,
    width: u32,
    height: u32,
    pitch_bytes: usize,
    output_width: u32,
    output_height: u32,
    format: PixelFormat,
}

#[cfg(target_os = "macos")]
impl<'a> JpegBaselineMetalEncodeTile<'a> {
    /// Describe one Metal-resident source tile for baseline JPEG encoding.
    ///
    /// # Safety
    ///
    /// All commands that write the described source range must have completed
    /// before construction. The caller must keep that range immutable to both
    /// CPU and GPU writers while this tile or any copy can be used, and through
    /// actual completion of every GPU read submitted from one. The provided
    /// encode functions are synchronous and wait for those reads before
    /// returning. The buffer must be usable by the device behind each session
    /// passed to the safe encode functions.
    pub unsafe fn new(
        buffer: &'a Buffer,
        byte_offset: usize,
        dimensions: (u32, u32),
        pitch_bytes: usize,
        output_dimensions: (u32, u32),
        format: PixelFormat,
    ) -> Self {
        Self {
            buffer,
            byte_offset,
            width: dimensions.0,
            height: dimensions.1,
            pitch_bytes,
            output_width: output_dimensions.0,
            output_height: output_dimensions.1,
            format,
        }
    }

    /// Byte offset of the first source pixel.
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Dimensions of the valid input region.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Number of bytes between consecutive input rows.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    /// Dimensions of the encoded JPEG frame.
    pub fn output_dimensions(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }

    /// Pixel format of the source buffer.
    pub fn pixel_format(&self) -> PixelFormat {
        self.format
    }

    /// Return the raw Metal source buffer.
    ///
    /// # Safety
    ///
    /// The caller must preserve the synchronization and immutability contract
    /// established by [`JpegBaselineMetalEncodeTile::new`].
    pub unsafe fn buffer(&self) -> &BufferRef {
        self.buffer_trusted().as_ref()
    }

    pub(crate) fn buffer_trusted(&self) -> &'a Buffer {
        self.buffer
    }
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
/// Placeholder encode tile type for non-macOS builds.
pub struct JpegBaselineMetalEncodeTile<'a> {
    _private: core::marker::PhantomData<&'a ()>,
}

#[cfg(target_os = "macos")]
/// Encode one Metal-resident tile as a baseline JPEG frame.
pub fn encode_jpeg_baseline_from_metal_buffer(
    tile: JpegBaselineMetalEncodeTile<'_>,
    options: JpegEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJpeg, crate::Error> {
    let mut adapter = MetalJpegBaselineEncodeAdapter { session };
    encode_jpeg_baseline_gpu_tile(tile, options, &mut adapter)
}

#[cfg(target_os = "macos")]
/// Encode multiple Metal-resident tiles as baseline JPEG frames.
///
/// Consecutive tiles that share a source Metal buffer are submitted through a
/// single entropy-kernel batch where possible. The returned frames preserve the
/// input order.
pub fn encode_jpeg_baseline_batch_from_metal_buffers(
    tiles: &[JpegBaselineMetalEncodeTile<'_>],
    options: JpegEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJpeg>, crate::Error> {
    let mut adapter = MetalJpegBaselineEncodeAdapter { session };
    encode_jpeg_baseline_gpu_batch(tiles, options, &mut adapter)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for batch Metal encode requests on non-macOS hosts.
pub fn encode_jpeg_baseline_batch_from_metal_buffers(
    tiles: &[JpegBaselineMetalEncodeTile<'_>],
    options: JpegEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJpeg>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for Metal encode requests on non-macOS hosts.
pub fn encode_jpeg_baseline_from_metal_buffer(
    tile: JpegBaselineMetalEncodeTile<'_>,
    options: JpegEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJpeg, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
struct MetalJpegBaselineEncodeAdapter<'a> {
    session: &'a crate::MetalBackendSession,
}
