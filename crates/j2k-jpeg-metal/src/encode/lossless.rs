// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal lossless (SOF3) JPEG encode of a Metal-resident image.

#[cfg(target_os = "macos")]
use j2k_core::PixelFormat;
#[cfg(target_os = "macos")]
use j2k_jpeg::adapter::{
    plan_jpeg_lossless_gpu_encode, JpegLosslessGpuEncodeError, JpegLosslessGpuEncodeTile,
};
#[cfg(target_os = "macos")]
use j2k_jpeg::JpegBackend;
use j2k_jpeg::{EncodedJpeg, JpegLosslessEncodeOptions};
#[cfg(target_os = "macos")]
use j2k_metal_support::ResidentMetalImage;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer as _, MTLDevice as _, MTLResource as _};

#[cfg(target_os = "macos")]
use crate::metal_types::{BufferRef, DeviceRef};

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
/// Metal buffer and layout metadata for one lossless JPEG encode.
pub struct JpegLosslessMetalEncodeTile<'a> {
    buffer: &'a BufferRef,
    byte_offset: usize,
    width: u32,
    height: u32,
    pitch_bytes: usize,
    format: PixelFormat,
}

#[cfg(target_os = "macos")]
impl<'a> JpegLosslessMetalEncodeTile<'a> {
    /// Describe a Metal-resident `Gray8`, `Gray16`, `Rgb8`, or `Rgb16` image
    /// (native-endian 16-bit samples) for lossless encode.
    ///
    /// # Safety
    ///
    /// All commands that write the described range must have completed
    /// before construction, and the range must stay immutable to CPU and GPU
    /// writers while this tile or a copy can be used. The encode function is
    /// synchronous and waits for its reads before returning. The buffer must
    /// belong to the device of the session passed to it.
    pub unsafe fn new(
        buffer: &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        byte_offset: usize,
        dimensions: (u32, u32),
        pitch_bytes: usize,
        format: PixelFormat,
    ) -> Self {
        Self {
            buffer,
            byte_offset,
            width: dimensions.0,
            height: dimensions.1,
            pitch_bytes,
            format,
        }
    }

    /// Describe a validated resident image for lossless encode.
    #[must_use]
    pub fn from_resident(image: &'a ResidentMetalImage) -> Self {
        let layout = image.layout();
        Self {
            // SAFETY: the resident abstraction exposes the handle only for
            // audited read-only backend binding, and this tile borrows the
            // resident owner for its entire lifetime.
            buffer: unsafe { image.raw_buffer() },
            byte_offset: layout.byte_offset(),
            width: layout.dimensions().0,
            height: layout.dimensions().1,
            pitch_bytes: layout.pitch_bytes(),
            format: layout.pixel_format(),
        }
    }

    /// Byte offset of the first sample.
    #[must_use]
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Image dimensions in pixels.
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Bytes between consecutive rows.
    #[must_use]
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    /// Sample layout of the buffer.
    #[must_use]
    pub fn pixel_format(&self) -> PixelFormat {
        self.format
    }

    fn validate_device(&self, device: &DeviceRef) -> Result<(), crate::Error> {
        let image_registry_id = self.buffer.device().registryID();
        let requested_registry_id = device.registryID();
        if image_registry_id == requested_registry_id {
            return Ok(());
        }
        Err(crate::error::metal_kernel_support_error(
            "JPEG input belongs to a different Metal device",
            j2k_metal_support::MetalSupportError::MetalImageDeviceMismatch {
                image_registry_id,
                requested_registry_id,
            },
        ))
    }
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
/// Placeholder lossless encode tile type for non-macOS builds.
pub struct JpegLosslessMetalEncodeTile<'a> {
    _private: core::marker::PhantomData<&'a ()>,
}

/// Encode one Metal-resident image as a lossless (SOF3) JPEG.
///
/// Output is byte-identical to [`j2k_jpeg::encode_jpeg_lossless`] for the
/// same samples and options. `options.backend` must be `Auto` or `Metal`.
///
/// # Errors
///
/// Returns the CPU encoder's errors for invalid samples or options
/// (including a sample above the precision), and Metal errors for an input
/// range outside the buffer, misaligned 16-bit samples, a buffer on another
/// device, or a failed submission.
#[cfg(target_os = "macos")]
pub fn encode_jpeg_lossless_from_metal_buffer(
    tile: JpegLosslessMetalEncodeTile<'_>,
    options: JpegLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJpeg, crate::Error> {
    tile.validate_device(session.device())?;
    let plan = plan_jpeg_lossless_gpu_encode(
        JpegLosslessGpuEncodeTile {
            byte_offset: tile.byte_offset,
            width: tile.width,
            height: tile.height,
            pitch_bytes: tile.pitch_bytes,
            format: tile.format,
            buffer_len: tile.buffer.length(),
        },
        options,
        JpegBackend::Metal,
    )
    .map_err(lossless_plan_error)?;
    crate::compute::lossless_encode::encode_jpeg_lossless_with_session(session, tile.buffer, &plan)
}

/// Return `Error::MetalUnavailable` for lossless encode on non-macOS hosts.
#[cfg(not(target_os = "macos"))]
pub fn encode_jpeg_lossless_from_metal_buffer(
    tile: JpegLosslessMetalEncodeTile<'_>,
    options: JpegLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJpeg, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
fn lossless_plan_error(error: JpegLosslessGpuEncodeError) -> crate::Error {
    use j2k_core::CapabilityRejection;
    match error {
        JpegLosslessGpuEncodeError::Encode(error) => error.into(),
        JpegLosslessGpuEncodeError::UnsupportedBackend { .. } => crate::Error::capability_rejected(
            CapabilityRejection::unsupported_operation(
                "JPEG lossless Metal encode accepts only the Auto or Metal backend",
            ),
        ),
        JpegLosslessGpuEncodeError::PitchTooShort {
            row_bytes,
            pitch_bytes,
        } => crate::Error::MetalKernel {
            message: format!(
                "JPEG lossless Metal encode pitch is shorter than one row: need {row_bytes}, got {pitch_bytes}"
            ),
        },
        JpegLosslessGpuEncodeError::UnalignedSamples { alignment } => crate::Error::MetalKernel {
            message: format!(
                "JPEG lossless Metal encode needs {alignment}-byte aligned 16-bit samples"
            ),
        },
        JpegLosslessGpuEncodeError::InputRangeExceedsBuffer {
            required_end,
            buffer_len,
        } => crate::Error::MetalKernel {
            message: format!(
                "JPEG lossless Metal encode input range exceeds buffer length: need {required_end}, buffer has {buffer_len}"
            ),
        },
        JpegLosslessGpuEncodeError::TooLarge { reason } => crate::Error::capability_rejected(
            CapabilityRejection::resource_limit(reason),
        ),
    }
}
