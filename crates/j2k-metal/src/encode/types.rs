// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::EncodedJ2k;
#[cfg(target_os = "macos")]
use j2k_core::PixelFormat;
#[cfg(target_os = "macos")]
use metal::Buffer;
use std::time::Duration;

use super::MetalEncodedJ2k;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
/// Metal buffer and layout metadata for one lossless J2K encode tile.
pub struct MetalLosslessEncodeTile<'a> {
    pub(super) buffer: &'a Buffer,
    pub(super) byte_offset: usize,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pitch_bytes: usize,
    pub(super) output_width: u32,
    pub(super) output_height: u32,
    pub(super) format: PixelFormat,
}

#[cfg(target_os = "macos")]
impl<'a> MetalLosslessEncodeTile<'a> {
    /// Describe an immutable Metal-buffer region to the lossless encoder.
    ///
    /// Geometry and allocation bounds are validated by the encode operation.
    ///
    /// # Safety
    ///
    /// All CPU and Metal commands that can write the described source region
    /// must have completed before this call. The caller must prevent CPU and GPU
    /// mutation of that region from this call until every encode submission
    /// derived from the tile has actually completed, including deferred
    /// submissions that outlive the tile value. Dropping a submitted operation
    /// without waiting does not end this obligation unless completion is
    /// established independently. The obligation includes handles cloned before
    /// this call and outlives copies of the tile. The buffer must belong to the
    /// same Metal device as every [`crate::MetalBackendSession`] later used to
    /// encode, submit, or validate this tile; a buffer from another device is
    /// not compatible even when its layout and storage mode otherwise match.
    pub unsafe fn from_buffer(
        buffer: &'a Buffer,
        byte_offset: usize,
        dimensions: (u32, u32),
        pitch_bytes: usize,
        output_dimensions: (u32, u32),
        format: PixelFormat,
    ) -> Self {
        Self::from_trusted_buffer(
            buffer,
            byte_offset,
            dimensions,
            pitch_bytes,
            output_dimensions,
            format,
        )
    }

    pub(crate) fn from_trusted_buffer(
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
    pub fn byte_offset(self) -> usize {
        self.byte_offset
    }

    /// Dimensions of the valid source region.
    pub fn dimensions(self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Number of bytes between consecutive source rows.
    pub fn pitch_bytes(self) -> usize {
        self.pitch_bytes
    }

    /// Encoded output dimensions.
    pub fn output_dimensions(self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }

    /// Pixel format of the source region.
    pub fn pixel_format(self) -> PixelFormat {
        self.format
    }
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
/// Placeholder lossless encode tile type for non-macOS builds.
pub struct MetalLosslessEncodeTile<'a> {
    _private: core::marker::PhantomData<&'a ()>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Residency decisions used by a lossless Metal encode.
pub struct MetalLosslessEncodeResidency {
    /// Whether coefficient preparation ran on Metal.
    pub coefficient_prep_used: bool,
    /// Whether packetization ran on Metal.
    pub packetization_used: bool,
    /// Whether codestream assembly stayed resident on Metal.
    pub codestream_assembly_used: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Lossless Metal encode output with host codestream bytes and timings.
///
/// API note: this diagnostic report is constructed by this crate. It is not
/// `#[non_exhaustive]`, but adapter releases may add diagnostic fields as the
/// resident encode path gains more profiling detail.
#[doc(hidden)]
pub struct MetalLosslessEncodeOutcome {
    /// Encoded J2K codestream.
    pub encoded: EncodedJ2k,
    /// Whether the input buffer had to be copied or padded.
    pub input_copy_used: bool,
    /// Residency decisions for the encode stages.
    pub resident: MetalLosslessEncodeResidency,
    /// Time spent copying or padding the input.
    pub input_copy_duration: Duration,
    /// End-to-end encode duration for this tile.
    pub encode_duration: Duration,
    /// GPU-only duration when timestamp data is available.
    pub gpu_duration: Option<Duration>,
    /// Time spent validating the encoded output.
    pub validation_duration: Duration,
    /// Time spent materializing buffer-backed codestream bytes into host bytes.
    pub host_readback_duration: Duration,
}

/// Metal lossless encode report for buffer-backed codestream output.
#[doc(hidden)]
pub struct MetalLosslessBufferEncodeOutcome {
    /// Encoded codestream stored in a Metal buffer.
    pub encoded: MetalEncodedJ2k,
    /// Whether the input buffer had to be copied or padded.
    pub input_copy_used: bool,
    /// Residency decisions for the encode stages.
    pub resident: MetalLosslessEncodeResidency,
    /// Time spent copying or padding the input.
    pub input_copy_duration: Duration,
    /// End-to-end encode duration for this tile.
    pub encode_duration: Duration,
    /// GPU-only duration when timestamp data is available.
    pub gpu_duration: Option<Duration>,
    /// Time spent validating the encoded output.
    pub validation_duration: Duration,
}

/// Tuning knobs for resident Metal lossless J2K/HTJ2K tile batch encode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalLosslessEncodeConfig {
    /// Requested maximum number of tiles submitted concurrently.
    ///
    /// `None` uses the crate default and still clamps by the memory budget.
    pub gpu_encode_inflight_tiles: Option<usize>,
    /// Resident encode memory budget in bytes.
    ///
    /// `None` uses `min(10 GiB, hw_memsize * 0.40)` when host memory can be
    /// discovered.
    pub gpu_encode_memory_budget_bytes: Option<usize>,
}

/// Batched lossless encode request over Metal-resident tiles.
///
/// Collapses the former per-permutation entry points: pick the input
/// staging mode and batch tuning here, then submit through
/// [`crate::submit_lossless_batch`], [`crate::submit_lossless_batch_to_metal`],
/// or [`crate::encode_lossless_batch_with_report`].
#[derive(Clone, Copy)]
pub struct MetalLosslessEncodeBatchRequest<'a, 'b> {
    /// Metal-resident tiles to encode.
    pub tiles: &'a [MetalLosslessEncodeTile<'b>],
    /// How tile samples reach the encoder's padded staging layout.
    pub staging: MetalEncodeInputStaging,
    /// Batch tuning knobs (inflight tiles, memory budget).
    pub config: MetalLosslessEncodeConfig,
}

/// How tile samples reach the encoder's padded staging layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalEncodeInputStaging {
    /// Copy the tile into freshly padded staging storage.
    CopyAndPad,
    /// The tile is already padded and contiguous; encode it in place.
    AlreadyPaddedContiguous,
}
