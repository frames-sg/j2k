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
    /// Source Metal buffer containing Gray or RGB pixels.
    pub buffer: &'a Buffer,
    /// Byte offset of the first source pixel in `buffer`.
    pub byte_offset: usize,
    /// Width of the valid input region in pixels.
    pub width: u32,
    /// Height of the valid input region in pixels.
    pub height: u32,
    /// Number of bytes between consecutive input rows.
    pub pitch_bytes: usize,
    /// Encoded image width in pixels.
    pub output_width: u32,
    /// Encoded image height in pixels.
    pub output_height: u32,
    /// Pixel format of the source buffer.
    pub format: PixelFormat,
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
