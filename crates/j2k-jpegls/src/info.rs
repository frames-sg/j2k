// SPDX-License-Identifier: MIT OR Apache-2.0

//! Header metadata and decode options.

use crate::params::PresetCodingParameters;

/// Default limit on the decoded image size: 512 MiB.
const DEFAULT_MAX_OUTPUT_BYTES: usize = 512 * 1024 * 1024;

/// How a scan interleaves its components (the SOS `ILV` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum InterleaveMode {
    /// One component per scan (`ILV = 0`).
    #[default]
    None,
    /// Line interleaved: one line of each component in turn (`ILV = 1`).
    Line,
    /// Sample interleaved: all components of each pixel in turn (`ILV = 2`).
    Sample,
}

impl InterleaveMode {
    /// The value written in the SOS `ILV` field.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Line => 1,
            Self::Sample => 2,
        }
    }

    pub(crate) const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::None),
            1 => Some(Self::Line),
            2 => Some(Self::Sample),
            _ => None,
        }
    }
}

/// One scan of a JPEG-LS image.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ScanInfo {
    /// Zero-based indices of the frame components the scan codes, in scan
    /// order.
    pub components: Vec<usize>,
    /// Maximum absolute reconstruction error `NEAR`; 0 is lossless.
    pub near: u8,
    /// Interleave mode.
    pub interleave: InterleaveMode,
    /// Lines per restart interval, or `None` without restart markers.
    pub restart_interval: Option<u32>,
    /// Coding parameters the scan uses, with defaults filled in.
    pub parameters: PresetCodingParameters,
}

/// Header metadata of a JPEG-LS image, from [`Decoder::inspect`] or
/// [`Decoder::info`].
///
/// [`Decoder::inspect`]: crate::Decoder::inspect
/// [`Decoder::info`]: crate::Decoder::info
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Info {
    /// Image dimensions as `(width, height)` in pixels.
    pub dimensions: (u32, u32),
    /// Number of components.
    pub components: u8,
    /// Component identifiers from the frame header.
    pub component_ids: Vec<u8>,
    /// Bits per sample, 2 to 16.
    pub bit_depth: u8,
    /// `NEAR` of the first scan; 0 is lossless.
    pub near: u8,
    /// Interleave mode of the first scan.
    pub interleave: InterleaveMode,
    /// Lines per restart interval of the first scan, or `None` without
    /// restart markers.
    pub restart_interval: Option<u32>,
    /// Preset coding parameters as written in an LSE segment before the first
    /// scan, or `None` if the stream uses the defaults. Zero fields mean
    /// "default".
    pub preset: Option<PresetCodingParameters>,
    /// Every scan in stream order.
    pub scans: Vec<ScanInfo>,
}

impl Info {
    /// Bytes per decoded sample: 1 up to 8 bits, 2 above.
    #[must_use]
    pub const fn bytes_per_sample(&self) -> usize {
        if self.bit_depth > 8 {
            2
        } else {
            1
        }
    }

    /// Bytes in one tightly packed row of interleaved samples, or `None` if
    /// that overflows `usize`.
    #[must_use]
    pub fn row_bytes(&self) -> Option<usize> {
        usize::try_from(self.dimensions.0)
            .ok()?
            .checked_mul(usize::from(self.components))?
            .checked_mul(self.bytes_per_sample())
    }

    /// Bytes in the tightly packed decoded image, or `None` if that overflows
    /// `usize`.
    #[must_use]
    pub fn output_len(&self) -> Option<usize> {
        self.row_bytes()?
            .checked_mul(usize::try_from(self.dimensions.1).ok()?)
    }

    /// Exact decoded size in bytes, saturated at `u64::MAX`.
    pub(crate) fn output_bytes_u64(&self) -> u64 {
        u64::from(self.dimensions.0)
            .saturating_mul(u64::from(self.dimensions.1))
            .saturating_mul(u64::from(self.components))
            .saturating_mul(self.bytes_per_sample() as u64)
    }
}

/// Options for [`Decoder::with_options`](crate::Decoder::with_options).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    max_output_bytes: usize,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        }
    }
}

impl DecodeOptions {
    /// Set the largest decoded image, in bytes, the decoder accepts. Images
    /// above it are rejected before any pixel buffer is allocated. The
    /// default is 512 MiB.
    #[must_use]
    pub const fn with_max_output_bytes(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }

    /// The largest decoded image, in bytes, the decoder accepts.
    #[must_use]
    pub const fn max_output_bytes(&self) -> usize {
        self.max_output_bytes
    }
}
