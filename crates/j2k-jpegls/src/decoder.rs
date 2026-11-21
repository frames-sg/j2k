// SPDX-License-Identifier: MIT OR Apache-2.0

//! The public decoder.

mod scan;
mod sink;

use crate::allocation::zeroed;
use crate::error::JpegLsError;
use crate::info::{DecodeOptions, Info};
use crate::markers::{self, Stream};

use self::sink::{ByteSink, LineSink, WordSink};

/// Decoded samples, interleaved by pixel.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodedSamples {
    /// Up to 8 bits per sample.
    U8(Vec<u8>),
    /// 9 to 16 bits per sample.
    U16(Vec<u16>),
}

/// A decoded image from [`Decoder::decode`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecodedImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Samples per pixel.
    pub components: u8,
    /// Bits per sample.
    pub bit_depth: u8,
    /// `width * height * components` samples, row-major, interleaved by
    /// pixel.
    pub samples: DecodedSamples,
}

/// A parsed JPEG-LS stream, ready to decode.
///
/// Construction parses and validates every marker segment and checks the
/// decoded size against [`DecodeOptions::max_output_bytes`], so an
/// oversized or structurally broken stream fails before any pixel buffer is
/// allocated.
#[derive(Debug, Clone)]
pub struct Decoder<'a> {
    data: &'a [u8],
    stream: Stream,
    info: Info,
}

impl<'a> Decoder<'a> {
    /// Parse `data` with default options.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed or truncated marker segments,
    /// unsupported features, or an image above the default 512 MiB output
    /// limit.
    pub fn new(data: &'a [u8]) -> Result<Self, JpegLsError> {
        Self::with_options(data, DecodeOptions::default())
    }

    /// Parse `data` with explicit options.
    ///
    /// # Errors
    ///
    /// As [`Decoder::new`], with the output limit from `options`.
    pub fn with_options(data: &'a [u8], options: DecodeOptions) -> Result<Self, JpegLsError> {
        let stream = markers::parse(data)?;
        let info = stream.info();
        let required = info.output_bytes_u64();
        if usize::try_from(required).map_or(true, |bytes| bytes > options.max_output_bytes()) {
            return Err(JpegLsError::OutputLimitExceeded {
                required,
                limit: options.max_output_bytes(),
            });
        }
        Ok(Self { data, stream, info })
    }

    /// Read the headers of `data` without decoding pixels. This walks every
    /// marker segment and skips over the entropy-coded data, so it validates
    /// the stream structure but not the coded samples.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed or truncated marker segments and
    /// unsupported features.
    pub fn inspect(data: &[u8]) -> Result<Info, JpegLsError> {
        Ok(markers::parse(data)?.info())
    }

    /// The parsed header metadata.
    #[must_use]
    pub const fn info(&self) -> &Info {
        &self.info
    }

    /// Decode into `out` with rows `stride` bytes apart. Samples are
    /// interleaved by pixel; above 8 bits each sample is two little-endian
    /// bytes. `out` needs `stride * (height - 1) + row_bytes` bytes, where
    /// `row_bytes` is [`Info::row_bytes`].
    ///
    /// On error, `out` may hold part of the image.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer or stride is too small, or if the
    /// entropy-coded data is truncated or invalid.
    pub fn decode_into(&self, out: &mut [u8], stride: usize) -> Result<(), JpegLsError> {
        let row_bytes = self.info.row_bytes().unwrap_or(usize::MAX);
        if stride < row_bytes {
            return Err(JpegLsError::InvalidStride { stride, row_bytes });
        }
        let rows = self.info.dimensions.1 as usize;
        let required = stride
            .checked_mul(rows - 1)
            .and_then(|bytes| bytes.checked_add(row_bytes))
            .unwrap_or(usize::MAX);
        if out.len() < required {
            return Err(JpegLsError::OutputBufferTooSmall {
                required,
                actual: out.len(),
            });
        }
        let mut sink = ByteSink {
            out,
            stride,
            components: usize::from(self.info.components),
            wide: self.info.bit_depth > 8,
        };
        self.decode_scans(&mut sink)
    }

    /// Decode into a new buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer cannot be allocated, or if the
    /// entropy-coded data is truncated or invalid.
    pub fn decode(&self) -> Result<DecodedImage, JpegLsError> {
        let info = &self.info;
        let samples = info
            .output_len()
            .map(|bytes| bytes / info.bytes_per_sample())
            .ok_or(JpegLsError::AllocationFailed { bytes: usize::MAX })?;
        let samples = if info.bit_depth > 8 {
            let mut out = allocate::<u16>(samples)?;
            let mut sink = WordSink {
                out: &mut out,
                components: usize::from(info.components),
            };
            self.decode_scans(&mut sink)?;
            DecodedSamples::U16(out)
        } else {
            let mut out = allocate::<u8>(samples)?;
            let stride = info.row_bytes().unwrap_or(usize::MAX);
            self.decode_into(&mut out, stride)?;
            DecodedSamples::U8(out)
        };
        Ok(DecodedImage {
            width: info.dimensions.0,
            height: info.dimensions.1,
            components: info.components,
            bit_depth: info.bit_depth,
            samples,
        })
    }

    fn decode_scans(&self, sink: &mut dyn LineSink) -> Result<(), JpegLsError> {
        let width = usize::from(self.stream.frame.width);
        let height = u32::from(self.stream.frame.height);
        for (index, scan) in self.stream.scans.iter().enumerate() {
            scan::decode_scan(self.data, scan, index, width, height, sink)?;
        }
        Ok(())
    }
}

fn allocate<T: Clone + Default>(len: usize) -> Result<Vec<T>, JpegLsError> {
    zeroed(len).map_err(|error| JpegLsError::AllocationFailed { bytes: error.bytes })
}
