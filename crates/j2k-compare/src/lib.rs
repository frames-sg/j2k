// SPDX-License-Identifier: MIT OR Apache-2.0

//! In-process external JPEG 2000 comparators for benches and parity tests.

use std::time::Instant;

use j2k_core::Rect;

/// Large enough for a full 16k x 16k RGB8 decode (768 MiB); a smaller cap here
/// would read as a comparator failure on large images.
pub(crate) const MAX_EXTERNAL_OUTPUT_BYTES: usize = 4 * 1024 * 1024 * 1024;

#[cfg(any(have_grok, have_openhtj2k, have_openjph, test))]
fn checked_external_output_len(
    codec: &str,
    width: u32,
    height: u32,
    channels: u32,
) -> Result<usize, String> {
    if !matches!(channels, 1 | 3) {
        return Err(format!(
            "{codec}: unsupported channel count {channels}, expected 1 or 3"
        ));
    }
    if width == 0 || height == 0 {
        return Err(format!("{codec}: image has zero-sized output"));
    }
    let width =
        usize::try_from(width).map_err(|_| format!("{codec}: width exceeds platform usize"))?;
    let height =
        usize::try_from(height).map_err(|_| format!("{codec}: height exceeds platform usize"))?;
    let channels = usize::try_from(channels)
        .map_err(|_| format!("{codec}: channel count exceeds platform usize"))?;
    let len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| format!("{codec}: output byte count overflow"))?;
    if len > MAX_EXTERNAL_OUTPUT_BYTES {
        return Err(format!(
            "{codec}: output exceeds {MAX_EXTERNAL_OUTPUT_BYTES} byte cap"
        ));
    }
    Ok(len)
}

/// Color shape requested from an external comparator decode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalDecodeColor {
    /// Single-channel 8-bit grayscale output.
    Gray,
    /// Three-channel interleaved 8-bit RGB output.
    Rgb,
    /// Single-channel 16-bit grayscale output: native-endian `u16` samples
    /// holding each sample's value unscaled (clamped to `0..=65535`).
    Gray16,
}

impl ExternalDecodeColor {
    /// Number of interleaved output channels.
    #[must_use]
    pub const fn channels(self) -> u32 {
        match self {
            Self::Gray | Self::Gray16 => 1,
            Self::Rgb => 3,
        }
    }

    /// Bytes per output sample.
    #[must_use]
    pub const fn bytes_per_sample(self) -> u32 {
        match self {
            Self::Gray | Self::Rgb => 1,
            Self::Gray16 => 2,
        }
    }
}

/// Decode request shared by `OpenJPEG` and `Grok` wrappers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalDecodeRequest {
    /// Output color shape.
    pub color: ExternalDecodeColor,
    /// Optional OpenJPEG/Grok resolution reduction factor.
    pub reduce: Option<u32>,
    /// Optional source region.
    pub region: Option<Rect>,
    /// Decoder-internal threads for this one image; `1` decodes it on the
    /// calling thread only.
    pub threads: u32,
}

impl ExternalDecodeRequest {
    /// Full-resolution RGB decode.
    #[must_use]
    pub const fn rgb() -> Self {
        Self {
            color: ExternalDecodeColor::Rgb,
            reduce: None,
            region: None,
            threads: 1,
        }
    }

    /// Full-resolution 16-bit grayscale decode.
    #[must_use]
    pub const fn gray16() -> Self {
        Self {
            color: ExternalDecodeColor::Gray16,
            reduce: None,
            region: None,
            threads: 1,
        }
    }

    /// Full-resolution grayscale decode.
    #[must_use]
    pub const fn gray() -> Self {
        Self {
            color: ExternalDecodeColor::Gray,
            reduce: None,
            region: None,
            threads: 1,
        }
    }

    /// The same request decoded with `threads` decoder-internal threads
    /// (at least one).
    #[must_use]
    pub const fn with_threads(self, threads: u32) -> Self {
        Self {
            threads: if threads == 0 { 1 } else { threads },
            ..self
        }
    }

    /// Region RGB decode.
    #[must_use]
    pub const fn rgb_region(region: Rect) -> Self {
        Self {
            region: Some(region),
            ..Self::rgb()
        }
    }

    /// Region grayscale decode.
    #[must_use]
    pub const fn gray_region(region: Rect) -> Self {
        Self {
            region: Some(region),
            ..Self::gray()
        }
    }

    /// Reduced-resolution RGB decode.
    #[must_use]
    pub const fn rgb_scaled(reduce: u32) -> Self {
        Self {
            reduce: Some(reduce),
            ..Self::rgb()
        }
    }

    /// Reduced-resolution grayscale decode.
    #[must_use]
    pub const fn gray_scaled(reduce: u32) -> Self {
        Self {
            reduce: Some(reduce),
            ..Self::gray()
        }
    }

    /// Region plus reduced-resolution RGB decode.
    #[must_use]
    pub const fn rgb_region_scaled(region: Rect, reduce: u32) -> Self {
        Self {
            reduce: Some(reduce),
            region: Some(region),
            ..Self::rgb()
        }
    }

    /// Region plus reduced-resolution grayscale decode.
    #[must_use]
    pub const fn gray_region_scaled(region: Rect, reduce: u32) -> Self {
        Self {
            reduce: Some(reduce),
            region: Some(region),
            ..Self::gray()
        }
    }
}

macro_rules! external_decode_wrappers {
    ($decode:ident) => {
        pub fn decode_rgb(bytes: &[u8]) -> Result<Vec<u8>, String> {
            $decode(bytes, crate::ExternalDecodeRequest::rgb())
        }

        pub fn decode_gray(bytes: &[u8]) -> Result<Vec<u8>, String> {
            $decode(bytes, crate::ExternalDecodeRequest::gray())
        }

        /// Decode exactly `request` (color, region, reduction and threads).
        pub fn decode_request(
            bytes: &[u8],
            request: crate::ExternalDecodeRequest,
        ) -> Result<Vec<u8>, String> {
            $decode(bytes, request)
        }

        /// Full-resolution RGB decode using `threads` decoder-internal threads.
        pub fn decode_rgb_threads(bytes: &[u8], threads: u32) -> Result<Vec<u8>, String> {
            $decode(
                bytes,
                crate::ExternalDecodeRequest::rgb().with_threads(threads),
            )
        }

        /// Full-resolution grayscale decode using `threads` decoder-internal threads.
        pub fn decode_gray_threads(bytes: &[u8], threads: u32) -> Result<Vec<u8>, String> {
            $decode(
                bytes,
                crate::ExternalDecodeRequest::gray().with_threads(threads),
            )
        }

        pub fn decode_rgb_region(bytes: &[u8], roi: j2k_core::Rect) -> Result<Vec<u8>, String> {
            $decode(bytes, crate::ExternalDecodeRequest::rgb_region(roi))
        }

        pub fn decode_gray_region(bytes: &[u8], roi: j2k_core::Rect) -> Result<Vec<u8>, String> {
            $decode(bytes, crate::ExternalDecodeRequest::gray_region(roi))
        }

        pub fn decode_rgb_region_scaled(
            bytes: &[u8],
            roi: j2k_core::Rect,
            reduce: u32,
        ) -> Result<Vec<u8>, String> {
            $decode(
                bytes,
                crate::ExternalDecodeRequest::rgb_region_scaled(roi, reduce),
            )
        }

        pub fn decode_gray_region_scaled(
            bytes: &[u8],
            roi: j2k_core::Rect,
            reduce: u32,
        ) -> Result<Vec<u8>, String> {
            $decode(
                bytes,
                crate::ExternalDecodeRequest::gray_region_scaled(roi, reduce),
            )
        }

        pub fn decode_rgb_scaled(bytes: &[u8], reduce: u32) -> Result<Vec<u8>, String> {
            $decode(bytes, crate::ExternalDecodeRequest::rgb_scaled(reduce))
        }

        pub fn decode_gray_scaled(bytes: &[u8], reduce: u32) -> Result<Vec<u8>, String> {
            $decode(bytes, crate::ExternalDecodeRequest::gray_scaled(reduce))
        }
    };
}

pub(crate) use external_decode_wrappers;

pub mod common;
pub mod encode_compare;
pub mod fixture_compare;
pub mod grok;
pub mod openhtj2k;
pub mod openjpeg;
pub mod openjph;

/// Summary statistics for benchmark samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleStats {
    /// Median sample value.
    pub median: f64,
    /// Arithmetic mean sample value.
    pub mean: f64,
}

/// Parses a positive `usize` from command-line or environment input.
pub fn parse_positive_usize(value: &str, label: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {label} {value:?}: {error}"))?;
    if parsed == 0 {
        return Err(format!("{label} must be > 0"));
    }
    Ok(parsed)
}

/// Runs an untimed warmup followed by `repeats` timed samples.
pub fn measure_repeated<T>(
    repeats: usize,
    seconds_to_units: f64,
    mut run: impl FnMut() -> Result<T, String>,
) -> Result<(Vec<f64>, T), String> {
    let mut last = run()?;
    std::hint::black_box(&last);
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        last = run()?;
        samples.push(started.elapsed().as_secs_f64() * seconds_to_units);
        std::hint::black_box(&last);
    }
    Ok((samples, last))
}

/// Computes mean and median for non-empty benchmark samples.
pub fn sample_stats(samples: &[f64]) -> Result<SampleStats, String> {
    if samples.is_empty() {
        return Err("cannot summarize empty benchmark samples".to_string());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(SampleStats {
        median: sorted[sorted.len() / 2],
        mean: samples.iter().sum::<f64>() / usize_to_f64(samples.len()),
    })
}

/// Lossy but bounded conversion for benchmark ratios.
#[must_use]
pub fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod external_output_len_tests {
    use super::{checked_external_output_len, MAX_EXTERNAL_OUTPUT_BYTES};

    #[test]
    fn output_len_is_bounded_before_slice_construction() {
        assert!(checked_external_output_len("codec", 0, 1, 1).is_err());
        assert!(checked_external_output_len("codec", 1, 1, 2).is_err());
        assert!(checked_external_output_len("codec", u32::MAX, u32::MAX, 3).is_err());
        // The cap (4 GiB) is wider than one u32 dimension, so reach it with
        // a square image: exactly at the cap is accepted, one row more is not.
        let side = 1_u32 << 16;
        assert_eq!(MAX_EXTERNAL_OUTPUT_BYTES, (1_usize << 16) * (1_usize << 16));
        assert_eq!(
            checked_external_output_len("codec", side, side, 1),
            Ok(MAX_EXTERNAL_OUTPUT_BYTES)
        );
        assert!(checked_external_output_len("codec", side, side + 1, 1).is_err());
        assert_eq!(checked_external_output_len("codec", 2, 3, 3), Ok(18));
    }
}
