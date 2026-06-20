// SPDX-License-Identifier: Apache-2.0

//! In-process external JPEG 2000 comparators for benches and parity tests.

use std::time::Instant;

use j2k_core::Rect;

/// Color shape requested from an external comparator decode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalDecodeColor {
    /// Single-channel 8-bit grayscale output.
    Gray,
    /// Three-channel interleaved 8-bit RGB output.
    Rgb,
}

impl ExternalDecodeColor {
    /// Number of interleaved output channels.
    #[must_use]
    pub const fn channels(self) -> u32 {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
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
}

impl ExternalDecodeRequest {
    /// Full-resolution RGB decode.
    #[must_use]
    pub const fn rgb() -> Self {
        Self {
            color: ExternalDecodeColor::Rgb,
            reduce: None,
            region: None,
        }
    }

    /// Full-resolution grayscale decode.
    #[must_use]
    pub const fn gray() -> Self {
        Self {
            color: ExternalDecodeColor::Gray,
            reduce: None,
            region: None,
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

pub mod grok;
pub mod openjpeg;

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
