// SPDX-License-Identifier: MIT OR Apache-2.0

//! The public encoder.

mod scan;

use crate::bitio::BitWriter;
use crate::decoder::{DecodedSamples, Decoder};
use crate::error::JpegLsEncodeError;
use crate::info::{DecodeOptions, InterleaveMode};
use crate::markers;
use crate::params::{CodingParameters, ParameterError, PresetCodingParameters};

use self::scan::{ScanPlan, ScanSource, SourceSamples};

/// Borrowed samples to encode: row-major, interleaved by pixel.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum JpegLsSamples<'a> {
    /// 8-bit samples, for bit depths up to 8.
    U8 {
        /// `width * height * components` samples.
        data: &'a [u8],
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Samples per pixel, 1 to 255.
        components: u8,
    },
    /// 16-bit samples, for bit depths up to 16.
    U16 {
        /// `width * height * components` samples.
        data: &'a [u16],
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Samples per pixel, 1 to 255.
        components: u8,
    },
}

impl JpegLsSamples<'_> {
    /// `(width, height)` in pixels.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        match *self {
            Self::U8 { width, height, .. } | Self::U16 { width, height, .. } => (width, height),
        }
    }

    /// Samples per pixel.
    #[must_use]
    pub const fn components(&self) -> u8 {
        match *self {
            Self::U8 { components, .. } | Self::U16 { components, .. } => components,
        }
    }

    const fn len(&self) -> usize {
        match self {
            Self::U8 { data, .. } => data.len(),
            Self::U16 { data, .. } => data.len(),
        }
    }

    const fn source(&self) -> SourceSamples<'_> {
        match *self {
            Self::U8 { data, .. } => SourceSamples::U8(data),
            Self::U16 { data, .. } => SourceSamples::U16(data),
        }
    }

    fn sample(&self, index: usize) -> Option<u16> {
        match self {
            Self::U8 { data, .. } => data.get(index).copied().map(u16::from),
            Self::U16 { data, .. } => data.get(index).copied(),
        }
    }
}

/// How [`encode_jpegls`] checks its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum JpegLsEncodeValidation {
    /// Decode the codestream and compare it with the input: exactly for
    /// lossless coding, within `NEAR` otherwise.
    #[default]
    RoundTrip,
    /// Return the codestream without decoding it.
    Skip,
}

/// Options for [`encode_jpegls`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct JpegLsEncodeOptions {
    /// Bits per sample `P`, 2 to 16 (2 to 8 for 8-bit samples).
    pub bit_depth: u8,
    /// Maximum absolute error `NEAR`: 0 for lossless, up to
    /// `min(255, MAXVAL / 2)` for near-lossless.
    pub near: u8,
    /// Interleave mode for images with more than one component. Line and
    /// sample interleaving code at most four components. Ignored for one
    /// component, which is always coded with `ILV = 0`.
    pub interleave: InterleaveMode,
    /// Lines per restart interval, or `None` for no restart markers.
    pub restart_interval: Option<u16>,
    /// Preset coding parameters. Zero fields take the defaults, and an LSE
    /// segment is written only when the result differs from the defaults.
    pub preset: Option<PresetCodingParameters>,
    /// Whether to verify the output by decoding it.
    pub validation: JpegLsEncodeValidation,
}

impl JpegLsEncodeOptions {
    /// Lossless coding at `bit_depth` bits with default parameters, no
    /// interleaving, no restart markers, and round-trip validation.
    #[must_use]
    pub const fn new(bit_depth: u8) -> Self {
        Self {
            bit_depth,
            near: 0,
            interleave: InterleaveMode::None,
            restart_interval: None,
            preset: None,
            validation: JpegLsEncodeValidation::RoundTrip,
        }
    }

    /// Set `NEAR`.
    #[must_use]
    pub const fn with_near(mut self, near: u8) -> Self {
        self.near = near;
        self
    }

    /// Set the interleave mode.
    #[must_use]
    pub const fn with_interleave(mut self, interleave: InterleaveMode) -> Self {
        self.interleave = interleave;
        self
    }

    /// Write a restart marker every `lines` lines.
    #[must_use]
    pub const fn with_restart_interval(mut self, lines: u16) -> Self {
        self.restart_interval = Some(lines);
        self
    }

    /// Set the preset coding parameters.
    #[must_use]
    pub const fn with_preset(mut self, preset: PresetCodingParameters) -> Self {
        self.preset = Some(preset);
        self
    }

    /// Set the output validation.
    #[must_use]
    pub const fn with_validation(mut self, validation: JpegLsEncodeValidation) -> Self {
        self.validation = validation;
        self
    }
}

/// An encoded JPEG-LS codestream.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct EncodedJpegLs {
    /// The codestream, from SOI to EOI.
    pub data: Vec<u8>,
}

/// Encode samples as a JPEG-LS codestream.
///
/// The frame uses component identifiers 1 to N. Without line or sample
/// interleaving, each component is a separate scan.
///
/// # Errors
///
/// Returns an error for invalid dimensions, component count, bit depth,
/// sample count, `NEAR`, interleave mode, preset parameters, or restart
/// interval; for a sample above `MAXVAL`; when the output cannot be
/// allocated; and, with [`JpegLsEncodeValidation::RoundTrip`], when the
/// output does not decode to the input within `NEAR`.
pub fn encode_jpegls(
    samples: JpegLsSamples<'_>,
    options: &JpegLsEncodeOptions,
) -> Result<EncodedJpegLs, JpegLsEncodeError> {
    let plan = EncodePlan::new(&samples, options)?;
    check_sample_range(&samples, plan.params.max_value)?;

    let mut out = Vec::new();
    out.try_reserve(plan.reserve_bytes)
        .map_err(|_| JpegLsEncodeError::AllocationFailed {
            bytes: plan.reserve_bytes,
        })?;
    markers::write_frame(
        &mut out,
        plan.width,
        plan.height,
        options.bit_depth,
        plan.components,
    );
    if let Some(preset) = plan.explicit_preset {
        markers::write_preset(&mut out, preset);
    }
    if let Some(lines) = options.restart_interval {
        markers::write_restart_interval(&mut out, lines);
    }

    let source = ScanSource {
        samples: samples.source(),
        width: usize::from(plan.width),
        height: u32::from(plan.height),
        components: usize::from(plan.components),
    };
    let all: Vec<usize> = (0..usize::from(plan.components)).collect();
    let scan_size = if plan.interleave == InterleaveMode::None {
        1
    } else {
        all.len()
    };
    let mut writer = BitWriter::new(out);
    for components in all.chunks(scan_size) {
        let ids: Vec<u8> = components
            .iter()
            .map(|&index| u8::try_from(index + 1).unwrap_or(u8::MAX))
            .collect();
        markers::write_scan(writer.bytes_mut(), &ids, options.near, plan.interleave);
        let scan_plan = ScanPlan {
            components,
            interleave: plan.interleave,
            params: plan.params,
            restart_interval: options.restart_interval.map_or(0, u32::from),
        };
        scan::encode_scan(&mut writer, &source, &scan_plan)
            .map_err(|error| JpegLsEncodeError::AllocationFailed { bytes: error.bytes })?;
    }
    let mut data = writer.into_bytes();
    markers::write_end(&mut data);

    if options.validation == JpegLsEncodeValidation::RoundTrip {
        verify_round_trip(&data, &samples, options.near)?;
    }
    Ok(EncodedJpegLs { data })
}

/// Validated encode parameters.
struct EncodePlan {
    width: u16,
    height: u16,
    components: u8,
    interleave: InterleaveMode,
    params: CodingParameters,
    /// Parameters for an LSE segment, when they differ from the defaults.
    explicit_preset: Option<PresetCodingParameters>,
    reserve_bytes: usize,
}

impl EncodePlan {
    fn new(
        samples: &JpegLsSamples<'_>,
        options: &JpegLsEncodeOptions,
    ) -> Result<Self, JpegLsEncodeError> {
        let (width, height) = samples.dimensions();
        if width == 0 || height == 0 {
            return Err(JpegLsEncodeError::EmptyDimensions);
        }
        let (Ok(width16), Ok(height16)) = (u16::try_from(width), u16::try_from(height)) else {
            return Err(JpegLsEncodeError::DimensionsTooLarge { width, height });
        };
        let components = samples.components();
        if components == 0 {
            return Err(JpegLsEncodeError::NoComponents);
        }
        let (sample_type, max_bits) = match samples {
            JpegLsSamples::U8 { .. } => ("u8", 8),
            JpegLsSamples::U16 { .. } => ("u16", 16),
        };
        if !(2..=max_bits).contains(&options.bit_depth) {
            return Err(JpegLsEncodeError::InvalidBitDepth {
                bit_depth: options.bit_depth,
                sample_type,
            });
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(usize::from(components)))
            .unwrap_or(usize::MAX);
        if samples.len() != expected {
            return Err(JpegLsEncodeError::SampleLength {
                expected,
                actual: samples.len(),
            });
        }
        let interleave = if components == 1 {
            InterleaveMode::None
        } else {
            options.interleave
        };
        if interleave != InterleaveMode::None
            && usize::from(components) > crate::context::MAX_INTERLEAVED_COMPONENTS
        {
            return Err(JpegLsEncodeError::InvalidInterleave {
                interleave,
                components,
            });
        }
        if options.restart_interval == Some(0) {
            return Err(JpegLsEncodeError::InvalidRestartInterval);
        }
        let resolve = |preset| {
            CodingParameters::resolve(preset, options.bit_depth, options.near).map_err(|error| {
                match error {
                    ParameterError::Near { near, max } => {
                        JpegLsEncodeError::InvalidNear { near, max }
                    }
                    other => JpegLsEncodeError::InvalidPresetParameters {
                        reason: other.reason(),
                    },
                }
            })
        };
        let params = resolve(options.preset)?;
        let full_scale = u16::try_from((1_u32 << options.bit_depth) - 1).unwrap_or(u16::MAX);
        let resolved = params.preset();
        let explicit_preset = (resolved
            != PresetCodingParameters::defaults(full_scale, options.near))
        .then_some(resolved);
        // Zero fields resolve independently, so check that the LSE segment as
        // written (every field explicit) is valid too.
        if explicit_preset.is_some() {
            resolve(explicit_preset)?;
        }
        let sample_bytes = if options.bit_depth > 8 { 2 } else { 1 };
        Ok(Self {
            width: width16,
            height: height16,
            components,
            interleave,
            params,
            explicit_preset,
            reserve_bytes: expected.saturating_mul(sample_bytes) / 2 + 1024,
        })
    }
}

fn check_sample_range(
    samples: &JpegLsSamples<'_>,
    max_value: i32,
) -> Result<(), JpegLsEncodeError> {
    let max = u16::try_from(max_value).unwrap_or(u16::MAX);
    let first_over = match samples {
        JpegLsSamples::U8 { data, .. } => data
            .iter()
            .position(|&sample| u16::from(sample) > max)
            .map(|index| (index, u16::from(data[index]))),
        JpegLsSamples::U16 { data, .. } => data
            .iter()
            .position(|&sample| sample > max)
            .map(|index| (index, data[index])),
    };
    match first_over {
        Some((index, value)) => Err(JpegLsEncodeError::SampleOutOfRange {
            index,
            value,
            max_value: max,
        }),
        None => Ok(()),
    }
}

/// Decode `data` and compare it with the input.
fn verify_round_trip(
    data: &[u8],
    samples: &JpegLsSamples<'_>,
    near: u8,
) -> Result<(), JpegLsEncodeError> {
    let (width, height) = samples.dimensions();
    let bytes = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(usize::from(samples.components()))
        .saturating_mul(2);
    let options = DecodeOptions::default().with_max_output_bytes(bytes);
    let decoded = Decoder::with_options(data, options)
        .and_then(|decoder| decoder.decode())
        .map_err(JpegLsEncodeError::ValidationDecode)?;
    let decoded_sample = |index: usize| match &decoded.samples {
        DecodedSamples::U8(values) => values.get(index).copied().map(u16::from),
        DecodedSamples::U16(values) => values.get(index).copied(),
    };
    for index in 0..samples.len() {
        let expected = samples.sample(index).unwrap_or(0);
        let actual = decoded_sample(index).unwrap_or(u16::MAX);
        if expected.abs_diff(actual) > u16::from(near) {
            return Err(JpegLsEncodeError::ValidationMismatch {
                index,
                expected,
                decoded: actual,
                near,
            });
        }
    }
    Ok(())
}
