// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validated parameters of one lossless encode, shared by the CPU encoder and
//! the device encoders.

use j2k_codec_math::jpeg::lossless::{initial_predictor, MAX_PREDICTOR};
use j2k_core::PixelFormat;

use super::JpegLosslessEncodeOptions;
use crate::encoder::JpegEncodeError;

/// Validated geometry and coding parameters of one lossless encode.
///
/// Samples are coded in scan order: rows top to bottom, pixels left to
/// right, and for color the three components of a pixel together (one
/// interleaved scan, one MCU per pixel).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegLosslessEncodePlan {
    /// Image width in pixels, 1..=65535.
    pub width: u32,
    /// Image height in pixels, 1..=65535.
    pub height: u32,
    /// 1 for grayscale, 3 for RGB.
    pub components: u8,
    /// Sample precision `P`, 2..=16.
    pub precision: u8,
    /// Predictor selection value, 1..=7.
    pub predictor: u8,
    /// Point transform `Pt`, below `precision`.
    pub point_transform: u8,
    /// `2^(P - Pt - 1)`, the 1-D predictor seed.
    pub initial_predictor: u16,
    /// Rows per restart interval; 0 without restart markers.
    pub restart_rows: u32,
}

impl JpegLosslessEncodePlan {
    /// Samples in the image, `width * height * components`.
    #[must_use]
    pub const fn samples(&self) -> usize {
        self.width as usize * self.height as usize * self.components as usize
    }

    /// Largest valid input sample, `2^P - 1`.
    #[must_use]
    pub const fn max_sample(&self) -> u32 {
        (1 << self.precision) - 1
    }

    /// The DRI restart interval in MCUs, when restart markers are used.
    #[must_use]
    pub const fn restart_interval_mcus(&self) -> Option<u16> {
        if self.restart_rows == 0 {
            return None;
        }
        // The plan proved the product fits the 16-bit DRI field.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "validated against u16::MAX when the plan was built"
        )]
        let interval = (self.restart_rows * self.width) as u16;
        Some(interval)
    }

    /// Restart intervals in the scan (1 without restart markers).
    #[must_use]
    pub const fn restart_intervals(&self) -> usize {
        if self.restart_rows == 0 {
            1
        } else {
            self.height.div_ceil(self.restart_rows) as usize
        }
    }
}

/// Validate a lossless encode of `format` samples at `dimensions` with
/// `options`, exactly as the CPU encoder does.
///
/// # Errors
/// Returns an error for an unsupported sample format, invalid dimensions,
/// precision, predictor, point transform, or restart interval.
#[doc(hidden)]
pub fn plan_jpeg_lossless_encode(
    format: PixelFormat,
    (width, height): (u32, u32),
    options: JpegLosslessEncodeOptions,
) -> Result<JpegLosslessEncodePlan, JpegEncodeError> {
    let (components, max_precision, samples) = match format {
        PixelFormat::Gray8 => (1, 8, "Gray8"),
        PixelFormat::Rgb8 => (3, 8, "Rgb8"),
        PixelFormat::Gray16 => (1, 16, "Gray16"),
        PixelFormat::Rgb16 => (3, 16, "Rgb16"),
        _ => return Err(JpegEncodeError::UnsupportedPixelFormat { format }),
    };
    if width == 0 || height == 0 {
        return Err(JpegEncodeError::EmptyDimensions);
    }
    if width > u32::from(u16::MAX) || height > u32::from(u16::MAX) {
        return Err(JpegEncodeError::DimensionsTooLarge { width, height });
    }
    let precision = options.precision;
    if !(2..=max_precision).contains(&precision) {
        return Err(JpegEncodeError::UnsupportedPrecision { precision, samples });
    }
    if !(1..=MAX_PREDICTOR).contains(&options.predictor) {
        return Err(JpegEncodeError::InvalidPredictor {
            predictor: options.predictor,
        });
    }
    let point_transform = options.point_transform;
    let initial_predictor = initial_predictor(precision, point_transform).ok_or(
        JpegEncodeError::InvalidPointTransform {
            point_transform,
            precision,
        },
    )?;
    let restart_rows = match options.restart_rows {
        None => 0,
        Some(0) => return Err(JpegEncodeError::InvalidRestartInterval),
        Some(rows) if u32::from(rows) * width > u32::from(u16::MAX) => {
            return Err(JpegEncodeError::RestartIntervalTooLarge {
                rows,
                mcus_per_row: width,
            });
        }
        Some(rows) => u32::from(rows),
    };
    Ok(JpegLosslessEncodePlan {
        width,
        height,
        components,
        precision,
        predictor: options.predictor,
        point_transform,
        initial_predictor,
        restart_rows,
    })
}
