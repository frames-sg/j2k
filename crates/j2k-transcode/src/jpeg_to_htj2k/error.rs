// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{fmt, DctGridError, MetricsLengthError, TranscodeStageError};

/// Error returned by the experimental transcode path.
#[derive(Debug)]
pub enum JpegToHtj2kError {
    /// JPEG parse or entropy decode failed.
    Jpeg(j2k_jpeg::JpegError),
    /// Input is outside the currently implemented experimental slice.
    Unsupported(&'static str),
    /// DCT block grid metadata did not cover the component dimensions.
    Grid(String),
    /// DCT block grid metadata did not cover the component dimensions for the
    /// 9/7 path.
    Grid97(String),
    /// Optional transform acceleration failed.
    Accelerator(TranscodeStageError),
    /// Validation metric inputs were inconsistent.
    Metrics(String),
    /// Validation encountered an out-of-range or non-finite coefficient.
    Validation(&'static str),
    /// Native HTJ2K encode failed.
    Encode(&'static str),
}

impl fmt::Display for JpegToHtj2kError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Jpeg(err) => write!(f, "JPEG extraction failed: {err}"),
            Self::Unsupported(reason) => write!(f, "unsupported transcode input: {reason}"),
            Self::Grid(reason) | Self::Grid97(reason) => {
                write!(f, "DCT grid transform failed: {reason}")
            }
            Self::Accelerator(reason) => write!(f, "transform accelerator failed: {reason}"),
            Self::Metrics(reason) => write!(f, "validation metrics failed: {reason}"),
            Self::Validation(reason) => write!(f, "validation failed: {reason}"),
            Self::Encode(reason) => write!(f, "HTJ2K encode failed: {reason}"),
        }
    }
}

impl std::error::Error for JpegToHtj2kError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Jpeg(err) => Some(err),
            Self::Unsupported(_)
            | Self::Grid(_)
            | Self::Grid97(_)
            | Self::Accelerator(_)
            | Self::Metrics(_)
            | Self::Validation(_)
            | Self::Encode(_) => None,
        }
    }
}

impl From<j2k_jpeg::JpegError> for JpegToHtj2kError {
    fn from(value: j2k_jpeg::JpegError) -> Self {
        Self::Jpeg(value)
    }
}

pub(super) fn dct53_grid_error(value: DctGridError) -> JpegToHtj2kError {
    JpegToHtj2kError::Grid(value.to_string())
}

pub(super) fn dct97_grid_error(value: DctGridError) -> JpegToHtj2kError {
    JpegToHtj2kError::Grid97(value.to_string())
}

impl From<MetricsLengthError> for JpegToHtj2kError {
    fn from(value: MetricsLengthError) -> Self {
        Self::Metrics(value.to_string())
    }
}
