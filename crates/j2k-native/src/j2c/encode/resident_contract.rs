// SPDX-License-Identifier: MIT OR Apache-2.0

/// Failure returned by the backend-resident HTJ2K encode boundary.
#[doc(hidden)]
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResidentHtj2kEncodeError {
    /// Caller-provided resident geometry or encode options are invalid.
    InvalidInput(&'static str),
    /// Geometry or encode options cannot use the resident whole-tile route.
    Unsupported(&'static str),
    /// The resident accelerator explicitly declined the whole-tile job.
    Declined,
    /// The accelerator accepted the job but returned a backend failure.
    Accelerator(crate::J2kEncodeStageError),
    /// Native host-resource accounting rejected the resident output.
    Resource(crate::EncodeError),
    /// Native planning or codestream finalization failed.
    Backend(crate::EncodeError),
}

impl core::fmt::Display for ResidentHtj2kEncodeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput(reason) => {
                write!(formatter, "invalid resident HTJ2K encode input: {reason}")
            }
            Self::Unsupported(reason) => {
                write!(formatter, "unsupported resident HTJ2K encode: {reason}")
            }
            Self::Declined => {
                formatter.write_str("resident HTJ2K tile accelerator declined encode")
            }
            Self::Accelerator(source) => {
                write!(formatter, "resident HTJ2K accelerator failed: {source}")
            }
            Self::Resource(error) => write!(formatter, "resident HTJ2K resource failure: {error}"),
            Self::Backend(error) => write!(formatter, "resident HTJ2K encode failed: {error}"),
        }
    }
}

impl core::error::Error for ResidentHtj2kEncodeError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Accelerator(error) => Some(error),
            Self::Resource(error) | Self::Backend(error) => Some(error),
            Self::InvalidInput(_) | Self::Unsupported(_) | Self::Declined => None,
        }
    }
}
