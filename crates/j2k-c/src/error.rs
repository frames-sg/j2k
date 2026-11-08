// SPDX-License-Identifier: MIT OR Apache-2.0

//! Map codec failures without exposing Rust objects across the ABI.

#![forbid(unsafe_code)]

use crate::abi::{
    J2kCError, J2K_ALLOCATION_FAILED, J2K_BUFFER_TOO_SMALL, J2K_CANCELLED, J2K_DECODE_FAILED,
    J2K_ENCODE_FAILED, J2K_INTERNAL_ERROR, J2K_INVALID_ARGUMENT, J2K_INVALID_INPUT, J2K_PANIC,
    J2K_RESOURCE_LIMIT, J2K_UNSUPPORTED,
};
use j2k::{BatchInfrastructureError, BatchItemError, BufferError, CodecError, J2kError};
use std::borrow::Cow;

pub(crate) type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub(crate) struct Error {
    pub(crate) code: u32,
    pub(crate) message: Cow<'static, str>,
}

impl Error {
    pub(crate) fn new(code: u32, message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub(crate) fn argument(message: &'static str) -> Self {
        Self::new(J2K_INVALID_ARGUMENT, message)
    }

    pub(crate) fn limit(message: &'static str) -> Self {
        Self::new(J2K_RESOURCE_LIMIT, message)
    }

    pub(crate) fn internal(message: &'static str) -> Self {
        Self::new(J2K_INTERNAL_ERROR, message)
    }

    pub(crate) fn codec(error: &J2kError) -> Self {
        let code = match error {
            J2kError::Cancelled => J2K_CANCELLED,
            J2kError::Buffer(
                BufferError::AllocationTooLarge { .. } | BufferError::SizeOverflow { .. },
            )
            | J2kError::ResourceLimit { .. } => J2K_RESOURCE_LIMIT,
            J2kError::Buffer(BufferError::HostAllocationFailed { .. }) => J2K_ALLOCATION_FAILED,
            J2kError::Buffer(BufferError::OutputTooSmall { .. }) => J2K_BUFFER_TOO_SMALL,
            J2kError::InvalidRegion { .. }
            | J2kError::InvalidSamples { .. }
            | J2kError::DimensionOverflow { .. } => J2K_INVALID_ARGUMENT,
            J2kError::NativeEncode { .. }
            | J2kError::NativeResidentEncode { .. }
            | J2kError::RateTargetUnreachable { .. } => J2K_ENCODE_FAILED,
            J2kError::InternalInvariant { .. } | J2kError::BackendComponentPlaneTooShort { .. } => {
                J2K_INTERNAL_ERROR
            }
            _ if error.is_unsupported() || error.is_not_implemented() => J2K_UNSUPPORTED,
            _ if error.is_buffer_error() => J2K_INVALID_ARGUMENT,
            _ if error.is_truncated() => J2K_INVALID_INPUT,
            J2kError::NativeDecode { .. } | J2kError::Backend(_) => J2K_DECODE_FAILED,
            _ => J2K_INVALID_INPUT,
        };
        Self::new(code, error.to_string())
    }

    pub(crate) fn item(error: &BatchItemError) -> Self {
        match error {
            BatchItemError::Codec { source, .. } => Self::codec(source),
            BatchItemError::NonRepresentableBatchOutput { .. } => {
                Self::new(J2K_UNSUPPORTED, error.to_string())
            }
            _ => Self::new(J2K_INTERNAL_ERROR, error.to_string()),
        }
    }

    pub(crate) fn infrastructure(error: BatchInfrastructureError) -> Self {
        let code = match error {
            BatchInfrastructureError::AllocationTooLarge { .. } => J2K_RESOURCE_LIMIT,
            BatchInfrastructureError::HostAllocationFailed { .. }
            | BatchInfrastructureError::WorkerSpawnFailed { .. } => J2K_ALLOCATION_FAILED,
            BatchInfrastructureError::WorkerPanicked { .. }
            | BatchInfrastructureError::ParallelWorkerPanicked
            | BatchInfrastructureError::SchedulerPoisoned => J2K_PANIC,
            BatchInfrastructureError::UnsupportedContract { .. } => J2K_UNSUPPORTED,
            _ => J2K_INTERNAL_ERROR,
        };
        Self::new(code, error.to_string())
    }

    pub(crate) fn diagnostic(&self) -> J2kCError {
        let mut result = J2kCError {
            code: self.code,
            ..Default::default()
        };
        let mut end = self.message.len().min(result.message.len() - 1);
        while !self.message.is_char_boundary(end) {
            end -= 1;
        }
        for (dst, &byte) in result
            .message
            .iter_mut()
            .zip(&self.message.as_bytes()[..end])
        {
            // Preserve bytes on both signed-char and unsigned-char C platforms.
            *dst = std::ffi::c_char::from_ne_bytes([byte]);
        }
        result
    }
}

impl From<J2kError> for Error {
    fn from(value: J2kError) -> Self {
        Self::codec(&value)
    }
}
