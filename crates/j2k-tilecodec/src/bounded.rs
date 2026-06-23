// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::BufferError;
use std::{fmt, io::Read};

#[derive(Debug)]
pub(crate) enum BoundedReadError {
    OutputTooSmall(BufferError),
    Io(std::io::Error),
}

impl fmt::Display for BoundedReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputTooSmall(error) => error.fmt(f),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl From<BoundedReadError> for crate::TileCodecError {
    fn from(error: BoundedReadError) -> Self {
        match error {
            BoundedReadError::OutputTooSmall(error) => Self::Buffer(error),
            BoundedReadError::Io(error) => {
                crate::error::input_or_backend_io_error(&error, "bounded decode")
            }
        }
    }
}

pub(crate) fn read_to_scratch_bounded<R: Read>(
    mut reader: R,
    scratch: &mut Vec<u8>,
    out_len: usize,
) -> Result<usize, BoundedReadError> {
    scratch.clear();
    let limit = out_len.saturating_add(1);
    reader
        .by_ref()
        .take(limit as u64)
        .read_to_end(scratch)
        .map_err(BoundedReadError::Io)?;

    if scratch.len() > out_len {
        return Err(BoundedReadError::OutputTooSmall(observed_too_small(
            scratch.len(),
            out_len,
        )));
    }

    Ok(scratch.len())
}

pub(crate) fn read_to_output_bounded<R: Read>(
    reader: R,
    scratch: &mut Vec<u8>,
    out: &mut [u8],
) -> Result<usize, BoundedReadError> {
    read_to_scratch_bounded(reader, scratch, out.len())?;
    Ok(copy_scratch_to_output(scratch, out))
}

pub(crate) fn observed_too_small(required: usize, have: usize) -> BufferError {
    BufferError::OutputTooSmall { required, have }
}

pub(crate) fn copy_scratch_to_output(scratch: &[u8], out: &mut [u8]) -> usize {
    out[..scratch.len()].copy_from_slice(scratch);
    scratch.len()
}
