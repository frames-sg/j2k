// SPDX-License-Identifier: MIT OR Apache-2.0

//! ICC profile assembly from JPEG APP2 marker chunks.

use alloc::vec::Vec;

use crate::{iter_segments, JpegError};

const APP2_MARKER: u8 = 0xe2;
const ICC_SIGNATURE: &[u8; 12] = b"ICC_PROFILE\0";
const ICC_CHUNK_HEADER_LEN: usize = ICC_SIGNATURE.len() + 2;
const MAX_APP2_PAYLOAD_LEN: usize = u16::MAX as usize - 2;
const MAX_ICC_CHUNK_DATA_LEN: usize = MAX_APP2_PAYLOAD_LEN - ICC_CHUNK_HEADER_LEN;

pub(crate) fn is_icc_app2_payload(payload: &[u8]) -> bool {
    payload.starts_with(ICC_SIGNATURE)
}

/// Error returned while assembling an embedded JPEG ICC profile.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum IccProfileError {
    /// The JPEG marker stream itself is malformed.
    #[error(transparent)]
    Jpeg(#[from] JpegError),
    /// An ICC APP2 chunk declared an invalid sequence/count pair.
    #[error("invalid JPEG ICC chunk {sequence}/{count} at offset {offset}")]
    InvalidChunkNumber {
        /// APP2 marker byte offset.
        offset: usize,
        /// One-based chunk sequence number.
        sequence: u8,
        /// Declared total chunk count.
        count: u8,
    },
    /// ICC APP2 chunks disagreed about the total number of chunks.
    #[error(
        "inconsistent JPEG ICC chunk count at offset {offset}: expected {expected}, found {found}"
    )]
    InconsistentChunkCount {
        /// APP2 marker byte offset.
        offset: usize,
        /// Count declared by the first ICC chunk.
        expected: u8,
        /// Count declared by this ICC chunk.
        found: u8,
    },
    /// The same ICC sequence number appeared more than once.
    #[error("duplicate JPEG ICC chunk {sequence} at offset {offset}")]
    DuplicateChunk {
        /// APP2 marker byte offset.
        offset: usize,
        /// Duplicated one-based chunk sequence number.
        sequence: u8,
    },
    /// At least one chunk required by the sequence was absent.
    #[error("missing JPEG ICC chunk {sequence} of {count}")]
    MissingChunk {
        /// Missing one-based sequence number.
        sequence: u8,
        /// Declared total chunk count.
        count: u8,
    },
    /// The complete profile length overflowed the host address space.
    #[error("JPEG ICC profile length overflow")]
    LengthOverflow,
    /// Allocation for the assembled profile failed.
    #[error("failed to allocate {requested} bytes for JPEG ICC profile")]
    AllocationFailed {
        /// Requested profile byte count.
        requested: usize,
    },
    /// An empty byte string is not an ICC profile.
    #[error("JPEG ICC profile must not be empty")]
    EmptyProfile,
    /// A profile would require more than the 255 chunks representable by JPEG ICC APP2 metadata.
    #[error("JPEG ICC profile requires {chunks} APP2 chunks; maximum is 255")]
    TooManyChunks {
        /// Required chunk count.
        chunks: usize,
    },
    /// The JPEG already contains an ICC profile.
    #[error("JPEG already contains an ICC profile")]
    ProfileAlreadyPresent,
}

/// Assemble an ICC profile carried by JPEG APP2 `ICC_PROFILE` chunks.
///
/// Unrelated APP2 markers are ignored. ICC chunks may appear in any order, but
/// their one-based sequence numbers must be complete, unique, and agree on the
/// total chunk count.
///
/// # Errors
///
/// Returns [`IccProfileError`] when JPEG marker syntax is invalid, ICC chunk
/// metadata is inconsistent, or the output allocation fails.
pub fn extract_icc_profile(input: &[u8]) -> Result<Option<Vec<u8>>, IccProfileError> {
    let mut declared_count = None;
    let mut chunks: Vec<Option<&[u8]>> = Vec::new();

    for segment in iter_segments(input) {
        let segment = segment?;
        if segment.marker != APP2_MARKER || !is_icc_app2_payload(segment.payload) {
            continue;
        }
        if segment.payload.len() < ICC_CHUNK_HEADER_LEN {
            return Err(IccProfileError::InvalidChunkNumber {
                offset: segment.marker_offset,
                sequence: 0,
                count: 0,
            });
        }

        let sequence = segment.payload[ICC_SIGNATURE.len()];
        let count = segment.payload[ICC_SIGNATURE.len() + 1];
        if sequence == 0 || count == 0 || sequence > count {
            return Err(IccProfileError::InvalidChunkNumber {
                offset: segment.marker_offset,
                sequence,
                count,
            });
        }

        match declared_count {
            None => {
                chunks.try_reserve_exact(count as usize).map_err(|_| {
                    IccProfileError::AllocationFailed {
                        requested: count as usize * core::mem::size_of::<Option<&[u8]>>(),
                    }
                })?;
                chunks.resize(count as usize, None);
                declared_count = Some(count);
            }
            Some(expected) if expected != count => {
                return Err(IccProfileError::InconsistentChunkCount {
                    offset: segment.marker_offset,
                    expected,
                    found: count,
                });
            }
            Some(_) => {}
        }

        let slot = &mut chunks[usize::from(sequence - 1)];
        if slot.is_some() {
            return Err(IccProfileError::DuplicateChunk {
                offset: segment.marker_offset,
                sequence,
            });
        }
        *slot = Some(&segment.payload[ICC_CHUNK_HEADER_LEN..]);
    }

    let Some(count) = declared_count else {
        return Ok(None);
    };
    let mut profile_len = 0usize;
    for (index, chunk) in chunks.iter().enumerate() {
        let chunk = chunk.ok_or(IccProfileError::MissingChunk {
            sequence: u8::try_from(index + 1).unwrap_or(u8::MAX),
            count,
        })?;
        profile_len = profile_len
            .checked_add(chunk.len())
            .ok_or(IccProfileError::LengthOverflow)?;
    }

    let mut profile = Vec::new();
    profile
        .try_reserve_exact(profile_len)
        .map_err(|_| IccProfileError::AllocationFailed {
            requested: profile_len,
        })?;
    for chunk in chunks.into_iter().flatten() {
        profile.extend_from_slice(chunk);
    }
    Ok(Some(profile))
}

/// Insert an ICC profile into a complete JPEG interchange stream.
///
/// The profile is split into standard APP2 chunks and inserted immediately
/// after SOI. Existing marker and entropy bytes are preserved exactly.
///
/// # Errors
///
/// Returns [`IccProfileError`] for malformed JPEG input, an empty or oversized
/// profile, an existing embedded profile, length overflow, or allocation
/// failure.
pub fn insert_icc_profile(input: &[u8], profile: &[u8]) -> Result<Vec<u8>, IccProfileError> {
    if profile.is_empty() {
        return Err(IccProfileError::EmptyProfile);
    }
    let mut segments = iter_segments(input);
    let first = segments
        .next()
        .transpose()?
        .ok_or(JpegError::MissingMarker {
            marker: crate::MarkerKind::Soi,
        })?;
    if first.marker != 0xd8 {
        return Err(JpegError::UnexpectedMarker {
            offset: first.marker_offset,
            expected: crate::MarkerKind::Soi,
            found: first.marker,
        }
        .into());
    }
    for segment in segments {
        segment?;
    }
    if extract_icc_profile(input)?.is_some() {
        return Err(IccProfileError::ProfileAlreadyPresent);
    }

    let chunk_count = profile.len().div_ceil(MAX_ICC_CHUNK_DATA_LEN);
    if chunk_count > u8::MAX as usize {
        return Err(IccProfileError::TooManyChunks {
            chunks: chunk_count,
        });
    }
    let chunk_count_u8 = u8::try_from(chunk_count).map_err(|_| IccProfileError::TooManyChunks {
        chunks: chunk_count,
    })?;
    let marker_overhead = chunk_count
        .checked_mul(4 + ICC_CHUNK_HEADER_LEN)
        .ok_or(IccProfileError::LengthOverflow)?;
    let output_len = input
        .len()
        .checked_add(profile.len())
        .and_then(|len| len.checked_add(marker_overhead))
        .ok_or(IccProfileError::LengthOverflow)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| IccProfileError::AllocationFailed {
            requested: output_len,
        })?;
    output.extend_from_slice(&input[..2]);
    for (index, chunk) in profile.chunks(MAX_ICC_CHUNK_DATA_LEN).enumerate() {
        let segment_len = u16::try_from(2 + ICC_CHUNK_HEADER_LEN + chunk.len())
            .map_err(|_| IccProfileError::LengthOverflow)?;
        output.extend_from_slice(&[0xff, APP2_MARKER]);
        output.extend_from_slice(&segment_len.to_be_bytes());
        output.extend_from_slice(ICC_SIGNATURE);
        output.push(u8::try_from(index + 1).map_err(|_| IccProfileError::LengthOverflow)?);
        output.push(chunk_count_u8);
        output.extend_from_slice(chunk);
    }
    output.extend_from_slice(&input[2..]);
    Ok(output)
}

/// Set the ICC profile in a complete JPEG interchange stream.
///
/// When no profile is present this behaves like [`insert_icc_profile`]. When
/// an ordered, complete ICC profile already exists, all of its APP2 chunks are
/// replaced. Unrelated APP2 markers and every non-ICC byte are preserved.
///
/// # Errors
///
/// Returns [`IccProfileError`] for malformed JPEG or existing ICC metadata,
/// an empty or oversized replacement profile, length overflow, or allocation
/// failure.
pub fn set_icc_profile(input: &[u8], profile: &[u8]) -> Result<Vec<u8>, IccProfileError> {
    if extract_icc_profile(input)?.is_none() {
        return insert_icc_profile(input, profile);
    }

    let without_icc = remove_icc_chunks(input)?;
    insert_icc_profile(&without_icc, profile)
}

fn remove_icc_chunks(input: &[u8]) -> Result<Vec<u8>, IccProfileError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| IccProfileError::AllocationFailed {
            requested: input.len(),
        })?;
    let mut copied_through = 0usize;
    for segment in iter_segments(input) {
        let segment = segment?;
        if segment.marker != APP2_MARKER || !is_icc_app2_payload(segment.payload) {
            continue;
        }
        let segment_end = segment
            .payload_offset
            .checked_add(segment.payload.len())
            .ok_or(IccProfileError::LengthOverflow)?;
        output.extend_from_slice(&input[copied_through..segment.marker_offset]);
        copied_through = segment_end;
    }
    output.extend_from_slice(&input[copied_through..]);
    Ok(output)
}

#[cfg(test)]
mod tests;
