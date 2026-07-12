// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible aggregate allocation for a single-pass JP2/JPH writer.

use alloc::vec::Vec;

use crate::J2kError;
use j2k_core::{try_host_vec_with_capacity, BufferError, DEFAULT_MAX_HOST_ALLOCATION_BYTES};

pub(super) fn checked_retained_bytes(
    first: usize,
    second: usize,
    what: &'static str,
) -> Result<usize, J2kError> {
    let retained = first
        .checked_add(second)
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow { what }))?;
    if retained > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(J2kError::Buffer(BufferError::AllocationTooLarge {
            requested: retained,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what,
        }));
    }
    Ok(retained)
}

pub(super) fn allocate_output(
    output_len: usize,
    retained_bytes: usize,
) -> Result<Vec<u8>, J2kError> {
    checked_retained_bytes(
        retained_bytes,
        output_len,
        "JP2/JPH writer aggregate output",
    )?;
    let output = try_host_vec_with_capacity(output_len).map_err(|error| {
        J2kError::Buffer(BufferError::HostAllocationFailed {
            bytes: error.requested_bytes(),
            what: "JP2/JPH output",
        })
    })?;
    let actual_peak = retained_bytes
        .checked_add(output.capacity())
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow {
            what: "JP2/JPH writer actual aggregate output",
        }))?;
    if actual_peak > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(J2kError::Buffer(BufferError::AllocationTooLarge {
            requested: actual_peak,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what: "JP2/JPH writer actual aggregate output",
        }));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsed_metadata_source_and_output_share_one_exact_boundary() {
        let parsed_metadata = 3;
        let owned_codestream = 2;
        let output = DEFAULT_MAX_HOST_ALLOCATION_BYTES - parsed_metadata - owned_codestream;
        let retained = checked_retained_bytes(
            parsed_metadata,
            owned_codestream,
            "test parsed and source owners",
        )
        .expect("retained owners");
        assert_eq!(
            checked_retained_bytes(retained, output, "test wrapper peak").expect("exact cap"),
            DEFAULT_MAX_HOST_ALLOCATION_BYTES
        );
        assert!(matches!(
            checked_retained_bytes(retained, output + 1, "test wrapper peak"),
            Err(J2kError::Buffer(BufferError::AllocationTooLarge { .. }))
        ));
    }
}
