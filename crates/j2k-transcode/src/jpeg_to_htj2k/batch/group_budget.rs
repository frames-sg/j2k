// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{BatchComponentRef, JpegToHtj2kError};
use crate::allocation::{checked_add_allocation_bytes, checked_allocation_bytes};

pub(super) fn batch_component_count(
    mut component_counts: impl Iterator<Item = usize>,
) -> Result<usize, JpegToHtj2kError> {
    component_counts.try_fold(0usize, |total, count| {
        total
            .checked_add(count)
            .ok_or(JpegToHtj2kError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })
    })
}

pub(super) fn validate_group_workspace(component_count: usize) -> Result<(), JpegToHtj2kError> {
    let groups_bytes = checked_allocation_bytes::<Vec<BatchComponentRef>>(component_count)?;
    let refs_bytes = checked_allocation_bytes::<BatchComponentRef>(component_count)?;
    checked_add_allocation_bytes(groups_bytes, refs_bytes)?;
    Ok(())
}

pub(super) fn next_group_len(len: usize) -> Result<usize, JpegToHtj2kError> {
    len.checked_add(1)
        .ok_or(JpegToHtj2kError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })
}
