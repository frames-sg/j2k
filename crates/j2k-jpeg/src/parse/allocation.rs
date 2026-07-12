// SPDX-License-Identifier: MIT OR Apache-2.0

//! One live allocation budget for retained JPEG parse metadata.

use alloc::vec::Vec;
use core::mem::size_of;

use crate::allocation::try_reserve_for_len_with_live_budget;
use crate::context::MAX_DECODER_CONTEXT_ALLOCATION_BYTES;
use crate::error::JpegError;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

#[derive(Debug)]
pub(crate) struct ParsedMetadataBudget {
    baseline_bytes: usize,
    live_bytes: usize,
    cap: usize,
}

impl ParsedMetadataBudget {
    #[cfg(test)]
    pub(crate) const fn new() -> Self {
        Self::with_limits(
            MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
    }

    pub(crate) fn with_external_live(external_live_bytes: usize) -> Result<Self, JpegError> {
        Self::with_external_live_and_cap(
            MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
            external_live_bytes,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
    }

    fn with_external_live_and_cap(
        context_bytes: usize,
        external_live_bytes: usize,
        cap: usize,
    ) -> Result<Self, JpegError> {
        let baseline_bytes = context_bytes
            .checked_add(external_live_bytes)
            .ok_or_else(|| cap_error(cap))?;
        if baseline_bytes > cap {
            return Err(JpegError::MemoryCapExceeded {
                requested: baseline_bytes,
                cap,
            });
        }
        Ok(Self::with_limits(baseline_bytes, cap))
    }

    const fn with_limits(baseline_bytes: usize, cap: usize) -> Self {
        Self {
            baseline_bytes,
            live_bytes: baseline_bytes,
            cap,
        }
    }

    pub(crate) fn try_push<T>(&mut self, values: &mut Vec<T>, value: T) -> Result<(), JpegError> {
        let new_len = values
            .len()
            .checked_add(1)
            .ok_or_else(|| cap_error(self.cap))?;
        try_reserve_for_len_with_live_budget(values, new_len, &mut self.live_bytes, self.cap)?;
        values.push(value);
        Ok(())
    }

    pub(crate) fn finish(self, retained_metadata_bytes: usize) -> Result<(), JpegError> {
        let expected = self
            .baseline_bytes
            .checked_add(retained_metadata_bytes)
            .ok_or_else(|| cap_error(self.cap))?;
        if expected != self.live_bytes {
            return Err(JpegError::InternalInvariant {
                reason: "JPEG parser allocation ledger disagrees with retained metadata",
            });
        }
        ensure_retained_metadata_bytes(retained_metadata_bytes)?;
        Ok(())
    }
}

pub(crate) fn ensure_retained_metadata_bytes(metadata_bytes: usize) -> Result<(), JpegError> {
    let requested = MAX_DECODER_CONTEXT_ALLOCATION_BYTES
        .checked_add(metadata_bytes)
        .ok_or_else(|| cap_error(DEFAULT_MAX_HOST_ALLOCATION_BYTES))?;
    if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(JpegError::MemoryCapExceeded {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        });
    }
    Ok(())
}

pub(crate) fn capacity_bytes<T>(capacity: usize) -> Result<usize, JpegError> {
    capacity
        .checked_mul(size_of::<T>())
        .ok_or_else(|| cap_error(DEFAULT_MAX_HOST_ALLOCATION_BYTES))
}

fn cap_error(cap: usize) -> JpegError {
    JpegError::MemoryCapExceeded {
        requested: usize::MAX,
        cap,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsed_growth_counts_old_and_replacement_peak_exactly() {
        let mut values = alloc::vec::Vec::<u64>::with_capacity(2);
        values.extend([1, 2]);
        let retained = values.capacity() * size_of::<u64>();
        let replacement = 3 * size_of::<u64>();
        let exact_peak = retained + replacement;

        let mut exact = ParsedMetadataBudget::with_limits(retained, exact_peak);
        exact.try_push(&mut values, 3).expect("exact peak fits");

        let mut another = alloc::vec::Vec::<u64>::with_capacity(2);
        another.extend([1, 2]);
        let mut one_under = ParsedMetadataBudget::with_limits(retained, exact_peak - 1);
        assert!(matches!(
            one_under.try_push(&mut another, 3),
            Err(JpegError::MemoryCapExceeded {
                requested,
                cap
            }) if requested == exact_peak && cap == exact_peak - 1
        ));
        assert_eq!(another.capacity(), 2, "preflight runs before reserve");
    }

    #[test]
    fn external_and_context_baseline_accepts_exact_cap_and_rejects_one_over() {
        let context = 11;
        let external = 7;
        let exact_cap = context + external;
        ParsedMetadataBudget::with_external_live_and_cap(context, external, exact_cap)
            .expect("exact parser owner baseline");
        assert!(matches!(
            ParsedMetadataBudget::with_external_live_and_cap(context, external, exact_cap - 1),
            Err(JpegError::MemoryCapExceeded { requested, cap })
                if requested == exact_cap && cap == exact_cap - 1
        ));
    }
}
