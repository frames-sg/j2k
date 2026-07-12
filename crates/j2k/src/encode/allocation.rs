// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible, capped host owners created by facade encode adapters.

use alloc::vec::Vec;
use core::mem::size_of;

use j2k_core::{try_host_vec_with_capacity, BufferError, DEFAULT_MAX_HOST_ALLOCATION_BYTES};

use crate::J2kError;

pub(super) fn try_vec<T>(count: usize, what: &'static str) -> Result<Vec<T>, J2kError> {
    try_vec_with_cap(count, what, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
}

pub(super) fn try_collect_exact<T>(
    items: impl ExactSizeIterator<Item = T>,
    what: &'static str,
) -> Result<Vec<T>, J2kError> {
    let expected = items.len();
    let mut values = try_vec(expected, what)?;
    values.extend(items);
    if values.len() != expected {
        return Err(J2kError::InternalInvariant {
            what: "exact encode descriptor iterator changed length",
        });
    }
    Ok(values)
}

pub(super) fn try_collect_results_exact<T>(
    items: impl ExactSizeIterator<Item = Result<T, J2kError>>,
    what: &'static str,
) -> Result<Vec<T>, J2kError> {
    let expected = items.len();
    let mut values = try_vec(expected, what)?;
    for item in items {
        values.push(item?);
    }
    if values.len() != expected {
        return Err(J2kError::InternalInvariant {
            what: "exact fallible encode descriptor iterator changed length",
        });
    }
    Ok(values)
}

fn try_vec_with_cap<T>(count: usize, what: &'static str, cap: usize) -> Result<Vec<T>, J2kError> {
    let requested = element_bytes::<T>(count, what)?;
    ensure_within_cap(requested, cap, what)?;
    let values = try_host_vec_with_capacity(count).map_err(|error| {
        J2kError::Buffer(BufferError::HostAllocationFailed {
            bytes: error.requested_bytes(),
            what,
        })
    })?;
    let actual = element_bytes::<T>(values.capacity(), what)?;
    ensure_within_cap(actual, cap, what)?;
    Ok(values)
}

fn element_bytes<T>(count: usize, what: &'static str) -> Result<usize, J2kError> {
    count
        .checked_mul(size_of::<T>())
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow { what }))
}

const fn ensure_within_cap(
    requested: usize,
    cap: usize,
    what: &'static str,
) -> Result<(), J2kError> {
    if requested > cap {
        return Err(J2kError::Buffer(BufferError::AllocationTooLarge {
            requested,
            cap,
            what,
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_owner_accepts_exact_cap_and_rejects_one_byte_less() {
        let count = 4;
        let bytes = count * size_of::<u32>();
        let exact = try_vec_with_cap::<u32>(count, "test encode owner", bytes)
            .expect("exact encode owner cap");
        assert_eq!(exact.capacity(), count);

        assert!(matches!(
            try_vec_with_cap::<u32>(count, "test encode owner", bytes - 1),
            Err(J2kError::Buffer(BufferError::AllocationTooLarge {
                requested,
                cap,
                what: "test encode owner",
            })) if requested == bytes && cap == bytes - 1
        ));
    }

    #[test]
    fn descriptor_collection_preserves_exact_order() {
        let values = try_collect_exact((0_u8..4).map(u16::from), "test descriptors")
            .expect("descriptor collection");
        assert_eq!(values, [0, 1, 2, 3]);
    }
}
