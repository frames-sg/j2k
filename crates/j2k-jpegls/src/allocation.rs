// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible allocation: every buffer the codec allocates reports failure as
//! an error instead of aborting.

/// An allocation of `bytes` bytes failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocationError {
    pub(crate) bytes: usize,
}

/// A vector of `len` default values.
pub(crate) fn zeroed<T: Clone + Default>(len: usize) -> Result<Vec<T>, AllocationError> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| AllocationError {
        bytes: len.saturating_mul(core::mem::size_of::<T>()),
    })?;
    out.resize(len, T::default());
    Ok(out)
}
