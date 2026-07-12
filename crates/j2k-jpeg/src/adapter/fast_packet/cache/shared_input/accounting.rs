// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::sync::Arc;

use super::{SharedJpegInput, SharedJpegInputInner, SharedJpegInputStorage};
use crate::adapter::fast_packet::cache::shared_allocation::{
    shared_owner_bytes, shared_slice_owner_bytes,
};
use crate::adapter::fast_packet::JpegPlanCacheError;

impl SharedJpegInput {
    /// Retained payload capacity of the copied vector or fixed Arc slice.
    #[must_use]
    pub fn data_capacity(&self) -> usize {
        match &self.0 {
            SharedJpegInputStorage::Copied(input) => input.bytes.capacity(),
            SharedJpegInputStorage::ArcSlice(input) => input.len(),
        }
    }

    /// Whether two handles share the same input allocation.
    #[must_use]
    pub fn ptr_eq(left: &Self, right: &Self) -> bool {
        match (&left.0, &right.0) {
            (SharedJpegInputStorage::Copied(left), SharedJpegInputStorage::Copied(right)) => {
                Arc::ptr_eq(left, right)
            }
            (SharedJpegInputStorage::ArcSlice(left), SharedJpegInputStorage::ArcSlice(right)) => {
                Arc::ptr_eq(left, right)
            }
            (SharedJpegInputStorage::Copied(_), SharedJpegInputStorage::ArcSlice(_))
            | (SharedJpegInputStorage::ArcSlice(_), SharedJpegInputStorage::Copied(_)) => false,
        }
    }

    /// Retained bytes charged to a cache entry for this shared input.
    ///
    /// Byte-vector capacity or fixed Arc-slice length is exact. See the type
    /// documentation for the Arc control-block limitation.
    ///
    /// # Errors
    ///
    /// Returns an invariant error if retained-byte arithmetic overflows.
    pub fn retained_cache_bytes(&self) -> Result<usize, JpegPlanCacheError> {
        match &self.0 {
            SharedJpegInputStorage::Copied(input) => {
                shared_owner_bytes::<SharedJpegInputInner>(input.bytes.capacity())
            }
            SharedJpegInputStorage::ArcSlice(input) => shared_slice_owner_bytes(input.len()),
        }
    }
}
