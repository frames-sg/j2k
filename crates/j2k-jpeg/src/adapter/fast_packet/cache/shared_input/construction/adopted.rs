// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::sync::Arc;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use super::super::{SharedJpegInput, SharedJpegInputStorage};
use crate::adapter::fast_packet::cache::shared_allocation::{
    checked_live_bytes, shared_slice_owner_bytes,
};
use crate::adapter::fast_packet::JpegPlanCacheError;

impl SharedJpegInput {
    /// Move immutable shared input into the cache owner without copying its payload.
    ///
    /// # Errors
    ///
    /// Returns a typed limit error.
    pub fn try_from_arc(input: Arc<[u8]>) -> Result<Self, JpegPlanCacheError> {
        Self::try_from_arc_with_external_live(input, 0)
    }

    /// Adopt immutable input while charging owners already live in the operation.
    ///
    /// # Errors
    ///
    /// Returns a typed aggregate-limit error.
    #[doc(hidden)]
    pub fn try_from_arc_with_external_live(
        input: Arc<[u8]>,
        external_live_bytes: usize,
    ) -> Result<Self, JpegPlanCacheError> {
        Self::try_from_arc_with_external_live_and_cap(
            input,
            external_live_bytes,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
    }

    #[cfg(test)]
    pub(in crate::adapter::fast_packet::cache) fn try_from_arc_with_cap(
        input: Arc<[u8]>,
        cap: usize,
    ) -> Result<Self, JpegPlanCacheError> {
        Self::try_from_arc_with_external_live_and_cap(input, 0, cap)
    }

    pub(in crate::adapter::fast_packet::cache) fn try_from_arc_with_external_live_and_cap(
        input: Arc<[u8]>,
        external_live_bytes: usize,
        cap: usize,
    ) -> Result<Self, JpegPlanCacheError> {
        checked_live_bytes(
            "shared JPEG Arc input owner graph",
            external_live_bytes,
            shared_slice_owner_bytes(input.len())?,
            cap,
        )?;
        Ok(Self(SharedJpegInputStorage::ArcSlice(input)))
    }
}
