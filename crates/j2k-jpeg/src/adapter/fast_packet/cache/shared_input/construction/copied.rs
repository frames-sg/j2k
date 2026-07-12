// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{sync::Arc, vec::Vec};
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use super::super::{SharedJpegInput, SharedJpegInputInner, SharedJpegInputStorage};
use crate::adapter::fast_packet::cache::shared_allocation::{
    checked_live_bytes, shared_owner_bytes,
};
use crate::adapter::fast_packet::JpegPlanCacheError;

impl SharedJpegInput {
    /// Copy an input into a fallibly reserved shared byte owner.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation or limit error.
    pub fn try_copy_from_slice(input: &[u8]) -> Result<Self, JpegPlanCacheError> {
        Self::try_copy_from_slice_with_external_live(input, 0)
    }

    /// Copy input while charging owners already live in the adapter operation.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation or aggregate-limit error.
    #[doc(hidden)]
    pub fn try_copy_from_slice_with_external_live(
        input: &[u8],
        external_live_bytes: usize,
    ) -> Result<Self, JpegPlanCacheError> {
        Self::try_copy_from_slice_with_external_live_and_cap(
            input,
            external_live_bytes,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
    }

    #[cfg(test)]
    pub(in crate::adapter::fast_packet::cache) fn try_copy_from_slice_with_cap(
        input: &[u8],
        cap: usize,
    ) -> Result<Self, JpegPlanCacheError> {
        Self::try_copy_from_slice_with_external_live_and_cap(input, 0, cap)
    }

    pub(in crate::adapter::fast_packet::cache) fn try_copy_from_slice_with_external_live_and_cap(
        input: &[u8],
        external_live_bytes: usize,
        cap: usize,
    ) -> Result<Self, JpegPlanCacheError> {
        checked_live_bytes(
            "shared JPEG copied input owner graph",
            external_live_bytes,
            shared_owner_bytes::<SharedJpegInputInner>(input.len())?,
            cap,
        )?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(input.len()).map_err(|source| {
            JpegPlanCacheError::allocation("shared JPEG input bytes", input.len(), source)
        })?;
        checked_live_bytes(
            "shared JPEG copied input owner graph",
            external_live_bytes,
            shared_owner_bytes::<SharedJpegInputInner>(bytes.capacity())?,
            cap,
        )?;
        bytes.extend_from_slice(input);
        Ok(Self(SharedJpegInputStorage::Copied(Arc::new(
            SharedJpegInputInner { bytes },
        ))))
    }
}
