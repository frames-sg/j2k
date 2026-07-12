// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared-input parsing, decoder construction, and cached-plan assembly.

use crate::adapter::fast_packet::cache::shared_allocation::checked_live_bytes;
use crate::adapter::fast_packet::cache::{
    JpegCachedPlan, JpegCachedPlanBuildError, SharedJpegInput,
};
use crate::decoder::{Decoder, JpegView};

impl JpegCachedPlan {
    /// Parse one input once and build its backend-neutral accelerator plan.
    ///
    /// Ordinary fast-path capability mismatches produce an explicit
    /// [`crate::adapter::JpegFastPacketState::Unsupported`] plan. Malformed input, resource
    /// failures, and invariant failures remain hard errors and must not be
    /// inserted as a negative cache entry.
    ///
    /// # Errors
    ///
    /// Returns a typed parse, packet-build, allocation, or invariant error.
    pub fn build(input: SharedJpegInput) -> Result<Self, JpegCachedPlanBuildError> {
        let parse_owner = input.clone();
        let input_live_bytes = input.retained_cache_bytes()?;
        let view = JpegView::parse_with_external_live(parse_owner.as_bytes(), input_live_bytes)?;
        let (plan, decoder) = Self::build_shared_from_view_with_decoder(input, view, 0)?;
        drop(decoder);
        Ok(plan)
    }

    /// Parse borrowed bytes once, copy them fallibly, and return one plan and decoder.
    ///
    /// `external_live_bytes` excludes the copied input, parsed metadata, and
    /// returned decoder constructed by this method.
    ///
    /// # Errors
    ///
    /// Returns a typed parse, packet-build, allocation, or invariant error.
    #[doc(hidden)]
    pub fn build_with_decoder(
        input: &[u8],
        external_live_bytes: usize,
    ) -> Result<(Self, Decoder<'_>), JpegCachedPlanBuildError> {
        let view = JpegView::parse_with_external_live(input, external_live_bytes)?;
        Self::build_from_view_with_decoder(view, external_live_bytes)
    }

    /// Copy a previously parsed view fallibly and return one plan and decoder.
    ///
    /// The still-live parsed-header allocations are included while the shared
    /// input owner is created. `external_live_bytes` excludes the view itself.
    ///
    /// # Errors
    ///
    /// Returns a typed packet-build, allocation, decoder, or invariant error.
    #[doc(hidden)]
    pub fn build_from_view_with_decoder(
        view: JpegView<'_>,
        external_live_bytes: usize,
    ) -> Result<(Self, Decoder<'_>), JpegCachedPlanBuildError> {
        let parsed_bytes = view.parsed_header().retained_allocation_bytes()?;
        let copy_live_bytes = checked_live_bytes(
            "external and parsed JPEG owners before shared input copy",
            external_live_bytes,
            parsed_bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        let input =
            SharedJpegInput::try_copy_from_slice_with_external_live(view.bytes(), copy_live_bytes)?;
        Self::build_shared_from_view_with_decoder(input, view, external_live_bytes)
    }
}
