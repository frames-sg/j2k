// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::JpegPlanCache;
use crate::adapter::fast_packet::{JpegCachedPlan, JpegCachedPlanBuildError, SharedJpegInput};
use crate::decoder::{Decoder, JpegView};

impl JpegPlanCache {
    /// Resolve an existing immutable shared input without another payload copy.
    ///
    /// # Errors
    ///
    /// Returns typed decode, packet, allocation, or invariant failures.
    pub fn resolve_shared(
        &mut self,
        input: SharedJpegInput,
    ) -> Result<JpegCachedPlan, JpegCachedPlanBuildError> {
        self.resolve_shared_with_external_live(input, 0)
    }

    /// Resolve an immutable shared input while charging adapter-owned live bytes.
    ///
    /// # Errors
    ///
    /// Returns typed parse, packet, allocation, or invariant failures.
    pub fn resolve_shared_with_external_live(
        &mut self,
        input: SharedJpegInput,
        external_live_bytes: usize,
    ) -> Result<JpegCachedPlan, JpegCachedPlanBuildError> {
        if let Some(plan) = self.get(input.as_bytes()) {
            drop(input);
            return Ok(plan);
        }
        let (plan, decoder) = self.resolve_shared_miss_with_decoder(&input, external_live_bytes)?;
        drop(decoder);
        drop(input);
        Ok(plan)
    }

    /// Resolve a shared input and construct one decoder without copying payload bytes.
    ///
    /// # Errors
    ///
    /// Returns typed parse, packet, allocation, or invariant failures.
    pub fn resolve_shared_with_decoder_and_external_live<'a>(
        &mut self,
        input: &'a SharedJpegInput,
        external_live_bytes: usize,
    ) -> Result<(JpegCachedPlan, Decoder<'a>), JpegCachedPlanBuildError> {
        if let Some(plan) = self.get(input.as_bytes()) {
            return self.decoder_for_shared_hit(plan, input, external_live_bytes);
        }
        self.resolve_shared_miss_with_decoder(input, external_live_bytes)
    }

    fn decoder_for_shared_hit<'a>(
        &self,
        plan: JpegCachedPlan,
        input: &'a SharedJpegInput,
        external_live_bytes: usize,
    ) -> Result<(JpegCachedPlan, Decoder<'a>), JpegCachedPlanBuildError> {
        let live_bytes = self.operation_live_bytes(external_live_bytes)?;
        let owner_live_bytes =
            crate::adapter::fast_packet::cache::shared_allocation::checked_live_bytes(
                "JPEG plan cache hit external and caller input owners",
                live_bytes,
                input.retained_cache_bytes()?,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )?;
        let view = JpegView::parse_with_external_live(input.as_bytes(), owner_live_bytes)?;
        let decoder = Decoder::from_view_with_external_live(view, owner_live_bytes)?;
        Ok((plan, decoder))
    }
}
