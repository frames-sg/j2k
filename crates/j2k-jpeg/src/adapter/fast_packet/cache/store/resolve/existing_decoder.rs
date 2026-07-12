// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use super::super::super::JpegPlanCache;
use crate::adapter::device_plan::retained_decoder_allocation_bytes;
use crate::adapter::fast_packet::cache::shared_allocation::checked_live_bytes;
use crate::adapter::fast_packet::{JpegCachedPlan, JpegCachedPlanBuildError, SharedJpegInput};
use crate::decoder::{Decoder, JpegView};

impl JpegPlanCache {
    /// Resolve a plan using an already-constructed decoder without building another.
    ///
    /// # Errors
    ///
    /// Returns typed parse, packet, allocation, or invariant failures.
    pub fn resolve_from_decoder_with_external_live(
        &mut self,
        decoder: &Decoder<'_>,
        external_live_bytes: usize,
    ) -> Result<JpegCachedPlan, JpegCachedPlanBuildError> {
        if let Some(plan) = self.get(decoder.bytes) {
            return Ok(plan);
        }
        let decoder_live_bytes = retained_decoder_allocation_bytes(decoder)?;
        let preexisting_live_bytes = checked_live_bytes(
            "JPEG adapter and existing decoder before cache metadata reserve",
            external_live_bytes,
            decoder_live_bytes,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        self.inner
            .prepare_for_miss(preexisting_live_bytes, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
        let cache_live_bytes = self.operation_live_bytes(external_live_bytes)?;
        let parse_live_bytes = checked_live_bytes(
            "JPEG plan cache and existing decoder before packet inspection",
            cache_live_bytes,
            decoder_live_bytes,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        let view = JpegView::parse_with_external_live(decoder.bytes, parse_live_bytes)?;
        let copy_live_bytes = checked_live_bytes(
            "JPEG plan cache, decoder, and parsed metadata before input copy",
            parse_live_bytes,
            view.parsed_header().retained_allocation_bytes()?,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        let input = SharedJpegInput::try_copy_from_slice_with_external_live(
            decoder.bytes,
            copy_live_bytes,
        )?;
        let plan = JpegCachedPlan::build_shared_from_view_and_decoder(
            input,
            view,
            decoder,
            cache_live_bytes,
        )?;
        self.insert(plan.clone())?;
        Ok(plan)
    }
}
