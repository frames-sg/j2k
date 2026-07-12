// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use super::super::super::JpegPlanCache;
use crate::adapter::fast_packet::cache::shared_allocation::checked_live_bytes;
use crate::adapter::fast_packet::{JpegCachedPlan, JpegCachedPlanBuildError, SharedJpegInput};
use crate::decoder::{Decoder, JpegView};

impl JpegPlanCache {
    pub(super) fn resolve_miss_with_decoder<'a>(
        &mut self,
        input: &'a [u8],
        external_live_bytes: usize,
        input_cap: usize,
    ) -> Result<(JpegCachedPlan, Decoder<'a>), JpegCachedPlanBuildError> {
        self.inner
            .prepare_for_miss(external_live_bytes, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
        let live_bytes = self.operation_live_bytes(external_live_bytes)?;
        let view = JpegView::parse_with_external_live(input, live_bytes)?;
        let parsed_bytes = view.parsed_header().retained_allocation_bytes()?;
        let copy_live_bytes = checked_live_bytes(
            "JPEG plan cache miss owners before input copy",
            live_bytes,
            parsed_bytes,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        let shared_input = SharedJpegInput::try_copy_from_slice_with_external_live_and_cap(
            input,
            copy_live_bytes,
            input_cap,
        )?;
        let (plan, decoder) =
            JpegCachedPlan::build_shared_from_view_with_decoder(shared_input, view, live_bytes)?;
        self.insert(plan.clone())?;
        Ok((plan, decoder))
    }

    pub(super) fn resolve_shared_miss_with_decoder<'a>(
        &mut self,
        input: &'a SharedJpegInput,
        external_live_bytes: usize,
    ) -> Result<(JpegCachedPlan, Decoder<'a>), JpegCachedPlanBuildError> {
        let input_live_bytes = input.retained_cache_bytes()?;
        let preexisting_live_bytes = checked_live_bytes(
            "JPEG adapter and shared-input owners before cache metadata reserve",
            external_live_bytes,
            input_live_bytes,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        self.inner
            .prepare_for_miss(preexisting_live_bytes, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
        let live_bytes = self.operation_live_bytes(external_live_bytes)?;
        let owner_live_bytes = checked_live_bytes(
            "JPEG plan cache miss external and shared-input owners",
            live_bytes,
            input_live_bytes,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        let view = JpegView::parse_with_external_live(input.as_bytes(), owner_live_bytes)?;
        let (plan, decoder) =
            JpegCachedPlan::build_shared_from_view_with_decoder(input.clone(), view, live_bytes)?;
        self.insert(plan.clone())?;
        Ok((plan, decoder))
    }
}
