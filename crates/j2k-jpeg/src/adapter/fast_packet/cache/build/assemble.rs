// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared-view decoder construction and one-family packet-state assembly.

use super::packet::{clear_fast_families, inspect_fast_header, materialize_packet_state};
use crate::adapter::device_plan::summarize_device_batch;
use crate::adapter::fast_packet::cache::shared_allocation::checked_live_bytes;
use crate::adapter::fast_packet::cache::{
    JpegCachedPlan, JpegCachedPlanBuildError, JpegFastPacketState, JpegPlanCacheError,
    SharedJpegInput,
};
use crate::decoder::{Decoder, JpegView};

const JPEG_PLAN_CACHE_CADENCE_MCUS: u32 = 4;

impl JpegCachedPlan {
    pub(in crate::adapter::fast_packet::cache) fn build_shared_from_view_with_decoder(
        input: SharedJpegInput,
        view: JpegView<'_>,
        external_live_bytes: usize,
    ) -> Result<(Self, Decoder<'_>), JpegCachedPlanBuildError> {
        let owner_live_bytes = checked_live_bytes(
            "cached JPEG plan input and external owner graph",
            external_live_bytes,
            input.retained_cache_bytes()?,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        let inspected = inspect_fast_header(&view)?;
        let decoder = Decoder::from_view_with_external_live(view, owner_live_bytes)?;
        let plan = Self::build_from_decoder(input, &decoder, inspected, owner_live_bytes)?;
        Ok((plan, decoder))
    }

    pub(in crate::adapter::fast_packet::cache) fn build_shared_from_view_and_decoder(
        input: SharedJpegInput,
        view: JpegView<'_>,
        decoder: &Decoder<'_>,
        external_live_bytes: usize,
    ) -> Result<Self, JpegCachedPlanBuildError> {
        if view.bytes() != decoder.bytes {
            return Err(JpegPlanCacheError::Invariant(
                "cached JPEG plan view and existing decoder inputs differ",
            )
            .into());
        }
        let owner_live_bytes = checked_live_bytes(
            "cached JPEG plan input and external owner graph",
            external_live_bytes,
            input.retained_cache_bytes()?,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        let inspected = inspect_fast_header(&view)?;
        drop(view);
        Self::build_from_decoder(input, decoder, inspected, owner_live_bytes)
    }

    fn build_from_decoder(
        input: SharedJpegInput,
        decoder: &Decoder<'_>,
        inspected: Option<(
            crate::adapter::fast_packet::JpegFastPacketFamily,
            crate::adapter::fast_packet::header::ColorFastHeader,
        )>,
        owner_live_bytes: usize,
    ) -> Result<Self, JpegCachedPlanBuildError> {
        let mut summary = summarize_device_batch(decoder, JPEG_PLAN_CACHE_CADENCE_MCUS);
        let packet_state = if let Some((family, header)) = inspected {
            materialize_packet_state(
                input.as_bytes(),
                decoder,
                family,
                header,
                owner_live_bytes,
                &mut summary,
            )?
        } else {
            clear_fast_families(&mut summary);
            JpegFastPacketState::Unsupported
        };
        Self::try_new(input, summary, decoder.info().color_space, packet_state).map_err(Into::into)
    }
}
