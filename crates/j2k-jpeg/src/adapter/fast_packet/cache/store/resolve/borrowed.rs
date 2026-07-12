// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use super::super::super::JpegPlanCache;
use crate::adapter::fast_packet::{JpegCachedPlan, JpegCachedPlanBuildError};
use crate::decoder::{Decoder, JpegView};

impl JpegPlanCache {
    /// Resolve one borrowed input through the authoritative hit/build/admit sequence.
    ///
    /// # Errors
    ///
    /// Returns typed decode, packet, allocation, or invariant failures.
    pub fn resolve(&mut self, input: &[u8]) -> Result<JpegCachedPlan, JpegCachedPlanBuildError> {
        self.resolve_with_external_live(input, 0)
    }

    /// Resolve a plan while charging adapter owners that remain live during a miss.
    ///
    /// # Errors
    ///
    /// Returns typed parse, packet, allocation, or invariant failures.
    pub fn resolve_with_external_live(
        &mut self,
        input: &[u8],
        external_live_bytes: usize,
    ) -> Result<JpegCachedPlan, JpegCachedPlanBuildError> {
        if let Some(plan) = self.get(input) {
            return Ok(plan);
        }
        let (plan, decoder) = self.resolve_miss_with_decoder(
            input,
            external_live_bytes,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        drop(decoder);
        Ok(plan)
    }

    /// Resolve a plan and construct exactly one decoder from the same parsed view.
    ///
    /// # Errors
    ///
    /// Returns typed parse, packet, allocation, or invariant failures.
    pub fn resolve_with_decoder_and_external_live<'a>(
        &mut self,
        input: &'a [u8],
        external_live_bytes: usize,
    ) -> Result<(JpegCachedPlan, Decoder<'a>), JpegCachedPlanBuildError> {
        if let Some(plan) = self.get(input) {
            let live_bytes = self.operation_live_bytes(external_live_bytes)?;
            let view = JpegView::parse_with_external_live(input, live_bytes)?;
            let decoder = Decoder::from_view_with_external_live(view, live_bytes)?;
            return Ok((plan, decoder));
        }
        self.resolve_miss_with_decoder(
            input,
            external_live_bytes,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
    }
}
