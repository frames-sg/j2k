// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use super::super::super::JpegPlanCache;
use crate::adapter::fast_packet::cache::shared_allocation::checked_live_bytes;
#[cfg(test)]
use crate::adapter::fast_packet::JpegCachedPlan;
use crate::adapter::fast_packet::JpegCachedPlanBuildError;

impl JpegPlanCache {
    pub(super) fn operation_live_bytes(
        &self,
        external_live_bytes: usize,
    ) -> Result<usize, JpegCachedPlanBuildError> {
        self.operation_live_bytes_with_cap(external_live_bytes, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }

    fn operation_live_bytes_with_cap(
        &self,
        external_live_bytes: usize,
        cap: usize,
    ) -> Result<usize, JpegCachedPlanBuildError> {
        checked_live_bytes(
            "JPEG plan cache and adapter owner graph",
            self.diagnostics().retained_bytes,
            external_live_bytes,
            cap,
        )
        .map_err(Into::into)
    }

    #[cfg(test)]
    pub(in crate::adapter::fast_packet::cache) fn operation_live_bytes_with_cap_for_test(
        &self,
        external_live_bytes: usize,
        cap: usize,
    ) -> Result<usize, JpegCachedPlanBuildError> {
        self.operation_live_bytes_with_cap(external_live_bytes, cap)
    }

    #[cfg(test)]
    pub(in crate::adapter::fast_packet::cache) fn resolve_with_input_cap_for_test(
        &mut self,
        input: &[u8],
        input_cap: usize,
    ) -> Result<JpegCachedPlan, JpegCachedPlanBuildError> {
        if let Some(plan) = self.get(input) {
            return Ok(plan);
        }
        let (plan, decoder) = self.resolve_miss_with_decoder(input, 0, input_cap)?;
        drop(decoder);
        Ok(plan)
    }
}
