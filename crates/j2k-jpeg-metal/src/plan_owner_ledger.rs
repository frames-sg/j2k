// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded owner-graph admission for collections of queued JPEG plans.

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
use j2k_jpeg::adapter::{JpegPlanCacheError, SharedJpegFastPacket, SharedJpegInput};

use crate::{batch::QueuedRequest, Error};

mod execution;
mod request_count;

pub(crate) use execution::batch_execution_budget;
#[cfg(target_os = "macos")]
pub(crate) use execution::preflight_collective_metadata;
use request_count::preflight_request_count;

pub(crate) struct PlanOwnerAdmission {
    retained_bytes: usize,
}

impl PlanOwnerAdmission {
    pub(crate) const fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    const fn into_retained_bytes(self) -> usize {
        self.retained_bytes
    }
}

#[derive(Debug)]
pub(crate) struct PlanOwnerLedger {
    retained_bytes: usize,
    host_byte_limit: usize,
}

impl Default for PlanOwnerLedger {
    fn default() -> Self {
        Self {
            retained_bytes: 0,
            host_byte_limit: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        }
    }
}

impl PlanOwnerLedger {
    /// Compose this collection's distinct retained owners with another live
    /// adapter domain before starting a size-dependent JPEG plan operation.
    pub(crate) fn external_live_bytes(&self, additional: usize) -> Result<usize, Error> {
        self.retained_bytes.checked_add(additional).ok_or_else(|| {
            JpegPlanCacheError::Invariant("JPEG Metal external plan baseline overflow").into()
        })
    }

    pub(crate) fn preflight(
        &self,
        retained: &[QueuedRequest],
        request: &QueuedRequest,
        cache_retained_bytes: usize,
    ) -> Result<PlanOwnerAdmission, Error> {
        preflight_request_count(retained.len())?;
        let input_is_retained = retained
            .iter()
            .any(|queued| SharedJpegInput::ptr_eq(&queued.input, &request.input));
        let packet_is_retained = request.fast_packet.as_ref().is_some_and(|packet| {
            retained.iter().any(|queued| {
                queued
                    .fast_packet
                    .as_ref()
                    .is_some_and(|owner| SharedJpegFastPacket::ptr_eq(owner, packet))
            })
        });
        let additional_input_bytes = if input_is_retained {
            0
        } else {
            request.retained_input_bytes()?
        };
        let additional_packet_bytes = if packet_is_retained {
            0
        } else {
            request.retained_packet_bytes()?
        };
        let retained_bytes = self
            .retained_bytes
            .checked_add(additional_input_bytes)
            .and_then(|bytes| bytes.checked_add(additional_packet_bytes))
            .ok_or(JpegPlanCacheError::Invariant(
                "JPEG Metal queued plan ledger overflow",
            ))?;
        // Cache and queue handles can share the same Arc owner. Charging the
        // complete graph to both domains is intentionally conservative and
        // keeps the bound stable after an entry is evicted from the cache.
        let requested = cache_retained_bytes.checked_add(retained_bytes).ok_or(
            JpegPlanCacheError::Invariant("JPEG Metal queued and cached plan ledger overflow"),
        )?;
        if requested > self.host_byte_limit {
            return Err(JpegPlanCacheError::Limit {
                what: "JPEG Metal queued and cached plan owner graphs",
                requested,
                cap: self.host_byte_limit,
            }
            .into());
        }
        Ok(PlanOwnerAdmission { retained_bytes })
    }

    pub(crate) const fn commit(&mut self, admission: PlanOwnerAdmission) {
        self.retained_bytes = admission.into_retained_bytes();
    }

    pub(crate) const fn reset(&mut self) {
        self.retained_bytes = 0;
    }

    pub(crate) const fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    #[cfg(test)]
    pub(crate) const fn host_byte_limit(&self) -> usize {
        self.host_byte_limit
    }

    #[cfg(test)]
    pub(crate) const fn set_host_byte_limit(&mut self, limit: usize) {
        self.host_byte_limit = limit;
    }
}
