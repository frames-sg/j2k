// SPDX-License-Identifier: MIT OR Apache-2.0

//! Execution-time owner baselines derived from queued request plans.

use j2k_jpeg::adapter::JpegPlanCacheError;

use super::PlanOwnerLedger;
use crate::{batch::QueuedRequest, Error};

impl PlanOwnerLedger {
    pub(crate) fn from_requests(
        requests: &[QueuedRequest],
        cache_retained_bytes: usize,
    ) -> Result<Self, Error> {
        let mut ledger = Self::default();
        for (index, request) in requests.iter().enumerate() {
            let admission = ledger.preflight(&requests[..index], request, cache_retained_bytes)?;
            ledger.commit(admission);
        }
        Ok(ledger)
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn preflight_collective_metadata(
    phase: &'static str,
    owner_bytes: usize,
    cache_retained_bytes: usize,
    metadata_live_bytes: usize,
) -> Result<(), Error> {
    let owner_and_cache_bytes =
        owner_bytes
            .checked_add(cache_retained_bytes)
            .ok_or(JpegPlanCacheError::Invariant(
                "JPEG Metal collective metadata baseline overflow",
            ))?;
    crate::batch_allocation::BatchMetadataBudget::with_external_live(phase, owner_and_cache_bytes)
        .preflight(&[crate::batch_allocation::BatchMetadataRequest::of::<u8>(
            metadata_live_bytes,
        )])
        .map_err(Error::from)
}

pub(crate) fn batch_execution_budget(
    phase: &'static str,
    requests: &[QueuedRequest],
) -> Result<crate::batch_allocation::BatchMetadataBudget, Error> {
    let (cache_retained_bytes, external_live_bytes, collective_owner_bytes) =
        crate::batch::execution_owner_baseline(requests)?;
    let retained_owner_bytes = if collective_owner_bytes == 0 {
        PlanOwnerLedger::from_requests(requests, cache_retained_bytes)?.retained_bytes()
    } else {
        collective_owner_bytes
    };
    let owner_and_external_bytes = retained_owner_bytes
        .checked_add(cache_retained_bytes)
        .and_then(|bytes| bytes.checked_add(external_live_bytes))
        .ok_or(JpegPlanCacheError::Invariant(
            "JPEG Metal execution owner baseline overflow",
        ))?;
    let budget = crate::batch_allocation::BatchMetadataBudget::with_external_live(
        phase,
        owner_and_external_bytes,
    );
    budget.preflight(&[])?;
    Ok(budget)
}
