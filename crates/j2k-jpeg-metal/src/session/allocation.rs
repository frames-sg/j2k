// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked capacity projections for session-owned metadata vectors.

use crate::Error;

pub(super) struct QueueGrowth {
    pub(super) queued: Option<Vec<crate::batch::QueuedRequest>>,
    pub(super) completed: Option<Vec<Option<Result<crate::Surface, crate::Error>>>>,
}

impl QueueGrowth {
    pub(super) fn queued_capacity(&self, current: usize) -> usize {
        self.queued.as_ref().map_or(current, Vec::capacity)
    }

    pub(super) fn completed_capacity(&self, current: usize) -> usize {
        self.completed.as_ref().map_or(current, Vec::capacity)
    }
}

pub(super) fn prepare_queue_growth(
    state: &super::SessionState,
    queued_owner_bytes: usize,
    execution_metadata_bytes: usize,
    projected_queued_capacity: usize,
    projected_completed_capacity: usize,
) -> Result<QueueGrowth, Error> {
    let queued_needs_growth = state.queued.len() == state.queued.capacity();
    let completed_needs_growth = state.completed.len() == state.completed.capacity();
    if !queued_needs_growth && !completed_needs_growth {
        state.preflight_collective_queue_state(
            queued_owner_bytes,
            execution_metadata_bytes,
            state.queued.capacity(),
            state.completed.capacity(),
        )?;
        return Ok(QueueGrowth {
            queued: None,
            completed: None,
        });
    }

    let old_metadata_bytes = state.session_metadata_live_bytes()?;
    let owner_cache_and_old_metadata = queued_owner_bytes
        .checked_add(state.jpeg_plans.diagnostics().retained_bytes)
        .and_then(|bytes| bytes.checked_add(execution_metadata_bytes))
        .and_then(|bytes| bytes.checked_add(old_metadata_bytes))
        .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
            "JPEG Metal transactional queue growth baseline overflow",
        ))?;
    let mut growth_budget = crate::batch_allocation::BatchMetadataBudget::with_external_live(
        "JPEG Metal transactional queue growth",
        owner_cache_and_old_metadata,
    );
    growth_budget.preflight(&[])?;
    let replacement_queued = queued_needs_growth
        .then(|| {
            growth_budget.try_vec(
                projected_queued_capacity,
                "JPEG Metal replacement queued requests",
            )
        })
        .transpose()?;
    let replacement_completed = completed_needs_growth
        .then(|| {
            growth_budget.try_vec(
                projected_completed_capacity,
                "JPEG Metal replacement completion slots",
            )
        })
        .transpose()?;

    #[cfg(test)]
    if let Some((queued_capacity, completed_capacity)) = state.queue_growth_capacity_override {
        let actual_queued = replacement_queued.as_ref().map_or(0, Vec::capacity);
        let actual_completed = replacement_completed.as_ref().map_or(0, Vec::capacity);
        growth_budget.preflight(&[
            crate::batch_allocation::BatchMetadataRequest::of::<crate::batch::QueuedRequest>(
                queued_capacity.saturating_sub(actual_queued),
            ),
            crate::batch_allocation::BatchMetadataRequest::of::<
                Option<Result<crate::Surface, crate::Error>>,
            >(completed_capacity.saturating_sub(actual_completed)),
        ])?;
    }
    Ok(QueueGrowth {
        queued: replacement_queued,
        completed: replacement_completed,
    })
}

pub(super) fn projected_push_capacity(
    len: usize,
    capacity: usize,
    what: &'static str,
) -> Result<usize, Error> {
    if len < capacity {
        return Ok(capacity);
    }
    let required =
        len.checked_add(1)
            .ok_or(j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what,
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
    capacity
        .checked_mul(2)
        .map(|doubled| doubled.max(required))
        .ok_or_else(|| {
            j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what,
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            }
            .into()
        })
}

pub(super) fn capacity_bytes<T>(capacity: usize, what: &'static str) -> Result<usize, Error> {
    capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| {
            j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what,
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            }
            .into()
        })
}

pub(crate) fn submission_capacity_bytes(capacity: usize) -> Result<usize, Error> {
    capacity_bytes::<crate::batch::MetalSubmission>(
        capacity,
        "JPEG Metal retained submission metadata",
    )
}
