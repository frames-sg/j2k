// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic, bounded partitioning of compatible queued requests.

use super::{
    execution_cache_retained_bytes, stamp_execution_collective_owner_bytes, BatchKey, BatchKind,
    PlaneModeHint, QueuedRequest, SamplingFamily,
};
use crate::Error;
use j2k_core::{BackendRequest, PixelFormat};

type BatchSortKey = (u8, u8, u8, u32, u32, u32, Option<u16>, usize, u8, u8);

impl BatchKey {
    fn sort_key(self) -> BatchSortKey {
        let backend = match self.backend {
            BackendRequest::Auto => 0,
            BackendRequest::Cpu => 1,
            BackendRequest::Metal => 2,
            BackendRequest::Cuda => 3,
        };
        let fmt = match self.fmt {
            PixelFormat::Rgb8 => 0,
            PixelFormat::Rgba8 => 1,
            PixelFormat::Gray8 => 2,
            PixelFormat::Rgb16 => 3,
            PixelFormat::Rgba16 => 4,
            PixelFormat::Gray16 => 5,
            _ => u8::MAX,
        };
        let (kind, width, height, scale) = match self.kind {
            BatchKind::Full => (0, 0, 0, 0),
            BatchKind::Region { dims } => (1, dims.0, dims.1, 0),
            BatchKind::Scaled { scale } => (2, 0, 0, scale.denominator()),
            BatchKind::RegionScaled { dims, scale } => (3, dims.0, dims.1, scale.denominator()),
        };
        let sampling_family = match self.shape.sampling_family {
            SamplingFamily::Unknown => 0,
            SamplingFamily::Fast420 => 1,
            SamplingFamily::Fast422 => 2,
            SamplingFamily::Fast444 => 3,
            SamplingFamily::Other => 4,
        };
        let plane_mode = match self.shape.plane_mode {
            PlaneModeHint::Unknown => 0,
            PlaneModeHint::YCbCr => 1,
            PlaneModeHint::Rgb => 2,
        };
        (
            backend,
            fmt,
            kind,
            width,
            height,
            scale,
            self.shape.restart_interval,
            self.shape.checkpoint_count,
            sampling_family,
            plane_mode,
        )
    }
}

pub(super) fn add_execution_external_live_bytes(
    requests: &mut [QueuedRequest],
    additional_live_bytes: usize,
) -> Result<(), Error> {
    for request in requests.iter() {
        request
            .execution_external_live_bytes()
            .checked_add(additional_live_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal grouped execution metadata baseline overflow",
            ))?;
    }
    for request in requests.iter_mut() {
        request.execution_external_live_bytes += additional_live_bytes;
    }
    Ok(())
}

pub(super) fn group_compatible_requests(
    queued: &mut Vec<QueuedRequest>,
) -> Result<Vec<Vec<QueuedRequest>>, Error> {
    let cache_retained_bytes = execution_cache_retained_bytes(queued)?;
    let collective_owner_bytes =
        crate::plan_owner_ledger::PlanOwnerLedger::from_requests(queued, cache_retained_bytes)?
            .retained_bytes();
    stamp_execution_collective_owner_bytes(queued, collective_owner_bytes);
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal compatible request grouping",
        queued,
    )?;
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(queued.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<Vec<QueuedRequest>>(queued.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<QueuedRequest>(queued.len()),
    ])?;
    queued.sort_unstable_by_key(|request| request.key().sort_key());
    let mut batch_counts: Vec<usize> =
        budget.try_vec(queued.len(), "JPEG Metal compatible batch request counts")?;
    let mut previous_key = None;
    for request in queued.iter() {
        let key = request.key().sort_key();
        if previous_key == Some(key) {
            let count =
                batch_counts
                    .last_mut()
                    .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                        "JPEG Metal compatible batch count is missing",
                    ))?;
            *count = count.checked_add(1).ok_or(
                j2k_core::BatchInfrastructureError::AllocationTooLarge {
                    what: "JPEG Metal compatible batch request counts",
                    requested: usize::MAX,
                    cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                },
            )?;
        } else {
            batch_counts.push(1);
            previous_key = Some(key);
        }
    }

    let mut batches =
        budget.try_vec(batch_counts.len(), "JPEG Metal compatible request batches")?;
    for &count in &batch_counts {
        batches.push(budget.try_vec(count, "JPEG Metal compatible batch requests")?);
    }
    let mut requests = queued.drain(..);
    for (batch, &count) in batches.iter_mut().zip(&batch_counts) {
        for _ in 0..count {
            let request =
                requests
                    .next()
                    .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                        "JPEG Metal compatible request partition ended early",
                    ))?;
            batch.push(request);
        }
    }
    if requests.next().is_some() {
        return Err(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
            "JPEG Metal compatible request partition left unassigned requests",
        )
        .into());
    }
    let grouped_metadata_bytes = grouped_request_metadata_bytes(batches.capacity(), &batches)?;
    for batch in &mut batches {
        add_execution_external_live_bytes(batch, grouped_metadata_bytes)?;
    }
    Ok(batches)
}

pub(super) fn grouped_request_metadata_bytes(
    outer_capacity: usize,
    batches: &[Vec<QueuedRequest>],
) -> Result<usize, Error> {
    let outer_bytes = crate::batch_allocation::checked_count_product(
        outer_capacity,
        core::mem::size_of::<Vec<QueuedRequest>>(),
        "JPEG Metal compatible batch outer metadata",
    )?;
    batches.iter().try_fold(outer_bytes, |bytes, batch| {
        let inner_bytes = crate::batch_allocation::checked_count_product(
            batch.capacity(),
            core::mem::size_of::<QueuedRequest>(),
            "JPEG Metal compatible batch request metadata",
        )?;
        crate::batch_allocation::checked_count_sum(
            [bytes, inner_bytes],
            "JPEG Metal compatible grouped metadata",
        )
        .map_err(Error::from)
    })
}
