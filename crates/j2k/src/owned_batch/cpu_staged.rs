// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded retained owners for staged cross-image CPU entropy dispatch.

use alloc::vec::Vec;
use core::{mem::size_of, ops::Range, sync::atomic::AtomicBool};
use std::sync::Mutex;

use j2k_core::BatchInfrastructureError;

use super::cpu_fast::{CpuFlattenedPayloadJob, CpuGroupFastWorkspace};
use crate::batch::allocation::{J2K_BATCH_METADATA_ALLOWANCE_BYTES, MAX_GENERIC_BATCH_WORKERS};
use crate::J2kError;

pub(super) const CPU_ENTROPY_IMAGES_PER_WINDOW: usize = MAX_GENERIC_BATCH_WORKERS * 2;

#[derive(Debug, Default)]
pub(super) struct CpuStagedWorkspace {
    image_scratch: Vec<Mutex<j2k_native::J2kDirectCpuScratch>>,
    failed: Vec<AtomicBool>,
    entropy_jobs: Vec<CpuFlattenedPayloadJob>,
    entropy_results: Vec<Option<CpuEntropyOutcome>>,
}

#[derive(Debug)]
pub(super) enum CpuEntropyOutcome {
    Complete,
    Error(J2kError),
}

pub(super) struct CpuStagedExecution<'a> {
    pub(super) image_scratch: &'a [Mutex<j2k_native::J2kDirectCpuScratch>],
    pub(super) failed: &'a [AtomicBool],
    pub(super) jobs: &'a mut [CpuFlattenedPayloadJob],
    pub(super) results: &'a mut [Option<CpuEntropyOutcome>],
}

impl CpuStagedWorkspace {
    pub(super) fn prepare_window(
        &mut self,
        _flattened: &CpuGroupFastWorkspace,
        image_range: Range<usize>,
    ) -> Result<(), BatchInfrastructureError> {
        let image_count = image_range.len();
        if image_count == 0 || image_count > CPU_ENTROPY_IMAGES_PER_WINDOW {
            return Err(BatchInfrastructureError::ResultIndexOutOfBounds {
                index: image_count,
                job_count: CPU_ENTROPY_IMAGES_PER_WINDOW,
            });
        }
        ensure_metadata_capacity(image_count, 0)?;
        reserve_for_len(
            &mut self.image_scratch,
            image_count,
            "J2K staged CPU image scratch owners",
        )?;
        while self.image_scratch.len() < image_count {
            self.image_scratch
                .push(Mutex::new(j2k_native::J2kDirectCpuScratch::new()));
        }
        self.image_scratch.truncate(image_count);

        reserve_for_len(
            &mut self.failed,
            image_count,
            "J2K staged CPU failure flags",
        )?;
        while self.failed.len() < image_count {
            self.failed.push(AtomicBool::new(false));
        }
        self.failed.truncate(image_count);
        for failed in &self.failed {
            failed.store(false, core::sync::atomic::Ordering::Relaxed);
        }

        self.entropy_jobs.clear();
        self.entropy_results.clear();
        Ok(())
    }

    pub(super) fn prepare_tile_jobs(
        &mut self,
        flattened: &CpuGroupFastWorkspace,
        image_range: Range<usize>,
        tile_index: usize,
    ) -> Result<(), BatchInfrastructureError> {
        let entropy_count = flattened
            .jobs()
            .iter()
            .filter(|job| {
                image_range.contains(&job.image_slot) && job.block_index.tile == tile_index
            })
            .count();
        ensure_metadata_capacity(image_range.len(), entropy_count)?;
        reserve_for_len(
            &mut self.entropy_jobs,
            entropy_count,
            "J2K staged CPU entropy jobs",
        )?;
        self.entropy_jobs.clear();
        self.entropy_jobs.extend(
            flattened
                .jobs()
                .iter()
                .filter(|job| {
                    image_range.contains(&job.image_slot) && job.block_index.tile == tile_index
                })
                .copied(),
        );

        reserve_for_len(
            &mut self.entropy_results,
            entropy_count,
            "J2K staged CPU entropy results",
        )?;
        self.entropy_results.clear();
        self.entropy_results.resize_with(entropy_count, || None);
        Ok(())
    }

    pub(super) fn execution(&mut self) -> CpuStagedExecution<'_> {
        CpuStagedExecution {
            image_scratch: &self.image_scratch,
            failed: &self.failed,
            jobs: &mut self.entropy_jobs,
            results: &mut self.entropy_results,
        }
    }
}

fn ensure_metadata_capacity(
    image_count: usize,
    entropy_count: usize,
) -> Result<(), BatchInfrastructureError> {
    let image_bytes = image_count
        .checked_mul(size_of::<Mutex<j2k_native::J2kDirectCpuScratch>>() + size_of::<AtomicBool>())
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K staged CPU metadata",
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        })?;
    let entropy_bytes = entropy_count
        .checked_mul(size_of::<CpuFlattenedPayloadJob>() + size_of::<Option<CpuEntropyOutcome>>())
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K staged CPU metadata",
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        })?;
    let requested = image_bytes.checked_add(entropy_bytes).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what: "J2K staged CPU metadata",
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        },
    )?;
    if requested > J2K_BATCH_METADATA_ALLOWANCE_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K staged CPU metadata",
            requested,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        });
    }
    Ok(())
}

fn reserve_for_len<T>(
    values: &mut Vec<T>,
    required: usize,
    what: &'static str,
) -> Result<(), BatchInfrastructureError> {
    if values.capacity() >= required {
        return Ok(());
    }
    let additional = required.saturating_sub(values.len());
    let bytes = required.saturating_mul(size_of::<T>());
    values
        .try_reserve_exact(additional)
        .map_err(|_| BatchInfrastructureError::HostAllocationFailed { what, bytes })
}
