// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::mem::size_of;
use std::sync::Mutex;

use j2k_core::{tile_batch_worker_count, BatchInfrastructureError};

use super::allocation::{
    ensure_live_domains, ensure_metadata_bytes, ensure_planning_phase, select_batch_plan,
    try_vec_with_capacity, vec_capacity_bytes, BatchMetadataLayout, BatchPlan, PlannedJob,
    JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
};
use super::worker::WorkerSlot;
use super::{
    available_tile_batch_workers, BatchResultSlot, JpegBatchSession, SMALL_OUTPUT_BYTES,
    SMALL_OUTPUT_DEFAULT_WORKER_CAP,
};
use crate::context::DecoderContext;

impl JpegBatchSession {
    pub(super) fn prepare_job_planning(
        &mut self,
        job_count: usize,
    ) -> Result<usize, BatchInfrastructureError> {
        let requested_plan_bytes = job_count.checked_mul(size_of::<PlannedJob>()).ok_or(
            BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG batch job plans",
                requested: usize::MAX,
                cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
            },
        )?;

        // One existing context may participate in the planning decoder's
        // authoritative codec claim. All scratch and every other context are
        // evicted before parsing so no stale codec owner sits beside that full
        // claim. Worker-slot descriptors remain metadata and are charged below.
        for (index, slot) in self.workers.iter_mut().enumerate() {
            let worker = slot
                .get_mut()
                .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?;
            if index == 0 {
                worker.prepare_for_planning();
            } else {
                worker.release_allocations();
            }
        }

        let mut worker_metadata_bytes = vec_capacity_bytes(&self.workers)?;
        let combined = match ensure_metadata_bytes(
            worker_metadata_bytes,
            requested_plan_bytes,
            "JPEG planning metadata",
        ) {
            Ok(bytes) => bytes,
            Err(BatchInfrastructureError::AllocationTooLarge { .. }) => {
                self.workers = Vec::new();
                worker_metadata_bytes = 0;
                ensure_metadata_bytes(
                    worker_metadata_bytes,
                    requested_plan_bytes,
                    "JPEG planning metadata",
                )?
            }
            Err(error) => return Err(error),
        };
        ensure_planning_phase(combined)?;
        Ok(worker_metadata_bytes)
    }

    pub(super) fn planning_context(
        &mut self,
    ) -> Result<Option<&mut DecoderContext>, BatchInfrastructureError> {
        self.workers
            .first_mut()
            .map(|slot| {
                slot.get_mut()
                    .map(WorkerSlot::planning_context)
                    .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)
            })
            .transpose()
    }

    pub(super) fn prepare_batch<R, O>(
        &mut self,
        plans: &[PlannedJob],
        plan_capacity_bytes: usize,
        min_output_len: usize,
    ) -> Result<BatchPlan, BatchInfrastructureError> {
        let mut desired_workers =
            tile_batch_worker_count(plans.len(), self.options, available_tile_batch_workers());
        let small_output_default_batch = self.cap_small_output_default_workers
            && self.options.workers.is_none()
            && min_output_len <= SMALL_OUTPUT_BYTES;
        if small_output_default_batch {
            desired_workers = desired_workers.min(SMALL_OUTPUT_DEFAULT_WORKER_CAP);
        }

        let mut retained = self.prepare_retained_summary(plan_capacity_bytes)?;
        let mut metadata = self
            .batch_metadata_layout::<R, O>(plan_capacity_bytes, vec_capacity_bytes(&retained)?)?;
        let mut plan = select_batch_plan(plans, desired_workers, metadata, |worker| {
            retained.get(worker).copied().unwrap_or(0)
        });
        if plan.is_err() && retained.iter().any(|&bytes| bytes != 0) {
            for slot in &mut self.workers {
                slot.get_mut()
                    .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?
                    .release_allocations();
            }
            retained.fill(0);
            plan = select_batch_plan(plans, desired_workers, metadata, |_| 0);
        }
        if plan.is_err() && self.workers.capacity() > desired_workers {
            self.workers = Vec::new();
            retained = Vec::new();
            metadata = self.batch_metadata_layout::<R, O>(
                plan_capacity_bytes,
                vec_capacity_bytes(&retained)?,
            )?;
            plan = select_batch_plan(plans, desired_workers, metadata, |_| 0);
        }
        let mut plan = plan?;

        self.ensure_worker_slots(plan.worker_count)?;
        drop(retained);
        let requested_post_summary_bytes =
            self.workers.len().checked_mul(size_of::<usize>()).ok_or(
                BatchInfrastructureError::AllocationTooLarge {
                    what: "JPEG post-reservation worker allocation summary",
                    requested: usize::MAX,
                    cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
                },
            )?;
        self.ensure_planning_live_bytes(plan_capacity_bytes, requested_post_summary_bytes)?;
        let mut post_retained = try_vec_with_capacity(
            self.workers.len(),
            "JPEG post-reservation worker allocation summary",
        )?;
        let actual_post_summary_bytes = vec_capacity_bytes(&post_retained)?;
        self.ensure_planning_live_bytes(plan_capacity_bytes, actual_post_summary_bytes)?;
        for slot in &mut self.workers {
            post_retained.push(
                slot.get_mut()
                    .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?
                    .retained_bytes(),
            );
        }
        metadata = self.batch_metadata_layout::<R, O>(
            plan_capacity_bytes,
            vec_capacity_bytes(&post_retained)?,
        )?;
        metadata.worker_slot_capacity = self.workers.capacity();
        plan = select_batch_plan(plans, plan.worker_count, metadata, |worker| {
            post_retained.get(worker).copied().unwrap_or(0)
        })?;
        self.workers.truncate(plan.worker_count);
        self.active_workers = plan.worker_count;
        Ok(plan)
    }

    fn batch_metadata_layout<R, O>(
        &self,
        plan_capacity_bytes: usize,
        retained_capacity_bytes: usize,
    ) -> Result<BatchMetadataLayout, BatchInfrastructureError> {
        let fixed_bytes = plan_capacity_bytes
            .checked_add(retained_capacity_bytes)
            .ok_or(BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG batch planning metadata",
                requested: usize::MAX,
                cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
            })?;
        let handle_bytes = if self.options.workers.is_some() {
            size_of::<(
                usize,
                std::thread::ScopedJoinHandle<'static, Result<(), BatchInfrastructureError>>,
            )>()
        } else {
            0
        };
        Ok(BatchMetadataLayout {
            fixed_bytes,
            worker_slot_capacity: self.workers.capacity(),
            worker_slot_bytes: size_of::<Mutex<WorkerSlot>>(),
            worker_result_bytes: size_of::<BatchResultSlot<R>>(),
            ordered_result_bytes: size_of::<O>(),
            handle_bytes,
        })
    }

    fn prepare_retained_summary(
        &mut self,
        plan_capacity_bytes: usize,
    ) -> Result<Vec<usize>, BatchInfrastructureError> {
        let mut requested_summary_bytes =
            self.workers.len().checked_mul(size_of::<usize>()).ok_or(
                BatchInfrastructureError::AllocationTooLarge {
                    what: "JPEG retained worker allocation summary",
                    requested: usize::MAX,
                    cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
                },
            )?;
        if self.planning_live_exceeds_cap(plan_capacity_bytes, requested_summary_bytes)? {
            self.release_worker_allocations()?;
        }
        if self.planning_live_exceeds_cap(plan_capacity_bytes, requested_summary_bytes)? {
            self.workers = Vec::new();
            requested_summary_bytes = 0;
        }
        self.ensure_planning_live_bytes(plan_capacity_bytes, requested_summary_bytes)?;

        let mut retained = try_vec_with_capacity(
            self.workers.len(),
            "JPEG retained worker allocation summary",
        )?;
        for slot in &mut self.workers {
            retained.push(
                slot.get_mut()
                    .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?
                    .retained_bytes(),
            );
        }
        let actual_summary_bytes = vec_capacity_bytes(&retained)?;
        match self.ensure_planning_live_bytes(plan_capacity_bytes, actual_summary_bytes) {
            Ok(()) => return Ok(retained),
            Err(BatchInfrastructureError::AllocationTooLarge { .. }) => {}
            Err(error) => return Err(error),
        }
        // Allocator over-capacity is reconciled before any worker is
        // scheduled. Drop the summary first so clearing workers never creates
        // a second simultaneous caller-sized owner.
        drop(retained);
        self.release_worker_allocations()?;
        self.workers = Vec::new();
        self.ensure_planning_live_bytes(plan_capacity_bytes, 0)?;
        Ok(Vec::new())
    }

    fn release_worker_allocations(&mut self) -> Result<(), BatchInfrastructureError> {
        for slot in &mut self.workers {
            slot.get_mut()
                .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?
                .release_allocations();
        }
        Ok(())
    }

    fn planning_domain_bytes(
        &mut self,
        plan_capacity_bytes: usize,
        summary_capacity_bytes: usize,
    ) -> Result<(usize, usize), BatchInfrastructureError> {
        let mut metadata_bytes = checked_runtime_add(
            plan_capacity_bytes,
            vec_capacity_bytes(&self.workers)?,
            "JPEG batch planning metadata",
            JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
        )?;
        metadata_bytes = checked_runtime_add(
            metadata_bytes,
            summary_capacity_bytes,
            "JPEG batch planning metadata",
            JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
        )?;
        let mut codec_bytes = 0usize;
        for slot in &mut self.workers {
            codec_bytes = checked_runtime_add(
                codec_bytes,
                slot.get_mut()
                    .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?
                    .retained_bytes(),
                "JPEG retained worker codec claims",
                super::allocation::JPEG_CODEC_HOST_CAP_BYTES,
            )?;
        }
        Ok((codec_bytes, metadata_bytes))
    }

    fn ensure_planning_live_bytes(
        &mut self,
        plan_capacity_bytes: usize,
        summary_capacity_bytes: usize,
    ) -> Result<(), BatchInfrastructureError> {
        let (codec_bytes, metadata_bytes) =
            self.planning_domain_bytes(plan_capacity_bytes, summary_capacity_bytes)?;
        ensure_live_domains(codec_bytes, metadata_bytes, "JPEG batch planning live set")?;
        Ok(())
    }

    fn planning_live_exceeds_cap(
        &mut self,
        plan_capacity_bytes: usize,
        summary_capacity_bytes: usize,
    ) -> Result<bool, BatchInfrastructureError> {
        match self.ensure_planning_live_bytes(plan_capacity_bytes, summary_capacity_bytes) {
            Ok(()) => Ok(false),
            Err(BatchInfrastructureError::AllocationTooLarge { .. }) => Ok(true),
            Err(error) => Err(error),
        }
    }

    fn ensure_worker_slots(&mut self, worker_count: usize) -> Result<(), BatchInfrastructureError> {
        if self.workers.len() < worker_count {
            // Release stale workers before a fresh fallible exact reservation;
            // this avoids a transient old-plus-new worker-vector live set.
            self.workers = Vec::new();
            let mut workers = try_vec_with_capacity(worker_count, "JPEG batch worker slots")?;
            workers.resize_with(worker_count, || Mutex::new(WorkerSlot::default()));
            self.workers = workers;
        } else {
            self.workers.truncate(worker_count);
        }
        Ok(())
    }
}

fn checked_runtime_add(
    left: usize,
    right: usize,
    what: &'static str,
    cap: usize,
) -> Result<usize, BatchInfrastructureError> {
    left.checked_add(right)
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap,
        })
}
