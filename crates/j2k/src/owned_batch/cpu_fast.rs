// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained cross-image compressed arenas for the CPU prepared fast path.

use alloc::vec::Vec;
use core::mem::size_of;

use j2k_core::{BatchInfrastructureError, DEFAULT_MAX_HOST_ALLOCATION_BYTES};
use j2k_native::{
    HtCodeBlockPayloadRanges, J2kClassicCodeBlockPayload, J2kCodestreamRange,
    J2kDirectCodeBlockIndex, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
    J2kReferencedClassicPlan, J2kReferencedHtj2kPlan,
};

use super::{BatchCodecRoute, PreparedBatchGroup, PreparedImage};
use crate::batch::allocation::J2K_BATCH_METADATA_ALLOWANCE_BYTES;

mod plan;
use self::plan::{
    append_input_range, checked_metadata_bytes, classic_group_requirements, empty_range, ht_bucket,
    ht_bucket_index, ht_group_requirements, reserve_reused, visit_classic_jobs, visit_ht_jobs,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuPayloadBucket {
    Cleanup,
    SigProp,
    MagRef,
    Classic,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CpuFlattenedPayloadJob {
    pub(super) source_index: usize,
    pub(super) image_slot: usize,
    pub(super) payload_index: usize,
    pub(super) destination_index: usize,
    pub(super) block_index: J2kDirectCodeBlockIndex,
    bucket: CpuPayloadBucket,
    bucket_ordinal: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct CpuImagePayloadSpan {
    start: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct CpuFastWorkspaceStats {
    pub(super) flattened_group_plans: u64,
    pub(super) flattened_payload_jobs: u64,
    pub(super) flattened_cleanup_jobs: u64,
    pub(super) flattened_sigprop_jobs: u64,
    pub(super) flattened_magref_jobs: u64,
    pub(super) flattened_classic_jobs: u64,
    pub(super) entropy_job_dispatches: u64,
    pub(super) cross_image_entropy_windows: u64,
    pub(super) compressed_arena_reuses: u64,
    pub(super) retained_compressed_arena_bytes: usize,
    pub(super) output_group_allocations: u64,
    pub(super) output_compaction_copied_samples: u64,
}

#[derive(Debug, Default)]
pub(super) struct CpuGroupFastWorkspace {
    compressed_arena: Vec<u8>,
    jobs: Vec<CpuFlattenedPayloadJob>,
    image_spans: Vec<CpuImagePayloadSpan>,
    ht_payloads: Vec<HtCodeBlockPayloadRanges>,
    classic_payloads: Vec<J2kClassicCodeBlockPayload>,
    classic_ranges: Vec<J2kCodestreamRange>,
    route: Option<BatchCodecRoute>,
    stats: CpuFastWorkspaceStats,
}

impl CpuGroupFastWorkspace {
    pub(super) fn prepare_group(
        &mut self,
        group: &PreparedBatchGroup,
    ) -> Result<bool, BatchInfrastructureError> {
        self.clear_active_plan();
        let flattened = match group.info.route {
            BatchCodecRoute::Htj2k
                if group
                    .images
                    .iter()
                    .all(|image| image.htj2k_plan().is_some()) =>
            {
                self.prepare_htj2k(group)?;
                true
            }
            BatchCodecRoute::Classic
                if group
                    .images
                    .iter()
                    .all(|image| image.classic_plan().is_some()) =>
            {
                self.prepare_classic(group)?;
                true
            }
            BatchCodecRoute::Classic | BatchCodecRoute::Htj2k => false,
        };
        if flattened {
            self.stats.flattened_group_plans = self.stats.flattened_group_plans.saturating_add(1);
            self.stats.flattened_payload_jobs = self
                .stats
                .flattened_payload_jobs
                .saturating_add(self.jobs.len() as u64);
            self.record_job_buckets();
        }
        Ok(flattened)
    }

    pub(super) fn jobs(&self) -> &[CpuFlattenedPayloadJob] {
        &self.jobs
    }

    pub(super) fn arena(&self) -> &[u8] {
        &self.compressed_arena
    }

    pub(super) fn route(&self) -> Option<BatchCodecRoute> {
        self.route
    }

    pub(super) fn ht_payload(
        &self,
        job: CpuFlattenedPayloadJob,
    ) -> Option<HtCodeBlockPayloadRanges> {
        self.ht_payloads.get(job.destination_index).copied()
    }

    pub(super) fn classic_payload_range(
        &self,
        job: CpuFlattenedPayloadJob,
    ) -> Option<J2kCodestreamRange> {
        let descriptor = self.classic_payloads.get(job.destination_index)?;
        if descriptor.range_count != 1 {
            return None;
        }
        self.classic_ranges.get(job.destination_index).copied()
    }

    pub(super) fn record_output_group(
        &mut self,
        copied_samples: usize,
    ) -> Result<(), BatchInfrastructureError> {
        self.stats.output_group_allocations = self.stats.output_group_allocations.saturating_add(1);
        let copied_samples = u64::try_from(copied_samples).map_err(|_| {
            BatchInfrastructureError::AllocationTooLarge {
                what: "J2K CPU output compaction diagnostics",
                requested: copied_samples,
                cap: usize::MAX,
            }
        })?;
        self.stats.output_compaction_copied_samples = self
            .stats
            .output_compaction_copied_samples
            .saturating_add(copied_samples);
        Ok(())
    }

    pub(super) fn record_entropy_dispatch(&mut self, jobs: usize, images: usize) {
        self.stats.entropy_job_dispatches = self
            .stats
            .entropy_job_dispatches
            .saturating_add(jobs as u64);
        if jobs != 0 && images > 1 {
            self.stats.cross_image_entropy_windows =
                self.stats.cross_image_entropy_windows.saturating_add(1);
        }
    }

    pub(super) fn stats(&self) -> CpuFastWorkspaceStats {
        CpuFastWorkspaceStats {
            retained_compressed_arena_bytes: self.compressed_arena.capacity(),
            ..self.stats
        }
    }

    fn prepare_htj2k(
        &mut self,
        group: &PreparedBatchGroup,
    ) -> Result<(), BatchInfrastructureError> {
        let (payload_count, payload_bytes) = ht_group_requirements(group)?;
        self.prepare_storage::<HtCodeBlockPayloadRanges>(
            group.images.len(),
            payload_count,
            payload_bytes,
        )?;
        reserve_reused(
            &mut self.ht_payloads,
            payload_count,
            "J2K CPU flattened HT payload ranges",
        )?;
        self.ht_payloads.resize(
            payload_count,
            HtCodeBlockPayloadRanges {
                cleanup: empty_range(),
                refinement: None,
            },
        );
        self.assign_image_spans(group, |image| {
            image
                .htj2k_plan()
                .map_or(0, super::PreparedHtj2kPlan::payload_count)
        })?;

        for bucket in [
            CpuPayloadBucket::Cleanup,
            CpuPayloadBucket::SigProp,
            CpuPayloadBucket::MagRef,
        ] {
            for (image_slot, image) in group.images.iter().enumerate() {
                let plan = image
                    .htj2k_plan()
                    .ok_or(BatchInfrastructureError::MissingResult { index: image_slot })?;
                let span = self.image_spans[image_slot];
                let mut bucket_ordinals = [0_usize; 3];
                visit_ht_jobs(
                    plan.native_plan(),
                    |payload_index, block_index, coding_passes| {
                        if ht_bucket(coding_passes) == bucket {
                            let bucket_index = ht_bucket_index(bucket);
                            let bucket_ordinal = bucket_ordinals[bucket_index];
                            bucket_ordinals[bucket_index] = bucket_ordinal.saturating_add(1);
                            self.jobs.push(CpuFlattenedPayloadJob {
                                source_index: group.source_indices[image_slot],
                                image_slot,
                                payload_index,
                                destination_index: span.start + payload_index,
                                block_index,
                                bucket,
                                bucket_ordinal,
                            });
                        }
                    },
                );
            }
        }
        self.jobs.sort_unstable_by_key(|job| {
            (
                ht_bucket_index(job.bucket),
                job.bucket_ordinal,
                job.image_slot,
            )
        });
        if self.jobs.len() != payload_count {
            return Err(BatchInfrastructureError::MissingResult {
                index: self.jobs.len(),
            });
        }
        for job in &self.jobs {
            if group.source_indices.get(job.image_slot).copied() != Some(job.source_index) {
                return Err(BatchInfrastructureError::ResultIndexOutOfBounds {
                    index: job.source_index,
                    job_count: group.images.len(),
                });
            }
            let image = &group.images[job.image_slot];
            let payload = image
                .htj2k_plan()
                .and_then(|plan| plan.native_plan().payloads().get(job.payload_index))
                .copied()
                .ok_or(BatchInfrastructureError::MissingResult {
                    index: job.source_index,
                })?;
            let cleanup = append_input_range(
                &mut self.compressed_arena,
                image,
                payload.cleanup,
                job.source_index,
            )?;
            let refinement = payload
                .refinement
                .map(|range| {
                    append_input_range(&mut self.compressed_arena, image, range, job.source_index)
                })
                .transpose()?;
            self.ht_payloads[job.destination_index] = HtCodeBlockPayloadRanges {
                cleanup,
                refinement,
            };
        }
        self.finish_group(BatchCodecRoute::Htj2k, payload_bytes)
    }

    fn prepare_classic(
        &mut self,
        group: &PreparedBatchGroup,
    ) -> Result<(), BatchInfrastructureError> {
        let (payload_count, payload_bytes) = classic_group_requirements(group)?;
        self.prepare_storage::<(J2kClassicCodeBlockPayload, J2kCodestreamRange)>(
            group.images.len(),
            payload_count,
            payload_bytes,
        )?;
        reserve_reused(
            &mut self.classic_payloads,
            payload_count,
            "J2K CPU flattened classic payloads",
        )?;
        reserve_reused(
            &mut self.classic_ranges,
            payload_count,
            "J2K CPU flattened classic ranges",
        )?;
        self.classic_payloads.resize(
            payload_count,
            J2kClassicCodeBlockPayload {
                first_range: 0,
                range_count: 0,
                combined_length: 0,
            },
        );
        self.classic_ranges.resize(payload_count, empty_range());
        self.assign_image_spans(group, |image| {
            image
                .classic_plan()
                .map_or(0, super::PreparedClassicPlan::payload_count)
        })?;

        for (image_slot, image) in group.images.iter().enumerate() {
            let plan = image
                .classic_plan()
                .ok_or(BatchInfrastructureError::MissingResult { index: image_slot })?;
            let span = self.image_spans[image_slot];
            visit_classic_jobs(plan.native_plan(), |payload_index, block_index| {
                self.jobs.push(CpuFlattenedPayloadJob {
                    source_index: group.source_indices[image_slot],
                    image_slot,
                    payload_index,
                    destination_index: span.start + payload_index,
                    block_index,
                    bucket: CpuPayloadBucket::Classic,
                    bucket_ordinal: payload_index,
                });
            });
        }
        self.jobs
            .sort_unstable_by_key(|job| (job.bucket_ordinal, job.image_slot));
        for job in &self.jobs {
            let image = &group.images[job.image_slot];
            let plan = image
                .classic_plan()
                .ok_or(BatchInfrastructureError::MissingResult {
                    index: job.source_index,
                })?;
            let payload = plan.native_plan().payloads().get(job.payload_index).ok_or(
                BatchInfrastructureError::MissingResult {
                    index: job.source_index,
                },
            )?;
            let end_range =
                payload
                    .end_range()
                    .ok_or(BatchInfrastructureError::ResultIndexOutOfBounds {
                        index: payload.first_range,
                        job_count: plan.native_plan().ranges().len(),
                    })?;
            let fragments = plan
                .native_plan()
                .ranges()
                .get(payload.first_range..end_range)
                .ok_or(BatchInfrastructureError::ResultIndexOutOfBounds {
                    index: end_range,
                    job_count: plan.native_plan().ranges().len(),
                })?;
            let start = self.compressed_arena.len();
            for range in fragments {
                append_input_range(&mut self.compressed_arena, image, *range, job.source_index)?;
            }
            let length = self.compressed_arena.len() - start;
            if length != payload.combined_length {
                return Err(BatchInfrastructureError::MissingResult {
                    index: job.source_index,
                });
            }
            self.classic_ranges[job.destination_index] = J2kCodestreamRange {
                offset: start,
                length,
            };
            self.classic_payloads[job.destination_index] = J2kClassicCodeBlockPayload {
                first_range: job.payload_index,
                range_count: 1,
                combined_length: length,
            };
        }
        self.finish_group(BatchCodecRoute::Classic, payload_bytes)
    }

    fn prepare_storage<T>(
        &mut self,
        image_count: usize,
        payload_count: usize,
        payload_bytes: usize,
    ) -> Result<(), BatchInfrastructureError> {
        let metadata_bytes = checked_metadata_bytes::<T>(image_count, payload_count)?;
        if metadata_bytes > J2K_BATCH_METADATA_ALLOWANCE_BYTES {
            return Err(BatchInfrastructureError::AllocationTooLarge {
                what: "J2K CPU flattened group metadata",
                requested: metadata_bytes,
                cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
            });
        }
        if payload_bytes > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
            return Err(BatchInfrastructureError::AllocationTooLarge {
                what: "J2K CPU flattened compressed arena",
                requested: payload_bytes,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            });
        }
        if payload_bytes != 0 && self.compressed_arena.capacity() >= payload_bytes {
            self.stats.compressed_arena_reuses =
                self.stats.compressed_arena_reuses.saturating_add(1);
        }
        reserve_reused(
            &mut self.compressed_arena,
            payload_bytes,
            "J2K CPU flattened compressed arena",
        )?;
        reserve_reused(&mut self.jobs, payload_count, "J2K CPU flattened jobs")?;
        reserve_reused(
            &mut self.image_spans,
            image_count,
            "J2K CPU flattened image spans",
        )?;
        Ok(())
    }

    fn assign_image_spans(
        &mut self,
        group: &PreparedBatchGroup,
        payload_count: impl Fn(&PreparedImage) -> usize,
    ) -> Result<(), BatchInfrastructureError> {
        let mut start = 0usize;
        for image in &group.images {
            let len = payload_count(image);
            self.image_spans.push(CpuImagePayloadSpan { start });
            start = start
                .checked_add(len)
                .ok_or(BatchInfrastructureError::AllocationTooLarge {
                    what: "J2K CPU flattened image spans",
                    requested: usize::MAX,
                    cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
                })?;
        }
        Ok(())
    }

    fn finish_group(
        &mut self,
        route: BatchCodecRoute,
        expected_payload_bytes: usize,
    ) -> Result<(), BatchInfrastructureError> {
        if self.compressed_arena.len() != expected_payload_bytes {
            return Err(BatchInfrastructureError::MissingResult {
                index: self.compressed_arena.len(),
            });
        }
        self.route = Some(route);
        Ok(())
    }

    fn record_job_buckets(&mut self) {
        let (cleanup, sigprop, magref, classic) = self.jobs.iter().fold(
            (0_u64, 0_u64, 0_u64, 0_u64),
            |(cleanup, sigprop, magref, classic), job| match job.bucket {
                CpuPayloadBucket::Cleanup => (cleanup + 1, sigprop, magref, classic),
                CpuPayloadBucket::SigProp => (cleanup, sigprop + 1, magref, classic),
                CpuPayloadBucket::MagRef => (cleanup, sigprop, magref + 1, classic),
                CpuPayloadBucket::Classic => (cleanup, sigprop, magref, classic + 1),
            },
        );
        self.stats.flattened_cleanup_jobs =
            self.stats.flattened_cleanup_jobs.saturating_add(cleanup);
        self.stats.flattened_sigprop_jobs =
            self.stats.flattened_sigprop_jobs.saturating_add(sigprop);
        self.stats.flattened_magref_jobs = self.stats.flattened_magref_jobs.saturating_add(magref);
        self.stats.flattened_classic_jobs =
            self.stats.flattened_classic_jobs.saturating_add(classic);
    }

    fn clear_active_plan(&mut self) {
        self.compressed_arena.clear();
        self.jobs.clear();
        self.image_spans.clear();
        self.ht_payloads.clear();
        self.classic_payloads.clear();
        self.classic_ranges.clear();
        self.route = None;
    }
}
