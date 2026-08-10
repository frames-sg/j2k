// SPDX-License-Identifier: MIT OR Apache-2.0

//! HTJ2K compressed-arena preparation for the retained CPU fast path.

use j2k_core::BatchInfrastructureError;
use j2k_native::HtCodeBlockPayloadRanges;

use super::super::{BatchCodecRoute, PreparedBatchGroup};
use super::plan::{
    append_input_range, empty_range, ht_bucket, ht_bucket_index, ht_group_requirements,
    reserve_reused, visit_ht_jobs,
};
use super::{CpuFlattenedPayloadJob, CpuGroupFastWorkspace, CpuPayloadBucket};

impl CpuGroupFastWorkspace {
    pub(super) fn prepare_htj2k(
        &mut self,
        group: &PreparedBatchGroup,
    ) -> Result<(), BatchInfrastructureError> {
        let (payload_count, payload_bytes) = ht_group_requirements(group)?;
        self.initialize_ht_storage(group, payload_count, payload_bytes)?;
        self.collect_ht_jobs(group, payload_count)?;
        self.materialize_ht_payloads(group)?;
        self.finish_group(BatchCodecRoute::Htj2k, payload_bytes)
    }

    fn initialize_ht_storage(
        &mut self,
        group: &PreparedBatchGroup,
        payload_count: usize,
        payload_bytes: usize,
    ) -> Result<(), BatchInfrastructureError> {
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
            let plan = image
                .htj2k_plan()
                .ok_or(BatchInfrastructureError::MissingResult { index: 0 })?;
            visit_ht_jobs(plan.native_plan(), |_, _, _, _| {})
        })
    }

    fn collect_ht_jobs(
        &mut self,
        group: &PreparedBatchGroup,
        payload_count: usize,
    ) -> Result<(), BatchInfrastructureError> {
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
                    |job_index, payload_records, block_index, coding_passes| {
                        if ht_bucket(coding_passes) == bucket {
                            let bucket_index = ht_bucket_index(bucket);
                            let bucket_ordinal = bucket_ordinals[bucket_index];
                            bucket_ordinals[bucket_index] = bucket_ordinal.saturating_add(1);
                            self.jobs.push(CpuFlattenedPayloadJob {
                                source_index: group.source_indices[image_slot],
                                image_slot,
                                payload_index: payload_records.first_record,
                                payload_record_count: payload_records.record_count,
                                destination_index: span.start + job_index,
                                block_index,
                                bucket,
                                bucket_ordinal,
                            });
                        }
                    },
                )?;
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
        Ok(())
    }

    fn materialize_ht_payloads(
        &mut self,
        group: &PreparedBatchGroup,
    ) -> Result<(), BatchInfrastructureError> {
        for job in &self.jobs {
            if group.source_indices.get(job.image_slot).copied() != Some(job.source_index) {
                return Err(BatchInfrastructureError::ResultIndexOutOfBounds {
                    index: job.source_index,
                    job_count: group.images.len(),
                });
            }
            let image = &group.images[job.image_slot];
            let payload_end = job
                .payload_index
                .checked_add(job.payload_record_count)
                .ok_or(BatchInfrastructureError::ResultIndexOutOfBounds {
                    index: job.payload_index,
                    job_count: image.bytes().len(),
                })?;
            let payloads = image
                .htj2k_plan()
                .and_then(|plan| {
                    plan.native_plan()
                        .payloads()
                        .get(job.payload_index..payload_end)
                })
                .ok_or(BatchInfrastructureError::MissingResult {
                    index: job.source_index,
                })?;
            let payload =
                payloads
                    .first()
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
            let refinement_start = self.compressed_arena.len();
            for (record_index, payload) in payloads.iter().enumerate() {
                if record_index != 0 {
                    append_input_range(
                        &mut self.compressed_arena,
                        image,
                        payload.cleanup,
                        job.source_index,
                    )?;
                }
                if let Some(range) = payload.refinement {
                    append_input_range(&mut self.compressed_arena, image, range, job.source_index)?;
                }
            }
            let refinement_length = self.compressed_arena.len() - refinement_start;
            let refinement = (refinement_length != 0).then_some(j2k_native::J2kCodestreamRange {
                offset: refinement_start,
                length: refinement_length,
            });
            self.ht_payloads[job.destination_index] = HtCodeBlockPayloadRanges {
                cleanup,
                refinement,
            };
        }
        Ok(())
    }
}
