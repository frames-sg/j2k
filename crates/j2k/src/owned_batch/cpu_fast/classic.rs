// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic JPEG 2000 compressed-arena preparation for the retained CPU fast path.

use j2k_core::BatchInfrastructureError;
use j2k_native::{J2kClassicCodeBlockPayload, J2kCodestreamRange};

use super::super::{BatchCodecRoute, PreparedBatchGroup, PreparedClassicPlan};
use super::plan::{
    append_input_range, classic_group_requirements, empty_range, reserve_reused, visit_classic_jobs,
};
use super::{CpuFlattenedPayloadJob, CpuGroupFastWorkspace, CpuPayloadBucket};

impl CpuGroupFastWorkspace {
    pub(super) fn prepare_classic(
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
                .map_or(0, PreparedClassicPlan::payload_count)
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
}
