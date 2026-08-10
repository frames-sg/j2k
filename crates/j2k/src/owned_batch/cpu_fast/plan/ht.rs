// SPDX-License-Identifier: MIT OR Apache-2.0

//! HTJ2K payload-record traversal and flattened-plan sizing.

use j2k_core::BatchInfrastructureError;
use j2k_native::{
    HtCodeBlockPayloadRanges, J2kDirectCodeBlockIndex, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, J2kReferencedHtj2kPlan, J2kReferencedPayloadRecordSpan,
};

use super::checked_add;
use crate::owned_batch::{cpu_fast::CpuPayloadBucket, PreparedBatchGroup};

#[cfg(test)]
#[path = "ht_tests.rs"]
mod tests;

pub(in crate::owned_batch::cpu_fast) fn ht_group_requirements(
    group: &PreparedBatchGroup,
) -> Result<(usize, usize), BatchInfrastructureError> {
    let mut payload_count = 0usize;
    let mut payload_bytes = 0usize;
    for (image_slot, image) in group.images.iter().enumerate() {
        let plan = image
            .htj2k_plan()
            .ok_or(BatchInfrastructureError::MissingResult { index: image_slot })?;
        let job_count = visit_ht_jobs(plan.native_plan(), |_, _, _, _| {})?;
        payload_count = checked_add(payload_count, job_count, "payload jobs")?;
        for payload in plan.native_plan().payloads() {
            payload_bytes = checked_add(
                payload_bytes,
                payload.cleanup.length,
                "compressed payload bytes",
            )?;
            if let Some(refinement) = payload.refinement {
                payload_bytes =
                    checked_add(payload_bytes, refinement.length, "compressed payload bytes")?;
            }
        }
    }
    Ok((payload_count, payload_bytes))
}

pub(in crate::owned_batch::cpu_fast) fn visit_ht_jobs(
    plan: &J2kReferencedHtj2kPlan,
    mut visit: impl FnMut(usize, J2kReferencedPayloadRecordSpan, J2kDirectCodeBlockIndex, u8),
) -> Result<usize, BatchInfrastructureError> {
    let mut job_index = 0usize;
    let mut payload_record = 0usize;
    for (tile_index, tile) in plan.tiles().iter().enumerate() {
        if let Some(geometry) = tile.grayscale_geometry() {
            visit_ht_component_jobs(
                tile_index,
                0,
                geometry,
                plan.payloads(),
                &mut payload_record,
                &mut job_index,
                &mut visit,
            )?;
        } else if let Some(geometry) = tile.color_geometry() {
            for (component_index, component) in geometry.component_plans.iter().enumerate() {
                visit_ht_component_jobs(
                    tile_index,
                    component_index,
                    component,
                    plan.payloads(),
                    &mut payload_record,
                    &mut job_index,
                    &mut visit,
                )?;
            }
        } else if let Some(geometry) = tile.rgba_geometry() {
            for (component_index, component) in geometry.component_plans.iter().enumerate() {
                visit_ht_component_jobs(
                    tile_index,
                    component_index,
                    component,
                    plan.payloads(),
                    &mut payload_record,
                    &mut job_index,
                    &mut visit,
                )?;
            }
        }
    }
    if payload_record != plan.payloads().len() {
        return Err(HT_PAYLOAD_RECORD_MISMATCH);
    }
    Ok(job_index)
}

fn visit_ht_component_jobs(
    tile_index: usize,
    component_index: usize,
    plan: &J2kDirectGrayscalePlan,
    payloads: &[HtCodeBlockPayloadRanges],
    payload_record: &mut usize,
    job_index: &mut usize,
    visit: &mut impl FnMut(usize, J2kReferencedPayloadRecordSpan, J2kDirectCodeBlockIndex, u8),
) -> Result<(), BatchInfrastructureError> {
    for (step_index, step) in plan.steps.iter().enumerate() {
        if let J2kDirectGrayscaleStep::HtSubBand(sub_band) = step {
            for (code_block, job) in sub_band.jobs.iter().enumerate() {
                let payload_records = next_ht_payload_record_span(
                    payloads,
                    payload_record,
                    job.cleanup_length,
                    job.refinement_length,
                )?;
                visit(
                    *job_index,
                    payload_records,
                    J2kDirectCodeBlockIndex {
                        tile: tile_index,
                        component: component_index,
                        step: step_index,
                        code_block,
                    },
                    job.number_of_coding_passes,
                );
                *job_index = checked_add(*job_index, 1, "payload jobs")?;
            }
        }
    }
    Ok(())
}

const HT_PAYLOAD_RECORD_MISMATCH: BatchInfrastructureError =
    BatchInfrastructureError::UnsupportedContract {
        what: "retained HT payload records do not match code-block geometry",
    };

fn next_ht_payload_record_span(
    payloads: &[HtCodeBlockPayloadRanges],
    cursor: &mut usize,
    cleanup_length: u32,
    refinement_length: u32,
) -> Result<J2kReferencedPayloadRecordSpan, BatchInfrastructureError> {
    let first_record = *cursor;
    let first = payloads
        .get(first_record)
        .ok_or(HT_PAYLOAD_RECORD_MISMATCH)?;
    if first.cleanup.length != cleanup_length as usize {
        return Err(HT_PAYLOAD_RECORD_MISMATCH);
    }
    let expected_refinement = refinement_length as usize;
    let mut refinement_bytes = first.refinement.map_or(0, |range| range.length);
    if refinement_bytes > expected_refinement {
        return Err(HT_PAYLOAD_RECORD_MISMATCH);
    }
    let mut next_record = checked_add(first_record, 1, "HT payload record cursor")?;
    while refinement_bytes < expected_refinement {
        let continuation = payloads
            .get(next_record)
            .ok_or(HT_PAYLOAD_RECORD_MISMATCH)?;
        let continuation_length = continuation
            .refinement
            .filter(|range| range.length != 0)
            .ok_or(HT_PAYLOAD_RECORD_MISMATCH)?
            .length;
        if continuation.cleanup.length != 0 {
            return Err(HT_PAYLOAD_RECORD_MISMATCH);
        }
        refinement_bytes = refinement_bytes
            .checked_add(continuation_length)
            .ok_or(HT_PAYLOAD_RECORD_MISMATCH)?;
        if refinement_bytes > expected_refinement {
            return Err(HT_PAYLOAD_RECORD_MISMATCH);
        }
        next_record = checked_add(next_record, 1, "HT payload record cursor")?;
    }
    let record_count = next_record - first_record;
    *cursor = next_record;
    Ok(J2kReferencedPayloadRecordSpan {
        first_record,
        record_count,
    })
}

pub(in crate::owned_batch::cpu_fast) const fn ht_bucket(coding_passes: u8) -> CpuPayloadBucket {
    match coding_passes {
        0 | 1 => CpuPayloadBucket::Cleanup,
        2 => CpuPayloadBucket::SigProp,
        _ => CpuPayloadBucket::MagRef,
    }
}

pub(in crate::owned_batch::cpu_fast) const fn ht_bucket_index(bucket: CpuPayloadBucket) -> usize {
    match bucket {
        CpuPayloadBucket::Cleanup => 0,
        CpuPayloadBucket::SigProp => 1,
        CpuPayloadBucket::MagRef => 2,
        CpuPayloadBucket::Classic => 3,
    }
}
