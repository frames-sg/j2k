// SPDX-License-Identifier: MIT OR Apache-2.0

//! HT/classic payload traversal and checked flattened-plan sizing.

use super::{
    size_of, BatchInfrastructureError, CpuFlattenedPayloadJob, CpuImagePayloadSpan,
    CpuPayloadBucket, J2kCodestreamRange, J2kDirectCodeBlockIndex, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, PreparedBatchGroup,
    PreparedImage, Vec, DEFAULT_MAX_HOST_ALLOCATION_BYTES, J2K_BATCH_METADATA_ALLOWANCE_BYTES,
};

pub(super) fn ht_group_requirements(
    group: &PreparedBatchGroup,
) -> Result<(usize, usize), BatchInfrastructureError> {
    let mut payload_count = 0usize;
    let mut payload_bytes = 0usize;
    for (image_slot, image) in group.images.iter().enumerate() {
        let plan = image
            .htj2k_plan()
            .ok_or(BatchInfrastructureError::MissingResult { index: image_slot })?;
        let mut job_count = 0usize;
        visit_ht_jobs(plan.native_plan(), |_, _, _| job_count += 1);
        if job_count != plan.payload_count() {
            return Err(BatchInfrastructureError::MissingResult { index: image_slot });
        }
        payload_count = checked_add(payload_count, plan.payload_count(), "payload jobs")?;
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

pub(super) fn classic_group_requirements(
    group: &PreparedBatchGroup,
) -> Result<(usize, usize), BatchInfrastructureError> {
    let mut payload_count = 0usize;
    let mut payload_bytes = 0usize;
    for (image_slot, image) in group.images.iter().enumerate() {
        let plan = image
            .classic_plan()
            .ok_or(BatchInfrastructureError::MissingResult { index: image_slot })?;
        payload_count = checked_add(payload_count, plan.payload_count(), "payload jobs")?;
        for payload in plan.native_plan().payloads() {
            payload_bytes = checked_add(
                payload_bytes,
                payload.combined_length,
                "compressed payload bytes",
            )?;
        }
    }
    Ok((payload_count, payload_bytes))
}

pub(super) fn visit_ht_jobs(
    plan: &J2kReferencedHtj2kPlan,
    mut visit: impl FnMut(usize, J2kDirectCodeBlockIndex, u8),
) {
    let mut payload_index = 0usize;
    for (tile_index, tile) in plan.tiles().iter().enumerate() {
        if let Some(geometry) = tile.grayscale_geometry() {
            visit_ht_component_jobs(tile_index, 0, geometry, &mut payload_index, &mut visit);
        } else if let Some(geometry) = tile.color_geometry() {
            for (component_index, component) in geometry.component_plans.iter().enumerate() {
                visit_ht_component_jobs(
                    tile_index,
                    component_index,
                    component,
                    &mut payload_index,
                    &mut visit,
                );
            }
        } else if let Some(geometry) = tile.rgba_geometry() {
            for (component_index, component) in geometry.component_plans.iter().enumerate() {
                visit_ht_component_jobs(
                    tile_index,
                    component_index,
                    component,
                    &mut payload_index,
                    &mut visit,
                );
            }
        }
    }
}

fn visit_ht_component_jobs(
    tile_index: usize,
    component_index: usize,
    plan: &J2kDirectGrayscalePlan,
    payload_index: &mut usize,
    visit: &mut impl FnMut(usize, J2kDirectCodeBlockIndex, u8),
) {
    for (step_index, step) in plan.steps.iter().enumerate() {
        if let J2kDirectGrayscaleStep::HtSubBand(sub_band) = step {
            for (code_block, job) in sub_band.jobs.iter().enumerate() {
                visit(
                    *payload_index,
                    J2kDirectCodeBlockIndex {
                        tile: tile_index,
                        component: component_index,
                        step: step_index,
                        code_block,
                    },
                    job.number_of_coding_passes,
                );
                *payload_index = payload_index.saturating_add(1);
            }
        }
    }
}

pub(super) fn visit_classic_jobs(
    plan: &J2kReferencedClassicPlan,
    mut visit: impl FnMut(usize, J2kDirectCodeBlockIndex),
) {
    let mut payload_index = 0usize;
    for (tile_index, tile) in plan.tiles().iter().enumerate() {
        if let Some(geometry) = tile.grayscale_geometry() {
            visit_classic_component_jobs(tile_index, 0, geometry, &mut payload_index, &mut visit);
        } else if let Some(geometry) = tile.color_geometry() {
            for (component_index, component) in geometry.component_plans.iter().enumerate() {
                visit_classic_component_jobs(
                    tile_index,
                    component_index,
                    component,
                    &mut payload_index,
                    &mut visit,
                );
            }
        } else if let Some(geometry) = tile.rgba_geometry() {
            for (component_index, component) in geometry.component_plans.iter().enumerate() {
                visit_classic_component_jobs(
                    tile_index,
                    component_index,
                    component,
                    &mut payload_index,
                    &mut visit,
                );
            }
        }
    }
}

fn visit_classic_component_jobs(
    tile_index: usize,
    component_index: usize,
    plan: &J2kDirectGrayscalePlan,
    payload_index: &mut usize,
    visit: &mut impl FnMut(usize, J2kDirectCodeBlockIndex),
) {
    for (step_index, step) in plan.steps.iter().enumerate() {
        if let J2kDirectGrayscaleStep::ClassicSubBand(sub_band) = step {
            for code_block in 0..sub_band.jobs.len() {
                visit(
                    *payload_index,
                    J2kDirectCodeBlockIndex {
                        tile: tile_index,
                        component: component_index,
                        step: step_index,
                        code_block,
                    },
                );
                *payload_index = payload_index.saturating_add(1);
            }
        }
    }
}

pub(super) const fn ht_bucket(coding_passes: u8) -> CpuPayloadBucket {
    match coding_passes {
        0 | 1 => CpuPayloadBucket::Cleanup,
        2 => CpuPayloadBucket::SigProp,
        _ => CpuPayloadBucket::MagRef,
    }
}

pub(super) const fn ht_bucket_index(bucket: CpuPayloadBucket) -> usize {
    match bucket {
        CpuPayloadBucket::Cleanup => 0,
        CpuPayloadBucket::SigProp => 1,
        CpuPayloadBucket::MagRef => 2,
        CpuPayloadBucket::Classic => 3,
    }
}

pub(super) fn append_input_range(
    arena: &mut Vec<u8>,
    image: &PreparedImage,
    range: J2kCodestreamRange,
    source_index: usize,
) -> Result<J2kCodestreamRange, BatchInfrastructureError> {
    let end = range.offset.checked_add(range.length).ok_or(
        BatchInfrastructureError::ResultIndexOutOfBounds {
            index: range.offset,
            job_count: image.bytes().len(),
        },
    )?;
    let bytes = image.bytes().get(range.offset..end).ok_or(
        BatchInfrastructureError::ResultIndexOutOfBounds {
            index: source_index,
            job_count: image.bytes().len(),
        },
    )?;
    let offset = arena.len();
    arena.extend_from_slice(bytes);
    Ok(J2kCodestreamRange {
        offset,
        length: bytes.len(),
    })
}

pub(super) fn checked_metadata_bytes<T>(
    image_count: usize,
    payload_count: usize,
) -> Result<usize, BatchInfrastructureError> {
    image_count
        .checked_mul(size_of::<CpuImagePayloadSpan>())
        .and_then(|bytes| {
            payload_count
                .checked_mul(size_of::<CpuFlattenedPayloadJob>() + size_of::<T>())
                .and_then(|payload_bytes| bytes.checked_add(payload_bytes))
        })
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K CPU flattened group metadata",
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        })
}

pub(super) fn checked_add(
    current: usize,
    additional: usize,
    what: &'static str,
) -> Result<usize, BatchInfrastructureError> {
    current
        .checked_add(additional)
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })
}

pub(super) fn reserve_reused<T>(
    values: &mut Vec<T>,
    required: usize,
    what: &'static str,
) -> Result<(), BatchInfrastructureError> {
    values.clear();
    if values.capacity() >= required {
        return Ok(());
    }
    let bytes = required.saturating_mul(size_of::<T>());
    values
        .try_reserve_exact(required)
        .map_err(|_| BatchInfrastructureError::HostAllocationFailed { what, bytes })
}

pub(super) const fn empty_range() -> J2kCodestreamRange {
    J2kCodestreamRange {
        offset: 0,
        length: 0,
    }
}
