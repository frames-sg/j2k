// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Buffer, DirectScratchBuffer, DirectStatusCheck, Error, PreparedDirectGrayscalePlan,
    PreparedDirectGrayscaleStep,
};

const MAX_RETAINED_BUFFERS_PER_DIRECT_STEP: usize = 4;
const MAX_RETAINED_BUFFERS_PER_HT_JOB: usize = 2;
const MAX_SCRATCH_BUFFERS_PER_DIRECT_STEP: usize = 3;

pub(super) struct DirectExecutionMetadata {
    pub(super) retained_buffers: Vec<Buffer>,
    pub(super) status_checks: Vec<DirectStatusCheck>,
    pub(super) scratch_buffers: Vec<DirectScratchBuffer>,
}

pub(super) fn allocate_direct_execution_metadata(
    step_count: usize,
    ht_job_count: usize,
    mut budget: crate::batch_allocation::BatchMetadataBudget,
) -> Result<DirectExecutionMetadata, Error> {
    // A classic Tier-1 step retains at most coded/jobs/segments/state buffers
    // and at most coefficient/state/output scratch buffers. Other steps use
    // fewer owners.
    let base_retained_capacity = crate::batch_allocation::checked_count_product(
        step_count,
        MAX_RETAINED_BUFFERS_PER_DIRECT_STEP,
        "J2K Metal direct retained buffer metadata",
    )?;
    let ht_retained_capacity = crate::batch_allocation::checked_count_product(
        ht_job_count,
        MAX_RETAINED_BUFFERS_PER_HT_JOB,
        "J2K Metal chunked HT retained buffer metadata",
    )?;
    let retained_capacity = crate::batch_allocation::checked_count_sum(
        [base_retained_capacity, ht_retained_capacity],
        "J2K Metal aggregate retained buffer metadata",
    )?;
    let scratch_capacity = crate::batch_allocation::checked_count_product(
        step_count,
        MAX_SCRATCH_BUFFERS_PER_DIRECT_STEP,
        "J2K Metal direct scratch buffer metadata",
    )?;
    let status_capacity = step_count;
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<Buffer>(retained_capacity),
        crate::batch_allocation::BatchMetadataRequest::of::<DirectStatusCheck>(status_capacity),
        crate::batch_allocation::BatchMetadataRequest::of::<DirectScratchBuffer>(scratch_capacity),
    ])?;
    Ok(DirectExecutionMetadata {
        retained_buffers: budget.try_vec(
            retained_capacity,
            "J2K Metal direct retained buffer metadata",
        )?,
        status_checks: budget.try_vec(status_capacity, "J2K Metal direct status metadata")?,
        scratch_buffers: budget
            .try_vec(scratch_capacity, "J2K Metal direct scratch buffer metadata")?,
    })
}

pub(super) fn direct_ht_job_count<'a>(
    plans: impl IntoIterator<Item = &'a PreparedDirectGrayscalePlan>,
    what: &'static str,
) -> Result<usize, Error> {
    Ok(crate::batch_allocation::checked_count_sum(
        plans.into_iter().flat_map(|plan| {
            plan.steps.iter().filter_map(|step| match step {
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => Some(sub_band.jobs.len()),
                PreparedDirectGrayscaleStep::ClassicSubBand(_)
                | PreparedDirectGrayscaleStep::Idwt(_)
                | PreparedDirectGrayscaleStep::Store(_) => None,
            })
        }),
        what,
    )?)
}

pub(in crate::compute) fn extend_preallocated_retained_buffers(
    retained_buffers: &mut Vec<Buffer>,
    buffers: Vec<Buffer>,
) -> Result<(), Error> {
    ensure_preallocated_retained_buffer_capacity(
        retained_buffers.len(),
        buffers.len(),
        retained_buffers.capacity(),
    )?;
    retained_buffers.extend(buffers);
    Ok(())
}

fn ensure_preallocated_retained_buffer_capacity(
    current_len: usize,
    additional: usize,
    capacity: usize,
) -> Result<(), Error> {
    let required = current_len
        .checked_add(additional)
        .ok_or(Error::MetalStateInvariant {
            state: "J2K Metal direct retained buffer metadata",
            reason: "retained owner count overflowed after allocation preflight",
        })?;
    if required > capacity {
        return Err(Error::MetalStateInvariant {
            state: "J2K Metal direct retained buffer metadata",
            reason: "chunked execution exceeded its preallocated retained owner capacity",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use j2k_core::BatchInfrastructureError;

    use super::*;
    use crate::batch_allocation::BatchMetadataBudget;

    #[test]
    fn direct_execution_resources_honor_exact_cap_and_one_byte_over() {
        let step_count = 3;
        let retained_capacity = step_count * MAX_RETAINED_BUFFERS_PER_DIRECT_STEP;
        let status_capacity = step_count;
        let scratch_capacity = step_count * MAX_SCRATCH_BUFFERS_PER_DIRECT_STEP;
        let exact_cap = retained_capacity * size_of::<Buffer>()
            + status_capacity * size_of::<DirectStatusCheck>()
            + scratch_capacity * size_of::<DirectScratchBuffer>();
        let owners = allocate_direct_execution_metadata(
            step_count,
            0,
            BatchMetadataBudget::with_cap(
                "J2K Metal direct execution resource metadata",
                exact_cap,
            ),
        )
        .expect("exact direct execution metadata cap");
        assert_eq!(owners.retained_buffers.capacity(), retained_capacity);
        assert_eq!(owners.status_checks.capacity(), status_capacity);
        assert_eq!(owners.scratch_buffers.capacity(), scratch_capacity);

        assert!(matches!(
            allocate_direct_execution_metadata(
                step_count,
                0,
                BatchMetadataBudget::with_cap(
                    "J2K Metal direct execution resource metadata",
                    exact_cap - 1,
                ),
            ),
            Err(Error::BatchInfrastructure(
                BatchInfrastructureError::AllocationTooLarge {
                    requested,
                    cap,
                    ..
                }
            )) if requested == exact_cap && cap == exact_cap - 1
        ));
    }

    #[test]
    fn chunked_ht_buffer_owners_are_preallocated_and_cannot_grow_past_the_budget() {
        let step_count = 1;
        let ht_job_count = 3;
        let retained_capacity = step_count * MAX_RETAINED_BUFFERS_PER_DIRECT_STEP
            + ht_job_count * MAX_RETAINED_BUFFERS_PER_HT_JOB;
        let status_capacity = step_count;
        let scratch_capacity = step_count * MAX_SCRATCH_BUFFERS_PER_DIRECT_STEP;
        let exact_cap = retained_capacity * size_of::<Buffer>()
            + status_capacity * size_of::<DirectStatusCheck>()
            + scratch_capacity * size_of::<DirectScratchBuffer>();
        let owners = allocate_direct_execution_metadata(
            step_count,
            ht_job_count,
            BatchMetadataBudget::with_cap("chunked HT execution owners", exact_cap),
        )
        .expect("chunked HT owners fit their exact aggregate cap");
        assert_eq!(owners.retained_buffers.capacity(), retained_capacity);
        ensure_preallocated_retained_buffer_capacity(
            step_count * MAX_RETAINED_BUFFERS_PER_DIRECT_STEP,
            ht_job_count * MAX_RETAINED_BUFFERS_PER_HT_JOB,
            owners.retained_buffers.capacity(),
        )
        .expect("preallocated HT chunk owners");
        assert!(ensure_preallocated_retained_buffer_capacity(
            step_count * MAX_RETAINED_BUFFERS_PER_DIRECT_STEP,
            ht_job_count * MAX_RETAINED_BUFFERS_PER_HT_JOB + 1,
            owners.retained_buffers.capacity(),
        )
        .is_err());
    }
}
