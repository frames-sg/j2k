// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Buffer, DirectScratchBuffer, DirectStatusCheck, Error};

const MAX_RETAINED_BUFFERS_PER_DIRECT_STEP: usize = 4;
const MAX_SCRATCH_BUFFERS_PER_DIRECT_STEP: usize = 3;

pub(super) struct DirectExecutionMetadata {
    pub(super) retained_buffers: Vec<Buffer>,
    pub(super) status_checks: Vec<DirectStatusCheck>,
    pub(super) scratch_buffers: Vec<DirectScratchBuffer>,
}

pub(super) fn allocate_direct_execution_metadata(
    step_count: usize,
    extra_status_count: usize,
    mut budget: crate::batch_allocation::BatchMetadataBudget,
) -> Result<DirectExecutionMetadata, Error> {
    // A classic Tier-1 step retains at most coded/jobs/segments/state buffers
    // and at most coefficient/state/output scratch buffers. Other steps use
    // fewer owners. Color MCT contributes one additional status per plan.
    let retained_capacity = crate::batch_allocation::checked_count_product(
        step_count,
        MAX_RETAINED_BUFFERS_PER_DIRECT_STEP,
        "J2K Metal direct retained buffer metadata",
    )?;
    let scratch_capacity = crate::batch_allocation::checked_count_product(
        step_count,
        MAX_SCRATCH_BUFFERS_PER_DIRECT_STEP,
        "J2K Metal direct scratch buffer metadata",
    )?;
    let status_capacity = crate::batch_allocation::checked_count_sum(
        [step_count, extra_status_count],
        "J2K Metal direct status metadata",
    )?;
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

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use j2k_core::BatchInfrastructureError;

    use super::*;
    use crate::batch_allocation::BatchMetadataBudget;

    #[test]
    fn direct_execution_resources_honor_exact_cap_and_one_byte_over() {
        let step_count = 3;
        let extra_status_count = 2;
        let retained_capacity = step_count * MAX_RETAINED_BUFFERS_PER_DIRECT_STEP;
        let status_capacity = step_count + extra_status_count;
        let scratch_capacity = step_count * MAX_SCRATCH_BUFFERS_PER_DIRECT_STEP;
        let exact_cap = retained_capacity * size_of::<Buffer>()
            + status_capacity * size_of::<DirectStatusCheck>()
            + scratch_capacity * size_of::<DirectScratchBuffer>();
        let owners = allocate_direct_execution_metadata(
            step_count,
            extra_status_count,
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
                extra_status_count,
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
}
