// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) use j2k_core::{
    checked_batch_count_product as checked_count_product,
    checked_batch_count_sum as checked_count_sum,
    try_batch_reserve_for_push as try_reserve_for_push, try_batch_reserve_to as try_reserve_to,
    BatchAllocationBudget as BatchMetadataBudget, BatchAllocationRequest as BatchMetadataRequest,
};

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use j2k_core::{BatchInfrastructureError, CodecError};

    use super::{BatchMetadataBudget, BatchMetadataRequest};
    use crate::Error;

    #[test]
    fn j2k_grouped_result_plan_honors_exact_cap_and_one_byte_over() {
        type ResultSlot = Option<Result<u16, Error>>;
        let count = 3;
        let exact_cap = count * size_of::<ResultSlot>() * 2;
        let requests = [
            BatchMetadataRequest::of::<ResultSlot>(count),
            BatchMetadataRequest::of::<ResultSlot>(count),
        ];
        let mut exact =
            BatchMetadataBudget::with_cap("J2K Metal grouped result collection", exact_cap);
        exact
            .preflight(&requests)
            .expect("exact grouped result cap");
        let first = exact
            .try_vec::<ResultSlot>(count, "J2K Metal grouped result slots")
            .expect("first result vector");
        let second = exact
            .try_vec::<ResultSlot>(count, "J2K Metal ordered result slots")
            .expect("second result vector");
        assert_eq!(first.capacity(), count);
        assert_eq!(second.capacity(), count);

        assert_eq!(
            BatchMetadataBudget::with_cap("J2K Metal grouped result collection", exact_cap - 1)
                .preflight(&requests)
                .expect_err("one byte over grouped result cap"),
            BatchInfrastructureError::AllocationTooLarge {
                what: "J2K Metal grouped result collection",
                requested: exact_cap,
                cap: exact_cap - 1,
            }
        );
    }

    #[test]
    fn allocator_failure_keeps_batch_infrastructure_source_and_category() {
        let mut budget = BatchMetadataBudget::with_cap("test batch metadata", usize::MAX);
        let source = budget
            .try_vec::<u8>(usize::MAX, "test result slots")
            .expect_err("impossible reservation");
        let error = Error::from(source);

        assert!(matches!(
            &error,
            Error::BatchInfrastructure(BatchInfrastructureError::HostAllocationFailed {
                what: "test result slots",
                bytes: usize::MAX,
            })
        ));
        assert!(std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<BatchInfrastructureError>())
            .is_some());
        assert!(!error.is_unsupported());
        assert!(!error.is_buffer_error());
    }
}
