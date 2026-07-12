// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) use j2k_core::{
    checked_batch_count_product as checked_count_product,
    checked_batch_count_sum as checked_count_sum,
    try_batch_reserve_for_push as try_reserve_for_push,
    BatchAllocationBudget as BatchMetadataBudget, BatchAllocationRequest as BatchMetadataRequest,
};

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use j2k_core::{AdapterErrorKind, AdapterErrorParts, BatchInfrastructureError};

    use super::{checked_count_product, BatchMetadataBudget, BatchMetadataRequest};
    use crate::abi::JpegEntropyCheckpointHost;
    use crate::Error;

    #[test]
    fn jpeg_entropy_checkpoint_plan_honors_exact_cap_and_one_byte_over() {
        let tile_count = 2;
        let segment_count = 3;
        let checkpoint_count = checked_count_product(
            tile_count,
            segment_count,
            "JPEG Metal batch entropy checkpoints",
        )
        .expect("checkpoint count");
        let exact_cap = 5
            + tile_count * size_of::<u32>() * 2
            + checkpoint_count * size_of::<JpegEntropyCheckpointHost>();
        let requests = [
            BatchMetadataRequest::of::<u8>(5),
            BatchMetadataRequest::of::<u32>(tile_count),
            BatchMetadataRequest::of::<u32>(tile_count),
            BatchMetadataRequest::of::<JpegEntropyCheckpointHost>(checkpoint_count),
        ];
        BatchMetadataBudget::with_cap("JPEG Metal batch entropy host data", exact_cap)
            .preflight(&requests)
            .expect("exact entropy cap");

        assert_eq!(
            BatchMetadataBudget::with_cap("JPEG Metal batch entropy host data", exact_cap - 1)
                .preflight(&requests)
                .expect_err("one byte over entropy cap"),
            BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG Metal batch entropy host data",
                requested: exact_cap,
                cap: exact_cap - 1,
            }
        );
    }

    #[test]
    fn jpeg_entropy_checkpoint_count_overflow_is_typed() {
        assert_eq!(
            checked_count_product(usize::MAX, 2, "JPEG Metal batch entropy checkpoints")
                .expect_err("checkpoint count overflow"),
            BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG Metal batch entropy checkpoints",
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            }
        );
    }

    #[test]
    fn grouped_result_plan_reconciles_two_simultaneous_result_vectors() {
        type ResultSlot = Option<Result<u16, Error>>;
        let count = 3;
        let exact_cap = count * size_of::<ResultSlot>() * 2;
        let mut budget =
            BatchMetadataBudget::with_cap("JPEG Metal grouped result collection", exact_cap);
        let merged = budget
            .try_filled(count, None::<Result<u16, Error>>, "merged result slots")
            .expect("merged result slots at exact cap");
        let ordered = budget
            .try_vec::<ResultSlot>(count, "ordered result slots")
            .expect("ordered result slots at exact cap");
        assert_eq!(merged.capacity(), count);
        assert_eq!(ordered.capacity(), count);

        let over =
            BatchMetadataBudget::with_cap("JPEG Metal grouped result collection", exact_cap - 1)
                .preflight(&[
                    BatchMetadataRequest::of::<ResultSlot>(count),
                    BatchMetadataRequest::of::<ResultSlot>(count),
                ])
                .expect_err("grouped result plan one byte over cap");
        assert_eq!(
            over,
            BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG Metal grouped result collection",
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
        assert_eq!(error.adapter_error_kind(), AdapterErrorKind::Other);
    }
}
