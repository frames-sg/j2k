// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;
use core::num::NonZeroUsize;
use std::sync::Arc;

use j2k_core::HtGpuJobChunkLimits;

use super::super::super::{HtBatchInput, HtPayloadSource, J2kHtCleanupBatchJob};
use super::entry::{fits_empty_cache, input_key_host_bytes, PreparedMetalHtInputKey};
use super::{prepared_metal_ht_execution, PreparedMetalHtExecutionCache};
use crate::compute::test_counters::{
    ht_immutable_job_uploads_for_test, ht_immutable_payload_uploads_for_test,
    reset_ht_immutable_job_uploads_for_test, reset_ht_immutable_payload_uploads_for_test,
};
use crate::compute::{Error, MetalRuntime, PreparedHtExecutionOwner};
use crate::session::{PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES, PREPARED_PLAN_CACHE_MAX_HOST_BYTES};

fn cleanup_job(output_offset: u32) -> J2kHtCleanupBatchJob {
    J2kHtCleanupBatchJob {
        coded_offset: 0,
        width: 1,
        height: 1,
        coded_len: 1,
        cleanup_length: 1,
        refinement_length: 0,
        missing_msbs: 0,
        num_bitplanes: 1,
        roi_shift: 0,
        number_of_coding_passes: 1,
        output_stride: 1,
        output_offset,
        dequantization_step: 1.0,
        stripe_causal: 0,
    }
}

#[test]
fn host_pressure_evicts_the_oldest_prepared_execution_instead_of_bypassing_cache() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let limits = HtGpuJobChunkLimits::new(
        NonZeroUsize::new(1).expect("nonzero job cap"),
        1,
        size_of::<J2kHtCleanupBatchJob>(),
    );
    let payload = [0_u8];
    let first_jobs = [cleanup_job(0)];
    let second_jobs = [cleanup_job(1)];
    let first_owner = Arc::new(PreparedHtExecutionOwner);
    let second_owner = Arc::new(PreparedHtExecutionOwner);
    let first = [HtBatchInput {
        source_index: 0,
        payload: HtPayloadSource::Contiguous(&payload),
        jobs: &first_jobs,
        output_base: 0,
        execution_owner: &first_owner,
    }];
    let second = [HtBatchInput {
        source_index: 1,
        payload: HtPayloadSource::Contiguous(&payload),
        jobs: &second_jobs,
        output_base: 1,
        execution_owner: &second_owner,
    }];

    let mut cache = PreparedMetalHtExecutionCache::new();
    cache
        .get_or_prepare(&device, &first, limits)
        .expect("seed prepared execution cache");

    let metadata_bytes = cache
        .retained_host_bytes
        .checked_sub(cache.entries[0].host_bytes())
        .expect("cache metadata accounting");
    let pressured_entry_bytes = PREPARED_PLAN_CACHE_MAX_HOST_BYTES
        .checked_sub(metadata_bytes)
        .expect("cache metadata fits host limit");
    cache.entries[0].set_host_bytes_for_test(pressured_entry_bytes);
    cache.retained_host_bytes = PREPARED_PLAN_CACHE_MAX_HOST_BYTES;

    cache
        .get_or_prepare(&device, &second, limits)
        .expect("rotate prepared execution cache under host pressure");
    reset_ht_immutable_payload_uploads_for_test();
    reset_ht_immutable_job_uploads_for_test();
    cache
        .get_or_prepare(&device, &second, limits)
        .expect("reuse rotated prepared execution");

    assert_eq!(ht_immutable_payload_uploads_for_test(), 0);
    assert_eq!(ht_immutable_job_uploads_for_test(), 0);
}

#[test]
fn prepared_execution_key_weight_uses_allocated_capacity() {
    let capacity = 8;
    let len = 1;

    assert_eq!(
        input_key_host_bytes(capacity, len).expect("key weight"),
        capacity * size_of::<PreparedMetalHtInputKey>() + 2 * size_of::<usize>()
    );
}

#[test]
fn otherwise_identical_prepared_inputs_are_isolated_by_owner_identity() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let limits = HtGpuJobChunkLimits::new(
        NonZeroUsize::new(1).expect("nonzero job cap"),
        1,
        size_of::<J2kHtCleanupBatchJob>(),
    );
    let payload = [0_u8];
    let jobs = [cleanup_job(0)];
    let first_owner = Arc::new(PreparedHtExecutionOwner);
    let second_owner = Arc::new(PreparedHtExecutionOwner);
    let first = [HtBatchInput {
        source_index: 0,
        payload: HtPayloadSource::Contiguous(&payload),
        jobs: &jobs,
        output_base: 0,
        execution_owner: &first_owner,
    }];
    let second = [HtBatchInput {
        execution_owner: &second_owner,
        ..first[0]
    }];

    let mut cache = PreparedMetalHtExecutionCache::new();
    cache
        .get_or_prepare(&device, &first, limits)
        .expect("cache first owner");
    reset_ht_immutable_payload_uploads_for_test();
    reset_ht_immutable_job_uploads_for_test();
    cache
        .get_or_prepare(&device, &second, limits)
        .expect("prepare identical slices under second owner");
    assert_eq!(ht_immutable_payload_uploads_for_test(), 1);
    assert_eq!(ht_immutable_job_uploads_for_test(), 1);

    reset_ht_immutable_payload_uploads_for_test();
    reset_ht_immutable_job_uploads_for_test();
    cache
        .get_or_prepare(&device, &first, limits)
        .expect("reuse first owner after isolated miss");
    assert_eq!(ht_immutable_payload_uploads_for_test(), 0);
    assert_eq!(ht_immutable_job_uploads_for_test(), 0);
}

#[test]
fn poisoned_prepared_execution_cache_reports_poisoned_state() {
    let runtime = MetalRuntime::new().expect("isolated Metal runtime");
    let poison = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = runtime
            .prepared_ht_execution_cache
            .lock()
            .expect("cache starts healthy");
        panic!("poison prepared HT cache for test");
    }));
    assert!(poison.is_err());

    let limits = HtGpuJobChunkLimits::new(
        NonZeroUsize::new(1).expect("nonzero job cap"),
        1,
        size_of::<J2kHtCleanupBatchJob>(),
    );
    let Err(error) = prepared_metal_ht_execution(&runtime, &[], limits) else {
        panic!("poisoned cache must fail")
    };
    assert!(matches!(
        error,
        Error::MetalStatePoisoned {
            state: "J2K Metal prepared HT execution cache"
        }
    ));
}

#[test]
fn oversized_prepared_execution_is_not_cacheable() {
    assert!(fits_empty_cache(
        1,
        PREPARED_PLAN_CACHE_MAX_HOST_BYTES - 1,
        PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES,
    ));
    assert!(!fits_empty_cache(1, PREPARED_PLAN_CACHE_MAX_HOST_BYTES, 0,));
    assert!(!fits_empty_cache(
        0,
        0,
        PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES + 1,
    ));
}

#[test]
fn thousand_prepared_cache_hits_keep_retained_memory_stable() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let limits = HtGpuJobChunkLimits::new(
        NonZeroUsize::new(1).expect("nonzero job cap"),
        1,
        size_of::<J2kHtCleanupBatchJob>(),
    );
    let payload = [0_u8];
    let jobs = [cleanup_job(0)];
    let owner = Arc::new(PreparedHtExecutionOwner);
    let input = [HtBatchInput {
        source_index: 0,
        payload: HtPayloadSource::Contiguous(&payload),
        jobs: &jobs,
        output_base: 0,
        execution_owner: &owner,
    }];
    let mut cache = PreparedMetalHtExecutionCache::new();
    cache
        .get_or_prepare(&device, &input, limits)
        .expect("warm prepared execution cache");
    let retained = (
        cache.entries.len(),
        cache.retained_host_bytes,
        cache.retained_device_bytes,
    );
    reset_ht_immutable_payload_uploads_for_test();
    reset_ht_immutable_job_uploads_for_test();

    for _ in 0..1_000 {
        cache
            .get_or_prepare(&device, &input, limits)
            .expect("reuse prepared execution cache");
    }

    assert_eq!(
        (
            cache.entries.len(),
            cache.retained_host_bytes,
            cache.retained_device_bytes,
        ),
        retained
    );
    assert_eq!(ht_immutable_payload_uploads_for_test(), 0);
    assert_eq!(ht_immutable_job_uploads_for_test(), 0);
}
