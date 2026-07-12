// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

mod device;

const LIMITS: CudaBufferPoolLimits = CudaBufferPoolLimits {
    max_cached_bytes: 1_024,
    max_cached_buffers: 2,
    max_size_buckets: 2,
};

#[test]
fn exact_cache_limits_admit() {
    let inventory = CacheInventory {
        buffers: 1,
        bytes: 512,
        size_buckets: 1,
    };
    assert_eq!(
        cache_admission_decision(LIMITS, inventory, 512, true),
        CacheAdmissionDecision::Admit
    );
}

#[test]
fn one_over_each_cache_limit_evicts_deterministically() {
    let one_buffer = CacheInventory {
        buffers: 1,
        bytes: 512,
        size_buckets: 1,
    };
    assert_eq!(
        cache_admission_decision(LIMITS, one_buffer, 513, false),
        CacheAdmissionDecision::Evict
    );
    let two_buffers = CacheInventory {
        buffers: 2,
        bytes: 512,
        size_buckets: 1,
    };
    assert_eq!(
        cache_admission_decision(LIMITS, two_buffers, 1, false),
        CacheAdmissionDecision::Evict
    );
    let two_buckets = CacheInventory {
        buffers: 2,
        bytes: 512,
        size_buckets: 2,
    };
    assert_eq!(
        cache_admission_decision(LIMITS, two_buckets, 1, true),
        CacheAdmissionDecision::Evict
    );
}

#[test]
fn impossible_candidate_and_disabled_cache_reject() {
    assert_eq!(
        cache_admission_decision(
            LIMITS,
            CacheInventory {
                buffers: 1,
                bytes: 1,
                size_buckets: 1,
            },
            1_025,
            false,
        ),
        CacheAdmissionDecision::Reject
    );
    assert_eq!(
        cache_admission_decision(
            CudaBufferPoolLimits {
                max_cached_bytes: 0,
                max_cached_buffers: 0,
                max_size_buckets: 0,
            },
            CacheInventory {
                buffers: 0,
                bytes: 0,
                size_buckets: 0,
            },
            0,
            false,
        ),
        CacheAdmissionDecision::Reject
    );
}

#[test]
fn deferred_byte_ledger_is_checked() {
    assert_eq!(checked_deferred_bytes(512, 512).expect("exact sum"), 1_024);
    assert!(checked_deferred_bytes(usize::MAX, 1).is_err());
}
