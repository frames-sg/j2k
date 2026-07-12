// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use super::super::super::super::{assert_pattern_checks, PatternCheck};

pub(super) fn assert_policy(root: &Path) {
    let read = |relative: &str| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    };
    let pool = read("crates/j2k-cuda-runtime/src/memory/pool.rs");
    let cache = read("crates/j2k-cuda-runtime/src/memory/pool/cache_policy.rs");
    let tests = read("crates/j2k-cuda-runtime/src/memory/pool/cache_policy/tests.rs");
    let device_tests = read("crates/j2k-cuda-runtime/src/memory/pool/cache_policy/tests/device.rs");
    let reuse_guard = read("crates/j2k-cuda-runtime/src/memory/pool/reuse_guard.rs");
    let buckets = read("crates/j2k-cuda-runtime/src/memory/pool/size_buckets.rs");
    let inventory = read("crates/j2k-cuda-runtime/src/memory/pool/size_buckets/inventory.rs");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA deferred retention safety ledger", &pool)
            .required(&[
                "checked_deferred_bytes(state.deferred_bytes, buffer.byte_len())",
                "state.deferred.try_reserve(1)",
                "state.deferred_bytes = deferred_bytes",
                "observe_deferred_high_water(&mut state)",
                "let deferred = std::mem::take(&mut state.deferred)",
                "state.deferred_bytes = 0",
                "self.recycle_completed_buffer(buffer)?",
            ])
            .forbidden(&["BTreeMap", "recycle_free_buffer"]),
        PatternCheck::new("CUDA completed-buffer bounded cache policy", &cache)
            .required(&[
                "DEFAULT_MAX_CACHED_BYTES",
                "pub struct CudaBufferPoolLimits",
                "pub struct CudaBufferPoolDiagnostics",
                "pub fn diagnostics(&self)",
                "candidate_bytes > limits.max_cached_bytes",
                "inventory.bytes.checked_add(candidate_bytes)",
                "CacheAdmissionDecision::Evict",
                "state.free.evict_deterministic()",
                "Self::FirstFit(buffers) => (!buffers.is_empty()).then(|| buffers.remove(0))",
                "drop(state)",
                "drop(evicted)",
                "std::mem::forget(candidate.take())",
                "buffer.byte_len()",
            ])
            .forbidden(&["BTreeMap", "HashMap"]),
        PatternCheck::new("CUDA cache exact-boundary regressions", &tests).required(&[
            "exact_cache_limits_admit",
            "one_over_each_cache_limit_evicts_deterministically",
            "impossible_candidate_and_disabled_cache_reject",
            "deferred_byte_ledger_is_checked",
        ]),
        PatternCheck::new("CUDA cache device high-water regressions", &device_tests).required(&[
            "first_fit_cache_respects_actual_byte_and_count_high_water",
            "best_fit_cache_evicts_largest_oldest_at_bucket_limit",
            "reuse_hold_accounts_oversize_buffer_until_completion",
            "diagnostics.peak_cached_bytes <= limits.max_cached_bytes",
            "deferred.deferred_bytes, 40",
        ]),
        PatternCheck::new(
            "CUDA deferred guard releases only after completion",
            &reuse_guard,
        )
        .required(&[
            "Err(error @ CudaError::HostAllocationFailed { .. })",
            "if self.release_inner().is_err() && self.active",
            "leave the hold active rather than",
        ]),
        PatternCheck::new("CUDA best-fit cache uses fallible sorted vectors", &buckets)
            .required(&[
                "self.buckets.try_reserve(1)",
                "buffers.try_reserve(1)",
                "binary_search_by_key",
                "partition_point",
                "fn evict_largest_oldest",
                "buffers.remove(0)",
            ])
            .forbidden(&["BTreeMap"]),
        PatternCheck::new("CUDA best-fit actual-byte inventory", &inventory).required(&[
            "fn cached_bytes",
            "bucket.size.saturating_mul(bucket.buffers.len())",
            "fn bucket_count",
            "fn contains_size",
        ]),
    ]);
}
