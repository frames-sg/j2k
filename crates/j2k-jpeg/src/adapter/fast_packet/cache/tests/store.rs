// SPDX-License-Identifier: MIT OR Apache-2.0

use std::error::Error as _;

use super::super::store::{metadata_entry_size_for_test, PlanCache};
use super::super::{
    JpegPlanCache, JpegPlanCacheError, JpegPlanCacheInsert, SharedJpegInput,
    DEFAULT_JPEG_PLAN_CACHE_ENTRIES, DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES,
};
use super::{unsupported_plan, ConstantDigestBuilder};

#[test]
fn defaults_are_eight_entries_and_sixty_four_mib() {
    let cache = JpegPlanCache::new();
    let diagnostics = cache.diagnostics();
    assert_eq!(diagnostics.entry_limit, DEFAULT_JPEG_PLAN_CACHE_ENTRIES);
    assert_eq!(
        diagnostics.host_byte_limit,
        DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES
    );
    assert_eq!(DEFAULT_JPEG_PLAN_CACHE_ENTRIES, 8);
    assert_eq!(DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES, 64 * 1024 * 1024);
}

#[test]
fn repeated_full_input_hits_reuse_shared_owners() {
    let plan = unsupported_plan(b"same complete input");
    let retained_input = plan.input().clone();
    let mut cache = JpegPlanCache::with_limits(4, usize::MAX);
    assert_eq!(
        cache.insert(plan).expect("insert plan"),
        JpegPlanCacheInsert::Cached
    );

    for _ in 0..2 {
        let hit = cache.get(b"same complete input").expect("full-input hit");
        assert!(SharedJpegInput::ptr_eq(hit.input(), &retained_input));
    }
    assert!(cache.get(b"same complete Input").is_none());
    assert_eq!(cache.diagnostics().hits, 2);
    assert_eq!(cache.diagnostics().misses, 1);
}

#[test]
fn reused_source_pointer_with_new_bytes_cannot_cross_hit() {
    let mut source = b"first complete bytes".to_vec();
    let source_pointer = source.as_ptr();
    let original = source.clone();
    let mut cache = JpegPlanCache::with_limits(2, usize::MAX);
    cache
        .insert(unsupported_plan(&source))
        .expect("insert copied input");
    source.fill(b'X');

    assert_eq!(source.as_ptr(), source_pointer);
    assert!(cache.get(&source).is_none());
    assert!(cache.get(&original).is_some());
}

#[test]
fn forced_digest_collision_still_requires_full_byte_equality() {
    let mut cache = PlanCache::with_limits_and_digest_builder(4, usize::MAX, ConstantDigestBuilder);
    cache
        .insert(unsupported_plan(b"collision first"))
        .expect("insert first");
    cache
        .insert(unsupported_plan(b"collision second"))
        .expect("insert second");

    assert_eq!(
        cache
            .get(b"collision first")
            .expect("first collision hit")
            .input()
            .as_bytes(),
        b"collision first"
    );
    assert_eq!(
        cache
            .get(b"collision second")
            .expect("second collision hit")
            .input()
            .as_bytes(),
        b"collision second"
    );
    assert!(cache.get(b"collision third").is_none());
}

#[test]
fn exact_retained_limit_is_cached_and_one_byte_over_is_not_retained() {
    let plan = unsupported_plan(b"exact byte limit");
    let mut probe = PlanCache::with_limits_and_digest_builder(1, usize::MAX, ConstantDigestBuilder);
    probe.insert(plan.clone()).expect("probe insertion");
    let exact_limit = probe.diagnostics().retained_bytes;

    let mut exact =
        PlanCache::with_limits_and_digest_builder(1, exact_limit, ConstantDigestBuilder);
    assert_eq!(
        exact.insert(plan.clone()).expect("exact admission"),
        JpegPlanCacheInsert::Cached
    );
    let exact_diagnostics = exact.diagnostics();
    assert_eq!(exact_diagnostics.retained_bytes, exact_limit);
    assert_eq!(exact_diagnostics.peak_bytes, exact_limit);
    assert_eq!(exact_diagnostics.peak_entries, 1);
    assert!(exact_diagnostics.metadata_capacity_bytes > 0);

    let mut over =
        PlanCache::with_limits_and_digest_builder(1, exact_limit - 1, ConstantDigestBuilder);
    assert_eq!(
        over.insert(plan).expect("oversized non-error"),
        JpegPlanCacheInsert::SkippedOversized
    );
    assert_eq!(over.diagnostics().entries, 0);
    assert_eq!(over.diagnostics().metadata_capacity_bytes, 0);
    assert_eq!(over.diagnostics().oversized_rejections, 1);
}

#[test]
fn oversized_admission_does_not_evict_or_replace_existing_content() {
    let retained = unsupported_plan(b"retained");
    let oversized = unsupported_plan(&vec![7_u8; 4096]);
    let mut probe = PlanCache::with_limits_and_digest_builder(2, usize::MAX, ConstantDigestBuilder);
    probe.insert(retained.clone()).expect("probe retained plan");
    let limit = probe.diagnostics().retained_bytes;

    let mut cache = PlanCache::with_limits_and_digest_builder(2, limit, ConstantDigestBuilder);
    cache.insert(retained).expect("insert retained plan");
    let retained_bytes = cache.diagnostics().retained_bytes;
    assert_eq!(
        cache.insert(oversized).expect("oversized non-error"),
        JpegPlanCacheInsert::SkippedOversized
    );
    assert_eq!(cache.diagnostics().entries, 1);
    assert_eq!(cache.diagnostics().retained_bytes, retained_bytes);
    assert_eq!(cache.diagnostics().evictions, 0);
    assert_eq!(cache.diagnostics().oversized_rejections, 1);
    assert!(cache.get(b"retained").is_some());
}

#[test]
fn deterministic_lru_evicts_the_oldest_entry_after_a_hit() {
    let mut cache = JpegPlanCache::with_limits(2, usize::MAX);
    for input in [b"one".as_slice(), b"two".as_slice()] {
        cache.insert(unsupported_plan(input)).expect("fill cache");
    }
    assert!(cache.get(b"one").is_some());
    cache
        .insert(unsupported_plan(b"three"))
        .expect("insert with eviction");

    assert!(cache.get(b"two").is_none());
    assert!(cache.get(b"one").is_some());
    assert!(cache.get(b"three").is_some());
    assert_eq!(cache.diagnostics().evictions, 1);
    assert_eq!(cache.diagnostics().peak_entries, 2);
}

#[test]
fn metadata_reservation_failure_preserves_source_and_diagnostics() {
    let entry_bytes = metadata_entry_size_for_test();
    let impossible_entries = (isize::MAX as usize / entry_bytes) + 1;
    let mut cache = PlanCache::with_limits_and_digest_builder(
        impossible_entries,
        usize::MAX,
        ConstantDigestBuilder,
    );
    let error = cache
        .insert(unsupported_plan(b"metadata failure"))
        .expect_err("impossible metadata reservation");
    let cloned = error.clone();

    assert!(matches!(error, JpegPlanCacheError::Allocation { .. }));
    assert!(matches!(cloned, JpegPlanCacheError::Allocation { .. }));
    assert!(error.source().is_some());
    assert_eq!(cache.diagnostics().metadata_allocation_failures, 1);
    assert_eq!(cache.diagnostics().entries, 0);
    assert_eq!(cache.diagnostics().metadata_capacity_bytes, 0);
}

#[test]
fn miss_metadata_reserve_counts_external_owners_before_mutating_cache() {
    let mut probe = PlanCache::with_limits_and_digest_builder(1, usize::MAX, ConstantDigestBuilder);
    probe
        .prepare_for_miss(0, usize::MAX)
        .expect("probe metadata reserve");
    let metadata_bytes = probe.diagnostics().metadata_capacity_bytes;
    assert!(metadata_bytes > 0);

    let external = 19;
    let exact_cap = external + metadata_bytes;
    let mut exact = PlanCache::with_limits_and_digest_builder(1, usize::MAX, ConstantDigestBuilder);
    exact
        .prepare_for_miss(external, exact_cap)
        .expect("external plus metadata exact cap");
    assert_eq!(exact.diagnostics().metadata_capacity_bytes, metadata_bytes);

    let mut rejected =
        PlanCache::with_limits_and_digest_builder(1, usize::MAX, ConstantDigestBuilder);
    assert!(matches!(
        rejected.prepare_for_miss(external, exact_cap - 1),
        Err(JpegPlanCacheError::Limit { requested, cap, .. })
            if requested == exact_cap && cap == exact_cap - 1
    ));
    let diagnostics = rejected.diagnostics();
    assert_eq!(diagnostics.entries, 0);
    assert_eq!(diagnostics.retained_bytes, 0);
    assert_eq!(diagnostics.metadata_capacity_bytes, 0);
    assert_eq!(diagnostics.metadata_allocation_failures, 0);
}

#[test]
fn disabled_cache_is_a_non_error_without_metadata_allocation() {
    let mut cache = JpegPlanCache::with_limits(0, usize::MAX);
    assert_eq!(
        cache
            .insert(unsupported_plan(b"disabled"))
            .expect("disabled non-error"),
        JpegPlanCacheInsert::SkippedDisabled
    );
    let diagnostics = cache.diagnostics();
    assert_eq!(diagnostics.disabled_rejections, 1);
    assert_eq!(diagnostics.entries, 0);
    assert_eq!(diagnostics.retained_bytes, 0);
    assert_eq!(diagnostics.metadata_capacity_bytes, 0);
}

#[test]
fn same_full_input_replacement_keeps_one_entry_and_uses_new_owner() {
    let first = unsupported_plan(b"replacement");
    let first_owner = first.input().clone();
    let replacement = unsupported_plan(b"replacement");
    let replacement_owner = replacement.input().clone();
    let mut cache = JpegPlanCache::with_limits(2, usize::MAX);
    cache.insert(first).expect("insert original");
    cache.insert(replacement).expect("replace original");

    let hit = cache.get(b"replacement").expect("replacement hit");
    assert!(!SharedJpegInput::ptr_eq(hit.input(), &first_owner));
    assert!(SharedJpegInput::ptr_eq(hit.input(), &replacement_owner));
    assert_eq!(cache.diagnostics().entries, 1);
    assert_eq!(cache.diagnostics().evictions, 0);
}
