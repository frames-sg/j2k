// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::store::metadata_entry_size_for_test;
use super::super::{
    JpegCachedPlanBuildError, JpegFastPacketState, JpegPlanCache, JpegPlanCacheError,
    SharedJpegFastPacket, SharedJpegInput,
};
use super::{rewrite_first_sof_quant_table_selector, unsupported_plan};
use crate::Decoder;
use j2k_test_support::{JPEG_BASELINE_420_16X16, JPEG_GRAYSCALE_8X8};
use std::sync::Arc;

#[test]
fn resolve_hit_reuses_the_authoritative_input_and_packet_owners() {
    let mut cache = JpegPlanCache::with_limits(2, usize::MAX);
    let first = cache
        .resolve(JPEG_BASELINE_420_16X16)
        .expect("resolve first plan");
    let second = cache
        .resolve(JPEG_BASELINE_420_16X16)
        .expect("resolve cached plan");

    assert!(SharedJpegInput::ptr_eq(first.input(), second.input()));
    assert!(SharedJpegFastPacket::ptr_eq(
        first.fast_packet().expect("first ready packet"),
        second.fast_packet().expect("cached ready packet")
    ));
    let diagnostics = cache.diagnostics();
    assert_eq!(diagnostics.entries, 1);
    assert_eq!(diagnostics.misses, 1);
    assert_eq!(diagnostics.hits, 1);
}

#[test]
fn resolve_with_decoder_preserves_caller_bytes_and_reuses_cached_plan_owners() {
    let mut cache = JpegPlanCache::with_limits(2, usize::MAX);
    let (first, first_decoder) = cache
        .resolve_with_decoder_and_external_live(JPEG_BASELINE_420_16X16, 0)
        .expect("resolve first plan and decoder");
    assert_eq!(
        first_decoder.bytes.as_ptr(),
        JPEG_BASELINE_420_16X16.as_ptr()
    );
    assert_ne!(
        first.input().as_bytes().as_ptr(),
        JPEG_BASELINE_420_16X16.as_ptr()
    );
    drop(first_decoder);

    let (second, second_decoder) = cache
        .resolve_with_decoder_and_external_live(JPEG_BASELINE_420_16X16, 0)
        .expect("resolve hit and decoder");
    assert_eq!(
        second_decoder.bytes.as_ptr(),
        JPEG_BASELINE_420_16X16.as_ptr()
    );
    assert!(SharedJpegInput::ptr_eq(first.input(), second.input()));
    assert_eq!(cache.diagnostics().misses, 1);
    assert_eq!(cache.diagnostics().hits, 1);
}

#[test]
fn existing_decoder_resolve_builds_no_replacement_decoder_and_hits_without_reparse() {
    let decoder = Decoder::new(JPEG_BASELINE_420_16X16).expect("construct one decoder");
    let mut cache = JpegPlanCache::with_limits(2, usize::MAX);
    let first = cache
        .resolve_from_decoder_with_external_live(&decoder, 0)
        .expect("resolve from existing decoder");
    let second = cache
        .resolve_from_decoder_with_external_live(&decoder, 0)
        .expect("resolve existing-decoder hit");

    assert_eq!(decoder.bytes.as_ptr(), JPEG_BASELINE_420_16X16.as_ptr());
    assert!(SharedJpegInput::ptr_eq(first.input(), second.input()));
    assert!(SharedJpegFastPacket::ptr_eq(
        first.fast_packet().expect("first packet"),
        second.fast_packet().expect("second packet")
    ));
    assert_eq!(cache.diagnostics().misses, 1);
    assert_eq!(cache.diagnostics().hits, 1);
}

#[test]
fn resolve_shared_preserves_arc_owner_and_full_equality_hits_existing_owner() {
    let owner = Arc::<[u8]>::from(JPEG_BASELINE_420_16X16);
    let shared = SharedJpegInput::try_from_arc(owner).expect("adopt shared fixture");
    let retained_owner = shared.clone();
    let mut cache = JpegPlanCache::with_limits(2, usize::MAX);
    let first = cache
        .resolve_shared(shared)
        .expect("resolve first shared plan");
    assert!(SharedJpegInput::ptr_eq(first.input(), &retained_owner));

    let equal_distinct = SharedJpegInput::try_from_arc(Arc::from(JPEG_BASELINE_420_16X16))
        .expect("adopt equal distinct owner");
    let second = cache
        .resolve_shared(equal_distinct.clone())
        .expect("resolve full-equality shared hit");
    assert!(SharedJpegInput::ptr_eq(first.input(), second.input()));
    assert!(!SharedJpegInput::ptr_eq(second.input(), &equal_distinct));
    assert_eq!(cache.diagnostics().misses, 1);
    assert_eq!(cache.diagnostics().hits, 1);
}

#[test]
fn shared_decoder_resolve_borrows_each_caller_owner_but_reuses_cached_plan_owner() {
    let first_input = SharedJpegInput::try_from_arc(Arc::from(JPEG_BASELINE_420_16X16)).unwrap();
    let mut cache = JpegPlanCache::with_limits(2, usize::MAX);
    let (first, first_decoder) = cache
        .resolve_shared_with_decoder_and_external_live(&first_input, 0)
        .expect("resolve shared miss");
    assert_eq!(
        first_decoder.bytes.as_ptr(),
        first_input.as_bytes().as_ptr()
    );
    assert!(SharedJpegInput::ptr_eq(first.input(), &first_input));
    drop(first_decoder);

    let distinct_input = SharedJpegInput::try_from_arc(Arc::from(JPEG_BASELINE_420_16X16)).unwrap();
    let (second, second_decoder) = cache
        .resolve_shared_with_decoder_and_external_live(&distinct_input, 0)
        .expect("resolve shared hit");
    assert_eq!(
        second_decoder.bytes.as_ptr(),
        distinct_input.as_bytes().as_ptr()
    );
    assert!(SharedJpegInput::ptr_eq(first.input(), second.input()));
    assert!(!SharedJpegInput::ptr_eq(second.input(), &distinct_input));
}

#[test]
fn prepopulated_cache_and_external_baseline_use_one_checked_operation_ledger() {
    let mut cache = JpegPlanCache::with_limits(2, usize::MAX);
    cache.insert(unsupported_plan(b"already retained")).unwrap();
    let before = cache.diagnostics();
    let external = 23;
    let exact_cap = before.retained_bytes + external;

    assert_eq!(
        cache
            .operation_live_bytes_with_cap_for_test(external, exact_cap)
            .expect("exact cache plus external cap"),
        exact_cap
    );
    assert!(matches!(
        cache.operation_live_bytes_with_cap_for_test(external, exact_cap - 1),
        Err(JpegCachedPlanBuildError::Cache(JpegPlanCacheError::Limit {
            requested,
            cap,
            ..
        })) if requested == exact_cap && cap == exact_cap - 1
    ));
    assert_eq!(cache.diagnostics(), before);
}

#[test]
fn resolve_returns_current_plan_when_cache_is_disabled() {
    let mut cache = JpegPlanCache::with_limits(0, usize::MAX);
    let plan = cache
        .resolve(JPEG_GRAYSCALE_8X8)
        .expect("resolve with disabled cache");

    assert!(matches!(
        plan.packet_state(),
        JpegFastPacketState::Unsupported
    ));
    let diagnostics = cache.diagnostics();
    assert_eq!(diagnostics.entries, 0);
    assert_eq!(diagnostics.metadata_capacity_bytes, 0);
    assert_eq!(diagnostics.disabled_rejections, 1);
    assert_eq!(diagnostics.misses, 1);
}

#[test]
fn resolve_returns_current_plan_when_admission_is_one_byte_oversized() {
    let mut probe = JpegPlanCache::with_limits(1, usize::MAX);
    probe
        .resolve(JPEG_BASELINE_420_16X16)
        .expect("probe retained size");
    let exact_retained = probe.diagnostics().retained_bytes;

    let mut cache = JpegPlanCache::with_limits(1, exact_retained - 1);
    let plan = cache
        .resolve(JPEG_BASELINE_420_16X16)
        .expect("oversized admission remains usable");

    assert!(plan.fast_packet().is_some());
    let diagnostics = cache.diagnostics();
    assert_eq!(diagnostics.entries, 0);
    assert!(diagnostics.metadata_capacity_bytes > 0);
    assert_eq!(
        diagnostics.retained_bytes,
        diagnostics.metadata_capacity_bytes
    );
    assert_eq!(diagnostics.oversized_rejections, 1);
    assert_eq!(diagnostics.misses, 1);
}

#[test]
fn resolve_never_caches_decode_or_fast_packet_hard_errors() {
    let mut cache = JpegPlanCache::new();
    assert!(matches!(
        cache.resolve(&[0xff, 0xd8]),
        Err(JpegCachedPlanBuildError::Decode(_))
    ));
    let invalid_table =
        rewrite_first_sof_quant_table_selector(JPEG_BASELINE_420_16X16.to_vec(), u8::MAX);
    assert!(matches!(
        cache.resolve(&invalid_table),
        Err(JpegCachedPlanBuildError::FastPacket(_))
    ));

    let diagnostics = cache.diagnostics();
    assert_eq!(diagnostics.entries, 0);
    assert_eq!(diagnostics.misses, 2);
    assert_eq!(diagnostics.disabled_rejections, 0);
    assert_eq!(diagnostics.oversized_rejections, 0);
}

#[test]
fn resolve_preflights_impossible_metadata_before_allocator_entry() {
    let metadata_bytes = metadata_entry_size_for_test();
    let impossible_entries = (isize::MAX as usize / metadata_bytes) + 1;
    let mut cache = JpegPlanCache::with_limits(impossible_entries, usize::MAX);
    let error = cache
        .resolve(JPEG_GRAYSCALE_8X8)
        .expect_err("metadata preflight must fail");

    assert!(matches!(
        error,
        JpegCachedPlanBuildError::Cache(JpegPlanCacheError::Limit { .. })
    ));
    assert_eq!(cache.diagnostics().entries, 0);
    assert_eq!(cache.diagnostics().metadata_allocation_failures, 0);
}

#[test]
fn resolve_reports_malformed_input_before_copy_limit_failure() {
    let mut cache = JpegPlanCache::new();
    assert!(matches!(
        cache.resolve_with_input_cap_for_test(&[0xff, 0xd8], 0),
        Err(JpegCachedPlanBuildError::Decode(_))
    ));

    assert!(matches!(
        cache.resolve_with_input_cap_for_test(JPEG_GRAYSCALE_8X8, JPEG_GRAYSCALE_8X8.len() - 1,),
        Err(JpegCachedPlanBuildError::Cache(
            JpegPlanCacheError::Limit { .. }
        ))
    ));
    assert_eq!(cache.diagnostics().entries, 0);
}
