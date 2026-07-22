// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    hash::{BuildHasher, Hasher},
    mem::size_of,
    sync::Arc,
};

use j2k::DecodeRequest;
use j2k_core::{Downscale, PixelFormat, Rect};

use super::*;

#[derive(Clone, Copy, Debug, Default)]
struct ConstantDigestBuilder;

impl BuildHasher for ConstantDigestBuilder {
    type Hasher = ConstantDigest;

    fn build_hasher(&self) -> Self::Hasher {
        ConstantDigest
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ConstantDigest;

impl Hasher for ConstantDigest {
    fn finish(&self) -> u64 {
        7
    }

    fn write(&mut self, _bytes: &[u8]) {}
}

struct TestValue {
    id: usize,
    weight: PreparedPlanCacheWeight,
    owner: Arc<()>,
}

impl PreparedPlanCacheValue for TestValue {
    fn retained_cache_weight(&self) -> Result<PreparedPlanCacheWeight, PreparedPlanCacheError> {
        Ok(self.weight)
    }
}

fn value(id: usize, host_bytes: usize, device_bytes: usize) -> TestValue {
    TestValue {
        id,
        weight: PreparedPlanCacheWeight::new(host_bytes, device_bytes),
        owner: Arc::new(()),
    }
}

fn roi(x: u32) -> Rect {
    Rect {
        x,
        y: 2,
        w: 16,
        h: 12,
    }
}

fn id(value: Option<&TestValue>) -> Option<usize> {
    value.map(|value| value.id)
}

#[test]
fn prepared_image_keys_match_only_the_exact_arc_request_and_format() {
    let bytes = Arc::<[u8]>::from(b"same codestream".as_slice());
    let alias = bytes.clone();
    let distinct_owner = Arc::<[u8]>::from(b"same codestream".as_slice());
    let mut cache = PreparedPlanCache::new(8);
    cache
        .insert(
            PreparedPlanCacheKey::prepared_gray(&bytes, DecodeRequest::Full, PixelFormat::Gray16),
            value(41, 0, 0),
        )
        .expect("insert exact prepared-image identity");

    assert_eq!(
        id(cache.get(PreparedPlanCacheKey::prepared_gray(
            &alias,
            DecodeRequest::Full,
            PixelFormat::Gray16,
        ))),
        Some(41)
    );
    for key in [
        PreparedPlanCacheKey::prepared_gray(
            &distinct_owner,
            DecodeRequest::Full,
            PixelFormat::Gray16,
        ),
        PreparedPlanCacheKey::prepared_gray(
            &bytes,
            DecodeRequest::Reduced {
                scale: Downscale::Half,
            },
            PixelFormat::Gray16,
        ),
        PreparedPlanCacheKey::prepared_gray(&bytes, DecodeRequest::Full, PixelFormat::GrayI16),
        PreparedPlanCacheKey::prepared_color(&bytes, DecodeRequest::Full, PixelFormat::Gray16),
    ] {
        assert_eq!(id(cache.get(key)), None);
    }
}

#[test]
fn prepared_image_arc_bytes_drive_eviction_and_oversized_admission() {
    let first = Arc::<[u8]>::from(vec![1_u8; 32]);
    let second = Arc::<[u8]>::from(vec![2_u8; 32]);
    let first_key =
        PreparedPlanCacheKey::prepared_color(&first, DecodeRequest::Full, PixelFormat::Rgb8);
    let second_key =
        PreparedPlanCacheKey::prepared_color(&second, DecodeRequest::Full, PixelFormat::Rgb8);

    let mut probe = PreparedPlanCache::with_limits_and_digest_builder(
        2,
        usize::MAX,
        usize::MAX,
        ConstantDigestBuilder,
    );
    probe
        .insert(first_key, value(1, 0, 0))
        .expect("probe one prepared Arc entry");
    let one_entry_limit = probe.retained_host_bytes().expect("probe retained bytes");

    let mut bounded = PreparedPlanCache::with_limits_and_digest_builder(
        2,
        one_entry_limit,
        usize::MAX,
        ConstantDigestBuilder,
    );
    assert_eq!(
        bounded.insert(first_key, value(1, 0, 0)).unwrap(),
        PreparedPlanCacheInsert::Cached
    );
    assert_eq!(
        bounded.insert(second_key, value(2, 0, 0)).unwrap(),
        PreparedPlanCacheInsert::Cached
    );
    assert_eq!(id(bounded.get(first_key)), None);
    assert_eq!(id(bounded.get(second_key)), Some(2));
    assert!(bounded.retained_host_bytes().unwrap() <= one_entry_limit);

    let oversized = Arc::<[u8]>::from(vec![3_u8; 33]);
    let oversized_key =
        PreparedPlanCacheKey::prepared_color(&oversized, DecodeRequest::Full, PixelFormat::Rgb8);
    let mut rejecting = PreparedPlanCache::with_limits_and_digest_builder(
        2,
        one_entry_limit,
        usize::MAX,
        ConstantDigestBuilder,
    );
    assert_eq!(
        rejecting
            .insert(oversized_key, value(3, 0, 0))
            .expect("oversized prepared Arc admission"),
        PreparedPlanCacheInsert::SkippedOversized
    );
    assert_eq!(rejecting.len(), 0);
    assert_eq!(Arc::strong_count(&oversized), 1);
}

#[test]
fn forced_digest_collision_does_not_cross_hit_distinct_inputs() {
    let mut cache = PreparedPlanCache::with_digest_builder(4, ConstantDigestBuilder);
    cache
        .insert(
            PreparedPlanCacheKey::direct_color(b"first codestream", PixelFormat::Rgb8),
            value(11, 0, 0),
        )
        .expect("insert first cache entry");
    cache
        .insert(
            PreparedPlanCacheKey::direct_color(b"second codestream", PixelFormat::Rgb8),
            value(17, 0, 0),
        )
        .expect("insert colliding second cache entry");

    assert_eq!(
        id(cache.get(PreparedPlanCacheKey::direct_color(
            b"first codestream",
            PixelFormat::Rgb8,
        ))),
        Some(11)
    );
    assert_eq!(
        id(cache.get(PreparedPlanCacheKey::direct_color(
            b"second codestream",
            PixelFormat::Rgb8,
        ))),
        Some(17)
    );
}

#[test]
fn forced_digest_collision_does_not_cross_hit_semantic_dimensions() {
    let mut cache = PreparedPlanCache::with_digest_builder(8, ConstantDigestBuilder);
    let input = b"shared codestream";
    let first = PreparedPlanCacheKey::region_scaled_color(
        input,
        PixelFormat::Rgb8,
        roi(1),
        Downscale::Half,
    );
    cache
        .insert(first, value(23, 0, 0))
        .expect("insert region-scaled cache entry");

    let distinct = PreparedPlanCacheKey::region_scaled_color(
        input,
        PixelFormat::Rgb8,
        roi(9),
        Downscale::Half,
    );
    assert_eq!(id(cache.get(distinct)), None);
    cache
        .insert(distinct, value(29, 0, 0))
        .expect("insert distinct ROI");
    assert_eq!(id(cache.get(first)), Some(23));

    for key in [
        PreparedPlanCacheKey::region_scaled_color(
            input,
            PixelFormat::Rgb8,
            roi(1),
            Downscale::Quarter,
        ),
        PreparedPlanCacheKey::region_scaled_color(
            input,
            PixelFormat::Rgba8,
            roi(1),
            Downscale::Half,
        ),
        PreparedPlanCacheKey::direct_color(input, PixelFormat::Rgb8),
    ] {
        assert_eq!(id(cache.get(key)), None);
    }
}

#[test]
fn cache_owns_identity_bytes_and_hits_share_the_same_owner() {
    let mut source = b"owned codestream".to_vec();
    let mut cache = PreparedPlanCache::new(4);
    let inserted = value(31, 0, 0);
    let owner = inserted.owner.clone();
    cache
        .insert(
            PreparedPlanCacheKey::direct_gray(&source, PixelFormat::Gray8),
            inserted,
        )
        .expect("insert cache entry");
    source[0] = b'X';

    let hit = cache
        .get(PreparedPlanCacheKey::direct_gray(
            b"owned codestream",
            PixelFormat::Gray8,
        ))
        .expect("owned key hit");
    assert!(Arc::ptr_eq(&owner, &hit.owner));
    assert_eq!(
        id(cache.get(PreparedPlanCacheKey::direct_gray(
            &source,
            PixelFormat::Gray8,
        ))),
        None
    );
}

#[test]
fn deterministic_lru_eviction_honors_hits() {
    let mut cache = PreparedPlanCache::new(2);
    for (input, id) in [(b"one".as_slice(), 1), (b"two".as_slice(), 2)] {
        cache
            .insert(
                PreparedPlanCacheKey::direct_color(input, PixelFormat::Rgb8),
                value(id, 0, 0),
            )
            .expect("fill cache");
    }
    assert_eq!(
        id(cache.get(PreparedPlanCacheKey::direct_color(
            b"one",
            PixelFormat::Rgb8,
        ))),
        Some(1)
    );
    cache
        .insert(
            PreparedPlanCacheKey::direct_color(b"three", PixelFormat::Rgb8),
            value(3, 0, 0),
        )
        .expect("insert with eviction");

    assert_eq!(
        id(cache.get(PreparedPlanCacheKey::direct_color(
            b"two",
            PixelFormat::Rgb8,
        ))),
        None
    );
    assert_eq!(
        id(cache.get(PreparedPlanCacheKey::direct_color(
            b"one",
            PixelFormat::Rgb8,
        ))),
        Some(1)
    );
}

#[test]
fn one_hundred_twenty_eight_entry_limit_evicts_the_oldest_exactly() {
    let mut cache = PreparedPlanCache::new(128);
    let inputs = (0_u16..=128).map(u16::to_le_bytes).collect::<Vec<_>>();
    for (index, input) in inputs.iter().take(128).enumerate() {
        cache
            .insert(
                PreparedPlanCacheKey::direct_color(input, PixelFormat::Rgb8),
                value(index, 0, 0),
            )
            .expect("fill 128-entry cache");
    }
    cache
        .insert(
            PreparedPlanCacheKey::direct_color(&inputs[128], PixelFormat::Rgb8),
            value(128, 0, 0),
        )
        .expect("insert 129th entry");

    assert_eq!(cache.len(), 128);
    assert_eq!(
        id(cache.get(PreparedPlanCacheKey::direct_color(
            &inputs[0],
            PixelFormat::Rgb8,
        ))),
        None
    );
    assert_eq!(
        id(cache.get(PreparedPlanCacheKey::direct_color(
            &inputs[128],
            PixelFormat::Rgb8,
        ))),
        Some(128)
    );
}

#[test]
fn disabled_cache_skips_insertion_without_allocating_metadata() {
    let mut cache = PreparedPlanCache::new(0);
    let outcome = cache
        .insert(
            PreparedPlanCacheKey::direct_color(b"uncached", PixelFormat::Rgb8),
            value(1, 0, 0),
        )
        .expect("disabled optional cache");

    assert_eq!(outcome, PreparedPlanCacheInsert::SkippedDisabled);
    assert_eq!(cache.entries.capacity(), 0);
    assert_eq!(cache.len(), 0);
}

#[test]
fn actual_owned_key_capacity_not_logical_length_is_the_weight() {
    let key = OwnedPreparedPlanCacheKey::with_input_capacity_for_test(64, b"key");
    assert_eq!(key.input_capacity(), 64);
    assert!(key.input_capacity() > b"key".len());
}

#[test]
fn host_value_exact_limit_is_cached_and_one_byte_over_is_skipped() {
    let key = PreparedPlanCacheKey::direct_color(b"key", PixelFormat::Rgb8);
    let mut probe = PreparedPlanCache::with_limits_and_digest_builder(
        1,
        usize::MAX,
        usize::MAX,
        ConstantDigestBuilder,
    );
    probe.insert(key, value(1, 0, 0)).expect("probe insert");
    let baseline = probe.retained_host_bytes().expect("probe retained bytes");

    let mut exact = PreparedPlanCache::with_limits_and_digest_builder(
        1,
        baseline + 7,
        usize::MAX,
        ConstantDigestBuilder,
    );
    assert_eq!(
        exact.insert(key, value(7, 7, 0)).expect("exact insert"),
        PreparedPlanCacheInsert::Cached
    );
    assert_eq!(exact.retained_host_bytes().unwrap(), baseline + 7);

    let mut over = PreparedPlanCache::with_limits_and_digest_builder(
        1,
        baseline + 6,
        usize::MAX,
        ConstantDigestBuilder,
    );
    assert_eq!(
        over.insert(key, value(8, 7, 0)).expect("oversize skip"),
        PreparedPlanCacheInsert::SkippedOversized
    );
    assert_eq!(over.len(), 0);
}

#[test]
fn device_value_exact_limit_is_cached_and_one_byte_over_is_skipped() {
    let key = PreparedPlanCacheKey::direct_color(b"device", PixelFormat::Rgb8);
    let mut exact =
        PreparedPlanCache::with_limits_and_digest_builder(1, usize::MAX, 9, ConstantDigestBuilder);
    assert_eq!(
        exact
            .insert(key, value(9, 0, 9))
            .expect("exact device insert"),
        PreparedPlanCacheInsert::Cached
    );
    assert_eq!(exact.retained_device_bytes(), 9);

    let mut over =
        PreparedPlanCache::with_limits_and_digest_builder(1, usize::MAX, 8, ConstantDigestBuilder);
    assert_eq!(
        over.insert(key, value(10, 0, 9))
            .expect("device oversize skip"),
        PreparedPlanCacheInsert::SkippedOversized
    );
    assert_eq!(over.len(), 0);
}

#[test]
fn replacement_reuses_owned_key_and_evicts_before_committing_larger_value() {
    let first = PreparedPlanCacheKey::direct_color(b"first", PixelFormat::Rgb8);
    let second = PreparedPlanCacheKey::direct_color(b"second", PixelFormat::Rgb8);
    let mut probe = PreparedPlanCache::with_limits_and_digest_builder(
        2,
        usize::MAX,
        usize::MAX,
        ConstantDigestBuilder,
    );
    probe.insert(first, value(1, 10, 0)).unwrap();
    probe.insert(second, value(2, 10, 0)).unwrap();
    let limit = probe.retained_host_bytes().unwrap();

    let mut cache = PreparedPlanCache::with_limits_and_digest_builder(
        2,
        limit,
        usize::MAX,
        ConstantDigestBuilder,
    );
    cache.insert(first, value(1, 10, 0)).unwrap();
    cache.insert(second, value(2, 10, 0)).unwrap();
    assert_eq!(
        cache.insert(first, value(3, 15, 0)).unwrap(),
        PreparedPlanCacheInsert::Cached
    );
    assert_eq!(cache.len(), 1);
    assert_eq!(id(cache.get(first)), Some(3));
    assert_eq!(id(cache.get(second)), None);
    assert!(cache.retained_host_bytes().unwrap() <= limit);
}

#[test]
fn oversized_admission_does_not_mutate_existing_entries() {
    let mut cache =
        PreparedPlanCache::with_limits_and_digest_builder(2, usize::MAX, 5, ConstantDigestBuilder);
    let retained = PreparedPlanCacheKey::direct_color(b"retained", PixelFormat::Rgb8);
    cache.insert(retained, value(1, 0, 5)).unwrap();
    assert_eq!(
        cache
            .insert(
                PreparedPlanCacheKey::direct_color(b"oversized", PixelFormat::Rgb8),
                value(2, 0, 6),
            )
            .unwrap(),
        PreparedPlanCacheInsert::SkippedOversized
    );
    assert_eq!(cache.len(), 1);
    assert_eq!(id(cache.get(retained)), Some(1));
}

#[test]
fn metadata_reservation_failure_keeps_allocator_source() {
    let entry_size = size_of::<PreparedPlanCacheEntry<TestValue>>();
    let impossible_entries = (isize::MAX as usize / entry_size) + 1;
    let mut cache = PreparedPlanCache::with_limits_and_digest_builder(
        impossible_entries,
        usize::MAX,
        usize::MAX,
        ConstantDigestBuilder,
    );
    let error = cache
        .insert(
            PreparedPlanCacheKey::direct_color(b"key", PixelFormat::Rgb8),
            value(1, 0, 0),
        )
        .expect_err("impossible metadata allocation");
    assert!(matches!(error, PreparedPlanCacheError::Allocation(_)));
    assert!(std::error::Error::source(&error).is_none());
}
