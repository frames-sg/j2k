// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::contains_normalized;

pub(super) fn assert_lookup_and_eviction_policy(source: &str) {
    assert!(
        contains_normalized(
            source,
            "pub(crate) fn get(&mut self, key: PreparedPlanCacheKey<'_>) -> Option<&V> { let digest = self.digest_builder.hash_one(key); let index = self.find_index(digest, key)?;",
        ),
        "cache lookup must use randomized digest selection before full-key lookup"
    );
    assert!(
        contains_normalized(
            source,
            ".position(|entry| entry.digest == digest && entry.key.matches(key))",
        ),
        "equal digests must still resolve through full-key equality"
    );
    assert!(
        contains_normalized(
            source,
            ".min_by_key(|(index, entry)| (entry.last_used, *index))",
        ),
        "eviction must remain deterministic least-recently-used selection"
    );
    assert!(
        !source.contains("HashMap<u64"),
        "digest buckets must not reintroduce nondeterministic map eviction"
    );
}
