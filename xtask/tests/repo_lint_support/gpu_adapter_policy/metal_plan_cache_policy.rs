// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move-only, collision-safe, byte-bounded Metal prepared-plan cache policy.

use super::super::{assert_pattern_checks, contains_normalized, PatternCheck};

mod identity;
mod optional_outcomes;
mod source;
use self::source::{cache_family_sources, owner_graph_sources, read};

#[test]
fn metal_prepared_plan_cache_uses_actual_separate_host_and_device_weights() {
    let sources = cache_family_sources();
    let combined = sources
        .iter()
        .map(|(_, source)| source.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let prepared = read("crates/j2k-metal/src/compute/direct_plan_types/allocation.rs");
    let direct_cache = read("crates/j2k-metal/src/compute/direct_cache.rs");
    let native = read("crates/j2k-native/src/direct_plan/allocation.rs");

    assert_pattern_checks(&[
        PatternCheck::new("Metal prepared-plan cache limits", &combined).required(&[
            "PREPARED_PLAN_CACHE_MAX_HOST_BYTES",
            "PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES",
            "host_limit: usize",
            "device_limit: usize",
            "entry_host_bytes: usize",
            "device_bytes: usize",
            "PreparedPlanCacheWeight",
            "retained_cache_weight",
        ]),
        PatternCheck::new("Metal actual cache allocation accounting", &combined).required(&[
            "input.capacity()",
            "entries.capacity()",
            "size_of::<PreparedPlanCacheEntry<V>>()",
            "try_reserve_exact(self.entry_limit)",
            "metadata_host_bytes",
            "retained_host_bytes",
            "evict_until_fits",
        ]),
        PatternCheck::new("prepared host and Metal-buffer owner accounting", &prepared).required(
            &[
                "steps.capacity()",
                "classic_groups.capacity()",
                "ht_groups.capacity()",
                "coded_data.capacity()",
                "jobs.capacity()",
                "segments.capacity()",
                "members.capacity()",
                "buffer.length()",
                "retained_cache_bytes",
            ],
        ),
        PatternCheck::new("native direct-plan owner accounting", &native).required(&[
            "component_plans.capacity()",
            "plan.steps.capacity()",
            "sub_band.jobs.capacity()",
            "job.data.capacity()",
            "job.segments.capacity()",
            "retained_allocation_bytes",
        ]),
        PatternCheck::new("fallible CPU Tier-1 cache-hit copies", &direct_cache)
            .required(&["drop(state)", "budget.try_vec("])
            .forbidden(&["entry.coefficients.to_vec()"]),
    ]);

    assert!(
        contains_normalized(
            &combined,
            "let owned_key = OwnedPreparedPlanCacheKey::try_from_borrowed(key)?; if !self.ensure_metadata_capacity()?",
        ),
        "owned-key and cache metadata growth must remain fallible"
    );
    assert!(
        contains_normalized(
            &combined,
            "self.evict_until_fits(new_entry_host, value_weight.device_bytes, None)?; let stamp = self.next_access_stamp(); self.entries.push",
        ),
        "weighted eviction must complete before cache ownership is committed"
    );
    optional_outcomes::assert_insert_policy(&combined);
}

#[test]
fn metal_prepared_plan_cache_is_move_only_arc_shared_and_collision_safe() {
    let cache_sources = cache_family_sources();
    let combined = cache_sources
        .iter()
        .map(|(_, source)| source.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let owner_sources = owner_graph_sources();
    let owners = owner_sources
        .iter()
        .map(|(_, source)| source.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let session = read("crates/j2k-metal/src/session.rs");
    let hybrid = read("crates/j2k-metal/src/hybrid.rs");

    identity::assert_lookup_and_eviction_policy(&combined);
    optional_outcomes::assert_route_policy(&session, &hybrid);
    assert_pattern_checks(&[
        PatternCheck::new("Arc-shared native and prepared cache owners", &session).required(&[
            "plan: Arc<J2kDirectGrayscalePlan>",
            "plan: Arc<J2kDirectColorPlan>",
            "prepared: Arc<crate::compute::PreparedDirectGrayscalePlan>",
            "prepared: Arc<crate::compute::PreparedDirectColorPlan>",
            "entry.plan.clone()",
            "entry.prepared.clone()",
        ]),
        PatternCheck::new("flat randomized full-key cache", &combined).required(&[
            "entries: Vec<PreparedPlanCacheEntry<V>>",
            "RandomState::new()",
            "hash_one(key)",
            "entry.digest == digest && entry.key.matches(key)",
            "self.input.as_slice() == key.input",
            "&& self.format == key.format",
            "&& self.roi == key.roi",
            "&& self.scale == key.scale",
            "&& self.kind == key.kind",
        ]),
        PatternCheck::new("cache adversarial regressions", &combined).required(&[
            "cache_owns_identity_bytes_and_hits_share_the_same_owner",
            "forced_digest_collision_does_not_cross_hit_distinct_inputs",
            "host_value_exact_limit_is_cached_and_one_byte_over_is_skipped",
            "device_value_exact_limit_is_cached_and_one_byte_over_is_skipped",
            "replacement_reuses_owned_key_and_evicts_before_committing_larger_value",
            "one_hundred_twenty_eight_entry_limit_evicts_the_oldest_exactly",
            "metadata_reservation_failure_keeps_allocator_source",
        ]),
        PatternCheck::new("session owner-sharing regression", &owners)
            .required(&["repeated_session_hits_share_native_and_prepared_plan_owners"]),
    ]);

    for (relative, source) in owner_sources {
        assert!(
            !source.contains("plan: J2kDirectGrayscalePlan")
                && !source.contains("plan: J2kDirectColorPlan"),
            "{relative} must not reintroduce cache-owned native plan values"
        );
    }
    let native_types = read("crates/j2k-native/src/direct_plan.rs");
    let prepared_types = read("crates/j2k-metal/src/compute/direct_plan_types.rs");
    assert!(
        !native_types.contains("#[derive(Debug, Clone)]"),
        "entropy-owning native direct-plan types must remain move-only"
    );
    assert!(
        !prepared_types.contains("#[derive(Clone)]"),
        "prepared direct-plan owner types must remain move-only"
    );
    assert!(owners.contains("disable_dynamic_cpu_tier1_retention"));
}

#[test]
fn metal_prepared_plan_cache_policy_modules_stay_focused() {
    for (path, maximum) in [
        ("metal_plan_cache_policy.rs", 210),
        ("metal_plan_cache_policy/source.rs", 100),
        ("metal_plan_cache_policy/identity.rs", 80),
        ("metal_plan_cache_policy/optional_outcomes.rs", 100),
    ] {
        let lines = match path {
            "metal_plan_cache_policy.rs" => {
                include_str!("metal_plan_cache_policy.rs").lines().count()
            }
            "metal_plan_cache_policy/source.rs" => {
                include_str!("metal_plan_cache_policy/source.rs")
                    .lines()
                    .count()
            }
            "metal_plan_cache_policy/identity.rs" => {
                include_str!("metal_plan_cache_policy/identity.rs")
                    .lines()
                    .count()
            }
            _ => include_str!("metal_plan_cache_policy/optional_outcomes.rs")
                .lines()
                .count(),
        };
        assert!(lines < maximum, "{path} has grown beyond its focused limit");
    }
}
