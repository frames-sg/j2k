// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::contains_normalized;

pub(super) fn assert_insert_policy(source: &str) {
    for required in [
        "if self.entry_limit == 0 { return Ok(PreparedPlanCacheInsert::SkippedDisabled); }",
        "if value_weight.device_bytes > self.device_limit { return Ok(PreparedPlanCacheInsert::SkippedOversized); }",
        "return Ok(PreparedPlanCacheInsert::SkippedOversized);",
        "PreparedPlanCacheError::Allocation",
        "PreparedPlanCacheError::Invariant",
    ] {
        assert!(
            contains_normalized(source, required),
            "Metal optional cache policy is missing `{required}`"
        );
    }
    assert!(
        !source.contains("KeyBytesTooLarge"),
        "oversized optional admission must not reject a valid decode"
    );
}

pub(super) fn assert_route_policy(direct_plan_cache: &str, hybrid: &str) {
    for (label, source) in [
        ("session direct-plan cache", direct_plan_cache),
        ("global/session hybrid", hybrid),
    ] {
        assert!(
            contains_normalized(source, ".insert(key, value).map(|_| ()).map_err(|error|"),
            "{label} cache route must treat optional admission outcomes as decode continuation"
        );
    }
    assert!(
        direct_plan_cache.contains("prepared_plan_cache_error"),
        "hard cache failures must preserve ERR-009 typed routing"
    );
    assert!(
        hybrid.contains("disable_dynamic_cpu_tier1_retention"),
        "cached hybrid plans must not grow unaccounted coefficient owners after admission"
    );
}
