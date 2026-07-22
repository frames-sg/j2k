// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::CodecError;

use super::{cache, prepared_plan_cache_error};
use crate::Error;

#[test]
fn prepared_plan_cache_allocation_keeps_its_source_and_classification() {
    let source = Vec::<u8>::new()
        .try_reserve(usize::MAX)
        .expect_err("capacity overflow must fail before allocation");
    let source_message = source.to_string();
    let error = prepared_plan_cache_error(
        "Metal prepared-plan cache update failed",
        cache::PreparedPlanCacheError::Allocation(source),
    );

    assert!(matches!(
        &error,
        Error::PreparedPlanCacheAllocation {
            context: "Metal prepared-plan cache update failed",
            ..
        }
    ));
    let chained = std::error::Error::source(&error).expect("cache allocation source");
    assert_eq!(chained.to_string(), source_message);
    assert!(!error.is_unsupported());
    assert!(!error.is_buffer_error());
}

#[test]
fn prepared_plan_cache_invariant_keeps_static_reason_without_source() {
    let error = prepared_plan_cache_error(
        "Metal region-scaled prepared-plan cache update failed",
        cache::PreparedPlanCacheError::Invariant("test cache invariant"),
    );

    assert_eq!(
        error.to_string(),
        "Metal kernel error: Metal region-scaled prepared-plan cache update failed: cache invariant failed: test cache invariant"
    );
    assert!(matches!(
        &error,
        Error::PreparedPlanCacheInvariant {
            context: "Metal region-scaled prepared-plan cache update failed",
            reason: "test cache invariant",
        }
    ));
    assert!(std::error::Error::source(&error).is_none());
    assert!(!error.is_unsupported());
    assert!(!error.is_buffer_error());
}
