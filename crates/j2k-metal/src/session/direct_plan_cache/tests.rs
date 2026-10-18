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

#[cfg(target_os = "macos")]
#[test]
fn grayscale_cache_weight_counts_the_codestream_its_payload_ranges_point_into() {
    use super::DirectGrayPlanCacheEntry;
    use cache::PreparedPlanCacheValue as _;
    use std::sync::Arc;

    let pixels = j2k_test_support::gradient_u8(64, 48, 1);
    let options = j2k_native::EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..j2k_native::EncodeOptions::default()
    };
    let bytes: Arc<[u8]> = j2k_native::encode_htj2k(&pixels, 64, 48, 1, 8, false, &options)
        .expect("encode HT grayscale fixture")
        .into();
    let image = j2k_native::Image::new(&bytes, &j2k_native::DecodeSettings::default())
        .expect("parse HT grayscale fixture");
    let mut context = j2k_native::DecoderContext::default();
    let referenced = image
        .build_referenced_htj2k_plan_region_with_context(&mut context, (0, 0, 64, 48))
        .expect("referenced plan");
    let prepared = Arc::new(
        crate::engine::prepare_referenced_htj2k_grayscale_plan(&referenced, &bytes)
            .expect("prepared referenced plan"),
    );
    assert_eq!(prepared.referenced_input_bytes(), Ok(bytes.len()));

    let plan_weight = prepared.retained_cache_weight().expect("plan weight");
    let entry_weight = DirectGrayPlanCacheEntry { prepared }
        .retained_cache_weight()
        .expect("cache entry weight");
    assert_eq!(
        entry_weight.host_bytes,
        plan_weight.host_bytes + bytes.len()
    );
    assert_eq!(entry_weight.device_bytes, plan_weight.device_bytes);
}
