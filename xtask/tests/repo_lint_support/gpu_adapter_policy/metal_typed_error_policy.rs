// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed shared-Metal and prepared-plan cache error crossing policy.

use super::super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn metal_support_crossings_keep_typed_sources_and_existing_routing() {
    let root = repo_root();
    let j2k_error = read_source_files(root, &["crates/j2k-metal/src/error.rs"]);
    let jpeg_error = read_source_files(root, &["crates/j2k-jpeg-metal/src/error.rs"]);
    let crossings = read_source_files(
        root,
        &[
            "crates/j2k-metal/src/compute/direct_buffers.rs",
            "crates/j2k-metal/src/compute/direct_surface_pack.rs",
            "crates/j2k-metal/src/compute/runtime.rs",
            "crates/j2k-metal/src/surface.rs",
            "crates/j2k-metal/src/encode/encoded.rs",
            "crates/j2k-metal/src/encode/host_fallback.rs",
            "crates/j2k-metal/src/lib.rs",
            "crates/j2k-jpeg-metal/src/buffers.rs",
            "crates/j2k-jpeg-metal/src/compute.rs",
            "crates/j2k-jpeg-metal/src/compute/tests.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("J2K Metal typed support error", &j2k_error).required(&[
            "MetalSupport {",
            "source: MetalSupportError",
            "fn metal_kernel_support_error(",
            "fn metal_runtime_support_error(",
            "if source.is_unavailable()",
            "Error::MetalUnavailable",
            "| Self::MetalSupport { .. }",
            "metal_support_error_keeps_display_source_and_other_classification",
            "runtime_unavailability_keeps_existing_unsupported_route",
        ]),
        PatternCheck::new("JPEG Metal typed support error", &jpeg_error).required(&[
            "#[derive(Clone, Debug, thiserror::Error)]",
            "MetalSupport {",
            "source: MetalSupportError",
            "fn metal_kernel_support_error(",
            "fn metal_runtime_support_error(",
            "| Self::MetalSupport { .. }",
            "cloned_metal_support_error_keeps_display_source_and_classification",
            "runtime_unavailability_keeps_existing_unsupported_route",
        ]),
        PatternCheck::new("Metal support crossing routes", &crossings)
            .required(&[
                "metal_kernel_support_error(",
                "metal_runtime_support_error(error)",
                "J2K Metal surface buffer is not host-addressable",
                "J2K Metal codestream buffer is not CPU-readable",
                "J2K Metal encode input buffer is not host-visible",
                "Error::UnsupportedMetalRequest",
                "readback_allocation_errors_keep_the_typed_element_count_without_fake_bytes",
            ])
            .forbidden(&[
                "message: error.to_string()",
                "buffer_access_error(context, &error)",
                "buffer_readback_error::<",
                "saturating_mul(size_of::<T>())",
                "Err(error) => Err(Error::MetalKernel {",
                "Err(error) => Err(crate::Error::MetalKernel {",
                "map_err(|error| Error::MetalKernel {",
                "map_err(|error| crate::Error::MetalKernel {",
            ]),
    ]);
}

#[test]
fn prepared_plan_cache_crossings_separate_allocation_and_invariant_failures() {
    let root = repo_root();
    let error = read_source_files(root, &["crates/j2k-metal/src/error.rs"]);
    let routes = read_source_files(
        root,
        &[
            "crates/j2k-metal/src/session.rs",
            "crates/j2k-metal/src/hybrid.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("public prepared-plan cache error categories", &error)
            .required(&[
                "PreparedPlanCacheAllocation {",
                "source: std::collections::TryReserveError",
                "PreparedPlanCacheInvariant {",
                "reason: &'static str",
            ])
            .forbidden(&["source: PreparedPlanCacheError"]),
        PatternCheck::new("typed prepared-plan cache adapters", &routes)
            .required(&[
                "fn prepared_plan_cache_error(",
                "PreparedPlanCacheError::Allocation(source)",
                "Error::PreparedPlanCacheAllocation",
                "PreparedPlanCacheError::Invariant(reason)",
                "Error::PreparedPlanCacheInvariant",
                "Metal prepared-plan cache update failed",
                "Metal region-scaled prepared-plan cache update failed",
                "prepared_plan_cache_allocation_keeps_its_source_and_classification",
                "prepared_plan_cache_invariant_keeps_static_reason_without_source",
            ])
            .forbidden(&[
                "direct_plan_cache_error",
                "region_scaled_color_plan_cache_error",
                "map_err(|error| direct_plan_cache_error(&error))",
                "format!(\"Metal prepared-plan cache update failed: {error}\")",
                "format!(\"Metal region-scaled prepared-plan cache update failed: {error}\")",
            ]),
    ]);
}

#[test]
fn metal_native_encode_crossings_preserve_concrete_sources() {
    let root = repo_root();
    let error = read_source_files(
        root,
        &[
            "crates/j2k-metal/src/error.rs",
            "crates/j2k-metal/src/error/native_source.rs",
        ],
    );
    let crossings = read_source_files(
        root,
        &[
            "crates/j2k-metal/src/compute/resident_tier1/counter_validation/validate.rs",
            "crates/j2k-metal/src/compute/tier1_encode/test_support/ordered_pack.rs",
            "crates/j2k-metal/src/compute/tier1_encode/test_support/split_cpu_pack.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("J2K Metal native encode source variant", &error).required(&[
            "NativeEncode {",
            "operation: &'static str",
            "source: NativeBackendError",
            "fn native_encode_error(",
            "source: NativeBackendError::encode(source)",
            "pub struct NativeBackendError",
            "NativeBackendErrorSource::Encode(EncodeError::Unsupported { .. })",
            "native_encode_crossing_preserves_operation_and_concrete_source",
            "concrete.downcast_ref::<NativeEncodeError>()",
        ]),
        PatternCheck::new("J2K Metal native token-pack crossings", &crossings)
            .required(&[
                "crate::error::native_encode_error(\"classic Tier-1 token pack\"",
                "crate::error::native_encode_error(\"classic Tier-1 ordered-token CPU pack\"",
                "crate::error::native_encode_error(\"classic Tier-1 split-token CPU pack\"",
            ])
            .forbidden(&[
                ".map_err(|source| Error::NativeEncode {",
                "map_err(|message| format!(\"J2K Metal classic Tier-1 token-pack failed",
                "map_err(|message| Error::MetalKernel",
            ]),
    ]);
}
