// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{assert_pattern_checks, PatternCheck};
use super::{calls, read};

#[test]
fn metal_float_projection_preflight_counts_both_live_outer_metadata_arrays() {
    let allocation = read("crates/j2k-transcode-metal/src/metal/geometry/allocation.rs");
    let tests = read("crates/j2k-transcode-metal/src/metal/tests.rs");
    let preflight = calls(
        "Metal float projection allocation preflight",
        &allocation,
        "validate_float_projection_allocations",
    );
    preflight.assert_count(
        "Metal output and source/destination metadata element counts",
        "checked_host_element_count",
        3,
    );
    preflight.assert_propagated(
        "Metal float projection host cap propagation",
        &["checked_host_element_count", "checked_host_workspace_bytes"],
    );
    assert_pattern_checks(&[
        PatternCheck::new("Metal float projection transient peak", &allocation).required(&[
            "checked_host_element_count::<ProjectedBands>",
            "checked_host_element_count::<Dwt97TwoDimensional<f64>>",
            "let readback_peak = output_bytes",
            ".saturating_add(source_metadata_bytes)",
            ".saturating_add(METAL_READBACK_CHUNK_BYTES)",
            "let conversion_peak = output_bytes",
            ".saturating_add(destination_metadata_bytes)",
            ".max(conversion_peak)",
            ".max(weight_device_bytes)",
        ]),
        PatternCheck::new("Metal conversion-peak regression", &tests).required(&[
            "projection_conversion_counts_both_live_outer_metadata_arrays",
            "let source_metadata_bytes =",
            "let destination_metadata_bytes =",
            "assert!(conversion_peak > readback_peak);",
            "DEFAULT_MAX_HOST_ALLOCATION_BYTES - conversion_peak + 1",
        ]),
    ]);
}

#[test]
fn metal_float_projection_entry_points_preflight_before_runtime_or_host_growth() {
    let irreversible = read("crates/j2k-transcode-metal/src/metal/irreversible.rs");
    let single = calls(
        "Metal single 9/7 projection",
        &irreversible,
        "dispatch_dct_grid_to_dwt97",
    );
    single.assert_ordered(
        "Metal single 9/7 preflight before weights and runtime",
        &[
            "validate_grid",
            "validate_float_projection_allocations",
            "SparseDwt97WeightRows::for_len",
            "with_runtime",
        ],
    );
    single.assert_propagated(
        "Metal single 9/7 preflight propagation",
        &["validate_float_projection_allocations"],
    );

    let batch = calls(
        "Metal batch 9/7 projection",
        &irreversible,
        "dispatch_dct_grid_to_dwt97_batch",
    );
    batch.assert_ordered(
        "Metal batch 9/7 preflight before runtime",
        &[
            "validate_dwt97_batch_geometry",
            "checked_mul",
            "validate_float_projection_allocations",
            "with_runtime",
        ],
    );
    batch.assert_propagated(
        "Metal batch 9/7 preflight propagation",
        &[
            "validate_dwt97_batch_geometry",
            "validate_float_projection_allocations",
        ],
    );

    let conversion = calls(
        "Metal projected-band conversion",
        &irreversible,
        "dwt97_outputs_from_projected_bands",
    );
    conversion.assert_ordered(
        "Metal projected-band fallible destination metadata",
        &["try_transcode_vec_with_capacity", "push"],
    );
    conversion.assert_propagated(
        "Metal projected-band allocation propagation",
        &["try_transcode_vec_with_capacity"],
    );
    conversion.assert_absent(
        "Metal projected-band conversion",
        &["collect", "Vec::with_capacity", "vec", "to_vec"],
    );
}

#[test]
fn metal_host_allocation_failures_retain_typed_stage_categories() {
    let error = read("crates/j2k-transcode-metal/src/error.rs");
    assert_pattern_checks(&[
        PatternCheck::new("Metal typed allocation stage mapping", &error)
            .required(&[
                "impl From<MetalTranscodeError> for TranscodeStageError",
                "MetalTranscodeError::HostAllocationTooLarge { requested, cap, .. }",
                "Self::MemoryCapExceeded { requested, cap }",
                "MetalTranscodeError::HostAllocationFailed { requested, .. }",
                "Self::HostAllocationFailed { bytes: requested }",
                "host_cap_failure_preserves_typed_stage_error",
                "host_allocator_failure_preserves_typed_stage_error",
                "source: Box<dyn std::error::Error + Send + Sync + 'static>",
                "Self::backend(\"metal\", operation, failure)",
                "Self::backend(\"metal\", operation, source)",
                "Self::DeviceMemoryCapExceeded {",
                "Self::DeviceAllocationFailed {",
                "device_allocation_failures_preserve_stage_resource_categories",
                "fn source(&self) -> Option<&(dyn std::error::Error + 'static)>",
            ])
            .forbidden(&[
                "detail: String",
                "source.to_string()",
                "Self::Backend(format!",
            ]),
    ]);
}

#[test]
fn metal_float_projection_policy_stays_focused() {
    assert!(
        include_str!("float_projection.rs").lines().count() < 150,
        "Metal float projection allocation policy must stay focused"
    );
}
