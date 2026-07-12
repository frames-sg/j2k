// SPDX-License-Identifier: MIT OR Apache-2.0

//! Standard-plan and post-encode validation ownership ratchets.

use super::{production_source, read};
use crate::repo_lint_support::{assert_pattern_checks, PatternCheck};

#[test]
fn standard_single_tile_plan_stays_fallible_phase_accounted_and_split() {
    let files = [
        ("crates/j2k-native/src/j2c/encode/single_tile/plan.rs", 80),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/plan/validation.rs",
            260,
        ),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/plan/construction.rs",
            200,
        ),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/plan/construction/roi.rs",
            140,
        ),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/plan/construction/sampling.rs",
            80,
        ),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/plan/build.rs",
            180,
        ),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/plan/build/owners.rs",
            140,
        ),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/plan/build/params.rs",
            80,
        ),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/plan/tests.rs",
            130,
        ),
    ];
    for (relative, ceiling) in files {
        let source = read(relative);
        assert!(
            source.lines().count() <= ceiling,
            "{relative} exceeds its focused single-tile plan ceiling of {ceiling} lines"
        );
    }

    let construction =
        production_source("crates/j2k-native/src/j2c/encode/single_tile/plan/construction.rs");
    let sampling = format!(
        "{}\n{}",
        production_source("crates/j2k-native/src/j2c/encode/single_tile/plan/validation.rs"),
        production_source(
            "crates/j2k-native/src/j2c/encode/single_tile/plan/construction/sampling.rs"
        )
    );
    let owners =
        production_source("crates/j2k-native/src/j2c/encode/single_tile/plan/build/owners.rs");
    let roi_construction =
        production_source("crates/j2k-native/src/j2c/encode/single_tile/plan/construction/roi.rs");
    let tests = read("crates/j2k-native/src/j2c/encode/single_tile/plan/tests.rs");
    assert_pattern_checks(&[
        PatternCheck::new("single-tile fallible plan construction", &construction)
            .required(&[
                "try_reserve_exact(count)",
                "host_allocation_failed(what, requested)",
                "self.session.checked_phase",
                "checked_element_bytes::<T>(values.capacity(), what)",
            ])
            .forbidden(&[
                "Vec::with_capacity",
                "collect::<Vec",
                ".to_vec()",
                ".clone()",
                "vec![",
            ]),
        PatternCheck::new("single-tile sampling materialization", &sampling).required(&[
            "validate_component_sampling(options, num_components)",
            "try_component_sampling(options, num_components, session)?",
        ]),
        PatternCheck::new("single-tile nested owner construction", &owners).required(&[
            "try_component_step_sizes",
            "try_component_quantization",
            "try_roi_plans",
            "try_copy_slice",
        ]),
        PatternCheck::new("single-tile scratch-free ROI planning", &roi_construction)
            .required(&[
                "planned_region_count",
                "construction.try_vec(count, \"single-tile ROI region owners\")?",
            ])
            .forbidden(&["region_counts", "try_vec::<usize>"]),
        PatternCheck::new("single-tile exact plan boundary regressions", &tests).required(&[
            "single_tile_plan_construction_accepts_its_exact_measured_cap",
            "single_tile_plan_construction_rejects_one_byte_below_measured_cap",
        ]),
    ]);
}

#[test]
fn ht_self_validation_keeps_encoded_output_in_the_decode_peak() {
    let exact = read("crates/j2k-native/src/j2c/encode/exact.rs");
    let entrypoints = production_source("crates/j2k-native/src/j2c/encode.rs");
    let tests = read("crates/j2k-native/src/j2c/encode/exact/tests.rs");
    assert_pattern_checks(&[
        PatternCheck::new("HT self-validation retained output", &exact)
            .required(&[
                "Image::new_with_retained_baseline(",
                "codestream_capacity",
                "retained_allocation_bytes()",
                "decode_native_with_context_and_retained_baseline",
                "DecodeError::AllocationTooLarge",
                "DecodeError::HostAllocationFailed",
            ])
            .forbidden(&[".decode_native()"]),
        PatternCheck::new("HT self-validation actual output capacity", &entrypoints)
            .required(&["codestream.capacity()"]),
        PatternCheck::new("HT self-validation ownership regressions", &tests).required(&[
            "ht_self_validation_counts_the_retained_codestream_during_decode",
            "ht_self_validation_preserves_host_allocation_category",
        ]),
    ]);
}
