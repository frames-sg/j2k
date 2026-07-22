// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn exact_native_color_store_separates_planning_from_encoder_dispatch() {
    let root =
        repo_root().join("crates/j2k-metal/src/compute/direct_grayscale_execute/color_destination");
    let dispatch =
        fs::read_to_string(root.join("store.rs")).expect("read exact native color store dispatch");
    let plan =
        fs::read_to_string(root.join("store/plan.rs")).expect("read exact native color store plan");

    assert_pattern_checks(&[
        PatternCheck::new("exact native color store dispatch", &dispatch)
            .required(&[
                "mod plan;",
                "struct NativeColorStoreConfig",
                "plan_exact_native_color_store(",
            ])
            .forbidden(&[
                "validate_stacked_color_destination_indices(",
                "J2kNativeColorBatchStoreParams {",
                "fn native_color_store_pipeline(",
            ]),
        PatternCheck::new("exact native color store planning", &plan).required(&[
            "struct NativeColorStorePlan",
            "fn plan_exact_native_color_store<'a>(",
            "validate_stacked_color_destination_indices(",
            "J2kNativeColorBatchStoreParams {",
            "fn native_color_store_pipeline(",
        ]),
    ]);
    assert!(dispatch.lines().count() < 100);
    assert!(plan.lines().count() < 150);
    assert!(!plan.lines().any(|line| line.trim() == "use super::*;"));
}
