// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::repo_root;

#[test]
fn metal_direct_plan_validation_is_split_by_runtime_and_shape_ownership() {
    let root = repo_root();
    let compute = fs::read_to_string(root.join("crates/j2k-metal/src/compute.rs"))
        .expect("read Metal compute module");
    let validation =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_plan_validation.rs"))
            .expect("read direct-plan validation façade");
    let runtime = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/direct_plan_validation/runtime.rs"),
    )
    .expect("read direct-plan runtime validation");
    let shape = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/direct_plan_validation/shape.rs"),
    )
    .expect("read direct-plan shape validation");
    let compute_test_support =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/tests/hybrid_support.rs"))
            .expect("read Metal compute hybrid test support");

    assert!(compute.contains("mod direct_plan_validation;"));
    assert!(!compute.contains("mod direct_plan_support;"));
    assert!(validation.lines().count() < 40);
    assert!(validation.contains("mod runtime;"));
    assert!(validation.contains("mod shape;"));
    assert!(runtime.contains("fn prepared_direct_color_plan_supports_runtime"));
    assert!(runtime.contains("fn classic_prepared_job_supports_runtime"));
    assert!(shape.contains("fn classic_group_shapes_match"));
    assert!(shape.contains("fn store_shapes_match"));
    assert!(compute_test_support.contains("fn prepared_direct_color_tier1_input_count"));
    assert!(!root
        .join("crates/j2k-metal/src/compute/direct_plan_support.rs")
        .exists());
}
