// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::Path;

use super::super::exclusion_policy::{
    matching_exclusion, validate_exclusion_policy, ExclusionMatcher, COVERAGE_EXCLUSIONS,
};

#[test]
fn metal_raw_shader_span_is_narrower_than_the_host_source_file() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let path = "crates/j2k-metal/src/compute/shader_source.rs";
    let source = fs::read_to_string(root.join(path)).unwrap();
    let lines = source.lines().collect::<Vec<_>>();
    let embedded_shader_line = lines
        .iter()
        .position(|line| line.contains("#include <metal_stdlib>"))
        .expect("embedded shader marker")
        + 1;
    let included_shader_line = lines
        .iter()
        .position(|line| line.contains("include_str!(\"../store.metal\")"))
        .expect("included shader marker")
        + 1;

    assert_eq!(
        matching_exclusion(path, embedded_shader_line, &lines)
            .unwrap()
            .map(|rule| rule.id),
        Some("metal-embedded-shader-body")
    );
    assert!(matching_exclusion(path, included_shader_line, &lines)
        .unwrap()
        .is_none());
}

#[test]
fn cuda_simt_exclusion_covers_split_device_modules_only() {
    let split_device_module =
        "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src/exports.rs";
    assert_eq!(
        matching_exclusion(split_device_module, 1, &[])
            .unwrap()
            .map(|rule| rule.id),
        Some("cuda-simt-device-rust")
    );

    let host_module = "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/src/main.rs";
    assert_eq!(
        matching_exclusion(host_module, 1, &[])
            .unwrap()
            .map(|rule| rule.id),
        Some("cuda-generated-host-scaffold")
    );
    assert!(
        matching_exclusion("crates/j2k-cuda-runtime/src/j2k_encode.rs", 1, &[])
            .unwrap()
            .is_none()
    );
}

#[test]
fn exclusion_policy_maps_every_narrow_rule_to_existing_tests() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    validate_exclusion_policy(&root).unwrap();
    assert!(COVERAGE_EXCLUSIONS
        .iter()
        .all(|rule| !rule.evidence.is_empty()));
    assert!(!COVERAGE_EXCLUSIONS.iter().any(|rule| {
        matches!(
            rule.matcher,
            ExclusionMatcher::WholeFile {
                path: "crates/j2k-cuda/" | "crates/j2k-metal/"
            }
        )
    }));
}
