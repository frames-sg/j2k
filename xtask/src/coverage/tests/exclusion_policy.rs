// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::Path;

use super::super::exclusion_policy::{
    matching_exclusion, validate_exclusion_policy, ExclusionMatcher, COVERAGE_EXCLUSIONS,
};

#[test]
fn metal_shader_composition_is_fully_host_covered() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let path = "crates/j2k-metal/src/engine/shader_source.rs";
    let source = fs::read_to_string(root.join(path)).unwrap();
    let lines = source.lines().collect::<Vec<_>>();
    assert!(source.contains("include_str!(\"../store.metal\")"));
    for line in 1..=lines.len() {
        assert!(
            matching_exclusion(path, line, &lines).unwrap().is_none(),
            "shader composition is host Rust and must remain covered (line {line})"
        );
    }
}

#[test]
fn cuda_simt_exclusion_covers_split_device_modules_only() {
    for root in [
        "crates/j2k-cuda-runtime",
        "crates/j2k-cuda-j2k-engine",
        "crates/j2k-cuda-jpeg-engine",
        "crates/j2k-cuda-transcode-engine",
    ] {
        let split_device_module = format!("{root}/src/cuda_oxide_demo/simt/src/exports.rs");
        assert_eq!(
            matching_exclusion(&split_device_module, 1, &[])
                .unwrap()
                .map(|rule| rule.id),
            Some("cuda-simt-device-rust"),
            "{root}"
        );

        let host_module = format!("{root}/src/cuda_oxide_demo/src/main.rs");
        assert_eq!(
            matching_exclusion(&host_module, 1, &[])
                .unwrap()
                .map(|rule| rule.id),
            Some("cuda-generated-host-scaffold"),
            "{root}"
        );
    }
    for near_miss in [
        "crates/j2k-cuda-j2k-engine/src/not_cuda_oxide/simt/src/exports.rs",
        "crates/j2k-cuda-other-engine/src/cuda_oxide_demo/simt/src/exports.rs",
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_demo/host/src/main.rs",
    ] {
        assert!(matching_exclusion(near_miss, 1, &[]).unwrap().is_none());
    }
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
