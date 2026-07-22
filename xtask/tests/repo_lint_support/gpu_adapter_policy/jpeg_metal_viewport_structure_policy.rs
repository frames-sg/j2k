// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

struct ViewportSources {
    viewport: String,
    cpu: String,
    model: String,
    policy: String,
    resident_policy: String,
    resident: String,
    tests: String,
    budget_tests: String,
}

impl ViewportSources {
    fn read() -> Self {
        let root = repo_root();
        let read = |relative: &str| {
            fs::read_to_string(root.join(relative))
                .unwrap_or_else(|error| panic!("read {relative}: {error}"))
        };
        Self {
            viewport: read("crates/j2k-jpeg-metal/src/viewport.rs"),
            cpu: read("crates/j2k-jpeg-metal/src/viewport/cpu.rs"),
            model: read("crates/j2k-jpeg-metal/src/viewport/model.rs"),
            policy: read("crates/j2k-jpeg-metal/src/viewport/policy.rs"),
            resident_policy: read("crates/j2k-jpeg-metal/src/viewport/policy/resident.rs"),
            resident: read("crates/j2k-jpeg-metal/src/viewport/resident.rs"),
            tests: read("crates/j2k-jpeg-metal/src/viewport/tests.rs"),
            budget_tests: read("crates/j2k-jpeg-metal/src/viewport/tests/budget.rs"),
        }
    }
}

#[test]
fn jpeg_metal_viewport_layers_live_in_focused_modules() {
    let ViewportSources {
        viewport,
        cpu,
        model,
        policy,
        resident_policy,
        resident,
        tests,
        budget_tests,
    } = ViewportSources::read();
    assert_pattern_checks(&[
        PatternCheck::new("JPEG Metal viewport policy facade", &viewport)
            .required(&[
                "mod cpu;",
                "mod model;",
                "mod policy;",
                "mod resident;",
                "pub use self::model::{",
                "pub use self::policy::{choose_viewport_surface_strategy, ViewportSurfaceStrategy};",
                "pub(crate) use self::resident::{",
                "use self::policy::choose_viewport_surface_strategy_for_decoder;",
            ])
            .forbidden(&[
                "pub enum ViewportSurfaceStrategy",
                "pub struct ViewportWorkload",
                "pub fn is_contiguous_viewport_workload(",
                "pub fn suggest_viewport_workload(",
                "fn decode_viewport_to_resizable_metal_buffer_with_session(",
                "fn validate_explicit_metal_viewport_request_with_packets(",
                "fn validate_resident_viewport_composition_request(",
            ]),
        PatternCheck::new("JPEG Metal viewport CPU owner", &cpu)
            .required(&[
                "fn cpu_viewport_allocation_budget_with_cap(",
                "fn compose_viewport_cpu_with_metadata_capacity(",
                "fn decode_viewport_region_cpu(",
                "fn decode_viewport_region_cpu_to_surface(",
                "fn compose_viewport_cpu_to_surface(",
                "fn blit_into_viewport(",
            ])
            .forbidden(&[
                "include!(",
                "use super::*;",
                concat!("#!", "[allow("),
                "#[cfg(target_os = \"macos\")]\n/// Decode the contiguous source region on CPU",
                "#[cfg(not(target_os = \"macos\"))]\n/// Decode the contiguous source region on CPU",
            ]),
        PatternCheck::new("JPEG Metal viewport model owner", &model)
            .required(&[
                "pub struct ViewportTile",
                "pub struct ViewportWorkload",
                "pub fn viewport_source_bounds(",
                "pub fn is_contiguous_viewport_workload(",
                "pub fn suggest_viewport_workload(",
            ])
            .forbidden(&["include!(", "use super::*;", concat!("#!", "[allow(")]),
        PatternCheck::new("JPEG Metal viewport policy owner", &policy)
            .required(&[
                "pub enum ViewportSurfaceStrategy",
                "pub fn choose_viewport_surface_strategy(",
                "fn choose_viewport_surface_strategy_for_decoder(",
                "fn validate_explicit_metal_viewport_request_with_packets(",
                "mod resident;",
                "pub(super) use resident::{",
            ])
            .forbidden(&[
                "include!(",
                "use super::*;",
                concat!("#!", "[allow("),
                "fn validate_resident_viewport_composition_request(",
            ]),
        PatternCheck::new("JPEG Metal viewport resident policy owner", &resident_policy)
            .required(&[
                "fn validate_resident_viewport_composition_request(",
                "fn choose_resizable_metal_viewport_strategy(",
                "fn choose_resizable_metal_viewport_strategy_for_decoder(",
            ])
            .forbidden(&["include!(", "use super::*;", concat!("#!", "[allow(")]),
        PatternCheck::new("JPEG Metal viewport resident adapter owner", &resident)
            .required(&[
                "fn compose_viewport_to_resizable_metal_buffer_with_session(",
                "fn decode_viewport_to_resizable_metal_buffer_with_decoder_session(",
                "fn decode_viewport_region_to_resizable_metal_textures_with_session(",
            ])
            .forbidden(&["include!(", "use super::*;", concat!("#!", "[allow(")]),
    ]);
    assert_viewport_test_contract(&tests, &budget_tests);
    assert_viewport_caps(&[
        ("facade", viewport.as_str(), 225),
        ("CPU owner", cpu.as_str(), 300),
        ("model", model.as_str(), 200),
        ("policy", policy.as_str(), 300),
        ("resident policy", resident_policy.as_str(), 150),
        ("resident adapters", resident.as_str(), 375),
        ("focused tests", tests.as_str(), 150),
        ("budget tests", budget_tests.as_str(), 75),
    ]);
}

fn assert_viewport_test_contract(tests: &str, budget_tests: &str) {
    let focused_tests = format!("{tests}\n{budget_tests}");
    assert_pattern_checks(&[PatternCheck::new(
        "JPEG Metal viewport focused tests",
        &focused_tests,
    )
    .required(&[
        "fn cpu_viewport_live_budget_honors_exact_cap_and_one_byte_over()",
        "fn auto_strategy_prefers_hybrid_for_restart_coded_contiguous_workloads()",
        "fn viewport_direct_packet_detection_includes_fast422()",
    ])]);
}

fn assert_viewport_caps(sources: &[(&str, &str, usize)]) {
    for &(name, source, cap) in sources {
        assert!(
            source.lines().count() < cap,
            "JPEG Metal viewport {name} must remain below its {cap}-line ratchet"
        );
    }
}
