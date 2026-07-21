// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn metal_batch_heuristics_live_in_focused_module() {
    let root = repo_root();
    let batch = fs::read_to_string(root.join("crates/j2k-metal/src/batch.rs"))
        .expect("read j2k-metal batch module");
    let heuristics = fs::read_to_string(root.join("crates/j2k-metal/src/batch/heuristics.rs"))
        .expect("read j2k-metal batch heuristics module");
    let execute = fs::read_to_string(root.join("crates/j2k-metal/src/batch/execute.rs"))
        .expect("read j2k-metal batch execute module");
    let session = fs::read_to_string(root.join("crates/j2k-metal/src/batch/session.rs"))
        .expect("read j2k-metal batch session module");
    let routes = fs::read_to_string(root.join("crates/j2k-metal/src/batch/routes.rs"))
        .expect("read j2k-metal batch routes module");
    let heuristic_consumers = [execute.as_str(), session.as_str(), routes.as_str()].join("\n");

    let heuristic_items = [
        "pub(super) enum BatchRoute",
        "pub(super) struct GroupedRequests",
        "pub(super) fn group_metal_requests",
        "pub(super) fn profile_route_label",
        "pub(super) fn is_region_scaled_direct_batch_candidate",
        "pub(super) fn should_auto_use_metal_for_region_scaled_direct_batch",
        "pub(super) fn can_decode_requests_as_repeated_region_scaled_batch",
    ];
    let heuristic_required = [
        "pub(super) enum BatchRoute",
        "pub(super) struct GroupedRequests",
        "pub(super) fn group_metal_requests",
        "pub(super) fn profile_route_label",
        "pub(super) fn is_region_scaled_direct_batch_candidate",
        "pub(super) fn should_auto_use_metal_for_region_scaled_direct_batch",
        "pub(super) fn can_decode_requests_as_repeated_region_scaled_batch",
        "AUTO_REGION_SCALED_DIRECT_BATCH64_MIN_DIM",
        "REGION_SCALED_DIRECT_FORMATS",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal batch heuristic module shell", &batch)
            .required(&["mod heuristics;"])
            .forbidden(&heuristic_items),
        PatternCheck::new("j2k-metal batch heuristic ownership", &heuristics)
            .required(&heuristic_required),
        PatternCheck::new("j2k-metal batch heuristic consumers", &heuristic_consumers)
            .required(&["use super::heuristics::{", "group_metal_requests"]),
    ]);
}

#[test]
fn metal_batch_cpu_fallback_lives_in_focused_module() {
    let root = repo_root();
    let batch = fs::read_to_string(root.join("crates/j2k-metal/src/batch.rs"))
        .expect("read j2k-metal batch module");
    let cpu = fs::read_to_string(root.join("crates/j2k-metal/src/batch/cpu.rs"))
        .expect("read j2k-metal batch CPU module");

    let cpu_items = [
        "pub(super) fn decode_cpu_host_batch",
        "fn decode_cpu_full_batch",
        "fn decode_cpu_region_scaled_batch",
        "fn checked_cpu_batch_surface",
        "fn cpu_batch_error",
        "fn host_surface",
        "decode_tiles_into",
        "decode_tiles_region_scaled_into",
        "BatchDecodeError::Tile(error)",
        "BatchDecodeError::Infrastructure(error)",
        "BufferError::AllocationTooLarge",
        "BufferError::HostAllocationFailed",
        "Error::BatchInfrastructure(other)",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal batch CPU fallback module shell", &batch)
            .required(&["mod cpu;", "use self::cpu::decode_cpu_host_batch;"])
            .forbidden(&cpu_items),
        PatternCheck::new("j2k-metal batch CPU fallback ownership", &cpu).required(&cpu_items),
    ]);
}

#[test]
fn metal_batch_execute_lives_in_focused_module() {
    let root = repo_root();
    let batch = fs::read_to_string(root.join("crates/j2k-metal/src/batch.rs"))
        .expect("read j2k-metal batch module");
    let execute = fs::read_to_string(root.join("crates/j2k-metal/src/batch/execute.rs"))
        .expect("read j2k-metal batch execute module");
    let session = fs::read_to_string(root.join("crates/j2k-metal/src/batch/session.rs"))
        .expect("read j2k-metal batch session module");

    let execute_items = [
        "pub(super) fn process_batch",
        "fn process_batch_inner",
        "fn complete_cpu_host_fallback",
        "fn complete_batch_surfaces",
        "fn profile_completed_outcome",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal batch execute module shell", &batch)
            .required(&["mod execute;"])
            .forbidden(&execute_items),
        PatternCheck::new("j2k-metal batch execute ownership", &execute).required(&execute_items),
        PatternCheck::new("j2k-metal batch execute consumer", &session).required(&[
            "use super::execute::process_batch;",
            "process_batch(session, batch, backend);",
        ]),
    ]);
    assert_eq!(
        execute
            .matches("session.completed[request.output_slot] = Some(Ok(surface));")
            .count(),
        1,
        "batch execution must use one shared successful-completion block"
    );
}

#[test]
fn metal_batch_facade_keeps_request_session_routes_and_tests_in_focused_modules() {
    let root = repo_root();
    let batch_root = root.join("crates/j2k-metal/src/batch");
    let shell = fs::read_to_string(root.join("crates/j2k-metal/src/batch.rs"))
        .expect("read j2k-metal batch module");
    assert!(
        shell.lines().count() < 100,
        "j2k-metal batch.rs must remain a focused module shell"
    );

    for (module, max_lines, owned_item) in [
        ("request", 150, "pub(crate) enum BatchOp"),
        ("session", 275, "pub struct MetalSubmission"),
        ("routes", 375, "fn decode_repeated_full_grayscale"),
        ("tests", 300, "fn auto_rgb_region_scaled_request"),
    ] {
        assert!(shell.contains(&format!("mod {module};")));
        assert!(
            !shell.contains(owned_item),
            "batch shell must not retain {module} implementation"
        );
        let relative = batch_root.join(format!("{module}.rs"));
        let source = fs::read_to_string(&relative)
            .unwrap_or_else(|error| panic!("read {}: {error}", relative.display()));
        assert!(source.contains(owned_item));
        assert!(
            source.lines().count() < max_lines,
            "{} exceeded its focused line-count ratchet of {max_lines}",
            relative.display()
        );
        assert!(!source.lines().any(|line| line.trim() == "use super::*;"));
    }
}

#[test]
fn metal_batch_routes_share_session_aware_implementations() {
    let root = repo_root();
    let direct_paths =
        fs::read_to_string(root.join("crates/j2k-metal/src/decoder/direct_paths.rs"))
            .expect("read j2k-metal direct paths");
    let hybrid = fs::read_to_string(root.join("crates/j2k-metal/src/hybrid/batch.rs"))
        .expect("read j2k-metal hybrid routes");
    let routes = fs::read_to_string(root.join("crates/j2k-metal/src/batch/routes.rs"))
        .expect("read j2k-metal batch routes");

    assert_pattern_checks(&[
        PatternCheck::new(
            "Metal full-batch shared route implementations",
            &direct_paths,
        )
        .required(&[
            "fn decode_repeated_grayscale_direct_to_device_routed",
            "fn decode_repeated_color_direct_to_device_routed",
            "fn decode_full_grayscale_batch_direct_to_device_routed",
            "fn decode_full_color_batch_direct_to_device_routed",
        ]),
        PatternCheck::new("Metal region-scaled shared route implementations", &hybrid).required(&[
            "fn decode_region_scaled_grayscale_batch_direct_to_device_routed",
            "fn decode_region_scaled_color_batch_direct_to_device_routed",
            "fn decode_repeated_region_scaled_color_batch_direct_to_device_routed",
        ]),
        PatternCheck::new("Metal batch scheduler shared route calls", &routes).required(&[
            "decode_repeated_grayscale_direct_to_device_routed",
            "decode_full_grayscale_batch_direct_to_device_routed",
            "decode_region_scaled_color_batch_direct_to_device_routed",
        ]),
    ]);
    assert_eq!(
        direct_paths
            .matches("full grayscale batch does not support")
            .count(),
        1,
        "full grayscale validation must live in one session-aware route"
    );
    assert_eq!(
        direct_paths
            .matches("full color batch does not support")
            .count(),
        1,
        "full color validation must live in one session-aware route"
    );
    assert_eq!(
        hybrid
            .matches("region-scaled grayscale batch does not support")
            .count(),
        1,
        "region-scaled grayscale validation must live in one session-aware route"
    );
}

#[test]
fn metal_multitile_tests_are_split_by_pixel_contract() {
    let root = repo_root();
    let test_root = root.join("crates/j2k-metal/tests/device/multitile_color");
    let shell = fs::read_to_string(root.join("crates/j2k-metal/tests/device/multitile_color.rs"))
        .expect("read Metal multi-tile test shell");
    for module in ["batch_inputs", "classic", "gray12", "rgb", "signed"] {
        assert!(shell.contains(&format!("mod {module};")));
        assert!(test_root.join(format!("{module}.rs")).exists());
    }
    assert!(shell.lines().count() < 25);
    for symbol in [
        "fn independent_openjph_multitile_gray12_decodes_exactly_on_metal(",
        "fn independent_openjph_multitile_rgb_decodes_exactly_on_metal(",
        "fn classic_multitile_rgb8_decodes_exactly_on_metal(",
    ] {
        assert!(!shell.contains(symbol));
    }
    assert!(fs::read_to_string(test_root.join("gray12.rs"))
        .expect("read Gray12 multi-tile tests")
        .contains("fn independent_openjph_multitile_gray12_decodes_exactly_on_metal("));
    assert!(fs::read_to_string(test_root.join("rgb.rs"))
        .expect("read RGB multi-tile tests")
        .contains("fn independent_openjph_multitile_rgb_decodes_exactly_on_metal("));
}

#[test]
fn hybrid_tests_have_a_focused_external_owner() {
    let root = repo_root();
    let facade = fs::read_to_string(root.join("crates/j2k-metal/src/hybrid.rs"))
        .expect("read Metal hybrid facade");
    let tests = fs::read_to_string(root.join("crates/j2k-metal/src/hybrid/tests.rs"))
        .expect("read Metal hybrid tests");
    assert!(facade.contains("#[cfg(test)]\nmod tests;"));
    assert!(!facade.contains("mod tests {"));
    assert!(!facade.contains("fn explicit_session_region_scaled_plan_builds_use_session_runtime"));
    for symbol in [
        "fn explicit_session_gray_region_scaled_plans_use_session_runtime",
        "fn explicit_session_color_region_scaled_plans_use_session_runtime",
        "fn explicit_session_repeated_color_plan_uses_session_runtime",
    ] {
        assert!(tests.contains(symbol), "hybrid tests must own {symbol}");
    }
    assert!(facade.lines().count() < 725);
    assert!(tests.lines().count() < 350);
    assert!(!tests.lines().any(|line| line.trim() == "use super::*;"));
}

#[test]
fn metal_queue_ordering_tests_are_split_by_contract() {
    let root = repo_root();
    let test_root = root.join("crates/j2k-metal/src/batch_decoder/queue_ordering_tests");
    let shell =
        fs::read_to_string(root.join("crates/j2k-metal/src/batch_decoder/queue_ordering_tests.rs"))
            .expect("read Metal queue-ordering test shell");

    assert!(shell.lines().count() < 25);
    for module in ["fixtures", "exact_queue", "cross_queue", "lifecycle"] {
        assert!(shell.contains(&format!("mod {module};")));
        let source = fs::read_to_string(test_root.join(format!("{module}.rs")))
            .unwrap_or_else(|error| panic!("read Metal queue-ordering {module}: {error}"));
        assert!(source.lines().count() < 250);
        assert!(!source.lines().any(|line| line.trim() == "use super::*;"));
    }
}
