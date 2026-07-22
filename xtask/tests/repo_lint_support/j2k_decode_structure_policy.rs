// SPDX-License-Identifier: MIT OR Apache-2.0

//! Responsibility and regression ratchets for facade decode orchestration.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, rust_sources, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn facade_decode_keeps_pixel_layout_conversion_in_explicit_children() {
    let root = read("crates/j2k/src/decode.rs");
    let output = read_source_files(
        repo_root(),
        &[
            "crates/j2k/src/decode/output.rs",
            "crates/j2k/src/decode/output/u8.rs",
            "crates/j2k/src/decode/output/u16.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("facade decode orchestrator", &root)
            .required(&[
                "mod component_handoff;",
                "mod output;",
                "decode_image_into_with_native_context",
                "decode_image_region_into_with_native_context",
                "decode_warnings_for_settings",
            ])
            .forbidden(&[
                "fn write_u8_output(",
                "fn write_u16_output(",
                "fn write_components_u8_output(",
                "fn convert_or_copy_u16(",
                "include!(",
            ]),
        PatternCheck::new("facade decode output modules", &output)
            .required(&[
                "mod u16;",
                "mod u8;",
                "pub(in crate::decode) fn write_u8_output(",
                "pub(in crate::decode) fn write_u16_output(",
                "direct_u8_decode_accepts_exact_rgb_and_gray_layouts",
                "eight_bit_samples_widen_across_the_complete_u16_domain",
                "synthesized_alpha_matches_native_sample_storage",
            ])
            .forbidden(&["use super::*", "include!("]),
    ]);
}

#[test]
fn facade_decode_responsibility_modules_stay_focused() {
    for (relative, max_lines) in [
        ("crates/j2k/src/decode.rs", 220),
        ("crates/j2k/src/decode/output.rs", 25),
        ("crates/j2k/src/decode/output/u8.rs", 375),
        ("crates/j2k/src/decode/output/u16.rs", 260),
    ] {
        let lines = read(relative).lines().count();
        assert!(
            lines < max_lines,
            "{relative} must stay below its {max_lines}-line responsibility ratchet"
        );
    }
}

#[test]
fn owned_batch_tests_are_split_by_responsibility() {
    let root = repo_root();
    let test_root = root.join("crates/j2k/tests/owned_batch");
    let shell = fs::read_to_string(root.join("crates/j2k/tests/owned_batch.rs"))
        .expect("read owned-batch integration-test shell");

    for (module, owned_symbol, max_lines) in [
        ("fixtures", "fn htj2k_gray8_fixture(", 260usize),
        ("oracles", "fn native_request_oracle(", 140usize),
        (
            "payload_plan",
            "fn assert_prepared_ht_payload_ranges_reconstruct_owned_bytes(",
            180usize,
        ),
        (
            "native_types_and_requests",
            "fn prepared_htj2k_gray_and_rgb_support_native_types_and_requests_exactly(",
            150usize,
        ),
    ] {
        assert!(shell.contains(&format!("mod {module};")));
        let path = test_root.join(format!("{module}.rs"));
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(source.contains(owned_symbol));
        assert!(source.lines().count() < max_lines);
    }
    assert!(shell.lines().count() < 40);
    for forbidden in [
        "fn htj2k_gray8_fixture(",
        "fn native_request_oracle(",
        "fn assert_prepared_ht_payload_ranges_reconstruct_owned_bytes(",
    ] {
        assert!(!shell.contains(forbidden));
    }
    let rgba = fs::read_to_string(test_root.join("rgba.rs")).expect("read owned-batch RGBA tests");
    assert!(rgba.lines().count() < 400);
    assert!(
        !rgba.contains("fn prepared_htj2k_gray_and_rgb_support_native_types_and_requests_exactly(")
    );
    for path in rust_sources(&test_root) {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(!source.lines().any(|line| line.trim() == "use super::*;"));
    }
}

#[test]
fn owned_batch_ht_matrix_tests_have_three_explicit_owners() {
    let root = repo_root();
    let test_root = root.join("crates/j2k/tests/owned_batch_fixtures/ht_matrix");
    let shell_relative = "crates/j2k/tests/owned_batch_fixtures/ht_matrix.rs";
    let shell = fs::read_to_string(root.join(shell_relative))
        .unwrap_or_else(|error| panic!("read {shell_relative}: {error}"));

    for (module, owned_symbol, max_lines) in [
        (
            "reversible_native",
            "fn reversible_batch_matrix_preserves_native_gray_and_rgb_samples(",
            100usize,
        ),
        (
            "raw_jph_request_geometry",
            "fn independent_odd_openhtj2k_fixture_supports_roi_and_reduction(",
            175usize,
        ),
        (
            "pass_bucket_parity",
            "fn external_cleanup_magref_and_generated_sigprop_jobs_decode_in_batches(",
            300usize,
        ),
    ] {
        assert!(
            shell.contains(&format!(
                "#[path = \"ht_matrix/{module}.rs\"]\nmod {module};"
            )),
            "{shell_relative} must declare the {module} responsibility owner"
        );
        let relative = format!("crates/j2k/tests/owned_batch_fixtures/ht_matrix/{module}.rs");
        let source = fs::read_to_string(test_root.join(format!("{module}.rs")))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.contains(owned_symbol),
            "{relative} must own {owned_symbol}"
        );
        assert!(
            source.lines().count() < max_lines,
            "{relative} exceeded its {max_lines}-line responsibility ratchet"
        );
        assert!(
            !source.lines().any(|line| line.trim() == "use super::*;"),
            "{relative} must keep explicit imports"
        );
    }

    assert!(
        shell.lines().count() < 15,
        "{shell_relative} must remain a focused module shell"
    );
    for forbidden in [
        "fn reversible_batch_matrix_preserves_native_gray_and_rgb_samples(",
        "fn independent_openhtj2k_raw_and_derived_jph_outputs_are_exact_and_indexed(",
        "fn independent_odd_openhtj2k_fixture_supports_roi_and_reduction(",
        "fn external_cleanup_magref_and_generated_sigprop_jobs_decode_in_batches(",
    ] {
        assert!(
            !shell.contains(forbidden),
            "{shell_relative} must not retain {forbidden}"
        );
    }
}

#[test]
fn cpu_fast_route_preparation_lives_in_focused_children() {
    let shell = read("crates/j2k/src/owned_batch/cpu_fast.rs");
    let ht = read("crates/j2k/src/owned_batch/cpu_fast/ht.rs");
    let classic = read("crates/j2k/src/owned_batch/cpu_fast/classic.rs");

    assert_pattern_checks(&[
        PatternCheck::new("CPU fast workspace lifecycle", &shell)
            .required(&[
                "mod classic;",
                "mod ht;",
                "mod plan;",
                "fn prepare_storage<T>(",
                "fn assign_image_spans(",
                "fn finish_group(",
                "fn record_job_buckets(",
                "fn clear_active_plan(",
            ])
            .forbidden(&["fn prepare_htj2k(", "fn prepare_classic("]),
        PatternCheck::new("CPU fast HT preparation", &ht)
            .required(&["impl CpuGroupFastWorkspace", "fn prepare_htj2k("])
            .forbidden(&["use super::*;", "fn prepare_classic("]),
        PatternCheck::new("CPU fast classic preparation", &classic)
            .required(&["impl CpuGroupFastWorkspace", "fn prepare_classic("])
            .forbidden(&["use super::*;", "fn prepare_htj2k("]),
    ]);

    for (relative, source, max_lines) in [
        ("crates/j2k/src/owned_batch/cpu_fast.rs", &shell, 350usize),
        ("crates/j2k/src/owned_batch/cpu_fast/ht.rs", &ht, 180usize),
        (
            "crates/j2k/src/owned_batch/cpu_fast/classic.rs",
            &classic,
            180usize,
        ),
    ] {
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its {max_lines}-line responsibility ratchet"
        );
    }
}

#[test]
fn cpu_staged_group_orchestrator_delegates_explicit_phases() {
    let source = read("crates/j2k/src/owned_batch/cpu_staged_execute.rs");
    let (_, function_tail) = source
        .split_once("pub(super) fn run_staged_typed_group")
        .expect("find staged group orchestrator");
    let (orchestrator, _) = function_tail
        .split_once("\nfn prepare_staged_window<")
        .expect("isolate staged group orchestrator");

    for (phase, owner) in [
        ("prepare_staged_window(", "fn prepare_staged_window<T"),
        (
            "execute_staged_window_tiles(",
            "fn execute_staged_window_tiles<T",
        ),
        (
            "materialize_staged_window(",
            "fn materialize_staged_window<T",
        ),
    ] {
        assert!(
            orchestrator.contains(phase),
            "orchestrator must call {phase}"
        );
        assert!(
            source.matches(owner).count() == 1,
            "{phase} must have one concrete phase owner"
        );
    }
    for implementation_detail in [
        "prepare_staged_image(",
        "staged_tile_count(",
        "prepare_staged_tile_window(",
        "finish_staged_plan_samples(",
    ] {
        assert!(
            !orchestrator.contains(implementation_detail),
            "orchestrator must not inline {implementation_detail}"
        );
    }
    assert!(orchestrator.lines().count() < 75);
    assert!(!source.contains("clippy::too_many_lines"));
}
