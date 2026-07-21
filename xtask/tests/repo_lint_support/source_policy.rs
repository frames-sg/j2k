// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{ffi::OsStr, fs};

use super::{
    assert_file_pattern_checks, assert_rust_source_scan_checks, repo_root, rust_sources,
    FilePatternCheck, RustSourceScanCheck,
};

#[test]
fn batch_work_keeps_referenced_plans_and_boundary_tests_in_focused_modules() {
    let root = repo_root();
    let direct_plan = fs::read_to_string(root.join("crates/j2k-native/src/direct_plan.rs"))
        .expect("read j2k-native direct plan");
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-native/src/direct_plan.rs")
                .required(&["mod referenced_ht;", "pub use referenced_ht::"])
                .forbidden(&["pub enum J2kReferencedHtj2kPlan"]),
            FilePatternCheck::new("crates/j2k-native/src/direct_plan/referenced_ht.rs")
                .required(&["pub enum J2kReferencedHtj2kPlan"])
                .forbidden(&["use super::*;"]),
        ],
    );
    assert!(
        direct_plan.lines().count() < 250,
        "direct_plan.rs must remain a focused public geometry owner"
    );

    for (owner, module_decl, child, symbol, max_lines) in [
        (
            "crates/j2k-native/src/tests.rs",
            "mod workspace_reuse;",
            "crates/j2k-native/src/tests/workspace_reuse.rs",
            "fn decoder_workspace_reuses_component_owners_across_distinct_input_lifetimes",
            300usize,
        ),
        (
            "crates/j2k-cuda-runtime/src/tests.rs",
            "mod context_diagnostics;",
            "crates/j2k-cuda-runtime/src/tests/context_diagnostics.rs",
            "fn runtime_diagnostics_count_device_to_host_transfers_when_required",
            100usize,
        ),
        (
            "crates/j2k-cuda-runtime/src/tests/pipeline.rs",
            "mod native_store;",
            "crates/j2k-cuda-runtime/src/tests/pipeline/native_store.rs",
            "fn j2k_native_grayscale_batch_store_preserves_unsigned_and_signed_samples_when_runtime_required",
            220usize,
        ),
        (
            "crates/j2k-metal/src/compute/tests.rs",
            "mod referenced_plan;",
            "crates/j2k-metal/src/compute/tests/referenced_plan.rs",
            "fn referenced_htj2k_payload_ranges_reconstruct_owned_direct_plan_bytes",
            100usize,
        ),
        (
            "crates/j2k-cuda/src/session.rs",
            "mod tests;",
            "crates/j2k-cuda/src/session/tests.rs",
            "fn uninitialized_decode_pool_diagnostics_are_empty",
            120usize,
        ),
    ] {
        let owner_source = fs::read_to_string(root.join(owner))
            .unwrap_or_else(|error| panic!("read {owner}: {error}"));
        let child_source = fs::read_to_string(root.join(child))
            .unwrap_or_else(|error| panic!("read {child}: {error}"));
        assert!(owner_source.contains(module_decl), "{owner} must declare {child}");
        assert!(!owner_source.contains(symbol), "{owner} must not retain {symbol}");
        assert!(child_source.contains(symbol), "{child} must own {symbol}");
        assert!(
            child_source.lines().count() < max_lines,
            "{child} exceeded its focused {max_lines}-line limit"
        );
        assert!(!child_source.lines().any(|line| line.trim() == "use super::*;"));
    }
}

#[test]
fn metal_batch_policy_checks_live_in_their_focused_child_module() {
    let root = repo_root();
    let owner_path = root.join("xtask/tests/repo_lint_support/gpu_adapter_policy.rs");
    let child_path = root
        .join("xtask/tests/repo_lint_support/gpu_adapter_policy/metal_batch_structure_policy.rs");
    let owner = fs::read_to_string(&owner_path).expect("read GPU adapter policy root");
    let child = fs::read_to_string(&child_path).expect("read Metal batch structure policy");

    assert!(owner.contains("mod metal_batch_structure_policy;"));
    assert!(!owner.contains("fn metal_batch_heuristics_live_in_focused_module("));
    assert!(!owner.contains("fn metal_batch_routes_share_session_aware_implementations("));
    assert!(child.contains("fn metal_batch_heuristics_live_in_focused_module("));
    assert!(child.contains("fn metal_batch_routes_share_session_aware_implementations("));
    assert!(
        owner.lines().count() < 1_550,
        "GPU adapter policy root must not absorb focused Metal batch checks"
    );
    assert!(child.lines().count() < 300);
}

#[test]
fn metal_batch_execution_policy_checks_live_in_their_focused_child_module() {
    let root = repo_root();
    let owner_path = root.join("xtask/tests/repo_lint_support/metal_compute_structure_policy.rs");
    let child_path = root
        .join("xtask/tests/repo_lint_support/metal_compute_structure_policy/batch_execution.rs");
    let owner = fs::read_to_string(&owner_path).expect("read Metal compute policy root");
    let child = fs::read_to_string(&child_path).expect("read Metal batch execution policy");

    assert!(owner.contains("mod batch_execution;"));
    assert!(
        !owner.contains("fn metal_ht_chunk_tests_are_split_by_planning_cache_and_status_behavior(")
    );
    assert!(
        !owner.contains("fn metal_direct_destination_is_split_by_submission_and_group_encoding(")
    );
    assert!(!owner
        .contains("fn metal_distinct_classic_batch_execution_is_split_from_cleanup_dispatch("));
    assert!(
        child.contains("fn metal_ht_chunk_tests_are_split_by_planning_cache_and_status_behavior(")
    );
    assert!(
        child.contains("fn metal_direct_destination_is_split_by_submission_and_group_encoding(")
    );
    assert!(
        child.contains("fn metal_distinct_classic_batch_execution_is_split_from_cleanup_dispatch(")
    );
    assert!(
        owner.lines().count() < 550,
        "Metal compute policy root must not absorb batch execution structure checks"
    );
    assert!(child.lines().count() < 250);
}

#[test]
fn metal_multitile_device_tests_are_split_by_pixel_contract() {
    let root = repo_root();
    let test_root = root.join("crates/j2k-metal/tests/device/multitile_color");
    let shell = fs::read_to_string(root.join("crates/j2k-metal/tests/device/multitile_color.rs"))
        .expect("read Metal multi-tile test shell");

    for module in ["batch_inputs", "classic", "gray12", "rgb", "signed"] {
        assert!(shell.contains(&format!("mod {module};")));
        assert!(test_root.join(format!("{module}.rs")).exists());
    }
    assert!(shell.lines().count() < 25);
    assert!(!shell.contains("fn independent_openjph_multitile_gray12_decodes_exactly_on_metal("));
    assert!(!shell.contains("fn independent_openjph_multitile_rgb_decodes_exactly_on_metal("));
    assert!(!shell.contains("fn classic_multitile_rgb8_decodes_exactly_on_metal("));
    assert!(fs::read_to_string(test_root.join("gray12.rs"))
        .expect("read Gray12 multi-tile tests")
        .contains("fn independent_openjph_multitile_gray12_decodes_exactly_on_metal("));
    assert!(fs::read_to_string(test_root.join("rgb.rs"))
        .expect("read RGB multi-tile tests")
        .contains("fn independent_openjph_multitile_rgb_decodes_exactly_on_metal("));
}

#[test]
fn owned_batch_test_support_is_split_by_responsibility() {
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
        let module_path = test_root.join(format!("{module}.rs"));
        let source = fs::read_to_string(&module_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", module_path.display()));
        assert!(source.contains(owned_symbol));
        assert!(
            source.lines().count() < max_lines,
            "{} exceeded its focused {max_lines}-line limit",
            module_path
                .strip_prefix(root)
                .unwrap_or(&module_path)
                .display()
        );
    }
    assert!(
        shell.lines().count() < 40,
        "owned_batch.rs must remain a focused integration-test module shell"
    );
    assert!(!shell.contains("fn htj2k_gray8_fixture("));
    assert!(!shell.contains("fn native_request_oracle("));
    assert!(!shell.contains("fn assert_prepared_ht_payload_ranges_reconstruct_owned_bytes("));

    let rgba_path = test_root.join("rgba.rs");
    let rgba = fs::read_to_string(&rgba_path).expect("read focused owned-batch RGBA tests");
    assert!(
        rgba.lines().count() < 400,
        "{} must remain focused on RGBA behavior",
        rgba_path.strip_prefix(root).unwrap_or(&rgba_path).display()
    );
    assert!(
        !rgba.contains("fn prepared_htj2k_gray_and_rgb_support_native_types_and_requests_exactly(")
    );

    for path in rust_sources(&test_root) {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(
            !source.lines().any(|line| line.trim() == "use super::*;"),
            "{} must import only the owned-batch contracts it uses",
            path.strip_prefix(root).unwrap_or(&path).display()
        );
    }
}

#[test]
fn adapter_crates_do_not_import_codec_private_modules() {
    assert_rust_source_scan_checks(
        repo_root(),
        &[RustSourceScanCheck::new(
            "adapter codec-private module imports",
            &[
                "crates/j2k-jpeg-metal",
                "crates/j2k-jpeg-cuda",
                "crates/j2k-metal",
                "crates/j2k-cuda",
            ],
        )
        .forbidden(&["::__private", " __private::"])],
    );
}

#[test]
fn production_j2k_cuda_code_does_not_reference_nvjpeg() {
    assert_rust_source_scan_checks(
        repo_root(),
        &[RustSourceScanCheck::new(
            "production J2K CUDA nvJPEG references",
            &[
                "crates/j2k-cuda-runtime/src",
                "crates/j2k-jpeg-cuda/src",
                "crates/j2k-jpeg-cuda/benches",
            ],
        )
        .forbidden(&["nvjpeg", "nvJPEG", "Nvjpeg", "NVJPEG"])],
    );
}

#[test]
fn cuda_runtime_rejects_product_cuda_c_and_checked_in_ptx() {
    let root = repo_root();
    let cuda_runtime_src = root.join("crates/j2k-cuda-runtime/src");
    let mut forbidden = Vec::new();
    let mut pending = vec![cuda_runtime_src.clone()];

    while let Some(dir) = pending.pop() {
        for entry in
            fs::read_dir(&dir).unwrap_or_else(|err| panic!("read {}: {err}", dir.display()))
        {
            let entry = entry.expect("read directory entry");
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if !matches!(path.extension().and_then(OsStr::to_str), Some("cu" | "ptx")) {
                continue;
            }
            let rel = path
                .strip_prefix(&cuda_runtime_src)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            if rel.starts_with("fixtures/") || rel.starts_with("test-fixtures/") {
                continue;
            }
            forbidden.push(rel);
        }
    }

    forbidden.sort();
    assert!(
        forbidden.is_empty(),
        "product CUDA C sources and checked-in product PTX are retired; use CUDA Oxide projects instead:\n{}",
        forbidden.join("\n")
    );
}

#[test]
fn cuda_runtime_dispatch_does_not_read_deprecated_oxide_route_selectors() {
    let root = repo_root();
    let runtime_src = root.join("crates/j2k-cuda-runtime/src");
    let deprecated_selector = ["J2K", "CUDA", "USE", "OXIDE"].join("_");
    let mut violations = Vec::new();

    for path in rust_sources(&runtime_src) {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        if source.contains(&deprecated_selector) {
            violations.push(
                path.strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            );
        }
    }

    assert!(
        violations.is_empty(),
        "CUDA Oxide route selection must be feature-driven, not runtime-env driven:\n{}",
        violations.join("\n")
    );
}

#[test]
fn cuda_trace_export_is_non_clobbering_and_documented() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-cuda/src/profile/trace.rs")
                .named("CUDA profile trace writer")
                .required(&[
                    "OpenOptions::new().write(true).create_new(true).open(path)",
                    "fn write_trace_file",
                    "emit_trace_write_error(\"cuda_htj2k_trace_write\"",
                    "emit_trace_write_error(\"cuda_htj2k_encode_trace_write\"",
                    "std::io::ErrorKind::AlreadyExists",
                    "CUDA trace path already exists",
                ])
                .forbidden(&["std::fs::write(&trace_path"]),
            FilePatternCheck::new("crates/j2k-cuda/src/profile/tests.rs")
                .named("CUDA trace non-clobber regression")
                .required(&[
                    "write_trace_file(&path, trace)",
                    "write_trace_file(&path, \"replace\")",
                    "ErrorKind::AlreadyExists",
                ]),
            FilePatternCheck::new("docs/env-vars.md")
                .named("environment variable docs")
                .required(&[
                    "operator-supplied path",
                    "Existing files are not overwritten",
                    "parent directories are not created",
                ]),
        ],
    );
}

#[test]
fn cuda_adapter_crates_keep_public_libs_as_module_shells() {
    let root = repo_root();
    let expected_modules = [
        (
            "crates/j2k-jpeg-cuda",
            [
                "codec.rs",
                "decoder.rs",
                "error.rs",
                "runtime.rs",
                "session.rs",
                "surface.rs",
            ]
            .as_slice(),
        ),
        (
            "crates/j2k-cuda",
            [
                "codec.rs",
                "decoder.rs",
                "encode.rs",
                "error.rs",
                "runtime.rs",
                "session.rs",
                "surface.rs",
            ]
            .as_slice(),
        ),
    ];

    for (crate_dir, modules) in expected_modules {
        let src_dir = root.join(crate_dir).join("src");
        let lib_path = src_dir.join("lib.rs");
        let lib = fs::read_to_string(&lib_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", lib_path.display()));
        let line_count = lib.lines().count();
        assert!(
            line_count <= 220,
            "{} should stay a thin public module shell; found {line_count} lines",
            lib_path.strip_prefix(root).unwrap_or(&lib_path).display()
        );

        for module in modules {
            let module_path = src_dir.join(module);
            assert!(
                module_path.exists(),
                "{} must exist to keep CUDA adapter responsibilities focused",
                module_path
                    .strip_prefix(root)
                    .unwrap_or(&module_path)
                    .display()
            );
        }
    }
}

#[test]
fn reusable_benchmark_generators_live_in_test_support() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("crates/j2k-test-support/src/lib.rs")
            .named("j2k-test-support")
            .required(&[
                "pub fn gradient_u8",
                "pub fn patterned_rgb8_tiles",
                "pub fn gpu_bench_rgb8",
            ])],
    );
}

#[test]
fn gpu_runtime_tests_do_not_silently_return_on_missing_hardware_gates() {
    assert_rust_source_scan_checks(
        repo_root(),
        &[RustSourceScanCheck::new(
            "GPU runtime tests must use j2k-test-support gates so optional local skips are visible and CI require gates fail closed",
            &[
                "crates/j2k-cuda-runtime/src",
                "crates/j2k-cuda/src",
                "crates/j2k-cuda/tests",
                "crates/j2k-jpeg-cuda/tests",
                "crates/j2k-transcode-cuda/tests",
                "crates/j2k-metal/src",
                "crates/j2k-metal/tests",
                "crates/j2k-jpeg-metal/tests",
                "crates/j2k-transcode-metal/tests",
            ],
        )
        .forbidden(&[
            "if !cuda_runtime_required() {\n        return;",
            "if !cuda_strict_oxide_required() {\n        return;",
            "if !cuda_jpeg_hardware_decode_required() {\n        return;",
            "if std::env::var_os(\"J2K_REQUIRE_CUDA_RUNTIME\").is_none() {\n        return;",
            "if Device::system_default().is_none() {\n            eprintln!(\"skipping",
            "no Metal device is available",
        ])],
    );
}
