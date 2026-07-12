// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

fn read(root: &Path, relative: &str) -> String {
    fs::read_to_string(root.join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

type FocusedModule = (&'static str, String, usize);

fn encode_modules(root: &Path) -> [FocusedModule; 9] {
    [
        (
            "api",
            read(root, "crates/j2k-cuda-runtime/src/htj2k_encode/api.rs"),
            300,
        ),
        (
            "host execution",
            read(
                root,
                "crates/j2k-cuda-runtime/src/htj2k_encode/api/host_execution.rs",
            ),
            80,
        ),
        (
            "resource upload",
            read(
                root,
                "crates/j2k-cuda-runtime/src/htj2k_encode/api/resources.rs",
            ),
            50,
        ),
        (
            "completion",
            read(
                root,
                "crates/j2k-cuda-runtime/src/htj2k_encode/completion.rs",
            ),
            380,
        ),
        (
            "context validation",
            read(
                root,
                "crates/j2k-cuda-runtime/src/htj2k_encode/context_validation.rs",
            ),
            80,
        ),
        (
            "launch",
            read(root, "crates/j2k-cuda-runtime/src/htj2k_encode/launch.rs"),
            150,
        ),
        (
            "planning",
            read(root, "crates/j2k-cuda-runtime/src/htj2k_encode/planning.rs"),
            310,
        ),
        (
            "compact planning",
            read(
                root,
                "crates/j2k-cuda-runtime/src/htj2k_encode/planning/compact.rs",
            ),
            125,
        ),
        (
            "types",
            read(root, "crates/j2k-cuda-runtime/src/htj2k_encode/types.rs"),
            400,
        ),
    ]
}

fn decode_modules(root: &Path) -> [FocusedModule; 10] {
    [
        (
            "api",
            read(root, "crates/j2k-cuda-runtime/src/htj2k_decode/api.rs"),
            200,
        ),
        (
            "completion",
            read(
                root,
                "crates/j2k-cuda-runtime/src/htj2k_decode/completion.rs",
            ),
            525,
        ),
        (
            "dequant completion",
            read(
                root,
                "crates/j2k-cuda-runtime/src/htj2k_decode/completion/dequant.rs",
            ),
            125,
        ),
        (
            "context validation",
            read(
                root,
                "crates/j2k-cuda-runtime/src/htj2k_decode/context_validation.rs",
            ),
            130,
        ),
        (
            "launch",
            read(root, "crates/j2k-cuda-runtime/src/htj2k_decode/launch.rs"),
            260,
        ),
        (
            "output regions",
            read(
                root,
                "crates/j2k-cuda-runtime/src/htj2k_decode/output_regions.rs",
            ),
            150,
        ),
        (
            "planning",
            read(root, "crates/j2k-cuda-runtime/src/htj2k_decode/planning.rs"),
            220,
        ),
        (
            "queued completion",
            read(root, "crates/j2k-cuda-runtime/src/htj2k_decode/queued.rs"),
            180,
        ),
        (
            "status",
            read(root, "crates/j2k-cuda-runtime/src/htj2k_decode/status.rs"),
            80,
        ),
        (
            "types",
            read(root, "crates/j2k-cuda-runtime/src/htj2k_decode/types.rs"),
            390,
        ),
    ]
}

fn module_source<'a>(modules: &'a [FocusedModule], name: &str) -> &'a str {
    modules
        .iter()
        .find_map(|(candidate, source, _)| (*candidate == name).then_some(source.as_str()))
        .unwrap_or_else(|| panic!("missing HTJ2K {name} policy source"))
}

fn assert_module_boundaries(encode: &[FocusedModule], decode: &[FocusedModule]) {
    for (name, source, max_lines) in encode.iter().chain(decode) {
        assert!(
            source.lines().count() < *max_lines,
            "HTJ2K {name} module exceeded {max_lines} lines"
        );
        assert!(
            !source.contains("include!("),
            "HTJ2K {name} must be a real Rust module"
        );
        assert!(
            !source.lines().any(|line| line.trim() == "use super::*;"),
            "HTJ2K {name} must use explicit dependencies"
        );
    }
}

fn assert_facade_contract(name: &str, source: &str, public_owners: &[&str]) {
    assert!(
        source.lines().count() < 80,
        "HTJ2K {name} root must stay a facade"
    );
    let mut required = vec![
        "mod api;",
        "mod completion;",
        "mod launch;",
        "mod planning;",
        "mod types;",
    ];
    required.extend_from_slice(public_owners);
    assert_pattern_checks(&[PatternCheck::new("HTJ2K runtime facade", source)
        .required(&required)
        .forbidden(&["impl CudaContext", "#[repr(C)]", "include!("])]);
}

fn assert_encode_ownership(modules: &[FocusedModule]) {
    assert_pattern_checks(&[
        PatternCheck::new("HTJ2K encode API ownership", module_source(modules, "api")).required(&[
            "encode_htj2k_codeblocks_resident_with_resources_and_pool",
            "encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool",
        ]),
        PatternCheck::new(
            "HTJ2K encode resource upload ownership",
            module_source(modules, "resource upload"),
        )
        .required(&[
            "upload_htj2k_encode_resources",
            "HTJ2K_UVLC_ENCODE_TABLE_BYTES",
        ]),
        PatternCheck::new(
            "HTJ2K encode completion ownership",
            module_source(modules, "completion"),
        )
        .required(&[
            "status_readback",
            "htj2k_encode_compact_jobs",
            "CudaHtj2kEncodeStageTimings::from_parts",
            "copy_kernel_dispatches",
        ]),
        PatternCheck::new(
            "HTJ2K encode launch ownership",
            module_source(modules, "launch"),
        )
        .required(&[
            "launch_htj2k_encode_codeblocks",
            "launch_htj2k_encode_multi_input_kernel",
            "Htj2kEncodeCodeblocksMultiInputCleanup64",
            "launch_htj2k_compact_codeblocks",
        ]),
        PatternCheck::new(
            "HTJ2K encode planning ownership",
            module_source(modules, "planning"),
        )
        .required(&[
            "validate_htj2k_encode_codeblock_shape",
            "htj2k_encode_multi_input_kernel_jobs",
        ]),
        PatternCheck::new(
            "HTJ2K compact planning ownership",
            module_source(modules, "compact planning"),
        )
        .required(&[
            "trait Htj2kCompactPlanJob",
            "htj2k_encode_compact_jobs_impl",
        ]),
    ]);
}

fn assert_decode_ownership(modules: &[FocusedModule]) {
    assert_pattern_checks(&[
        PatternCheck::new("HTJ2K decode API ownership", module_source(modules, "api")).required(&[
            "upload_htj2k_decode_table_resources",
            "upload_htj2k_decode_resources_with_tables_and_pool",
            "allocate_htj2k_codeblock_coefficients_with_pool",
        ]),
        PatternCheck::new(
            "HTJ2K decode completion ownership",
            module_source(modules, "completion"),
        )
        .required(&[
            "decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool",
            "select_status_release_result",
            "submit_htj2k_decode_and_dequantize",
            "CudaHtj2kDecodeStageTimings",
        ]),
        PatternCheck::new(
            "HTJ2K decode dequant completion ownership",
            module_source(modules, "dequant completion"),
        )
        .required(&[
            "submit_htj2k_dequantize_htj2k_codeblocks",
            "j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_and_live_host_bytes",
        ]),
        PatternCheck::new(
            "HTJ2K decode launch ownership",
            module_source(modules, "launch"),
        )
        .required(&[
            "launch_htj2k_decode_codeblocks_multi",
            "launch_j2k_dequantize_htj2k_cleanup_jobs_multi",
            "CudaLaunchMode::Async",
        ]),
        PatternCheck::new(
            "HTJ2K decode planning ownership",
            module_source(modules, "planning"),
        )
        .required(&[
            "htj2k_kernel_jobs",
            "htj2k_cleanup_multi_kernel_jobs",
            "htj2k_decode_multi_cleanup_dequant_kernel_for_jobs",
        ]),
    ]);
}

#[test]
fn cuda_htj2k_runtime_owners_remain_focused_real_modules() {
    let root = repo_root();
    let encode_root = read(root, "crates/j2k-cuda-runtime/src/htj2k_encode.rs");
    let decode_root = read(root, "crates/j2k-cuda-runtime/src/htj2k_decode.rs");
    let encode = encode_modules(root);
    let decode = decode_modules(root);

    assert_module_boundaries(&encode, &decode);
    assert_facade_contract(
        "encode",
        &encode_root,
        &["CudaHtj2kEncodeResources", "CudaHtj2kEncodedCodeBlocks"],
    );
    assert_facade_contract(
        "decode",
        &decode_root,
        &["CudaQueuedHtj2kCleanup", "CudaHtj2kDecodeResources"],
    );
    assert_encode_ownership(&encode);
    assert_decode_ownership(&decode);
}

#[test]
fn cuda_htj2k_split_keeps_abi_and_behavior_ratchets() {
    let root = repo_root();
    let encode_types = read(root, "crates/j2k-cuda-runtime/src/htj2k_encode/types.rs");
    let decode_types = read(root, "crates/j2k-cuda-runtime/src/htj2k_decode/types.rs");
    let abi = read(root, "crates/j2k-cuda-runtime/src/bytes/abi.rs");
    let behavior = [
        read(root, "crates/j2k-cuda-runtime/src/tests.rs"),
        read(root, "crates/j2k-cuda-runtime/src/tests/pipeline.rs"),
        read(
            root,
            "crates/j2k-cuda-runtime/src/htj2k_encode/context_validation/tests.rs",
        ),
        read(
            root,
            "crates/j2k-cuda-runtime/src/htj2k_decode/status/tests.rs",
        ),
    ]
    .concat();

    assert_pattern_checks(&[
        PatternCheck::new("HTJ2K encode ABI owners", &encode_types).required(&[
            "#[repr(C)]",
            "CudaHtj2kEncodeKernelJob",
            "CudaHtj2kEncodeMultiInputKernelJob",
            "CudaHtj2kEncodeStatus",
        ]),
        PatternCheck::new("HTJ2K decode ABI owners", &decode_types).required(&[
            "#[repr(C)]",
            "CudaHtj2kCodeBlockKernelJob",
            "CudaHtj2kCleanupMultiKernelJob",
            "CudaHtj2kDequantizeKernelJob",
            "CudaHtj2kStatus",
        ]),
        PatternCheck::new("HTJ2K compile-time ABI ledger", &abi).required(&[
            "CudaHtj2kEncodeKernelJob {",
            "CudaHtj2kEncodeMultiInputKernelJob {",
            "CudaHtj2kCleanupMultiKernelJob {",
            "CudaHtj2kDequantizeKernelJob {",
        ]),
        PatternCheck::new("HTJ2K split behavior coverage", &behavior).required(&[
            "htj2k_encode_compact_jobs_pack_actual_payloads",
            "context_match_validation_rejects_each_mismatched_category",
            "htj2k_decode_multi_kernel_routes_cleanup_only_jobs",
            "kernel_and_release_failures_are_both_preserved",
        ]),
    ]);
}
