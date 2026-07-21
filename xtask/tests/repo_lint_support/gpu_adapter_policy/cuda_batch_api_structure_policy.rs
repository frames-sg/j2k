// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::repo_root;

#[test]
fn cuda_batch_decoder_api_tests_use_focused_real_modules() {
    let root = repo_root();
    let shell_relative = "crates/j2k-cuda/tests/batch_decoder_api.rs";
    let shell = fs::read_to_string(root.join(shell_relative))
        .unwrap_or_else(|error| panic!("read {shell_relative}: {error}"));
    assert!(
        shell.lines().count() < 100,
        "{shell_relative} must remain a focused integration-test module shell"
    );

    for (module, max_lines) in [
        ("support", 75),
        ("basic_contracts", 175),
        ("multitile", 200),
        ("exact_color", 375),
        ("external_lifecycle", 325),
        ("async_resident", 150),
        ("classic_native", 450),
        ("refinement_overlap", 100),
        ("rgba", 275),
        ("session_soak", 150),
        ("signed_rgb", 200),
    ] {
        assert!(
            shell.contains(&format!("mod {module};")),
            "{shell_relative} must declare {module} as a real module"
        );
        let relative = format!("crates/j2k-cuda/tests/batch_decoder_api/{module}.rs");
        let source = fs::read_to_string(root.join(&relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "{relative} exceeded its focused line-count ratchet of {max_lines}"
        );
        assert!(
            !source.lines().any(|line| line.trim() == "use super::*;"),
            "{relative} must keep explicit imports"
        );
    }
}

#[test]
fn cuda_grayscale_contiguous_store_keeps_zero_fill_and_store_on_one_stream_boundary() {
    let root = repo_root();
    let relative = "crates/j2k-cuda-runtime/src/j2k_decode/store/grayscale_batch/api.rs";
    let source = fs::read_to_string(root.join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"));

    assert!(
        !source.contains("self.synchronize()?;"),
        "{relative} must not synchronize the host between ordered zero-fill and final-store work"
    );
    assert!(
        source.contains("attach_zero_fill_completion"),
        "{relative} must retain a completion event when zero-fill is the only queued work"
    );
    assert!(
        source.contains("retire_failed_zero_fill"),
        "{relative} must retire or quarantine zero-fill work when final-store enqueue fails"
    );
}

#[test]
fn cuda_resident_batch_group_has_one_required_dense_owner() {
    let root = repo_root();
    let relative = "crates/j2k-cuda/src/batch/types.rs";
    let source = fs::read_to_string(root.join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"));

    assert!(source.contains("pub const fn dense_output(&self) -> &CudaResidentBatchBuffer"));
    assert!(
        !source.contains("pub const fn dense_output(&self) -> Option<&CudaResidentBatchBuffer>")
    );
    assert!(!source.contains("Option<CudaResidentBatchBuffer>"));
    assert!(!source.contains("Some(self.dense_output)"));
}
