// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

#[test]
fn j2k_ml_policy_keeps_domain_specific_children() {
    let root = repo_root().join("xtask/tests/repo_lint_support");
    let shell_path = root.join("j2k_ml_policy.rs");
    let shell = fs::read_to_string(&shell_path).expect("read j2k-ml policy root");

    for module in [
        "mod adapter;",
        "mod benchmark_evidence;",
        "mod benchmark_prepare_policy;",
        "mod benchmark_support_structure;",
        "mod features;",
        "mod interop;",
    ] {
        assert!(
            shell.contains(module),
            "j2k-ml policy root must own {module}"
        );
    }
    assert!(shell.contains("fn read(relative: &str) -> String"));
    assert!(!shell.contains("#[test]"));
    assert!(!shell.contains("RustEvidence"));
    assert!(!shell.contains("include!"));
    assert!(
        shell.lines().count() < 30,
        "j2k-ml policy root must remain a thin module/shared-read owner"
    );

    for (relative, symbols, max_lines) in [
        (
            "j2k_ml_policy/adapter.rs",
            &[
                "fn j2k_ml_is_a_thin_persistent_batch_adapter(",
                "fn metal_burn_decoder_keeps_batch_options_in_the_codec_session_only(",
            ][..],
            125usize,
        ),
        (
            "j2k_ml_policy/features.rs",
            &[
                "fn j2k_ml_stays_independent_experimental_and_explicitly_feature_gated(",
                "fn j2k_ml_uses_a_portable_arm_linux_test_backend(",
            ][..],
            100usize,
        ),
        (
            "j2k_ml_policy/interop.rs",
            &["fn j2k_ml_accelerator_zero_copy_contracts_are_source_enforced("][..],
            100usize,
        ),
        (
            "j2k_ml_policy/benchmark_evidence.rs",
            &[
                "struct RustEvidence",
                "impl<'ast> Visit<'ast> for RustEvidence",
                "fn rust_evidence(",
                "fn j2k_ml_batch_benchmarks_cover_native_medical_outputs_and_all_requests(",
            ][..],
            250usize,
        ),
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        for symbol in symbols {
            assert!(source.contains(symbol), "{relative} must own {symbol}");
            assert!(
                !shell.contains(symbol),
                "j2k-ml policy root must not own {symbol}"
            );
        }
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its domain-policy line-count ratchet"
        );
        assert!(!source.lines().any(|line| line.trim() == "use super::*;"));
        assert!(!source.contains("include!"));
    }
}
