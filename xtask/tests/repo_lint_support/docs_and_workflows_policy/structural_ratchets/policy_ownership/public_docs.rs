// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

#[test]
fn public_docs_policy_is_a_shell_over_focused_contract_domains() {
    let root = repo_root();
    let shell_relative = "xtask/tests/repo_lint_support/public_docs_policy.rs";
    let shell = fs::read_to_string(root.join(shell_relative))
        .unwrap_or_else(|error| panic!("read {shell_relative}: {error}"));
    assert!(shell.lines().count() < 25);

    for (module, owned_symbol, max_lines) in [
        ("environment", "fn supported_j2k_env_vars_are_documented(", 125),
        (
            "accelerator_evidence",
            "fn accelerator_support_and_benchmark_evidence_have_single_document_owners(",
            250,
        ),
        (
            "navigation_packaging",
            "fn published_crates_have_crates_io_landing_readmes(",
            400,
        ),
        (
            "benchmark_publication",
            "fn benchmark_docs_define_publication_gate_for_openjpeg_and_grok(",
            250,
        ),
        (
            "metal_safety",
            "fn metal_consistency_cleanup_keeps_names_status_buffers_and_marker_sizes_single_sourced(",
            300,
        ),
        (
            "repository_hygiene",
            "fn public_text_does_not_embed_local_user_home_paths(",
            175,
        ),
    ] {
        assert!(shell.contains(&format!("mod {module};")));
        assert!(!shell.contains(owned_symbol));
        let relative = format!("xtask/tests/repo_lint_support/public_docs_policy/{module}.rs");
        let source = fs::read_to_string(root.join(&relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(source.contains(owned_symbol));
        assert!(source.lines().count() < max_lines);
        assert!(!source.lines().any(|line| line.trim() == "use super::*;"));
        assert!(!source.lines().any(|line| line.trim_start().starts_with("include!(")));
    }
}
