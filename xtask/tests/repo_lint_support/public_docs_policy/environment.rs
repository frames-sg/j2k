// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs};

use super::super::{
    assert_file_pattern_checks, documented_j2k_env_vars, is_archived_handoff,
    is_internal_j2k_token, is_repo_lint_test_source, j2k_env_tokens, repo_root, repo_text_files,
    FilePatternCheck,
};

#[test]
fn supported_j2k_env_vars_are_documented() {
    let root = repo_root();
    let docs_path = root.join("docs/env-vars.md");
    let docs = fs::read_to_string(&docs_path)
        .unwrap_or_else(|err| panic!("read {}: {err}", docs_path.display()));
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("README.md")
                .named("README environment-variable reference")
                .required(&["docs/env-vars.md"]),
            FilePatternCheck::new("docs/env-vars.md")
                .named("supported environment-variable reference")
                .required(&["| `J2K_SEMVER_TOOLCHAIN` | Rejected by `cargo xtask semver`; Rust `1.96` is pinned in source and CI. | Must not be set | Test/CI |"])
                .forbidden(&["J2K_JPEG_METAL_SPLIT_FAST420_BATCH"]),
        ],
    );
    let documented = documented_j2k_env_vars(&docs);
    assert!(
        !documented.is_empty(),
        "docs/env-vars.md must document supported J2K_* environment variables"
    );

    let mut missing = Vec::new();
    let mut referenced = BTreeSet::new();
    for path in repo_text_files(root) {
        if is_archived_handoff(&path)
            || path.ends_with("docs/env-vars.md")
            || path.ends_with("engineering/ai-codebase-audit-remediation-plan.md")
            || is_repo_lint_test_source(root, &path)
        {
            continue;
        }
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for token in j2k_env_tokens(&source) {
            referenced.insert(token.clone());
            if is_internal_j2k_token(&token) {
                continue;
            }
            if !documented.contains(&token) {
                missing.push(format!(
                    "{}: {token}",
                    path.strip_prefix(root).unwrap_or(&path).display()
                ));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "supported J2K_* environment variables must be documented in docs/env-vars.md:\n{}",
        missing.join("\n")
    );
    let stale = documented
        .difference(&referenced)
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        stale.is_empty(),
        "docs/env-vars.md documents J2K_* variables with no repo reference:\n{}",
        stale.join("\n")
    );
    for phantom in [
        "J2K_LEVEL1_CUDA_HT_MIN_MPS",
        "J2K_LEVEL1_CUDA_HT_MIN_SPEEDUP_VS_NVIDIA",
        "J2K_LEVEL2_CUDA_HT_MIN_MPS",
        "J2K_LEVEL2_CUDA_HT_MIN_SPEEDUP_VS_NVIDIA",
    ] {
        assert!(
            !documented.contains(phantom),
            "phantom GPU validation env var `{phantom}` must not be documented without an implementation"
        );
    }
}
