// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::repo_lint_support::{assert_file_pattern_checks, repo_root, FilePatternCheck};

#[test]
fn j2k_ml_stays_independent_publishable_and_explicitly_feature_gated() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-ml/Cargo.toml")
                .named("j2k-ml manifest")
                .required(&[
                    "name = \"j2k-ml\"",
                    "[package.metadata.docs.rs]",
                    "default = []",
                    "cpu = []",
                    "cuda = [",
                    "metal = [",
                ]),
            FilePatternCheck::new("crates/j2k-ml/README.md")
                .named("j2k-ml independence notice")
                .required(&[
                    "independent integration",
                    "not an official Tracel or Burn crate",
                ]),
        ],
    );
}

#[test]
fn j2k_ml_uses_a_portable_arm_linux_test_backend() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-ml/Cargo.toml")
                .named("j2k-ml target-specific test backends")
                .required(&[
                    "target.'cfg(all(target_arch = \"aarch64\", target_os = \"linux\"))'.dev-dependencies",
                    "burn-ndarray = { workspace = true }",
                    "target.'cfg(not(all(target_arch = \"aarch64\", target_os = \"linux\")))'.dev-dependencies",
                    "burn-flex = { workspace = true }",
                ]),
            FilePatternCheck::new("docs/j2k-ml.md")
                .named("j2k-ml ARM backend rationale")
                .required(&[
                    "Linux AArch64",
                    "https://github.com/sarah-quinones/gemm/issues/31",
                    "https://github.com/sarah-quinones/gemm/pull/43",
                ]),
        ],
    );
}
