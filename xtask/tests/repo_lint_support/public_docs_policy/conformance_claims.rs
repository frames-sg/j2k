// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::repo_lint_support::{assert_file_pattern_checks, repo_root, FilePatternCheck};

#[test]
fn t803_claims_remain_exact_and_release_scoped() {
    let root = repo_root();
    assert!(
        !root.join("corpus/j2k-conformance/manifest.tsv").exists(),
        "the optional decode-smoke manifest must stay retired"
    );
    assert!(
        !root.join("crates/j2k/tests/iso_conformance.rs").exists(),
        "the environment-gated decode-smoke test must stay retired"
    );
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("README.md")
                .required(&[
                    "docs/t803-conformance.md",
                    "Profile-1 Cclass-1 compliant",
                    "Profile-1 Cclass-1HF compliant",
                    "Annex G JP2 reader compliant",
                    "CUDA: 0/90 device-native, 48/90 hybrid, 42/90 CPU-routed",
                    "Metal: 0/90 device-native, 48/90 hybrid, 42/90 CPU-routed",
                ])
                .forbidden(&[
                    "candidate/pending",
                    "full JPEG 2000 Part 1 codestream support",
                ]),
            FilePatternCheck::new("docs/public-support.md")
                .required(&["docs/t803-conformance.md", "support-inventory.tsv"])
                .forbidden(&["full JPEG 2000 Part 1"]),
            FilePatternCheck::new("docs/t803-conformance.md")
                .required(&[
                    "ISO/IEC 15444-4:2024 / ITU-T T.803 v3",
                    "Status: **0.8.1 release-scoped**",
                    "Formal decoder claim:",
                    "Profile-1 Cclass-1 compliant",
                    "Profile-1 Cclass-1HF compliant",
                    "Annex G JP2 reader compliant",
                    "0/90 device-native, 48/90 hybrid, and 42/90 CPU-routed",
                    "zero skips",
                    "c1-c0p0-13",
                    "adapter IUT",
                    "informative",
                    "T.803 does not establish robustness, security, adoption, or performance",
                ])
                .forbidden(&["candidate/pending", "Formal claim: **not made**"]),
            FilePatternCheck::new("corpus/j2k-conformance/README.md").required(&[
                "t803-v3.toml",
                "encoder-matrix-v1.toml",
                "support-inventory.tsv",
                "copyrighted electronic attachment",
                "must not be committed",
            ]),
        ],
    );
}
