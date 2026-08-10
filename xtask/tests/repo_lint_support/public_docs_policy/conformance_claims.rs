// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::repo_lint_support::{assert_file_pattern_checks, repo_root, FilePatternCheck};

#[test]
fn t803_candidate_claims_remain_exact_and_release_scoped() {
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
                    "Profile-1 Cclass-1",
                    "Profile-1 Cclass-1HF",
                    "Annex G JP2 reader",
                    "DS1-HM Cclass-1h, MMAGB 15",
                    "Cclass-1HFh, MMAGB 20",
                    "Annex G JPH",
                    "0/160 device-native, 81/160",
                    "development-dirty-worktree",
                    "not part of the published `0.8.1` claim",
                ])
                .forbidden(&[
                    "full JPEG 2000 Part 1 codestream support",
                    "Formal decoder claim for `0.8.1`",
                    "planned `0.8.1` release",
                ]),
            FilePatternCheck::new("docs/public-support.md")
                .required(&["docs/t803-conformance.md", "support-inventory.tsv"])
                .forbidden(&["full JPEG 2000 Part 1"]),
            FilePatternCheck::new("docs/t803-conformance.md")
                .required(&[
                    "ISO/IEC 15444-4:2024 / ITU-T T.803 v3",
                    "Status: **Part 1 published for 0.8.1; Part 15 unversioned development evidence;",
                    "Published formal decoder wording for release `0.8.1`:",
                    "Planned Part 15 decoder wording after a future exact-clean-SHA verification:",
                    "Profile-1 Cclass-1 compliant",
                    "Profile-1 Cclass-1HF compliant",
                    "Annex G JP2 reader compliant",
                    "DS1-HM Cclass-1h, MMAGB 15",
                    "Cclass-1HFh, MMAGB 20",
                    "Annex G JPH reader compliant at MMAGB 15",
                    "0/160 device-native, 81/160 hybrid, and 79/160 CPU-routed",
                    "development-dirty-worktree",
                    "zero skips",
                    "c1-c0p0-13",
                    "adapter IUT",
                    "informative",
                    "T.803 does not establish robustness, security, adoption, or performance",
                ])
                .forbidden(&[
                    "Formal claim: **not made**",
                    "0.8.1 candidate development evidence",
                ]),
            FilePatternCheck::new("corpus/j2k-conformance/README.md").required(&[
                "t803-v3.toml",
                "encoder-matrix-v2.toml",
                "support-inventory.tsv",
                "copyrighted electronic attachment",
                "must not be committed",
            ]),
        ],
    );
}
