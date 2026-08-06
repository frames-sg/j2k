// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::repo_lint_support::{assert_file_pattern_checks, repo_root, FilePatternCheck};

#[test]
fn t803_claims_remain_candidate_scoped_until_release_evidence_passes() {
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
                .required(&["docs/t803-conformance.md", "candidate/pending"])
                .forbidden(&["full JPEG 2000 Part 1 codestream support"]),
            FilePatternCheck::new("docs/public-support.md")
                .required(&["docs/t803-conformance.md", "support-inventory.tsv"])
                .forbidden(&["full JPEG 2000 Part 1"]),
            FilePatternCheck::new("docs/t803-conformance.md").required(&[
                "ISO/IEC 15444-4:2024 / ITU-T T.803 v3",
                "Formal claim: **not made**",
                "Profile-1 Cclass-1",
                "Profile-1 Cclass-1HF",
                "Annex G JP2 reader",
                "c1-c0p0-13",
                "adapter IUT",
                "informative",
                "T.803 does not establish robustness, security, adoption, or performance",
            ]),
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
