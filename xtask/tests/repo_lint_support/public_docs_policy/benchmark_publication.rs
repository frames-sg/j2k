// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::{assert_pattern_checks, repo_root, xtask_sources, PatternCheck};

#[test]
fn benchmark_docs_define_publication_gate_for_openjpeg_and_grok() {
    let root = repo_root();
    let benchmark_corpora = fs::read_to_string(root.join("docs/benchmark-corpora.md"))
        .expect("read benchmark corpus docs");
    let benchmark_evidence = fs::read_to_string(root.join("docs/benchmark-evidence.md"))
        .expect("read benchmark evidence docs");
    let env_vars = fs::read_to_string(root.join("docs/env-vars.md")).expect("read env var docs");
    let benchmark_docs = format!("{benchmark_corpora}\n{benchmark_evidence}\n{env_vars}");
    let xtask = xtask_sources(root);
    let ci = fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read CI workflow");

    assert_pattern_checks(&[
        PatternCheck::new("benchmark publication docs", &benchmark_docs).required(&[
            "published benchmark",
            "J2K_COMPARE_THREADS",
            "J2K_REQUIRE_OPENJPEG=1",
            "J2K_REQUIRE_GROK=1",
            "comparator availability",
            "comparator version",
            "input source",
            "j2k-generated",
        ]),
        PatternCheck::new("xtask benchmark signoff", &xtask).required(&[
            "\"j2k-bench-signoff\"",
            "grok_parity",
            "libjpeg_turbo_compare",
            "bench-libjpeg-turbo",
            "J2K_REQUIRE_LIBJPEG_TURBO",
            "passed_test_count",
            "expected at least",
        ]),
        PatternCheck::new("CI comparator parity job", &ci).required(&[
            "comparator-parity:",
            "grokj2k-tools",
            "libgrokj2k1-dev",
            "libopenjp2-tools",
            "libturbojpeg0-dev",
            "pkg-config --modversion libgrokj2k",
            "pkg-config --modversion libturbojpeg",
            "J2K_REQUIRE_OPENJPEG: \"1\"",
            "J2K_REQUIRE_GROK: \"1\"",
            "J2K_REQUIRE_LIBJPEG_TURBO: \"1\"",
            "cargo xtask j2k-bench-signoff",
        ]),
    ]);
}

#[test]
fn adoption_starter_corpus_fallback_is_pinned() {
    let root = repo_root();
    let workflow = fs::read_to_string(root.join(".github/workflows/gpu-validation.yml"))
        .expect("read GPU validation workflow");
    let benchmark_docs = fs::read_to_string(root.join("docs/benchmark-corpora.md"))
        .expect("read benchmark corpus docs");
    let openjpeg_commit = "39524bd3a601d90ed8e0177559400d23945f96a9";

    let workflow_required = [
        "sha256sum -c - <<'SHA256'".to_string(),
        "a56e27cbf5f843c048b6af1d6e090760e9c92fadba88b7dee0205918a37523bd  kodim01.png".to_string(),
        "1071c68372cc5a01435c2c225a5cf7d4bb803846ec08bb6b3d6721b156d7cb96  kodim24.png".to_string(),
        "downloaded-from-r0k-us-kodak-lossless-true-color-sha256-pinned".to_string(),
        format!("J2K_STARTER_OPENJPEG_DATA_COMMIT={openjpeg_commit}"),
        "git -C target/j2k-public-corpora/openjpeg-data fetch --depth 1 origin".to_string(),
        "git -C target/j2k-public-corpora/openjpeg-data checkout --detach".to_string(),
        format!("source-native-openjpeg-data-conformance-dir@{openjpeg_commit}"),
        format!("source-native-openjpeg-data-nonregression-dir@{openjpeg_commit}"),
    ];
    let workflow_required = workflow_required
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let benchmark_docs_required = [
        "sha256sum -c - <<'SHA256'",
        openjpeg_commit,
        "downloaded-from-r0k-us-kodak-lossless-true-color-sha256-pinned",
        "OpenJPEG-data commit",
        "SHA-256-checked Kodak PNGs",
        "fixed OpenJPEG-data commit",
    ];

    assert_pattern_checks(&[
        PatternCheck::new(
            "GPU adoption fallback pinned starter corpus inputs",
            &workflow,
        )
        .required(&workflow_required)
        .forbidden(&["git clone --depth 1 https://github.com/uclouvain/openjpeg-data"]),
        PatternCheck::new(
            "benchmark corpus docs pinned starter corpus inputs",
            &benchmark_docs,
        )
        .required(&benchmark_docs_required),
    ]);
}

#[test]
fn benchmark_publication_gate_rules_are_single_sourced() {
    let root = repo_root();
    let gate = fs::read_to_string(root.join("xtask/src/publication_gate.rs"))
        .expect("read publication gate module");
    let benchmark = fs::read_to_string(root.join("xtask/src/adoption_benchmark/support.rs"))
        .expect("read adoption benchmark publication support module");
    let report = fs::read_to_string(root.join("xtask/src/adoption_report.rs"))
        .expect("read adoption report module");

    assert_pattern_checks(&[
        PatternCheck::new("publication gate module", &gate).required(&[
            "PUBLICATION_GATE_KEYS",
            "PublicationGateEvaluation",
            "pub(crate) fn collect_publication_gate_issues",
            "publication_eligible",
            "publication_blockers",
            "benchmark_complete",
        ]),
        PatternCheck::new("adoption benchmark writer publication gate use", &benchmark)
            .required(&[
                "PUBLICATION_GATE_KEYS",
                "collect_publication_gate_issues",
                "Some(&fixture_metadata)",
                "Some(&encode_metadata)",
            ])
            .forbidden(&["fn collect_publication_gate_issues("]),
        PatternCheck::new("adoption report checker publication gate use", &report)
            .required(&[
                "collect_publication_gate_issues",
                "summary.get(\"cpu_fixture_compare\")",
                "summary.get(\"cpu_encode_compare\")",
            ])
            .forbidden(&["fn collect_gate_issue("]),
    ]);
}
