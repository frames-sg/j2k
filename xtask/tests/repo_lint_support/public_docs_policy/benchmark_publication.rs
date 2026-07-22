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
    let full_validation = fs::read_to_string(root.join(".github/workflows/full-validation.yml"))
        .expect("read full validation workflow");

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
        PatternCheck::new("full validation comparator parity job", &full_validation).required(&[
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
fn adoption_gpu_workflow_requires_explicit_external_corpora() {
    let root = repo_root();
    let workflow = fs::read_to_string(root.join(".github/workflows/gpu-benchmarks.yml"))
        .expect("read GPU benchmark workflow");
    let benchmark_docs = fs::read_to_string(root.join("docs/benchmark-corpora.md"))
        .expect("read benchmark corpus docs");

    let workflow_required = [
        "inputs.suite == 'adoption'".to_string(),
        ": \"${J2K_ADOPTION_FIXTURES:?Set J2K_ADOPTION_FIXTURES}\"".to_string(),
        ": \"${J2K_ADOPTION_MANIFEST:?Set J2K_ADOPTION_MANIFEST}\"".to_string(),
        ": \"${J2K_ADOPTION_ENCODE_FIXTURES:?Set J2K_ADOPTION_ENCODE_FIXTURES}\"".to_string(),
        ": \"${J2K_ADOPTION_ENCODE_MANIFEST:?Set J2K_ADOPTION_ENCODE_MANIFEST}\"".to_string(),
        "--require-cuda".to_string(),
        "--require-metal".to_string(),
    ];
    let workflow_required = workflow_required
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let benchmark_docs_required = [
        "dispatch `GPU benchmarks` with `suite=adoption`",
        "fails closed when any variable is absent",
        "does not synthesize or download a fallback corpus",
        "--require-cuda",
        "--require-metal",
    ];

    assert_pattern_checks(&[
        PatternCheck::new(
            "GPU adoption workflow strict external corpus inputs",
            &workflow,
        )
        .required(&workflow_required)
        .forbidden(&["git clone", "curl ", "wget "]),
        PatternCheck::new(
            "benchmark corpus docs strict external corpus inputs",
            &benchmark_docs,
        )
        .normalized_required(&benchmark_docs_required),
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
