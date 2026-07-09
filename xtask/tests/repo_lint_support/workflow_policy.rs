// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::*;

const CUDA_OXIDE_STRICT_BUILD_DOC_PATTERNS: &[&str] = &[
    "cuda-runtime",
    "CUDA Driver API dispatch",
    "CUDA Oxide",
    "J2K_REQUIRE_CUDA_OXIDE_BUILD=1",
    "placeholder PTX",
    "CUDA Oxide PTX was not built",
];

#[test]
fn github_workflows_parse_as_yaml() {
    let root = repo_root();
    let workflow_dir = root.join(".github/workflows");
    let mut workflow_paths = fs::read_dir(&workflow_dir)
        .expect("read .github/workflows")
        .map(|entry| entry.expect("read workflow directory entry").path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|value| value.to_str()),
                Some("yml" | "yaml")
            )
        })
        .collect::<Vec<_>>();
    workflow_paths.sort();
    assert!(!workflow_paths.is_empty(), "no GitHub workflow YAML found");

    for path in workflow_paths {
        let relative_path = path.strip_prefix(root).unwrap_or(&path);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", relative_path.display()));
        serde_yaml_ng::from_str::<serde_yaml_ng::Value>(&source)
            .unwrap_or_else(|err| panic!("parse {} as YAML: {err}", relative_path.display()));
    }
}

#[test]
fn ci_miri_job_is_a_required_gate() {
    let workflow =
        fs::read_to_string(repo_root().join(".github/workflows/ci.yml")).expect("read CI workflow");
    let miri_job = workflow_job(&workflow, "miri");

    assert_pattern_checks(&[PatternCheck::new("CI miri job", miri_job).required(&[
        "toolchain: nightly",
        "components: miri",
        "cargo xtask miri",
    ])]);
}

#[test]
fn ci_fuzz_run_budgets_are_nontrivial() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new(".github/workflows/ci.yml")
            .named("CI fuzz budgets")
            .required(&[
                "J2K_FUZZ_RUNS: \"512\"",
                "J2K_FUZZ_MAX_TOTAL_TIME_SECONDS: \"60\"",
                "J2K_FUZZ_RUNS: \"20000\"",
                "J2K_FUZZ_MAX_TOTAL_TIME_SECONDS: \"900\"",
            ])],
    );
}

#[test]
fn deny_paste_advisory_ignore_has_review_metadata() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("deny.toml")
            .named("deny.toml paste advisory metadata")
            .required(&[
                "RUSTSEC-2024-0436",
                "Review-by: 2026-12-31",
                "https://crates.io/crates/metal",
                "https://github.com/gfx-rs/metal-rs/blob/master/Cargo.toml",
                "https://rustsec.org/advisories/RUSTSEC-2024-0436.html",
                "https://github.com/gfx-rs/metal-rs/issues/349",
            ])],
    );
}

#[test]
fn unsafe_audit_rows_include_invariants_and_regression_guards() {
    let audit =
        fs::read_to_string(repo_root().join("docs/unsafe-audit.md")).expect("read unsafe audit");

    assert_pattern_checks(&[PatternCheck::new("unsafe audit table header", &audit)
        .required(&["| Path | Scope | Invariants | Regression guards |"])]);
    for line in audit
        .lines()
        .filter(|line| line.trim().starts_with("| `crates/"))
    {
        let cells = line.split('|').map(str::trim).collect::<Vec<_>>();
        assert!(
            cells.len() >= 6 && !cells[3].is_empty() && !cells[4].is_empty(),
            "unsafe audit row must include invariant and regression guard: {line}"
        );
    }
}

#[test]
fn ci_workflow_has_read_only_permissions_and_gpu_path_policy() {
    let root = repo_root();
    let workflow =
        fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read CI workflow");
    let gpu_policy = workflow_job(&workflow, "gpu-path-policy");
    let verifier = fs::read_to_string(root.join("scripts/github_actions_verify.py"))
        .expect("read GitHub Actions verifier");
    let codeowners = fs::read_to_string(root.join(".github/CODEOWNERS")).expect("read CODEOWNERS");

    assert_pattern_checks(&[
        PatternCheck::new("CI workflow default permissions", &workflow)
            .normalized_required(&["permissions:\n  contents: read"]),
        PatternCheck::new("CI GPU path policy job", gpu_policy)
            .required(&[
                "pull-requests: read",
                "actions: read",
                "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5",
                "scripts/github_actions_verify.py pr-gpu-policy",
                "--repository \"${REPOSITORY}\"",
                "--pr-number \"${PR_NUMBER}\"",
                "--head-sha \"${HEAD_SHA}\"",
                "--workflow gpu-validation.yml",
            ])
            .forbidden(&["urllib.request", "python3 <<'PY'"]),
        PatternCheck::new("repository-owned GitHub Actions verifier", &verifier).required(&[
            "def fetch_pull_request_paths(",
            "def classify_gpu_paths(",
            "def verify_workflow_run(",
            "def peel_annotated_tag(",
            "def verify_release_evidence(",
            "CUDA API compatibility on x86_64",
            "Metal validation on Apple Silicon",
            "Release candidate aggregate",
            "workflow run pagination exceeded",
            "workflow job pagination exceeded",
            "must be annotated",
        ]),
        PatternCheck::new("CODEOWNERS GPU path coverage", &codeowners).required(&[
            ".github/workflows/ci.yml",
            ".github/workflows/gpu-validation.yml",
            ".github/workflows/publish.yml",
            "scripts/github_actions_verify.py",
            "crates/j2k-cuda-runtime/",
            "crates/j2k-jpeg-cuda/",
            "crates/j2k-cuda/",
            "crates/j2k-transcode-cuda/",
            "crates/j2k-metal-support/",
            "crates/j2k-jpeg-metal/",
            "crates/j2k-metal/",
            "crates/j2k-transcode-metal/",
        ]),
    ]);
}

#[test]
fn release_candidate_and_publish_evidence_are_fail_closed() {
    let root = repo_root();
    let ci = fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read CI workflow");
    let publish = fs::read_to_string(root.join(".github/workflows/publish.yml"))
        .expect("read publish workflow");
    let aggregate = workflow_job(&ci, "release-candidate");
    let preflight = workflow_job(&publish, "preflight");
    let verifier_tests =
        fs::read_to_string(root.join("scripts/tests/test_github_actions_verify.py"))
            .expect("read GitHub Actions verifier tests");

    assert_pattern_checks(&[
        PatternCheck::new("release candidate aggregate", aggregate).required(&[
            "name: Release candidate aggregate",
            "if: ${{ always() }}",
            "github-actions-verifier",
            "gpu-path-policy",
            "fmt",
            "clippy",
            "panic-surface",
            "comparator-parity",
            "semver",
            "docs",
            "stable-api",
            "release-integrity",
            "unsafe-audit",
            "typos",
            "test",
            "release-cpu",
            "metal-compile",
            "no-std",
            "miri",
            "bench-build",
            "fuzz-build",
            "fuzz-run",
            "package",
            "coverage",
            "deny",
            "REQUIRED_RESULTS: ${{ toJSON(needs) }}",
        ]),
        PatternCheck::new("publish workflow exact-SHA policy", &publish)
            .required(&[
                "actions: read",
                "CRATES_IO_ALLOW_PUBLISHED_RERUN: ${{ vars.CRATES_IO_ALLOW_PUBLISHED_RERUN || 'false' }}",
                "DRY_RUN_ONLY: ${{ github.event_name == 'workflow_dispatch' }}",
                "Verify annotated tag and exact-SHA release evidence",
                "scripts/github_actions_verify.py verify-release",
                "--ci-branch main",
                "--aggregate-job \"Release candidate aggregate\"",
                "--cuda-job \"CUDA API compatibility on x86_64\"",
                "--metal-job \"Metal validation on Apple Silicon\"",
            ])
            .forbidden(&["inputs.dry-run-only"]),
        PatternCheck::new("publish preflight exact-SHA policy", preflight).required(&[
            "fetch-depth: 0",
            "Enforce dry-run-only manual publishing",
            "if: ${{ github.event_name == 'workflow_dispatch' }}",
            "if: ${{ github.event_name == 'push' }}",
            "candidate_sha=\"$(git rev-parse HEAD)\"",
            "cargo xtask release-integrity",
        ]),
        PatternCheck::new("GitHub Actions verifier mocked tests", &verifier_tests).required(&[
            "test_pull_request_files_are_paginated",
            "test_runs_and_jobs_are_paginated",
            "test_successes_from_different_runs_cannot_be_combined",
            "test_incomplete_skipped_missing_and_stale_evidence_is_rejected",
            "test_annotated_tag_is_peeled",
            "test_http_failure_does_not_expose_token",
        ]),
    ]);
}

#[test]
fn gpu_validation_workflow_is_self_hosted_and_explicit() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new(".github/workflows/gpu-validation.yml")
                .named("GPU validation workflow")
                .required(&[
                    "workflow_dispatch",
                    "run-timed-benchmarks",
                    "run-metal-validation",
                    "self-hosted",
                    "metal",
                    "cuda",
                    "J2K_REQUIRE_CUDA_OXIDE_BUILD",
                    "J2K_REQUIRE_METAL_RUNTIME",
                    "cargo xtask release-metal",
                    "cargo test -p j2k-jpeg-cuda",
                    "cargo test -p j2k-cuda",
                ]),
            FilePatternCheck::new("CONTRIBUTING.md")
                .named("contributor GPU validation policy")
                .required(&[
                    "The GPU validation workflow is intentionally `workflow_dispatch` only.",
                    "`CUDA API compatibility on x86_64`",
                    "`Metal validation on Apple Silicon`",
                ])
                .normalized_required(&[
                    "It does not run automatically on `pull_request` or `push`",
                    "successful manual `gpu-validation.yml` dispatch for the PR head SHA before merge",
                    "The normal CI `gpu-path-policy` job checks the PR diff",
                    "queries `gpu-validation.yml` runs by head SHA",
                    "Hosted macOS CI runs `cargo xtask metal-compile`",
                    "The self-hosted Metal job runs `cargo xtask release-metal`",
                    "fails on skipped runtime tests or a missing Metal device",
                    "Do not add `pull_request` or `push` triggers to `gpu-validation.yml` without an explicit policy decision.",
                ]),
        ],
    );
}

#[test]
fn cuda_gpu_validation_job_stays_cuda_focused() {
    let root = repo_root();
    let workflow_path = root.join(".github/workflows/gpu-validation.yml");
    let workflow = fs::read_to_string(&workflow_path).expect("read GPU validation workflow");
    let cuda_job = workflow_job(&workflow, "cuda-x86_64-compatibility");

    let forbidden_j2k_metal_compare_bench =
        ["cargo bench -p ", "j2k-metal", " --bench compare --no-run"].concat();
    let forbidden = [
        forbidden_j2k_metal_compare_bench.as_str(),
        "cargo bench -p j2k-jpeg --no-run",
        "cargo test -p j2k-jpeg-metal",
        "cargo test -p j2k-metal",
    ];
    assert_pattern_checks(&[PatternCheck::new("CUDA GPU validation job", cuda_job)
        .required(&[
            "runs-on: [self-hosted, Linux, X64, cuda]",
            "J2K_REQUIRE_CUDA_RUNTIME",
            "J2K_REQUIRE_CUDA_OXIDE_BUILD",
            "J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE",
            "J2K_GPU_BENCH_DIM",
            "J2K_GPU_BENCH_BATCH",
            "J2K_GPU_BENCH_BATCH_DIM",
            "uname -a",
            "rustc -Vv",
            "cargo -V",
            "nvidia-smi",
            "CUDA runtime validation requires a working CUDA driver",
            "cargo test -p j2k-jpeg-cuda --all-targets --features cuda-runtime",
            "cargo test -p j2k-cuda --all-targets --features cuda-runtime",
            "cargo bench -p j2k-jpeg-cuda --bench device_decode --features cuda-runtime --no-run",
            "cargo bench -p j2k-jpeg-cuda --bench device_decode --features cuda-runtime -- --noplot",
        ])
        .forbidden(&forbidden)]);
}

#[test]
fn ci_hosted_metal_job_is_compile_only() {
    let root = repo_root();
    let workflow =
        fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read CI workflow");
    let metal_job = workflow_job(&workflow, "metal-compile");

    assert_pattern_checks(&[PatternCheck::new("hosted Metal compile job", metal_job)
        .required(&[
            "name: Metal compile and pure tests",
            "runs-on: macos-latest",
            "components: clippy",
            "cargo xtask metal-compile",
        ])
        .forbidden(&[
            "cargo xtask release-metal",
            "J2K_REQUIRE_METAL_RUNTIME",
            "self-hosted",
        ])]);
}

#[test]
fn metal_gpu_validation_job_fails_closed_and_stays_metal_focused() {
    let root = repo_root();
    let workflow_path = root.join(".github/workflows/gpu-validation.yml");
    let workflow = fs::read_to_string(&workflow_path).expect("read GPU validation workflow");
    let metal_job = workflow_job(&workflow, "metal-apple-silicon");

    assert_pattern_checks(&[PatternCheck::new("Metal GPU validation job", metal_job)
        .required(&[
            "runs-on: [self-hosted, macOS, ARM64, metal]",
            "J2K_REQUIRE_METAL_RUNTIME: \"1\"",
            "RUST_TEST_THREADS: \"1\"",
            "Run fail-closed Metal release validation",
            "cargo xtask release-metal",
            "cargo bench -p j2k-jpeg-metal --no-run",
        ])
        .forbidden(&[
            "nvidia-smi",
            "nvcc --version",
            "cargo test -p j2k-jpeg-cuda",
            "cargo test -p j2k-cuda",
            "J2K_REQUIRE_CUDA_RUNTIME",
            "executed-count floor",
            "passed=$(echo",
            "cargo test -p j2k-jpeg-metal",
            "cargo test -p j2k-metal",
            "cargo test -p j2k-transcode-metal",
        ])]);
}

#[test]
fn metal_xtask_owns_complete_compile_and_runtime_policy() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("xtask/src/metal.rs")
            .named("Metal xtask policy")
            .required(&[
                "METAL_COMPILE_PACKAGES",
                "j2k-metal-support",
                "j2k-jpeg-metal",
                "j2k-metal",
                "j2k-transcode-metal",
                "J2K public facade",
                "J2K_REQUIRE_METAL_RUNTIME",
                "RUST_TEST_THREADS",
                "J2K_GPU_TEST_SKIPPED",
                "J2K_METAL_REQUIRED_IGNORED_TESTS",
                "validate_required_ignored_inventory",
                "validate_exact_ignored_run",
                "passed != J2K_METAL_REQUIRED_IGNORED_TESTS.len()",
                "metal-compile requires J2K_REQUIRE_METAL_RUNTIME to be unset",
                "refusing to report Metal success without the required platform",
            ])
            .forbidden(&[
                "skipping Metal release tests",
                "J2K_RUN_HOSTED_J2K_METAL_RUNTIME_TESTS",
            ])],
    );
}

#[test]
fn cuda_runtime_build_script_does_not_use_nvcc() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("crates/j2k-cuda-runtime/build.rs")
            .named("CUDA runtime build script")
            .required(&[
                ".env_remove(\"RUSTC\")",
                ".env_remove(\"RUSTC_WRAPPER\")",
                ".env_remove(\"RUSTC_WORKSPACE_WRAPPER\")",
            ])
            .forbidden(&["NVCC", "nvcc"])],
    );
}

#[test]
fn cuda_oxide_shared_strict_build_gate_is_wired_and_documented() {
    let per_family_gates = [
        "J2K_REQUIRE_CUDA_OXIDE_COPY_U8",
        "J2K_REQUIRE_CUDA_OXIDE_J2K_ENCODE",
        "J2K_REQUIRE_CUDA_OXIDE_J2K_DECODE_STORE",
        "J2K_REQUIRE_CUDA_OXIDE_J2K_DEQUANTIZE",
        "J2K_REQUIRE_CUDA_OXIDE_J2K_IDWT",
        "J2K_REQUIRE_CUDA_OXIDE_HTJ2K_DECODE",
        "J2K_REQUIRE_CUDA_OXIDE_HTJ2K_ENCODE",
        "J2K_REQUIRE_CUDA_OXIDE_TRANSCODE",
        "J2K_REQUIRE_CUDA_OXIDE_JPEG_DECODE",
        "J2K_REQUIRE_CUDA_OXIDE_JPEG_ENCODE",
    ];
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-cuda-runtime/build.rs")
                .named("CUDA runtime shared strict build gate")
                .required(&["J2K_REQUIRE_CUDA_OXIDE_BUILD"])
                .forbidden(&per_family_gates),
            FilePatternCheck::new("docs/env-vars.md")
                .named("env docs shared strict CUDA Oxide build gate")
                .required(&["J2K_REQUIRE_CUDA_OXIDE_BUILD"])
                .forbidden(&per_family_gates),
            FilePatternCheck::new("crates/j2k-cuda/README.md")
                .required(CUDA_OXIDE_STRICT_BUILD_DOC_PATTERNS),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/README.md")
                .required(CUDA_OXIDE_STRICT_BUILD_DOC_PATTERNS),
            FilePatternCheck::new("crates/j2k-transcode-cuda/README.md")
                .required(CUDA_OXIDE_STRICT_BUILD_DOC_PATTERNS),
            FilePatternCheck::new("docs/cuda-jpeg2000-rust/index.html")
                .required(CUDA_OXIDE_STRICT_BUILD_DOC_PATTERNS),
        ],
    );
}

#[test]
fn cuda_decode_profile_workflow_exports_rca_artifacts() {
    let root = repo_root();
    let workflow_path = root.join(".github/workflows/gpu-validation.yml");
    let workflow = fs::read_to_string(&workflow_path).expect("read GPU validation workflow");
    let cuda_job = workflow_job(&workflow, "cuda-x86_64-compatibility");

    assert_pattern_checks(&[PatternCheck::new(
        "CUDA decode profile workflow/job",
        &format!("{workflow}\n{cuda_job}"),
    )
    .required(&[
        "run-cuda-htj2k-decode-profile",
        "CUDA HTJ2K decode RCA profile",
        "J2K_REQUIRE_CUDA_BENCH: \"1\"",
        "J2K_PROFILE_STAGES: summary",
        "J2K_CUDA_TRACE: ${{ github.workspace }}/target/cuda_htj2k_decode_trace.json",
        "/proc/sys/kernel/perf_event_paranoid",
        "cargo install samply --version 0.13.1 --locked",
        "samply record --save-only -o target/cuda_htj2k_decode_samply.json.gz",
        "target/cuda_htj2k_decode_samply_status.txt",
        "samply_status=blocked",
        "passwordless sudo",
        "--features cuda-runtime,cuda-profiling",
        "2>&1 | tee target/cuda_htj2k_decode_profile.log",
        "cuda-htj2k-decode-rca-profile",
        "target/cuda_htj2k_decode_profile.log",
        "target/cuda_htj2k_decode_trace.json",
        "target/cuda_htj2k_decode_samply.json.gz",
        "target/criterion",
    ])]);
}

#[test]
fn nvidia_baseline_workflow_is_retired() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new(".github/workflows/gpu-validation.yml")
                .named("GPU validation retired NVIDIA baseline")
                .forbidden(&[
                    "run-nvidia-baseline",
                    "--bin transcode_compare",
                    "tests/nvidia-baseline/scripts/assert_transcode_perf.py",
                    "--bin decode_compare",
                    "nvidia-baseline-comparison",
                ]),
        ],
    );
}

#[test]
fn nvidia_codec_comparator_is_historical_only() {
    let root = repo_root();
    let needles = [
        "tests/nvidia-baseline",
        "j2k-nvidia-baseline",
        "nvjpeg2000",
        "nvjpeg2k",
        "nvidia-baseline",
        "run-nvidia-baseline",
    ];
    let mut violations = Vec::new();

    for path in repo_text_files(root) {
        let rel = path.strip_prefix(root).unwrap_or(&path);
        let rel_s = format!("./{}", rel.display()).replace('\\', "/");
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for (line_idx, line) in source.lines().enumerate() {
            let lower = line.to_ascii_lowercase();
            if !needles.iter().any(|needle| lower.contains(needle)) {
                continue;
            }
            let allowed =
                rel_s == "./docs/benchmark-evidence.md" || is_repo_lint_test_source(root, &path);
            if !allowed {
                violations.push(format!("{}:{}:{}", rel_s, line_idx + 1, line));
            }
        }
    }

    let evidence = fs::read_to_string(root.join("docs/benchmark-evidence.md"))
        .expect("read benchmark evidence");
    assert_pattern_checks(&[PatternCheck::new(
        "benchmark evidence historical NVIDIA comparator record",
        &evidence,
    )
    .required(&["Final NVIDIA comparator capture"])]);
    assert!(
        violations.is_empty(),
        "NVIDIA codec comparator references must be historical docs only:\n{}",
        violations.join("\n")
    );
}
