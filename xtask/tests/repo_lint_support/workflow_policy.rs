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
fn cuda_runtime_resolves_codec_math_from_packaged_build_metadata() {
    let root = repo_root();
    let codec_math_manifest = fs::read_to_string(root.join("crates/j2k-codec-math/Cargo.toml"))
        .expect("read codec-math manifest");
    let codec_math_build = fs::read_to_string(root.join("crates/j2k-codec-math/build.rs"))
        .expect("read codec-math build script");
    let runtime_manifest = fs::read_to_string(root.join("crates/j2k-cuda-runtime/Cargo.toml"))
        .expect("read CUDA runtime manifest");
    let runtime_build = fs::read_to_string(root.join("crates/j2k-cuda-runtime/build.rs"))
        .expect("read CUDA runtime build script");

    assert!(codec_math_manifest.contains("links = \"j2k_codec_math\""));
    assert!(codec_math_build.contains("cargo::metadata=manifest_dir="));
    assert!(runtime_manifest.contains("[build-dependencies]"));
    assert!(runtime_build.contains("DEP_J2K_CODEC_MATH_MANIFEST_DIR"));
    assert!(runtime_build.contains("codec_math_crate_path(&manifest_dir)?"));
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
fn deny_bincode_advisory_ignore_and_license_exceptions_are_scoped() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("deny.toml")
            .named("deny.toml burn dependency policy")
            .required(&[
                "RUSTSEC-2025-0141",
                "Review-by: 2027-01-31",
                "https://rustsec.org/advisories/RUSTSEC-2025-0141.html",
                "https://github.com/tracel-ai/burn/tree/v0.21.0/crates/burn-core",
                "{ crate = \"colored@3.1.1\", allow = [\"MPL-2.0\"] }",
                "{ crate = \"hexf-parse@0.2.1\", allow = [\"CC0-1.0\"] }",
                "{ crate = \"option-ext@0.2.0\", allow = [\"MPL-2.0\"] }",
                "{ crate = \"xxhash-rust@0.8.16\", allow = [\"BSL-1.0\"] }",
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
            "def verify_repository_origin(",
            "def require_private_vulnerability_reporting(",
            "/private-vulnerability-reporting",
            "def require_github_release_absent(",
            "def verify_candidate_evidence(",
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
#[expect(
    clippy::too_many_lines,
    reason = "release evidence remains fail-closed by checking the complete workflow matrix together"
)]
fn release_candidate_and_publish_evidence_are_fail_closed() {
    let root = repo_root();
    let ci = fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read CI workflow");
    let secret_scan_workflow = fs::read_to_string(root.join(".github/workflows/secret-scan.yml"))
        .expect("read secret scan workflow");
    let publish = fs::read_to_string(root.join(".github/workflows/publish.yml"))
        .expect("read publish workflow");
    let aggregate = workflow_job(&ci, "release-candidate");
    let strict_clippy = workflow_job(&ci, "clippy-strict");
    let diff_check = workflow_job(&ci, "diff-check");
    let secret_scan = workflow_job(&ci, "secret-scan");
    let codec_math_codegen = workflow_job(&ci, "codec-math-codegen");
    let public_support_final = workflow_job(&ci, "public-support-final");
    let machete = workflow_job(&ci, "machete");
    let repo_lint = workflow_job(&ci, "repo-lint");
    let preflight = workflow_job(&publish, "preflight");
    let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask main");
    let release_status = fs::read_to_string(root.join("xtask/src/release_status.rs"))
        .expect("read release-status task");
    let verifier_tests =
        fs::read_to_string(root.join("scripts/tests/test_github_actions_verify.py"))
            .expect("read GitHub Actions verifier tests");
    let verifier = fs::read_to_string(root.join("scripts/github_actions_verify.py"))
        .expect("read GitHub Actions verifier");
    let crates_io_version = fs::read_to_string(root.join("scripts/crates_io_version.py"))
        .expect("read crates.io version verifier");
    let crates_io_version_tests =
        fs::read_to_string(root.join("scripts/tests/test_crates_io_version.py"))
            .expect("read crates.io version verifier tests");

    assert_pattern_checks(&[
        PatternCheck::new("release candidate aggregate", aggregate).required(&[
            "name: Release candidate aggregate",
            "if: ${{ always() }}",
            "github-actions-verifier",
            "gpu-path-policy",
            "fmt",
            "diff-check",
            "secret-scan",
            "clippy",
            "clippy-strict",
            "panic-surface",
            "clone-audit",
            "comparator-parity",
            "semver",
            "docs",
            "stable-api",
            "codec-math-codegen",
            "public-support-final",
            "machete",
            "repo-lint",
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
        PatternCheck::new("CI authoritative strict Clippy gate", strict_clippy).required(&[
            "runs-on: ubuntu-latest",
            "components: clippy",
            "cargo xtask clippy-strict",
        ]),
        PatternCheck::new("CI submitted-delta whitespace gate", diff_check).required(&[
            "fetch-depth: 0",
            "BASE_SHA: ${{ github.event.pull_request.base.sha || github.event.before }}",
            "git diff --check \"${BASE_SHA}...${GITHUB_SHA}\"",
            "git show --check --format=fuller \"${GITHUB_SHA}\"",
        ]),
        PatternCheck::new("CI exact-SHA secret scan gate", secret_scan).required(&[
            "name: Secret scan",
            "uses: ./.github/workflows/secret-scan.yml",
        ]),
        PatternCheck::new("reusable pinned secret scan", &secret_scan_workflow).required(&[
            "workflow_call:",
            "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5",
            "GITLEAKS_VERSION: \"8.30.1\"",
            "GITLEAKS_SHA256: \"551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb\"",
            "./gitleaks detect --source . --redact --verbose",
        ]),
        PatternCheck::new("CI codec-math freshness gate", codec_math_codegen).required(&[
            "runs-on: ubuntu-latest",
            "cargo xtask codec-math-codegen",
        ]),
        PatternCheck::new("CI final public-support gate", public_support_final).required(&[
            "runs-on: ubuntu-latest",
            "cargo xtask public-support --final",
        ]),
        PatternCheck::new("CI unused-dependency gate", machete).required(&[
            "cargo-machete@0.9.2",
            "cargo xtask machete",
        ]),
        PatternCheck::new("CI normal and strict repository policy gate", repo_lint).required(&[
            "runs-on: macos-latest",
            "toolchain: nightly-2026-06-28",
            "targets: aarch64-apple-darwin",
            "cargo-public-api@0.52.0",
            "cargo install cargo-public-api --version 0.52.0 --locked",
            "Run normal and strict repository policy",
            "cargo xtask repo-lint --strict",
        ]),
        PatternCheck::new("xtask release-status dispatch", &xtask).required(&[
            "mod release_status;",
            "\"release-status\" => release_status::release_status(env::args().skip(2))",
            "release-status verify one frozen SHA's CI aggregate and both GPU jobs",
        ]),
        PatternCheck::new("read-only exact-SHA release-status handoff", &release_status)
            .required(&[
                "verify-candidate",
                "--candidate-sha",
                "--repository",
                "GITHUB_REPOSITORY",
                "remote.origin.url",
                "GH_TOKEN",
                "GITHUB_TOKEN",
                "Release candidate aggregate",
                "CUDA API compatibility on x86_64",
                "Metal validation on Apple Silicon",
            ])
            .forbidden(&["verify-release", "--tag"]),
        PatternCheck::new("candidate private-reporting prerequisite", &verifier).required(&[
            "def verify_candidate_evidence(",
            "require_private_vulnerability_reporting(api)",
            "api.get_json(\"/private-vulnerability-reporting\")",
        ]),
        PatternCheck::new("publish workflow exact-SHA policy", &publish)
            .required(&[
                "actions: read",
                "CRATES_IO_ALLOW_PUBLISHED_RERUN: ${{ vars.CRATES_IO_ALLOW_PUBLISHED_RERUN || 'false' }}",
                "DRY_RUN_ONLY: ${{ github.event_name == 'workflow_dispatch' }}",
                "Verify annotated tag and exact-SHA release evidence",
                "scripts/github_actions_verify.py verify-release",
                "--origin-url \"${origin_url}\"",
                "--server-url \"${GITHUB_SERVER_URL}\"",
                "--ci-workflow ci.yml",
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
            "origin_url=\"$(git remote get-url origin)\"",
            "cargo xtask release-integrity",
            "Verify final publish metadata",
            "cargo xtask release-integrity --publish",
            "Verify canonical tag and current registry prefix",
            "scripts/publish-crate.sh --preflight-all",
            "Package and checksum the complete release manifest",
            "python3 scripts/publish_release.py preflight",
        ]),
        PatternCheck::new("GitHub Actions verifier mocked tests", &verifier_tests).required(&[
            "test_pull_request_files_are_paginated",
            "test_runs_and_jobs_are_paginated",
            "test_successes_from_different_runs_cannot_be_combined",
            "test_incomplete_skipped_missing_and_stale_evidence_is_rejected",
            "test_annotated_tag_is_peeled",
            "test_post_freeze_candidate_verifies_ci_and_gpu_without_a_tag",
            "test_post_freeze_candidate_requires_private_vulnerability_reporting",
            "test_verify_candidate_parser_smoke",
            "test_verify_release_parser_requires_origin_context",
            "test_repository_origin_is_exact_and_credential_free",
            "test_private_vulnerability_reporting_must_be_enabled",
            "test_existing_github_release_in_any_state_is_rejected",
            "test_missing_token_fails_closed",
            "test_http_failure_does_not_expose_token",
            "test_only_http_404_is_optional_absence",
        ]),
        PatternCheck::new("fail-closed crates.io version verifier", &crates_io_version)
            .required(&[
                "error.code == 404",
                "VersionState.AVAILABLE",
                "VersionState.PUBLISHED",
                "dependency-order prefix",
                "could not classify every crates.io target version",
            ]),
        PatternCheck::new("crates.io version verifier mocked tests", &crates_io_version_tests)
            .required(&[
                "test_http_404_is_available",
                "test_exact_http_200_payload_is_published_without_authorization",
                "test_non_404_http_failures_are_not_treated_as_available",
                "test_initial_publish_rejects_any_existing_version_after_checking_all",
                "test_idempotent_retry_accepts_only_a_published_prefix",
                "test_non_prefix_publication_is_rejected",
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
                    "cargo xtask release-cuda",
                    "cargo xtask release-metal",
                    "cargo run -p xtask --features adoption -- adoption-materialize",
                    "cargo run -p xtask --features adoption -- adoption-curate",
                    "cargo run -p xtask --features adoption -- adoption-benchmark",
                    "cargo run -p xtask --features adoption -- adoption-report",
                ])
                .forbidden(&["cargo xtask adoption-"]),
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
        "cargo test -p j2k-cuda-runtime",
        "cargo test -p j2k-jpeg-cuda",
        "cargo test -p j2k-cuda",
        "cargo test -p j2k-transcode-cuda",
        "cargo clippy -p j2k-cuda-runtime",
        "cargo clippy -p j2k-jpeg-cuda",
        "cargo clippy -p j2k-cuda",
        "cargo clippy -p j2k-transcode-cuda",
        "executed-count floor",
        "passed=$(echo",
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
            "Run fail-closed CUDA release validation",
            "cargo xtask release-cuda",
            "cargo bench -p j2k-jpeg-cuda --bench device_decode --features cuda-runtime --no-run",
            "cargo bench -p j2k-jpeg-cuda --bench device_decode --features cuda-runtime -- --noplot",
        ])
        .forbidden(&forbidden)]);
    assert_eq!(
        cuda_job.matches("cargo xtask release-cuda").count(),
        1,
        "CUDA GPU validation must delegate exactly once to the repository-owned release gate"
    );
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
    let root = repo_root();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("xtask/src/metal.rs")
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
                    "METAL_OPTIONAL_IGNORED_TESTS",
                    "fn run_metal_compile()",
                    "fn run_release_metal()",
                    "validate_required_ignored_inventory",
                    "validate_exact_ignored_run",
                    "passed != J2K_METAL_REQUIRED_IGNORED_TESTS.len()",
                    "metal-compile requires J2K_REQUIRE_METAL_RUNTIME to be unset",
                    "refusing to report Metal success without the required platform",
                    "mod tests;",
                ])
                .forbidden(&[
                    "skipping Metal release tests",
                    "J2K_RUN_HOSTED_J2K_METAL_RUNTIME_TESTS",
                ]),
            FilePatternCheck::new("xtask/src/metal/tests.rs")
                .named("Metal xtask command regressions")
                .required(&[
                    "fn metal_commands_execute_complete_hermetic_compile_and_release_plans()",
                    "use_test_cargo_program",
                    "RecordingProgram",
                    "--ignored --list",
                    "--ignored --show-output",
                ]),
        ],
    );

    for (relative, max_lines) in [
        ("xtask/src/metal.rs", 400),
        ("xtask/src/metal/tests.rs", 200),
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its focused {max_lines}-line ownership ratchet"
        );
    }
}

#[test]
fn release_status_command_boundary_has_a_hermetic_regression() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("xtask/tests/command_orchestration.rs")
                .named("release-status command regression")
                .required(&[
                    "fn release_status_executes_exact_sha_verification_without_exposing_tokens()",
                    "release-status with explicit repository",
                    "release-status with remote-derived repository",
                    "--token-env GH_TOKEN",
                    "--token-env GITHUB_TOKEN",
                    "token values reached command log",
                ]),
            FilePatternCheck::new("xtask/tests/command_orchestration/support.rs")
                .named("release-status isolated process boundaries")
                .required(&[
                    "remote.origin.url",
                    "git@example.invalid:frames-sg/j2k.git",
                    "fake Python",
                    "GITHUB_REPOSITORY",
                ]),
        ],
    );
}

#[test]
fn cuda_xtask_owns_complete_compile_and_runtime_policy() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("xtask/src/cuda.rs")
                .named("CUDA xtask policy")
                .required(&[
                    "CUDA_RELEASE_ENV",
                    "J2K_REQUIRE_CUDA_RUNTIME",
                    "J2K_REQUIRE_CUDA_OXIDE_BUILD",
                    "J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE",
                    "RUST_TEST_THREADS",
                    "nvidia-smi",
                    "release-cuda requires Linux x86_64",
                    "j2k-cuda-runtime",
                    "j2k-jpeg-cuda",
                    "j2k-cuda",
                    "j2k-transcode-cuda",
                    "HTJ2K_ENCODE_PARITY_TESTS",
                    "TRANSCODE_PARITY_TESTS",
                    "fn run_release_cuda(",
                    "validate_exact_inventory",
                    "validate_exact_named_run",
                    "J2K_GPU_TEST_SKIPPED",
                    "skipping cuda",
                    "passed zero tests",
                    "was partial",
                    "mod test_support;",
                    "mod tests;",
                ])
                .forbidden(&["minimum_passed", "executed-count floor"]),
            FilePatternCheck::new("xtask/src/cuda/tests.rs")
                .named("CUDA xtask command regressions")
                .required(&[
                    "fn cuda_release_executes_the_complete_hermetic_command_plan()",
                    "fn cuda_device_override_is_nested_transactional_and_fail_closed()",
                    "fn exact_inventory_and_captured_cargo_report_subprocess_failures()",
                    "use_test_cargo_program",
                    "RecordingProgram",
                ]),
            FilePatternCheck::new("xtask/src/cuda/test_support.rs")
                .named("CUDA xtask test process boundary")
                .required(&[
                    "thread_local!",
                    "PhantomData<Rc<()>>",
                    "impl Drop for TestNvidiaSmiProgramGuard",
                ])
                .forbidden(&["Mutex"]),
        ],
    );

    for (relative, max_lines) in [
        ("xtask/src/cuda.rs", 525),
        ("xtask/src/cuda/tests.rs", 325),
        ("xtask/src/cuda/test_support.rs", 75),
    ] {
        let source = fs::read_to_string(repo_root().join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its focused {max_lines}-line ownership ratchet"
        );
    }
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
