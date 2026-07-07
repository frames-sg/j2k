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
    let codeowners = fs::read_to_string(root.join(".github/CODEOWNERS")).expect("read CODEOWNERS");

    assert_pattern_checks(&[
        PatternCheck::new("CI workflow default permissions", &workflow)
            .normalized_required(&["permissions:\n  contents: read"]),
        PatternCheck::new("CI GPU path policy job", gpu_policy).required(&[
            "pull-requests: read",
            "actions: read",
            "cuda_prefixes = (",
            "metal_prefixes = (",
            "shared_gpu_exact_paths = {",
            "requires_cuda = bool(cuda_changes or shared_gpu_changes)",
            "requires_metal = bool(metal_changes or shared_gpu_changes)",
            "crates/j2k-cuda-runtime/",
            "crates/j2k-jpeg-cuda/",
            "crates/j2k-cuda/",
            "crates/j2k-transcode-cuda/",
            "crates/j2k-metal-support/",
            "crates/j2k-jpeg-metal/",
            "crates/j2k-metal/",
            "crates/j2k-transcode-metal/",
            "gpu-validation.yml/runs?head_sha=",
            "/actions/runs/{run.get('id')}/jobs?per_page=100",
            "CUDA API compatibility on x86_64",
            "Metal validation on Apple Silicon",
            "required_jobs - successful_jobs",
            "No GPU path changes detected.",
        ]),
        PatternCheck::new("CODEOWNERS GPU path coverage", &codeowners).required(&[
            ".github/workflows/gpu-validation.yml",
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
                    "Assert Metal runtime gate is set",
                    "cargo test -p j2k-transcode-metal --all-targets",
                    "executed-count floor",
                    "cargo test -p j2k-jpeg-metal",
                    "cargo test -p j2k-metal",
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
fn metal_gpu_validation_job_fails_closed_and_stays_metal_focused() {
    let root = repo_root();
    let workflow_path = root.join(".github/workflows/gpu-validation.yml");
    let workflow = fs::read_to_string(&workflow_path).expect("read GPU validation workflow");
    let metal_job = workflow_job(&workflow, "metal-apple-silicon");

    assert_pattern_checks(&[PatternCheck::new("Metal GPU validation job", metal_job)
        .required(&[
            "runs-on: [self-hosted, macOS, ARM64, metal]",
            "J2K_REQUIRE_METAL_RUNTIME: \"1\"",
            "Assert Metal runtime gate is set",
            "J2K_REQUIRE_METAL_RUNTIME not set",
            "cargo test -p j2k-jpeg-metal --all-targets -- --nocapture",
            "cargo test -p j2k-metal --all-targets -- --nocapture",
            "cargo test -p j2k-transcode-metal --all-targets -- --nocapture",
            "Expected at least 100 executed JPEG Metal tests",
            "Expected at least 150 executed J2K Metal tests",
            "Expected at least 20 executed transcode Metal tests",
            "cargo bench -p j2k-jpeg-metal --no-run",
        ])
        .forbidden(&[
            "nvidia-smi",
            "nvcc --version",
            "cargo test -p j2k-jpeg-cuda",
            "cargo test -p j2k-cuda",
            "J2K_REQUIRE_CUDA_RUNTIME",
        ])]);
}

#[test]
fn cuda_runtime_build_script_does_not_use_nvcc() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("crates/j2k-cuda-runtime/build.rs")
            .named("CUDA runtime build script")
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
