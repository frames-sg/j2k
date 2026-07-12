// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use crate::repo_lint_support::{
    assert_file_pattern_checks, assert_pattern_checks, repo_root, rust_sources, workflow_job,
    xtask_sources, FilePatternCheck, PatternCheck,
};

fn fuzz_target_names(manifest: &Path) -> Vec<String> {
    let text = fs::read_to_string(manifest).expect("read fuzz manifest");
    let mut names = Vec::new();
    let mut in_bin = false;

    for line in text.lines().map(str::trim) {
        if line == "[[bin]]" {
            in_bin = true;
            continue;
        }
        if line.starts_with('[') {
            in_bin = false;
            continue;
        }
        if !in_bin || !line.starts_with("name") {
            continue;
        }
        let Some((_, value)) = line.split_once('=') else {
            continue;
        };
        let name = value.trim().trim_matches('"');
        if !name.is_empty() {
            names.push(name.to_string());
        }
    }

    assert!(
        !names.is_empty(),
        "fuzz manifest {} must declare at least one [[bin]] target",
        manifest.display()
    );
    names
}

#[test]
fn xtask_test_does_not_run_benchmarks_as_tests() {
    let xtask = xtask_sources(repo_root());
    let test_section = xtask
        .split("fn test()")
        .nth(1)
        .and_then(|rest| rest.split("fn doc()").next())
        .expect("xtask test section");

    assert_pattern_checks(&[PatternCheck::new("xtask test function", test_section)
        .required(&["\"--lib\"", "\"--bins\"", "\"--tests\"", "\"--doc\""])
        .forbidden(&["\"--all-targets\""])]);
}

#[test]
fn benchmark_targets_are_not_test_targets() {
    let root = repo_root();
    let mut violations = Vec::new();

    for entry in fs::read_dir(root.join("crates")).expect("read crates dir") {
        let manifest_path = entry.expect("read crate entry").path().join("Cargo.toml");
        if !manifest_path.exists() {
            continue;
        }

        let manifest = fs::read_to_string(&manifest_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", manifest_path.display()));
        for section in manifest.split("[[bench]]").skip(1) {
            let block = section.split("\n[").next().unwrap_or(section);
            let name = block
                .lines()
                .find_map(|line| line.trim().strip_prefix("name = "))
                .map_or("<unnamed>", |value| value.trim_matches('"'));
            if !block.lines().any(|line| line.trim() == "test = false") {
                violations.push(format!(
                    "{}: bench target `{name}` must set `test = false`",
                    manifest_path
                        .strip_prefix(root)
                        .unwrap_or(&manifest_path)
                        .display()
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "benchmark targets must not execute under cargo test --all-targets:\n{}",
        violations.join("\n")
    );
}

#[test]
fn xtask_exposes_nextest_machete_and_strict_clippy_gates() {
    let xtask = xtask_sources(repo_root());
    let workflow =
        fs::read_to_string(repo_root().join(".github/workflows/ci.yml")).expect("read CI workflow");
    let strict_clippy_job = workflow_job(&workflow, "clippy-strict");
    let release_candidate_job = workflow_job(&workflow, "release-candidate");
    let help_section = xtask
        .split("fn print_help()")
        .nth(1)
        .expect("xtask help section");

    assert_pattern_checks(&[
        PatternCheck::new("xtask nextest/machete/strict clippy dispatch", &xtask).required(&[
            "\"nextest\" =>",
            "\"machete\" =>",
            "\"clippy-strict\" =>",
        ]),
        PatternCheck::new("xtask nextest/machete/strict clippy help", help_section).required(&[
            "nextest",
            "machete",
            "clippy-strict",
        ]),
        PatternCheck::new("xtask nextest/machete/strict clippy gates", &xtask).required(&[
            "\"nextest\"",
            "\"run\"",
            "\"cargo-machete\"",
            "\"--with-metadata\"",
            "\"--no-deps\"",
            "\"clippy::pedantic\"",
            "\"clippy::nursery\"",
            "\"j2k-native\"",
            "\"j2k\"",
        ]),
        PatternCheck::new("hosted authoritative strict Clippy gate", strict_clippy_job)
            .required(&["components: clippy", "cargo xtask clippy-strict"]),
        PatternCheck::new(
            "release candidate requires strict Clippy",
            release_candidate_job,
        )
        .required(&["clippy-strict"]),
    ]);
}

#[test]
fn xtask_fuzz_build_checks_every_fuzz_manifest() {
    let root = repo_root();
    let xtask = xtask_sources(root);
    let mut manifests = Vec::new();

    for entry in fs::read_dir(root.join("crates")).expect("read crates dir") {
        let entry = entry.expect("read crate entry");
        let manifest = entry.path().join("fuzz/Cargo.toml");
        if manifest.exists() {
            manifests.push(manifest);
        }
    }
    manifests.sort();
    assert!(
        !manifests.is_empty(),
        "repository must keep fuzz targets under crates/*/fuzz"
    );

    for manifest in manifests {
        let relative_path = manifest
            .strip_prefix(root)
            .expect("fuzz manifest under repo root");
        let relative = relative_path
            .iter()
            .map(|part| part.to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        assert!(
            xtask.contains(&relative),
            "xtask fuzz-build must check {relative}"
        );

        for target in fuzz_target_names(&manifest) {
            let crate_dir = relative
                .strip_suffix("/fuzz/Cargo.toml")
                .expect("fuzz manifest suffix");
            let expected = format!("(\"{crate_dir}\", \"{target}\")");
            assert!(
                xtask.contains(&expected),
                "xtask FUZZ_TARGETS must include {expected}"
            );
        }
    }
}

#[test]
fn ci_coverage_job_is_a_required_gate() {
    const INSTALL_ACTION_SHA: &str = "91534edaf9fd796a162759d80d49cdff574bff2c";

    let workflow =
        fs::read_to_string(repo_root().join(".github/workflows/ci.yml")).expect("read CI workflow");
    let coverage_job = workflow_job(&workflow, "coverage");

    let install_action = format!("taiki-e/install-action@{INSTALL_ACTION_SHA}");
    assert_pattern_checks(&[PatternCheck::new("CI coverage job", coverage_job)
        .required(&[
            install_action.as_str(),
            "tool: cargo-llvm-cov@0.8.7",
            "fetch-depth: 0",
            "J2K_COVERAGE_BASE: ${{ github.event_name == 'pull_request' && github.event.pull_request.base.sha || github.event_name == 'push' && github.event.before || 'HEAD^' }}",
            "cargo xtask coverage host",
            "name: j2k-host-coverage",
            "lcov-host.info",
            "coverage-host-summary.json",
            "if-no-files-found: error",
        ])
        .forbidden(&["taiki-e/install-action@cargo-llvm-cov", "continue-on-error"])]);
}

#[test]
fn coverage_measures_accelerator_host_rust_with_narrow_test_backed_exclusions() {
    let root = repo_root();
    let coverage_dir = root.join("xtask/src/coverage");
    let mut coverage_sources = rust_sources(&coverage_dir);
    coverage_sources.retain(|path| {
        let relative = path
            .strip_prefix(&coverage_dir)
            .unwrap_or_else(|error| panic!("normalize {}: {error}", path.display()));
        !relative
            .components()
            .any(|component| component.as_os_str() == "tests")
            && relative.file_name().is_none_or(|name| name != "tests.rs")
    });
    assert!(
        !coverage_sources.is_empty(),
        "coverage production source inventory"
    );
    coverage_sources.push(root.join("xtask/src/coverage.rs"));
    coverage_sources.sort();
    let coverage = coverage_sources
        .into_iter()
        .map(|path| {
            fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("xtask/src/main.rs")
                .named("coverage command delegation")
                .required(&["coverage::coverage(env::args().skip(2))"])
                .forbidden(&["GPU_COVERAGE_EXCLUSION_REGEX", "--ignore-filename-regex"]),
            FilePatternCheck::new(".gitignore")
                .named("generated coverage evidence")
                .required(&[
                    "lcov-*.info",
                    "coverage-*-summary.json",
                    "coverage-*-regions.json",
                ]),
            FilePatternCheck::new(".github/workflows/ci.yml")
                .named("host coverage artifacts")
                .required(&["coverage-host-regions.json"]),
            FilePatternCheck::new(".github/workflows/gpu-validation.yml")
                .named("accelerator coverage artifacts")
                .required(&["coverage-metal-regions.json", "coverage-cuda-regions.json"]),
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("changed accelerator coverage policy", &coverage)
            .required(&[
                "CHANGED_LINE_THRESHOLD_PERCENT: u64 = 80",
                "Self::Host => !is_accelerator_path(path)",
                "AcceleratorLaneSpec",
                "SHARED_ACCELERATOR_SOURCES",
                "shared_accelerator_packages",
                "validate_shared_accelerator_registry",
                "EvidenceClass::Primary",
                "require_primary_evidence",
                "enclosing_cfg_is_conditional",
                "accelerator host lines",
                "--include-build-script",
                "j2k-changed-line-coverage-v4",
                "head_sha",
                "lane_scope",
                "changed_functions_without_covered_body",
                "changed_deferred_bodies_without_covered_compiler_region",
                "compiler_noninstrumentable_deferred_bodies",
                "coverage-host-regions.json",
                "mixed_test_production_lines",
                "cuda-simt-device-rust",
                "cuda-generated-host-scaffold",
                "cuda-driver-ffi-declarations",
                "metal-embedded-shader-body",
                "generated-codec-math-fragment",
                "vendored-block-ffi-binding",
                "cuda_facade_byte_matches_native_across_matrix_when_required",
                "runtime_raii_primitives_smoke_when_required",
                "metal_kernels_are_wired_to_host_pipelines",
                "full_classic_grayscale_decode_to_metal_matches_host_decode",
                "lcov-host.info",
                "lcov-metal.info",
                "lcov-cuda.info",
            ])
            .forbidden(&["GPU_COVERAGE_EXCLUSION_REGEX"]),
    ]);
}

#[test]
fn self_hosted_accelerator_jobs_publish_distinct_coverage_evidence() {
    let workflow = fs::read_to_string(repo_root().join(".github/workflows/gpu-validation.yml"))
        .expect("read GPU validation workflow");
    let metal_job = workflow_job(&workflow, "metal-apple-silicon");
    let cuda_job = workflow_job(&workflow, "cuda-x86_64-compatibility");

    assert_pattern_checks(&[
        PatternCheck::new("GPU coverage baseline", &workflow).required(&[
            "coverage-base-ref:",
            "default: \"v0.6.2\"",
            "J2K_COVERAGE_BASE: ${{ inputs.coverage-base-ref }}",
        ]),
    ]);

    assert_pattern_checks(&[
        PatternCheck::new("Metal hardware coverage", metal_job)
            .required(&[
                "fetch-depth: 0",
                "tool: cargo-llvm-cov@0.8.7",
                "cargo xtask coverage metal",
                "name: j2k-metal-coverage",
                "lcov-metal.info",
                "coverage-metal-summary.json",
                "if-no-files-found: error",
            ])
            .forbidden(&["continue-on-error"]),
        PatternCheck::new("CUDA hardware coverage", cuda_job)
            .required(&[
                "fetch-depth: 0",
                "tool: cargo-llvm-cov@0.8.7",
                "cargo xtask coverage cuda",
                "name: j2k-cuda-coverage",
                "lcov-cuda.info",
                "coverage-cuda-summary.json",
                "if-no-files-found: error",
            ])
            .forbidden(&["continue-on-error"]),
    ]);
}

#[test]
fn xtask_adoption_stack_is_feature_gated() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("xtask/Cargo.toml")
                .named("xtask adoption feature and optional dependencies")
                .required(&[
                    "image = { workspace = true, optional = true }",
                    "j2k = { path = \"../crates/j2k\", optional = true }",
                    "j2k-compare = { path = \"../crates/j2k-compare\", optional = true }",
                    "j2k-native = { path = \"../crates/j2k-native\", optional = true }",
                    "j2k-profile = { path = \"../crates/j2k-profile\", optional = true }",
                    "j2k-test-support = { path = \"../crates/j2k-test-support\", optional = true }",
                ])
                .normalized_required(&["[features]\ndefault = []", "adoption = ["]),
            FilePatternCheck::new("xtask/src/main.rs")
                .named("xtask adoption module cfg gates")
                .normalized_required(&[
                    "#[cfg(feature = \"adoption\")]\nmod adoption_benchmark;",
                    "#[cfg(feature = \"adoption\")]\nmod adoption_corpus;",
                    "#[cfg(feature = \"adoption\")]\nmod adoption_curate;",
                    "#[cfg(feature = \"adoption\")]\nmod adoption_manifest;",
                    "#[cfg(feature = \"adoption\")]\nmod adoption_materialize;",
                    "#[cfg(feature = \"adoption\")]\nmod adoption_report;",
                    "#[cfg(not(feature = \"adoption\"))]\nmod adoption_disabled;",
                ]),
            FilePatternCheck::new("xtask/src/adoption_disabled.rs")
                .named("disabled adoption command shim")
                .required(&[
                    "\"adoption-benchmark\"",
                    "\"adoption-curate\"",
                    "\"adoption-manifest\"",
                    "\"adoption-materialize\"",
                    "\"adoption-report\"",
                ]),
        ],
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the adoption benchmark module map is a single fail-closed ownership policy"
)]
fn adoption_benchmark_lives_in_focused_modules() {
    let root = repo_root();
    let coordinator = fs::read_to_string(root.join("xtask/src/adoption_benchmark.rs"))
        .expect("read adoption benchmark coordinator");
    let options = fs::read_to_string(root.join("xtask/src/adoption_benchmark/options.rs"))
        .expect("read adoption benchmark options");
    let runner = fs::read_to_string(root.join("xtask/src/adoption_benchmark/runner.rs"))
        .expect("read adoption benchmark runner");
    let existing = fs::read_to_string(root.join("xtask/src/adoption_benchmark/existing.rs"))
        .expect("read adoption benchmark existing-result discovery");
    let parsing = fs::read_to_string(root.join("xtask/src/adoption_benchmark/parsing.rs"))
        .expect("read adoption benchmark parsing");
    let summary = fs::read_to_string(root.join("xtask/src/adoption_benchmark/summary.rs"))
        .expect("read adoption benchmark summary");
    let readme = fs::read_to_string(root.join("xtask/src/adoption_benchmark/readme.rs"))
        .expect("read adoption benchmark README renderer");
    let support = fs::read_to_string(root.join("xtask/src/adoption_benchmark/support.rs"))
        .expect("read adoption benchmark publication/path support");

    for (path, source, max_lines) in [
        ("adoption_benchmark.rs", coordinator.as_str(), 600),
        ("adoption_benchmark/options.rs", options.as_str(), 250),
        ("adoption_benchmark/runner.rs", runner.as_str(), 700),
        ("adoption_benchmark/existing.rs", existing.as_str(), 200),
        ("adoption_benchmark/parsing.rs", parsing.as_str(), 700),
        ("adoption_benchmark/summary.rs", summary.as_str(), 300),
        ("adoption_benchmark/readme.rs", readme.as_str(), 300),
        ("adoption_benchmark/support.rs", support.as_str(), 150),
    ] {
        assert!(
            source.lines().count() < max_lines,
            "xtask/src/{path} must stay below its focused-module line-count ratchet of {max_lines}"
        );
        assert!(
            !source.contains("use super::*") && !source.contains("include!("),
            "xtask/src/{path} must keep explicit real-Rust module boundaries"
        );
    }

    assert_pattern_checks(&[
        PatternCheck::new("adoption benchmark coordinator wiring", &coordinator).required(&[
            "mod existing;",
            "mod options;",
            "mod parsing;",
            "mod readme;",
            "mod runner;",
            "mod summary;",
            "mod support;",
            "pub(crate) fn adoption_benchmark(",
        ]),
        PatternCheck::new(
            "adoption benchmark coordinator ownership exclusions",
            &coordinator,
        )
        .forbidden(&[
            "const SCRUBBED_BENCH_ENV_VARS",
            "struct AdoptionStep",
            "fn run_cpu_encode_compare(",
            "fn existing_steps(",
            "fn write_summary(",
            "fn criterion_summary_json(",
            "fn write_readme(",
            "fn enforce_publication_gate(",
            "impl AdoptionBenchmarkOptions",
        ]),
        PatternCheck::new("adoption benchmark option ownership", &options).required(&[
            "pub(crate) struct AdoptionBenchmarkOptions",
            "pub(super) fn parse(",
            "pub(super) fn help_text(",
            "pub(super) fn parse_batch_size_list(",
        ]),
        PatternCheck::new("adoption benchmark runner ownership", &runner).required(&[
            "pub(super) const SCRUBBED_BENCH_ENV_VARS",
            "pub(super) fn run_cpu_encode_compare(",
            "pub(super) fn run_cuda_htj2k_decode(",
            "pub(super) fn run_metal_transcode_benchmark(",
            "pub(super) fn run_logged_owned(",
            "pub(super) fn display_command(",
        ]),
        PatternCheck::new("adoption benchmark existing-result ownership", &existing).required(&[
            "pub(super) fn existing_steps(",
            "pub(super) fn existing_ran_step(",
        ]),
        PatternCheck::new("adoption benchmark parser ownership", &parsing).required(&[
            "pub(super) fn criterion_summary_json(",
            "pub(super) fn read_metal_decode_summary(",
            "pub(super) fn read_metal_encode_summary(",
            "pub(super) fn read_metal_transcode_summary(",
            "pub(super) fn read_tsv_metadata(",
        ]),
        PatternCheck::new("adoption benchmark summary/model ownership", &summary).required(&[
            "pub(super) struct AdoptionStep",
            "pub(super) enum StepStatus",
            "pub(super) fn write_summary(",
            "pub(super) fn step_json(",
        ]),
        PatternCheck::new("adoption benchmark README ownership", &readme)
            .required(&["pub(super) fn write_readme("]),
        PatternCheck::new("adoption benchmark publication/path ownership", &support).required(&[
            "pub(super) fn enforce_publication_gate(",
            "pub(super) fn benchmark_env_path(",
            "pub(super) fn benchmark_env_path_list(",
            "pub(super) fn canonical_benchmark_path(",
        ]),
    ]);
}
