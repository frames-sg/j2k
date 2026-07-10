// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use super::*;

fn source_before_cfg_test_module<'a>(source: &'a str, relative: &str) -> &'a str {
    source
        .split_once("#[cfg(test)]\nmod tests")
        .unwrap_or_else(|| panic!("{relative} must end production code before its test module"))
        .0
}

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
fn codec_api_guide_covers_public_surfaces() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("README.md")
            .named("README.md codec API surface")
            .required(&[
                "Codec contracts",
                "decode_region_scaled_into",
                "decode_rows",
                "TileBatchDecode",
                "BackendRequest::Auto",
                "BackendRequest::Metal",
                "BackendRequest::Cuda",
                "DeviceSurface",
                "ScratchPool",
                "DecoderContext",
            ])],
    );
}

#[test]
fn ci_workflow_keeps_docs_and_benchmark_compile_gates() {
    let xtask = xtask_sources(repo_root());
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new(".github/workflows/ci.yml")
            .named("CI workflow docs and benchmark compile gates")
            .required(&[
                "cargo xtask doc",
                "cargo xtask stable-api",
                "cargo xtask bench-build",
                "cargo-public-api@0.52.0",
                "macos-latest",
            ])
            .forbidden(&["macos-13"])],
    );
    assert_pattern_checks(&[
        PatternCheck::new("xtask benchmark compile gate", &xtask).required(&[
            "\"doc\"",
            "\"--workspace\"",
            "\"--all-features\"",
            "\"--no-deps\"",
            "\"j2k-jpeg-metal\"",
            "\"j2k-metal\"",
            "\"--no-run\"",
        ]),
    ]);
}

#[test]
fn ci_workflow_runs_semver_checks_for_stable_library_crates() {
    let root = repo_root();
    let workflow =
        fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read CI workflow");
    let xtask = xtask_sources(root);
    let semver_impl =
        fs::read_to_string(root.join("xtask/src/semver.rs")).expect("read semver xtask");
    let stable_api_doc =
        fs::read_to_string(root.join("docs/stable-api-1.0.md")).expect("read stable API policy");
    let semver_job = workflow_job(&workflow, "semver");

    assert_pattern_checks(&[
        PatternCheck::new("CI semver job", semver_job)
            .required(&[
                "cargo install cargo-semver-checks --version 0.48.0 --locked",
                "cargo xtask semver",
                "runs-on: macos-latest",
                "toolchain: \"1.96\"",
            ])
            .forbidden(&["release-type: minor"]),
        PatternCheck::new("xtask semver policy", &semver_impl)
            .required(&[
                "CARGO_SEMVER_CHECKS_VERSION: &str = \"0.48.0\"",
                "SEMVER_BASELINE_VERSION: &str = \"0.6.2\"",
                "SEMVER_BASELINE_TAG: &str = \"v0.6.2\"",
                "const SEMVER_NEW_PACKAGES: &[&str] = &[\"j2k-codec-math\"]",
                "computed_release_type",
                "release_type.as_str()",
                "--baseline-version",
                "--write-report",
                "API_DIFF_REPORT",
                "API_REVIEW_CONFIG",
                "validate_reviews",
                "is stale",
                "unwrap_or_else(|_| \"1.96\".to_string())",
            ])
            .forbidden(&[
                "skipping semver baseline for unpublished package",
                "crates_io_package_exists",
            ]),
    ]);

    let stable = const_array_values(&xtask, "STABLE_SEMVER_PACKAGES")
        .into_iter()
        .collect::<BTreeSet<_>>();
    let baseline = const_array_values(&semver_impl, "SEMVER_BASELINE_PACKAGES")
        .into_iter()
        .collect::<BTreeSet<_>>();
    let new = ["j2k-codec-math".to_string()]
        .into_iter()
        .collect::<BTreeSet<_>>();
    assert!(
        baseline.is_disjoint(&new),
        "semver baseline/new package lists overlap"
    );
    assert_eq!(
        baseline.union(&new).cloned().collect::<BTreeSet<_>>(),
        stable,
        "semver baseline/new package lists must partition STABLE_SEMVER_PACKAGES"
    );
    assert_eq!(
        new,
        ["j2k-codec-math".to_string()].into_iter().collect(),
        "new 0.7 library packages must be listed explicitly"
    );

    for package in stable {
        assert!(
            stable_api_doc.contains(&format!("`{package}`")),
            "docs/stable-api-1.0.md must list semver-gated package `{package}`"
        );
    }
    assert_pattern_checks(&[PatternCheck::new(
        "docs/stable-api-1.0.md prerequisites",
        &stable_api_doc,
    )
    .required(&["cargo-public-api", "0.52.0", "macOS"])]);

    let package = "j2k-cli";
    assert_pattern_checks(&[PatternCheck::new(
        "CI semver job experimental package exclusion",
        semver_job,
    )
    .forbidden(&[package])]);
}

#[test]
fn reviewed_api_diff_artifacts_are_consistent() {
    let root = repo_root();
    let report = fs::read_to_string(root.join("engineering/reviewed-public-api-diff-0.7.0.md"))
        .expect("read reviewed API diff report");
    let config_source = fs::read_to_string(root.join("engineering/public-api-review-0.7.0.yml"))
        .expect("read public API review config");
    let config: serde_yaml_ng::Value =
        serde_yaml_ng::from_str(&config_source).expect("parse public API review config");
    let config = config.as_mapping().expect("review config root mapping");
    assert_eq!(
        config
            .get("candidate_version")
            .and_then(serde_yaml_ng::Value::as_str),
        Some("0.7.0"),
        "review config candidate version must match the 0.7 report"
    );
    let reviews = config
        .get("reviews")
        .and_then(serde_yaml_ng::Value::as_mapping)
        .expect("review config reviews mapping");

    let mut summary = std::collections::BTreeMap::new();
    for line in report.lines().filter(|line| line.starts_with("| `")) {
        let cells = line.split('|').map(str::trim).collect::<Vec<_>>();
        assert_eq!(cells.len(), 10, "malformed API diff summary row: {line}");
        let package = cells[1].trim_matches('`').to_string();
        let removed = cells[6]
            .parse::<usize>()
            .unwrap_or_else(|error| panic!("invalid removed count for {package}: {error}"));
        let removed_fingerprint = cells[7].trim_matches('`').to_string();
        let added_fingerprint = cells[8].trim_matches('`').to_string();
        assert!(
            summary
                .insert(package, (removed, removed_fingerprint, added_fingerprint))
                .is_none(),
            "duplicate package in API diff summary"
        );
    }
    assert_eq!(summary.len(), 17, "API diff must list every stable library");

    for (package, (removed, removed_fingerprint, _)) in &summary {
        if *removed == 0 {
            continue;
        }
        let review = reviews
            .get(package.as_str())
            .and_then(serde_yaml_ng::Value::as_mapping)
            .unwrap_or_else(|| panic!("{package} removals lack a review entry"));
        assert_eq!(
            review
                .get("removed_fingerprint")
                .and_then(serde_yaml_ng::Value::as_str),
            Some(removed_fingerprint.as_str()),
            "{package} removed fingerprint review is stale"
        );
        let rationale = review
            .get("rationale")
            .and_then(serde_yaml_ng::Value::as_str)
            .unwrap_or_else(|| panic!("{package} review lacks a rationale"));
        assert!(
            rationale.trim().len() >= 20,
            "{package} review rationale is too short"
        );
    }
    for (package, review) in reviews {
        let package = package.as_str().expect("review package name");
        let (_, removed_fingerprint, added_fingerprint) = summary
            .get(package)
            .unwrap_or_else(|| panic!("review config contains unknown package {package}"));
        let review = review.as_mapping().expect("package review mapping");
        if let Some(reviewed) = review
            .get("removed_fingerprint")
            .and_then(serde_yaml_ng::Value::as_str)
        {
            assert_eq!(reviewed, removed_fingerprint, "{package} removal review");
        }
        if let Some(reviewed) = review
            .get("added_fingerprint")
            .and_then(serde_yaml_ng::Value::as_str)
        {
            assert_eq!(reviewed, added_fingerprint, "{package} addition review");
        }
    }

    assert_pattern_checks(&[
        PatternCheck::new("reviewed API diff SAFE changes", &report).required(&[
            "j2k_jpeg_metal::Surface::as_bytes(&self) -> alloc::borrow::Cow<'_, [u8]>",
            "j2k_metal::MetalEncodedJ2k::codestream_bytes(&self) -> core::result::Result<alloc::vec::Vec<u8>",
            "j2k_metal::Surface::as_bytes(&self) -> alloc::borrow::Cow<'_, [u8]>",
            "j2k_metal_support::checked_buffer_contents_slice",
            "j2k_metal_support::checked_buffer_contents_slice_mut",
            "j2k_metal_support::checked_buffer_fill_bytes",
            "j2k_metal_support::checked_buffer_read_vec",
            "j2k_metal_support::checked_buffer_write",
            "j2k_metal_support::MetalSupportError::BufferZeroSizedType",
            "j2k_metal_support::MetalSupportError::BufferReadbackAllocation",
            "j2k_native::DecodeErrorClass",
            "j2k_native::DecodeError::classify",
        ]),
        PatternCheck::new("reviewed API diff new-package classification", &report).required(&[
            "## New packages without a 0.6.2 registry baseline",
            "`j2k-codec-math` `0.7.0`: 72 public API items",
        ]),
    ]);
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
fn large_test_files_stay_split_by_axis() {
    let root = repo_root();
    for (relative, max_lines) in [
        ("crates/j2k-metal/src/encode/tests.rs", 150),
        ("crates/j2k-metal/src/encode/tests/batch.rs", 450),
        ("crates/j2k-metal/src/encode/tests/dwt_parity.rs", 250),
        ("crates/j2k-metal/src/encode/tests/kernels.rs", 1_300),
        ("crates/j2k-metal/src/encode/tests/layouts.rs", 250),
        ("crates/j2k-metal/src/encode/tests/resident_batches.rs", 725),
        ("crates/j2k-metal/src/encode/tests/resident_buffers.rs", 950),
        ("crates/j2k-metal/src/encode/tests/routing.rs", 850),
        ("crates/j2k-metal/src/encode/tests/stage_validation.rs", 650),
        ("crates/j2k-metal/src/encode/tests/stats_inflight.rs", 950),
        ("crates/j2k-jpeg-metal/src/tests.rs", 2_400),
        ("crates/j2k-jpeg-metal/src/tests/reusable_output.rs", 250),
        ("crates/j2k-jpeg-metal/src/tests/textures.rs", 2_400),
        ("crates/j2k-cuda-runtime/src/tests.rs", 2_300),
        ("crates/j2k-cuda-runtime/src/tests/pipeline.rs", 2_400),
        ("crates/j2k-jpeg/tests/decode_into.rs", 2_000),
        ("crates/j2k-jpeg/tests/decode_into/lossless.rs", 1_600),
        ("crates/j2k-jpeg/tests/decode_into/color.rs", 900),
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below the split test-file line-count ratchet"
        );
    }
}

#[test]
fn repo_lint_policy_support_files_stay_split_by_axis() {
    let root = repo_root();
    for (relative, max_lines) in [
        (
            "xtask/tests/repo_lint_support/docs_and_workflows_policy.rs",
            2_750,
        ),
        (
            "xtask/tests/repo_lint_support/encode_compare_structure_policy.rs",
            250,
        ),
        (
            "xtask/tests/repo_lint_support/fixture_compare_structure_policy.rs",
            250,
        ),
        ("xtask/tests/repo_lint_support/gpu_adapter_policy.rs", 1_800),
        (
            "xtask/tests/repo_lint_support/gpu_device_structure_policy.rs",
            500,
        ),
        (
            "xtask/tests/repo_lint_support/jpeg_decoder_structure_policy.rs",
            425,
        ),
        (
            "xtask/tests/repo_lint_support/jpeg_metal_resource_safety_policy.rs",
            350,
        ),
        (
            "xtask/tests/repo_lint_support/metal_compute_structure_policy.rs",
            550,
        ),
        ("xtask/tests/repo_lint_support/transcode_api_policy.rs", 125),
        (
            "xtask/tests/repo_lint_support/transcode_structure_policy.rs",
            375,
        ),
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below the split repo-lint policy line-count ratchet"
        );
    }
}

#[test]
fn xtask_exposes_nextest_machete_and_strict_clippy_gates() {
    let xtask = xtask_sources(repo_root());
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
            "tool: cargo-llvm-cov",
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
    let coverage = [
        "xtask/src/coverage.rs",
        "xtask/src/coverage/model.rs",
        "xtask/src/coverage/lane.rs",
        "xtask/src/coverage/parsing.rs",
        "xtask/src/coverage/evaluation.rs",
        "xtask/src/coverage/summary.rs",
        "xtask/src/coverage/exclusion_policy.rs",
    ]
    .into_iter()
    .map(|relative| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
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
                .required(&["lcov-*.info", "coverage-*-summary.json"]),
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("changed accelerator coverage policy", &coverage)
            .required(&[
                "CHANGED_LINE_THRESHOLD_PERCENT: u64 = 80",
                "Self::Host => is_production_rust(path)",
                "accelerator host lines",
                "cuda-simt-device-rust",
                "cuda-generated-host-scaffold",
                "cuda-driver-ffi-declarations",
                "metal-embedded-shader-body",
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
                "tool: cargo-llvm-cov",
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
                "tool: cargo-llvm-cov",
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
fn decode_capability_correctness_regressions_are_guarded() {
    let root = repo_root();
    let native_codestream = read_source_files(
        root,
        &[
            "crates/j2k-native/src/j2c/codestream/header.rs",
            "crates/j2k-native/src/j2c/codestream/tests.rs",
        ],
    );
    assert_pattern_checks(&[PatternCheck::new(
        "target-resolution shrink-factor arithmetic",
        &native_codestream,
    )
    .required(&[
        ".checked_shl(u32::from(skipped_resolution_levels))",
        ".checked_mul(resolution_shrink_factor)",
        "size_data.checked_image_width()?;",
        "size_data.checked_image_height()?;",
        "checked_image_dimensions_reject_shrink_factor_overflow",
    ])]);
    assert_file_pattern_checks(
        root,
        &[FilePatternCheck::new("crates/j2k-jpeg/tests/inspect.rs")
            .named("JPEG progressive inspect/decode agreement fixtures")
            .required(&[
                "fn inspect_and_decoder_info_agree_for_progressive_fixtures()",
                "progressive_8x8_jpeg()",
                "progressive_12bit_grayscale_8x8_jpeg()",
                "progressive_12bit_rgb_8x8_jpeg()",
                "assert_eq!(decoder.info(), &inspected, \"{label}\");",
            ])],
    );
}

#[test]
fn panic_hotspot_production_paths_do_not_use_unwrap_or_expect() {
    let root = repo_root();
    for relative in [
        "crates/j2k-cuda/src/encode.rs",
        "crates/j2k-jpeg/src/entropy/block.rs",
        "crates/j2k-jpeg/src/entropy/huffman.rs",
        "crates/j2k-jpeg/src/entropy/progressive.rs",
        "crates/j2k-jpeg/src/entropy/sequential.rs",
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|err| panic!("read {relative}: {err}"));
        let production = source_before_cfg_test_module(&source, relative);
        for forbidden in [".unwrap(", ".expect("] {
            assert!(
                !production.contains(forbidden),
                "{relative} production path must not use panic-on-error `{forbidden}`"
            );
        }
    }
}

#[test]
fn too_many_arguments_suppressions_stay_below_current_ratchet() {
    let root = repo_root();
    let mut sources = rust_sources(&root.join("crates"));
    sources.extend(rust_sources(&root.join("xtask")));
    assert!(
        !sources.is_empty(),
        "too_many_arguments ratchet must scan Rust sources"
    );

    let mut count = 0usize;
    for path in sources {
        let source = fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {path:?}: {err}"));
        count += count_too_many_arguments_suppressions(&source);
    }

    assert!(
        count <= 4,
        "too_many_arguments suppression count must not exceed the current ratchet: found {count}, expected <= 4"
    );
}

fn count_too_many_arguments_suppressions(source: &str) -> usize {
    let bytes = source.as_bytes();
    let mut count = 0usize;
    let mut index = 0usize;

    while index < bytes.len() {
        if bytes[index] != b'#' {
            index += 1;
            continue;
        }

        let bracket = match bytes.get(index + 1) {
            Some(b'[') => index + 1,
            Some(b'!') if bytes.get(index + 2) == Some(&b'[') => index + 2,
            _ => {
                index += 1;
                continue;
            }
        };

        let mut depth = 0usize;
        let mut end = bracket;
        while end < bytes.len() {
            match bytes[end] {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        end += 1;
                        break;
                    }
                }
                _ => {}
            }
            end += 1;
        }

        let attribute = &source[index..end.min(source.len())];
        if attribute.contains("allow") && attribute.contains("clippy::too_many_arguments") {
            count += 1;
        }
        index = end.max(index + 1);
    }

    count
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

#[test]
fn mq_qe_table_is_shared_by_encoder_and_decoder() {
    let root = repo_root();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-native/src/j2c/mq.rs")
                .named("native MQ table module")
                .required(&[
                    "pub(crate) struct QeData",
                    "pub(crate) static QE_TABLE: [QeData; 47]",
                    "Shared MQ arithmetic-coder probability table",
                ]),
            FilePatternCheck::new("crates/j2k-native/src/j2c/arithmetic_decoder.rs")
                .named("native arithmetic decoder")
                .required(&["use super::mq::QE_TABLE;"])
                .forbidden(&["struct QeData", "static QE_TABLE"]),
            FilePatternCheck::new("crates/j2k-native/src/j2c/arithmetic_encoder.rs")
                .named("native arithmetic encoder")
                .required(&["use super::mq::QE_TABLE;"])
                .forbidden(&["struct QeData", "static QE_TABLE"]),
        ],
    );
}

#[test]
fn component_plane_metadata_accessors_are_shared() {
    let root = repo_root();
    let native_color = fs::read_to_string(root.join("crates/j2k-native/src/color.rs"))
        .expect("read native color module");
    let facade_decode =
        fs::read_to_string(root.join("crates/j2k/src/decode.rs")).expect("read j2k decode");

    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-native/src/lib.rs")
                .named("native component-plane accessor macro")
                .required(&[
                    "#[doc(hidden)]",
                    "#[macro_export]",
                    "macro_rules! __j2k_component_plane_metadata_accessors",
                ]),
            FilePatternCheck::new("crates/j2k/src/decode.rs")
                .named("j2k decode facade")
                .forbidden(&["macro_rules! impl_component_plane_metadata_accessors"]),
        ],
    );
    for (name, source, expected_macro, expected_calls) in [
        (
            "native color",
            native_color.as_str(),
            "crate::__j2k_component_plane_metadata_accessors!();",
            2,
        ),
        (
            "j2k decode facade",
            facade_decode.as_str(),
            "j2k_native::__j2k_component_plane_metadata_accessors!();",
            2,
        ),
    ] {
        assert_eq!(
            source.matches(expected_macro).count(),
            expected_calls,
            "{name} must use the shared component-plane accessor macro"
        );
    }
}

#[test]
fn jpeg_cache_digests_use_shared_fnv1a64_helpers() {
    let root = repo_root();
    let jpeg_context =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/context.rs")).expect("read JPEG context");
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-core/src/lib.rs")
                .named("j2k-core FNV-1a helper macros")
                .required(&[
                    "macro_rules! __j2k_fnv1a64_init",
                    "macro_rules! __j2k_fnv1a64_update",
                    "macro_rules! __j2k_fnv1a64_bytes",
                    "0xcbf2_9ce4_8422_2325_u64",
                    "0x0000_0100_0000_01B3_u64",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg/src/context.rs")
                .named("JPEG context FNV helper use")
                .required(&["j2k_core::__j2k_fnv1a64_bytes!(bytes)"])
                .forbidden(&["FNV_OFFSET", "FNV_PRIME"]),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/src/session.rs")
                .named("JPEG CUDA session FNV helper use")
                .required(&["j2k_core::__j2k_fnv1a64_bytes!(bytes)"])
                .forbidden(&["FNV_OFFSET", "FNV_PRIME"]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/session.rs")
                .named("JPEG Metal session FNV helper use")
                .required(&["j2k_core::__j2k_fnv1a64_bytes!(bytes)"])
                .forbidden(&["FNV_OFFSET", "FNV_PRIME"]),
        ],
    );
    assert!(
        jpeg_context.contains("j2k_core::__j2k_fnv1a64_init!()")
            && jpeg_context.contains("j2k_core::__j2k_fnv1a64_update!(hash, byte)"),
        "JPEG table digest continuations must use the shared FNV-1a init/update helpers"
    );
}

#[test]
fn jpeg_fast_packet_accessors_stay_out_of_public_api() {
    let root = repo_root();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-jpeg/src/adapter/fast_packet.rs")
                .named("shared JPEG fast-packet adapter")
                .required(&[
                    "pub struct JpegFast420PacketV1",
                    "pub struct JpegFast422PacketV1",
                    "pub struct JpegFast444PacketV1",
                ])
                .forbidden(&[
                    "pub trait JpegColorFastPacket",
                    "impl_color_fast_packet_access",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg/src/adapter/mod.rs")
                .named("j2k-jpeg adapter facade")
                .forbidden(&["JpegColorFastPacket"]),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/src/owned_decode.rs")
                .named("CUDA owned decode")
                .required(&[
                    "macro_rules! fast_rgb8_packet_parts",
                    "fn build_cuda_rgb8_plan_data",
                ])
                .forbidden(&[
                    "JpegColorFastPacket",
                    "trait JpegColorFastPacket",
                    "macro_rules! impl_color_fast_packet_access",
                    "macro_rules! cuda_decode_plan",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/compute/fast_packets/descriptors.rs")
                .named("Metal fast packets")
                .required(&[
                    "trait FastSubsampledPacket",
                    "macro_rules! impl_fast_subsampled_packet_accessors",
                ])
                .forbidden(&[
                    "JpegColorFastPacket",
                    "trait JpegColorFastPacket",
                    "macro_rules! impl_color_fast_packet_access",
                ]),
            FilePatternCheck::new("docs/stable-api-1.0.public-api.txt")
                .named("stable API snapshot")
                .forbidden(&[
                    "pub trait j2k_jpeg::adapter::JpegColorFastPacket",
                    "j2k_jpeg::adapter::JpegColorFastPacket::",
                ]),
        ],
    );
}

#[test]
fn ht_code_block_scalar_fallback_lives_in_trait_default() {
    let root = repo_root();
    let backend = fs::read_to_string(root.join("crates/j2k-native/src/backend.rs"))
        .expect("read native backend trait");
    let trait_source = backend
        .split_once("pub trait HtCodeBlockDecoder")
        .expect("native backend trait must define HtCodeBlockDecoder")
        .1;
    assert_contains_all_normalized(
        "HT code-block scalar fallback",
        trait_source,
        &[
            "fn decode_code_block(\n        &mut self,",
            "decode_ht_code_block_scalar(job, output)",
        ],
    );

    for relative in [
        "crates/j2k-metal/src/classic.rs",
        "crates/j2k-metal/src/idwt.rs",
        "crates/j2k-metal/src/mct.rs",
        "crates/j2k-metal/src/store.rs",
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source.as_str(), |(prod, _)| prod);
        assert!(
            !production.contains("fn decode_code_block("),
            "{relative} must inherit the shared scalar HT fallback instead of restating it"
        );
    }

    let composite =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/code_block_decoder.rs"))
            .expect("read Metal composite code-block decoder");
    assert!(
        composite.contains("self.ht.decode_code_block(job, output)")
            && !composite.contains("decode_ht_code_block_scalar("),
        "Metal composite decoder must delegate HT blocks instead of copying the scalar fallback"
    );
}

#[test]
fn packet_progression_ordering_uses_shared_packetization_contract() {
    let root = repo_root();
    let packet_contract =
        fs::read_to_string(root.join("crates/j2k-types/src/lib.rs")).expect("read j2k-types");
    let native_encode =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/packet_plan.rs"))
            .expect("read native encode packet plan");
    let native_encode_options =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/options.rs"))
            .expect("read native encode options");
    let native_codestream =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/codestream_write.rs"))
            .expect("read native codestream writer");
    let metal_packet_plan =
        fs::read_to_string(root.join("crates/j2k-metal/src/encode/packet_plan.rs"))
            .expect("read Metal packet plan");
    let metal_capacity =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/encode_capacity.rs"))
            .expect("read Metal encode capacity");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-types packet progression contract", &packet_contract).required(&[
            "pub fn sort_packet_descriptors_for_progression",
            "pub const fn codestream_order_code",
        ]),
        PatternCheck::new("native encode packetization option", &native_encode_options)
            .required(&["packetization_order(self)"]),
        PatternCheck::new("native encode packet descriptor ordering", &native_encode)
            .required(&["crate::sort_packet_descriptors_for_progression("])
            .forbidden(&["fn sort_packet_descriptors_for_progression("]),
        PatternCheck::new(
            "native codestream progression byte mapping",
            &native_codestream,
        )
        .required(&[".codestream_order_code()"]),
        PatternCheck::new("Metal packet plan progression ordering", &metal_packet_plan)
            .required(&["sort_packet_descriptors_for_progression("])
            .forbidden(&["fn sort_lossless_device_packet_descriptors("]),
        PatternCheck::new("Metal capacity progression byte mapping", &metal_capacity)
            .required(&[".codestream_order_code()"]),
    ]);
}

#[test]
fn idwt_required_region_propagation_uses_shared_native_helper() {
    let root = repo_root();
    let direct_roi = fs::read_to_string(root.join("crates/j2k-native/src/direct_roi.rs"))
        .expect("read native direct ROI helper");
    let native_roi = fs::read_to_string(root.join("crates/j2k-native/src/j2c/roi.rs"))
        .expect("read native ROI planner");
    let cuda_direct = fs::read_to_string(root.join("crates/j2k-cuda/src/direct_plan.rs"))
        .expect("read CUDA direct plan");
    let metal_direct_roi =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
            .expect("read Metal direct ROI");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-native direct ROI IDWT helper", &direct_roi).required(&[
            "pub fn idwt_required_input_windows",
            "pub fn idwt_required_input_window_for_rects",
            "pub const fn idwt_required_output_margin",
            "pub struct J2kRequiredBandRegion",
        ]),
    ]);

    for (relative, source) in [
        ("crates/j2k-cuda/src/direct_plan.rs", &cuda_direct),
        (
            "crates/j2k-metal/src/compute/direct_roi.rs",
            &metal_direct_roi,
        ),
    ] {
        assert_pattern_checks(&[PatternCheck::new(relative, source)
            .required(&["idwt_required_input_windows(", "expanded_within_band("])
            .forbidden(&[
                "fn idwt_input_required_region(",
                "fn idwt_required_output_margin(",
                "struct RequiredBandRegion",
                "struct BandRequiredRegion",
                "j2k_native::idwt_band_index",
            ])]);
    }

    assert_pattern_checks(&[
        PatternCheck::new("native ROI IDWT window arithmetic", &native_roi)
            .required(&[
                "idwt_required_input_window_for_rects(",
                "crate::idwt_required_output_margin(",
            ])
            .forbidden(&["fn idwt_input_required_region(", "fn idwt_band_index("]),
    ]);
}

#[test]
fn metal_direct_required_region_retain_uses_shared_job_helper() {
    let root = repo_root();
    let compute = fs::read_to_string(root.join("crates/j2k-metal/src/compute.rs"))
        .expect("read Metal compute root");
    let direct_roi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
        .expect("read Metal direct ROI");

    assert_pattern_checks(&[
        PatternCheck::new("Metal compute ROI module", &compute).required(&["mod direct_roi;"]),
        PatternCheck::new("Metal direct required-region retain helper", &direct_roi).required(&[
            "trait RequiredRegionJob",
            "impl RequiredRegionJob for J2kClassicCleanupBatchJob",
            "impl RequiredRegionJob for J2kHtCleanupBatchJob",
            "fn retain_jobs_for_required_region<J: RequiredRegionJob>",
            "retain_jobs_for_required_region(jobs, required);",
        ]),
    ]);
    assert_eq!(
        direct_roi.matches("jobs.retain(|job|").count(),
        1,
        "Metal direct classic/HT required-region retain must have one shared retain body"
    );
}

#[test]
fn metal_direct_sub_band_group_scan_uses_shared_helper() {
    let root = repo_root();
    let compute = fs::read_to_string(root.join("crates/j2k-metal/src/compute.rs"))
        .expect("read Metal compute root");
    let direct_prepare =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_prepare.rs"))
            .expect("read Metal direct prepare");

    assert_pattern_checks(&[
        PatternCheck::new("Metal compute prepare module", &compute)
            .required(&["mod direct_prepare;"]),
        PatternCheck::new("Metal direct sub-band grouping helper", &direct_prepare).required(&[
            "fn prepare_sub_band_groups<'a, SubBand: 'a, Group>",
            "prepare_sub_band_groups(",
            "PreparedDirectGrayscaleStep::ClassicSubBand(sub_band)",
            "PreparedDirectGrayscaleStep::HtSubBand(sub_band)",
            "prepare_classic_sub_band_group,",
            "prepare_ht_sub_band_group,",
        ]),
    ]);
    assert_eq!(
        direct_prepare
            .matches("while step_idx < steps.len()")
            .count(),
        1,
        "Metal direct classic/HT sub-band grouping must have one shared scan loop"
    );
}

#[test]
fn metal_hybrid_region_scaled_cache_uses_shared_scope() {
    let root = repo_root();
    let hybrid =
        fs::read_to_string(root.join("crates/j2k-metal/src/hybrid.rs")).expect("read hybrid");

    assert_pattern_checks(&[
        PatternCheck::new("Metal hybrid region-scaled cache scope", &hybrid)
            .required(&[
                "enum RegionScaledColorPlanCache",
                "Uncached",
                "Global",
                "Session(&'a crate::MetalBackendSession)",
                "fn build_region_scaled_direct_plan_with_cache(",
                "fn build_region_scaled_direct_color_plan_cached_with_cache(",
                "RegionScaledColorPlanCache::Uncached",
                "RegionScaledColorPlanCache::Session(session)",
            ])
            .forbidden(&["fn build_region_scaled_direct_color_plan_cached_with_session("]),
    ]);
    assert_eq!(
        hybrid.matches("match fmt {").count(),
        1,
        "Metal hybrid direct region-scaled format dispatch must stay single-sourced"
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "wavelet constant provenance is checked across every backend in one policy matrix"
)]
fn wavelet_and_idct_constants_use_codec_math_sources() {
    let root = repo_root();
    let codec_math = fs::read_to_string(root.join("crates/j2k-codec-math/src/lib.rs"))
        .expect("read j2k-codec-math lib");
    let metal_shader_source =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/shader_source.rs"))
            .expect("read j2k-metal shader source");
    let metal_forward_transform =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/forward_transform.rs"))
            .expect("read j2k-metal forward transform");
    let forward_dwt_shader = fs::read_to_string(root.join("crates/j2k-metal/src/fdwt.metal"))
        .expect("read j2k-metal fdwt shader");
    let inverse_dwt_shader = fs::read_to_string(root.join("crates/j2k-metal/src/idwt.metal"))
        .expect("read j2k-metal idwt shader");
    let transcode_metal =
        fs::read_to_string(root.join("crates/j2k-transcode-metal/src/metal/runtime.rs"))
            .expect("read transcode Metal runtime");
    let metal_transcode_dct97 =
        fs::read_to_string(root.join("crates/j2k-transcode-metal/src/dct97.metal"))
            .expect("read transcode Metal dct97 shader");
    let cpu_transcode_dct97 = fs::read_to_string(root.join("crates/j2k-transcode/src/dct97_2d.rs"))
        .expect("read transcode CPU dct97 module");
    let cuda_transcode = read_source_files(
        root,
        &[
            "crates/j2k-cuda-runtime/src/cuda_oxide_transcode/simt/src/main.rs",
            "crates/j2k-cuda-runtime/src/cuda_oxide_transcode/simt/src/constants.rs",
        ],
    );

    assert_file_pattern_checks(
        root,
        &[FilePatternCheck::new("crates/j2k-metal/Cargo.toml")
            .named("j2k-metal codec math dependency")
            .required(&["j2k-codec-math"])],
    );
    assert_pattern_checks(&[PatternCheck::new(
        "j2k-codec-math generated Metal DWT97 constants",
        &codec_math,
    )
    .required(&[
        "pub const DWT97_CONSTANTS_METAL",
        "include_str!(\"../generated/dwt97_constants.metal\")",
    ])]);
    let generated_idx = metal_shader_source
        .find("j2k_codec_math::generated::DWT97_CONSTANTS_METAL")
        .expect("j2k-metal shader source must splice generated DWT97 constants");
    let idwt_idx = metal_shader_source
        .find("../idwt.metal")
        .expect("j2k-metal shader source must include IDWT shader");
    let fdwt_idx = metal_shader_source
        .find("../fdwt.metal")
        .expect("j2k-metal shader source must include FDWT shader");
    assert!(
        generated_idx < idwt_idx && generated_idx < fdwt_idx,
        "j2k-metal shader source must splice generated DWT97 constants before IDWT/FDWT shaders"
    );
    assert_pattern_checks(&[
        PatternCheck::new(
            "j2k-transcode-metal generated DWT97 constants",
            &transcode_metal,
        )
        .required(&["j2k_codec_math::generated::DWT97_CONSTANTS_METAL"]),
        PatternCheck::new("j2k-transcode CPU DCT97 constants", &cpu_transcode_dct97)
            .required(&[
                "j2k_codec_math::dwt::DWT97_ALPHA_F64",
                "j2k_codec_math::dwt::DWT97_BETA_F64",
                "j2k_codec_math::dwt::DWT97_GAMMA_F64",
                "j2k_codec_math::dwt::DWT97_DELTA_F64",
                "j2k_codec_math::dwt::DWT97_KAPPA_F64",
                "j2k_codec_math::dwt::DWT97_INV_KAPPA_F64",
            ])
            .forbidden(&[
                "-1.586_134_342",
                "-0.052_980_118",
                "0.882_911_075",
                "0.443_506_852",
                "1.230_174_104",
            ]),
        PatternCheck::new("j2k-metal host DWT97 constants", &metal_forward_transform).required(&[
            "j2k_codec_math::dwt::DWT97_ALPHA_F32",
            "j2k_codec_math::dwt::DWT97_BETA_F32",
            "j2k_codec_math::dwt::DWT97_GAMMA_F32",
            "j2k_codec_math::dwt::DWT97_DELTA_F32",
        ]),
    ]);

    for (relative, source) in [
        ("crates/j2k-metal/src/fdwt.metal", &forward_dwt_shader),
        ("crates/j2k-metal/src/idwt.metal", &inverse_dwt_shader),
        (
            "crates/j2k-transcode-metal/src/dct97.metal",
            &metal_transcode_dct97,
        ),
    ] {
        assert!(
            source.contains("CODEC_MATH_DWT97") || source.contains("CODEC_MATH_IDWT97"),
            "{relative} must use generated codec-math DWT constants"
        );
        assert_pattern_checks(&[PatternCheck::new(relative, source).forbidden(&[
            "1.586134",
            "0.052980",
            "0.882911",
            "0.443506",
            "1.230174",
            "J2K_FDWT97_",
            "DCT97_ALPHA",
            "DCT97_BETA",
            "DCT97_GAMMA",
            "DCT97_DELTA",
            "DCT97_KAPPA",
        ])]);
    }

    assert_pattern_checks(&[PatternCheck::new(
        "CUDA Oxide transcode SIMT IDCT constants",
        &cuda_transcode,
    )
    .required(&["use j2k_codec_math::jpeg::idct;", "idct::FIX_0_298631336"])
    .forbidden(&[
        "const CONST_BITS: i32 = 13",
        "const FIX_0_298631336: i32 = 2446",
    ])]);
}

#[test]
fn jp2_box_parsing_is_native_owned_with_facade_adapter_only() {
    let root = repo_root();
    let native_jp2 = fs::read_to_string(root.join("crates/j2k-native/src/jp2/mod.rs"))
        .expect("read native JP2 parser");
    let native_lib =
        fs::read_to_string(root.join("crates/j2k-native/src/lib.rs")).expect("read native lib");
    let facade_boxes = fs::read_to_string(root.join("crates/j2k/src/parse/boxes.rs"))
        .expect("read facade JP2 adapter");

    assert!(
        native_jp2.contains("pub fn inspect_jp2_container")
            && native_jp2.contains("fn parse_jp2_container_with_strict")
            && native_jp2.contains("parse_jp2_container_with_strict(data, settings.strict)?"),
        "j2k-native must own the JP2/JPH container box walk used by native decode"
    );
    assert!(
        native_lib.contains("inspect_jp2_container"),
        "j2k-native must re-export the JP2 container inspection bridge for facade adapters"
    );
    assert!(
        facade_boxes.contains("inspect_jp2_container(input)")
            && !facade_boxes.contains("fn read_box_header")
            && !facade_boxes.contains("fn parse_jp2h")
            && !facade_boxes.contains("fn parse_pclr")
            && !facade_boxes.contains("fn parse_cmap")
            && !facade_boxes.contains("fn parse_cdef")
            && !facade_boxes.contains("fn walk_top_level_boxes"),
        "j2k facade JP2 parsing must be an adapter over j2k-native, not a second box parser"
    );
}

#[test]
fn native_classic_and_ht_parallel_copyback_share_one_helper() {
    let root = repo_root();
    let decode_shell = fs::read_to_string(root.join("crates/j2k-native/src/j2c/decode.rs"))
        .expect("read native J2K decode module");
    let decode_subband =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/decode/subband.rs"))
            .expect("read native J2K subband decode module");
    let decode = format!("{decode_shell}\n{decode_subband}");

    assert_pattern_checks(&[PatternCheck::new(
        "native classic/HT decoded-block copyback",
        decode.as_str(),
    )
    .required(&[
        "trait DecodedSubBandBlock",
        "impl DecodedSubBandBlock for DecodedClassicBlock",
        "impl DecodedSubBandBlock for DecodedHtBlock",
        "fn copy_decoded_blocks_to_sub_band<B: DecodedSubBandBlock>",
        "copy_decoded_blocks_to_sub_band(decoded_blocks, sub_band, storage)",
        "decoded_classic_block_copyback_covers_full_block",
        "decoded_ht_block_copyback_covers_partial_edge_block",
        "decoded_block_copyback_rejects_out_of_bounds_blocks",
    ])]);
    let helper_start = decode
        .find("fn copy_decoded_blocks_to_sub_band<B: DecodedSubBandBlock>")
        .expect("shared decoded-block copyback helper");
    let helper_rest = &decode[helper_start..];
    let helper_end = helper_rest
        .find("fn decode_ht_sub_band_blocks_parallel")
        .expect("end of shared decoded-block copyback helper");
    let helper = &helper_rest[..helper_end];
    assert_eq!(
        helper.matches("let dst_start =").count(),
        1,
        "native decoded-block copyback destination bounds/indexing must have one implementation"
    );
    assert_eq!(
        decode
            .matches(".copy_from_slice(&block.coefficients()")
            .count(),
        1,
        "native decoded-block coefficient row copy must have one implementation"
    );
}

#[test]
fn copied_test_fixture_helpers_live_in_shared_support() {
    let root = repo_root();
    let test_support = fs::read_to_string(root.join("crates/j2k-test-support/src/lib.rs"))
        .expect("read j2k-test-support");
    let compare = fs::read_to_string(root.join("crates/j2k-compare/src/encode_compare/images.rs"))
        .expect("read compare encode image module");
    let dct97 = fs::read_to_string(root.join("crates/j2k-transcode/src/dct97_2d.rs"))
        .expect("read transcode 9/7 DCT module");
    let dct97_test = fs::read_to_string(root.join("crates/j2k-transcode/tests/dct97_2d.rs"))
        .expect("read transcode 9/7 DCT test");
    let dwt_diff =
        fs::read_to_string(root.join("crates/j2k-transcode-test-support/src/dwt_diff.rs"))
            .expect("read shared transcode DWT diff test support");
    let jpeg_batch = fs::read_to_string(root.join("crates/j2k-jpeg/tests/batch.rs"))
        .expect("read JPEG batch tests");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-test-support shared PNM helper", &test_support).required(&[
            "pub fn write_pnm(",
            "pub fn read_pnm_image(",
            "pub struct PnmImage",
            "fn read_pnm_token",
        ]),
        PatternCheck::new("compare encode shared PNM helper use", &compare)
            .required(&[
                "j2k_test_support::write_pnm",
                "j2k_test_support::read_pnm_image",
            ])
            .forbidden(&[
                "struct PnmParser",
                "fn parse_pnm_u32(",
                "fs::File::create(path)",
                "write!(file,",
            ]),
    ]);

    assert_pattern_checks(&[
        PatternCheck::new("9/7 transcode internal diff helper", &dct97)
            .required(&[
                "#[cfg(test)]\nimpl Dwt97TwoDimensional<f64>",
                "pub(crate) fn max_abs_diff(&self, other: &Self) -> f64",
            ])
            .forbidden(&["pub fn max_abs_diff(&self, other: &Self) -> f64"]),
        PatternCheck::new("shared transcode DWT diff helper", &dwt_diff).required(&[
            "pub fn max_abs_diff_53(",
            "pub fn max_abs_diff_97(",
            "fn max_abs_diff_bands(",
        ]),
        PatternCheck::new(
            "9/7 transcode integration test shared diff helper",
            &dct97_test,
        )
        .required(&[
            "use j2k_transcode_test_support::max_abs_diff_97;",
            "max_abs_diff_97(&",
        ])
        .forbidden(&["mod dwt_diff;", "fn max_abs_diff("]),
    ]);

    let ycbcr12_start = jpeg_batch
        .find("fn session_batch_decode_extended12_ycbcr444_matches_single_tile_decode")
        .expect("12-bit YCbCr session batch test section");
    let ycbcr12_end = jpeg_batch
        .find("fn session_batch_decode_12bit_rgba16_matches_single_tile_decode")
        .expect("end of 12-bit YCbCr session batch test section");
    let ycbcr12_section = &jpeg_batch[ycbcr12_start..ycbcr12_end];
    assert_eq!(
        ycbcr12_section
            .matches("assert_session_batch_decode(")
            .count(),
        8,
        "12-bit YCbCr session batch cases must share the batch assertion helper"
    );
    assert!(
        !ycbcr12_section.contains("let mut outputs = vec![vec![0u8; expected.len()]"),
        "12-bit YCbCr session batch cases must not reintroduce duplicated output/job setup"
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "native encode ownership checks are intentionally reviewed as one fail-closed matrix"
)]
fn native_encode_options_and_tile_parts_live_in_focused_modules() {
    let root = repo_root();
    let encode = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode.rs"))
        .expect("read native encode module");
    let options = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/options.rs"))
        .expect("read native encode options module");
    let tile_parts =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/tile_parts.rs"))
            .expect("read native encode tile-part module");
    let precomputed_shell =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed.rs"))
            .expect("read native encode precomputed module");
    let precomputed_accelerator = fs::read_to_string(
        root.join("crates/j2k-native/src/j2c/encode/precomputed/accelerator.rs"),
    )
    .expect("read native precomputed accelerator module");
    let precomputed_api53 =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed/api53.rs"))
            .expect("read native precomputed 5-3 API module");
    let precomputed_api97 =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed/api97.rs"))
            .expect("read native precomputed 9-7 API module");
    let precomputed_batch97 =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed/batch97.rs"))
            .expect("read native precomputed 9-7 batch module");
    let precomputed_packets =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed/packets.rs"))
            .expect("read native precomputed packet module");
    let precomputed_validation =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed/validation.rs"))
            .expect("read native precomputed validation module");
    let precomputed = [
        precomputed_shell.as_str(),
        precomputed_accelerator.as_str(),
        precomputed_api53.as_str(),
        precomputed_api97.as_str(),
        precomputed_batch97.as_str(),
        precomputed_packets.as_str(),
        precomputed_validation.as_str(),
    ]
    .join("\n");
    let packet_plan =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/packet_plan.rs"))
            .expect("read native encode packet-plan module");
    let rate_control =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/rate_control.rs"))
            .expect("read native encode rate-control module");
    let roi_plan = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/roi_plan.rs"))
        .expect("read native encode ROI planning module");
    let samples = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/samples.rs"))
        .expect("read native encode sample helper module");
    let i64_packetize =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/i64_packetize.rs"))
            .expect("read native encode i64 packetization module");
    let single_tile =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/single_tile.rs"))
            .expect("read native encode single-tile module");
    let api_helpers =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/api_helpers.rs"))
            .expect("read native encode public API helper module");
    let subband = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/subband.rs"))
        .expect("read native encode subband preparation module");

    assert!(
        encode.lines().count() < 800,
        "j2c/encode.rs must stay below the post-split line-count ratchet"
    );
    assert!(
        precomputed_api97.lines().count() < 800,
        "j2c/encode/precomputed/api97.rs must stay below the focused-module line-count ratchet"
    );
    assert!(
        precomputed_batch97.lines().count() < 150,
        "j2c/encode/precomputed/batch97.rs must stay below the focused-module line-count ratchet"
    );
    assert!(
        roi_plan.lines().count() < 300,
        "j2c/encode/roi_plan.rs must stay below the ROI planning line-count ratchet"
    );
    assert!(
        subband.lines().count() < 650,
        "j2c/encode/subband.rs must stay below the subband preparation line-count ratchet"
    );

    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode.rs option module shell", &encode).required(&[
            "mod options;",
            "pub use self::options",
            "EncodeOptions",
        ]),
    ]);
    for option_type in [
        "pub struct EncodeOptions",
        "pub struct EncodeComponentPlane",
        "pub struct EncodeTypedComponentPlane",
        "pub struct EncodeRoiRegion",
        "pub enum EncodeProgressionOrder",
    ] {
        assert_pattern_checks(&[
            PatternCheck::new("j2c/encode.rs option type exclusion", &encode)
                .forbidden(&[option_type]),
            PatternCheck::new("j2c/encode/options.rs option type ownership", &options)
                .required(&[option_type]),
        ]);
    }
    for helper in [
        "fn validate_irreversible_quantization_scale",
        "pub(super) fn validate_irreversible_quantization_profile",
        "pub(super) fn precinct_exponents_for_options",
    ] {
        assert_pattern_checks(&[
            PatternCheck::new("j2c/encode.rs option validation helper exclusion", &encode)
                .forbidden(&[helper]),
            PatternCheck::new(
                "j2c/encode/options.rs option validation helper ownership",
                &options,
            )
            .required(&[helper]),
        ]);
    }
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode.rs tile-part module shell", &encode).required(&[
            "mod tile_parts;",
            "write_single_tile_packetized_codestream",
            "validate_packet_header_marker_payloads",
        ]),
    ]);
    for helper in [
        "struct EncodedTilePart",
        "pub(super) fn split_packetized_tile_into_tile_parts",
        "pub(super) fn write_single_tile_packetized_codestream",
        "fn validate_packet_header_marker_payload",
        "pub(super) fn validate_packet_header_marker_payloads",
    ] {
        assert_pattern_checks(&[
            PatternCheck::new("j2c/encode.rs tile-part helper exclusion", &encode)
                .forbidden(&[helper]),
            PatternCheck::new("j2c/encode/tile_parts.rs helper ownership", &tile_parts)
                .required(&[helper]),
        ]);
    }
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode.rs focused module wiring", &encode).required(&[
            "mod precomputed;",
            "pub use self::precomputed::{",
            "mod packet_plan;",
            "mod rate_control;",
            "mod roi_plan;",
            "mod samples;",
            "mod i64_packetize;",
            "mod single_tile;",
            "mod subband;",
        ]),
        PatternCheck::new("precomputed.rs 9-7 batch wiring", &precomputed_shell).required(&[
            "mod batch97;",
            "pub use self::batch97::encode_precomputed_htj2k_97_batch_with_accelerator;",
        ]),
        PatternCheck::new("precomputed/api97.rs batch exclusion", &precomputed_api97)
            .forbidden(&["pub fn encode_precomputed_htj2k_97_batch_with_accelerator("]),
        PatternCheck::new(
            "precomputed/batch97.rs batch ownership",
            &precomputed_batch97,
        )
        .required(&["pub fn encode_precomputed_htj2k_97_batch_with_accelerator("]),
    ]);
    for helper in [
        "struct I64PacketizeRequest",
        "struct I64CodestreamPacketRequest",
        "pub(super) fn encode_i64_component_resolution_packets",
        "pub(super) fn packetize_i64_component_resolution_packets",
    ] {
        assert_pattern_checks(&[
            PatternCheck::new("j2c/encode.rs i64 packetization helper exclusion", &encode)
                .forbidden(&[helper]),
            PatternCheck::new(
                "j2c/encode/i64_packetize.rs helper ownership",
                &i64_packetize,
            )
            .required(&[helper]),
        ]);
    }
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode.rs API helper module shell", &encode).required(&[
            "mod api_helpers;",
            "public_sub_band_type",
            "internal_sub_band_type",
            "deinterleave_to_f32",
            "max_decomposition_levels",
        ]),
    ]);
    assert_pattern_checks(&[
        PatternCheck::new(
            "j2c/encode.rs single-tile implementation exclusion",
            &encode,
        )
        .forbidden(&["fn encode_impl("]),
        PatternCheck::new(
            "j2c/encode/single_tile.rs single-tile implementation",
            &single_tile,
        )
        .required(&["pub(super) fn encode_impl("]),
    ]);
    let precomputed_helpers = [
        "pub fn encode_precomputed_j2k_53",
        "pub fn encode_precomputed_htj2k_97",
        "pub fn encode_preencoded_htj2k_97",
        "pub(in crate::j2c::encode) fn validate_precomputed_dwt97_geometry",
    ];
    assert_pattern_checks(&[
        PatternCheck::new(
            "j2c/encode/precomputed.rs precomputed helpers",
            precomputed.as_str(),
        )
        .required(&precomputed_helpers),
        PatternCheck::new("j2c/encode.rs precomputed helper exclusion", &encode)
            .forbidden(&precomputed_helpers),
        PatternCheck::new(
            "precomputed DWT adapter forwarding macro",
            precomputed.as_str(),
        )
        .required(&["macro_rules! forward_precomputed_encode_stage_hooks"]),
    ]);
    assert_eq!(
        precomputed
            .matches("forward_precomputed_encode_stage_hooks!();")
            .count(),
        2,
        "both precomputed DWT adapters must use the shared forwarding macro"
    );
    for forwarded in [
        "fn dispatch_report(",
        "fn encode_quantize_subband(",
        "fn encode_tier1_code_block(",
        "fn encode_tier1_code_blocks(",
        "fn encode_ht_code_block(",
        "fn encode_ht_code_blocks(",
        "fn prefer_parallel_cpu_code_block_fallback(",
        "fn prefer_parallel_cpu_tile_encode(",
        "fn encode_packetization(",
    ] {
        assert_eq!(
            precomputed.matches(forwarded).count(),
            1,
            "precomputed DWT forwarding hook `{forwarded}` must live once in the shared macro"
        );
    }
    for defaulted in [
        "fn encode_deinterleave(",
        "fn encode_forward_rct(",
        "fn encode_forward_ict(",
        "fn encode_ht_subband(",
        "fn encode_htj2k_tile(",
    ] {
        assert_pattern_checks(&[PatternCheck::new(
            "precomputed DWT defaulted hook exclusions",
            precomputed.as_str(),
        )
        .forbidden(&[defaulted])]);
    }
    let packet_plan_helpers = [
        "pub(super) fn packet_descriptors_for_order",
        "pub(super) fn packetize_resolution_packets_with_options",
        "pub(super) fn ordered_prepared_resolution_packets",
        "pub(super) fn public_packetization_resolutions_from_compact",
    ];
    assert_pattern_checks(&[
        PatternCheck::new(
            "j2c/encode/packet_plan.rs packet-plan helpers",
            &packet_plan,
        )
        .required(&packet_plan_helpers),
        PatternCheck::new("j2c/encode.rs packet-plan helper exclusion", &encode)
            .forbidden(&packet_plan_helpers),
    ]);
    let rate_control_helpers = [
        "pub(super) fn classic_multilayer_code_block_style",
        "pub(super) struct ClassicLayerBudgetAllocator",
        "pub(super) fn assign_classic_segment_layers_by_slope",
        "pub(super) fn assign_ht_segment_layers_by_budget",
    ];
    assert_pattern_checks(&[
        PatternCheck::new(
            "j2c/encode/rate_control.rs rate-control helpers",
            &rate_control,
        )
        .required(&rate_control_helpers),
        PatternCheck::new("j2c/encode.rs rate-control helper exclusion", &encode)
            .forbidden(&rate_control_helpers),
    ]);
    let roi_plan_helpers = [
        "pub(super) struct ComponentRoiEncodePlan",
        "pub(super) struct ComponentRoiEncodeRegion",
        "pub(super) fn component_sampling_for_options",
        "pub(super) fn roi_encode_plans_for_options",
        "fn roi_component_shifts_for_options",
        "fn validate_roi_shift(",
        "pub(super) fn roi_subband_scale",
        "pub(super) fn max_total_bitplanes_for_components",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode/roi_plan.rs ROI planning helpers", &roi_plan)
            .required(&roi_plan_helpers),
        PatternCheck::new("j2c/encode.rs ROI planning helper exclusion", &encode)
            .forbidden(&roi_plan_helpers),
    ]);
    let sample_helpers = [
        "pub(super) fn raw_pixel_bytes_per_sample",
        "pub(super) fn read_le_sample_value",
        "pub(super) fn sign_extend_sample",
        "pub(super) fn native_samples_equal",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode/samples.rs sample helpers", &samples)
            .required(&sample_helpers),
        PatternCheck::new("j2c/encode.rs sample helper exclusion", &encode)
            .forbidden(&sample_helpers),
    ]);
    let subband_helpers = [
        "fn apply_roi_maxshift_encode",
        "fn apply_roi_maxshift_encode_i64",
        "fn roi_region_subband_window",
        "pub(super) fn prepare_subband(",
        "pub(super) struct I64SubbandEncodeSettings",
        "pub(super) fn prepare_subband_i64",
        "pub(super) fn prepare_subband_cpu_quantized",
        "fn code_block_shapes",
        "fn subband_range_bits",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode/subband.rs subband helpers", &subband)
            .required(&subband_helpers),
        PatternCheck::new("j2c/encode.rs subband helper exclusion", &encode)
            .forbidden(&subband_helpers),
    ]);
    let api_helpers_patterns = [
        "pub(super) fn public_sub_band_type",
        "pub(super) fn internal_sub_band_type",
        "pub(super) fn default_public_code_block_style",
        "pub(crate) fn deinterleave_to_f32",
        "pub(super) fn deinterleave_rgb8_unsigned_to_f32",
        "pub(super) fn max_decomposition_levels",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode/api_helpers.rs API helpers", &api_helpers)
            .required(&api_helpers_patterns),
        PatternCheck::new("j2c/encode.rs API helper exclusion", &encode)
            .forbidden(&api_helpers_patterns),
    ]);
}

#[test]
fn j2k_encode_facade_lives_in_focused_modules() {
    let root = repo_root();
    let facade =
        fs::read_to_string(root.join("crates/j2k/src/encode.rs")).expect("read J2K encode facade");
    let contracts = fs::read_to_string(root.join("crates/j2k/src/encode/contracts.rs"))
        .expect("read J2K encode contracts");
    let samples = fs::read_to_string(root.join("crates/j2k/src/encode/samples.rs"))
        .expect("read J2K encode samples");
    let native = fs::read_to_string(root.join("crates/j2k/src/encode/native.rs"))
        .expect("read J2K native encode bridge");
    let routing = fs::read_to_string(root.join("crates/j2k/src/encode/routing.rs"))
        .expect("read J2K encode routing");
    let lossy = fs::read_to_string(root.join("crates/j2k/src/encode/lossy.rs"))
        .expect("read J2K lossy encode targeting");
    let validation = fs::read_to_string(root.join("crates/j2k/src/encode/validation.rs"))
        .expect("read J2K encode validation");

    for (path, source) in [
        ("encode.rs", facade.as_str()),
        ("encode/contracts.rs", contracts.as_str()),
        ("encode/samples.rs", samples.as_str()),
        ("encode/native.rs", native.as_str()),
        ("encode/routing.rs", routing.as_str()),
        ("encode/lossy.rs", lossy.as_str()),
        ("encode/validation.rs", validation.as_str()),
    ] {
        assert!(
            source.lines().count() < 800,
            "crates/j2k/src/{path} must stay below the focused-module line-count ratchet"
        );
        assert!(
            !source.contains("use super::*") && !source.contains("include!("),
            "crates/j2k/src/{path} must keep explicit Rust module boundaries"
        );
    }

    assert_pattern_checks(&[
        PatternCheck::new("J2K encode facade wiring", &facade).required(&[
            "mod contracts;",
            "pub use self::contracts::{",
            "mod samples;",
            "pub use self::samples::{",
            "mod native;",
            "mod routing;",
            "mod lossy;",
            "mod validation;",
            "pub fn encode_j2k_lossless(",
            "pub fn encode_j2k_lossy(",
            "pub fn j2k_lossless_decomposition_levels(",
        ]),
        PatternCheck::new("J2K encode facade ownership exclusions", &facade).forbidden(&[
            "pub enum EncodeBackendPreference",
            "pub struct J2kLosslessSamples",
            "struct RequiredEncodeStages",
            "struct LossyAttempt",
            "fn validate_lossless_roundtrip(",
            "use self::contracts::*",
            "use self::samples::*",
        ]),
        PatternCheck::new("J2K encode contract ownership", &contracts).required(&[
            "pub enum EncodeBackendPreference",
            "pub struct J2kLosslessEncodeOptions",
            "pub enum J2kRateTarget",
            "pub struct J2kLossyEncodeOptions",
            "pub struct EncodedJ2k",
            "pub struct EncodedLossyJ2k",
        ]),
        PatternCheck::new("J2K encode sample ownership", &samples).required(&[
            "pub struct J2kLosslessSamples",
            "pub struct J2kLosslessComponentSamples",
            "pub struct J2kLosslessTypedComponentSamples",
            "pub struct J2kLossySamples",
            "pub(super) fn raw_pixel_bytes_per_sample",
        ]),
        PatternCheck::new("J2K native encode bridge ownership", &native).required(&[
            "pub(super) fn encode_cpu(",
            "pub(super) fn native_roi_regions_for_samples(",
            "pub(super) fn native_lossless_options(",
            "pub(super) fn native_lossy_options(",
        ]),
        PatternCheck::new("J2K encode routing ownership", &routing).required(&[
            "pub(super) fn resolve_accelerated_encode_backend(",
            "pub(super) struct RequiredEncodeStages",
            "pub(super) fn required_encode_stages(",
            "pub(super) fn required_lossy_encode_stages(",
        ]),
        PatternCheck::new("J2K lossy target ownership", &lossy).required(&[
            "pub(super) struct LossyAttempt",
            "pub(super) fn encode_lossy_targeted(",
            "pub(super) fn encode_lossy_to_byte_target(",
            "pub(super) fn encode_lossy_to_psnr_target(",
            "pub(super) fn target_bytes_for_bpp(",
        ]),
        PatternCheck::new("J2K encode validation ownership", &validation).required(&[
            "pub(super) fn validate_lossy_roundtrip(",
            "pub(super) fn validate_lossless_roundtrip(",
            "pub(super) fn validate_lossless_component_roundtrip(",
            "pub(super) fn validate_lossless_typed_component_roundtrip(",
        ]),
    ]);
}

#[test]
fn jpeg_to_htj2k_options_live_in_focused_module() {
    let root = repo_root();
    let transcode = fs::read_to_string(root.join("crates/j2k-transcode/src/jpeg_to_htj2k.rs"))
        .expect("read JPEG-to-HTJ2K transcode module");
    let options =
        fs::read_to_string(root.join("crates/j2k-transcode/src/jpeg_to_htj2k/options.rs"))
            .expect("read JPEG-to-HTJ2K options module");
    let report = fs::read_to_string(root.join("crates/j2k-transcode/src/jpeg_to_htj2k/report.rs"))
        .expect("read JPEG-to-HTJ2K report module");
    let error = fs::read_to_string(root.join("crates/j2k-transcode/src/jpeg_to_htj2k/error.rs"))
        .expect("read JPEG-to-HTJ2K error module");
    let batch = fs::read_to_string(root.join("crates/j2k-transcode/src/jpeg_to_htj2k/batch.rs"))
        .expect("read JPEG-to-HTJ2K batch module");

    assert!(
        transcode.lines().count() < 1_770,
        "jpeg_to_htj2k.rs must stay below the post-split line-count ratchet"
    );

    let option_items = [
        "pub const JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE",
        "pub struct JpegToHtj2kEncodeOptions",
        "pub struct JpegToHtj2kOptions",
        "pub enum JpegToHtj2kCoefficientPath",
        "fn native_progression_order",
    ];
    let report_items = [
        "pub struct BatchTranscodeReport",
        "pub enum TranscodeBatchProfileRequest",
        "pub struct TranscodeTimingReport",
        "pub struct TranscodeReport",
    ];
    let error_items = [
        "pub enum JpegToHtj2kError",
        "pub(super) fn dct53_grid_error",
    ];
    let batch_facade_items = [
        "pub fn jpeg_to_htj2k_batch",
        "pub(super) fn jpeg_tile_batch_to_htj2k_with_scratch",
    ];
    let batch_items = [
        batch_facade_items[0],
        batch_facade_items[1],
        "pub(super) struct IntegerBatchTile",
        "pub(super) fn encode_float97_batch_tile",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("jpeg_to_htj2k options module shell", &transcode)
            .required(&[
                "mod options;",
                "pub use self::options",
                "JpegToHtj2kOptions",
            ])
            .forbidden(&option_items),
        PatternCheck::new("jpeg_to_htj2k option item ownership", &options).required(&option_items),
        PatternCheck::new("jpeg_to_htj2k support module wiring", &transcode).required(&[
            "mod report;",
            "mod error;",
            "mod batch;",
        ]),
        PatternCheck::new("jpeg_to_htj2k report item exclusion", &transcode)
            .forbidden(&report_items),
        PatternCheck::new("jpeg_to_htj2k error item exclusion", &transcode).forbidden(&error_items),
        PatternCheck::new("jpeg_to_htj2k batch item exclusion", &transcode).forbidden(&batch_items),
        PatternCheck::new("jpeg_to_htj2k report item ownership", &report).required(&report_items),
        PatternCheck::new("jpeg_to_htj2k error item ownership", &error).required(&error_items),
        PatternCheck::new("jpeg_to_htj2k batch facade ownership", &batch)
            .required(&batch_facade_items),
    ]);
}

#[test]
fn native_public_contracts_live_in_focused_modules() {
    let root = repo_root();
    let lib =
        fs::read_to_string(root.join("crates/j2k-native/src/lib.rs")).expect("read j2k-native lib");
    let backend = fs::read_to_string(root.join("crates/j2k-native/src/backend.rs"))
        .expect("read j2k-native backend module");
    let color = fs::read_to_string(root.join("crates/j2k-native/src/color.rs"))
        .expect("read j2k-native color module");
    let ht_adapter = fs::read_to_string(root.join("crates/j2k-native/src/ht_adapter.rs"))
        .expect("read j2k-native HT adapter module");
    let roi = fs::read_to_string(root.join("crates/j2k-native/src/roi.rs"))
        .expect("read j2k-native ROI module");
    let types = fs::read_to_string(root.join("crates/j2k-types/src/lib.rs"))
        .expect("read j2k-types module");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-native focused public module wiring", &lib).required(&[
            "mod backend;",
            "mod color;",
            "mod ht_adapter;",
            "mod roi;",
            "pub use backend::{",
            "pub use color::{",
            "pub use ht_adapter::{",
            "pub use roi::idwt_band_index;",
        ]),
    ]);
    assert!(
        lib.lines().count() < 2_260,
        "j2k-native lib.rs must keep shrinking after the test-module split"
    );

    let backend_items = [
        "pub trait HtCodeBlockDecoder",
        "pub struct HtCodeBlockDecodeJob",
        "pub struct J2kCodeBlockDecodeJob",
        "pub struct J2kRect",
    ];
    let color_items = [
        "pub enum ColorSpace",
        "pub struct Bitmap",
        "pub struct RawBitmap",
        "pub struct DecodedNativeComponents",
        "pub(crate) fn resolve_alpha_and_color_space",
        "pub(crate) fn convert_color_space",
        "pub(crate) fn cielab_to_rgb",
    ];
    let roi_items = [
        "pub fn idwt_band_index",
        "pub(crate) fn add_roi_shift_to_bitplanes",
        "pub(crate) fn apply_roi_maxshift_inverse_i64",
        "pub(crate) fn apply_roi_maxshift_inverse_i32",
        "pub(crate) fn validate_roi",
    ];
    let ht_adapter_items = [
        "pub struct HtSigPropBenchmarkState",
        "pub fn prepare_ht_sigprop_benchmark_state",
        "pub fn decode_ht_sigprop_benchmark_state",
        "pub fn ht_vlc_table0",
        "pub fn ht_uvlc_encode_table_bytes",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-types encode-stage accelerator ownership", &types)
            .required(&["pub trait J2kEncodeStageAccelerator"]),
        PatternCheck::new("j2k-native backend accelerator exclusion", &backend)
            .forbidden(&["pub trait J2kEncodeStageAccelerator"]),
        PatternCheck::new("j2k-native backend contract exclusion", &lib).forbidden(&backend_items),
        PatternCheck::new("j2k-native backend contract ownership", &backend)
            .required(&backend_items),
        PatternCheck::new("j2k-native color/output exclusion", &lib).forbidden(&color_items),
        PatternCheck::new("j2k-native color/output ownership", &color).required(&color_items),
        PatternCheck::new("j2k-native ROI helper exclusion", &lib).forbidden(&roi_items),
        PatternCheck::new("j2k-native ROI helper ownership", &roi).required(&roi_items),
        PatternCheck::new("j2k-native HT adapter helper exclusion", &lib)
            .forbidden(&ht_adapter_items),
        PatternCheck::new("j2k-native HT adapter helper ownership", &ht_adapter)
            .required(&ht_adapter_items),
    ]);
}

#[test]
fn native_adapter_exports_are_doc_hidden() {
    let root = repo_root();
    let lib =
        fs::read_to_string(root.join("crates/j2k-native/src/lib.rs")).expect("read j2k-native lib");
    let scalar_encode = fs::read_to_string(root.join("crates/j2k-native/src/scalar/encode.rs"))
        .expect("read j2k-native scalar encode module");
    let scalar_decode =
        fs::read_to_string(root.join("crates/j2k-native/src/scalar/classic_decode.rs"))
            .expect("read j2k-native scalar classic decode module");
    let image = fs::read_to_string(root.join("crates/j2k-native/src/image.rs"))
        .expect("read j2k-native image module");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-native hidden adapter exports", &lib).required(&[
            "#[doc(hidden)]\npub use backend::",
            "#[doc(hidden)]\npub use direct_plan::",
            "#[doc(hidden)]\npub use ht_adapter::",
            "#[doc(hidden)]\npub use j2k_types::",
            "#[doc(hidden)]\npub use scalar::",
        ]),
        PatternCheck::new("j2k-native hidden scalar encode adapters", &scalar_encode)
            .required(&["#[doc(hidden)]\n#[must_use]\npub fn forward_dwt53_reference"]),
        PatternCheck::new("j2k-native hidden scalar decode adapters", &scalar_decode)
            .required(&["#[doc(hidden)]\npub fn decode_j2k_code_block_scalar"]),
        PatternCheck::new("j2k-native image contract ownership", &image)
            .required(&["pub struct DecodeSettings", "pub struct Image"]),
    ]);
}

#[test]
fn jpeg_decoder_upsample_sample_width_twins_use_generic_helpers() {
    let root = repo_root();
    let decoder = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/decoder/extended12.rs",
            "crates/j2k-jpeg/src/decoder/extended12/planes.rs",
            "crates/j2k-jpeg/src/decoder/extended12/upsample.rs",
            "crates/j2k-jpeg/src/decoder/extended12/writers.rs",
            "crates/j2k-jpeg/src/decoder/lossless_helpers.rs",
        ],
    );

    assert_pattern_checks(&[PatternCheck::new(
        "JPEG decoder sample-width upsample helpers",
        &decoder,
    )
    .required(&[
        "trait UpsampleSample",
        "impl UpsampleSample for u8",
        "impl UpsampleSample for u16",
        "fn upsample_h2v1_sample_at<S: UpsampleSample>",
        "fn upsample_h2v2_rows_at<S: UpsampleSample>",
        "upsample_h2v1_sample_at(",
        "upsample_h2v2_rows_at(current, near, output_width, output_x)",
        "upsample_h2v2_u16_rows_at(",
    ])
    .forbidden(&[
        "fn upsample_h2v1_12bit_at",
        "fn upsample_h2v2_12bit_at",
        "3 * u32::from(row[sample])",
        "let colsum = |index: usize| 3 * u32::from(current[index])",
    ])]);
}

#[test]
fn mirrored_twin_unification_record_is_current() {
    let root = repo_root();
    let record = fs::read_to_string(root.join("engineering/mirrored-twin-unification.md"))
        .expect("read mirrored-twin unification record");
    let compute = fs::read_to_string(root.join("crates/j2k-metal/src/compute.rs"))
        .expect("read Metal compute root");
    let direct_prepare =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_prepare.rs"))
            .expect("read Metal direct prepare");
    let direct_roi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
        .expect("read Metal direct ROI");
    let hybrid =
        fs::read_to_string(root.join("crates/j2k-metal/src/hybrid.rs")).expect("read hybrid");
    let decoder = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/decoder/extended12.rs",
            "crates/j2k-jpeg/src/decoder/extended12/planes.rs",
            "crates/j2k-jpeg/src/decoder/extended12/upsample.rs",
            "crates/j2k-jpeg/src/decoder/extended12/writers.rs",
        ],
    );
    let neon = fs::read_to_string(root.join("crates/j2k-jpeg/src/backend/neon.rs"))
        .expect("read JPEG NEON backend");
    let native_idwt = read_source_files(
        root,
        &[
            "crates/j2k-native/src/j2c/idwt.rs",
            "crates/j2k-native/src/j2c/idwt/orchestrate.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("mirrored-twin unification record", &record).required(&[
            "## Unified Families",
            "## Documented Waivers",
            "## Golden Checks",
            "Metal direct required-region retain",
            "Metal direct sub-band group scanning",
            "Metal hybrid region-scaled planning",
            "JPEG sample-width upsample helpers",
            "Extended12 versus Progressive12 JPEG decode is not a safe merge target",
            "NEON `dual` versus `top_only` row-pair kernels are intentionally separate",
            "Native IDWT f32 versus i64 remains separate",
            "cargo test -p j2k-jpeg --test decode_into progressive12_ycbcr420",
            "cargo test -p j2k-jpeg --test neon_hot_paths",
        ]),
        PatternCheck::new("Metal direct twin-unification module shell", &compute)
            .required(&["mod direct_prepare;", "mod direct_roi;"]),
        PatternCheck::new("Metal direct retain unification", &direct_roi)
            .required(&["fn retain_jobs_for_required_region<J: RequiredRegionJob>"]),
        PatternCheck::new(
            "Metal direct sub-band grouping unification",
            &direct_prepare,
        )
        .required(&["fn prepare_sub_band_groups<'a, SubBand: 'a, Group>"]),
        PatternCheck::new("Metal hybrid region-scaled planning unification", &hybrid)
            .required(&["enum RegionScaledColorPlanCache"]),
        PatternCheck::new("JPEG sample-width and extended12 waiver evidence", &decoder).required(
            &[
                "trait UpsampleSample",
                "struct Extended12WriteRegion",
                "fn render_progressive12_color_planes(",
                "fn decode_extended12_color_planes(",
            ],
        ),
        PatternCheck::new("JPEG NEON row-pair waiver evidence", &neon).required(&[
            "unsafe fn fill_rgb_row_pair_from_420_neon(",
            "unsafe fn fill_rgb_row_pair_from_420_neon_top_only(",
        ]),
        PatternCheck::new("native IDWT f32/i64 waiver evidence", &native_idwt)
            .required(&["pub(crate) fn apply(", "fn apply_i64("]),
    ]);
}

#[test]
fn jpeg_fixture_builders_tables_and_reference_decode_are_split() {
    let root = repo_root();
    let module = fs::read_to_string(root.join("crates/j2k-test-support/src/jpeg_fixtures.rs"))
        .expect("read JPEG fixture module");
    let builders =
        fs::read_to_string(root.join("crates/j2k-test-support/src/jpeg_fixtures/builders.rs"))
            .expect("read JPEG fixture builders");
    let reference = fs::read_to_string(
        root.join("crates/j2k-test-support/src/jpeg_fixtures/reference_decode.rs"),
    )
    .expect("read JPEG fixture reference decode helpers");
    let tables =
        fs::read_to_string(root.join("crates/j2k-test-support/src/jpeg_fixtures/tables.rs"))
            .expect("read JPEG fixture tables");

    assert_pattern_checks(&[
        PatternCheck::new("jpeg_fixtures.rs small re-export shell", &module)
            .required(&[
                "mod builders;",
                "mod reference_decode;",
                "mod tables;",
                "pub use builders::*;",
                "pub use tables::*;",
            ])
            .forbidden(&["pub fn "]),
        PatternCheck::new("jpeg fixture builders.rs fixture builders", &builders)
            .required(&[
                "pub fn minimal_baseline_420_jpeg",
                "pub fn extended_12bit_rgb_8x8_jpeg",
                "pub fn lossless_predictor_rgb_3x3_jpeg",
                "pub fn progressive_12bit_cmyk_8x8_jpeg",
            ])
            .forbidden(&[
                "pub const LOSSLESS_GRAYSCALE_3X3_PIXELS",
                "fn ycbcr8_pixels_to_rgb8",
                "enum ColorSpaceFixture",
                "fn upsample_h2v2_12bit_for_fixture",
            ]),
        PatternCheck::new("jpeg fixture tables.rs table ownership", &tables).required(&[
            "pub const LOSSLESS_GRAYSCALE_3X3_PIXELS",
            "pub(super) const LOSSLESS_RGB_8BIT_422_4X2_C0",
            "pub(super) struct Lossless422Planes",
            "pub(super) c0:",
        ]),
        PatternCheck::new(
            "jpeg fixture reference_decode.rs helper ownership",
            &reference,
        )
        .required(&[
            "pub(super) fn ycbcr8_pixels_to_rgb8",
            "pub(super) fn ycbcr16_pixels_to_rgb16",
            "pub(super) enum ColorSpaceFixture",
            "pub(super) fn upsample_h2v2_12bit_for_fixture",
        ])
        .forbidden(&["_jpeg() -> Vec<u8>"]),
    ]);
    assert!(
        module.lines().count() < 50,
        "jpeg_fixtures.rs must stay below the re-export shell line-count ratchet"
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "compare-binary ownership and duplication guards form one cohesive policy"
)]
fn compare_bins_use_library_common_helpers() {
    let root = repo_root();
    let common = fs::read_to_string(root.join("crates/j2k-compare/src/common.rs"))
        .expect("read j2k-compare common library module");
    let lib = fs::read_to_string(root.join("crates/j2k-compare/src/lib.rs"))
        .expect("read j2k-compare lib");
    let fixture = fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare.rs"))
        .expect("read fixture compare module");
    let fixture_cli =
        fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare/cli.rs"))
            .expect("read fixture compare CLI module");
    let fixture_manifest =
        fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare/manifest.rs"))
            .expect("read fixture compare manifest module");
    let fixture_rows =
        fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare/rows.rs"))
            .expect("read fixture compare rows module");
    let fixture_comparators =
        fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare/comparators.rs"))
            .expect("read fixture compare comparators module");
    let fixture_gates =
        fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare/gates.rs"))
            .expect("read fixture compare publication gates module");
    let fixture_types =
        fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare/types.rs"))
            .expect("read fixture compare types module");
    let encode_cli = fs::read_to_string(root.join("crates/j2k-compare/src/encode_compare/cli.rs"))
        .expect("read encode compare CLI module");
    let fixture_bin =
        fs::read_to_string(root.join("crates/j2k-compare/src/bin/jp2k_fixture_compare.rs"))
            .expect("read fixture compare bin");
    let encode_bin =
        fs::read_to_string(root.join("crates/j2k-compare/src/bin/jp2k_encode_compare.rs"))
            .expect("read encode compare bin");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-compare library modules", &lib).required(&[
            "pub mod common;",
            "pub mod fixture_compare;",
            "pub mod encode_compare;",
        ]),
        PatternCheck::new("fixture compare bin launcher", &fixture_bin)
            .required(&["j2k_compare::fixture_compare::main();"]),
        PatternCheck::new("encode compare bin launcher", &encode_bin)
            .required(&["j2k_compare::encode_compare::main();"]),
    ]);
    assert!(
        fixture.lines().count() < 600,
        "fixture_compare.rs must stay below the focused-coordinator line-count ratchet"
    );
    assert_pattern_checks(&[
        PatternCheck::new("fixture_compare manifest module shell", &fixture)
            .required(&["mod manifest;"])
            .forbidden(&[
                "fn fixture_manifest_from_env(",
                "fn external_fixture_metadata(",
            ]),
        PatternCheck::new("fixture_compare/manifest.rs ownership", &fixture_manifest).required(&[
            "pub(super) fn fixture_manifest_from_env",
            "pub(super) fn external_fixture_metadata",
        ]),
        PatternCheck::new("fixture_compare rows module shell", &fixture)
            .required(&["mod rows;"])
            .forbidden(&[
                "fn measurement_row(",
                "fn mixed_measurement_row(",
                "fn skip_row(",
                "fn mixed_skip_row(",
            ]),
        PatternCheck::new("fixture_compare/rows.rs ownership", &fixture_rows).required(&[
            "pub(super) fn measurement_row",
            "pub(super) fn mixed_measurement_row",
            "pub(super) fn skip_row",
            "pub(super) fn mixed_skip_row",
        ]),
        PatternCheck::new("fixture_compare comparators module shell", &fixture)
            .required(&["mod comparators;"])
            .forbidden(&[
                "fn decode_openjph_once(",
                "fn decode_kakadu_once(",
                "fn read_cli_pnm_output(",
                "OPENJPH_TEMP_COUNTER",
                "KAKADU_TEMP_COUNTER",
            ]),
        PatternCheck::new(
            "fixture_compare/comparators.rs ownership",
            &fixture_comparators,
        )
        .required(&[
            "pub(super) fn decode_openjph_once",
            "pub(super) fn decode_kakadu_once",
            "pub(super) fn openjph_is_available",
            "pub(super) fn kakadu_is_available",
            "fn read_cli_pnm_output",
        ]),
        PatternCheck::new("fixture_compare gates module shell", &fixture)
            .required(&["mod gates;"])
            .forbidden(&[
                "fn publication_blockers(",
                "fn publication_gate_skipped_comparators_label(",
                "fn require_mixed_fixture_group(",
                "fn external_unique_input_count_for_format_operation(",
            ]),
        PatternCheck::new("fixture_compare/gates.rs ownership", &fixture_gates).required(&[
            "pub(super) fn publication_blockers",
            "pub(super) fn publication_gate_skipped_comparators_label",
            "fn require_mixed_fixture_group",
            "fn external_unique_input_count_for_format_operation",
        ]),
        PatternCheck::new("fixture_compare types module shell", &fixture)
            .required(&["mod types;"])
            .forbidden(&[
                "enum BenchmarkMode",
                "enum Codec",
                "enum Container",
                "enum Operation",
                "enum OperationClass",
                "enum DecoderKind",
            ]),
        PatternCheck::new("fixture_compare/types.rs ownership", &fixture_types).required(&[
            "pub(super) enum BenchmarkMode",
            "pub(super) enum Codec",
            "pub(super) enum Container",
            "pub(super) enum Operation",
            "pub(super) enum OperationClass",
            "pub(super) enum DecoderKind",
        ]),
    ]);
    assert!(
        !root
            .join("crates/j2k-compare/src/bin/common/mod.rs")
            .exists(),
        "compare bins must not reintroduce a bin-local common module"
    );
    assert_pattern_checks(&[
        PatternCheck::new("j2k-compare common helper ownership", &common).required(&[
            "pub struct BatchSizeConfig",
            "pub struct BatchSizeEnv",
            "pub fn batch_size_config_from_env",
            "pub fn batch_size_config_from_values",
            "pub fn legacy_batch_sizes_from_env",
        ]),
        PatternCheck::new("fixture_compare shared batch-size helper use", &fixture_cli)
            .required(&[
                "use super::{",
                "common::batch_size_config_from_env(",
                "J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES",
            ])
            .forbidden(&[
                "mod common;",
                "struct BatchSizeConfig",
                "fn batch_size_config_from_values",
                "fn legacy_batch_sizes_from_env",
            ]),
        PatternCheck::new("encode_compare shared batch-size helper use", &encode_cli)
            .required(&[
                "use super::{",
                "common::batch_size_config_from_env(",
                "J2K_ENCODE_COMPARE_CASE_BATCH_SIZES",
            ])
            .forbidden(&[
                "mod common;",
                "struct BatchSizeConfig",
                "fn batch_size_config_from_values",
                "fn legacy_batch_sizes_from_env",
            ]),
    ]);
}

#[test]
fn deinterleave_reference_has_checked_public_entrypoint() {
    let root = repo_root();
    let native = fs::read_to_string(root.join("crates/j2k-native/src/scalar/encode.rs"))
        .expect("read native scalar encode module");
    let cuda_parity = fs::read_to_string(root.join("crates/j2k-cuda/tests/htj2k_encode_parity.rs"))
        .expect("read CUDA parity tests");
    let metal_parity = fs::read_to_string(root.join("crates/j2k-metal/src/encode/tests.rs"))
        .expect("read Metal encode tests");
    let metal_bench =
        fs::read_to_string(root.join("crates/j2k-metal/tests/encode_auto_routing_benchmark.rs"))
            .expect("read Metal encode benchmark tests");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-native checked deinterleave reference", &native).required(&[
            "pub fn try_deinterleave_reference",
            "checked_deinterleave_reference_bytes_per_sample",
            "checked_decode_byte_len3",
            "ValidationError::InvalidComponentMetadata",
            "pub fn deinterleave_reference",
            "try_deinterleave_reference(pixels",
        ]),
        PatternCheck::new(
            "CUDA parity tests checked deinterleave reference",
            &cuda_parity,
        )
        .required(&["try_deinterleave_reference"])
        .forbidden(&[" deinterleave_reference,", "= deinterleave_reference("]),
        PatternCheck::new(
            "Metal encode tests checked deinterleave reference",
            &metal_parity,
        )
        .required(&["try_deinterleave_reference"])
        .forbidden(&[" deinterleave_reference,", "= deinterleave_reference("]),
        PatternCheck::new(
            "Metal encode routing tests checked deinterleave reference",
            &metal_bench,
        )
        .required(&["try_deinterleave_reference"])
        .forbidden(&[" deinterleave_reference,", "= deinterleave_reference("]),
    ]);
}

#[test]
fn decode_strictness_policy_is_explicit_and_warns_on_lenient_default() {
    let root = repo_root();
    let native = fs::read_to_string(root.join("crates/j2k-native/src/image.rs"))
        .expect("read native image module");
    let facade_decode =
        fs::read_to_string(root.join("crates/j2k/src/decode.rs")).expect("read facade decode");
    let facade_view =
        fs::read_to_string(root.join("crates/j2k/src/view.rs")).expect("read facade view");
    let facade_batch =
        fs::read_to_string(root.join("crates/j2k/src/batch.rs")).expect("read facade batch");
    let crate_readme =
        fs::read_to_string(root.join("crates/j2k/README.md")).expect("read j2k README");
    let architecture =
        fs::read_to_string(root.join("docs/architecture.md")).expect("read architecture docs");

    assert_pattern_checks(&[
        PatternCheck::new("native DecodeSettings strictness constructors", &native).required(&[
            "pub const fn lenient() -> Self",
            "pub const fn strict() -> Self",
            "pub const fn lenient_tolerance_enabled",
            "Self::lenient()",
        ]),
        PatternCheck::new("j2k facade decode warnings", &facade_decode).required(&[
            "pub enum J2kDecodeWarning",
            "LenientDecodeMode",
            "decode_warnings_for_settings",
            "DecodeOutcome<J2kDecodeWarning>",
        ]),
        PatternCheck::new("j2k facade view warning propagation", &facade_view).required(&[
            "type Warning = J2kDecodeWarning",
            "decode_warnings_for_settings(DecodeSettings::default())",
        ]),
        PatternCheck::new("j2k facade batch warning propagation", &facade_batch).required(&[
            "DecodeOutcome<J2kDecodeWarning>",
            "decode_warnings_for_settings(DecodeSettings::default())",
        ]),
        PatternCheck::new("j2k README decode strictness policy", &crate_readme).required(&[
            "DecodeSettings::strict()",
            "J2kDecodeWarning::LenientDecodeMode",
        ]),
        PatternCheck::new("architecture decode strictness policy", &architecture).required(&[
            "DecodeSettings::strict()",
            "J2kDecodeWarning::LenientDecodeMode",
        ]),
    ]);
}
