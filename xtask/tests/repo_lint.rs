// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::BTreeSet,
    ffi::OsStr,
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
};

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
}

fn const_array_block<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source
        .find(&format!("const {name}:"))
        .unwrap_or_else(|| panic!("missing const {name}"));
    let rest = &source[start..];
    let end = rest
        .find("];")
        .unwrap_or_else(|| panic!("unterminated const {name}"));
    &rest[..end]
}

mod corpus_policy {
    use super::*;

    #[test]
    fn conformance_manifest_lists_all_committed_jpeg_inputs() {
        let root = repo_root();
        let conformance = root.join("corpus/conformance");
        let manifest = fs::read_to_string(conformance.join("manifest.json"))
            .expect("read conformance manifest");

        for entry in fs::read_dir(&conformance).expect("read conformance dir") {
            let entry = entry.expect("read conformance entry");
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("jpg") {
                continue;
            }
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("utf-8 fixture filename");
            assert!(
                manifest.contains(&format!("\"{filename}\"")),
                "conformance fixture {filename} is missing from manifest.json"
            );
        }
    }

    #[test]
    fn corpus_readme_does_not_claim_committed_fixtures_are_absent() {
        let readme =
            fs::read_to_string(repo_root().join("corpus/README.md")).expect("read corpus README");

        assert!(
            !readme.contains("intentionally empty"),
            "corpus README still claims the committed fixture corpus is empty"
        );
    }
}

mod source_policy {
    use super::*;

    #[test]
    fn adapter_crates_do_not_import_codec_private_modules() {
        let root = repo_root();
        let adapter_crates = [
            "crates/signinum-jpeg-metal",
            "crates/signinum-jpeg-cuda",
            "crates/signinum-j2k-metal",
            "crates/signinum-j2k-cuda",
        ];

        for crate_dir in adapter_crates {
            for path in rust_sources(&root.join(crate_dir)) {
                let source = fs::read_to_string(&path)
                    .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
                assert!(
                !source.contains("::__private") && !source.contains(" __private::"),
                "adapter source {} imports a codec __private module; use the public adapter API",
                path.strip_prefix(root).unwrap_or(&path).display()
            );
            }
        }
    }

    #[test]
    fn production_signinum_cuda_code_does_not_reference_nvjpeg() {
        let root = repo_root();
        let checked_dirs = [
            "crates/signinum-cuda-runtime/src",
            "crates/signinum-jpeg-cuda/src",
            "crates/signinum-jpeg-cuda/benches",
        ];

        for dir in checked_dirs {
            for path in rust_sources(&root.join(dir)) {
                let source = fs::read_to_string(&path)
                    .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
                assert!(
                !["nvjpeg", "nvJPEG", "Nvjpeg", "NVJPEG"]
                    .iter()
                    .any(|token| source.contains(token)),
                "production Signinum CUDA source {} still references nvJPEG; JPEG CUDA decode must use Signinum-owned paths only",
                path.strip_prefix(root).unwrap_or(&path).display()
            );
            }
        }
    }

    #[test]
    fn cuda_adapter_crates_keep_public_libs_as_module_shells() {
        let root = repo_root();
        let expected_modules = [
            (
                "crates/signinum-jpeg-cuda",
                [
                    "codec.rs",
                    "decoder.rs",
                    "error.rs",
                    "runtime.rs",
                    "session.rs",
                    "surface.rs",
                ]
                .as_slice(),
            ),
            (
                "crates/signinum-j2k-cuda",
                [
                    "codec.rs",
                    "decoder.rs",
                    "encode.rs",
                    "error.rs",
                    "runtime.rs",
                    "session.rs",
                    "surface.rs",
                ]
                .as_slice(),
            ),
        ];

        for (crate_dir, modules) in expected_modules {
            let src_dir = root.join(crate_dir).join("src");
            let lib_path = src_dir.join("lib.rs");
            let lib = fs::read_to_string(&lib_path)
                .unwrap_or_else(|err| panic!("read {}: {err}", lib_path.display()));
            let line_count = lib.lines().count();
            assert!(
                line_count <= 220,
                "{} should stay a thin public module shell; found {line_count} lines",
                lib_path.strip_prefix(root).unwrap_or(&lib_path).display()
            );

            for module in modules {
                let module_path = src_dir.join(module);
                assert!(
                    module_path.exists(),
                    "{} must exist to keep CUDA adapter responsibilities focused",
                    module_path
                        .strip_prefix(root)
                        .unwrap_or(&module_path)
                        .display()
                );
            }
        }
    }

    #[test]
    fn reusable_benchmark_generators_live_in_test_support() {
        let root = repo_root();
        let support = fs::read_to_string(root.join("crates/signinum-test-support/src/lib.rs"))
            .expect("read signinum-test-support");

        for required in [
            "pub fn gradient_u8",
            "pub fn patterned_rgb8_tiles",
            "pub fn gpu_bench_rgb8",
        ] {
            assert!(
                support.contains(required),
                "signinum-test-support must expose reusable generator `{required}`"
            );
        }
    }
}

mod architecture_policy {
    use super::*;

    #[test]
    fn workspace_contains_public_signinum_facade_crate() {
        let root = repo_root();
        let manifest_path = root.join("crates/signinum/Cargo.toml");
        let manifest = fs::read_to_string(&manifest_path).unwrap_or_else(|err| {
            panic!("read {}: {err}", manifest_path.display());
        });

        for required in [
            "name = \"signinum\"",
            "signinum-core",
            "signinum-jpeg",
            "signinum-j2k",
            "signinum-tilecodec",
        ] {
            assert!(
                manifest.contains(required),
                "{} must contain `{required}`",
                manifest_path
                    .strip_prefix(root)
                    .unwrap_or(&manifest_path)
                    .display()
            );
        }

        let root_manifest =
            fs::read_to_string(root.join("Cargo.toml")).expect("read workspace manifest");
        assert!(
            root_manifest.contains("\"crates/signinum\""),
            "workspace members must include the public signinum facade crate"
        );
    }

    #[test]
    fn signinum_facade_uses_explicit_upstream_reexports() {
        let root = repo_root();
        let facade_path = root.join("crates/signinum/src/lib.rs");
        let facade = fs::read_to_string(&facade_path).unwrap_or_else(|err| {
            panic!("read {}: {err}", facade_path.display());
        });

        let glob_reexports = facade
            .lines()
            .enumerate()
            .filter_map(|(idx, line)| {
                let trimmed = line.trim();
                (trimmed.starts_with("pub use signinum_") && trimmed.ends_with("::*;"))
                    .then(|| format!("{}:{}", idx + 1, trimmed))
            })
            .collect::<Vec<_>>();
        assert!(
            glob_reexports.is_empty(),
            "signinum facade must explicitly list upstream reexports:\n{}",
            glob_reexports.join("\n")
        );

        let root_reexports = facade
            .split("pub mod prelude")
            .next()
            .expect("facade root reexports");
        let public_root = root_reexports
            .split("pub mod tilecodec")
            .nth(1)
            .expect("facade root-level public uses");
        for required in ["TileBatchDecodeDevice", "TileBatchDecodeSubmit"] {
            assert!(
                public_root.contains(required),
                "signinum facade root must re-export {required}"
            );
        }
    }

    #[test]
    fn architecture_dependency_graph_matches_cargo_metadata() {
        let root = repo_root();
        let metadata_edges = cargo_metadata_workspace_edges(root);
        let docs =
            fs::read_to_string(root.join("docs/architecture.md")).expect("read architecture docs");
        let docs_edges = architecture_doc_dependency_edges(&docs);

        let missing = metadata_edges
            .difference(&docs_edges)
            .map(format_edge)
            .collect::<Vec<_>>();
        let extra = docs_edges
            .difference(&metadata_edges)
            .map(format_edge)
            .collect::<Vec<_>>();

        assert!(
            missing.is_empty() && extra.is_empty(),
            "docs/architecture.md crate dependency graph drifted from cargo metadata\n\
         missing from docs: {missing:#?}\n\
         not in cargo metadata: {extra:#?}"
        );
    }

    #[test]
    fn architecture_docs_classify_workspace_and_in_repo_tool_crates() {
        let root = repo_root();
        let docs =
            fs::read_to_string(root.join("docs/architecture.md")).expect("read architecture docs");

        for required in [
            "`signinum-test-support`",
            "dev helper",
            "`xtask`",
            "workspace tool",
            "`xtask/`",
        ] {
            assert!(
                docs.contains(required),
                "docs/architecture.md must classify `{required}`"
            );
        }
        assert!(
            !docs.contains("All crates live under `crates/`"),
            "docs/architecture.md must not claim xtask lives under crates/"
        );
    }

    #[test]
    fn tooling_and_validation_crates_stay_unpublished() {
        let root = repo_root();
        let workspace =
            fs::read_to_string(root.join("Cargo.toml")).expect("read workspace manifest");
        let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
        let publishable = const_array_block(&xtask, "PUBLISHABLE_PACKAGES");

        for (manifest, package) in [
            (
                "crates/signinum-test-support/Cargo.toml",
                "signinum-test-support",
            ),
            ("xtask/Cargo.toml", "xtask"),
            (
                "tests/nvidia-baseline/Cargo.toml",
                "signinum-nvidia-baseline",
            ),
        ] {
            let source = fs::read_to_string(root.join(manifest))
                .unwrap_or_else(|err| panic!("read {manifest}: {err}"));
            assert!(
                source.contains("publish = false"),
                "{manifest} must stay unpublished"
            );
            assert!(
                !publishable.contains(&format!("\"{package}\"")),
                "xtask publishable package gate must not include {package}"
            );
        }

        assert!(
            workspace.contains("\"crates/signinum-test-support\""),
            "root workspace must include signinum-test-support for shared test helpers"
        );
        assert!(
            workspace.contains("\"xtask\""),
            "root workspace must include xtask for cargo xtask automation"
        );
        assert!(
            !workspace.contains("\"tests/nvidia-baseline\""),
            "root workspace must not include the standalone NVIDIA baseline workspace"
        );
    }

    #[test]
    fn public_crates_do_not_reexport_signinum_j2k_native() {
        let root = repo_root();
        let mut offenders = Vec::new();

        for crate_dir in [
            "crates/signinum/src",
            "crates/signinum-j2k/src",
            "crates/signinum-transcode/src",
            "crates/signinum-j2k-metal/src",
            "crates/signinum-j2k-cuda/src",
            "crates/signinum-transcode-metal/src",
            "crates/signinum-transcode-cuda/src",
        ] {
            for path in rust_sources(&root.join(crate_dir)) {
                let source = fs::read_to_string(&path)
                    .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
                for (line_idx, line) in source.lines().enumerate() {
                    let trimmed = line.trim_start();
                    if trimmed.starts_with("pub use signinum_j2k_native")
                        || trimmed.starts_with("pub type ")
                            && trimmed.contains("signinum_j2k_native")
                    {
                        offenders.push(format!(
                            "{}:{}:{}",
                            path.strip_prefix(root).unwrap_or(&path).display(),
                            line_idx + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }

        assert!(
            offenders.is_empty(),
            "public crates must not re-export native J2K implementation types:\n{}",
            offenders.join("\n")
        );
    }

    #[test]
    fn rendered_public_api_does_not_expose_signinum_j2k_native() {
        let root = repo_root();
        let stable_api_snapshot =
            fs::read_to_string(root.join("docs/stable-api-1.0.public-api.txt"))
                .expect("read stable API snapshot");

        for package in [
            "signinum",
            "signinum-j2k",
            "signinum-transcode",
            "signinum-j2k-metal",
            "signinum-j2k-cuda",
            "signinum-transcode-metal",
            "signinum-transcode-cuda",
        ] {
            let api = cargo_public_api(root, package)
                .unwrap_or_else(|| stable_api_snapshot_section(&stable_api_snapshot, package));
            assert!(
                !api.contains("signinum_j2k_native"),
                "public API for package {package} exposes signinum_j2k_native:\n{api}"
            );
        }
    }
}

mod docs_and_workflows_policy {
    use super::*;

    #[test]
    fn wsi_decode_api_guide_covers_public_surfaces() {
        let root = repo_root();
        let readme = fs::read_to_string(root.join("README.md")).expect("read README");

        assert!(
            readme.contains("WSI decode contracts"),
            "README must document the WSI decode API surface"
        );

        for required in [
            "decode_region_scaled_into",
            "decode_rows",
            "TileBatchDecode",
            "BackendRequest::Auto",
            "BackendRequest::Metal",
            "BackendRequest::Cuda",
            "DeviceSurface",
            "ScratchPool",
            "DecoderContext",
        ] {
            assert!(
                readme.contains(required),
                "README.md must document WSI decode surface `{required}`"
            );
        }
    }

    #[test]
    fn ci_workflow_keeps_docs_and_benchmark_compile_gates() {
        let workflow = fs::read_to_string(repo_root().join(".github/workflows/ci.yml"))
            .expect("read CI workflow");
        let xtask = fs::read_to_string(repo_root().join("xtask/src/main.rs")).expect("read xtask");

        for required in [
            "cargo xtask doc",
            "cargo xtask stable-api",
            "cargo xtask bench-build",
            "cargo-public-api@0.52.0",
            "macos-latest",
        ] {
            assert!(
                workflow.contains(required),
                "CI workflow must contain `{required}`"
            );
        }
        assert!(
            !workflow.contains("macos-13"),
            "hosted CI must not gate releases on unsupported Intel macOS runners"
        );

        for required in [
            "\"doc\"",
            "\"--workspace\"",
            "\"--all-features\"",
            "\"--no-deps\"",
            "\"signinum-jpeg-metal\"",
            "\"signinum-j2k-metal\"",
            "\"--no-run\"",
        ] {
            assert!(xtask.contains(required), "xtask must contain `{required}`");
        }
    }

    #[test]
    fn ci_workflow_runs_semver_checks_for_stable_library_crates() {
        let workflow = fs::read_to_string(repo_root().join(".github/workflows/ci.yml"))
            .expect("read CI workflow");
        let xtask = fs::read_to_string(repo_root().join("xtask/src/main.rs")).expect("read xtask");
        let semver_job = workflow_job(&workflow, "semver");
        let semver_packages = const_array_block(&xtask, "STABLE_SEMVER_PACKAGES");

        assert!(
            semver_job.contains("cargo install cargo-semver-checks --version 0.48.0 --locked"),
            "CI semver job must install the pinned cargo-semver-checks version"
        );
        assert!(
            semver_job.contains("cargo xtask semver"),
            "CI semver job must use the repo-owned semver gate"
        );
        assert!(
            semver_job.contains("toolchain: \"1.96\""),
            "CI semver job must install the workspace Rust toolchain"
        );
        assert!(
            xtask.contains("unwrap_or_else(|_| \"1.96\".to_string())"),
            "xtask semver must default to the workspace Rust toolchain"
        );
        assert!(
            !semver_job.contains("release-type: minor"),
            "CI semver job must not treat the 0.5.0 boundary as a compatible minor release"
        );

        for package in [
            "signinum",
            "signinum-core",
            "signinum-jpeg",
            "signinum-j2k",
            "signinum-tilecodec",
            "signinum-jpeg-metal",
            "signinum-j2k-metal",
            "signinum-jpeg-cuda",
            "signinum-j2k-cuda",
            "signinum-transcode",
            "signinum-transcode-cuda",
            "signinum-metal-support",
            "signinum-transcode-metal",
            "signinum-j2k-native",
            "signinum-cuda-runtime",
            "signinum-profile",
        ] {
            assert!(
                semver_packages.contains(&format!("\"{package}\"")),
                "repo semver policy must cover stable library crate `{package}`"
            );
        }

        let package = "signinum-cli";
        assert!(
            !semver_job.contains(package),
            "CI semver job must not gate experimental or CLI crate `{package}`"
        );
    }

    #[test]
    fn xtask_test_does_not_run_benchmarks_as_tests() {
        let xtask = fs::read_to_string(repo_root().join("xtask/src/main.rs")).expect("read xtask");
        let test_section = xtask
            .split("fn test()")
            .nth(1)
            .and_then(|rest| rest.split("fn doc()").next())
            .expect("xtask test section");

        for required in ["\"--lib\"", "\"--bins\"", "\"--tests\"", "\"--doc\""] {
            assert!(
                test_section.contains(required),
                "xtask test must include cargo test selector `{required}`"
            );
        }
        assert!(
        !test_section.contains("\"--all-targets\""),
        "xtask test must not pass --all-targets because harness=false benchmark binaries would run as tests"
    );
    }

    #[test]
    fn xtask_exposes_nextest_machete_and_strict_clippy_gates() {
        let xtask = fs::read_to_string(repo_root().join("xtask/src/main.rs")).expect("read xtask");
        let help_section = xtask
            .split("fn print_help()")
            .nth(1)
            .expect("xtask help section");

        for task in ["nextest", "machete", "clippy-strict"] {
            assert!(
                xtask.contains(&format!("\"{task}\" =>")),
                "xtask must dispatch `{task}`"
            );
            assert!(
                help_section.contains(task),
                "xtask help must document `{task}`"
            );
        }

        for required in [
            "\"nextest\"",
            "\"run\"",
            "\"cargo-machete\"",
            "\"--no-deps\"",
            "\"clippy::pedantic\"",
            "\"clippy::nursery\"",
            "\"signinum-j2k-native\"",
            "\"signinum-j2k\"",
        ] {
            assert!(xtask.contains(required), "xtask must contain `{required}`");
        }
    }

    #[test]
    fn xtask_fuzz_build_checks_every_fuzz_manifest() {
        let root = repo_root();
        let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
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
        }
    }

    #[test]
    fn ci_coverage_job_is_a_required_gate() {
        let workflow = fs::read_to_string(repo_root().join(".github/workflows/ci.yml"))
            .expect("read CI workflow");
        let coverage_job = workflow_job(&workflow, "coverage");
        const INSTALL_ACTION_SHA: &str = "91534edaf9fd796a162759d80d49cdff574bff2c";

        assert!(
            coverage_job.contains(&format!("taiki-e/install-action@{INSTALL_ACTION_SHA}"))
                && coverage_job.contains("tool: cargo-llvm-cov")
                && coverage_job.contains("cargo xtask coverage"),
            "coverage job must install cargo-llvm-cov from a pinned action and run xtask coverage"
        );
        assert!(
            !coverage_job.contains("taiki-e/install-action@cargo-llvm-cov"),
            "coverage job must not use mutable install-action tool tags"
        );
        assert!(
            !coverage_job.contains("continue-on-error"),
            "coverage job must not be allowed to fail silently"
        );
    }

    #[test]
    fn ci_miri_job_is_a_required_gate() {
        let workflow = fs::read_to_string(repo_root().join(".github/workflows/ci.yml"))
            .expect("read CI workflow");
        let miri_job = workflow_job(&workflow, "miri");

        for required in ["toolchain: nightly", "components: miri", "cargo xtask miri"] {
            assert!(
                miri_job.contains(required),
                "miri job must contain `{required}`"
            );
        }
    }

    #[test]
    fn ci_fuzz_run_budgets_are_nontrivial() {
        let workflow = fs::read_to_string(repo_root().join(".github/workflows/ci.yml"))
            .expect("read CI workflow");

        for required in [
            "SIGNINUM_FUZZ_RUNS: \"512\"",
            "SIGNINUM_FUZZ_MAX_TOTAL_TIME_SECONDS: \"60\"",
            "SIGNINUM_FUZZ_RUNS: \"20000\"",
            "SIGNINUM_FUZZ_MAX_TOTAL_TIME_SECONDS: \"900\"",
        ] {
            assert!(
                workflow.contains(required),
                "CI fuzz budgets must contain `{required}`"
            );
        }
    }

    #[test]
    fn deny_paste_advisory_ignore_has_review_metadata() {
        let deny = fs::read_to_string(repo_root().join("deny.toml")).expect("read deny.toml");

        for required in [
            "RUSTSEC-2024-0436",
            "Review-by: 2026-12-31",
            "https://crates.io/crates/metal",
            "https://github.com/gfx-rs/metal-rs/blob/master/Cargo.toml",
            "https://rustsec.org/advisories/RUSTSEC-2024-0436.html",
            "https://github.com/gfx-rs/metal-rs/issues/349",
        ] {
            assert!(
                deny.contains(required),
                "deny.toml paste advisory rationale must contain `{required}`"
            );
        }
    }

    #[test]
    fn unsafe_audit_rows_include_invariants_and_regression_guards() {
        let audit = fs::read_to_string(repo_root().join("docs/unsafe-audit.md"))
            .expect("read unsafe audit");

        assert!(
            audit.contains("| Path | Scope | Invariants | Regression guards |"),
            "unsafe audit must include invariant and regression guard columns"
        );
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
        let codeowners =
            fs::read_to_string(root.join(".github/CODEOWNERS")).expect("read CODEOWNERS");

        assert!(
            workflow.contains("\npermissions:\n  contents: read\n"),
            "CI workflow must default the GITHUB_TOKEN to contents: read"
        );
        for required in [
            "pull-requests: read",
            "actions: read",
            "crates/signinum-cuda-runtime/",
            "crates/signinum-jpeg-cuda/",
            "crates/signinum-j2k-cuda/",
            "crates/signinum-transcode-cuda/",
            "crates/signinum-metal-support/",
            "crates/signinum-jpeg-metal/",
            "crates/signinum-j2k-metal/",
            "crates/signinum-transcode-metal/",
            "gpu-validation.yml/runs?head_sha=",
            "run.get(\"conclusion\") == \"success\"",
            "No GPU path changes detected.",
        ] {
            assert!(
                gpu_policy.contains(required),
                "GPU path policy job must contain `{required}`"
            );
        }
        for required in [
            ".github/workflows/gpu-validation.yml",
            "crates/signinum-cuda-runtime/",
            "crates/signinum-jpeg-cuda/",
            "crates/signinum-j2k-cuda/",
            "crates/signinum-transcode-cuda/",
            "crates/signinum-metal-support/",
            "crates/signinum-jpeg-metal/",
            "crates/signinum-j2k-metal/",
            "crates/signinum-transcode-metal/",
        ] {
            assert!(
                codeowners.contains(required),
                "CODEOWNERS must cover GPU path `{required}`"
            );
        }
    }

    #[test]
    fn gpu_validation_workflow_is_self_hosted_and_explicit() {
        let root = repo_root();
        let workflow_path = root.join(".github/workflows/gpu-validation.yml");
        let workflow = fs::read_to_string(&workflow_path).expect("read GPU validation workflow");

        for required in [
            "workflow_dispatch",
            "run-timed-benchmarks",
            "run-metal-validation",
            "self-hosted",
            "metal",
            "cuda",
            "NVCC: /usr/local/cuda/bin/nvcc",
            "cargo test -p signinum-jpeg-metal",
            "cargo test -p signinum-j2k-metal",
            "cargo test -p signinum-jpeg-cuda",
            "cargo test -p signinum-j2k-cuda",
        ] {
            assert!(
                workflow.contains(required),
                "{} must contain `{required}`",
                workflow_path
                    .strip_prefix(root)
                    .unwrap_or(&workflow_path)
                    .display()
            );
        }
    }

    #[test]
    fn cuda_gpu_validation_job_stays_cuda_focused() {
        let root = repo_root();
        let workflow_path = root.join(".github/workflows/gpu-validation.yml");
        let workflow = fs::read_to_string(&workflow_path).expect("read GPU validation workflow");
        let cuda_job = workflow_job(&workflow, "cuda-x86_64-compatibility");

        for required in [
            "runs-on: [self-hosted, Linux, X64, cuda]",
            "NVCC: /usr/local/cuda/bin/nvcc",
            "SIGNINUM_REQUIRE_CUDA_RUNTIME",
        "SIGNINUM_REQUIRE_CUDA_JPEG_HARDWARE_DECODE",
        "SIGNINUM_GPU_BENCH_DIM",
        "SIGNINUM_GPU_BENCH_BATCH",
        "SIGNINUM_GPU_BENCH_BATCH_DIM",
        "uname -a",
        "rustc -Vv",
        "cargo -V",
        "nvidia-smi",
        "CUDA runtime validation requires a working CUDA driver",
        "cargo test -p signinum-jpeg-cuda --all-targets --features cuda-runtime",
        "cargo test -p signinum-j2k-cuda --all-targets --features cuda-runtime",
        "cargo bench -p signinum-jpeg-cuda --bench device_decode --features cuda-runtime --no-run",
        "cargo bench -p signinum-jpeg-cuda --bench device_decode --features cuda-runtime -- --noplot",
    ] {
        assert!(
            cuda_job.contains(required),
            "{} CUDA job must contain `{required}`",
            workflow_path
                .strip_prefix(root)
                .unwrap_or(&workflow_path)
                .display()
        );
    }

        let forbidden_j2k_metal_compare_bench = [
            "cargo bench -p ",
            "signinum-j2k-metal",
            " --bench compare --no-run",
        ]
        .concat();
        for forbidden in [
            forbidden_j2k_metal_compare_bench.as_str(),
            "cargo bench -p signinum-jpeg --no-run",
            "cargo test -p signinum-jpeg-metal",
            "cargo test -p signinum-j2k-metal",
        ] {
            assert!(
                !cuda_job.contains(forbidden),
                "{} CUDA job must not contain Metal validation command `{forbidden}`",
                workflow_path
                    .strip_prefix(root)
                    .unwrap_or(&workflow_path)
                    .display()
            );
        }
    }

    #[test]
    fn cuda_build_scripts_do_not_probe_default_nvcc() {
        let root = repo_root();
        for relative in [
            "crates/signinum-cuda-runtime/build.rs",
            "tests/nvidia-baseline/build.rs",
        ] {
            let source = fs::read_to_string(root.join(relative))
                .unwrap_or_else(|err| panic!("read {relative}: {err}"));
            assert!(
                !source.contains("unwrap_or_else(|| \"nvcc\".into())"),
                "{relative} must not default to PATH nvcc"
            );
            assert!(
                source.contains("requires absolute NVCC"),
                "{relative} must require absolute NVCC in strict mode"
            );
        }
    }

    #[test]
    fn cuda_decode_profile_workflow_exports_rca_artifacts() {
        let root = repo_root();
        let workflow_path = root.join(".github/workflows/gpu-validation.yml");
        let workflow = fs::read_to_string(&workflow_path).expect("read GPU validation workflow");
        let cuda_job = workflow_job(&workflow, "cuda-x86_64-compatibility");

        for required in [
            "run-cuda-htj2k-decode-profile",
            "CUDA HTJ2K decode RCA profile",
            "SIGNINUM_REQUIRE_CUDA_BENCH: \"1\"",
            "SIGNINUM_J2K_PROFILE_STAGES: summary",
            "SIGNINUM_J2K_CUDA_TRACE: ${{ github.workspace }}/target/cuda_htj2k_decode_trace.json",
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
        ] {
            assert!(
                workflow.contains(required) || cuda_job.contains(required),
                "{} CUDA decode profile gate must contain `{required}`",
                workflow_path
                    .strip_prefix(root)
                    .unwrap_or(&workflow_path)
                    .display()
            );
        }
    }

    #[test]
    fn nvidia_baseline_workflow_exports_direct_decode_artifacts() {
        let root = repo_root();
        let workflow_path = root.join(".github/workflows/gpu-validation.yml");
        let workflow = fs::read_to_string(&workflow_path).expect("read GPU validation workflow");
        let cuda_job = workflow_job(&workflow, "cuda-x86_64-compatibility");

        for required in [
            "run-nvidia-baseline",
            "--bin transcode_compare",
            "--decomposition-levels 1",
            "--decomposition-levels 2",
            "target/transcode_compare_level1.json",
            "target/transcode_compare_level2.json",
            "tests/nvidia-baseline/scripts/assert_transcode_perf.py",
            "SIGNINUM_LEVEL1_CUDA_HT_MIN_MPS",
            "SIGNINUM_LEVEL2_CUDA_HT_MIN_MPS",
            "--bin decode_compare",
            "--jpeg-dir \"${SIGNINUM_BENCH_JPEG_DIR}\"",
            "--min-inputs 100",
            "target/decode_compare.json",
            "target/decode_compare.csv",
            "python3 -m json.tool target/decode_compare.json",
            "nvidia-baseline-comparison",
        ] {
            assert!(
                workflow.contains(required) || cuda_job.contains(required),
                "{} NVIDIA baseline gate must contain `{required}`",
                workflow_path
                    .strip_prefix(root)
                    .unwrap_or(&workflow_path)
                    .display()
            );
        }
    }

    #[test]
    fn nvidia_codec_comparator_stays_test_only() {
        let root = repo_root();
        let needles = [
            "signinum-nvidia-baseline",
            "nvjpeg2000",
            "nvjpeg2k",
            "nvidia-baseline",
        ];
        let mut seen = 0usize;
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
                seen += 1;
                let allowed = rel_s.starts_with("./tests/nvidia-baseline/")
                    || rel_s == "./.github/workflows/gpu-validation.yml"
                    || rel_s == "./xtask/tests/repo_lint.rs";
                if !allowed {
                    violations.push(format!("{}:{}:{}", rel_s, line_idx + 1, line));
                }
            }
        }

        assert!(
            seen > 0,
            "repo must contain the test-only NVIDIA comparator guard input"
        );
        assert!(
            violations.is_empty(),
            "NVIDIA codec comparator references must stay test-only:\n{}",
            violations.join("\n")
        );
    }
}

mod release_policy {
    use super::*;

    #[test]
    fn crates_io_publish_policy_is_explicit() {
        let root = repo_root();
        let workspace =
            fs::read_to_string(root.join("Cargo.toml")).expect("read workspace manifest");
        let changelog = fs::read_to_string(root.join("CHANGELOG.md")).expect("read changelog");
        let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
        let publishable = const_array_block(&xtask, "PUBLISHABLE_PACKAGES");
        let publish_workflow = fs::read_to_string(root.join(".github/workflows/publish.yml"))
            .expect("read publish workflow");
        let version = workspace_package_version(&workspace);

        assert!(
            changelog.contains(&format!("## [{version}]")),
            "CHANGELOG.md must contain a section for the current staged release version {version}"
        );

        for package in [
            "signinum-core",
            "signinum-cuda-runtime",
            "signinum-profile",
            "signinum-j2k-native",
            "signinum-tilecodec",
            "signinum-jpeg",
            "signinum-j2k",
            "signinum-jpeg-metal",
            "signinum-jpeg-cuda",
            "signinum-j2k-metal",
            "signinum-j2k-cuda",
            "signinum-cli",
            "signinum",
        ] {
            assert!(
                publishable.contains(&format!("\"{package}\"")),
                "xtask package gate must include publishable package {package}"
            );
            assert!(
                publish_workflow.contains(&format!("publish-{package}:")),
                "publish workflow must include package {package}"
            );
        }

        let package = "signinum-j2k-compare";
        assert!(
            !publishable.contains(&format!("\"{package}\"")),
            "xtask package gate must not package local comparator package {package}"
        );
        assert!(
            !publish_workflow.contains(&format!("publish-{package}:")),
            "publish workflow must not publish local comparator package {package}"
        );
    }

    #[test]
    fn release_docs_use_manifest_versions_for_publish_order() {
        let release =
            fs::read_to_string(repo_root().join("docs/release.md")).expect("read release docs");

        assert!(
        release.contains("manifest versions"),
        "release docs must describe publishing the current manifest versions instead of stale hard-coded versions"
    );
        assert!(
            !release.contains("`signinum-j2k` `1.1.0`")
                && !release.contains("`signinum-j2k-native` `0.3.0`")
                && !release.contains("`signinum` `1.0.0`"),
            "release docs must not carry stale pre-facade publish versions"
        );
    }

    fn const_array_block<'a>(source: &'a str, name: &str) -> &'a str {
        let start = source
            .find(&format!("const {name}:"))
            .unwrap_or_else(|| panic!("missing const {name}"));
        let rest = &source[start..];
        let end = rest
            .find("];")
            .unwrap_or_else(|| panic!("unterminated const {name}"));
        &rest[..end]
    }

    fn workspace_package_version(workspace_manifest: &str) -> &str {
        workspace_manifest
            .lines()
            .find_map(|line| {
                let line = line.trim();
                line.strip_prefix("version")
                    .and_then(|rest| rest.split('"').nth(1))
            })
            .expect("workspace package version")
    }

    #[test]
    fn j2k_compare_stays_unpublished_and_out_of_j2k_package_deps() {
        let root = repo_root();
        let compare_manifest =
            fs::read_to_string(root.join("crates/signinum-j2k-compare/Cargo.toml"))
                .expect("read signinum-j2k-compare manifest");
        let j2k_manifest = fs::read_to_string(root.join("crates/signinum-j2k/Cargo.toml"))
            .expect("read signinum-j2k manifest");

        assert!(
            compare_manifest.contains("publish = false"),
            "signinum-j2k-compare must remain an unpublished local oracle helper"
        );
        assert!(
            !j2k_manifest.contains("signinum-j2k-compare"),
            "signinum-j2k must not package a dev-dependency on signinum-j2k-compare"
        );
    }

    #[test]
    fn package_preflight_is_staged_dependency_aware() {
        let root = repo_root();
        let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
        let publish_script =
            fs::read_to_string(root.join("scripts/publish-crate.sh")).expect("read publish script");
        let release = fs::read_to_string(root.join("docs/release.md")).expect("read release docs");

        assert!(
        xtask.contains("STAGED_DEPENDENCY_PACKAGES"),
        "xtask package preflight must explicitly model crates blocked by unpublished staged dependencies"
    );
        let strict_packages = const_array_block(&xtask, "REGISTRY_INDEPENDENT_PACKAGES");
        let staged_packages = const_array_block(&xtask, "STAGED_DEPENDENCY_PACKAGES");
        for (package, dependency) in cargo_metadata_workspace_edges(root) {
            assert!(
                !strict_packages.contains(&format!("\"{package}\"")),
                "strict package preflight must not include `{package}` while it depends on staged workspace crate `{dependency}`"
            );
        }
        assert!(
            staged_packages.contains("\"signinum-cuda-runtime\""),
            "signinum-cuda-runtime depends on staged signinum-core and must not run strict package verification before publication"
        );
        assert!(
        xtask.contains("\"--list\"") && xtask.contains("unpublished workspace dependencies"),
        "xtask package preflight must validate package contents for staged downstream crates without hiding why strict packaging is skipped"
    );
        assert!(
        publish_script.contains("dry-run package list only")
            && publish_script.contains("signinum-cli")
            && publish_script.contains("cargo package -p \"$crate\" --list"),
        "publish workflow dry-run must not fail downstream crates only because staged dependency versions are not published yet"
    );
        assert!(
            release.contains("cargo package --list")
                && release.contains("cargo publish --dry-run")
                && release.contains("unpublished workspace dependencies"),
            "release docs must explain the pre-publish package validation limits"
        );
    }

    #[test]
    fn publish_script_covers_all_publishable_crates() {
        let root = repo_root();
        let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
        let publish_script =
            fs::read_to_string(root.join("scripts/publish-crate.sh")).expect("read publish script");
        let publishable = const_array_block(&xtask, "PUBLISHABLE_PACKAGES");

        for line in publishable.lines() {
            let package = line.trim().trim_matches([',', '"']);
            if package.is_empty()
                || package.starts_with("const ")
                || package.starts_with(']')
                || package.starts_with('&')
            {
                continue;
            }
            assert!(
                publish_script.contains(package),
                "publish script must allow publishable package `{package}`"
            );
        }
    }
}

mod public_docs_policy {
    use super::*;

    #[test]
    fn supported_signinum_env_vars_are_documented() {
        let root = repo_root();
        let docs_path = root.join("docs/env-vars.md");
        let docs = fs::read_to_string(&docs_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", docs_path.display()));
        let readme = fs::read_to_string(root.join("README.md")).expect("read README");
        assert!(
            readme.contains("docs/env-vars.md"),
            "README must link the supported environment-variable reference"
        );

        let mut missing = Vec::new();
        for path in repo_text_files(root) {
            if is_archived_handoff(&path)
                || path.ends_with("docs/env-vars.md")
                || path.ends_with("xtask/tests/repo_lint.rs")
            {
                continue;
            }
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
            for token in signinum_env_tokens(&source) {
                if is_internal_signinum_token(&token) {
                    continue;
                }
                if !docs.contains(&format!("`{token}`")) {
                    missing.push(format!(
                        "{}: {token}",
                        path.strip_prefix(root).unwrap_or(&path).display()
                    ));
                }
            }
        }

        assert!(
            missing.is_empty(),
            "supported SIGNINUM_* environment variables must be documented in docs/env-vars.md:\n{}",
            missing.join("\n")
        );
        assert!(
            !docs.contains("SIGNINUM_JPEG_METAL_SPLIT_FAST420_BATCH"),
            "removed experiment-only JPEG Metal fast420 split switch must not be documented as supported"
        );
    }

    #[test]
    fn public_docs_describe_facade_auto_and_cuda_runtime_surface_scope() {
        let root = repo_root();
        let readme = fs::read_to_string(root.join("README.md")).expect("read README");
        let changelog = fs::read_to_string(root.join("CHANGELOG.md")).expect("read changelog");
        let architecture =
            fs::read_to_string(root.join("docs/architecture.md")).expect("read architecture docs");
        let release = fs::read_to_string(root.join("docs/release.md")).expect("read release docs");

        for (name, docs) in [
            ("README.md", readme.as_str()),
            ("docs/architecture.md", architecture.as_str()),
            ("docs/release.md", release.as_str()),
        ] {
            assert!(
                docs.contains("facade release")
                    && docs.contains("Runtime backend selection defaults to `Auto`"),
                "{name} must name the facade release posture and Auto backend policy"
            );
        }

        for (name, docs) in [
            ("README.md", readme.as_str()),
            ("CHANGELOG.md", changelog.as_str()),
            ("docs/release.md", release.as_str()),
        ] {
            assert!(
            docs.contains("cuda-runtime")
                && docs.contains("CUDA device memory")
                && docs.contains("Signinum-owned CUDA")
                && docs.contains("NVIDIA performance"),
            "{name} must describe CUDA device-memory output and owned CUDA scope without overclaiming NVIDIA performance"
        );
            assert!(
            !docs.contains("compatibility-only with no runtime CUDA decode"),
            "{name} must not describe CUDA as compatibility-only after runtime surface support exists"
        );
        }
    }

    #[test]
    fn public_docs_route_users_to_current_crates() {
        let root = repo_root();
        let readme = fs::read_to_string(root.join("README.md")).expect("read README");

        for required in [
            "Which crate should I use?",
            "Fast Path For LLM-Assisted Use",
            "cargo add signinum",
            "statumen",
            "wsi-dicom",
            "signinum-jpeg",
            "signinum-j2k",
            "signinum-cli",
        ] {
            assert!(
                readme.contains(required),
                "README.md must route users to `{required}` after the rename"
            );
        }

        let legacy_terms = [
            format!("{}{}", "ash", "lar"),
            format!("{}{}", "zig", "gurat"),
        ];
        for legacy in &legacy_terms {
            assert!(
                !readme.to_ascii_lowercase().contains(legacy),
                "README.md must use current package names only"
            );
        }
    }

    #[test]
    fn public_repo_excludes_agent_private_artifacts() {
        let root = repo_root();
        let private_docs_name = ["super", "powers"].concat();
        let private_dir = ["docs", private_docs_name.as_str()].join("/");
        let migration_doc = ["MIGRATION", ".md"].concat();
        let migration_doc_lower = migration_doc.to_ascii_lowercase();
        let mut offenders = Vec::new();

        for path in repo_text_files(root) {
            let relative = path.strip_prefix(root).unwrap_or(&path);
            let relative_text = relative.to_string_lossy();
            let file_name = path
                .file_name()
                .and_then(OsStr::to_str)
                .unwrap_or_default()
                .to_ascii_lowercase();
            if relative_text.starts_with(&private_dir) || file_name == migration_doc_lower {
                offenders.push(relative_text.to_string());
            }
        }

        assert!(
        offenders.is_empty(),
        "public repo must not track agent-private planning docs or migration scratch files: {offenders:?}"
    );
    }

    #[test]
    fn published_crates_have_crates_io_landing_readmes() {
        let root = repo_root();

        for crate_dir in publishable_crate_dirs() {
            let manifest_path = root.join(crate_dir).join("Cargo.toml");
            let manifest = fs::read_to_string(&manifest_path)
                .unwrap_or_else(|err| panic!("read {}: {err}", manifest_path.display()));
            let readme_path = root.join(crate_dir).join("README.md");

            assert!(
                manifest.contains("readme"),
                "{} must declare a readme for crates.io landing pages",
                manifest_path
                    .strip_prefix(root)
                    .unwrap_or(&manifest_path)
                    .display()
            );
            assert!(
                readme_path.exists(),
                "{} must exist for crates.io landing pages",
                readme_path
                    .strip_prefix(root)
                    .unwrap_or(&readme_path)
                    .display()
            );
        }
    }

    #[test]
    fn publishable_crates_configure_docs_rs_metadata() {
        let root = repo_root();

        for crate_dir in publishable_crate_dirs() {
            let manifest_path = root.join(crate_dir).join("Cargo.toml");
            let manifest = fs::read_to_string(&manifest_path)
                .unwrap_or_else(|err| panic!("read {}: {err}", manifest_path.display()));

            assert!(
                manifest.contains("[package.metadata.docs.rs]"),
                "{} must configure docs.rs metadata",
                manifest_path
                    .strip_prefix(root)
                    .unwrap_or(&manifest_path)
                    .display()
            );
            assert!(
                manifest.contains("all-features = true"),
                "{} must build docs.rs with all features enabled",
                manifest_path
                    .strip_prefix(root)
                    .unwrap_or(&manifest_path)
                    .display()
            );
            assert!(
                manifest.contains("targets = []"),
                "{} must keep docs.rs targets explicit",
                manifest_path
                    .strip_prefix(root)
                    .unwrap_or(&manifest_path)
                    .display()
            );
        }
    }

    #[test]
    fn support_matrix_is_linked_and_covers_adoption_surfaces() {
        let root = repo_root();
        let readme = fs::read_to_string(root.join("README.md")).expect("read README");

        for required in [
            "Stable APIs",
            "Experimental APIs",
            "BackendRequest::Auto",
            "Security",
            "Benchmark and parity policy",
            "MSRV",
            "OpenJPEG",
            "Grok",
        ] {
            assert!(
                readme.contains(required),
                "README.md must cover `{required}`"
            );
        }
    }

    #[test]
    fn facade_and_transcode_examples_are_publicly_linked() {
        let root = repo_root();
        let readme = fs::read_to_string(root.join("README.md")).expect("read README");

        for example in [
            "crates/signinum/examples/inspect_and_decode.rs",
            "crates/signinum/examples/tile_decompress.rs",
            "crates/signinum-transcode/examples/jpeg_to_htj2k.rs",
        ] {
            assert!(
                root.join(example).exists(),
                "expected runnable example `{example}`"
            );
            assert!(readme.contains(example), "README must link `{example}`");
        }
    }

    #[test]
    fn benchmark_docs_define_publication_gate_for_openjpeg_and_grok() {
        let root = repo_root();
        let readme = fs::read_to_string(root.join("README.md")).expect("read README");
        let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");

        for required in [
            "published benchmark",
            "SIGNINUM_J2K_COMPARE_THREADS",
            "SIGNINUM_REQUIRE_OPENJPEG=1",
            "SIGNINUM_REQUIRE_GROK=1",
            "comparator availability",
            "comparator version",
            "input source",
            "signinum-generated",
        ] {
            assert!(
                readme.contains(required),
                "README.md benchmark policy must contain `{required}`"
            );
        }

        assert!(
            xtask.contains("\"j2k-bench-signoff\""),
            "xtask must expose a no-silent-skip J2K benchmark signoff task"
        );
    }

    #[test]
    fn j2k_metal_bench_surface_stays_clean_after_reset() {
        let root = repo_root();
        let cargo_toml = fs::read_to_string(root.join("crates/signinum-j2k-metal/Cargo.toml"))
            .expect("read J2K Metal manifest");
        let readme = fs::read_to_string(root.join("README.md")).expect("read README");
        let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
        let openjpeg = fs::read_to_string(root.join("crates/signinum-j2k-compare/src/openjpeg.rs"))
            .expect("read OpenJPEG comparator");
        let grok = fs::read_to_string(root.join("crates/signinum-j2k-compare/src/grok.rs"))
            .expect("read Grok comparator");

        assert!(
        !cargo_toml.contains("[[bench]]"),
        "signinum-j2k-metal bench targets must stay reset until new profiling benches are added"
    );

        for forbidden in [
            "criterion =",
            "signinum-j2k-compare =",
            "name = \"device_upload\"",
            "name = \"compare\"",
            "name = \"encode_stages\"",
            "name = \"decode_stages\"",
        ] {
            assert!(
                !cargo_toml.contains(forbidden),
                "signinum-j2k-metal manifest must not contain legacy bench entry `{forbidden}`"
            );
        }

        let benches_dir = root.join("crates/signinum-j2k-metal/benches");
        if benches_dir.exists() {
            let stale_entries: Vec<_> = fs::read_dir(&benches_dir)
                .expect("read J2K Metal benches dir")
                .map(|entry| {
                    let path = entry.expect("read J2K Metal bench entry").path();
                    path.strip_prefix(root)
                        .expect("bench entry under repo root")
                        .display()
                        .to_string()
                })
                .collect();
            assert!(
                stale_entries.is_empty(),
                "signinum-j2k-metal benches dir must stay empty after reset: {stale_entries:?}"
            );
        }

        let removed_j2k_metal_bench_command =
            ["cargo bench -p ", "signinum-j2k-metal", " --bench"].concat();
        assert!(
            !readme.contains(&removed_j2k_metal_bench_command),
            "README.md must not publish removed signinum-j2k-metal bench commands"
        );
        assert!(
            !xtask.contains(&removed_j2k_metal_bench_command),
            "xtask must not run removed signinum-j2k-metal bench commands"
        );
        assert!(
            openjpeg.contains("pub fn version"),
            "OpenJPEG comparator must expose version metadata"
        );
        assert!(
            grok.contains("pub fn version") && grok.contains("pub fn library_path"),
            "Grok comparator must expose version and path metadata"
        );
    }

    #[test]
    fn public_text_does_not_embed_local_user_home_paths() {
        let root = repo_root();
        let mut offenders = Vec::new();

        for path in repo_text_files(root) {
            if is_archived_handoff(&path) {
                continue;
            }
            if path.ends_with("xtask/tests/repo_lint.rs") {
                continue;
            }
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
            if source.contains("/Users/") || source.contains("C:\\Users\\") {
                offenders.push(
                    path.strip_prefix(root)
                        .unwrap_or(&path)
                        .display()
                        .to_string(),
                );
            }
        }

        assert!(
        offenders.is_empty(),
        "public text must not embed local user-home paths; use env vars or repo-relative defaults: {offenders:?}"
    );
    }

    #[test]
    fn referenced_shell_scripts_exist() {
        let root = repo_root();
        let mut missing = Vec::new();

        for path in repo_text_files(root) {
            if is_archived_handoff(&path) {
                continue;
            }
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
            for script in referenced_shell_scripts(&source) {
                let root_relative = root.join(&script);
                let file_relative = path.parent().expect("text file has parent").join(&script);
                if !root_relative.exists() && !file_relative.exists() {
                    missing.push(format!(
                        "{} references missing script {script}",
                        path.strip_prefix(root).unwrap_or(&path).display()
                    ));
                }
            }
        }

        assert!(
            missing.is_empty(),
            "all referenced shell scripts must exist: {missing:?}"
        );
    }

    #[test]
    fn public_narrative_docs_do_not_carry_stale_zeiss_claims() {
        let root = repo_root();
        let mut offenders = Vec::new();

        for relative in [
            "README.md",
            "docs/architecture.md",
            "docs/release.md",
            "paper/paper.md",
            "paper/arxiv/main.tex",
        ] {
            let path = root.join(relative);
            let Ok(source) = fs::read_to_string(&path) else {
                if relative.starts_with("paper/") {
                    continue;
                }
                panic!("read {}: missing required narrative doc", path.display());
            };
            if source.contains("Zeiss") {
                offenders.push(relative);
            }
        }

        assert!(
            offenders.is_empty(),
            "public narrative docs must not carry stale Zeiss integration claims: {offenders:?}"
        );
    }

    #[test]
    fn packaged_rust_sources_do_not_include_files_outside_their_crate() {
        let root = repo_root();
        let workspace_crates = root.join("crates");
        let mut escaping = Vec::new();

        for source_path in rust_sources(&workspace_crates) {
            let Ok(relative_to_crates) = source_path.strip_prefix(&workspace_crates) else {
                continue;
            };
            let Some(crate_name) = relative_to_crates.components().next() else {
                continue;
            };
            let member_root = workspace_crates.join(crate_name.as_os_str());
            let source = fs::read_to_string(&source_path)
                .unwrap_or_else(|err| panic!("read {}: {err}", source_path.display()));

            for include_path in rust_include_paths(&source) {
                let resolved = normalize_path(
                    &source_path
                        .parent()
                        .expect("source file has parent")
                        .join(&include_path),
                );
                if !resolved.starts_with(&member_root) {
                    escaping.push(format!(
                        "{} includes {} outside package root",
                        source_path
                            .strip_prefix(root)
                            .unwrap_or(&source_path)
                            .display(),
                        include_path
                    ));
                }
            }
        }

        assert!(
        escaping.is_empty(),
        "package source include paths must stay inside their crate so packaged tests/benches/examples are not dead: {escaping:?}"
    );
    }
}

fn workflow_job<'a>(workflow: &'a str, job_name: &str) -> &'a str {
    let marker = format!("  {job_name}:");
    let start = workflow
        .find(&marker)
        .unwrap_or_else(|| panic!("missing workflow job {job_name}"));
    let rest = &workflow[start..];
    let mut search_start = marker.len();
    let mut end = rest.len();
    while let Some(relative) = rest[search_start..].find("\n  ") {
        let candidate = search_start + relative + 1;
        if !rest[candidate..].starts_with("    ") {
            end = candidate;
            break;
        }
        search_start = candidate + 1;
    }
    &rest[..end]
}

fn publishable_crate_dirs() -> &'static [&'static str] {
    &[
        "crates/signinum-core",
        "crates/signinum-cuda-runtime",
        "crates/signinum-profile",
        "crates/signinum-j2k-native",
        "crates/signinum-jpeg",
        "crates/signinum-tilecodec",
        "crates/signinum-j2k",
        "crates/signinum-transcode",
        "crates/signinum-jpeg-metal",
        "crates/signinum-j2k-metal",
        "crates/signinum-transcode-metal",
        "crates/signinum-jpeg-cuda",
        "crates/signinum-j2k-cuda",
        "crates/signinum-cli",
        "crates/signinum",
    ]
}

fn cargo_metadata_workspace_edges(root: &Path) -> BTreeSet<(String, String)> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version=1"])
        .current_dir(root)
        .output()
        .expect("run cargo metadata");
    assert!(
        output.status.success(),
        "cargo metadata failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let metadata =
        serde_json::from_slice::<serde_json::Value>(&output.stdout).expect("parse cargo metadata");
    let workspace_members = metadata["workspace_members"]
        .as_array()
        .expect("metadata workspace_members array")
        .iter()
        .map(|id| {
            id.as_str()
                .expect("workspace member id is string")
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    let workspace_packages = metadata["packages"]
        .as_array()
        .expect("metadata packages array")
        .iter()
        .filter(|package| {
            package["id"]
                .as_str()
                .is_some_and(|id| workspace_members.contains(id))
        })
        .filter_map(|package| package["name"].as_str())
        .collect::<BTreeSet<_>>();

    let mut edges = BTreeSet::new();
    for package in metadata["packages"]
        .as_array()
        .expect("metadata packages array")
        .iter()
        .filter(|package| {
            package["id"]
                .as_str()
                .is_some_and(|id| workspace_members.contains(id))
        })
    {
        let source = package["name"].as_str().expect("package name");
        for dependency in package["dependencies"]
            .as_array()
            .expect("package dependencies array")
            .iter()
            .filter(|dependency| dependency["kind"].is_null())
            .filter(|dependency| dependency["source"].is_null())
            .filter_map(|dependency| dependency["name"].as_str())
            .filter(|dependency| workspace_packages.contains(dependency))
        {
            edges.insert((source.to_owned(), dependency.to_owned()));
        }
    }
    edges
}

fn architecture_doc_dependency_edges(docs: &str) -> BTreeSet<(String, String)> {
    let graph_section = docs
        .split("## Crate dependency graph")
        .nth(1)
        .expect("architecture dependency graph section");
    let graph_block = graph_section
        .split("```")
        .nth(1)
        .expect("architecture dependency graph code block");
    let mut edges = BTreeSet::new();

    for line in graph_block.lines().filter(|line| line.contains("->")) {
        let (source, dependencies) = line.split_once("->").expect("graph edge line");
        let source = source.trim();
        for dependency in dependencies.split(',') {
            let dependency = dependency
                .split_whitespace()
                .next()
                .expect("graph dependency token");
            edges.insert((source.to_owned(), dependency.to_owned()));
        }
    }

    edges
}

fn format_edge(edge: &(String, String)) -> String {
    format!("{} -> {}", edge.0, edge.1)
}

fn cargo_public_api(root: &Path, package: &str) -> Option<String> {
    let output = Command::new("cargo")
        .args(["public-api", "-p", package, "--all-features"])
        .current_dir(root)
        .output();

    let output = match output {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("skipping cargo-public-api check for {package}: cargo not found");
            return None;
        }
        Err(err) => panic!("run cargo public-api for {package}: {err}"),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");

    if !output.status.success()
        && combined.contains("no such command")
        && combined.contains("public-api")
    {
        eprintln!("skipping cargo-public-api check for {package}: cargo-public-api not installed");
        return None;
    }

    assert!(
        output.status.success(),
        "cargo public-api failed for package {package}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    Some(combined)
}

fn stable_api_snapshot_section(snapshot: &str, package: &str) -> String {
    let heading = format!("## `{package}`");
    let start = snapshot
        .find(&heading)
        .unwrap_or_else(|| panic!("stable API snapshot is missing section {heading}"));
    let after_heading = &snapshot[start + heading.len()..];
    let end = after_heading.find("\n## `").unwrap_or(after_heading.len());
    after_heading[..end].to_owned()
}

fn rust_sources(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    collect_rust_sources(dir, &mut out);
    out
}

fn collect_rust_sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|err| panic!("read {}: {err}", dir.display())) {
        let entry = entry.expect("read directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rust_sources(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn repo_text_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_repo_text_files(root, &mut out);
    out
}

fn collect_repo_text_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|err| panic!("read {}: {err}", dir.display())) {
        let entry = entry.expect("read directory entry");
        let path = entry.path();
        if path.is_dir() {
            if should_skip_repo_dir(&path) {
                continue;
            }
            collect_repo_text_files(&path, out);
            continue;
        }
        if is_repo_text_file(&path) {
            out.push(path);
        }
    }
}

fn should_skip_repo_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| matches!(name, ".codewhale" | ".git" | ".venv" | "target"))
}

fn is_repo_text_file(path: &Path) -> bool {
    if path.file_name().and_then(OsStr::to_str) == Some("Cargo.lock") {
        return true;
    }
    matches!(
        path.extension().and_then(OsStr::to_str),
        Some(
            "bib"
                | "c"
                | "cc"
                | "cpp"
                | "cu"
                | "h"
                | "hpp"
                | "json"
                | "lock"
                | "md"
                | "rs"
                | "sh"
                | "tex"
                | "toml"
                | "txt"
                | "yaml"
                | "yml"
        )
    )
}

fn is_archived_handoff(path: &Path) -> bool {
    path.file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| name.starts_with("HANDOFF-"))
}

fn signinum_env_tokens(source: &str) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    for line in source.lines() {
        let mut rest = line;
        while let Some(start) = rest.find("SIGNINUM_") {
            let token_start = start;
            let token_end = rest[token_start..]
                .find(|ch: char| !(ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_'))
                .map_or(rest.len(), |end| token_start + end);
            tokens.insert(rest[token_start..token_end].to_string());
            rest = &rest[token_end..];
        }
    }
    tokens
}

fn is_internal_signinum_token(token: &str) -> bool {
    token == "SIGNINUM_"
        || token.starts_with("SIGNINUM_SIGNPOST_")
        || matches!(
            token,
            "SIGNINUM_JPEG_ZIGZAG" | "SIGNINUM_PROFILE_TEST_STAGE_MODE"
        )
}

fn referenced_shell_scripts(source: &str) -> Vec<String> {
    source
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_' | '/')))
        .filter(|token| {
            Path::new(token)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("sh"))
                && token.contains('/')
        })
        .filter(|token| !token.starts_with("http://") && !token.starts_with("https://"))
        .map(str::to_string)
        .collect()
}

fn rust_include_paths(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    for marker in ["include_bytes!(\"", "include_str!(\""] {
        let mut rest = source;
        while let Some(start) = rest.find(marker) {
            let after_marker = &rest[start + marker.len()..];
            let Some(end) = after_marker.find('"') else {
                break;
            };
            out.push(after_marker[..end].to_string());
            rest = &after_marker[end + 1..];
        }
    }
    out
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}
