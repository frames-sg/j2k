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
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new(".github/workflows/ci.yml")
                .named("CI workflow docs and benchmark compile gates")
                .required(&[
                    "cargo xtask doc",
                    "cargo xtask stable-api",
                    "cargo xtask bench-build",
                    "cargo-public-api@0.52.0",
                    "macos-latest",
                ])
                .forbidden(&["macos-13"]),
            FilePatternCheck::new("xtask/src/main.rs")
                .named("xtask benchmark compile gate")
                .required(&[
                    "\"doc\"",
                    "\"--workspace\"",
                    "\"--all-features\"",
                    "\"--no-deps\"",
                    "\"j2k-jpeg-metal\"",
                    "\"j2k-metal\"",
                    "\"--no-run\"",
                ]),
        ],
    );
}

#[test]
fn ci_workflow_runs_semver_checks_for_stable_library_crates() {
    let root = repo_root();
    let workflow =
        fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read CI workflow");
    let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
    let stable_api_doc =
        fs::read_to_string(root.join("docs/stable-api-1.0.md")).expect("read stable API policy");
    let semver_job = workflow_job(&workflow, "semver");
    let semver_packages = const_array_block(&xtask, "STABLE_SEMVER_PACKAGES");

    assert_pattern_checks(&[
        PatternCheck::new("CI semver job", semver_job)
            .required(&[
                "cargo install cargo-semver-checks --version 0.48.0 --locked",
                "cargo xtask semver",
                "runs-on: macos-latest",
                "toolchain: \"1.96\"",
            ])
            .forbidden(&["release-type: minor"]),
        PatternCheck::new("xtask semver toolchain default", &xtask)
            .required(&["unwrap_or_else(|_| \"1.96\".to_string())"]),
    ]);

    for package in [
        "j2k",
        "j2k-core",
        "j2k-jpeg",
        "j2k",
        "j2k-tilecodec",
        "j2k-jpeg-metal",
        "j2k-metal",
        "j2k-jpeg-cuda",
        "j2k-cuda",
        "j2k-transcode",
        "j2k-transcode-cuda",
        "j2k-metal-support",
        "j2k-transcode-metal",
        "j2k-native",
        "j2k-cuda-runtime",
        "j2k-profile",
    ] {
        assert!(
            semver_packages.contains(&format!("\"{package}\"")),
            "repo semver policy must cover stable library crate `{package}`"
        );
    }

    for package in const_array_values(&xtask, "STABLE_SEMVER_PACKAGES") {
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
fn xtask_test_does_not_run_benchmarks_as_tests() {
    let xtask = fs::read_to_string(repo_root().join("xtask/src/main.rs")).expect("read xtask");
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
                .map(|value| value.trim_matches('"'))
                .unwrap_or("<unnamed>");
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
    let xtask = fs::read_to_string(repo_root().join("xtask/src/main.rs")).expect("read xtask");
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
    let workflow =
        fs::read_to_string(repo_root().join(".github/workflows/ci.yml")).expect("read CI workflow");
    let coverage_job = workflow_job(&workflow, "coverage");
    const INSTALL_ACTION_SHA: &str = "91534edaf9fd796a162759d80d49cdff574bff2c";

    let install_action = format!("taiki-e/install-action@{INSTALL_ACTION_SHA}");
    assert_pattern_checks(&[PatternCheck::new("CI coverage job", coverage_job)
        .required(&[
            install_action.as_str(),
            "tool: cargo-llvm-cov",
            "cargo xtask coverage",
        ])
        .forbidden(&["taiki-e/install-action@cargo-llvm-cov", "continue-on-error"])]);
}

#[test]
fn coverage_excludes_hardware_only_gpu_adapter_crates() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("xtask/src/main.rs")
                .named("coverage exclusion regex")
                .required(&[
                    "crates/j2k-cuda-runtime/",
                    "crates/j2k-cuda/",
                    "crates/j2k-.*-cuda/",
                    "crates/j2k-metal/",
                    "crates/j2k-.*-metal/",
                    "crates/j2k-metal-support/",
                ]),
            FilePatternCheck::new("docs/release.md")
                .named("GPU coverage substitute release evidence")
                .required(&[
                    "GPU-heavy changes",
                    "`gpu-validation` runs",
                    "per-backend minimum test count floors",
                    "Shared CPU-runnable GPU path code",
                ]),
        ],
    );
}

#[test]
fn metal_resident_retry_uses_typed_error_classification() {
    let root = repo_root();
    let resident_estimate =
        fs::read_to_string(root.join("crates/j2k-metal/src/encode/resident_estimate.rs"))
            .expect("read resident estimate");
    let metal_error = fs::read_to_string(root.join("crates/j2k-metal/src/error.rs"))
        .expect("read j2k-metal error source");
    let tier1_encode =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/tier1_encode.rs"))
            .expect("read j2k-metal tier1 encode source");
    let classification_sources = [
        resident_estimate.as_str(),
        metal_error.as_str(),
        tier1_encode.as_str(),
    ]
    .join("\n");

    assert_pattern_checks(&[
        PatternCheck::new("Metal resident retry decision source", &resident_estimate)
            .forbidden(&[".contains("]),
        PatternCheck::new(
            "typed Metal retry classification sources",
            &classification_sources,
        )
        .required(&[
            "MetalKernelRetryable",
            "encode_status_retry_class",
            "ResidentClassicBatch",
            "ResidentHtBatch",
            "is_conservative_retry_candidate",
        ]),
    ]);
}

#[test]
fn gpu_adapter_error_classification_uses_shared_core_impl() {
    let root = repo_root();
    let core_error =
        fs::read_to_string(root.join("crates/j2k-core/src/error.rs")).expect("read core error");
    assert_pattern_checks(&[
        PatternCheck::new("j2k-core adapter error classifier", &core_error).required(&[
            "pub enum AdapterErrorKind",
            "pub trait AdapterErrorParts",
            "adapter_error_is_unsupported",
            "adapter_error_is_buffer_error",
        ]),
    ]);
    let jpeg_metal_lib = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/lib.rs"))
        .expect("read JPEG Metal lib module");
    let jpeg_metal_decode_request =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/decode_request.rs"))
            .expect("read JPEG Metal decode request module");
    let jpeg_metal_decoder = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/decoder.rs"))
        .expect("read JPEG Metal decoder module");
    let jpeg_metal_codec_batch =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/codec_batch.rs"))
            .expect("read JPEG Metal codec batch module");
    assert!(
        jpeg_metal_lib.lines().count() < 932,
        "j2k-jpeg-metal lib.rs must keep focused public paths re-exported under the post-request-type-split line ratchet"
    );
    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal public module shell", &jpeg_metal_lib)
            .required(&[
                "mod error;",
                "pub use error::Error;",
                "mod decode_request;",
                "pub use decode_request::{MetalDecodeOp, MetalDecodeRequest};",
                "mod decoder;",
                "pub use decoder::Decoder;",
                "mod codec_batch;",
                "pub use codec_batch::{",
            ])
            .forbidden(&[
                "pub enum MetalDecodeOp",
                "pub struct MetalDecodeRequest",
                "pub enum Rgb8MetalBatchOp",
                "pub struct Decoder<'a>",
                "impl Codec {",
            ]),
        PatternCheck::new(
            "j2k-jpeg-metal decode request module",
            &jpeg_metal_decode_request,
        )
        .required(&["pub enum MetalDecodeOp", "pub struct MetalDecodeRequest"]),
        PatternCheck::new("j2k-jpeg-metal decoder module", &jpeg_metal_decoder)
            .required(&["pub struct Decoder<'a>", "impl<'a> Decoder<'a>"]),
        PatternCheck::new("j2k-jpeg-metal codec batch module", &jpeg_metal_codec_batch).required(
            &[
                "impl Codec",
                "pub enum Rgb8MetalBatchOp",
                "pub fn inspect_rgb8_decoder_batch_metal_output(",
            ],
        ),
    ]);

    let adapter_classifier_patterns = [
        "impl AdapterErrorParts for Error",
        "adapter_error_is_truncated(self)",
        "adapter_error_is_not_implemented(self)",
        "adapter_error_is_unsupported(self)",
        "adapter_error_is_buffer_error(self)",
    ];
    for relative in [
        "crates/j2k-cuda/src/error.rs",
        "crates/j2k-metal/src/error.rs",
        "crates/j2k-jpeg-cuda/src/error.rs",
        "crates/j2k-jpeg-metal/src/error.rs",
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|err| panic!("read {relative}: {err}"));
        assert_pattern_checks(&[
            PatternCheck::new(relative, &source).required(&adapter_classifier_patterns)
        ]);
    }
}

#[test]
fn decode_capability_correctness_regressions_are_guarded() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-native/src/j2c/codestream.rs")
                .named("target-resolution shrink-factor arithmetic")
                .required(&[
                    ".checked_shl(u32::from(skipped_resolution_levels))",
                    ".checked_mul(resolution_shrink_factor)",
                    "size_data.checked_image_width()?;",
                    "size_data.checked_image_height()?;",
                    "checked_image_dimensions_reject_shrink_factor_overflow",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg/tests/inspect.rs")
                .named("JPEG progressive inspect/decode agreement fixtures")
                .required(&[
                    "fn inspect_and_decoder_info_agree_for_progressive_fixtures()",
                    "progressive_8x8_jpeg()",
                    "progressive_12bit_grayscale_8x8_jpeg()",
                    "progressive_12bit_rgb_8x8_jpeg()",
                    "assert_eq!(decoder.info(), &inspected, \"{label}\");",
                ]),
        ],
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
                .required(&["macro_rules! cuda_decode_plan"])
                .forbidden(&[
                    "JpegColorFastPacket",
                    "trait JpegColorFastPacket",
                    "macro_rules! impl_color_fast_packet_access",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/compute/fast_packets_impl.rs")
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
fn gpu_decoder_cpu_host_facades_use_core_blanket_impl() {
    let root = repo_root();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-core/src/traits.rs")
                .named("j2k-core CPU-backed ImageDecode blanket impl")
                .required(&[
                    "pub trait CpuBackedImageDecode<'a>",
                    "impl<'a, T> ImageDecode<'a> for T",
                    "T: CpuBackedImageDecode<'a>",
                ]),
            FilePatternCheck::new("crates/j2k-cuda/src/decoder.rs")
                .required(&["impl<'a> CpuBackedImageDecode<'a>"])
                .forbidden(&["impl<'a> ImageDecode<'a>"]),
            FilePatternCheck::new("crates/j2k-metal/src/decoder.rs")
                .required(&["impl<'a> CpuBackedImageDecode<'a>"])
                .forbidden(&["impl<'a> ImageDecode<'a>"]),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/src/decoder.rs")
                .required(&["impl<'a> CpuBackedImageDecode<'a>"])
                .forbidden(&["impl<'a> ImageDecode<'a>"]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/decoder.rs")
                .required(&["impl<'a> CpuBackedImageDecode<'a>"])
                .forbidden(&["impl<'a> ImageDecode<'a>"]),
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
    let native_encode = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode.rs"))
        .expect("read native encode");
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
    let direct_execute =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_execute_impl.rs"))
            .expect("read Metal direct execute");
    let direct_roi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
        .expect("read Metal direct ROI");

    assert_pattern_checks(&[
        PatternCheck::new("Metal direct shell ROI module", &direct_execute)
            .required(&["mod direct_roi;"]),
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
    let direct_execute =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_execute_impl.rs"))
            .expect("read Metal direct execute");
    let direct_prepare =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_prepare.rs"))
            .expect("read Metal direct prepare");

    assert_pattern_checks(&[
        PatternCheck::new("Metal direct shell prepare module", &direct_execute)
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
    let metal_fdwt = fs::read_to_string(root.join("crates/j2k-metal/src/fdwt.metal"))
        .expect("read j2k-metal fdwt shader");
    let metal_idwt = fs::read_to_string(root.join("crates/j2k-metal/src/idwt.metal"))
        .expect("read j2k-metal idwt shader");
    let transcode_metal = fs::read_to_string(root.join("crates/j2k-transcode-metal/src/metal.rs"))
        .expect("read transcode Metal runtime");
    let transcode_dct97 =
        fs::read_to_string(root.join("crates/j2k-transcode-metal/src/dct97.metal"))
            .expect("read transcode Metal dct97 shader");
    let transcode_cpu_dct97 = fs::read_to_string(root.join("crates/j2k-transcode/src/dct97_2d.rs"))
        .expect("read transcode CPU dct97 module");
    let cuda_transcode = fs::read_to_string(
        root.join("crates/j2k-cuda-runtime/src/cuda_oxide_transcode/simt/src/main.rs"),
    )
    .expect("read CUDA Oxide transcode SIMT source");

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
        PatternCheck::new("j2k-transcode CPU DCT97 constants", &transcode_cpu_dct97)
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
        ("crates/j2k-metal/src/fdwt.metal", &metal_fdwt),
        ("crates/j2k-metal/src/idwt.metal", &metal_idwt),
        (
            "crates/j2k-transcode-metal/src/dct97.metal",
            &transcode_dct97,
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
fn jpeg_gpu_encode_host_orchestration_uses_shared_adapter_helper() {
    let root = repo_root();
    let shared = fs::read_to_string(root.join("crates/j2k-jpeg/src/adapter/baseline_encode.rs"))
        .expect("read JPEG baseline encode adapter helper");
    let cuda_encode = fs::read_to_string(root.join("crates/j2k-jpeg-cuda/src/encode.rs"))
        .expect("read JPEG CUDA encode host");
    let metal_encode = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/encode.rs"))
        .expect("read JPEG Metal encode host");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg shared GPU encode helper", &shared).required(&[
            "pub struct JpegBaselineGpuEncodeTile",
            "pub struct JpegBaselineGpuEncodeParams",
            "pub trait JpegBaselineGpuEncodeHostAdapter",
            "pub enum JpegBaselineGpuEncodeError",
            "fn validate_jpeg_baseline_gpu_encode_tile",
            "fn jpeg_baseline_gpu_encode_params",
            "fn jpeg_baseline_gpu_entropy_capacity_bytes",
            "fn same_source_buffer_batch_end",
            "pub fn encode_jpeg_baseline_gpu_tile",
            "pub fn encode_jpeg_baseline_gpu_batch",
            "while start < tiles.len()",
            "assemble_jpeg_baseline_frame(",
        ]),
    ]);

    for (relative, source) in [
        ("crates/j2k-jpeg-cuda/src/encode.rs", &cuda_encode),
        ("crates/j2k-jpeg-metal/src/encode.rs", &metal_encode),
    ] {
        assert_pattern_checks(&[PatternCheck::new(relative, source)
            .required(&[
                "JpegBaselineGpuEncodeHostAdapter",
                "encode_jpeg_baseline_gpu_tile(tile, options, &mut adapter)",
                "encode_jpeg_baseline_gpu_batch(tiles, options, &mut adapter)",
                "fn encode_tile_entropy(",
                "fn encode_batch_entropy(",
            ])
            .forbidden(&[
                "baseline_encode_tables",
                "assemble_jpeg_baseline_frame",
                "jpeg_baseline_gpu_encode_tile_plan",
                "jpeg_baseline_gpu_encode_batch_plan",
                "same_source_buffer_batch_end",
                "while start < tiles.len()",
                "validate_jpeg_baseline_dimensions",
                "jpeg_baseline_entropy_capacity_bytes",
                "checked_mul(bytes_per_pixel)",
                "let mcu_width =",
                "let mcu_height =",
                "JpegSubsampling",
            ])]);
    }
    assert!(
        cuda_encode.lines().count() < 310 && metal_encode.lines().count() < 325,
        "JPEG GPU encode adapters must stay below the post-driver line ratchets"
    );
}

#[test]
fn metal_backend_session_lifecycle_lives_in_support_crate() {
    let root = repo_root();
    let support = fs::read_to_string(root.join("crates/j2k-metal-support/src/lib.rs"))
        .expect("read Metal support crate");
    let jpeg_metal = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/lib.rs"))
        .expect("read JPEG Metal lib");
    let jpeg_metal_session = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/session.rs"))
        .expect("read JPEG Metal session module");
    let j2k_metal_session = fs::read_to_string(root.join("crates/j2k-metal/src/session.rs"))
        .expect("read J2K Metal session module");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal-support session lifecycle helper", &support).required(&[
            "pub struct MetalRuntimeSession<R, E>",
            "runtime: Arc<OnceLock<Result<R, E>>>",
            "pub fn system_default() -> Result<Self, MetalSupportError>",
            "pub fn runtime_initialized(&self) -> bool",
            "pub fn get_or_init_runtime",
        ]),
        PatternCheck::new("j2k-jpeg-metal public session re-exports", &jpeg_metal)
            .required(&["pub use session::{MetalBackendSession, MetalSession};"])
            .forbidden(&["pub struct MetalBackendSession", "pub struct MetalSession"]),
        PatternCheck::new(
            "j2k-jpeg-metal session module public types",
            &jpeg_metal_session,
        )
        .required(&["pub struct MetalBackendSession", "pub struct MetalSession"]),
    ]);

    for (relative, source) in [
        ("crates/j2k-jpeg-metal/src/session.rs", &jpeg_metal_session),
        ("crates/j2k-metal/src/session.rs", &j2k_metal_session),
    ] {
        assert_pattern_checks(&[PatternCheck::new(relative, source)
            .required(&["MetalRuntimeSession<", "runtime_session:"])
            .forbidden(&[
                "runtime: Arc<OnceLock<Result",
                "system_default_device()\n            .map(Self::new)",
            ])]);
    }
}

#[test]
fn jpeg_metal_huffman_derivation_uses_shared_entropy_canonical_tables() {
    let root = repo_root();
    let codec_math = fs::read_to_string(root.join("crates/j2k-codec-math/src/jpeg.rs"))
        .expect("read codec-math JPEG helpers");
    let entropy_huffman = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/huffman.rs"))
        .expect("read JPEG entropy Huffman implementation");
    let fast_packet = fs::read_to_string(root.join("crates/j2k-jpeg/src/adapter/fast_packet.rs"))
        .expect("read JPEG fast packet adapter");
    let metal_abi = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/abi.rs"))
        .expect("read JPEG Metal ABI");
    let cuda_runtime = fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/jpeg.rs"))
        .expect("read CUDA JPEG runtime ABI");

    assert!(
        codec_math.contains("pub fn derive_canonical_huffman")
            && codec_math.contains("pub struct CanonicalHuffmanDerivation")
            && codec_math.contains("let mut huffsize")
            && codec_math.contains("let mut huffcode"),
        "j2k-codec-math must own the Annex C canonical Huffman derivation"
    );
    assert!(
        entropy_huffman.contains("pub(crate) fn derive_canonical_huffman")
            && entropy_huffman.contains("derive_canonical_huffman(raw)?"),
        "j2k-jpeg entropy must expose and use one shared Annex C canonical Huffman derivation"
    );
    assert!(
        fast_packet.contains("pub struct JpegCanonicalHuffmanTable")
            && fast_packet.contains("pub fn derive_canonical(&self)")
            && fast_packet.contains("derive_canonical_huffman(&raw)?"),
        "j2k-jpeg adapter must expose backend-facing canonical Huffman derivation"
    );
    assert!(
        metal_abi.contains(".derive_canonical()")
            && !metal_abi.contains("let mut huffsize")
            && !metal_abi.contains("let mut huffcode")
            && !metal_abi.contains("let mut code = 0u32")
            && !metal_abi.contains("for (len_minus_1, &count) in value.bits.iter().enumerate()"),
        "JPEG Metal ABI must pack shared canonical Huffman tables instead of deriving Annex C locally"
    );
    assert!(
        cuda_runtime.contains("j2k_codec_math::jpeg::derive_canonical_huffman")
            && !cuda_runtime.contains("let mut huffsize")
            && !cuda_runtime.contains("let mut huffcode")
            && !cuda_runtime.contains("let mut code = 0u32"),
        "CUDA JPEG runtime must use shared codec-math canonical Huffman derivation"
    );
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
fn fast444_region_scaled_batches_use_shared_region_scaled_metal_path() {
    let root = repo_root();
    let fast_packets =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/fast_packets_impl.rs"))
            .expect("read JPEG Metal fast packet implementation");
    let region_plan =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/region_scaled_plan.rs"))
            .expect("read JPEG Metal region scaled plan");
    let batch_decode =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/batch_decode_region.rs"))
            .expect("read JPEG Metal region batch decoder");

    assert_pattern_checks(&[
        PatternCheck::new("fast packet region-scaled Metal trait", &fast_packets).required(&[
            "trait FastRegionScaledMetal",
            "impl FastRegionScaledMetal for JpegFast444PacketV1",
            "fn chroma_width(width: u32) -> u32",
        ]),
        PatternCheck::new("region-scaled packet-family planning", &region_plan).required(&[
            "mode: PlaneMode",
            "plane_mode_to_u32(mode)",
            "P::chroma_width(source_window.w)",
        ]),
        PatternCheck::new("fast444 RGB region-scaled batch path", &batch_decode)
            .required(&[
                "try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast444PacketV1>",
            ])
            .forbidden(&[
                "fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces_with_output(",
                "fn try_decode_fast444_restart_region_scaled_rgb_batch_to_surfaces_with_output(",
                "fn try_decode_grouped_fast444_region_scaled_rgb_batch_to_surfaces_with_output(",
            ]),
    ]);
}

#[test]
fn fast444_full_batches_use_shared_fastsubsampled_metal_path() {
    let root = repo_root();
    let fast_packets =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/fast_packets_impl.rs"))
            .expect("read JPEG Metal fast packet implementation");
    let batch_decode =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/batch_decode_full.rs"))
            .expect("read JPEG Metal full batch decoder");

    assert_pattern_checks(&[
        PatternCheck::new("fast444 shared FastSubsampledMetal impl", &fast_packets)
            .required(&["impl FastSubsampledMetal for JpegFast444PacketV1"]),
        PatternCheck::new(
            "fast444 full shared region-scaled batch path",
            &batch_decode,
        )
        .required(&[
            "fn fast444_full_region_scaled_requests(",
            "scale: j2k_core::Downscale::None",
            "try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<",
            "JpegFast444PacketV1",
            "try_decode_fast444_region_scaled_rgba_batch_to_textures(",
        ])
        .forbidden(&[
            "struct Fast444FullRgbSurfaceShape",
            "struct Fast444FullRgbaTextureShape",
            "fn fast444_full_packets(",
            "fn try_decode_grouped_fast444_full_rgb_batch_to_surfaces_with_output(",
            "fn try_decode_grouped_fast444_full_rgba_batch_to_textures(",
            "fn encode_fast444_full_rgba_texture_decode(",
            "fast444_full_entropy",
        ]),
    ]);
}

#[test]
fn jpeg_fast420_profiled_decode_uses_shared_scan_loop() {
    let root = repo_root();
    let sequential = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential.rs"))
        .expect("read JPEG entropy sequential decoder");

    assert_pattern_checks(&[PatternCheck::new(
        "JPEG fast420 profiled/unprofiled scan paths",
        &sequential,
    )
    .required(&[
        "trait Fast420ScanProfiler",
        "struct NoopFast420ScanProfile",
        "impl Fast420ScanProfiler for BenchFast420Profile",
        "fn decode_scan_fast_tile_rgb_impl",
        "decode_scan_fast_tile_rgb_impl(plan, backend, scan_bytes, pool, writer, &mut profile)",
        "decode_scan_fast_tile_rgb_impl(plan, backend, scan_bytes, pool, writer, profile)",
        "fast_tile_profiled_rgb_matches_unprofiled_decode",
    ])
    .forbidden(&["let mcu_start = Instant::now();"])]);
    assert_eq!(
        sequential.matches("finish_fast_tile_scan(&mut br)").count(),
        1,
        "JPEG fast420 profiled/unprofiled scan paths must not duplicate the scan loop"
    );
}

#[test]
fn cuda_htj2k_compact_jobs_use_shared_planner() {
    let root = repo_root();
    let htj2k_encode = fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/htj2k_encode.rs"))
        .expect("read CUDA runtime HTJ2K encode module");
    let runtime_tests = fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/tests.rs"))
        .expect("read CUDA runtime tests");

    assert_pattern_checks(&[PatternCheck::new(
        "CUDA HTJ2K compact planner implementation",
        &htj2k_encode,
    )
    .required(&[
        "trait Htj2kCompactPlanJob",
        "impl Htj2kCompactPlanJob for CudaHtj2kEncodeKernelJob",
        "impl Htj2kCompactPlanJob for CudaHtj2kEncodeMultiInputKernelJob",
        "fn htj2k_encode_compact_jobs_impl<J: Htj2kCompactPlanJob>",
        "htj2k_encode_compact_jobs_impl(statuses, kernel_jobs)",
    ])]);
    assert_eq!(
        htj2k_encode.matches("let source_end =").count(),
        1,
        "compact output-range validation must live in one planner"
    );
    assert_pattern_checks(&[
        PatternCheck::new("CUDA HTJ2K compact planner tests", &runtime_tests).required(&[
            "assert_compact_jobs_match_for_single_and_multi_input",
            "htj2k_encode_compact_jobs_accept_empty_batches",
            "htj2k_encode_compact_jobs_accept_exact_capacity_payloads",
            "htj2k_encode_compact_jobs_reject_payloads_larger_than_capacity",
            "htj2k_encode_compact_jobs_pack_actual_payloads",
        ]),
    ]);
}

#[test]
fn native_classic_and_ht_parallel_copyback_share_one_helper() {
    let root = repo_root();
    let decode = fs::read_to_string(root.join("crates/j2k-native/src/j2c/decode.rs"))
        .expect("read native J2K decode module");

    assert_pattern_checks(&[PatternCheck::new(
        "native classic/HT decoded-block copyback",
        &decode,
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
fn cuda_oxide_simt_helpers_use_shared_prelude() {
    let root = repo_root();
    let prelude =
        fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/cuda_oxide_simt_prelude.rs"))
            .expect("read CUDA Oxide SIMT prelude");
    let build_script = fs::read_to_string(root.join("crates/j2k-cuda-runtime/build.rs"))
        .expect("read CUDA runtime build script");
    let unsafe_audit =
        fs::read_to_string(root.join("docs/unsafe-audit.md")).expect("read unsafe audit");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA Oxide SIMT prelude", &prelude).required(&[
            "fn simt_load<T: Copy>",
            "fn simt_store<T>",
            "fn simt_ptr_at<T>",
            "fn simt_mut_ptr_at<T>",
            "SAFETY: CUDA-Oxide kernels pass validated device buffers",
        ]),
    ]);
    assert_pattern_checks(&[
        PatternCheck::new("CUDA runtime SIMT prelude build dependency", &build_script).required(&[
            "cargo:rerun-if-changed=src/cuda_oxide_simt_prelude.rs",
            "stage_cuda_oxide_shared_prelude(out_dir);",
            "out_dir.join(\"cuda_oxide_simt_prelude.rs\")",
        ]),
        PatternCheck::new(
            "unsafe audit CUDA Oxide SIMT prelude invariants",
            &unsafe_audit,
        )
        .required(&[
            "cuda_oxide_simt_prelude.rs",
            "Shared cuda-oxide SIMT pointer prelude",
        ]),
    ]);

    let mut simt_sources = rust_sources(&root.join("crates/j2k-cuda-runtime/src"))
        .into_iter()
        .filter(|path| {
            path.ends_with(Path::new("simt/src/main.rs"))
                && path.components().any(|component| {
                    component
                        .as_os_str()
                        .to_string_lossy()
                        .starts_with("cuda_oxide_")
                })
        })
        .collect::<Vec<_>>();
    simt_sources.sort();
    assert!(
        simt_sources.len() >= 10,
        "expected all CUDA Oxide SIMT kernel sources to be discovered"
    );

    for path in simt_sources {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        let relative = path.strip_prefix(root).unwrap_or(&path).display();
        let relative_name = relative.to_string();
        assert_pattern_checks(&[PatternCheck::new(&relative_name, &source)
            .required(&["include!(\"../../../cuda_oxide_simt_prelude.rs\");"])]);

        if source.contains("fn load_")
            || source.contains("fn store_")
            || source.contains("fn offset_")
            || source.contains("pub unsafe fn j2k_copy_u8")
        {
            assert!(
                source.contains("simt_load")
                    || source.contains("simt_store")
                    || source.contains("simt_ptr_at")
                    || source.contains("simt_mut_ptr_at"),
                "{relative} helper wrappers must delegate to the shared SIMT prelude"
            );
        }

        assert_pattern_checks(&[PatternCheck::new(&relative_name, &source).forbidden(&[
            "unsafe { *ptr.add",
            "unsafe { ptr.add",
            "unsafe { *ptr }",
            "*dst.add(",
            "*src.add(",
            "*decoded_data.add(",
        ])]);
    }
}

#[test]
fn backend_surfaces_use_core_metadata_and_residency() {
    let root = repo_root();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-core/src/accelerator.rs")
                .named("j2k-core accelerator contracts")
                .required(&[
                    "pub struct SurfaceMetadata",
                    "pub enum SurfaceResidency",
                    "pub pitch_bytes: usize",
                    "pub byte_offset: usize",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/lib.rs")
                .named("JPEG Metal lib module")
                .required(&["mod surface;", "pub use surface::{"])
                .forbidden(&[
                    "pub struct Surface",
                    "pub struct MetalBatchOutputBuffer",
                    "pub struct MetalBatchTextureOutput",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/surface.rs")
                .named("JPEG Metal surface module")
                .required(&[
                    "pub struct Surface",
                    "pub struct MetalBatchOutputBuffer",
                    "pub struct MetalBatchTextureOutput",
                    "pub struct MetalTextureTile",
                    "pub struct ResidentPrivateJpegTile",
                ]),
        ],
    );

    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-cuda/src/surface.rs")
                .required(&["SurfaceMetadata", "fn metadata(&self)"])
                .forbidden(&["pub enum SurfaceResidency"]),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/src/surface.rs")
                .required(&["SurfaceMetadata", "fn metadata(&self)"])
                .forbidden(&["pub enum SurfaceResidency"]),
            FilePatternCheck::new("crates/j2k-metal/src/surface.rs")
                .required(&["SurfaceMetadata", "fn metadata(&self)"])
                .forbidden(&["pub enum SurfaceResidency"]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/surface.rs")
                .required(&["SurfaceMetadata", "fn metadata(&self)"])
                .forbidden(&["pub enum SurfaceResidency"]),
            FilePatternCheck::new("crates/j2k-cuda/src/surface.rs")
                .required(&["pub use j2k_core::SurfaceResidency;"]),
            FilePatternCheck::new("crates/j2k-metal/src/lib.rs")
                .required(&["pub use j2k_core::SurfaceResidency;"]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/lib.rs")
                .required(&["pub use j2k_core::SurfaceResidency;"]),
        ],
    );
}

#[test]
fn copied_test_fixture_helpers_live_in_shared_support() {
    let root = repo_root();
    let test_support = fs::read_to_string(root.join("crates/j2k-test-support/src/lib.rs"))
        .expect("read j2k-test-support");
    let compare = fs::read_to_string(root.join("crates/j2k-compare/src/encode_compare.rs"))
        .expect("read compare encode module");
    let dct97 = fs::read_to_string(root.join("crates/j2k-transcode/src/dct97_2d.rs"))
        .expect("read transcode 9/7 DCT module");
    let dct97_test = fs::read_to_string(root.join("crates/j2k-transcode/tests/dct97_2d.rs"))
        .expect("read transcode 9/7 DCT test");
    let dwt_diff = fs::read_to_string(root.join("crates/j2k-transcode/tests/support/dwt_diff.rs"))
        .expect("read transcode DWT diff test support");
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
        PatternCheck::new("transcode integration DWT diff helper", &dwt_diff).required(&[
            "pub(crate) fn max_abs_diff_53(",
            "pub(crate) fn max_abs_diff_97(",
            "fn max_abs_diff_bands(",
        ]),
        PatternCheck::new(
            "9/7 transcode integration test shared diff helper",
            &dct97_test,
        )
        .required(&["mod dwt_diff;", "max_abs_diff_97(&"])
        .forbidden(&["fn max_abs_diff("]),
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
fn metal_compute_runtime_registry_is_split_from_compute_god_file() {
    let root = repo_root();
    let compute = fs::read_to_string(root.join("crates/j2k-metal/src/compute.rs"))
        .expect("read Metal compute module");
    let runtime = fs::read_to_string(root.join("crates/j2k-metal/src/compute/runtime.rs"))
        .expect("read Metal compute runtime module");
    let forward_transform =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/forward_transform.rs"))
            .expect("read Metal compute forward-transform module");
    let resident_tier1 =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_tier1.rs"))
            .expect("read Metal compute resident tier1 module");
    let lossless_prepare =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/lossless_prepare.rs"))
            .expect("read Metal compute lossless prepare module");
    let decode_dispatch =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/decode_dispatch.rs"))
            .expect("read Metal compute decode dispatch module");
    let tier1_encode =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/tier1_encode.rs"))
            .expect("read Metal compute tier1 encode module");
    let resident_codestream =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_codestream.rs"))
            .expect("read Metal compute resident codestream module");
    let resident_codestream_ht_cleanup = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/resident_codestream/ht_cleanup.rs"),
    )
    .expect("read Metal compute resident codestream HT cleanup module");
    let resident_codestream_classic_labels = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/resident_codestream/classic_labels.rs"),
    )
    .expect("read Metal compute resident codestream classic labels module");
    let decode_cleanup =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/decode_cleanup.rs"))
            .expect("read Metal compute decode cleanup module");

    assert!(
        compute.lines().count() < 390,
        "compute.rs must stay below the post-split line-count ratchet"
    );
    assert!(
        resident_codestream.lines().count() < 2_785,
        "resident_codestream.rs must stay below the post-classic-label split line-count ratchet"
    );
    assert!(
        resident_codestream.contains("mod ht_cleanup;")
            && resident_codestream_ht_cleanup
                .contains("pub(in crate::compute) fn dispatch_ht_cleanup")
            && !resident_codestream.contains("fn dispatch_ht_cleanup("),
        "resident_codestream HT cleanup dispatch helpers must live in resident_codestream/ht_cleanup.rs"
    );
    assert!(
        resident_codestream.contains("mod classic_labels;")
            && resident_codestream_classic_labels.contains("CLASSIC_TIER1_DENSITY_LABEL")
            && resident_codestream_classic_labels.contains("next_enabled_classic_stage_label")
            && !resident_codestream.contains("const CLASSIC_TIER1_DENSITY_LABEL")
            && !resident_codestream.contains("fn next_enabled_classic_stage_label("),
        "resident_codestream classic profiling labels must live in resident_codestream/classic_labels.rs"
    );

    assert_pattern_checks(&[
        PatternCheck::new("Metal compute runtime module shell", &compute)
            .required(&[
                "mod runtime;",
                "pub(crate) use self::runtime",
                "MetalRuntime",
                "runtime_initialization_error",
            ])
            .forbidden(&[
                "pub(crate) struct MetalRuntime",
                "MetalPipelineLoader::new(device",
            ]),
        PatternCheck::new("Metal compute runtime implementation", &runtime).required(&[
            "pub(crate) struct MetalRuntime",
            "MetalPipelineLoader::new(device",
            "DEFAULT_METAL_SESSION",
            "METAL_RUNTIME_OVERRIDE",
            "with_runtime_for_session",
        ]),
    ]);
    for (module_wire, module_source, owned_item) in [
        (
            "mod forward_transform;",
            &forward_transform,
            "pub(crate) fn encode_forward_dwt53",
        ),
        (
            "mod resident_tier1;",
            &resident_tier1,
            "pub(crate) struct J2kLosslessDeviceCodeBlock",
        ),
        (
            "mod lossless_prepare;",
            &lossless_prepare,
            "pub(crate) fn prepare_lossless_device_code_blocks",
        ),
        (
            "mod decode_dispatch;",
            &decode_dispatch,
            "pub(crate) fn decode_inverse_mct",
        ),
        (
            "mod tier1_encode;",
            &tier1_encode,
            "pub(crate) fn encode_classic_tier1_code_blocks",
        ),
        (
            "mod resident_codestream;",
            &resident_codestream,
            "pub(crate) fn encode_tier2_packetization",
        ),
        (
            "mod decode_cleanup;",
            &decode_cleanup,
            "pub(crate) fn decode_classic_cleanup_code_block",
        ),
    ] {
        assert_pattern_checks(&[
            PatternCheck::new("Metal compute module wiring", &compute).required(&[module_wire]),
            PatternCheck::new("split Metal compute module owned item", module_source)
                .required(&[owned_item]),
            PatternCheck::new("Metal compute module shell owned-item exclusion", &compute)
                .forbidden(&[owned_item]),
        ]);
    }
}

#[test]
fn native_encode_options_and_tile_parts_live_in_focused_modules() {
    let root = repo_root();
    let encode = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode.rs"))
        .expect("read native encode module");
    let options = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/options.rs"))
        .expect("read native encode options module");
    let tile_parts =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/tile_parts.rs"))
            .expect("read native encode tile-part module");
    let precomputed =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed.rs"))
            .expect("read native encode precomputed module");
    let packet_plan =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/packet_plan.rs"))
            .expect("read native encode packet-plan module");
    let rate_control =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/rate_control.rs"))
            .expect("read native encode rate-control module");
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

    assert!(
        encode.lines().count() < 3_900,
        "j2c/encode.rs must stay below the post-split line-count ratchet"
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
            "pub use self::precomputed::*;",
            "mod packet_plan;",
            "mod rate_control;",
            "mod samples;",
            "mod i64_packetize;",
            "mod single_tile;",
        ]),
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
        "pub(super) fn validate_precomputed_dwt97_geometry",
    ];
    assert_pattern_checks(&[
        PatternCheck::new(
            "j2c/encode/precomputed.rs precomputed helpers",
            &precomputed,
        )
        .required(&precomputed_helpers),
        PatternCheck::new("j2c/encode.rs precomputed helper exclusion", &encode)
            .forbidden(&precomputed_helpers),
        PatternCheck::new("precomputed DWT adapter forwarding macro", &precomputed)
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
            &precomputed,
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
fn jpeg_decoder_view_and_output_format_live_in_focused_modules() {
    let root = repo_root();
    let decoder = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder.rs"))
        .expect("read JPEG decoder module");
    let view = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/view.rs"))
        .expect("read JPEG decoder view module");
    let output_format =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/output_format.rs"))
            .expect("read JPEG decoder output-format module");
    let extended12 = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/extended12.rs"))
        .expect("read JPEG decoder extended12 module");
    let lossless = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/lossless_helpers.rs"))
        .expect("read JPEG decoder lossless helper module");
    let lossless_region =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/lossless_region.rs"))
            .expect("read JPEG decoder lossless region module");
    let color_convert =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/color_convert.rs"))
            .expect("read JPEG decoder color-convert module");
    let core_traits = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/core_traits.rs"))
        .expect("read JPEG decoder core-traits module");
    let scratch = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/scratch.rs"))
        .expect("read JPEG decoder scratch module");
    let sink_writer = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/sink_writer.rs"))
        .expect("read JPEG decoder sink-writer module");
    let bench_support = fs::read_to_string(root.join("crates/j2k-jpeg/src/bench_support.rs"))
        .expect("read JPEG bench support module");

    assert!(
        decoder.lines().count() < 3_985,
        "decoder.rs must stay below the post-split line-count ratchet"
    );
    assert!(
        decoder.contains("fn decode_lossless_output_format_region_scaled(")
            && decoder.contains("self.lossless_plan.as_ref()?;")
            && decoder.matches("if self.lossless_plan.is_some()").count() == 1,
        "decoder.rs must route output-format lossless dispatch through one shared helper"
    );
    assert!(
        decoder.contains("fn decode_lossless_color8_output_into(")
            && decoder.contains("fn decode_lossless_color16_output_into(")
            && decoder
                .matches("match lossless_color_sampling(&self.info)")
                .count()
                == 2,
        "decoder.rs must keep lossless RGB/YCbCr sampling dispatch shared by bit depth"
    );
    assert!(
        lossless_region.contains("pub(super) enum LosslessRgbRegionFallback")
            && lossless_region.contains("YCbCr8")
            && lossless_region.contains("Rgb8")
            && lossless_region.contains("YCbCr16")
            && lossless_region.contains("Rgb16")
            && lossless_region.contains("decode_rgb_region_scaled_into(")
            && lossless_region.contains("decode_rgba_region_scaled_into(")
            && !decoder.contains("enum LosslessRgbRegionFallback")
            && !decoder.contains("fn decode_lossless_rgb_region_scaled_into(")
            && !decoder.contains("fn decode_lossless_rgba8_region_into(")
            && !decoder.contains("fn decode_lossless_rgba16_region_scaled_into("),
        "decoder.rs must keep lossless region fallback routing on the focused helper module"
    );

    assert_pattern_checks(&[
        PatternCheck::new("decoder.rs view module shell", &decoder)
            .required(&["mod view;", "pub use self::view::JpegView;"])
            .forbidden(&[
                "pub struct JpegView",
                "impl<'a> JpegView<'a>",
                "parse_header(input)?",
            ]),
        PatternCheck::new("decoder/view.rs parsed-view API", &view).required(&[
            "pub struct JpegView",
            "impl<'a> JpegView<'a>",
            "pub fn parse(",
            "pub fn parse_with_options(",
            "pub fn passthrough_candidate(",
            "pub fn restart_index(",
        ]),
        PatternCheck::new("decoder.rs output-format module shell", &decoder).required(&[
            "mod output_format;",
            "output_format_from_parts",
            "checked_output_geometry",
        ]),
    ]);
    let output_format_patterns = [
        "pub(super) fn output_format_profile_name",
        "pub(super) fn downscale_profile_name",
        "pub(super) fn jpeg_downscale",
        "pub(super) fn output_format_from_parts",
        "pub(super) fn allocate_output_buffer",
        "pub(super) fn scaled_dimensions",
        "pub(super) fn scaled_rect_covering",
        "pub(super) fn checked_output_geometry",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder.rs output-format helper exclusion", &decoder)
            .forbidden(&output_format_patterns),
        PatternCheck::new("decoder/output_format.rs helpers", &output_format)
            .required(&output_format_patterns),
        PatternCheck::new("decoder.rs focused helper module wiring", &decoder).required(&[
            "mod extended12;",
            "mod lossless_helpers;",
            "mod lossless_region;",
            "mod color_convert;",
            "mod core_traits;",
            "mod scratch;",
            "mod sink_writer;",
        ]),
    ]);
    let extended12_patterns = [
        "pub(super) enum Extended12Output",
        "pub(super) struct Extended12WriteRegion",
        "pub(super) fn decode_extended12_color_planes",
        "pub(super) fn render_progressive12_color_planes",
        "pub(super) trait UpsampleSample",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/extended12.rs helpers", &extended12)
            .required(&extended12_patterns),
        PatternCheck::new("decoder.rs extended12 helper exclusion", &decoder)
            .forbidden(&extended12_patterns),
    ]);
    let lossless_patterns = [
        "pub(super) fn restart_index_for_stream",
        "pub(super) fn consume_lossless_restart",
        "pub(super) struct LosslessRestartTracker",
        "pub(super) struct Extended12RestartTracker",
        "pub(super) fn validate_lossless_color_plan",
        "pub(super) fn decode_lossless_plane_sample",
        "pub(super) fn decode_lossless_color_sample<P, T>",
        "pub(super) struct LosslessColorIntoSample",
        "pub(super) struct LosslessColorRowSample",
        "pub(super) fn decode_lossless_sampled_color_mcu<P>",
        "pub(super) struct LosslessSampledColorPlanesMut",
        "pub(super) struct LosslessSampledMcu",
        "pub(super) fn write_lossless_color16_sampled_output",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/lossless_helpers.rs helpers", &lossless)
            .required(&lossless_patterns),
        PatternCheck::new("decoder.rs lossless helper exclusion", &decoder)
            .forbidden(&lossless_patterns),
    ]);
    assert!(
        decoder.contains("Extended12RestartTracker::new(self.plan.restart_interval, total_mcus)")
            && extended12
                .contains("Extended12RestartTracker::new(plan.restart_interval, total_mcus)")
            && !decoder.contains("consume_extended12_restart(")
            && !extended12.contains("consume_extended12_restart("),
        "extended-12 restart cadence must be centralized through Extended12RestartTracker"
    );
    assert!(
        decoder.matches("validate_lossless_color_plan::<P>").count() == 3,
        "decoder.rs lossless color paths must share validation through decoder/lossless_helpers.rs"
    );
    let color_convert_patterns = [
        "pub(super) fn merged_warnings",
        "pub(super) fn convert_ycbcr8_to_rgb8_in_place",
        "pub(super) fn copy_rgb16_scaled_rect",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/color_convert.rs helpers", &color_convert)
            .required(&color_convert_patterns),
        PatternCheck::new("decoder.rs color-convert helper exclusion", &decoder)
            .forbidden(&color_convert_patterns),
    ]);
    let scratch_patterns = [
        "pub(super) fn compute_decode_scratch_bytes",
        "pub(super) fn compute_lossless_scratch_bytes",
        "pub(super) fn compute_extended12_planes_scratch_bytes",
        "pub(super) fn checked_scratch_len",
        "pub(super) fn checked_usize_product",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/scratch.rs helpers", &scratch).required(&scratch_patterns),
        PatternCheck::new("decoder.rs scratch helper exclusion", &decoder)
            .forbidden(&scratch_patterns),
    ]);
    let core_trait_patterns = [
        "impl ImageCodec for Decoder<'_>",
        "impl TileBatchDecode for JpegCodec",
        "pub(super) struct CroppedWriter",
        "impl<W: ComponentRowWriter + ?Sized> OutputWriter for &mut W",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/core_traits.rs trait adapters", &core_traits)
            .required(&core_trait_patterns)
            .forbidden(&["ComponentWriterAdapter"]),
        PatternCheck::new("decoder.rs core trait adapter exclusion", &decoder)
            .forbidden(&core_trait_patterns),
        PatternCheck::new("decoder.rs component writer adapter exclusion", &decoder)
            .forbidden(&["ComponentWriterAdapter"]),
        PatternCheck::new("decoder.rs sink writer re-export", &decoder)
            .required(&["pub(crate) use self::sink_writer::SinkWriter;"]),
    ]);
    let sink_writer_patterns = [
        "pub(crate) struct SinkWriter",
        "pub(crate) fn into_rows",
        "impl<S> InterleavedRgbWriter for SinkWriter<'_, S>",
        "impl<S> OutputWriter for SinkWriter<'_, S>",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/sink_writer.rs helpers", &sink_writer)
            .required(&sink_writer_patterns),
        PatternCheck::new("decoder.rs sink writer helper exclusion", &decoder)
            .forbidden(&sink_writer_patterns),
        PatternCheck::new("bench profile shared sink writer reuse", &bench_support)
            .required(&[
                "struct BlackBoxRowSink",
                "impl RowSink<u8> for BlackBoxRowSink",
                "SinkWriter::new(&mut sink, rows, dec.backend)",
            ])
            .forbidden(&[
                "struct BenchProfileSinkWriter",
                "impl InterleavedRgbWriter for BenchProfileSinkWriter",
                "impl OutputWriter for BenchProfileSinkWriter",
            ]),
    ]);
}

#[test]
fn cuda_encode_api_and_resident_types_live_in_focused_modules() {
    let root = repo_root();
    let encode = fs::read_to_string(root.join("crates/j2k-cuda/src/encode.rs"))
        .expect("read CUDA encode module");
    let api = fs::read_to_string(root.join("crates/j2k-cuda/src/encode/api.rs"))
        .expect("read CUDA encode API module");
    let htj2k = fs::read_to_string(root.join("crates/j2k-cuda/src/encode/htj2k.rs"))
        .expect("read CUDA encode HTJ2K module");
    let packetization =
        fs::read_to_string(root.join("crates/j2k-cuda/src/encode/packetization.rs"))
            .expect("read CUDA encode packetization module");
    let resident = fs::read_to_string(root.join("crates/j2k-cuda/src/encode/resident.rs"))
        .expect("read CUDA encode resident module");
    let stage = fs::read_to_string(root.join("crates/j2k-cuda/src/encode/stage.rs"))
        .expect("read CUDA encode stage module");

    let api_helpers = [
        "pub fn encode_j2k_lossless_with_cuda(",
        "pub fn encode_j2k_lossless_with_cuda_and_profile(",
        "pub(super) fn strict_cuda_encode_options",
        "pub(super) fn reject_non_cuda_encode_backend",
    ];
    let resident_types = [
        "pub struct CudaLosslessEncodeTile",
        "pub struct CudaLosslessEncodeResidency",
        "pub struct CudaLosslessEncodeOutcome",
        "pub struct CudaResidentCodestreamBuffer",
        "pub struct CudaEncodedJ2k",
        "pub struct CudaLosslessBufferEncodeOutcome",
        "pub struct SubmittedJ2kLosslessCudaEncode",
        "pub struct SubmittedJ2kLosslessCudaEncodeBatch",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("CUDA encode API module shell", &encode)
            .required(&[
                "mod api;",
                "pub use self::api::{encode_j2k_lossless_with_cuda",
                "strict_cuda_encode_options",
            ])
            .forbidden(&api_helpers),
        PatternCheck::new("CUDA encode API helper ownership", &api).required(&api_helpers),
        PatternCheck::new("CUDA encode resident module shell", &encode)
            .required(&[
                "mod resident;",
                "pub use self::resident",
                "CudaLosslessEncodeTile",
            ])
            .forbidden(&resident_types),
        PatternCheck::new("CUDA encode resident type ownership", &resident)
            .required(&resident_types),
    ]);
    assert!(
        encode.lines().count() < 3_000,
        "j2k-cuda encode.rs must stay below the post-split god-file threshold"
    );
    let packetization_items = [
        "pub(super) struct CudaHtj2kPacketizationPlan",
        "pub(super) fn flatten_cuda_htj2k_packetization_job",
        "pub(super) fn cuda_packetization_packets",
        "pub(super) fn cuda_packetization_tag_nodes",
    ];
    let stage_items = [
        "pub struct CudaEncodeStageAccelerator",
        "pub struct CudaEncodeStageTimings",
        "impl J2kEncodeStageAccelerator for CudaEncodeStageAccelerator",
    ];
    let htj2k_items = [
        "pub(super) fn cuda_encode_ht_code_block",
        "pub(super) fn cuda_encode_htj2k_tile_body",
        "pub(super) fn cuda_encode_htj2k_device_tile_body",
        "pub(super) fn cuda_encode_ht_subband",
        "fn cuda_packetize_tile_body",
        "pub(super) fn cuda_htj2k_encode_tables",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("CUDA encode focused module shell", &encode).required(&[
            "mod packetization;",
            "mod stage;",
            "pub use self::stage::{CudaEncodeStageAccelerator",
            "mod htj2k;",
        ]),
        PatternCheck::new("CUDA encode packetization exclusion", &encode)
            .forbidden(&packetization_items),
        PatternCheck::new("CUDA encode stage exclusion", &encode).forbidden(&stage_items),
        PatternCheck::new("CUDA encode HTJ2K runtime exclusion", &encode).forbidden(&htj2k_items),
        PatternCheck::new("CUDA encode packetization ownership", &packetization)
            .required(&packetization_items),
        PatternCheck::new("CUDA encode stage ownership", &stage).required(&stage_items),
        PatternCheck::new("CUDA encode HTJ2K runtime ownership", &htj2k).required(&htj2k_items),
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
    let batch_items = [
        "pub fn jpeg_to_htj2k_batch",
        "pub(super) fn jpeg_tile_batch_to_htj2k_with_scratch",
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
        PatternCheck::new("jpeg_to_htj2k batch item ownership", &batch).required(&batch_items),
    ]);
}

#[test]
fn transcode_gpu_auto_threshold_policy_is_documented() {
    let root = repo_root();
    let cuda = fs::read_to_string(root.join("crates/j2k-transcode-cuda/src/lib.rs"))
        .expect("read CUDA transcode adapter");
    let metal = fs::read_to_string(root.join("crates/j2k-transcode-metal/src/lib.rs"))
        .expect("read Metal transcode adapter");
    let cuda_readme = fs::read_to_string(root.join("crates/j2k-transcode-cuda/README.md"))
        .expect("read CUDA transcode README");
    let metal_readme = fs::read_to_string(root.join("crates/j2k-transcode-metal/README.md"))
        .expect("read Metal transcode README");

    let shared_auto_batch_thresholds = [
        "const DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_JOBS: usize = 32;",
        "const DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_SAMPLES: usize = 224 * 224 * 32;",
        "const DEFAULT_AUTO_DWT97_BATCH_MIN_JOBS: usize = 32;",
        "const DEFAULT_AUTO_DWT97_BATCH_MIN_SAMPLES: usize = 224 * 224 * 32;",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("CUDA transcode Auto batch thresholds", &cuda)
            .required(&shared_auto_batch_thresholds),
        PatternCheck::new("Metal transcode Auto batch thresholds", &metal)
            .required(&shared_auto_batch_thresholds),
        PatternCheck::new("CUDA transcode Auto threshold rationale", &cuda)
            .required(&["Batch thresholds below intentionally match Metal"]),
        PatternCheck::new("CUDA transcode README threshold rationale", &cuda_readme).required(&[
            "shared `224 * 224` component-sample floor",
            "defaults are routing policy, not a speedup promise",
        ]),
        PatternCheck::new("Metal transcode Auto threshold policy", &metal).required(&[
            "single-job Auto dispatch is disabled",
            "const DEFAULT_AUTO_DWT97_MIN_SAMPLES: usize = usize::MAX;",
            "const DEFAULT_AUTO_REVERSIBLE_MIN_SAMPLES: usize = usize::MAX;",
            "const MAX_AUTO_DWT97_STAGED_BATCH_AXIS: usize = 1024;",
        ]),
        PatternCheck::new("Metal transcode README staged-axis policy", &metal_readme).required(&[
            "either tile axis exceeds 1024 samples",
            "defaults are routing policy, not a speedup promise",
        ]),
    ]);
}

#[test]
fn transcode_stage_counters_are_shared_between_gpu_adapters() {
    let root = repo_root();
    let accelerator = fs::read_to_string(root.join("crates/j2k-transcode/src/accelerator.rs"))
        .expect("read transcode accelerator contracts");
    let cuda = fs::read_to_string(root.join("crates/j2k-transcode-cuda/src/lib.rs"))
        .expect("read CUDA transcode adapter");
    let metal = fs::read_to_string(root.join("crates/j2k-transcode-metal/src/lib.rs"))
        .expect("read Metal transcode adapter");

    assert_pattern_checks(&[PatternCheck::new(
        "j2k-transcode accelerator shared counters",
        &accelerator,
    )
    .required(&[
        "pub struct DctToWaveletStageCounters",
        "pub enum DctToWaveletStageCounterEvent",
        "pub enum TranscodeStageDispatchMode",
        "pub const fn unavailable<T>",
        "pub fn recover<T, E>",
        "pub fn record(&mut self, event: DctToWaveletStageCounterEvent, count: usize)",
        "DctToWaveletStageCounterEvent::Htj2k97CodeblockBatchAttempt",
        "DctToWaveletStageCounterEvent::Htj2k97CodeblockBatchDispatch",
    ])]);

    for (label, source) in [("CUDA", cuda.as_str()), ("Metal", metal.as_str())] {
        let check_name = format!("{label} transcode shared counters and dispatch policy");
        assert_pattern_checks(&[PatternCheck::new(&check_name, source)
            .required(&[
                "DctToWaveletStageCounterEvent as CounterEvent",
                "counters: DctToWaveletStageCounters",
                "self.counters.record(CounterEvent::",
                "mode: TranscodeStageDispatchMode",
                "self.mode.unavailable()",
                ".recover(error, |error| error.is_recoverable())",
            ])
            .forbidden(&[
                "reversible_dwt53_attempts: usize",
                "dwt53_attempts: usize",
                "dwt97_attempts: usize",
                "htj2k97_codeblock_batch_attempts: usize",
                "enum CudaDispatchMode",
                "enum MetalDispatchMode",
                "fn unavailable<T>(&self)",
                "MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_)",
            ])]);
    }
}

#[test]
fn metal_direct_plan_types_live_in_focused_module() {
    let root = repo_root();
    let direct_execute =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_execute_impl.rs"))
            .expect("read Metal direct execute implementation");
    let plan_types =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_plan_types.rs"))
            .expect("read Metal direct plan types module");
    let plane_pack =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_plane_pack.rs"))
            .expect("read Metal direct plane-pack module");
    let prepare = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_prepare.rs"))
        .expect("read Metal direct prepare module");
    let roi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
        .expect("read Metal direct ROI module");
    let grayscale_execute =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_grayscale_execute.rs"))
            .expect("read Metal direct grayscale executor module");
    let stacked_batch =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_stacked_batch.rs"))
            .expect("read Metal direct stacked batch module");
    let surface_pack =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_surface_pack.rs"))
            .expect("read Metal direct surface-pack module");

    assert!(
        direct_execute.lines().count() < 200,
        "direct_execute_impl.rs must remain an index shell after the direct split"
    );

    assert_pattern_checks(&[PatternCheck::new(
        "direct_execute_impl.rs direct plan type module shell",
        &direct_execute,
    )
    .required(&[
        "mod direct_plan_types;",
        "pub(crate) use self::direct_plan_types",
        "PreparedDirectColorPlan",
    ])]);
    for item in [
        "pub(crate) struct PreparedDirectGrayscalePlan",
        "pub(crate) struct PreparedDirectColorPlan",
        "pub(super) enum PreparedDirectGrayscaleStep",
        "pub(super) struct PreparedDirectIdwt",
        "pub(super) struct PreparedClassicSubBand",
        "pub(super) struct PreparedClassicSubBandGroup",
        "pub(super) struct PreparedHtSubBand",
        "pub(super) struct PreparedHtSubBandGroup",
    ] {
        assert_pattern_checks(&[
            PatternCheck::new(
                "direct_execute_impl.rs direct plan type exclusion",
                &direct_execute,
            )
            .forbidden(&[item]),
            PatternCheck::new(
                "compute/direct_plan_types.rs direct plan type ownership",
                &plan_types,
            )
            .required(&[item]),
        ]);
    }
    for required in [
        (
            "mod direct_plane_pack;",
            &plane_pack,
            "pub(super) struct PlaneStage",
        ),
        (
            "mod direct_prepare;",
            &prepare,
            "pub(crate) fn prepare_direct_grayscale_plan",
        ),
        (
            "mod direct_roi;",
            &roi,
            "pub(crate) fn crop_prepared_direct_grayscale_plan_to_output_region",
        ),
        (
            "mod direct_grayscale_execute;",
            &grayscale_execute,
            "pub(super) fn encode_prepared_direct_component_plane_in_command_buffer",
        ),
        (
            "mod direct_stacked_batch;",
            &stacked_batch,
            "pub(super) fn encode_stacked_direct_component_plane_batch",
        ),
        (
            "mod direct_surface_pack;",
            &surface_pack,
            "pub(super) fn output_shape_for",
        ),
    ] {
        let (module_wire, module_source, owned_item) = required;
        assert_pattern_checks(&[
            PatternCheck::new(
                "direct_execute_impl.rs split module wiring",
                &direct_execute,
            )
            .required(&[module_wire]),
            PatternCheck::new("direct split module owned item", module_source)
                .required(&[owned_item]),
            PatternCheck::new(
                "direct_execute_impl.rs owned item exclusion",
                &direct_execute,
            )
            .forbidden(&[owned_item]),
        ]);
    }
}

#[test]
fn metal_public_error_lives_in_focused_module() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs"))
        .expect("read j2k-metal lib module");
    let error = fs::read_to_string(root.join("crates/j2k-metal/src/error.rs"))
        .expect("read j2k-metal error module");

    let error_items = [
        "pub enum Error",
        "pub enum MetalDirectFallbackReason",
        "pub enum MetalKernelRetryClass",
        "impl AdapterErrorParts for Error",
        "impl CodecError for Error",
    ];
    let error_helpers = [
        "adapter_error_is_truncated",
        "adapter_error_is_not_implemented",
        "adapter_error_is_unsupported",
        "adapter_error_is_buffer_error",
        "is_conservative_retry_candidate",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal error module shell", &lib)
            .required(&[
                "mod error;",
                "pub use self::error::{Error, MetalDirectFallbackReason, MetalKernelRetryClass};",
            ])
            .forbidden(&error_items),
        PatternCheck::new("j2k-metal error item ownership", &error).required(&error_items),
        PatternCheck::new("j2k-metal error classification helpers", &error)
            .required(&error_helpers),
    ]);
}

#[test]
fn metal_surface_lives_in_focused_module() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs"))
        .expect("read j2k-metal lib module");
    let surface = fs::read_to_string(root.join("crates/j2k-metal/src/surface.rs"))
        .expect("read j2k-metal surface module");

    let surface_items = [
        "pub struct Surface",
        "pub(crate) enum Storage",
        "impl Surface",
        "impl DeviceSurface for Surface",
        "fn checked_storage_range",
    ];
    let surface_helpers = [
        "SurfaceMetadata::new",
        "copy_tight_pixels_to_strided_output",
        "DeviceMemoryRange::new",
        "from_metal_buffer_with_offset",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal surface module shell", &lib)
            .required(&["mod surface;", "pub use self::surface::Surface;"])
            .forbidden(&surface_items),
        PatternCheck::new("j2k-metal surface item ownership", &surface).required(&surface_items),
        PatternCheck::new("j2k-metal surface helper ownership", &surface)
            .required(&surface_helpers),
    ]);
}

#[test]
fn metal_sessions_and_direct_plan_caches_live_in_focused_module() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs"))
        .expect("read j2k-metal lib module");
    let session = fs::read_to_string(root.join("crates/j2k-metal/src/session.rs"))
        .expect("read j2k-metal session module");

    let session_items = [
        "pub struct MetalBackendSession",
        "pub struct MetalSession",
        "struct DirectGrayPlanCacheEntry",
        "struct DirectColorPlanCacheEntry",
        "const DIRECT_PLAN_CACHE_CAP",
        "fn evict_one_direct_plan_if_needed",
        "pub(crate) fn record_submit",
    ];
    let session_helpers = [
        "pub(crate) fn direct_plan_cache_key",
        "pub(crate) fn direct_gray_plan_cache_key",
        "pub(crate) fn cached_session_direct_gray_plan",
        "pub(crate) fn store_session_direct_gray_plan",
        "pub(crate) fn cached_session_direct_color_plan",
        "pub(crate) fn store_session_direct_color_plan",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal session module shell", &lib)
            .required(&[
                "mod session;",
                "pub use self::session::{MetalBackendSession, MetalSession};",
            ])
            .forbidden(&session_items),
        PatternCheck::new("j2k-metal session item ownership", &session).required(&session_items),
        PatternCheck::new("j2k-metal direct-plan cache helper ownership", &session)
            .required(&session_helpers),
    ]);
}

#[test]
fn metal_tile_batch_lives_in_focused_module() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs"))
        .expect("read j2k-metal lib module");
    let tile_batch = fs::read_to_string(root.join("crates/j2k-metal/src/tile_batch.rs"))
        .expect("read j2k-metal tile batch module");

    let tile_batch_items = [
        "pub struct MetalTileBatch",
        "impl MetalTileBatch",
        "pub fn push_tile_request(",
        "pub fn push_shared_tile_request(",
        "pub fn decode_all(",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal tile batch module shell", &lib)
            .required(&[
                "mod tile_batch;",
                "pub use self::tile_batch::MetalTileBatch;",
            ])
            .forbidden(&tile_batch_items),
        PatternCheck::new("j2k-metal tile batch item ownership", &tile_batch)
            .required(&tile_batch_items),
    ]);
}

#[test]
fn metal_decoder_api_lives_in_focused_module() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs"))
        .expect("read j2k-metal lib module");
    let decoder = fs::read_to_string(root.join("crates/j2k-metal/src/decoder.rs"))
        .expect("read j2k-metal decoder module");

    assert!(
        lib.lines().count() < 300,
        "j2k-metal lib.rs must stay below 300 lines after the item 53 split"
    );
    let decoder_items = [
        "pub struct J2kDecoder",
        "pub struct Codec",
        "pub enum DecodeOperation",
        "pub struct DecodeRouteReport",
        "pub struct DecodeSurfaceWithReport",
        "fn upload_surface(",
        "pub(crate) fn decode_to_surface_impl",
        "macro_rules! define_ensure_prepared_direct_plan",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal decoder module shell", &lib)
            .required(&["mod decoder;", "pub use self::decoder::{", "J2kDecoder"])
            .forbidden(&decoder_items),
        PatternCheck::new("j2k-metal decoder item ownership", &decoder).required(&decoder_items),
    ]);
}

#[test]
fn metal_batch_heuristics_live_in_focused_module() {
    let root = repo_root();
    let batch = fs::read_to_string(root.join("crates/j2k-metal/src/batch.rs"))
        .expect("read j2k-metal batch module");
    let heuristics = fs::read_to_string(root.join("crates/j2k-metal/src/batch/heuristics.rs"))
        .expect("read j2k-metal batch heuristics module");

    let heuristic_items = [
        "pub(super) enum BatchRoute",
        "pub(super) struct GroupedRequests",
        "pub(super) fn group_metal_requests",
        "pub(super) fn profile_route_label",
        "pub(super) fn is_region_scaled_direct_batch_candidate",
        "pub(super) fn should_auto_use_metal_for_region_scaled_direct_batch",
        "pub(super) fn can_decode_requests_as_repeated_region_scaled_batch",
    ];
    let heuristic_required = [
        "pub(super) enum BatchRoute",
        "pub(super) struct GroupedRequests",
        "pub(super) fn group_metal_requests",
        "pub(super) fn profile_route_label",
        "pub(super) fn is_region_scaled_direct_batch_candidate",
        "pub(super) fn should_auto_use_metal_for_region_scaled_direct_batch",
        "pub(super) fn can_decode_requests_as_repeated_region_scaled_batch",
        "AUTO_REGION_SCALED_DIRECT_BATCH64_MIN_DIM",
        "REGION_SCALED_DIRECT_FORMATS",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal batch heuristic module shell", &batch)
            .required(&[
                "mod heuristics;",
                "use self::heuristics::{",
                "group_metal_requests",
            ])
            .forbidden(&heuristic_items),
        PatternCheck::new("j2k-metal batch heuristic ownership", &heuristics)
            .required(&heuristic_required),
    ]);
}

#[test]
fn metal_batch_cpu_fallback_lives_in_focused_module() {
    let root = repo_root();
    let batch = fs::read_to_string(root.join("crates/j2k-metal/src/batch.rs"))
        .expect("read j2k-metal batch module");
    let cpu = fs::read_to_string(root.join("crates/j2k-metal/src/batch/cpu.rs"))
        .expect("read j2k-metal batch CPU module");

    let cpu_items = [
        "pub(super) fn decode_cpu_host_batch",
        "fn decode_cpu_full_batch",
        "fn decode_cpu_region_scaled_batch",
        "fn checked_cpu_batch_surface",
        "fn host_surface",
        "decode_tiles_into",
        "decode_tiles_region_scaled_into",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal batch CPU fallback module shell", &batch)
            .required(&["mod cpu;", "use self::cpu::decode_cpu_host_batch;"])
            .forbidden(&cpu_items),
        PatternCheck::new("j2k-metal batch CPU fallback ownership", &cpu).required(&cpu_items),
    ]);
}

#[test]
fn metal_batch_execute_lives_in_focused_module() {
    let root = repo_root();
    let batch = fs::read_to_string(root.join("crates/j2k-metal/src/batch.rs"))
        .expect("read j2k-metal batch module");
    let execute = fs::read_to_string(root.join("crates/j2k-metal/src/batch/execute.rs"))
        .expect("read j2k-metal batch execute module");

    let execute_items = [
        "pub(super) fn process_batch",
        "fn process_batch_inner",
        "fn complete_cpu_host_fallback",
        "fn complete_batch_surfaces",
        "fn profile_completed_outcome",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal batch execute module shell", &batch)
            .required(&["mod execute;", "use self::execute::process_batch;"])
            .forbidden(&execute_items),
        PatternCheck::new("j2k-metal batch execute ownership", &execute).required(&execute_items),
    ]);
    assert_eq!(
        execute
            .matches("session.completed[request.output_slot] = Some(Ok(surface));")
            .count(),
        1,
        "batch execution must use one shared successful-completion block"
    );
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

    assert_pattern_checks(
        &[
            PatternCheck::new("j2k-native hidden adapter exports", &lib).required(&[
                "#[doc(hidden)]\npub use backend::",
                "#[doc(hidden)]\npub use direct_plan::",
                "#[doc(hidden)]\npub use ht_adapter::",
                "#[doc(hidden)]\npub use j2k_types::",
                "#[doc(hidden)]\npub fn forward_dwt53_reference",
                "#[doc(hidden)]\npub fn decode_j2k_code_block_scalar",
                "pub struct DecodeSettings",
                "pub struct Image",
            ]),
        ],
    );
}

#[test]
fn jpeg_metal_batch_decode_is_split_by_axis() {
    let root = repo_root();
    let compute = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute.rs"))
        .expect("read j2k-jpeg-metal compute");
    let monolith =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/batch_decode_impl.rs"))
            .expect("read j2k-jpeg-metal batch decode placeholder");

    let chunks = [
        (
            "compute/batch_decode_full.rs",
            "try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output",
            "full-image RGB batch path",
        ),
        (
            "compute/batch_decode_region.rs",
            "try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output",
            "region-scaled RGB batch path",
        ),
        (
            "compute/batch_decode_entry.rs",
            "decode_full_batch_to_surfaces",
            "public batch decode entry points",
        ),
    ];

    let mut previous_idx = 0;
    for (file, required_symbol, description) in chunks {
        let include = format!("include!(\"{file}\")");
        let idx = compute
            .find(&include)
            .unwrap_or_else(|| panic!("j2k-jpeg-metal compute.rs must include `{file}`"));
        assert!(
            idx >= previous_idx,
            "j2k-jpeg-metal batch decode chunk `{file}` must be included in source order"
        );
        previous_idx = idx;

        let source = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src").join(file))
            .unwrap_or_else(|_| panic!("read j2k-jpeg-metal batch decode chunk `{file}`"));
        let check_name = format!("j2k-jpeg-metal batch decode chunk {description}");
        assert_pattern_checks(&[
            PatternCheck::new(&check_name, &source).required(&[required_symbol])
        ]);
    }

    assert!(
        monolith.lines().count() < 50 && !monolith.contains("fn try_decode_"),
        "j2k-jpeg-metal batch_decode_impl.rs must remain a small split-file pointer"
    );
}

#[test]
fn jpeg_metal_viewport_plane_rows_use_shared_target() {
    let root = repo_root();
    let viewport_cache =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/viewport_cache.rs"))
            .expect("read j2k-jpeg-metal viewport cache");

    assert_pattern_checks(&[PatternCheck::new(
        "j2k-jpeg-metal viewport row writers",
        &viewport_cache,
    )
    .required(&[
        "struct PlaneRowTarget<'a>",
        "impl ComponentRowWriter for PlaneStage",
        "impl ComponentRowWriter for ViewportPlaneWriter<'_>",
        "impl PlaneRowTarget<'_>",
        "fn write_plane_row(&self, buffer: &Buffer, y: u32, src: &[u8]) -> Result<(), Error>",
        "fn checked_write_row_u8_at(",
        "checked_copy_bytes_to_buffer_at(",
    ])
    .normalized_required(&[
        "self.row_target() .write_gray_row(y, gray_row) .map_err(jpeg_plane_write_error)",
        "self.row_target() .write_ycbcr_row(y, y_row, chroma_blue_row, chroma_red_row) .map_err(jpeg_plane_write_error)",
        "self.row_target() .write_rgb_row(y, r_row, g_row, b_row) .map_err(jpeg_plane_write_error)",
    ])
    .forbidden(&["fn write_row_u8(", ".contents()"])]);
    assert!(
        viewport_cache
            .matches("fn row_target(&self) -> PlaneRowTarget<'_>")
            .count()
            == 2,
        "PlaneStage and ViewportPlaneWriter must both delegate through PlaneRowTarget"
    );
}

#[test]
fn jpeg_decoder_owned_outputs_use_decode_request() {
    let root = repo_root();
    let decoder = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder.rs"))
        .expect("read j2k-jpeg decoder");
    let lib =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/lib.rs")).expect("read j2k-jpeg lib");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg owned-output request API", &decoder).required(&[
            "pub struct DecodeRequest",
            "pub const fn full(fmt: PixelFormat) -> Self",
            "pub const fn scaled(fmt: PixelFormat, scale: Downscale) -> Self",
            "pub const fn region(fmt: PixelFormat, region: Rect) -> Self",
            "pub const fn region_scaled(fmt: PixelFormat, region: Rect, scale: Downscale) -> Self",
            "pub fn decode_request(",
            "fn decode_request_with_scratch(",
        ]),
        PatternCheck::new("j2k-jpeg owned-output wrapper removal", &decoder).forbidden(&[
            "pub fn decode(&self, fmt: PixelFormat)",
            "pub fn decode_scaled(",
            "pub fn decode_with_scratch(",
            "pub fn decode_scaled_with_scratch(",
            "pub fn decode_region(",
            "pub fn decode_region_scaled(",
            "pub fn decode_region_with_scratch(",
            "pub fn decode_region_scaled_with_scratch(",
        ]),
        PatternCheck::new("j2k-jpeg DecodeRequest re-export", &lib).required(&[
            "DecodeOutcome, DecodeRequest",
            "DecodedTile, Decoder, JpegView",
        ]),
    ]);
}

#[test]
fn jpeg_metal_single_decode_uses_request_api() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/lib.rs"))
        .expect("read j2k-jpeg-metal lib");
    let codec_batch = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/codec_batch.rs"))
        .expect("read j2k-jpeg-metal codec batch module");
    let decode_request =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/decode_request.rs"))
            .expect("read j2k-jpeg-metal decode request module");
    let decoder = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/decoder.rs"))
        .expect("read j2k-jpeg-metal decoder module");
    let tile_batch = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/tile_batch.rs"))
        .expect("read j2k-jpeg-metal tile batch module");
    let source = format!("{lib}\n{codec_batch}\n{decode_request}\n{decoder}\n{tile_batch}");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal request API routing", &source)
            .required(&[
                "pub enum MetalDecodeOp",
                "pub struct MetalDecodeRequest",
                "pub fn decode_request_to_device(",
                "pub fn push_tile_request(",
                "pub fn push_shared_tile_request(",
                "pub fn submit_tile_request_to_device(",
                "self.push_shared_tile_request(",
                "Self::submit_tile_request_to_device(",
                "MetalDecodeRequest::region_scaled(fmt, roi, scale, backend)",
            ])
            .forbidden(&[
                "pub fn decode_region_scaled_to_device(",
                "pub fn push_tile(",
                "pub fn push_shared_tile(",
                "pub fn push_tile_region(",
                "pub fn push_shared_tile_region(",
                "pub fn push_tile_scaled(",
                "pub fn push_shared_tile_scaled(",
                "pub fn push_tile_region_scaled(",
                "pub fn push_shared_tile_region_scaled(",
            ]),
    ]);
    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal tile batch module shell", &lib)
            .required(&["mod tile_batch;", "pub use tile_batch::JpegTileBatch;"])
            .forbidden(&["pub struct JpegTileBatch"]),
        PatternCheck::new("j2k-jpeg-metal tile batch ownership", &tile_batch)
            .required(&["pub struct JpegTileBatch", "impl JpegTileBatch"]),
        PatternCheck::new("j2k-jpeg-metal decoder module shell", &lib)
            .required(&["mod decoder;", "pub use decoder::Decoder;"])
            .forbidden(&["pub struct Decoder<'a>"]),
        PatternCheck::new("j2k-jpeg-metal decoder ownership", &decoder)
            .required(&["pub struct Decoder<'a>", "impl<'a> Decoder<'a>"]),
        PatternCheck::new("j2k-jpeg-metal codec batch module shell", &lib)
            .required(&["mod codec_batch;", "pub use codec_batch::{"])
            .forbidden(&["impl Codec {", "pub enum Rgb8MetalBatchOp"]),
        PatternCheck::new("j2k-jpeg-metal codec batch ownership", &codec_batch)
            .required(&[
                "impl Codec",
                "pub enum Rgb8MetalBatchOp",
                "pub fn submit_tile_request_to_device(",
                "pub fn decode_rgb8_batch_into_buffer_with_session(",
            ])
            .forbidden(&["pub fn submit_tile_region_scaled_to_device("]),
        PatternCheck::new("j2k-jpeg-metal decode request module shell", &lib)
            .required(&[
                "mod decode_request;",
                "pub use decode_request::{MetalDecodeOp, MetalDecodeRequest};",
            ])
            .forbidden(&["pub enum MetalDecodeOp", "pub struct MetalDecodeRequest"]),
        PatternCheck::new("j2k-jpeg-metal decode request ownership", &decode_request)
            .required(&["pub enum MetalDecodeOp", "pub struct MetalDecodeRequest"]),
    ]);
}

#[test]
fn jpeg_decoder_upsample_sample_width_twins_use_generic_helpers() {
    let root = repo_root();
    let extended12 = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/extended12.rs"))
        .expect("read JPEG decoder extended12 helpers");
    let lossless = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/lossless_helpers.rs"))
        .expect("read JPEG decoder lossless helpers");
    let decoder = format!("{extended12}\n{lossless}");

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
    let direct_execute =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_execute_impl.rs"))
            .expect("read Metal direct execute");
    let direct_prepare =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_prepare.rs"))
            .expect("read Metal direct prepare");
    let direct_roi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
        .expect("read Metal direct ROI");
    let hybrid =
        fs::read_to_string(root.join("crates/j2k-metal/src/hybrid.rs")).expect("read hybrid");
    let decoder = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/extended12.rs"))
        .expect("read JPEG decoder extended12 helpers");
    let neon = fs::read_to_string(root.join("crates/j2k-jpeg/src/backend/neon.rs"))
        .expect("read JPEG NEON backend");
    let native_idwt = fs::read_to_string(root.join("crates/j2k-native/src/j2c/idwt.rs"))
        .expect("read native IDWT");

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
        PatternCheck::new(
            "Metal direct twin-unification module shell",
            &direct_execute,
        )
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
fn j2k_metal_decode_and_tile_batch_use_request_api() {
    let root = repo_root();
    let lib =
        fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs")).expect("read j2k-metal lib");
    let decoder = fs::read_to_string(root.join("crates/j2k-metal/src/decoder.rs"))
        .expect("read j2k-metal decoder");
    let tile_batch = fs::read_to_string(root.join("crates/j2k-metal/src/tile_batch.rs"))
        .expect("read j2k-metal tile batch");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal decoder request API routing", &decoder)
            .required(&[
                "pub enum MetalDecodeOp",
                "pub struct MetalDecodeRequest",
                "pub fn decode_request_to_device(",
                "pub fn decode_request_to_device_with_report(",
                "pub fn decode_request_to_device_with_session(",
                "pub fn decode_request_to_host_surface(",
                "pub fn decode_request_to_cpu_staged_metal_surface_with_session(",
                "let request = MetalDecodeRequest::region_scaled(fmt, roi, scale, backend);",
                "request.op.batch_op()",
            ])
            .forbidden(&[
                "pub fn decode_to_device_with_report(",
                "pub fn decode_region_to_device_with_report(",
                "pub fn decode_scaled_to_device_with_report(",
                "pub fn decode_region_scaled_to_device_with_report(",
                "pub fn decode_to_device_with_session(",
                "pub fn decode_region_to_device_with_session(",
                "pub fn decode_scaled_to_device_with_session(",
                "pub fn decode_region_scaled_to_device_with_session(",
                "pub fn decode_to_host_surface(",
                "pub fn decode_region_to_host_surface(",
                "pub fn decode_scaled_to_host_surface(",
                "pub fn decode_region_scaled_to_host_surface(",
                "pub fn decode_to_cpu_staged_metal_surface_with_session(",
                "pub fn decode_region_to_cpu_staged_metal_surface_with_session(",
                "pub fn decode_scaled_to_cpu_staged_metal_surface_with_session(",
                "pub fn decode_region_scaled_to_cpu_staged_metal_surface_with_session(",
            ]),
        PatternCheck::new("j2k-metal decode request type re-export", &lib)
            .required(&["MetalDecodeOp", "MetalDecodeRequest"]),
    ]);
    assert_pattern_checks(&[PatternCheck::new(
        "j2k-metal tile batch request API routing",
        &tile_batch,
    )
    .required(&[
        "pub fn push_tile_request(",
        "pub fn push_shared_tile_request(",
        "self.push_shared_tile_request(",
    ])
    .forbidden(&[
        "pub fn push_tile(",
        "pub fn push_shared_tile(",
        "pub fn push_tile_region(",
        "pub fn push_shared_tile_region(",
        "pub fn push_tile_scaled(",
        "pub fn push_shared_tile_scaled(",
        "pub fn push_tile_region_scaled(",
        "pub fn push_shared_tile_region_scaled(",
    ])]);
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
fn compare_bins_use_library_common_helpers() {
    let root = repo_root();
    let common = fs::read_to_string(root.join("crates/j2k-compare/src/common.rs"))
        .expect("read j2k-compare common library module");
    let lib = fs::read_to_string(root.join("crates/j2k-compare/src/lib.rs"))
        .expect("read j2k-compare lib");
    let fixture = fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare.rs"))
        .expect("read fixture compare module");
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
    let encode = fs::read_to_string(root.join("crates/j2k-compare/src/encode_compare.rs"))
        .expect("read encode compare module");
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
        fixture.lines().count() < 2_295,
        "fixture_compare.rs must stay below the post-gate-split line-count ratchet"
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
        PatternCheck::new("fixture_compare shared batch-size helper use", &fixture)
            .required(&[
                "use crate::{common,",
                "common::batch_size_config_from_env(",
                "J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES",
            ])
            .forbidden(&[
                "mod common;",
                "struct BatchSizeConfig",
                "fn batch_size_config_from_values",
                "fn legacy_batch_sizes_from_env",
            ]),
        PatternCheck::new("encode_compare shared batch-size helper use", &encode)
            .required(&[
                "use crate::{common,",
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
    let native =
        fs::read_to_string(root.join("crates/j2k-native/src/lib.rs")).expect("read native lib");
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
    let native =
        fs::read_to_string(root.join("crates/j2k-native/src/lib.rs")).expect("read native lib");
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
