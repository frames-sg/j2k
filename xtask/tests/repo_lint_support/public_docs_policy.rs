// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, ffi::OsStr, fs};

use super::{
    assert_file_pattern_checks, assert_pattern_checks, documented_j2k_env_vars,
    is_allowed_legacy_name_history_reference, is_archived_handoff, is_internal_j2k_token,
    is_repo_lint_test_source, j2k_env_tokens, normalize_path, publishable_crate_dirs,
    read_source_files, referenced_shell_scripts, repo_root, repo_text_files, rust_include_paths,
    rust_sources, xtask_sources, FilePatternCheck, PatternCheck,
};

#[test]
fn supported_j2k_env_vars_are_documented() {
    let root = repo_root();
    let docs_path = root.join("docs/env-vars.md");
    let docs = fs::read_to_string(&docs_path)
        .unwrap_or_else(|err| panic!("read {}: {err}", docs_path.display()));
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("README.md")
                .named("README environment-variable reference")
                .required(&["docs/env-vars.md"]),
            FilePatternCheck::new("docs/env-vars.md")
                .named("supported environment-variable reference")
                .required(&["| `J2K_SEMVER_TOOLCHAIN` | Rejected by `cargo xtask semver`; Rust `1.96` is pinned in source and CI. | Must not be set | Test/CI |"])
                .forbidden(&["J2K_JPEG_METAL_SPLIT_FAST420_BATCH"]),
        ],
    );
    let documented = documented_j2k_env_vars(&docs);
    assert!(
        !documented.is_empty(),
        "docs/env-vars.md must document supported J2K_* environment variables"
    );

    let mut missing = Vec::new();
    let mut referenced = BTreeSet::new();
    for path in repo_text_files(root) {
        if is_archived_handoff(&path)
            || path.ends_with("docs/env-vars.md")
            || path.ends_with("engineering/ai-codebase-audit-remediation-plan.md")
            || is_repo_lint_test_source(root, &path)
        {
            continue;
        }
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for token in j2k_env_tokens(&source) {
            referenced.insert(token.clone());
            if is_internal_j2k_token(&token) {
                continue;
            }
            if !documented.contains(&token) {
                missing.push(format!(
                    "{}: {token}",
                    path.strip_prefix(root).unwrap_or(&path).display()
                ));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "supported J2K_* environment variables must be documented in docs/env-vars.md:\n{}",
        missing.join("\n")
    );
    let stale = documented
        .difference(&referenced)
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        stale.is_empty(),
        "docs/env-vars.md documents J2K_* variables with no repo reference:\n{}",
        stale.join("\n")
    );
    for phantom in [
        "J2K_LEVEL1_CUDA_HT_MIN_MPS",
        "J2K_LEVEL1_CUDA_HT_MIN_SPEEDUP_VS_NVIDIA",
        "J2K_LEVEL2_CUDA_HT_MIN_MPS",
        "J2K_LEVEL2_CUDA_HT_MIN_SPEEDUP_VS_NVIDIA",
    ] {
        assert!(
            !documented.contains(phantom),
            "phantom GPU validation env var `{phantom}` must not be documented without an implementation"
        );
    }
}

#[test]
fn public_docs_describe_public_crate_auto_and_cuda_runtime_surface_scope() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("README.md")
                .required(&[
                    "public crate release",
                    "Runtime backend selection defaults to `Auto`",
                    "cuda-runtime",
                    "CUDA device memory",
                    "J2K-owned CUDA",
                    "NVIDIA performance",
                ])
                .forbidden(&["compatibility-only with no runtime CUDA decode"]),
            FilePatternCheck::new("docs/architecture.md").required(&[
                "public crate release",
                "Runtime backend selection defaults to `Auto`",
            ]),
            FilePatternCheck::new("docs/release.md")
                .required(&[
                    "public crate release",
                    "Runtime backend selection defaults to `Auto`",
                    "cuda-runtime",
                    "CUDA device memory",
                    "J2K-owned CUDA",
                    "NVIDIA performance",
                ])
                .forbidden(&["compatibility-only with no runtime CUDA decode"]),
            FilePatternCheck::new("CHANGELOG.md")
                .required(&[
                    "cuda-runtime",
                    "CUDA device memory",
                    "J2K-owned CUDA",
                    "NVIDIA performance",
                ])
                .forbidden(&["compatibility-only with no runtime CUDA decode"]),
        ],
    );
}

#[test]
fn historical_metal_batch_claims_are_qualified_at_the_point_of_use() {
    let root = repo_root();
    let architecture =
        fs::read_to_string(root.join("docs/architecture.md")).expect("read architecture docs");
    let claim_anchor = architecture
        .find("A July 19, 2026 local M4 Pro diagnostic run")
        .expect("architecture historical Metal batch claim");
    let claim_start = architecture[..claim_anchor]
        .rfind("\n\n")
        .map_or(0, |offset| offset + 2);
    let claim_end = architecture[claim_anchor..]
        .find("\n\n")
        .map_or(architecture.len(), |offset| claim_anchor + offset);
    let claim = &architecture[claim_start..claim_end];

    for qualification in [
        "identical encoded content",
        "decode-once broadcast",
        "not a content-distinct acceptance baseline",
    ] {
        assert!(
            claim.contains(qualification),
            "historical Metal batch claim must include {qualification:?} at the point of use"
        );
    }
}

#[test]
fn public_docs_route_users_to_current_crates() {
    let root = repo_root();
    let readme = fs::read_to_string(root.join("README.md")).expect("read README");
    assert_file_pattern_checks(
        root,
        &[FilePatternCheck::new("README.md")
            .named("README current crate routing")
            .required(&[
                "Which crate should I use?",
                "Fast Path For LLM-Assisted Use",
                "cargo add j2k",
                "statumen",
                "wsi-dicom",
                "j2k-jpeg",
                "j2k",
                "j2k-cli",
            ])],
    );

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
fn active_repo_text_does_not_reintroduce_signinum_names() {
    let root = repo_root();
    let mut offenders = Vec::new();

    for path in repo_text_files(root) {
        if is_archived_handoff(&path)
            || is_allowed_legacy_name_history_reference(root, &path)
            || is_repo_lint_test_source(root, &path)
        {
            continue;
        }
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for (line_idx, line) in source.lines().enumerate() {
            let lower = line.to_ascii_lowercase();
            if lower.contains("signinum") {
                offenders.push(format!(
                    "{}:{}:{}",
                    path.strip_prefix(root).unwrap_or(&path).display(),
                    line_idx + 1,
                    line
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "active repo text must not reintroduce signinum names after the j2k rename:\n{}",
        offenders.join("\n")
    );
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

    for crate_dir in publishable_crate_dirs(root) {
        let manifest_path = crate_dir.join("Cargo.toml");
        let manifest = fs::read_to_string(&manifest_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", manifest_path.display()));
        let readme_path = crate_dir.join("README.md");
        let package = crate_dir
            .file_name()
            .and_then(OsStr::to_str)
            .expect("publishable crate dir has UTF-8 name");

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

        let readme = fs::read_to_string(&readme_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", readme_path.display()));
        let readme_source_name = readme_path
            .strip_prefix(root)
            .unwrap_or(&readme_path)
            .display()
            .to_string();
        let docs_url = format!("https://docs.rs/{package}");
        assert_pattern_checks(
            &[PatternCheck::new(&readme_source_name, &readme).required(&[
                docs_url.as_str(),
                "https://github.com/frames-sg/j2k",
                "docs/public-support.md",
            ])],
        );
    }
}

#[test]
fn publishable_crates_configure_docs_rs_metadata() {
    let root = repo_root();

    for crate_dir in publishable_crate_dirs(root) {
        let manifest_path = crate_dir.join("Cargo.toml");
        let manifest = fs::read_to_string(&manifest_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", manifest_path.display()));

        let manifest_source_name = manifest_path
            .strip_prefix(root)
            .unwrap_or(&manifest_path)
            .display()
            .to_string();
        assert_pattern_checks(&[
            PatternCheck::new(&manifest_source_name, &manifest).required(&[
                "[package.metadata.docs.rs]",
                "all-features = true",
                "targets = []",
            ]),
        ]);
    }
}

#[test]
fn support_matrix_is_linked_and_covers_adoption_surfaces() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("README.md")
            .named("README support matrix")
            .required(&[
                "Stable APIs",
                "Experimental APIs",
                "BackendRequest::Auto",
                "Security",
                "Benchmark and parity policy",
                "MSRV",
                "OpenJPEG",
                "Grok",
            ])],
    );
}

#[test]
fn public_codec_and_transcode_examples_are_publicly_linked() {
    let root = repo_root();
    let readme = fs::read_to_string(root.join("README.md")).expect("read README");

    let examples = [
        "crates/j2k/examples/decode_generated.rs",
        "crates/j2k-jpeg/examples/inspect.rs",
        "crates/j2k-metal/examples/decode_route_report.rs",
        "crates/j2k-metal/examples/htj2k_encode_auto_report.rs",
        "crates/j2k-metal/examples/resident_encode_buffer.rs",
        "crates/j2k-tilecodec/examples/decompress.rs",
        "crates/j2k-transcode/examples/jpeg_to_htj2k.rs",
        "crates/j2k-transcode-metal/examples/jpeg_to_htj2k_route_report.rs",
    ];
    for example in examples {
        assert!(
            root.join(example).exists(),
            "expected runnable example `{example}`"
        );
    }
    assert_pattern_checks(&[
        PatternCheck::new("README public example links", &readme).required(&examples)
    ]);
}

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

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the cross-module Metal consistency checks are one public-contract policy"
)]
fn metal_consistency_cleanup_keeps_names_status_buffers_and_marker_sizes_single_sourced() {
    let root = repo_root();
    let buffer_validation =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/buffer_validation.rs"))
            .expect("read buffer validation");
    let decode_dispatch =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/decode_dispatch.rs"))
            .expect("read decode dispatch");
    let lossless_prepare =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/lossless_prepare.rs"))
            .expect("read lossless prepare");
    let tier1_encode =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/tier1_encode.rs"))
            .expect("read tier1 encode");
    let resident_codestream =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_codestream.rs"))
            .expect("read resident codestream");
    let resident_tier1 =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_tier1.rs"))
            .expect("read resident tier1");
    let resident_tier1_types =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_tier1/types.rs"))
            .expect("read resident tier1 types");
    let direct_buffers =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_buffers.rs"))
            .expect("read direct buffers");
    let direct_roi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
        .expect("read direct ROI");
    let resident_types =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_types.rs"))
            .expect("read resident types");
    let resident_packet_plan =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_packet_plan.rs"))
            .expect("read resident packet plan");
    let encode_capacity =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/encode_capacity.rs"))
            .expect("read encode capacity");
    let jpeg_extended12 = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/decoder/extended12.rs",
            "crates/j2k-jpeg/src/decoder/extended12/upsample.rs",
        ],
    );
    let split_metal_status_users = [
        "crates/j2k-metal/src/compute/decode_dispatch/classic_cleanup.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/classic_subband.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/ht_distinct.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/ht_subband.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/idwt.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/mct.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/store.rs",
        "crates/j2k-metal/src/compute/lossless_prepare/batch.rs",
        "crates/j2k-metal/src/compute/lossless_prepare/batch_item.rs",
        "crates/j2k-metal/src/compute/lossless_prepare/commands.rs",
        "crates/j2k-metal/src/compute/lossless_prepare/forward_encode.rs",
        "crates/j2k-metal/src/compute/lossless_prepare/single.rs",
        "crates/j2k-metal/src/compute/lossless_prepare/sizes.rs",
        "crates/j2k-metal/src/compute/resident_tier1/profile_dispatch/analysis.rs",
        "crates/j2k-metal/src/compute/resident_tier1/profile_dispatch/tokens.rs",
        "crates/j2k-metal/src/compute/resident_tier1/readback.rs",
        "crates/j2k-metal/src/compute/resident_tier1/result_harvest.rs",
    ]
    .into_iter()
    .map(|relative| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    })
    .collect::<Vec<_>>()
    .join("\n");
    let metal_status_users = [
        buffer_validation.as_str(),
        decode_dispatch.as_str(),
        lossless_prepare.as_str(),
        tier1_encode.as_str(),
        resident_codestream.as_str(),
        resident_tier1.as_str(),
        split_metal_status_users.as_str(),
    ]
    .join("\n");

    assert_pattern_checks(&[
        PatternCheck::new(
            "Metal resident tier1 component-count field",
            &resident_tier1_types,
        )
        .required(&["pub(crate) component_count: u8"])
        .forbidden(&["pub(crate) components: u8", "pub(crate) num_components: u8"]),
        PatternCheck::new(
            "Metal resident types component-count field",
            &resident_types,
        )
        .required(&["pub(crate) component_count: u8"])
        .forbidden(&["pub(crate) num_components: u8"]),
        PatternCheck::new(
            "Metal resident packet plan component-count field",
            &resident_packet_plan,
        )
        .required(&["pub(super) component_count: u8"])
        .forbidden(&["pub(super) num_components: u8"]),
        PatternCheck::new("Metal direct buffer helper", &direct_buffers)
            .required(&["pub(super) fn zeroed_shared_buffer", "early-returning"]),
        PatternCheck::new("Metal status readback buffer users", &metal_status_users)
            .required(&["zeroed_shared_buffer(&runtime.device"])
            .forbidden(&[
                "let status_buffer = runtime.device.new_buffer(",
                "let status_buffer = runtime.device.new_buffer_with_data(",
                "let status_buffer = runtime\n                .device\n                .new_buffer",
            ]),
        PatternCheck::new(
            "codestream capacity marker-size constants",
            &encode_capacity,
        )
        .required(&[
            "JP2K_SIZ_FIXED_BYTES",
            "JP2K_SIZ_BYTES_PER_COMPONENT",
            "JP2K_CAP_MARKER_SEGMENT_BYTES",
            "JP2K_COD_MARKER_SEGMENT_BYTES",
            "JP2K_QCD_FIXED_BYTES",
            "JP2K_TLM_MARKER_SEGMENT_BYTES",
            "JP2K_SOT_MARKER_SEGMENT_BYTES",
            "JP2K_SOD_MARKER_BYTES",
            "JP2K_EOC_MARKER_BYTES",
        ])
        .forbidden(&[
            "40usize",
            "len.checked_add(14)",
            "if job.write_tlm { 12",
            "len.checked_add(12)",
            "len.checked_add(2)",
        ]),
        PatternCheck::new("IDWT margin explanations", &direct_roi)
            .required(&["16 samples", "40 for irreversible 9/7"]),
        PatternCheck::new(
            "extended 12-bit fancy upsample rounding explanation",
            &jpeg_extended12,
        )
        .required(&["IJG/libjpeg fancy h2v2 upsampling"]),
    ]);
}

#[test]
fn metal_raw_buffer_contents_access_stays_confined_to_checked_helpers() {
    let root = repo_root();
    let allowed = BTreeSet::from([
        "crates/j2k-metal-support/src/buffer_access.rs",
        "crates/j2k-metal/src/compute/direct_buffers.rs",
        "crates/j2k-jpeg-metal/src/buffers.rs",
    ]);

    for src_root in [
        "crates/j2k-metal-support/src",
        "crates/j2k-metal/src",
        "crates/j2k-jpeg-metal/src",
        "crates/j2k-transcode-metal/src",
    ] {
        for path in rust_sources(&root.join(src_root)) {
            let rel = path
                .strip_prefix(root)
                .expect("source path under repo root")
                .to_string_lossy()
                .replace('\\', "/");
            let source =
                fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {rel}: {err}"));
            if allowed.contains(rel.as_str()) {
                continue;
            }
            assert!(
                !source.contains(".contents()"),
                "raw Metal buffer contents access must stay inside checked helpers; found in {rel}"
            );
        }
    }
}

#[test]
fn j2k_metal_bench_surface_stays_clean_after_reset() {
    let root = repo_root();
    let removed_j2k_metal_bench_command = ["cargo bench -p ", "j2k-metal", " --bench"].concat();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-metal/Cargo.toml")
                .named("J2K Metal manifest")
                .forbidden(&[
                    "[[bench]]",
                    "criterion =",
                    "j2k-compare =",
                    "name = \"device_upload\"",
                    "name = \"compare\"",
                    "name = \"encode_stages\"",
                    "name = \"decode_stages\"",
                ]),
            FilePatternCheck::new("README.md")
                .forbidden(&[removed_j2k_metal_bench_command.as_str()]),
            FilePatternCheck::new("xtask/src/main.rs")
                .forbidden(&[removed_j2k_metal_bench_command.as_str()]),
            FilePatternCheck::new("crates/j2k-compare/src/openjpeg.rs")
                .named("OpenJPEG comparator")
                .required(&["pub fn version"]),
            FilePatternCheck::new("crates/j2k-compare/src/grok.rs")
                .named("Grok comparator")
                .required(&["pub fn version", "pub fn library_path"]),
        ],
    );

    let benches_dir = root.join("crates/j2k-metal/benches");
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
            "j2k-metal benches dir must stay empty after reset: {stale_entries:?}"
        );
    }
}

#[test]
fn public_text_does_not_embed_local_user_home_paths() {
    let root = repo_root();
    let mut offenders = Vec::new();

    for path in repo_text_files(root) {
        if is_archived_handoff(&path) {
            continue;
        }
        if is_repo_lint_test_source(root, &path) {
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

#[test]
fn public_search_metadata_routes_generic_queries_to_one_landing_page() {
    let root = repo_root();
    let read = |relative: &str| {
        let path = root.join(relative);
        fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
    };

    let root_readme = read("README.md");
    let workspace_manifest = read("Cargo.toml");
    let crate_manifest = read("crates/j2k/Cargo.toml");
    let crate_readme = read("crates/j2k/README.md");
    let crate_lib = read("crates/j2k/src/lib.rs");
    let home = read("docs/index.html");
    let landing = read("docs/rust-jpeg2000-codec/index.html");
    let sitemap = read("docs/sitemap.xml");

    assert!(root_readme.starts_with("# J2K — Pure-Rust JPEG 2000 and HTJ2K Codec\n"));
    assert!(root_readme.contains("[Pure-Rust JPEG 2000 codec documentation](https://frames-sg.github.io/j2k/rust-jpeg2000-codec/)"));
    assert!(workspace_manifest
        .contains("homepage     = \"https://frames-sg.github.io/j2k/rust-jpeg2000-codec/\""));
    assert!(crate_manifest.contains("description = \"Pure-Rust JPEG 2000"));
    assert!(crate_manifest.contains("homepage.workspace = true"));
    assert!(crate_readme.contains("[Pure-Rust JPEG 2000 codec documentation](https://frames-sg.github.io/j2k/rust-jpeg2000-codec/)"));
    assert!(crate_lib.contains("//! Pure-Rust JPEG 2000"));
    assert!(landing.contains("<title>Pure-Rust JPEG 2000 Codec — J2K</title>"));
    assert!(landing.contains("<h1>Pure-Rust JPEG 2000 Codec</h1>"));
    assert!(landing.contains(
        "<link rel=\"canonical\" href=\"https://frames-sg.github.io/j2k/rust-jpeg2000-codec/\">"
    ));
    assert!(!home.contains("<title>J2K: Rust JPEG 2000 / HTJ2K Codec</title>"));
    assert!(sitemap.contains("<loc>https://frames-sg.github.io/j2k/rust-jpeg2000-codec/</loc>\n    <priority>1.0</priority>"));
}
