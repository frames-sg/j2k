// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::{assert_file_pattern_checks, repo_root, FilePatternCheck};

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
fn accelerator_support_and_benchmark_evidence_have_single_document_owners() {
    let root = repo_root();
    let readme = fs::read_to_string(root.join("README.md")).expect("read workspace README");
    let architecture =
        fs::read_to_string(root.join("docs/architecture.md")).expect("read architecture docs");
    let public_support = fs::read_to_string(root.join("docs/public-support.md"))
        .expect("read public support matrix");
    let ml = fs::read_to_string(root.join("docs/j2k-ml.md")).expect("read Burn integration docs");
    let evidence = fs::read_to_string(root.join("docs/benchmark-evidence.md"))
        .expect("read benchmark evidence");
    let metal_telemetry = fs::read_to_string(root.join("crates/j2k-ml/benches/metal_telemetry.rs"))
        .expect("read Metal telemetry source");
    let normalized_ml = ml.split_whitespace().collect::<Vec<_>>().join(" ");

    for (name, document) in [("README", &readme), ("architecture", &architecture)] {
        for canonical in ["public-support.md", "j2k-ml.md", "benchmark-evidence.md"] {
            assert!(document.contains(canonical), "{name} must link {canonical}");
        }
        for duplicated_evidence in [
            "RTX 4070 hardware validation",
            "hardware-validated direct Metal batch matrix",
            "A July 19, 2026 local M4 Pro diagnostic run",
            "52 of 64 batch-32 cases",
        ] {
            assert!(
                !document.contains(duplicated_evidence),
                "{name} must not duplicate mutable accelerator evidence {duplicated_evidence:?}"
            );
        }
    }

    assert!(public_support.contains("## Owned batch codec boundary"));
    assert!(public_support.contains("j2k-ml.md"));
    assert!(normalized_ml.contains("not a Cartesian batch-greater-than-one validation"));
    assert!(!ml.contains("A July 19, 2026 local M4 Pro diagnostic run"));

    for qualification in [
        "identical encoded content",
        "decode-once broadcast",
        "none of those rows is an acceptance baseline",
    ] {
        assert!(
            evidence.contains(qualification),
            "benchmark evidence must retain historical qualification {qualification:?}"
        );
    }

    let snapshot = metal_telemetry
        .find("let before = metal_snapshot(decoder);")
        .expect("Metal snapshot before codec telemetry");
    let timer = metal_telemetry
        .find("let start = Instant::now();")
        .expect("Metal codec telemetry timer");
    assert!(
        snapshot < timer,
        "Metal telemetry snapshot must precede timing"
    );
    for timing_boundary in [
        "runtime and pipeline initialization before the timed interval",
        "cold for prepared-plan and execution-arena caches",
        "first prepared decode includes its immutable-arena upload",
    ] {
        assert!(
            normalized_ml.contains(timing_boundary),
            "Burn integration docs must disclose {timing_boundary:?}"
        );
    }
}

#[test]
fn metal_batch_api_and_benchmark_docs_match_the_implemented_contract() {
    let root = repo_root();
    let decoder = fs::read_to_string(root.join("crates/j2k-metal/src/batch_decoder/decoder.rs"))
        .expect("read Metal batch decoder");
    let external = fs::read_to_string(root.join("crates/j2k-metal/src/batch_decoder/external.rs"))
        .expect("read Metal external batch decoder");
    let ml = fs::read_to_string(root.join("docs/j2k-ml.md")).expect("read Burn integration docs");
    let architecture =
        fs::read_to_string(root.join("docs/architecture.md")).expect("read architecture docs");

    assert!(decoder.contains(
        "Decode one homogeneous shared codec group using the preparation policy captured by the group."
    ));
    assert!(!decoder.contains("using strict session defaults"));

    assert!(external.contains(
        "Decode one prepared homogeneous Gray, RGB, or RGBA group with native U8, U16, or I16 samples"
    ));
    assert!(!external.contains("Gray or unsigned RGB group"));

    for group_id in [
        "j2k_owned_batch_codec_cpu/input_{distinct|repeated}",
        "j2k_owned_batch_burn_cpu/input_{distinct|repeated}",
    ] {
        assert!(ml.contains(group_id), "Burn docs must name {group_id}");
    }
    assert!(ml.contains("`prepare_images` rows report images per second"));
    assert!(ml.contains("Decode rows report decoded spatial pixels per second"));
    assert!(!ml.contains("Criterion group names include `flex`"));

    assert!(architecture.contains("codec-side raw Objective-C resource-construction boundary"));
    assert!(architecture.contains("separate audited raw-handle adoption boundary"));
    assert!(!architecture.contains("sole raw Objective-C resource-construction boundary"));
}

#[test]
fn metal_failure_attribution_is_documented_without_guessing_a_source() {
    let root = repo_root();
    let readme = fs::read_to_string(root.join("README.md")).expect("read workspace README");
    let public_support = fs::read_to_string(root.join("docs/public-support.md"))
        .expect("read public support matrix");
    let ml = fs::read_to_string(root.join("docs/j2k-ml.md")).expect("read Burn integration docs");
    let contracts =
        fs::read_to_string(root.join("crates/j2k-metal/src/batch_decoder/contracts.rs"))
            .expect("read Metal batch contracts");
    let submission =
        fs::read_to_string(root.join("crates/j2k-metal/src/batch_decoder/submission.rs"))
            .expect("read Metal batch submission contracts");

    assert!(readme.contains("indexed preparation failures"));
    assert!(readme.contains("homogeneous group execution failures"));
    assert!(!readme.contains("returns source indices and indexed failures"));
    for required in [
        "Device classic and HT codec jobs retain their original source identity",
        "Metal command-buffer completion failure",
        "remains a group-level failure",
        "preserves every affected source index",
    ] {
        assert!(ml.contains(required), "j2k-ml docs must state {required:?}");
    }
    assert!(public_support.contains("Preparation failures remain indexed per input"));
    assert!(public_support.contains("command-buffer failures remain group-level"));
    assert!(contracts.contains(
        "Successful groups, indexed preparation failures, and homogeneous execution failures."
    ));
    assert!(
        submission.contains("Command-buffer failures remain group-level at the batch boundary.")
    );
}
