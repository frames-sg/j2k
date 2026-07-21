// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_below, read};
use crate::repo_lint_support::{assert_pattern_checks, PatternCheck};

#[test]
fn j2k_ml_benchmark_support_has_focused_ownership() {
    let root = read("crates/j2k-ml/benches/support/mod.rs");
    let decode_case = read("crates/j2k-ml/benches/support/decode_case.rs");
    let fixture = read("crates/j2k-ml/benches/support/fixture.rs");
    let input_selection = read("crates/j2k-ml/benches/support/input_selection.rs");
    let process_policy = read("crates/j2k-ml/benches/support/process_policy.rs");
    let workload = read("crates/j2k-ml/benches/support/workload.rs");

    for (path, source, limit) in [
        ("crates/j2k-ml/benches/support/mod.rs", root.as_str(), 40),
        (
            "crates/j2k-ml/benches/support/decode_case.rs",
            decode_case.as_str(),
            90,
        ),
        (
            "crates/j2k-ml/benches/support/fixture.rs",
            fixture.as_str(),
            190,
        ),
        (
            "crates/j2k-ml/benches/support/input_selection.rs",
            input_selection.as_str(),
            80,
        ),
        (
            "crates/j2k-ml/benches/support/process_policy.rs",
            process_policy.as_str(),
            120,
        ),
        (
            "crates/j2k-ml/benches/support/workload.rs",
            workload.as_str(),
            160,
        ),
    ] {
        assert_below(path, source, limit);
        assert!(
            !source.contains("use super::*") && !source.contains("include!("),
            "{path} must keep explicit Rust module boundaries"
        );
    }

    assert_pattern_checks(&[
        PatternCheck::new("benchmark support module wiring", &root)
            .required(&[
                "mod decode_case;",
                "mod fixture;",
                "mod input_selection;",
                "mod process_policy;",
                "mod workload;",
            ])
            .forbidden(&[
                "enum InputMode",
                "enum ProcessMode",
                "struct Workload",
                "fn encode_ht_fixture(",
                "fn requests(",
            ]),
        PatternCheck::new("benchmark decode-case ownership", &decode_case)
            .required(&[
                "fn requests(",
                "fn decoded_pixels_per_batch(",
                "fn require_prepared_success(",
            ])
            .forbidden(&["std::env::var", "encode_j2k_lossless"]),
        PatternCheck::new("benchmark fixture ownership", &fixture)
            .required(&["fn encode_ht_fixture(", "fn wrap_benchmark_rgba_jph("])
            .forbidden(&["struct Workload", "std::env::var"]),
        PatternCheck::new("benchmark input-selection ownership", &input_selection)
            .required(&["enum InputMode", "J2K_ML_BATCH_INPUT_MODE"])
            .forbidden(&["enum ProcessMode", "encode_j2k_lossless"]),
        PatternCheck::new("benchmark process-policy ownership", &process_policy)
            .required(&[
                "enum ProcessMode",
                "J2K_ML_BATCH_PROCESS_MODE",
                "fn ensure_metal_criterion_instrumentation_disabled(",
            ])
            .forbidden(&["enum InputMode", "encode_j2k_lossless"]),
        PatternCheck::new("benchmark workload ownership", &workload)
            .required(&[
                "struct WorkloadSpec",
                "struct Workload",
                "fn materialize_workload(",
            ])
            .forbidden(&["std::env::var", "wrap_j2k_codestream"]),
    ]);
}

#[test]
fn accelerator_benchmark_sessions_are_scoped_to_one_materialized_workload() {
    let cuda = read("crates/j2k-ml/benches/batch_decode_cuda.rs");
    let metal = read("crates/j2k-ml/benches/batch_decode_metal.rs");

    for (source, constructor, expected) in [
        (cuda.as_str(), "CudaBatchDecoder::with_options", 4),
        (cuda.as_str(), "CudaBurnDecoder::new", 4),
        (cuda.as_str(), "CpuBurnDecoder::<Cuda>::new", 1),
        (
            metal.as_str(),
            "MetalBatchDecoder::system_default_with_options",
            4,
        ),
        (metal.as_str(), "MetalBurnDecoder::system_default", 4),
        (metal.as_str(), "CpuBurnDecoder::<Wgpu>::new", 2),
    ] {
        let constructor_offsets = source
            .match_indices(constructor)
            .map(|(offset, _)| offset)
            .collect::<Vec<_>>();
        assert_eq!(
            constructor_offsets.len(),
            expected,
            "unexpected {constructor} benchmark session count"
        );
        for offset in constructor_offsets {
            let prefix = &source[..offset];
            let function_start = prefix.rfind("\nfn ").unwrap_or(0);
            let workload_start = prefix
                .rfind("let workload = materialize_workload(spec, input_mode);")
                .unwrap_or_else(|| panic!("{constructor} must follow workload materialization"));
            let loop_start = prefix
                .rfind("for &spec in workload_specs {")
                .unwrap_or_else(|| panic!("{constructor} must be inside a workload loop"));
            assert!(
                loop_start > function_start && workload_start > loop_start,
                "{constructor} must be created inside and after its workload loop so prepared caches cannot retain earlier inputs"
            );
        }
    }
}
