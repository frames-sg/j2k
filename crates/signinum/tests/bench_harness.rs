// SPDX-License-Identifier: Apache-2.0

#[test]
fn facade_bench_exposes_cpu_and_hybrid_encode_surfaces() {
    let bench = include_str!("../benches/facade.rs");

    for expected in [
        "facade_j2k_lossless_encode_cpu_matrix",
        "cpu_only_rgb8_512_classic_external",
        "cpu_only_rgb8_512_htj2k_external",
        "facade_j2k_lossless_encode_adaptive_matrix",
        "adaptive_rgb8_512_classic_external",
        "adaptive_rgb8_512_htj2k_external",
        "direct_metal_auto_stage_rgb8_512_classic_external",
        "direct_metal_cpu_rct_stage_rgb8_512_htj2k_external",
        "facade_j2k_htj2k_encode_backend_speed_matrix",
        "EncodeBackendPreference::STRICT_DEVICE",
        "SIGNINUM_REQUIRE_CUDA_BENCH",
    ] {
        assert!(
            bench.contains(expected),
            "facade benchmark is missing `{expected}`"
        );
    }

    for expected_row in [
        "cpu_rgb8_512_htj2k_external",
        "adaptive_rgb8_512_htj2k_perf_gate_external",
        "strict_metal_rgb8_512_htj2k_external",
        "strict_cuda_rgb8_512_htj2k_external",
        "cpu_rgb8_1024_htj2k_external",
        "adaptive_rgb8_1024_htj2k_perf_gate_external",
        "strict_metal_rgb8_1024_htj2k_external",
        "strict_cuda_rgb8_1024_htj2k_external",
        "cpu_rgba8_512_htj2k_external",
        "adaptive_rgba8_512_htj2k_perf_gate_external",
        "strict_metal_rgba8_512_htj2k_external",
        "strict_cuda_rgba8_512_htj2k_external",
        "cpu_rgba8_1024_htj2k_external",
        "adaptive_rgba8_1024_htj2k_perf_gate_external",
        "strict_metal_rgba8_1024_htj2k_external",
        "strict_cuda_rgba8_1024_htj2k_external",
    ] {
        assert!(
            !bench.contains(expected_row),
            "facade benchmark should generate `{expected_row}`, not satisfy the harness with a static string"
        );
    }

    for expected in [
        r#"label: "rgb8_512""#,
        r#"label: "rgb8_1024""#,
        r#"label: "rgba8_512""#,
        r#"label: "rgba8_1024""#,
        r#"format!("cpu_{}_htj2k_external", case.label)"#,
        r#"format!("adaptive_{}_htj2k_perf_gate_external", case.label)"#,
        r#"format!("strict_metal_{}_htj2k_external", case.label)"#,
        r#"format!("strict_cuda_{}_htj2k_external", case.label)"#,
    ] {
        assert!(
            bench.contains(expected),
            "facade benchmark row generation is missing `{expected}`"
        );
    }
}
