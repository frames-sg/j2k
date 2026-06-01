// SPDX-License-Identifier: Apache-2.0

#[test]
fn cuda_htj2k_decode_bench_exposes_gray_rgb_rgba_rows() {
    let bench = include_str!("../benches/htj2k_decode.rs");

    for expected in [
        "cpu_gray8",
        "cuda_gray8",
        "cpu_rgb8",
        "cuda_rgb8",
        "cpu_rgba8",
        "cuda_rgba8",
        "j2k_cuda_htj2k_full_tile_decode",
        "j2k_cuda_htj2k_roi_decode",
        "j2k_cuda_htj2k_scaled_decode",
        "j2k_cuda_htj2k_roi_scaled_decode",
        "j2k_cuda_htj2k_tile_batch_decode",
        "SIGNINUM_REQUIRE_CUDA_BENCH",
    ] {
        assert!(
            bench.contains(expected),
            "CUDA HTJ2K decode benchmark is missing `{expected}`"
        );
    }
}
