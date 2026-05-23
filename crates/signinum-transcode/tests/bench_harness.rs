// SPDX-License-Identifier: Apache-2.0

const DCT53_BENCH: &str = include_str!("../benches/dct53.rs");

#[test]
fn dct53_benchmark_group_names_are_stable() {
    for group_name in [
        "dct53_1d_single_block_scalar",
        "dct53_1d_multi_block_scalar",
        "dct53_2d_single_level_scalar",
        "dct53_2d_grid_scalar",
        "dct97_2d_grid_scalar",
        "direct_linear_13x11_scratch_reuse",
        "dct53_multilevel_scalar",
        "dct53_layout_candidates",
        "jpeg_dct_extract",
        "jpeg_to_htj2k",
        "grayscale_8x8_stateful_reuse",
        "grayscale_8x8_float_direct_97",
        "ycbcr_420_16x16_float_direct_97",
    ] {
        assert!(
            DCT53_BENCH.contains(group_name),
            "missing Criterion group {group_name}"
        );
    }
}
