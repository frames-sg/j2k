// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safe target-feature boundaries for benchmark-visible JPEG SIMD helpers.

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn jpeg_benchmark_simd_helpers_keep_safe_target_feature_boundaries() {
    let root = repo_root();
    let bench_support = fs::read_to_string(root.join("crates/j2k-jpeg/src/bench_support.rs"))
        .expect("read JPEG bench support");
    let parity = fs::read_to_string(root.join("crates/j2k-jpeg/tests/idct_parity.rs"))
        .expect("read JPEG IDCT parity tests");
    let unsafe_audit =
        fs::read_to_string(root.join("docs/unsafe-audit.md")).expect("read unsafe audit");

    assert_pattern_checks(&[
        PatternCheck::new("AArch64 baseline benchmark dispatch", &bench_support).required(&[
            "pub fn bench_idct_neon_block(input: &[i16; 64], output: &mut [u8; 64])",
            "Every supported AArch64 target provides NEON",
            "unsafe { crate::idct::neon::idct_islow(input, output) }",
        ]),
        PatternCheck::new("safe AVX2 benchmark dispatch", &bench_support)
            .required(&[
                "enum BenchAvx2Dispatch",
                "const fn select_bench_avx2_dispatch(avx2_available: bool)",
                "std::is_x86_feature_detected!(\"avx2\")",
                "BenchAvx2Dispatch::Scalar => idct_islow(input, output)",
                "BenchAvx2Dispatch::Avx2 =>",
                "avx2_dispatch_selects_scalar_when_feature_is_absent",
                "avx2_dispatch_selects_avx2_when_feature_is_present",
            ])
            .forbidden(&["support — call `std::is_x86_feature_detected!(\"avx2\")` first"]),
        PatternCheck::new("AVX2 parity through the safe wrapper", &parity)
            .required(&[
                "j2k_jpeg::bench_support::bench_idct_avx2_block(input, &mut avx_out)",
                "assert_eq!(",
                "scalar_out, avx_out",
            ])
            .forbidden(&["if !std::is_x86_feature_detected!(\"avx2\")"]),
        PatternCheck::new("AVX2 benchmark unsafe-audit contract", &unsafe_audit).required(&[
            "`crates/j2k-jpeg/src/bench_support.rs`",
            "the x86 AVX2 wrapper performs runtime detection and falls back to scalar",
            "AVX2 dispatch selector unit tests, IDCT parity tests",
        ]),
    ]);
}

#[test]
fn jpeg_simd_backends_share_safe_row_normalization() {
    let root = repo_root();
    let backend = fs::read_to_string(root.join("crates/j2k-jpeg/src/backend/mod.rs"))
        .expect("read JPEG backend module");
    let row_pair = fs::read_to_string(root.join("crates/j2k-jpeg/src/backend/row_pair.rs"))
        .expect("read JPEG row-pair normalizer");
    let x86 = fs::read_to_string(root.join("crates/j2k-jpeg/src/backend/x86.rs"))
        .expect("read JPEG x86 backend");
    let neon = fs::read_to_string(root.join("crates/j2k-jpeg/src/backend/neon.rs"))
        .expect("read JPEG NEON backend");

    assert_pattern_checks(&[
        PatternCheck::new("JPEG SIMD normalization module", &backend).required(&["mod row_pair;"]),
        PatternCheck::new("JPEG SIMD safe slice normalization", &row_pair).required(&[
            "fn normalize_ycbcr_row<'a>(",
            "fn normalize_simd_row_pair(",
            "chroma.min_width()",
            "row_pair_clamps_luma_chroma_and_both_destinations_together",
            "row_pair_rejects_zero_complete_pixels",
        ]),
        PatternCheck::new("AVX2 normalized row boundary", &x86).required(&[
            "normalize_ycbcr_row(y_row, cb_row, cr_row, dst)",
            "let Some(request) = normalize_simd_row_pair(request) else",
            "fill_rgb_row_pair_from_420_avx2(request, &mut scratch)",
        ]),
        PatternCheck::new("NEON normalized row boundary", &neon).required(&[
            "normalize_ycbcr_row(y_row, cb_row, cr_row, dst)",
            "let Some(request) = normalize_simd_row_pair(request) else",
            "fill_rgb_row_pair_from_420_neon(request)",
        ]),
    ]);
    assert!(
        row_pair.lines().count() < 130,
        "JPEG SIMD row normalization must remain focused"
    );
}
