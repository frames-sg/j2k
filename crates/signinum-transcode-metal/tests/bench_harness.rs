// SPDX-License-Identifier: Apache-2.0

const DCT97_BENCH: &str = include_str!("../benches/dct97.rs");

#[test]
fn dct97_benchmark_groups_are_stable() {
    for expected in [
        "dct53_metal_projection",
        "jpeg_to_htj2k_wsi_53",
        "dct97_metal_projection",
        "scalar_224x224",
        "metal_explicit_224x224",
        "scalar_512x512",
        "metal_explicit_512x512",
        "scalar_1024x1024",
        "metal_explicit_1024x1024",
        "scalar_2048x2048",
        "metal_explicit_2048x2048",
        "jpeg_to_htj2k_wsi_97",
        "srgb_ybr420_224",
        "srgb_ybr420_512",
        "srgb_ybr420_1024",
        "srgb_ybr420_2048",
        "p3_like_ybr444_224",
        "p3_like_ybr444_512",
        "p3_like_ybr444_1024",
        "p3_like_ybr444_2048",
        "ycbcr_like_ybr420_224",
        "ycbcr_like_ybr420_512",
        "ycbcr_like_ybr420_1024",
        "ycbcr_like_ybr420_2048",
    ] {
        assert!(
            DCT97_BENCH.contains(expected),
            "missing benchmark marker {expected}"
        );
    }
}
