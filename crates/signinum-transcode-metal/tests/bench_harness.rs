// SPDX-License-Identifier: Apache-2.0

const DCT97_BENCH: &str = include_str!("../benches/dct97.rs");

#[test]
fn dct97_benchmark_groups_are_stable() {
    for expected in [
        "dct53_metal_projection",
        "jpeg_to_htj2k_wsi_53",
        "reversible_dct53_metal_projection",
        "reversible_dct53_batch_metal_projection",
        "jpeg_to_htj2k_wsi_integer_53",
        "metal_auto",
        "rayon_224x224",
        "rayon_512x512",
        "rayon_1024x1024",
        "rayon_2048x2048",
        "batch_1",
        "batch_8",
        "batch_32",
        "batch_128",
        "batch_512",
        "rayon_224x224_batch_1",
        "metal_explicit_224x224_batch_1",
        "rayon_512x512_batch_512",
        "rayon_1024x1024_batch_128",
        "rayon_2048x2048_batch_32",
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
