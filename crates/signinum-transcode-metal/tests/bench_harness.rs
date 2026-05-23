// SPDX-License-Identifier: Apache-2.0

const DCT97_BENCH: &str = include_str!("../benches/dct97.rs");

#[test]
fn dct97_benchmark_groups_are_stable() {
    for expected in [
        "dct97_metal_projection",
        "scalar_16x16",
        "metal_explicit_16x16",
    ] {
        assert!(
            DCT97_BENCH.contains(expected),
            "missing benchmark marker {expected}"
        );
    }
}
