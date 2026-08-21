// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ownership ratchet for backend-neutral encode geometry policy.

use std::fs;

use super::{assert_file_pattern_checks, repo_root, rust_sources, FilePatternCheck};

#[test]
fn shared_encode_geometry_has_one_policy_owner_and_backend_consumers() {
    let root = repo_root();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-types/src/encode_geometry.rs").required(&[
                "pub const fn maximum_decomposition_levels(",
                "pub const fn lossless_decomposition_levels(",
                "pub const fn encode_dwt_level_dimensions_for_input(",
                "pub fn code_block_exponent(",
                "pub const fn code_block_dimensions(",
                "pub const fn reversible_subband_total_bitplanes(",
            ]),
            FilePatternCheck::new("crates/j2k/src/encode/geometry.rs")
                .required(&["shared_lossless_decomposition_levels("])
                .forbidden(&[
                    "MIN_LOSSLESS_DWT_DIMENSION",
                    "fn j2k_rpcl_lossless_decomposition_levels",
                ]),
            FilePatternCheck::new("crates/j2k-metal/src/encode/plan.rs")
                .required(&[
                    "lossless_decomposition_levels(",
                    "encode_dwt_level_dimensions(",
                    "code_block_exponent(edge)",
                    "reversible_subband_total_bitplanes(",
                ])
                .forbidden(&[
                    "MIN_LOSSLESS_DWT_DIMENSION",
                    "fn lossless_device_encode_levels",
                    "struct LosslessDwtLevelPlan",
                ]),
            FilePatternCheck::new("crates/j2k-native/src/j2c/encode.rs")
                .required(&["use j2k_types::encode_geometry::maximum_decomposition_levels;"]),
            FilePatternCheck::new("crates/j2k-native/src/j2c/fdwt/packed.rs")
                .required(&["encode_dwt_level_dimensions_for_input("]),
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/src/j2k_encode/dwt/validation.rs")
                .required(&["use j2k_types::encode_geometry::maximum_decomposition_levels;"]),
            FilePatternCheck::new("crates/j2k-transcode-metal/src/metal/geometry.rs")
                .required(&["encode_dwt_level_dimensions_for_input(width, height)"]),
            FilePatternCheck::new("crates/j2k-codec-math/src/dwt.rs")
                .required(&[
                    "pub const fn max_decomposition_levels(width: u32, height: u32) -> u8 {\n    j2k_types::encode_geometry::maximum_decomposition_levels(width, height)\n}",
                ])
                .forbidden(&[
                    "MIN_LOSSLESS_DWT_DIMENSION",
                    "while minimum_dimension > 1",
                ]),
        ],
    );
}

#[test]
fn gpu_adapters_do_not_reintroduce_lossless_level_policy() {
    let forbidden = [
        "MIN_LOSSLESS_DWT_DIMENSION",
        "fn lossless_device_encode_levels",
        "fn max_decomposition_levels",
    ];
    for directory in [
        "crates/j2k-cuda/src",
        "crates/j2k-cuda-runtime/src",
        "crates/j2k-metal/src",
        "crates/j2k-transcode-cuda/src",
        "crates/j2k-transcode-metal/src",
    ] {
        for path in rust_sources(&root_path(directory)) {
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            for pattern in forbidden {
                assert!(
                    !source.contains(pattern),
                    "{} must consume j2k-types encode geometry instead of owning `{pattern}`",
                    path.display()
                );
            }
        }
    }
}

fn root_path(relative: &str) -> std::path::PathBuf {
    repo_root().join(relative)
}
