// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

struct ResidentContractSources {
    types_root: String,
    types: String,
    native_shell: String,
    native_resident: String,
    facade: String,
    cuda_production: String,
    cuda_resident: String,
    cuda_stage: String,
    cuda_dwt_output: String,
}

impl ResidentContractSources {
    fn read(root: &Path) -> Self {
        let read = |relative: &str| {
            fs::read_to_string(root.join(relative))
                .unwrap_or_else(|error| panic!("read {relative}: {error}"))
        };
        let cuda = read("crates/j2k-cuda/src/encode.rs");
        Self {
            types_root: read("crates/j2k-types/src/lib.rs"),
            types: read("crates/j2k-types/src/resident.rs"),
            native_shell: read("crates/j2k-native/src/j2c/encode/single_tile.rs"),
            native_resident: read("crates/j2k-native/src/j2c/encode/single_tile/resident.rs"),
            facade: read("crates/j2k/src/encode/resident.rs"),
            cuda_production: cuda
                .split("#[cfg(test)]\nmod tests")
                .next()
                .expect("CUDA encode production prefix")
                .to_string(),
            cuda_resident: read("crates/j2k-cuda/src/encode/resident.rs"),
            cuda_stage: read("crates/j2k-cuda/src/encode/stage.rs"),
            cuda_dwt_output: read("crates/j2k-cuda/src/encode/stage/dwt_output.rs"),
        }
    }
}

#[test]
fn cuda_resident_encode_uses_an_explicit_no_host_input_contract() {
    let sources = ResidentContractSources::read(repo_root());
    let shared_contract = format!("{}\n{}", sources.types_root, sources.types);
    let device_output = sources
        .cuda_resident
        .split("pub struct CudaEncodedJ2k {")
        .nth(1)
        .and_then(|source| source.split("impl CudaEncodedJ2k").next())
        .expect("extract CUDA-resident output owner");

    assert!(sources.types_root.lines().count() < 1_200);
    assert!(sources.types.lines().count() < 225);
    assert!(sources.cuda_dwt_output.lines().count() < 200);
    assert_pattern_checks(&[
        PatternCheck::new("shared resident encode contract", &shared_contract).required(&[
            "pub struct J2kResidentEncodeInput",
            "pub struct J2kResidentHtj2kTileEncodeJob",
            "fn encode_resident_htj2k_tile(",
        ]),
        PatternCheck::new("native resident encode module", &sources.native_shell)
            .required(&["mod resident;", "encode_resident_impl"]),
        PatternCheck::new(
            "native fail-closed resident orchestration",
            &sources.native_resident,
        )
        .required(&[
            "validate_non_pixel_single_tile_request(",
            "NonPixelSingleTileRequest {",
            ".map_err(NativeEncodePipelineError::into_resident_error)?",
            "encode_complete_resident_ht_tile(",
            "finalize_accelerated_codestream(",
        ])
        .forbidden(&[
            "validate_encode_request(",
            "pixels_len",
            "J2kLosslessSamples",
            "prepare_accelerated_components",
            "encode_tile_packets",
            "encode_multitile_impl",
        ]),
        PatternCheck::new("resident facade dispatch contract", &sources.facade).required(&[
            "pub fn encode_j2k_lossless_resident_with_accelerator(",
            "J2kEncodeValidation::External",
            "EncodeBackendPreference::RequireDevice",
            "required_resident_encode_stages(",
        ]),
        PatternCheck::new("CUDA resident input routing", &sources.cuda_production)
            .required(&[
                "J2kResidentEncodeInput::new(",
                "encode_j2k_lossless_resident_with_accelerator(",
                "fn encode_resident_htj2k_tile(",
            ])
            .forbidden(&[
                "dummy_len",
                "let dummy =",
                "J2kLosslessSamples::new(",
                "encode_j2k_lossless_with_accelerator(",
                "encoded: host_outcome.encoded.clone()",
            ]),
        PatternCheck::new("CUDA resident codestream contracts", &sources.cuda_resident).required(
            &[
                "pub struct CudaEncodedJ2kMetadata",
                "pub struct CudaEncodedJ2k",
                "pub struct CudaLosslessBufferEncodeOutcome",
                "pub encoded: CudaEncodedJ2k",
            ],
        ),
        PatternCheck::new(
            "CUDA resident codestream metadata-only owner",
            device_output,
        )
        .required(&[
            "pub metadata: CudaEncodedJ2kMetadata",
            "pub codestream: CudaResidentCodestreamBuffer",
        ])
        .forbidden(&["j2k::EncodedJ2k", "host_outcome", "host_codestream"]),
        PatternCheck::new("CUDA DWT conversion stage ownership", &sources.cuda_stage)
            .required(&["mod dwt_output;", "cuda_dwt53_output_to_j2k"]),
        PatternCheck::new(
            "CUDA DWT conversion implementation",
            &sources.cuda_dwt_output,
        )
        .required(&[
            "fn cuda_dwt_output_parts",
            "try_vec_with_capacity",
            "fn extract_cuda_subband",
        ]),
    ]);
}
