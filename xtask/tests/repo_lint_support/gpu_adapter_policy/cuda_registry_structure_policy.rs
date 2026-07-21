// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

fn assert_below(relative: &str, source: &str, maximum: usize) {
    assert!(
        source.lines().count() < maximum,
        "{relative} must stay below its focused line-count ratchet of {maximum}"
    );
}

#[test]
fn cuda_kernel_registry_is_split_by_codec_family() {
    let root_path = "crates/j2k-cuda-runtime/src/kernels.rs";
    let j2k_path = "crates/j2k-cuda-runtime/src/kernels/j2k.rs";
    let jpeg_path = "crates/j2k-cuda-runtime/src/kernels/jpeg.rs";
    let transcode_path = "crates/j2k-cuda-runtime/src/kernels/transcode.rs";
    let shared_path = "crates/j2k-cuda-runtime/src/kernels/shared.rs";
    let root = read(root_path);
    let j2k = read(j2k_path);
    let jpeg = read(jpeg_path);
    let transcode = read(transcode_path);
    let shared = read(shared_path);

    assert_below(root_path, &root, 325);
    assert_below(j2k_path, &j2k, 650);
    assert_below(jpeg_path, &jpeg, 250);
    assert_below(transcode_path, &transcode, 250);
    assert_below(shared_path, &shared, 225);
    assert_pattern_checks(&[
        PatternCheck::new("CUDA kernel registry facade", &root)
            .required(&[
                "mod j2k;",
                "mod jpeg;",
                "mod shared;",
                "mod transcode;",
                "pub(crate) enum CudaKernel",
                "pub(crate) fn entrypoint",
            ])
            .forbidden(&[
                "include!(",
                "fn is_cuda_oxide_transcode_stage",
                "fn is_cuda_oxide_jpeg_decode_stage",
            ]),
        PatternCheck::new("J2K kernel family", &j2k).required(&[
            "impl CudaKernel",
            "is_cuda_oxide_j2k_encode_stage",
            "j2k_classic_codeblock_launch_geometry",
            "cuda_oxide_htj2k_decode_ptx",
        ]),
        PatternCheck::new("JPEG kernel family", &jpeg).required(&[
            "impl CudaKernel",
            "is_cuda_oxide_jpeg_encode_stage",
            "is_cuda_oxide_jpeg_decode_stage",
            "cuda_oxide_jpeg_decode_ptx",
        ]),
        PatternCheck::new("transcode kernel family", &transcode).required(&[
            "impl CudaKernel",
            "is_transcode_dwt97_batch_stage",
            "is_cuda_oxide_transcode_stage",
            "cuda_oxide_transcode_ptx",
        ]),
    ]);
}

#[test]
fn classic_cuda_decode_has_abi_preparation_and_launch_owners() {
    let root_path = "crates/j2k-cuda-runtime/src/classic_decode.rs";
    let abi_path = "crates/j2k-cuda-runtime/src/classic_decode/abi.rs";
    let prepare_path = "crates/j2k-cuda-runtime/src/classic_decode/prepare.rs";
    let launch_path = "crates/j2k-cuda-runtime/src/classic_decode/launch.rs";
    let root = read(root_path);
    let abi = read(abi_path);
    let prepare = read(prepare_path);
    let launch = read(launch_path);

    assert_below(root_path, &root, 50);
    assert_below(abi_path, &abi, 225);
    assert_below(prepare_path, &prepare, 350);
    assert_below(launch_path, &launch, 350);
    assert_pattern_checks(&[
        PatternCheck::new("classic decode facade", &root)
            .required(&["mod abi;", "mod launch;", "mod prepare;", "pub use abi::{"])
            .forbidden(&["include!(", "impl CudaContext", "fn validate_classic_job"]),
        PatternCheck::new("classic decode ABI", &abi).required(&[
            "pub struct CudaClassicCodeBlockJob",
            "pub struct CudaClassicDecodeTarget",
            "pub(crate) struct CudaClassicKernelJob",
        ]),
        PatternCheck::new("classic decode preparation", &prepare).required(&[
            "fn prepare_classic_decode",
            "fn validate_classic_job",
            "validate_disjoint_output_regions",
        ]),
        PatternCheck::new("classic decode launch", &launch).required(&[
            "impl CudaContext",
            "decode_classic_codeblocks_multi_with_resources_and_pool_timed",
            "pool_reuse_guard.release()",
        ]),
    ]);
}

#[test]
fn cuda_direct_plan_is_split_by_shared_classic_and_ht_ownership() {
    let root_path = "crates/j2k-cuda/src/direct_plan.rs";
    let shared_path = "crates/j2k-cuda/src/direct_plan/shared.rs";
    let classic_path = "crates/j2k-cuda/src/direct_plan/classic.rs";
    let ht_path = "crates/j2k-cuda/src/direct_plan/ht.rs";
    let root = read(root_path);
    let shared = read(shared_path);
    let classic = read(classic_path);
    let ht = read(ht_path);

    assert_below(root_path, &root, 500);
    assert_below(shared_path, &shared, 325);
    assert_below(classic_path, &classic, 300);
    assert_below(ht_path, &ht, 500);
    assert_pattern_checks(&[
        PatternCheck::new("CUDA direct-plan facade/types", &root)
            .required(&[
                "mod classic;",
                "mod ht;",
                "mod required_regions;",
                "mod shared;",
            ])
            .forbidden(&[
                "include!(",
                "fn validate_classic_job",
                "fn convert_store_step",
            ]),
        PatternCheck::new("CUDA direct-plan shared validation", &shared).required(&[
            "struct CudaPlanCapacityHint",
            "fn convert_idwt_step",
            "fn convert_store_step",
        ]),
        PatternCheck::new("CUDA direct-plan classic planning", &classic).required(&[
            "fn append_classic_subband",
            "fn validate_classic_job",
            "fn classic_style_flags",
        ]),
        PatternCheck::new("CUDA direct-plan HT planning", &ht).required(&[
            "fn append_ht_subband",
            "PLAN_BLOCK_LENGTH_MISMATCH",
            "ROI_MAXSHIFT_UNSUPPORTED",
        ]),
    ]);
}
