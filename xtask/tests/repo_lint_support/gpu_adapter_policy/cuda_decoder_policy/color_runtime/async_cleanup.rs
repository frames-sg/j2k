// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn normal_cuda_refinement_retains_metadata_without_a_host_completion_boundary() {
    let cleanup_enqueue = std::fs::read_to_string(
        repo_root().join("crates/j2k-cuda/src/decoder/resident/cleanup_dequant/enqueue.rs"),
    )
    .expect("read CUDA normal-path cleanup/dequant enqueue leaf");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA normal-path queued dequantization", &cleanup_enqueue)
            .required(&[
                "decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool_and_live_host_bytes(",
                "j2k_dequantize_queued_htj2k_cleanup_enqueue(&queued)",
            ])
            .forbidden(&[
                "j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_and_live_host_bytes(",
                "context.synchronize(",
            ]),
    ]);
}
