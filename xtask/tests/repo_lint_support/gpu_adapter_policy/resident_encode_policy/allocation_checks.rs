// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use crate::repo_lint_support::{assert_pattern_checks, PatternCheck};

mod adapter_contract;

pub(super) fn assert_cuda_image_derived_encode_allocation_contract(root: &Path) {
    let read = |relative: &str| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    };
    let allocation = read("crates/j2k-cuda/src/allocation.rs");
    let htj2k = [
        read("crates/j2k-cuda/src/encode/htj2k.rs"),
        read("crates/j2k-cuda/src/encode/htj2k/code_blocks.rs"),
        read("crates/j2k-cuda/src/encode/htj2k/host_budget.rs"),
        read("crates/j2k-cuda/src/encode/htj2k/ordering.rs"),
        read("crates/j2k-cuda/src/encode/htj2k/resident.rs"),
        read("crates/j2k-cuda/src/encode/htj2k/tile_packets.rs"),
    ]
    .concat();
    let packetization = [
        read("crates/j2k-cuda/src/encode/packetization.rs"),
        read("crates/j2k-cuda/src/encode/packetization/error.rs"),
        read("crates/j2k-cuda/src/encode/packetization/flatten.rs"),
        read("crates/j2k-cuda/src/encode/packetization/runtime.rs"),
        read("crates/j2k-cuda/src/encode/packetization/state.rs"),
        read("crates/j2k-cuda/src/encode/packetization/state/count.rs"),
        read("crates/j2k-cuda/src/encode/packetization/tag_tree.rs"),
        read("crates/j2k-cuda/src/encode/packetization/tag_tree/allocation.rs"),
        read("crates/j2k-cuda/src/encode/packetization/tag_tree/allocation/tests.rs"),
        read("crates/j2k-cuda/src/encode/packetization/tests.rs"),
        read("crates/j2k-cuda/src/encode/packetization/tests/ht_segment.rs"),
        read("crates/j2k-cuda/src/encode/packetization/types.rs"),
    ]
    .concat();
    let stage = read("crates/j2k-cuda/src/encode/stage.rs");
    let dwt_output = read("crates/j2k-cuda/src/encode/stage/dwt_output.rs");
    let runtime_encode = [
        read("crates/j2k-cuda-runtime/src/htj2k_encode/types.rs"),
        read("crates/j2k-cuda-runtime/src/htj2k_encode/planning.rs"),
        read("crates/j2k-cuda-runtime/src/htj2k_encode/api.rs"),
        read("crates/j2k-cuda-runtime/src/htj2k_encode/completion.rs"),
        read("crates/j2k-cuda-runtime/src/context/compact.rs"),
    ]
    .concat();
    let runtime_packetize = read("crates/j2k-cuda-runtime/src/htj2k_packetize.rs");
    let runtime_readback = read("crates/j2k-cuda-runtime/src/j2k_encode/readback.rs");

    adapter_contract::assert_policy(&allocation, &htj2k, &packetization, &stage, &dwt_output);
    assert_runtime_ownership_contracts(&runtime_encode, &runtime_packetize, &runtime_readback);

    assert_eq!(
        htj2k.matches("Vec::with_capacity").count(),
        1,
        "only the codec-bounded resolution vector may remain infallible"
    );
    assert_eq!(
        packetization.matches("Vec::with_capacity").count(),
        0,
        "packetization owner construction must remain fallible"
    );
    assert_eq!(
        packetization.matches("vec![0; total_nodes]").count(),
        0,
        "packetization tag-tree state must remain fallibly allocated"
    );
}

fn assert_runtime_ownership_contracts(
    runtime_encode: &str,
    runtime_packetize: &str,
    runtime_readback: &str,
) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA runtime consuming result accessors", runtime_encode).required(&[
            "pub fn into_code_blocks(self)",
            "pub fn into_parts(self)",
            "htj2k_encode_kernel_jobs_with_live_host_bytes(",
            "htj2k_encode_multi_input_kernel_jobs_with_live_host_bytes(",
            "HostPhaseBudget::with_live_bytes(",
            "copy_pooled_bytes_to_vec_uninit_with_budget(",
            "into_owned_code_blocks_with_live_host_bytes(",
            "host_budget.account_vec(&payload)?",
        ]),
        PatternCheck::new("CUDA runtime packetized data ownership", runtime_packetize).required(&[
            "pub fn into_data(self) -> Vec<u8>",
            "packetize_htj2k_cleanup_packets_with_tag_state_and_live_host_bytes(",
            "completion_budget.account_vec(&kernel_packets)?",
        ]),
        PatternCheck::new(
            "CUDA runtime quantized coefficient ownership",
            runtime_readback,
        )
        .required(&["pub fn into_coefficients(self) -> Vec<i32>"]),
    ]);
}
