// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::repo_lint_support::rust_function_policy::FunctionCalls;

pub(super) fn assert_policy(decode: &str, encode: &str, packetize: &str) {
    assert!(
        include_str!("ordering.rs").lines().count() < 75,
        "CUDA phase-allocation ordering policy must remain focused"
    );
    FunctionCalls::parse(
        "CUDA cleanup completion",
        decode,
        "run_htj2k_cleanup_multi_kernel",
    )
    .assert_ordered(
        "CUDA cleanup status allocation before launch",
        &["try_vec_filled", "launch_htj2k_decode_codeblocks_multi"],
    );
    for function in [
        "encode_htj2k_kernel_jobs_device_with_resources_and_pool",
        "encode_htj2k_multi_input_kernel_jobs_device_compact_with_resources_and_pool",
    ] {
        FunctionCalls::parse("CUDA HTJ2K encode completion", encode, function).assert_ordered(
            "CUDA encode status allocation before launch",
            &[
                "HostPhaseBudget::with_live_bytes",
                "try_vec_filled",
                "time_default_stream_named_us",
            ],
        );
    }
    FunctionCalls::parse(
        "CUDA packetization completion",
        packetize,
        "packetize_htj2k_cleanup_packets_with_tag_state_and_live_host_bytes",
    )
    .assert_ordered(
        "CUDA packetization launch and output ownership",
        &[
            "HostPhaseBudget::with_live_bytes",
            "htj2k_packetization_kernel_packets",
            "try_vec_filled",
            "drop",
            "time_default_stream_named_us",
            "complete_htj2k_packetization",
        ],
    );
    FunctionCalls::parse(
        "CUDA packetization host completion",
        packetize,
        "complete_htj2k_packetization",
    )
    .assert_ordered(
        "CUDA packetization output ownership",
        &[
            "HostPhaseBudget::with_live_bytes",
            "account_vec",
            "try_vec_filled",
            "try_vec_filled",
        ],
    );
}
