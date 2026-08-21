// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safety- and budget-critical phase ordering checked against the Rust AST.

use std::fs;

use super::{repo_root, rust_function_policy::FunctionCalls};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn native_decode_preflights_and_propagates_before_roi_or_allocation() {
    let decode = read("crates/j2k-native/src/j2c/decode.rs");
    let decode_tile = FunctionCalls::parse("native tile decode", &decode, "decode_tile");
    decode_tile.assert_ordered(
        "native full-tile preflight before ROI planning",
        &[
            "build::build",
            "RoiPlan::build",
            "segment::parse",
            "decode_component_tile_bit_planes_budgeted",
        ],
    );
    decode_tile.assert_propagated(
        "native tile workspace failures",
        &[
            "build::build",
            "segment::parse",
            "decode_component_tile_bit_planes_budgeted",
        ],
    );

    let allocation = read("crates/j2k-native/src/j2c/build/allocation.rs");
    FunctionCalls::parse(
        "native decomposition allocation",
        &allocation,
        "prepare_decomposition_storage",
    )
    .assert_ordered(
        "plan, stale-capacity normalization, and actual-capacity validation",
        &[
            "plan::build_allocation_plan",
            "reuse::discard_stale_capacity",
            "account_live_workspace",
            "reuse::reserve_decomposition_storage",
            "account_live_workspace",
        ],
    );
}

#[test]
fn native_context_and_tile_owner_handoffs_remain_transactional() {
    let decode = read("crates/j2k-native/src/j2c/decode.rs");
    FunctionCalls::parse("native decode", &decode, "decode").assert_ordered(
        "retained component accounting before parse and reset",
        &["prepare_reused_decode_baseline", "tile::parse", "reset"],
    );

    let tile = read("crates/j2k-native/src/j2c/tile.rs");
    FunctionCalls::parse("native tile parser", &tile, "parse").assert_ordered(
        "native tile construction and owner handoff",
        &[
            "TileMetadataBudget::for_image",
            "try_reserve_retained",
            "inherit_tile_metadata",
            "parse_tile_part",
            "validate_owner_graph",
            "ParsedTiles::new",
        ],
    );
}

#[test]
fn cuda_status_and_packet_owners_exist_before_launch_or_completion() {
    let decode = read("crates/j2k-cuda-j2k-engine/src/htj2k_decode/completion.rs");
    FunctionCalls::parse(
        "CUDA cleanup completion",
        &decode,
        "run_htj2k_cleanup_multi_kernel",
    )
    .assert_ordered(
        "CUDA cleanup status allocation before launch",
        &["try_vec_filled", "launch_htj2k_decode_codeblocks_multi"],
    );

    let encode = read("crates/j2k-cuda-j2k-engine/src/htj2k_encode/completion.rs");
    for function in [
        "encode_htj2k_kernel_jobs_device_with_resources_and_pool",
        "encode_htj2k_multi_input_kernel_jobs_device_compact_with_resources_and_pool",
    ] {
        FunctionCalls::parse("CUDA HTJ2K encode completion", &encode, function).assert_ordered(
            "CUDA encode status allocation before launch",
            &[
                "HostPhaseBudget::with_live_bytes",
                "try_vec_filled",
                "time_default_stream_named_us",
            ],
        );
    }

    let packetize = read("crates/j2k-cuda-j2k-engine/src/htj2k_packetize.rs");
    FunctionCalls::parse(
        "CUDA packetization completion",
        &packetize,
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
}

#[test]
fn cuda_grayscale_batch_preserves_prepare_execute_complete_order() {
    let execution = read("crates/j2k-cuda/src/decoder/grayscale_batch/execution.rs");
    FunctionCalls::parse(
        "CUDA grayscale batch orchestration",
        &execution,
        "decode_grayscale_cuda_batch_with_profile",
    )
    .assert_ordered(
        "CUDA grayscale batch phases",
        &[
            "prepare_grayscale_batch",
            "upload_grayscale_decode_resources",
            "build_grayscale_component_work",
            "enqueue_grayscale_entropy",
            "enqueue_grayscale_idwt",
            "finish_grayscale_components_and_store",
        ],
    );
}
