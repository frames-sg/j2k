// SPDX-License-Identifier: MIT OR Apache-2.0

mod allocation_sources;
mod checks;
mod encoder_checks;
mod entropy_checks;
mod frame_checks;
mod gpu_capacity;
mod gpu_checks;

struct JpegAllocationSources {
    checkpoint: String,
    checkpoint_allocation: String,
    checkpoint_build: String,
    checkpoint_cache: String,
    checkpoint_cache_allocation: String,
    checkpoint_eoi: String,
    checkpoint_planning: String,
    checkpoint_tests: String,
    checkpoint_allocation_tests: String,
    checkpoint_build_tests: String,
    checkpoint_eoi_tests: String,
    fast_packet: String,
    fast_packet_types: String,
    fast_packet_allocation: String,
    fast_packet_build_owner: String,
    fast_packet_build_gray: String,
    fast_packet_build_materialization: String,
    fast_packet_build: String,
    fast_packet_checkpoints: String,
    fast_packet_entropy: String,
    fast_packet_header: String,
    fast_packet_tests: String,
    fast_packet_allocation_tests: String,
    fast_packet_behavior_tests: String,
    fast_packet_checkpoint_tests: String,
    fast_packet_source_tests: String,
    device_plan: String,
    decoder_sequential: String,
    owned_decode: String,
    owned_decode_plan: String,
    owned_decode_tests: String,
}

struct JpegEncodeAllocationSources {
    encode_allocation: String,
    encode_allocation_tests: String,
    encoded_output: String,
    encoded_output_tests: String,
    encoder: String,
    encoder_contract: String,
    encoder_planning: String,
    encoder_tests: String,
    baseline_entropy: String,
    shared_allocation: String,
    entropy: String,
    entropy_restart: String,
    entropy_workspace: String,
    frame: String,
    orchestrate: String,
    orchestrate_batch_owner: String,
    orchestrate_batch_group: String,
    orchestrate_batch: String,
    planning_owner: String,
    planning_batch: String,
    planning: String,
    transcode: String,
    types: String,
    adapter_tests: String,
}

#[test]
fn jpeg_high_level_allocation_paths_stay_focused_fallible_and_single_pass() {
    checks::assert_policy(&JpegAllocationSources::read());
}

#[test]
fn jpeg_encode_output_paths_stay_preflighted_bounded_and_fallible() {
    let sources = JpegEncodeAllocationSources::read();
    encoder_checks::assert_policy(&sources);
    frame_checks::assert_policy(&sources);
    gpu_checks::assert_policy(&sources);
}
