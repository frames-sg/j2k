#include <metal_stdlib>
using namespace metal;

constant uint J2K_ENCODE_STATUS_OK = 0u;
constant uint J2K_ENCODE_STATUS_FAIL = 1u;
constant uint J2K_ENCODE_STATUS_UNSUPPORTED = 2u;
// Failure details that mean an output buffer was too small for what the
// stage wrote, which the host retries with conservative capacity. The host
// mirrors them in `engine::abi`, and a test there holds the two in step.
constant uint J2K_CLASSIC_TIER1_FAIL_PASS_CAPACITY = 4u;
constant uint J2K_CLASSIC_TIER1_FAIL_SEGMENT_CAPACITY = 5u;
constant uint J2K_CODESTREAM_FAIL_TILE_CAPACITY = 2u;
constant uint J2K_CODESTREAM_FAIL_EOC_CAPACITY = 3u;
constant uint J2K_PACKET_FAIL_HEADER_CAPACITY = 3u;
constant uint J2K_PACKET_FAIL_STUFFING_CAPACITY = 4u;
constant uint J2K_PACKET_FAIL_BODY_CAPACITY = 5u;
// A packet whose code-block failed Tier-1; the Tier-1 detail rides in the
// status's data length.
constant uint J2K_PACKET_FAIL_TIER1 = 7u;
constant uint J2K_PACKET_PAYLOAD_COPY_SMALL_JOB_BYTES = 64u;
constant uint J2K_PACKET_PAYLOAD_COPY_MEDIUM_JOB_BYTES = 512u;

struct J2kClassicEncodeBatchJob {
    uint coefficient_offset;
    uint output_offset;
    uint segment_offset;
    uint width;
    uint height;
    uint sub_band_type;
    uint total_bitplanes;
    uint style_flags;
    uint output_capacity;
    uint segment_capacity;
};
