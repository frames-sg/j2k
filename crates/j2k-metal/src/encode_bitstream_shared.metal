#include <metal_stdlib>
using namespace metal;

constant uint J2K_ENCODE_STATUS_OK = 0u;
constant uint J2K_ENCODE_STATUS_FAIL = 1u;
constant uint J2K_ENCODE_STATUS_UNSUPPORTED = 2u;
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
