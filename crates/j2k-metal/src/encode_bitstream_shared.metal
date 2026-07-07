#include <metal_stdlib>
using namespace metal;

constant uint J2K_ENCODE_STATUS_OK = 0u;
constant uint J2K_ENCODE_STATUS_FAIL = 1u;
constant uint J2K_ENCODE_STATUS_UNSUPPORTED = 2u;
constant uint J2K_PACKET_PAYLOAD_COPY_SMALL_JOB_BYTES = 64u;
constant uint J2K_PACKET_PAYLOAD_COPY_MEDIUM_JOB_BYTES = 512u;
