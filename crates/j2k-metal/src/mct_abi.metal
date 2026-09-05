// SPDX-License-Identifier: MIT OR Apache-2.0

#include <metal_stdlib>
using namespace metal;

struct J2kInverseMctParams {
    uint len;
    uint transform;
    float addend0;
    float addend1;
    float addend2;
};

struct J2kMctStatus {
    uint code;
    uint detail;
    uint reserved0;
    uint reserved1;
};

struct J2kForwardRctParams {
    uint len;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct J2kForwardIctParams {
    uint len;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct J2kFusedInputMctParams {
    uint len;
    uint bytes_per_sample;
    uint bit_depth;
    uint sample_offset;
    uint signed_samples;
    uint reversible;
};

constant uint J2K_MCT_TRANSFORM_REVERSIBLE53 = 0;
constant uint J2K_MCT_TRANSFORM_IRREVERSIBLE97 = 1;
constant uint J2K_MCT_STATUS_OK = 0;
