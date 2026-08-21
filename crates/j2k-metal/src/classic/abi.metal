#include <metal_stdlib>
using namespace metal;

struct J2kClassicCleanupBatchJob {
    uint coded_offset;
    uint coded_len;
    uint segment_offset;
    uint segment_count;
    uint width;
    uint height;
    uint output_stride;
    uint output_offset;
    uint missing_msbs;
    uint total_bitplanes;
    uint roi_shift;
    uint number_of_coding_passes;
    uint sub_band_type;
    uint style_flags;
    uint strict;
    uint irreversible_midpoint;
    float dequantization_step;
};

struct J2kClassicSegment {
    uint data_offset;
    uint data_length;
    uint start_coding_pass;
    uint end_coding_pass;
    uint use_arithmetic;
};

struct J2kClassicStatus {
    uint code;
    uint detail;
    uint reserved0;
    uint reserved1;
};

struct J2kClassicRepeatedBatchParams {
    uint job_count;
    uint output_plane_len;
    uint batch_count;
};

struct J2kQeData {
    uint qe;
    uchar nmps;
    uchar nlps;
    uchar switch_mps;
};

struct J2kArithmeticDecoder {
    device const uchar *data;
    uint data_len;
    uint c;
    uint a;
    uint base_pointer;
    uint shift_count;
};

struct J2kBypassDecoder {
    device const uchar *data;
    uint data_len;
    uint bit_pos;
    uint strict;
};
