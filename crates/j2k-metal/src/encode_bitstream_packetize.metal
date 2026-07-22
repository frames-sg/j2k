struct J2kPacketEncodeParams {
    uint resolution_count;
    uint num_layers;
    uint num_components;
    uint code_block_count;
    uint subband_count;
    uint descriptor_count;
    uint output_capacity;
    uint header_capacity;
    uint scratch_node_capacity;
};

struct J2kPacketDescriptor {
    uint packet_index;
    uint state_index;
    uint layer;
    uint resolution;
    uint component;
    uint precinct_lo;
    uint precinct_hi;
    uint state_block_offset;
};

struct J2kPacketResolution {
    uint subband_offset;
    uint subband_count;
};

struct J2kPacketSubband {
    uint block_offset;
    uint block_count;
    uint num_cbs_x;
    uint num_cbs_y;
};

struct J2kPacketBlock {
    uint data_offset;
    uint data_len;
    uint num_coding_passes;
    uint num_zero_bitplanes;
    uint previously_included;
    uint l_block;
    uint block_coding_mode;
    uint reserved0;
};

struct J2kResidentPacketBlock {
    uint tier1_job_index;
    uint previously_included;
    uint l_block;
    uint block_coding_mode;
};

struct J2kResidentPacketBlockParams {
    uint block_count;
    uint tier1_job_count;
};

struct J2kPacketStateBlock {
    uint previously_included;
    uint l_block;
};

struct J2kPacketEncodeStatus {
    uint code;
    uint detail;
    uint data_len;
    uint reserved0;
    uint payload_copy_bytes;
    uint payload_copy_small_jobs;
    uint payload_copy_medium_jobs;
    uint payload_copy_large_jobs;
};

struct J2kLosslessCodestreamAssemblyParams {
    uint width;
    uint height;
    uint num_components;
    uint bit_depth;
    uint signed_samples;
    uint num_decomposition_levels;
    uint use_mct;
    uint guard_bits;
    uint progression_order;
    uint write_tlm;
    uint high_throughput;
    uint code_block_style;
    uint code_block_width_exp;
    uint code_block_height_exp;
    uint output_capacity;
};

struct J2kCodestreamAssemblyStatus {
    uint code;
    uint detail;
    uint data_len;
    uint reserved0;
};

struct J2kPacketBitWriter {
    device uchar *data;
    uint capacity;
    uint len;
    uint buffer;
    uint bits_in_buffer;
    uint last_byte_was_ff;
    uint failed;
};

inline void j2k_set_packet_status(device J2kPacketEncodeStatus *status, uint code, uint detail, uint len) {
    status->code = code;
    status->detail = detail;
    status->data_len = len;
    status->reserved0 = 0u;
    status->payload_copy_bytes = 0u;
    status->payload_copy_small_jobs = 0u;
    status->payload_copy_medium_jobs = 0u;
    status->payload_copy_large_jobs = 0u;
}

inline void j2k_set_packet_status_payload_copy(
    device J2kPacketEncodeStatus *status,
    uint code,
    uint detail,
    uint len,
    uint payload_copy_bytes,
    uint payload_copy_small_jobs,
    uint payload_copy_medium_jobs,
    uint payload_copy_large_jobs
) {
    status->code = code;
    status->detail = detail;
    status->data_len = len;
    status->reserved0 = 0u;
    status->payload_copy_bytes = payload_copy_bytes;
    status->payload_copy_small_jobs = payload_copy_small_jobs;
    status->payload_copy_medium_jobs = payload_copy_medium_jobs;
    status->payload_copy_large_jobs = payload_copy_large_jobs;
}

inline void j2k_set_codestream_status(
    device J2kCodestreamAssemblyStatus *status,
    uint code,
    uint detail,
    uint len
) {
    status->code = code;
    status->detail = detail;
    status->data_len = len;
    status->reserved0 = 0u;
}

inline bool j2k_codestream_write_u8(
    device uchar *out,
    uint capacity,
    thread uint &cursor,
    uint value
) {
    if (cursor >= capacity) {
        return false;
    }
    out[cursor] = uchar(value & 0xFFu);
    cursor += 1u;
    return true;
}

inline bool j2k_codestream_write_u16(
    device uchar *out,
    uint capacity,
    thread uint &cursor,
    uint value
) {
    return j2k_codestream_write_u8(out, capacity, cursor, value >> 8u) &&
       j2k_codestream_write_u8(out, capacity, cursor, value);
}

inline bool j2k_codestream_write_u32(
    device uchar *out,
    uint capacity,
    thread uint &cursor,
    uint value
) {
    return j2k_codestream_write_u8(out, capacity, cursor, value >> 24u) &&
       j2k_codestream_write_u8(out, capacity, cursor, value >> 16u) &&
       j2k_codestream_write_u8(out, capacity, cursor, value >> 8u) &&
       j2k_codestream_write_u8(out, capacity, cursor, value);
}

inline bool j2k_codestream_write_marker(
    device uchar *out,
    uint capacity,
    thread uint &cursor,
    uint marker
) {
    return j2k_codestream_write_u8(out, capacity, cursor, 0xFFu) &&
       j2k_codestream_write_u8(out, capacity, cursor, marker);
}

kernel void j2k_assemble_lossless_classic_codestream(
    device const uchar *tile_data [[buffer(0)]],
    device const J2kPacketEncodeStatus *tile_status [[buffer(1)]],
    device uchar *out [[buffer(2)]],
    constant J2kLosslessCodestreamAssemblyParams &params [[buffer(3)]],
    device J2kCodestreamAssemblyStatus *status [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

   j2k_set_codestream_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u);
    const J2kPacketEncodeStatus packet_status = tile_status[0];
    if (packet_status.code != J2K_ENCODE_STATUS_OK) {
       j2k_set_codestream_status(status, J2K_ENCODE_STATUS_FAIL, packet_status.detail, 0u);
        return;
    }
    if (params.num_components == 0u || params.num_components > 255u ||
        params.bit_depth == 0u || params.bit_depth > 16u ||
        params.num_decomposition_levels > 31u) {
       j2k_set_codestream_status(status, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u);
        return;
    }

    const uint tile_len = packet_status.data_len;
    const uint tile_part_len = 14u + tile_len;
    uint cursor = 0u;
    bool ok = true;

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x4Fu);

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x51u);
    const uint siz_len = 38u + 3u * params.num_components;
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, siz_len);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, params.width);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, params.height);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, params.width);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, params.height);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, params.num_components);
    const uint ssiz = (params.bit_depth - 1u) | (params.signed_samples != 0u ? 0x80u : 0u);
    for (uint comp = 0u; comp < params.num_components && ok; ++comp) {
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, ssiz);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 1u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 1u);
    }

    if (params.high_throughput != 0u) {
        const uint magnitude_bits = params.bit_depth - 1u;
        const uint bp = magnitude_bits <= 8u ? 0u :
            (magnitude_bits < 28u ? magnitude_bits - 8u : 13u + (magnitude_bits >> 2u));
        ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x50u);
        ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 8u);
        ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0x00020000u);
        ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, bp);
    }

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x52u);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 12u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.progression_order);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 1u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.use_mct != 0u ? 1u : 0u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.num_decomposition_levels);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.code_block_width_exp);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.code_block_height_exp);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.code_block_style);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 1u);

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x5Cu);
    const uint qcd_steps = 1u + 3u * params.num_decomposition_levels;
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 3u + qcd_steps);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.guard_bits << 5u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.bit_depth << 3u);
    for (uint level = 0u; level < params.num_decomposition_levels && ok; ++level) {
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, (params.bit_depth + 1u) << 3u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, (params.bit_depth + 1u) << 3u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, (params.bit_depth + 2u) << 3u);
    }

    if (params.write_tlm != 0u) {
        ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x55u);
        ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 10u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 0u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 0x60u);
        ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 0u);
        ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, tile_part_len);
    }

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x90u);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 10u);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, tile_part_len);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 1u);
    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x93u);

    if (!ok || cursor + tile_len + 2u > params.output_capacity) {
       j2k_set_codestream_status(status, J2K_ENCODE_STATUS_FAIL, 2u, cursor);
        return;
    }
    for (uint idx = 0u; idx < tile_len; ++idx) {
        out[cursor + idx] = tile_data[idx];
    }
    cursor += tile_len;
    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0xD9u);
    if (!ok) {
       j2k_set_codestream_status(status, J2K_ENCODE_STATUS_FAIL, 3u, cursor);
        return;
    }

   j2k_set_codestream_status(status, J2K_ENCODE_STATUS_OK, 0u, cursor);
}

struct J2kBatchedCodestreamAssemblyJob {
    uint tile_data_offset;
    uint codestream_offset;
    uint width;
    uint height;
    uint num_components;
    uint bit_depth;
    uint signed_samples;
    uint num_decomposition_levels;
    uint use_mct;
    uint guard_bits;
    uint progression_order;
    uint write_tlm;
    uint high_throughput;
    uint code_block_style;
    uint code_block_width_exp;
    uint code_block_height_exp;
    uint output_capacity;
};

kernel void j2k_assemble_lossless_codestream_batched(
    device const uchar *tile_data [[buffer(0)]],
    device const J2kPacketEncodeStatus *tile_status [[buffer(1)]],
    device uchar *out [[buffer(2)]],
    device const J2kBatchedCodestreamAssemblyJob *jobs [[buffer(3)]],
    device J2kCodestreamAssemblyStatus *status [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const J2kBatchedCodestreamAssemblyJob job = jobs[gid];
    device J2kCodestreamAssemblyStatus *tile_status_out = status + gid;
    device uchar *tile_out = out + job.codestream_offset;
    const J2kPacketEncodeStatus packet_status = tile_status[gid];

   j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_FAIL, 0u, 0u);
    if (packet_status.code != J2K_ENCODE_STATUS_OK) {
       j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_FAIL, packet_status.detail, 0u);
        return;
    }
    if (job.num_components == 0u || job.num_components > 255u ||
        job.bit_depth == 0u || job.bit_depth > 16u ||
        job.num_decomposition_levels > 31u) {
       j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u);
        return;
    }

    const uint tile_len = packet_status.data_len;
    const uint tile_part_len = 14u + tile_len;
    uint cursor = 0u;
    bool ok = true;

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x4Fu);

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x51u);
    const uint siz_len = 38u + 3u * job.num_components;
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, siz_len);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, job.width);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, job.height);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, job.width);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, job.height);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, job.num_components);
    const uint ssiz = (job.bit_depth - 1u) | (job.signed_samples != 0u ? 0x80u : 0u);
    for (uint comp = 0u; comp < job.num_components && ok; ++comp) {
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, ssiz);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 1u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 1u);
    }

    if (job.high_throughput != 0u) {
        const uint magnitude_bits = job.bit_depth - 1u;
        const uint bp = magnitude_bits <= 8u ? 0u :
            (magnitude_bits < 28u ? magnitude_bits - 8u : 13u + (magnitude_bits >> 2u));
        ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x50u);
        ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 8u);
        ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0x00020000u);
        ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, bp);
    }

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x52u);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 12u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.progression_order);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 1u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.use_mct != 0u ? 1u : 0u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.num_decomposition_levels);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.code_block_width_exp);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.code_block_height_exp);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.code_block_style);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 1u);

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x5Cu);
    const uint qcd_steps = 1u + 3u * job.num_decomposition_levels;
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 3u + qcd_steps);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.guard_bits << 5u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.bit_depth << 3u);
    for (uint level = 0u; level < job.num_decomposition_levels && ok; ++level) {
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, (job.bit_depth + 1u) << 3u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, (job.bit_depth + 1u) << 3u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, (job.bit_depth + 2u) << 3u);
    }

    if (job.write_tlm != 0u) {
        ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x55u);
        ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 10u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 0u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 0x60u);
        ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 0u);
        ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, tile_part_len);
    }

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x90u);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 10u);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, tile_part_len);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 1u);
    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x93u);
    const uint payload_offset = cursor;

    if (!ok || cursor + tile_len + 2u > job.output_capacity) {
       j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_FAIL, 2u, cursor);
        return;
    }
    cursor += tile_len;
    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0xD9u);
    if (!ok) {
       j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_FAIL, 3u, cursor);
        return;
    }

   j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_OK, payload_offset, cursor);
}

inline J2kPacketBlock j2k_classic_packet_block_from_resident(
    device const J2kResidentPacketBlock *resident_blocks,
    device const J2kClassicEncodeBatchJob *tier1_jobs,
    device const J2kClassicEncodeStatus *tier1_statuses,
    uint tier1_job_count,
    uint block_index
) {
    const J2kResidentPacketBlock resident = resident_blocks[block_index];
    J2kPacketBlock packet;
    packet.data_offset = 0u;
    packet.data_len = 0u;
    packet.num_coding_passes = 1u;
    packet.num_zero_bitplanes = 0u;
    packet.previously_included = resident.previously_included;
    packet.l_block = resident.l_block;
    packet.block_coding_mode = 0xFFFFFFFFu;
    packet.reserved0 = 0u;

    if (resident.tier1_job_index < tier1_job_count) {
        const J2kClassicEncodeBatchJob job = tier1_jobs[resident.tier1_job_index];
        const J2kClassicEncodeStatus tier1_status = tier1_statuses[resident.tier1_job_index];
        if (tier1_status.code == J2K_ENCODE_STATUS_OK &&
            tier1_status.data_len <= job.output_capacity &&
            tier1_status.reserved0 <= 1u &&
            tier1_status.reserved1 == 0u &&
            tier1_status.data_len + tier1_status.reserved0 >= tier1_status.data_len &&
            tier1_status.data_len + tier1_status.reserved0 <= job.output_capacity &&
            tier1_status.segment_count <= job.segment_capacity) {
            packet.data_offset = job.output_offset + tier1_status.reserved0;
            packet.data_len = tier1_status.data_len;
            packet.num_coding_passes = tier1_status.number_of_coding_passes;
            packet.num_zero_bitplanes = tier1_status.missing_bit_planes;
            packet.block_coding_mode = resident.block_coding_mode;
        } else {
            packet.reserved0 = tier1_status.detail;
        }
    }

    return packet;
}

inline J2kPacketBlock j2k_ht_packet_block_from_resident(
    device const J2kResidentPacketBlock *resident_blocks,
    device const J2kHtEncodeBatchJob *tier1_jobs,
    device const J2kHtEncodeStatus *tier1_statuses,
    uint tier1_job_count,
    uint block_index
) {
    const J2kResidentPacketBlock resident = resident_blocks[block_index];
    J2kPacketBlock packet;
    packet.data_offset = 0u;
    packet.data_len = 0u;
    packet.num_coding_passes = 1u;
    packet.num_zero_bitplanes = 0u;
    packet.previously_included = resident.previously_included;
    packet.l_block = resident.l_block;
    packet.block_coding_mode = 0xFFFFFFFFu;
    packet.reserved0 = 0u;

    if (resident.tier1_job_index < tier1_job_count) {
        const J2kHtEncodeBatchJob job = tier1_jobs[resident.tier1_job_index];
        const J2kHtEncodeStatus tier1_status = tier1_statuses[resident.tier1_job_index];
        if (tier1_status.code == J2K_ENCODE_STATUS_OK &&
            tier1_status.data_len <= job.output_capacity) {
            packet.data_offset = job.output_offset;
            packet.data_len = tier1_status.data_len;
            packet.num_coding_passes = tier1_status.num_coding_passes;
            packet.num_zero_bitplanes = tier1_status.num_zero_bitplanes;
            packet.block_coding_mode = resident.block_coding_mode;
        } else {
            packet.reserved0 = tier1_status.detail;
        }
    }

    return packet;
}

kernel void j2k_prepare_packet_blocks_from_classic_status(
    device const J2kResidentPacketBlock *resident_blocks [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *tier1_jobs [[buffer(1)]],
    device const J2kClassicEncodeStatus *tier1_statuses [[buffer(2)]],
    device J2kPacketBlock *packet_blocks [[buffer(3)]],
    constant J2kResidentPacketBlockParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.block_count) {
        return;
    }

    packet_blocks[gid] = j2k_classic_packet_block_from_resident(
        resident_blocks,
        tier1_jobs,
        tier1_statuses,
        params.tier1_job_count,
        gid
    );
}

kernel void j2k_prepare_packet_blocks_from_ht_status(
    device const J2kResidentPacketBlock *resident_blocks [[buffer(0)]],
    device const J2kHtEncodeBatchJob *tier1_jobs [[buffer(1)]],
    device const J2kHtEncodeStatus *tier1_statuses [[buffer(2)]],
    device J2kPacketBlock *packet_blocks [[buffer(3)]],
    constant J2kResidentPacketBlockParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.block_count) {
        return;
    }

    packet_blocks[gid] = j2k_ht_packet_block_from_resident(
        resident_blocks,
        tier1_jobs,
        tier1_statuses,
        params.tier1_job_count,
        gid
    );
}

inline void j2k_packet_writer_init(thread J2kPacketBitWriter &writer, device uchar *data, uint capacity) {
    writer.data = data;
    writer.capacity = capacity;
    writer.len = 0u;
    writer.buffer = 0u;
    writer.bits_in_buffer = 0u;
    writer.last_byte_was_ff = 0u;
    writer.failed = 0u;
}

inline void j2k_packet_flush_byte(thread J2kPacketBitWriter &writer) {
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    const uchar byte = uchar(writer.buffer >> (writer.bits_in_buffer - limit));
    if (writer.len >= writer.capacity) {
        writer.failed = 1u;
        return;
    }
    writer.data[writer.len] = byte;
    writer.len += 1u;
    writer.last_byte_was_ff = byte == uchar(0xFFu) ? 1u : 0u;
    writer.bits_in_buffer -= limit;
    writer.buffer &= writer.bits_in_buffer == 0u ? 0u : ((1u << writer.bits_in_buffer) - 1u);
}

inline void j2k_packet_write_bit(thread J2kPacketBitWriter &writer, uint bit) {
    writer.buffer = (writer.buffer << 1u) | (bit & 1u);
    writer.bits_in_buffer += 1u;
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    if (writer.bits_in_buffer >= limit) {
       j2k_packet_flush_byte(writer);
    }
}

inline void j2k_packet_write_bits(thread J2kPacketBitWriter &writer, uint value, uint count) {
    for (int bit = int(count) - 1; bit >= 0; --bit) {
       j2k_packet_write_bit(writer, (value >> uint(bit)) & 1u);
    }
}

inline void j2k_packet_writer_finish(thread J2kPacketBitWriter &writer) {
    if (writer.bits_in_buffer > 0u) {
        const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
        const uint shift = limit - writer.bits_in_buffer;
        const uchar byte = uchar(writer.buffer << shift);
        if (writer.len >= writer.capacity) {
            writer.failed = 1u;
            return;
        }
        writer.data[writer.len] = byte;
        writer.len += 1u;
        writer.last_byte_was_ff = byte == uchar(0xFFu) ? 1u : 0u;
        writer.buffer = 0u;
        writer.bits_in_buffer = 0u;
    }
}

inline uint j2k_packet_ilog2(uint value) {
    return value == 0u ? 0u : 31u - clz(value);
}

inline bool j2k_packet_value_fits(uint value, uint bits) {
    return bits >= 32u || value < (1u << bits);
}

inline uint j2k_packet_bits_for_length(uint l_block, uint passes) {
    const uint log2_passes = passes <= 1u ? 0u : j2k_packet_ilog2(passes);
    return l_block + log2_passes;
}

inline uint j2k_packet_bits_for_ht_length(uint l_block, uint passes) {
    const uint placeholder_groups = (passes > 0u ? passes - 1u : 0u) / 3u;
    const uint placeholder_passes = placeholder_groups * 3u;
    return l_block + j2k_packet_ilog2(placeholder_passes + 1u);
}

inline void j2k_packet_encode_num_passes(uint passes, thread J2kPacketBitWriter &writer) {
    if (passes == 1u) {
       j2k_packet_write_bit(writer, 0u);
    } else if (passes == 2u) {
       j2k_packet_write_bits(writer, 0b10u, 2u);
    } else if (passes == 3u) {
       j2k_packet_write_bits(writer, 0b1100u, 4u);
    } else if (passes == 4u) {
       j2k_packet_write_bits(writer, 0b1101u, 4u);
    } else if (passes == 5u) {
       j2k_packet_write_bits(writer, 0b1110u, 4u);
    } else if (passes <= 36u) {
       j2k_packet_write_bits(writer, 0b1111u, 4u);
       j2k_packet_write_bits(writer, passes - 6u, 5u);
    } else {
       j2k_packet_write_bits(writer, 0x1FFu, 9u);
       j2k_packet_write_bits(writer, passes - 37u, 7u);
    }
}

inline void j2k_packet_encode_num_ht_passes(uint passes, thread J2kPacketBitWriter &writer) {
    if (passes == 1u) {
       j2k_packet_write_bit(writer, 0u);
    } else if (passes == 2u) {
       j2k_packet_write_bits(writer, 0b10u, 2u);
    } else if (passes <= 5u) {
       j2k_packet_write_bits(writer, 0b11u, 2u);
       j2k_packet_write_bits(writer, passes - 3u, 2u);
    } else if (passes <= 36u) {
       j2k_packet_write_bits(writer, 0b11u, 2u);
       j2k_packet_write_bits(writer, 0b11u, 2u);
       j2k_packet_write_bits(writer, passes - 6u, 5u);
    } else {
       j2k_packet_write_bits(writer, 0b11u, 2u);
       j2k_packet_write_bits(writer, 0b11u, 2u);
       j2k_packet_write_bits(writer, 31u, 5u);
       j2k_packet_write_bits(writer, passes - 37u, 7u);
    }
}

inline void j2k_packet_encode_length(
    uint length,
    thread uint &l_block,
    uint num_bits,
    thread J2kPacketBitWriter &writer
) {
    while (!j2k_packet_value_fits(length, num_bits)) {
       j2k_packet_write_bit(writer, 1u);
        l_block += 1u;
        num_bits += 1u;
    }
   j2k_packet_write_bit(writer, 0u);
   j2k_packet_write_bits(writer, length, num_bits);
}

inline bool j2k_packet_encode_classic_segment_lengths_resident(
    const J2kPacketBlock block,
    const J2kClassicEncodeBatchJob tier1_job,
    const J2kClassicEncodeStatus tier1_status,
    device const J2kClassicSegment *tier1_segments,
    thread uint &l_block,
    thread J2kPacketBitWriter &writer
) {
    if (tier1_status.segment_count <= 1u) {
        const uint num_bits = j2k_packet_bits_for_length(l_block, block.num_coding_passes);
       j2k_packet_encode_length(block.data_len, l_block, num_bits, writer);
        return true;
    }

    uint segment_data_sum = 0u;
    uint segment_pass_sum = 0u;
    uint required_l_block = l_block;
    for (uint segment_idx = 0u; segment_idx < tier1_status.segment_count; ++segment_idx) {
        const J2kClassicSegment segment =
            tier1_segments[tier1_job.segment_offset + segment_idx];
        if (segment.start_coding_pass >= segment.end_coding_pass ||
            segment.data_offset != segment_data_sum ||
            segment_data_sum > 0xffffffffu - segment.data_length) {
            return false;
        }
        const uint segment_passes = segment.end_coding_pass - segment.start_coding_pass;
        segment_data_sum += segment.data_length;
        segment_pass_sum += segment_passes;
        while (!j2k_packet_value_fits(
                segment.data_length,
               j2k_packet_bits_for_length(required_l_block, segment_passes))) {
            required_l_block += 1u;
        }
    }
    if (segment_data_sum != block.data_len || segment_pass_sum != block.num_coding_passes) {
        return false;
    }

    while (l_block < required_l_block) {
       j2k_packet_write_bit(writer, 1u);
        l_block += 1u;
    }
   j2k_packet_write_bit(writer, 0u);

    for (uint segment_idx = 0u; segment_idx < tier1_status.segment_count; ++segment_idx) {
        const J2kClassicSegment segment =
            tier1_segments[tier1_job.segment_offset + segment_idx];
        const uint segment_passes = segment.end_coding_pass - segment.start_coding_pass;
        const uint length_bits = j2k_packet_bits_for_length(l_block, segment_passes);
       j2k_packet_write_bits(writer, segment.data_length, length_bits);
    }
    return writer.failed == 0u;
}

inline uint j2k_packet_tree_offsets(
    uint width,
    uint height,
    thread uint *level_offsets,
    thread uint *level_widths,
    thread uint *level_heights,
    thread uint &levels
) {
    uint total = 0u;
    uint w = width;
    uint h = height;
    levels = 0u;
    while (true) {
        level_offsets[levels] = total;
        level_widths[levels] = w;
        level_heights[levels] = h;
        total += w * h;
        levels += 1u;
        if (w <= 1u && h <= 1u) {
            break;
        }
        w = (w + 1u) / 2u;
        h = (h + 1u) / 2u;
    }
    return total;
}

inline bool j2k_packet_prepare_tree(
    device const J2kPacketBlock *blocks,
    uint block_offset,
    uint block_count,
    uint num_cbs_x,
    uint num_cbs_y,
    bool zero_bitplanes,
    uint inclusion_layer,
    device uint *value,
    device uint *current,
    device uint *known,
    uint node_capacity,
    thread uint *level_offsets,
    thread uint *level_widths,
    thread uint *level_heights,
    thread uint &levels
) {
    if (num_cbs_x == 0u || num_cbs_y == 0u || num_cbs_x * num_cbs_y != block_count) {
        return false;
    }
    const uint node_count =
       j2k_packet_tree_offsets(num_cbs_x, num_cbs_y, level_offsets, level_widths, level_heights, levels);
    if (node_count > node_capacity || levels > 16u) {
        return false;
    }
    for (uint idx = 0u; idx < node_count; ++idx) {
        value[idx] = 0u;
        current[idx] = 0u;
        known[idx] = 0u;
    }
    for (uint idx = 0u; idx < block_count; ++idx) {
        const J2kPacketBlock block = blocks[block_offset + idx];
        value[idx] = zero_bitplanes
            ? block.num_zero_bitplanes
            : (block.num_coding_passes > 0u ? inclusion_layer : 0x7FFFFFFFu);
    }
    for (uint level = 1u; level < levels; ++level) {
        const uint prev_w = level_widths[level - 1u];
        const uint prev_h = level_heights[level - 1u];
        const uint cur_w = level_widths[level];
        const uint cur_h = level_heights[level];
        for (uint py = 0u; py < cur_h; ++py) {
            for (uint px = 0u; px < cur_w; ++px) {
                uint min_value = 0xFFFFFFFFu;
                for (uint dy = 0u; dy < 2u; ++dy) {
                    const uint cy = py * 2u + dy;
                    if (cy >= prev_h) {
                        continue;
                    }
                    for (uint dx = 0u; dx < 2u; ++dx) {
                        const uint cx = px * 2u + dx;
                        if (cx >= prev_w) {
                            continue;
                        }
                        const uint child = level_offsets[level - 1u] + cy * prev_w + cx;
                        min_value = min(min_value, value[child]);
                    }
                }
                value[level_offsets[level] + py * cur_w + px] = min_value;
            }
        }
    }
    return true;
}

inline bool j2k_packet_prepare_tree_resident_classic(
    device const J2kResidentPacketBlock *resident_blocks,
    device const J2kClassicEncodeBatchJob *tier1_jobs,
    device const J2kClassicEncodeStatus *tier1_statuses,
    uint tier1_job_count,
    uint block_offset,
    uint block_count,
    uint num_cbs_x,
    uint num_cbs_y,
    bool zero_bitplanes,
    uint inclusion_layer,
    device uint *value,
    device uint *current,
    device uint *known,
    uint node_capacity,
    thread uint *level_offsets,
    thread uint *level_widths,
    thread uint *level_heights,
    thread uint &levels
) {
    if (num_cbs_x == 0u || num_cbs_y == 0u || num_cbs_x * num_cbs_y != block_count) {
        return false;
    }
    const uint node_count =
       j2k_packet_tree_offsets(num_cbs_x, num_cbs_y, level_offsets, level_widths, level_heights, levels);
    if (node_count > node_capacity || levels > 16u) {
        return false;
    }
    for (uint idx = 0u; idx < node_count; ++idx) {
        value[idx] = 0u;
        current[idx] = 0u;
        known[idx] = 0u;
    }
    for (uint idx = 0u; idx < block_count; ++idx) {
        const J2kPacketBlock block = j2k_classic_packet_block_from_resident(
            resident_blocks,
            tier1_jobs,
            tier1_statuses,
            tier1_job_count,
            block_offset + idx
        );
        value[idx] = zero_bitplanes
            ? block.num_zero_bitplanes
            : (block.num_coding_passes > 0u ? inclusion_layer : 0x7FFFFFFFu);
    }
    for (uint level = 1u; level < levels; ++level) {
        const uint prev_w = level_widths[level - 1u];
        const uint prev_h = level_heights[level - 1u];
        const uint cur_w = level_widths[level];
        const uint cur_h = level_heights[level];
        for (uint py = 0u; py < cur_h; ++py) {
            for (uint px = 0u; px < cur_w; ++px) {
                uint min_value = 0xFFFFFFFFu;
                for (uint dy = 0u; dy < 2u; ++dy) {
                    const uint cy = py * 2u + dy;
                    if (cy >= prev_h) {
                        continue;
                    }
                    for (uint dx = 0u; dx < 2u; ++dx) {
                        const uint cx = px * 2u + dx;
                        if (cx >= prev_w) {
                            continue;
                        }
                        const uint child = level_offsets[level - 1u] + cy * prev_w + cx;
                        min_value = min(min_value, value[child]);
                    }
                }
                value[level_offsets[level] + py * cur_w + px] = min_value;
            }
        }
    }
    return true;
}

inline void j2k_packet_tree_encode(
    uint x,
    uint y,
    uint threshold,
    device uint *value,
    device uint *current,
    device uint *known,
    thread uint *level_offsets,
    thread uint *level_widths,
    uint levels,
    thread J2kPacketBitWriter &writer
) {
    thread uint path[16];
    uint cx = x;
    uint cy = y;
    for (uint level = 0u; level < levels; ++level) {
        path[level] = level_offsets[level] + cy * level_widths[level] + cx;
        cx /= 2u;
        cy /= 2u;
    }

    uint parent_val = 0u;
    for (int level = int(levels) - 1; level >= 0; --level) {
        const uint node = path[uint(level)];
        const uint start = max(current[node], parent_val);
        if (known[node] == 0u) {
            const uint target = min(value[node], threshold);
            for (uint v = start; v < target; ++v) {
               j2k_packet_write_bit(writer, 0u);
            }
            if (value[node] < threshold) {
               j2k_packet_write_bit(writer, 1u);
                known[node] = 1u;
            }
            current[node] = target;
        }
        parent_val = current[node];
    }
}

struct J2kPacketTreeScratch {
    device uint *inc_value;
    device uint *inc_current;
    device uint *inc_known;
    device uint *zbp_value;
    device uint *zbp_current;
    device uint *zbp_known;
};

inline J2kPacketTreeScratch j2k_packet_tree_scratch(
    device uint *tree_scratch,
    uint node_capacity
) {
    return {
        tree_scratch,
        tree_scratch + node_capacity,
        tree_scratch + node_capacity * 2u,
        tree_scratch + node_capacity * 3u,
        tree_scratch + node_capacity * 4u,
        tree_scratch + node_capacity * 5u,
    };
}

inline J2kPacketDescriptor j2k_packet_descriptor_for_order(
    device const J2kPacketDescriptor *descriptors,
    uint descriptor_count,
    uint packet_order_idx
) {
    return descriptor_count > 0u
        ? descriptors[packet_order_idx]
        : J2kPacketDescriptor{
            packet_order_idx,
            packet_order_idx,
            0u,
            packet_order_idx,
            0u,
            0u,
            0u,
            0u,
        };
}

inline bool j2k_packet_append_header(
    device uchar *out,
    device const uchar *header,
    thread J2kPacketBitWriter &writer,
    uint output_capacity,
    thread uint &out_len,
    device J2kPacketEncodeStatus *status
) {
    if (writer.failed != 0u || out_len + writer.len > output_capacity) {
       j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 3u, 0u);
        return false;
    }
    for (uint idx = 0u; idx < writer.len; ++idx) {
        out[out_len + idx] = header[idx];
    }
    out_len += writer.len;
    if (writer.len > 0u && header[writer.len - 1u] == uchar(0xFFu)) {
        if (out_len >= output_capacity) {
           j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 4u, 0u);
            return false;
        }
        out[out_len] = uchar(0u);
        out_len += 1u;
    }
    return true;
}

kernel void j2k_encode_packetization(
    device const J2kPacketResolution *resolutions [[buffer(0)]],
    device const J2kPacketSubband *subbands [[buffer(1)]],
    device const J2kPacketBlock *blocks [[buffer(2)]],
    device const uchar *payload [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    device uchar *header [[buffer(5)]],
    device uint *tree_scratch [[buffer(6)]],
    constant J2kPacketEncodeParams &params [[buffer(7)]],
    device J2kPacketEncodeStatus *status [[buffer(8)]],
    device const J2kPacketDescriptor *descriptors [[buffer(9)]],
    device J2kPacketStateBlock *state_blocks [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

   j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u);

    const uint node_capacity = params.scratch_node_capacity;
    const J2kPacketTreeScratch scratch = j2k_packet_tree_scratch(tree_scratch, node_capacity);

    uint out_len = 0u;
    const uint packet_count =
        params.descriptor_count > 0u ? params.descriptor_count : params.resolution_count;
    for (uint packet_order_idx = 0u; packet_order_idx < packet_count; ++packet_order_idx) {
        const bool has_descriptor = params.descriptor_count > 0u;
        const J2kPacketDescriptor descriptor = j2k_packet_descriptor_for_order(
            descriptors,
            params.descriptor_count,
            packet_order_idx
        );
        const uint packet_index = descriptor.packet_index;
        if (packet_index >= params.resolution_count) {
           j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 6u, 0u);
            return;
        }
        const J2kPacketResolution resolution = resolutions[packet_index];
        uint state_block_cursor = descriptor.state_block_offset;
        bool any_data = false;
        for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
            const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
            for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                if (blocks[subband.block_offset + block_idx].num_coding_passes > 0u) {
                    any_data = true;
                    break;
                }
            }
        }

        thread J2kPacketBitWriter writer;
       j2k_packet_writer_init(writer, header, params.header_capacity);
        if (!any_data) {
           j2k_packet_write_bit(writer, 0u);
           j2k_packet_writer_finish(writer);
        } else {
           j2k_packet_write_bit(writer, 1u);
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                const uint subband_state_block_offset = state_block_cursor;
                state_block_cursor += subband.block_count;
                thread uint level_offsets[16];
                thread uint level_widths[16];
                thread uint level_heights[16];
                uint levels = 0u;
                if (!j2k_packet_prepare_tree(
                        blocks,
                        subband.block_offset,
                        subband.block_count,
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                        false,
                        descriptor.layer,
                        scratch.inc_value,
                        scratch.inc_current,
                        scratch.inc_known,
                        node_capacity,
                        level_offsets,
                        level_widths,
                        level_heights,
                        levels)) {
                   j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 1u, 0u);
                    return;
                }
                thread uint z_level_offsets[16];
                thread uint z_level_widths[16];
                thread uint z_level_heights[16];
                uint z_levels = 0u;
                if (!j2k_packet_prepare_tree(
                        blocks,
                        subband.block_offset,
                        subband.block_count,
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                        true,
                        descriptor.layer,
                        scratch.zbp_value,
                        scratch.zbp_current,
                        scratch.zbp_known,
                        node_capacity,
                        z_level_offsets,
                        z_level_widths,
                        z_level_heights,
                        z_levels)) {
                   j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 2u, 0u);
                    return;
                }

                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const uint x = block_idx % subband.num_cbs_x;
                    const uint y = block_idx / subband.num_cbs_x;
                    const J2kPacketBlock block = blocks[subband.block_offset + block_idx];
                    const uint state_block_index = subband_state_block_offset + block_idx;
                    uint previously_included = block.previously_included;
                    uint local_l_block = block.l_block;
                    if (has_descriptor) {
                        previously_included = state_blocks[state_block_index].previously_included;
                        local_l_block = state_blocks[state_block_index].l_block;
                    }
                    if (previously_included == 0u) {
                       j2k_packet_tree_encode(
                            x,
                            y,
                            descriptor.layer + 1u,
                            scratch.inc_value,
                            scratch.inc_current,
                            scratch.inc_known,
                            level_offsets,
                            level_widths,
                            levels,
                            writer
                        );
                        if (block.num_coding_passes == 0u) {
                            continue;
                        }
                       j2k_packet_tree_encode(
                            x,
                            y,
                            block.num_zero_bitplanes + 1u,
                            scratch.zbp_value,
                            scratch.zbp_current,
                            scratch.zbp_known,
                            z_level_offsets,
                            z_level_widths,
                            z_levels,
                            writer
                        );
                    } else if (block.num_coding_passes > 0u) {
                       j2k_packet_write_bit(writer, 1u);
                    } else {
                       j2k_packet_write_bit(writer, 0u);
                        continue;
                    }

                    if (block.block_coding_mode == 0u) {
                        const uint num_bits =
                           j2k_packet_bits_for_length(local_l_block, block.num_coding_passes);
                       j2k_packet_encode_num_passes(block.num_coding_passes, writer);
                       j2k_packet_encode_length(block.data_len, local_l_block, num_bits, writer);
                    } else if (block.block_coding_mode == 1u) {
                        const uint num_bits =
                           j2k_packet_bits_for_ht_length(local_l_block, block.num_coding_passes);
                       j2k_packet_encode_num_ht_passes(block.num_coding_passes, writer);
                       j2k_packet_encode_length(block.data_len, local_l_block, num_bits, writer);
                    } else {
                       j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 7u, block.reserved0);
                        return;
                    }
                    if (has_descriptor) {
                        state_blocks[state_block_index].previously_included = 1u;
                        state_blocks[state_block_index].l_block = local_l_block;
                    }
                }
            }
           j2k_packet_writer_finish(writer);
        }

        if (!j2k_packet_append_header(out, header, writer, params.output_capacity, out_len, status)) {
            return;
        }

        if (any_data) {
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const J2kPacketBlock block = blocks[subband.block_offset + block_idx];
                    if (block.num_coding_passes == 0u) {
                        continue;
                    }
                    if (out_len + block.data_len > params.output_capacity) {
                       j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 5u, 0u);
                        return;
                    }
                    for (uint byte_idx = 0u; byte_idx < block.data_len; ++byte_idx) {
                        out[out_len + byte_idx] = payload[block.data_offset + byte_idx];
                    }
                    out_len += block.data_len;
                }
            }
        }
    }

   j2k_set_packet_status(status, J2K_ENCODE_STATUS_OK, 0u, out_len);
}

struct J2kPacketPayloadCopyJob {
    uint src_offset;
    uint dst_offset;
    uint byte_len;
    uint reserved0;
};

struct J2kPacketPayloadCopyParams {
    uint bytes_per_thread;
    uint stripes_per_job;
};

inline void j2k_packet_record_payload_copy_job(
    uint byte_len,
    thread uint &payload_copy_bytes,
    thread uint &payload_copy_small_jobs,
    thread uint &payload_copy_medium_jobs,
    thread uint &payload_copy_large_jobs
) {
    payload_copy_bytes = payload_copy_bytes > 0xffffffffu - byte_len
        ? 0xffffffffu
        : payload_copy_bytes + byte_len;
    if (byte_len <= J2K_PACKET_PAYLOAD_COPY_SMALL_JOB_BYTES) {
        payload_copy_small_jobs += 1u;
    } else if (byte_len <= J2K_PACKET_PAYLOAD_COPY_MEDIUM_JOB_BYTES) {
        payload_copy_medium_jobs += 1u;
    } else {
        payload_copy_large_jobs += 1u;
    }
}

inline bool j2k_packet_push_payload_copy_job(
    device J2kPacketPayloadCopyJob *payload_copy_jobs,
    uint payload_copy_capacity,
    uint src_offset,
    uint dst_offset,
    uint byte_len,
    thread uint &payload_copy_count,
    thread uint &payload_copy_bytes,
    thread uint &payload_copy_small_jobs,
    thread uint &payload_copy_medium_jobs,
    thread uint &payload_copy_large_jobs
) {
    if (byte_len == 0u) {
        return true;
    }
    if (payload_copy_count >= payload_copy_capacity) {
        return false;
    }
    payload_copy_jobs[payload_copy_count] =
        J2kPacketPayloadCopyJob{src_offset, dst_offset, byte_len, 0u};
    payload_copy_count += 1u;
   j2k_packet_record_payload_copy_job(
        byte_len,
        payload_copy_bytes,
        payload_copy_small_jobs,
        payload_copy_medium_jobs,
        payload_copy_large_jobs
    );
    return true;
}

struct J2kBatchedPacketEncodeJob {
    uint resolution_offset;
    uint subband_offset;
    uint block_offset;
    uint descriptor_offset;
    uint state_block_offset;
    uint output_offset;
    uint header_offset;
    uint scratch_offset;
    uint payload_copy_offset;
    uint payload_copy_capacity;
    uint resolution_count;
    uint num_layers;
    uint num_components;
    uint code_block_count;
    uint subband_count;
    uint descriptor_count;
    uint output_capacity;
    uint header_capacity;
    uint scratch_node_capacity;
};

kernel void j2k_encode_packetization_batched(
    device const J2kPacketResolution *all_resolutions [[buffer(0)]],
    device const J2kPacketSubband *all_subbands [[buffer(1)]],
    device const J2kPacketBlock *all_blocks [[buffer(2)]],
    device const uchar *payload [[buffer(3)]],
    device uchar *all_out [[buffer(4)]],
    device uchar *all_header [[buffer(5)]],
    device uint *all_tree_scratch [[buffer(6)]],
    device const J2kBatchedPacketEncodeJob *jobs [[buffer(7)]],
    device J2kPacketEncodeStatus *all_status [[buffer(8)]],
    device const J2kPacketDescriptor *all_descriptors [[buffer(9)]],
    device J2kPacketStateBlock *all_state_blocks [[buffer(10)]],
    device J2kPacketPayloadCopyJob *all_payload_copy_jobs [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    const J2kBatchedPacketEncodeJob job = jobs[gid];
    device const J2kPacketResolution *resolutions = all_resolutions + job.resolution_offset;
    device const J2kPacketSubband *subbands = all_subbands + job.subband_offset;
    device const J2kPacketBlock *blocks = all_blocks + job.block_offset;
    device uchar *out = all_out + job.output_offset;
    device uchar *header = all_header + job.header_offset;
    device uint *tree_scratch = all_tree_scratch + job.scratch_offset;
    device J2kPacketEncodeStatus *status = all_status + gid;
    device const J2kPacketDescriptor *descriptors = all_descriptors + job.descriptor_offset;
    device J2kPacketStateBlock *state_blocks = all_state_blocks + job.state_block_offset;
    device J2kPacketPayloadCopyJob *payload_copy_jobs =
        all_payload_copy_jobs + job.payload_copy_offset;

    J2kPacketEncodeParams params;
    params.resolution_count = job.resolution_count;
    params.num_layers = job.num_layers;
    params.num_components = job.num_components;
    params.code_block_count = job.code_block_count;
    params.subband_count = job.subband_count;
    params.descriptor_count = job.descriptor_count;
    params.output_capacity = job.output_capacity;
    params.header_capacity = job.header_capacity;
    params.scratch_node_capacity = job.scratch_node_capacity;

   j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u);

    const uint node_capacity = params.scratch_node_capacity;
    const J2kPacketTreeScratch scratch = j2k_packet_tree_scratch(tree_scratch, node_capacity);

    uint out_len = 0u;
    uint payload_copy_count = 0u;
    uint payload_copy_bytes = 0u;
    uint payload_copy_small_jobs = 0u;
    uint payload_copy_medium_jobs = 0u;
    uint payload_copy_large_jobs = 0u;
    const uint packet_count =
        params.descriptor_count > 0u ? params.descriptor_count : params.resolution_count;
    for (uint packet_order_idx = 0u; packet_order_idx < packet_count; ++packet_order_idx) {
        const bool has_descriptor = params.descriptor_count > 0u;
        const J2kPacketDescriptor descriptor = j2k_packet_descriptor_for_order(
            descriptors,
            params.descriptor_count,
            packet_order_idx
        );
        const uint packet_index = descriptor.packet_index;
        if (packet_index >= params.resolution_count) {
           j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 6u, 0u);
            return;
        }
        const J2kPacketResolution resolution = resolutions[packet_index];
        uint state_block_cursor = descriptor.state_block_offset;
        bool any_data = false;
        for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
            const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
            for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                if (blocks[subband.block_offset + block_idx].num_coding_passes > 0u) {
                    any_data = true;
                    break;
                }
            }
        }

        thread J2kPacketBitWriter writer;
       j2k_packet_writer_init(writer, header, params.header_capacity);
        if (!any_data) {
           j2k_packet_write_bit(writer, 0u);
           j2k_packet_writer_finish(writer);
        } else {
           j2k_packet_write_bit(writer, 1u);
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                const uint subband_state_block_offset = state_block_cursor;
                state_block_cursor += subband.block_count;
                thread uint level_offsets[16];
                thread uint level_widths[16];
                thread uint level_heights[16];
                uint levels = 0u;
                if (!j2k_packet_prepare_tree(
                        blocks,
                        subband.block_offset,
                        subband.block_count,
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                        false,
                        descriptor.layer,
                        scratch.inc_value,
                        scratch.inc_current,
                        scratch.inc_known,
                        node_capacity,
                        level_offsets,
                        level_widths,
                        level_heights,
                        levels)) {
                   j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 1u, 0u);
                    return;
                }
                thread uint z_level_offsets[16];
                thread uint z_level_widths[16];
                thread uint z_level_heights[16];
                uint z_levels = 0u;
                if (!j2k_packet_prepare_tree(
                        blocks,
                        subband.block_offset,
                        subband.block_count,
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                        true,
                        descriptor.layer,
                        scratch.zbp_value,
                        scratch.zbp_current,
                        scratch.zbp_known,
                        node_capacity,
                        z_level_offsets,
                        z_level_widths,
                        z_level_heights,
                        z_levels)) {
                   j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 2u, 0u);
                    return;
                }

                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const uint x = block_idx % subband.num_cbs_x;
                    const uint y = block_idx / subband.num_cbs_x;
                    const J2kPacketBlock block = blocks[subband.block_offset + block_idx];
                    const uint state_block_index = subband_state_block_offset + block_idx;
                    uint previously_included = block.previously_included;
                    uint local_l_block = block.l_block;
                    if (has_descriptor) {
                        previously_included = state_blocks[state_block_index].previously_included;
                        local_l_block = state_blocks[state_block_index].l_block;
                    }
                    if (previously_included == 0u) {
                       j2k_packet_tree_encode(
                            x,
                            y,
                            descriptor.layer + 1u,
                            scratch.inc_value,
                            scratch.inc_current,
                            scratch.inc_known,
                            level_offsets,
                            level_widths,
                            levels,
                            writer
                        );
                        if (block.num_coding_passes == 0u) {
                            continue;
                        }
                       j2k_packet_tree_encode(
                            x,
                            y,
                            block.num_zero_bitplanes + 1u,
                            scratch.zbp_value,
                            scratch.zbp_current,
                            scratch.zbp_known,
                            z_level_offsets,
                            z_level_widths,
                            z_levels,
                            writer
                        );
                    } else if (block.num_coding_passes > 0u) {
                       j2k_packet_write_bit(writer, 1u);
                    } else {
                       j2k_packet_write_bit(writer, 0u);
                        continue;
                    }

                    if (block.block_coding_mode == 0u) {
                        const uint num_bits =
                           j2k_packet_bits_for_length(local_l_block, block.num_coding_passes);
                       j2k_packet_encode_num_passes(block.num_coding_passes, writer);
                       j2k_packet_encode_length(block.data_len, local_l_block, num_bits, writer);
                    } else if (block.block_coding_mode == 1u) {
                        const uint num_bits =
                           j2k_packet_bits_for_ht_length(local_l_block, block.num_coding_passes);
                       j2k_packet_encode_num_ht_passes(block.num_coding_passes, writer);
                       j2k_packet_encode_length(block.data_len, local_l_block, num_bits, writer);
                    } else {
                       j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 7u, block.reserved0);
                        return;
                    }
                    if (has_descriptor) {
                        state_blocks[state_block_index].previously_included = 1u;
                        state_blocks[state_block_index].l_block = local_l_block;
                    }
                }
            }
           j2k_packet_writer_finish(writer);
        }

        if (!j2k_packet_append_header(out, header, writer, params.output_capacity, out_len, status)) {
            return;
        }

        if (any_data) {
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const J2kPacketBlock block = blocks[subband.block_offset + block_idx];
                    if (block.num_coding_passes == 0u) {
                        continue;
                    }
                    if (out_len + block.data_len > params.output_capacity) {
                       j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 5u, 0u);
                        return;
                    }
                    if (!j2k_packet_push_payload_copy_job(
                            payload_copy_jobs,
                            job.payload_copy_capacity,
                            block.data_offset,
                            out_len,
                            block.data_len,
                            payload_copy_count,
                            payload_copy_bytes,
                            payload_copy_small_jobs,
                            payload_copy_medium_jobs,
                            payload_copy_large_jobs)) {
                       j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 8u, payload_copy_count);
                        return;
                    }
                    out_len += block.data_len;
                }
            }
        }
    }

   j2k_set_packet_status_payload_copy(
        status,
        J2K_ENCODE_STATUS_OK,
        payload_copy_count,
        out_len,
        payload_copy_bytes,
        payload_copy_small_jobs,
        payload_copy_medium_jobs,
        payload_copy_large_jobs
    );
}

kernel void j2k_encode_packetization_resident_classic_batched(
    device const J2kPacketResolution *all_resolutions [[buffer(0)]],
    device const J2kPacketSubband *all_subbands [[buffer(1)]],
    device const J2kResidentPacketBlock *all_resident_blocks [[buffer(2)]],
    device const uchar *payload [[buffer(3)]],
    device uchar *all_out [[buffer(4)]],
    device uchar *all_header [[buffer(5)]],
    device uint *all_tree_scratch [[buffer(6)]],
    device const J2kBatchedPacketEncodeJob *jobs [[buffer(7)]],
    device J2kPacketEncodeStatus *all_status [[buffer(8)]],
    device const J2kPacketDescriptor *all_descriptors [[buffer(9)]],
    device J2kPacketStateBlock *all_state_blocks [[buffer(10)]],
    device J2kPacketPayloadCopyJob *all_payload_copy_jobs [[buffer(11)]],
    device const J2kClassicEncodeBatchJob *tier1_jobs [[buffer(12)]],
    device const J2kClassicEncodeStatus *tier1_statuses [[buffer(13)]],
    device const J2kClassicSegment *tier1_segments [[buffer(14)]],
    constant J2kResidentPacketBlockParams &resident_params [[buffer(15)]],
    uint gid [[thread_position_in_grid]]
) {
    const J2kBatchedPacketEncodeJob job = jobs[gid];
    device const J2kPacketResolution *resolutions = all_resolutions + job.resolution_offset;
    device const J2kPacketSubband *subbands = all_subbands + job.subband_offset;
    device const J2kResidentPacketBlock *resident_blocks = all_resident_blocks + job.block_offset;
    device uchar *out = all_out + job.output_offset;
    device uchar *header = all_header + job.header_offset;
    device uint *tree_scratch = all_tree_scratch + job.scratch_offset;
    device J2kPacketEncodeStatus *status = all_status + gid;
    device const J2kPacketDescriptor *descriptors = all_descriptors + job.descriptor_offset;
    device J2kPacketStateBlock *state_blocks = all_state_blocks + job.state_block_offset;
    device J2kPacketPayloadCopyJob *payload_copy_jobs =
        all_payload_copy_jobs + job.payload_copy_offset;

    J2kPacketEncodeParams params;
    params.resolution_count = job.resolution_count;
    params.num_layers = job.num_layers;
    params.num_components = job.num_components;
    params.code_block_count = job.code_block_count;
    params.subband_count = job.subband_count;
    params.descriptor_count = job.descriptor_count;
    params.output_capacity = job.output_capacity;
    params.header_capacity = job.header_capacity;
    params.scratch_node_capacity = job.scratch_node_capacity;

   j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u);

    const uint node_capacity = params.scratch_node_capacity;
    const J2kPacketTreeScratch scratch = j2k_packet_tree_scratch(tree_scratch, node_capacity);

    uint out_len = 0u;
    uint payload_copy_count = 0u;
    uint payload_copy_bytes = 0u;
    uint payload_copy_small_jobs = 0u;
    uint payload_copy_medium_jobs = 0u;
    uint payload_copy_large_jobs = 0u;
    const uint packet_count =
        params.descriptor_count > 0u ? params.descriptor_count : params.resolution_count;
    for (uint packet_order_idx = 0u; packet_order_idx < packet_count; ++packet_order_idx) {
        const bool has_descriptor = params.descriptor_count > 0u;
        const J2kPacketDescriptor descriptor = j2k_packet_descriptor_for_order(
            descriptors,
            params.descriptor_count,
            packet_order_idx
        );
        const uint packet_index = descriptor.packet_index;
        if (packet_index >= params.resolution_count) {
           j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 6u, 0u);
            return;
        }
        const J2kPacketResolution resolution = resolutions[packet_index];
        uint state_block_cursor = descriptor.state_block_offset;
        bool any_data = false;
        for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
            const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
            for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                const J2kPacketBlock block = j2k_classic_packet_block_from_resident(
                    resident_blocks,
                    tier1_jobs,
                    tier1_statuses,
                    resident_params.tier1_job_count,
                    subband.block_offset + block_idx
                );
                if (block.num_coding_passes > 0u) {
                    any_data = true;
                    break;
                }
            }
        }

        thread J2kPacketBitWriter writer;
       j2k_packet_writer_init(writer, header, params.header_capacity);
        if (!any_data) {
           j2k_packet_write_bit(writer, 0u);
           j2k_packet_writer_finish(writer);
        } else {
           j2k_packet_write_bit(writer, 1u);
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                const uint subband_state_block_offset = state_block_cursor;
                state_block_cursor += subband.block_count;
	                thread uint level_offsets[16];
	                thread uint level_widths[16];
	                thread uint level_heights[16];
	                uint levels = 0u;
	                if (!j2k_packet_prepare_tree_resident_classic(
	                        resident_blocks,
	                        tier1_jobs,
	                        tier1_statuses,
	                        resident_params.tier1_job_count,
	                        subband.block_offset,
	                        subband.block_count,
	                        subband.num_cbs_x,
	                        subband.num_cbs_y,
	                        false,
	                        descriptor.layer,
	                        scratch.inc_value,
	                        scratch.inc_current,
	                        scratch.inc_known,
	                        node_capacity,
	                        level_offsets,
	                        level_widths,
	                        level_heights,
	                        levels)) {
	                   j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 1u, 0u);
	                    return;
	                }
	                thread uint z_level_offsets[16];
	                thread uint z_level_widths[16];
	                thread uint z_level_heights[16];
	                uint z_levels = 0u;
	                if (!j2k_packet_prepare_tree_resident_classic(
	                        resident_blocks,
	                        tier1_jobs,
	                        tier1_statuses,
	                        resident_params.tier1_job_count,
	                        subband.block_offset,
	                        subband.block_count,
	                        subband.num_cbs_x,
	                        subband.num_cbs_y,
	                        true,
	                        descriptor.layer,
	                        scratch.zbp_value,
	                        scratch.zbp_current,
	                        scratch.zbp_known,
	                        node_capacity,
	                        z_level_offsets,
	                        z_level_widths,
	                        z_level_heights,
	                        z_levels)) {
	                   j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 2u, 0u);
	                    return;
	                }

                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const uint x = block_idx % subband.num_cbs_x;
                    const uint y = block_idx / subband.num_cbs_x;
                    const J2kPacketBlock block = j2k_classic_packet_block_from_resident(
                        resident_blocks,
                        tier1_jobs,
                        tier1_statuses,
                        resident_params.tier1_job_count,
                        subband.block_offset + block_idx
                    );
                    const uint state_block_index = subband_state_block_offset + block_idx;
                    uint previously_included = block.previously_included;
                    uint local_l_block = block.l_block;
                    if (has_descriptor) {
                        previously_included = state_blocks[state_block_index].previously_included;
                        local_l_block = state_blocks[state_block_index].l_block;
                    }
                    if (previously_included == 0u) {
                       j2k_packet_tree_encode(
                            x,
                            y,
                            descriptor.layer + 1u,
                            scratch.inc_value,
                            scratch.inc_current,
                            scratch.inc_known,
                            level_offsets,
                            level_widths,
                            levels,
                            writer
                        );
                        if (block.num_coding_passes == 0u) {
                            continue;
                        }
                       j2k_packet_tree_encode(
                            x,
                            y,
	                            block.num_zero_bitplanes + 1u,
	                            scratch.zbp_value,
	                            scratch.zbp_current,
	                            scratch.zbp_known,
	                            z_level_offsets,
	                            z_level_widths,
	                            z_levels,
	                            writer
	                        );
                    } else if (block.num_coding_passes > 0u) {
                       j2k_packet_write_bit(writer, 1u);
                    } else {
                       j2k_packet_write_bit(writer, 0u);
                        continue;
                    }

                    if (block.block_coding_mode == 0u) {
                       j2k_packet_encode_num_passes(block.num_coding_passes, writer);
                        const J2kResidentPacketBlock resident =
                            resident_blocks[subband.block_offset + block_idx];
                        if (resident.tier1_job_index >= resident_params.tier1_job_count) {
                           j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 7u, block.reserved0);
                            return;
                        }
                        const J2kClassicEncodeBatchJob tier1_job =
                            tier1_jobs[resident.tier1_job_index];
                        const J2kClassicEncodeStatus tier1_status =
                            tier1_statuses[resident.tier1_job_index];
                        if (!j2k_packet_encode_classic_segment_lengths_resident(
                                block,
                                tier1_job,
                                tier1_status,
                                tier1_segments,
                                local_l_block,
                                writer)) {
                           j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 9u, block.reserved0);
                            return;
                        }
                    } else {
                       j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 7u, block.reserved0);
                        return;
                    }
                    if (has_descriptor) {
                        state_blocks[state_block_index].previously_included = 1u;
                        state_blocks[state_block_index].l_block = local_l_block;
                    }
                }
            }
           j2k_packet_writer_finish(writer);
        }

        if (!j2k_packet_append_header(out, header, writer, params.output_capacity, out_len, status)) {
            return;
        }

        if (any_data) {
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const J2kPacketBlock block = j2k_classic_packet_block_from_resident(
                        resident_blocks,
                        tier1_jobs,
                        tier1_statuses,
                        resident_params.tier1_job_count,
                        subband.block_offset + block_idx
                    );
                    if (block.num_coding_passes == 0u) {
                        continue;
                    }
                    if (out_len + block.data_len > params.output_capacity) {
                       j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 5u, 0u);
                        return;
                    }
                    if (!j2k_packet_push_payload_copy_job(
                            payload_copy_jobs,
                            job.payload_copy_capacity,
                            block.data_offset,
                            out_len,
                            block.data_len,
                            payload_copy_count,
                            payload_copy_bytes,
                            payload_copy_small_jobs,
                            payload_copy_medium_jobs,
                            payload_copy_large_jobs)) {
                       j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 8u, payload_copy_count);
                        return;
                    }
                    out_len += block.data_len;
                }
            }
        }
    }

   j2k_set_packet_status_payload_copy(
        status,
        J2K_ENCODE_STATUS_OK,
        payload_copy_count,
        out_len,
        payload_copy_bytes,
        payload_copy_small_jobs,
        payload_copy_medium_jobs,
        payload_copy_large_jobs
    );
}

kernel void j2k_copy_packet_payload_batched(
    device const uchar *payload [[buffer(0)]],
    device uchar *all_out [[buffer(1)]],
    device const J2kBatchedPacketEncodeJob *jobs [[buffer(2)]],
    device const J2kPacketEncodeStatus *all_status [[buffer(3)]],
    device const J2kPacketPayloadCopyJob *all_payload_copy_jobs [[buffer(4)]],
    constant J2kPacketPayloadCopyParams &params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint copy_index = gid.x;
    const uint tile = gid.y;
    const uint stripe = gid.z;
    if (stripe >= params.stripes_per_job || params.bytes_per_thread == 0u) {
        return;
    }

    const J2kPacketEncodeStatus status = all_status[tile];
    if (status.code != J2K_ENCODE_STATUS_OK || copy_index >= status.detail) {
        return;
    }

    const J2kBatchedPacketEncodeJob job = jobs[tile];
    const J2kPacketPayloadCopyJob copy_job =
        all_payload_copy_jobs[job.payload_copy_offset + copy_index];
    device uchar *out = all_out + job.output_offset + copy_job.dst_offset;
    device const uchar *src = payload + copy_job.src_offset;
    const uint stride = params.stripes_per_job * params.bytes_per_thread;
    for (uint byte_base = stripe * params.bytes_per_thread;
         byte_base < copy_job.byte_len;
         byte_base += stride) {
        const uint byte_end = min(copy_job.byte_len, byte_base + params.bytes_per_thread);
        for (uint byte_idx = byte_base; byte_idx < byte_end; ++byte_idx) {
            out[byte_idx] = src[byte_idx];
        }
    }
}
