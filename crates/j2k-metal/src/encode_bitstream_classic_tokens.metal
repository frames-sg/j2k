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

struct J2kClassicTier1DensityCounters {
    uint sigprop_active_candidates;
    uint sigprop_new_significant;
    uint magref_active_candidates;
    uint cleanup_active_candidates;
    uint cleanup_new_significant;
    uint cleanup_rlc_stripes;
    uint cleanup_rlc_zero_stripes;
    uint arithmetic_sigprop_active_candidates;
    uint arithmetic_sigprop_new_significant;
    uint raw_sigprop_active_candidates;
    uint raw_sigprop_new_significant;
    uint arithmetic_magref_active_candidates;
    uint raw_magref_active_candidates;
    uint reserved0;
    uint reserved1;
};

struct J2kClassicTier1SymbolPlanCounters {
    uint code;
    uint detail;
    uint coding_passes;
    uint missing_bit_planes;
    uint segment_count;
    uint mq_symbol_count;
    uint raw_bit_count;
    uint cleanup_mq_symbol_count;
    uint sigprop_mq_symbol_count;
    uint magref_mq_symbol_count;
    uint raw_sigprop_bit_count;
    uint raw_magref_bit_count;
    uint cleanup_sign_symbol_count;
    uint sigprop_sign_symbol_count;
    uint mq_symbol_hash;
    uint raw_bit_hash;
};

struct J2kClassicTier1PassPlanCounters {
    uint code;
    uint detail;
    uint coding_passes;
    uint missing_bit_planes;
    uint segment_count;
    uint mq_symbol_count;
    uint raw_bit_count;
    uint nonempty_mq_passes;
    uint nonempty_raw_passes;
    uint max_mq_symbols_per_pass;
    uint max_raw_bits_per_pass;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    uint reserved3;
    uint reserved4;
    uint mq_symbols_by_pass[J2K_CLASSIC_TIER1_PASS_PLAN_CAPACITY];
    uint raw_bits_by_pass[J2K_CLASSIC_TIER1_PASS_PLAN_CAPACITY];
};

struct J2kClassicTier1TokenSegment {
    uint token_bit_offset;
    uint token_bit_count;
    uint pass_range;
    uint flags;
};

struct J2kClassicTier1TokenBitWriter {
    device uchar *data;
    uint capacity;
    uint len;
    uint bit_count;
    uint current_byte;
    uint bits_in_current;
    uint failed;
};

struct J2kClassicTier1TokenByteWriter {
    device uchar *data;
    uint capacity;
    uint len;
    uint failed;
};

struct J2kClassicTier1TokenEmitter {
    J2kClassicTier1SymbolPlanCounters counters;
    J2kClassicTier1TokenBitWriter writer;
    device J2kClassicTier1TokenSegment *segments;
    uint segment_capacity;
    uint current_segment_start_bit;
    uint segment_failed;
};

struct J2kClassicTier1SplitTokenEmitter {
    J2kClassicTier1SymbolPlanCounters counters;
    J2kClassicTier1TokenBitWriter mq_writer;
    J2kClassicTier1TokenBitWriter raw_writer;
    device J2kClassicTier1TokenSegment *segments;
    uint segment_capacity;
    uint current_segment_start_bit;
    uint segment_failed;
};

struct J2kClassicTier1SplitMqByteRawTokenEmitter {
    J2kClassicTier1SymbolPlanCounters counters;
    J2kClassicTier1TokenByteWriter mq_writer;
    J2kClassicTier1TokenBitWriter raw_writer;
    device J2kClassicTier1TokenSegment *segments;
    uint segment_capacity;
    uint current_segment_start_bit;
    uint segment_failed;
};

struct J2kClassicTier1PassPlanEmitter {
    J2kClassicTier1PassPlanCounters counters;
    uint current_pass;
};

struct J2kClassicTier1TokenBitReader {
    device const uchar *data;
    uint bit_pos;
    uint bit_capacity;
    uint failed;
};

inline void j2k_classic_tier1_density_zero(thread J2kClassicTier1DensityCounters &counters) {
    counters.sigprop_active_candidates = 0u;
    counters.sigprop_new_significant = 0u;
    counters.magref_active_candidates = 0u;
    counters.cleanup_active_candidates = 0u;
    counters.cleanup_new_significant = 0u;
    counters.cleanup_rlc_stripes = 0u;
    counters.cleanup_rlc_zero_stripes = 0u;
    counters.arithmetic_sigprop_active_candidates = 0u;
    counters.arithmetic_sigprop_new_significant = 0u;
    counters.raw_sigprop_active_candidates = 0u;
    counters.raw_sigprop_new_significant = 0u;
    counters.arithmetic_magref_active_candidates = 0u;
    counters.raw_magref_active_candidates = 0u;
    counters.reserved0 = 0u;
    counters.reserved1 = 0u;
}

inline void j2k_classic_tier1_density_store(
    device J2kClassicTier1DensityCounters *out,
    thread const J2kClassicTier1DensityCounters &counters
) {
    out->sigprop_active_candidates = counters.sigprop_active_candidates;
    out->sigprop_new_significant = counters.sigprop_new_significant;
    out->magref_active_candidates = counters.magref_active_candidates;
    out->cleanup_active_candidates = counters.cleanup_active_candidates;
    out->cleanup_new_significant = counters.cleanup_new_significant;
    out->cleanup_rlc_stripes = counters.cleanup_rlc_stripes;
    out->cleanup_rlc_zero_stripes = counters.cleanup_rlc_zero_stripes;
    out->arithmetic_sigprop_active_candidates = counters.arithmetic_sigprop_active_candidates;
    out->arithmetic_sigprop_new_significant = counters.arithmetic_sigprop_new_significant;
    out->raw_sigprop_active_candidates = counters.raw_sigprop_active_candidates;
    out->raw_sigprop_new_significant = counters.raw_sigprop_new_significant;
    out->arithmetic_magref_active_candidates = counters.arithmetic_magref_active_candidates;
    out->raw_magref_active_candidates = counters.raw_magref_active_candidates;
    out->reserved0 = counters.reserved0;
    out->reserved1 = counters.reserved1;
}

inline void j2k_classic_symbol_plan_zero(thread J2kClassicTier1SymbolPlanCounters &counters) {
    counters.code = J2K_ENCODE_STATUS_OK;
    counters.detail = 0u;
    counters.coding_passes = 0u;
    counters.missing_bit_planes = 0u;
    counters.segment_count = 0u;
    counters.mq_symbol_count = 0u;
    counters.raw_bit_count = 0u;
    counters.cleanup_mq_symbol_count = 0u;
    counters.sigprop_mq_symbol_count = 0u;
    counters.magref_mq_symbol_count = 0u;
    counters.raw_sigprop_bit_count = 0u;
    counters.raw_magref_bit_count = 0u;
    counters.cleanup_sign_symbol_count = 0u;
    counters.sigprop_sign_symbol_count = 0u;
    counters.mq_symbol_hash = 2166136261u;
    counters.raw_bit_hash = 2166136261u;
}

inline void j2k_classic_symbol_plan_store(
    device J2kClassicTier1SymbolPlanCounters *out,
    thread const J2kClassicTier1SymbolPlanCounters &counters
) {
    out->code = counters.code;
    out->detail = counters.detail;
    out->coding_passes = counters.coding_passes;
    out->missing_bit_planes = counters.missing_bit_planes;
    out->segment_count = counters.segment_count;
    out->mq_symbol_count = counters.mq_symbol_count;
    out->raw_bit_count = counters.raw_bit_count;
    out->cleanup_mq_symbol_count = counters.cleanup_mq_symbol_count;
    out->sigprop_mq_symbol_count = counters.sigprop_mq_symbol_count;
    out->magref_mq_symbol_count = counters.magref_mq_symbol_count;
    out->raw_sigprop_bit_count = counters.raw_sigprop_bit_count;
    out->raw_magref_bit_count = counters.raw_magref_bit_count;
    out->cleanup_sign_symbol_count = counters.cleanup_sign_symbol_count;
    out->sigprop_sign_symbol_count = counters.sigprop_sign_symbol_count;
    out->mq_symbol_hash = counters.mq_symbol_hash;
    out->raw_bit_hash = counters.raw_bit_hash;
}

inline void j2k_classic_pass_plan_zero(thread J2kClassicTier1PassPlanCounters &counters) {
    counters.code = J2K_ENCODE_STATUS_OK;
    counters.detail = 0u;
    counters.coding_passes = 0u;
    counters.missing_bit_planes = 0u;
    counters.segment_count = 0u;
    counters.mq_symbol_count = 0u;
    counters.raw_bit_count = 0u;
    counters.nonempty_mq_passes = 0u;
    counters.nonempty_raw_passes = 0u;
    counters.max_mq_symbols_per_pass = 0u;
    counters.max_raw_bits_per_pass = 0u;
    counters.reserved0 = 0u;
    counters.reserved1 = 0u;
    counters.reserved2 = 0u;
    counters.reserved3 = 0u;
    counters.reserved4 = 0u;
    for (uint pass = 0u; pass < J2K_CLASSIC_TIER1_PASS_PLAN_CAPACITY; ++pass) {
        counters.mq_symbols_by_pass[pass] = 0u;
        counters.raw_bits_by_pass[pass] = 0u;
    }
}

inline void j2k_classic_pass_plan_store(
    device J2kClassicTier1PassPlanCounters *out,
    thread const J2kClassicTier1PassPlanCounters &counters
) {
    out->code = counters.code;
    out->detail = counters.detail;
    out->coding_passes = counters.coding_passes;
    out->missing_bit_planes = counters.missing_bit_planes;
    out->segment_count = counters.segment_count;
    out->mq_symbol_count = counters.mq_symbol_count;
    out->raw_bit_count = counters.raw_bit_count;
    out->nonempty_mq_passes = counters.nonempty_mq_passes;
    out->nonempty_raw_passes = counters.nonempty_raw_passes;
    out->max_mq_symbols_per_pass = counters.max_mq_symbols_per_pass;
    out->max_raw_bits_per_pass = counters.max_raw_bits_per_pass;
    out->reserved0 = counters.reserved0;
    out->reserved1 = counters.reserved1;
    out->reserved2 = counters.reserved2;
    out->reserved3 = counters.reserved3;
    out->reserved4 = counters.reserved4;
    for (uint pass = 0u; pass < J2K_CLASSIC_TIER1_PASS_PLAN_CAPACITY; ++pass) {
        out->mq_symbols_by_pass[pass] = counters.mq_symbols_by_pass[pass];
        out->raw_bits_by_pass[pass] = counters.raw_bits_by_pass[pass];
    }
}

inline void j2k_classic_tier1_token_writer_init(
    thread J2kClassicTier1TokenBitWriter &writer,
    device uchar *data,
    uint capacity
) {
    writer.data = data;
    writer.capacity = capacity;
    writer.len = 0u;
    writer.bit_count = 0u;
    writer.current_byte = 0u;
    writer.bits_in_current = 0u;
    writer.failed = 0u;
}

inline void j2k_classic_tier1_token_writer_push_byte(
    thread J2kClassicTier1TokenBitWriter &writer,
    uint value
) {
    if (writer.len >= writer.capacity) {
        writer.failed = 1u;
        return;
    }
    writer.data[writer.len] = uchar(value & 0xFFu);
    writer.len += 1u;
}

inline void j2k_classic_tier1_token_writer_write_bit(
    thread J2kClassicTier1TokenBitWriter &writer,
    uint bit
) {
    writer.current_byte = (writer.current_byte << 1u) | (bit & 1u);
    writer.bits_in_current += 1u;
    writer.bit_count += 1u;
    if (writer.bits_in_current == 8u) {
       j2k_classic_tier1_token_writer_push_byte(writer, writer.current_byte);
        writer.current_byte = 0u;
        writer.bits_in_current = 0u;
    }
}

inline void j2k_classic_tier1_token_writer_write_bits(
    thread J2kClassicTier1TokenBitWriter &writer,
    uint value,
    uint width
) {
    for (uint bit_idx = 0u; bit_idx < width; ++bit_idx) {
        const uint shift = width - 1u - bit_idx;
       j2k_classic_tier1_token_writer_write_bit(writer, (value >> shift) & 1u);
    }
}

inline void j2k_classic_tier1_token_writer_finish(
    thread J2kClassicTier1TokenBitWriter &writer
) {
    if (writer.bits_in_current == 0u) {
        return;
    }
   j2k_classic_tier1_token_writer_push_byte(
        writer,
        writer.current_byte << (8u - writer.bits_in_current)
    );
    writer.current_byte = 0u;
    writer.bits_in_current = 0u;
}

inline void j2k_classic_tier1_token_byte_writer_init(
    thread J2kClassicTier1TokenByteWriter &writer,
    device uchar *data,
    uint capacity
) {
    writer.data = data;
    writer.capacity = capacity;
    writer.len = 0u;
    writer.failed = 0u;
}

inline void j2k_classic_tier1_token_byte_writer_push(
    thread J2kClassicTier1TokenByteWriter &writer,
    uint value
) {
    if (writer.len >= writer.capacity) {
        writer.failed = 1u;
        return;
    }
    writer.data[writer.len] = uchar(value & 0xFFu);
    writer.len += 1u;
}

inline void j2k_classic_tier1_token_emit_init(
    thread J2kClassicTier1TokenEmitter &emitter,
    device uchar *token_data,
    uint token_capacity,
    device J2kClassicTier1TokenSegment *segments,
    uint segment_capacity
) {
   j2k_classic_symbol_plan_zero(emitter.counters);
   j2k_classic_tier1_token_writer_init(emitter.writer, token_data, token_capacity);
    emitter.segments = segments;
    emitter.segment_capacity = segment_capacity;
    emitter.current_segment_start_bit = 0u;
    emitter.segment_failed = 0u;
}

inline void j2k_classic_tier1_token_emit_push_segment(
    thread J2kClassicTier1TokenEmitter &emitter,
    uint start_pass,
    uint end_pass,
    bool use_arithmetic
) {
    const uint segment_idx = emitter.counters.segment_count;
    emitter.counters.segment_count += 1u;
    if (segment_idx >= emitter.segment_capacity) {
        emitter.segment_failed = 1u;
        return;
    }
    const uint token_bit_count = emitter.writer.bit_count - emitter.current_segment_start_bit;
    emitter.segments[segment_idx].token_bit_offset = emitter.current_segment_start_bit;
    emitter.segments[segment_idx].token_bit_count = token_bit_count;
    emitter.segments[segment_idx].pass_range =
        (start_pass & 0xFFFFu) | ((end_pass & 0xFFFFu) << 16u);
    emitter.segments[segment_idx].flags = use_arithmetic ? 1u : 0u;
    emitter.current_segment_start_bit = emitter.writer.bit_count;
}

inline void j2k_classic_tier1_token_emit_store(
    device J2kClassicTier1SymbolPlanCounters *out,
    thread J2kClassicTier1TokenEmitter &emitter
) {
    if (emitter.writer.failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 21u;
    } else if (emitter.segment_failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 22u;
    }
   j2k_classic_symbol_plan_store(out, emitter.counters);
}

inline void j2k_classic_tier1_split_token_emit_init(
    thread J2kClassicTier1SplitTokenEmitter &emitter,
    device uchar *mq_token_data,
    uint mq_token_capacity,
    device uchar *raw_token_data,
    uint raw_token_capacity,
    device J2kClassicTier1TokenSegment *segments,
    uint segment_capacity
) {
   j2k_classic_symbol_plan_zero(emitter.counters);
   j2k_classic_tier1_token_writer_init(emitter.mq_writer, mq_token_data, mq_token_capacity);
   j2k_classic_tier1_token_writer_init(emitter.raw_writer, raw_token_data, raw_token_capacity);
    emitter.segments = segments;
    emitter.segment_capacity = segment_capacity;
    emitter.current_segment_start_bit = 0u;
    emitter.segment_failed = 0u;
}

inline uint j2k_classic_tier1_split_token_emit_stream_bit_count(
    thread J2kClassicTier1SplitTokenEmitter &emitter,
    bool use_arithmetic
) {
    return use_arithmetic ? emitter.mq_writer.bit_count : emitter.raw_writer.bit_count;
}

inline void j2k_classic_tier1_split_token_emit_push_segment(
    thread J2kClassicTier1SplitTokenEmitter &emitter,
    uint start_pass,
    uint end_pass,
    bool use_arithmetic
) {
    const uint segment_idx = emitter.counters.segment_count;
    emitter.counters.segment_count += 1u;
    if (segment_idx >= emitter.segment_capacity) {
        emitter.segment_failed = 1u;
        return;
    }
    const uint stream_bit_count =
       j2k_classic_tier1_split_token_emit_stream_bit_count(emitter, use_arithmetic);
    const uint token_bit_count = stream_bit_count - emitter.current_segment_start_bit;
    emitter.segments[segment_idx].token_bit_offset = emitter.current_segment_start_bit;
    emitter.segments[segment_idx].token_bit_count = token_bit_count;
    emitter.segments[segment_idx].pass_range =
        (start_pass & 0xFFFFu) | ((end_pass & 0xFFFFu) << 16u);
    emitter.segments[segment_idx].flags = use_arithmetic ? 1u : 0u;
}

inline void j2k_classic_tier1_split_token_emit_store(
    device J2kClassicTier1SymbolPlanCounters *out,
    thread J2kClassicTier1SplitTokenEmitter &emitter
) {
    if (emitter.mq_writer.failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 23u;
    } else if (emitter.raw_writer.failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 24u;
    } else if (emitter.segment_failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 22u;
    }
   j2k_classic_symbol_plan_store(out, emitter.counters);
}

inline void j2k_classic_tier1_split_mq_byte_token_emit_init(
    thread J2kClassicTier1SplitMqByteRawTokenEmitter &emitter,
    device uchar *mq_token_data,
    uint mq_token_capacity,
    device uchar *raw_token_data,
    uint raw_token_capacity,
    device J2kClassicTier1TokenSegment *segments,
    uint segment_capacity
) {
   j2k_classic_symbol_plan_zero(emitter.counters);
   j2k_classic_tier1_token_byte_writer_init(
        emitter.mq_writer,
        mq_token_data,
        mq_token_capacity
    );
   j2k_classic_tier1_token_writer_init(emitter.raw_writer, raw_token_data, raw_token_capacity);
    emitter.segments = segments;
    emitter.segment_capacity = segment_capacity;
    emitter.current_segment_start_bit = 0u;
    emitter.segment_failed = 0u;
}

inline uint j2k_classic_tier1_split_mq_byte_token_emit_stream_bit_count(
    thread J2kClassicTier1SplitMqByteRawTokenEmitter &emitter,
    bool use_arithmetic
) {
    return use_arithmetic ? emitter.mq_writer.len * 8u : emitter.raw_writer.bit_count;
}

inline void j2k_classic_tier1_split_mq_byte_token_emit_push_segment(
    thread J2kClassicTier1SplitMqByteRawTokenEmitter &emitter,
    uint start_pass,
    uint end_pass,
    bool use_arithmetic
) {
    const uint segment_idx = emitter.counters.segment_count;
    emitter.counters.segment_count += 1u;
    if (segment_idx >= emitter.segment_capacity) {
        emitter.segment_failed = 1u;
        return;
    }
    const uint stream_bit_count =
       j2k_classic_tier1_split_mq_byte_token_emit_stream_bit_count(emitter, use_arithmetic);
    const uint token_bit_count = stream_bit_count - emitter.current_segment_start_bit;
    emitter.segments[segment_idx].token_bit_offset = emitter.current_segment_start_bit;
    emitter.segments[segment_idx].token_bit_count = token_bit_count;
    emitter.segments[segment_idx].pass_range =
        (start_pass & 0xFFFFu) | ((end_pass & 0xFFFFu) << 16u);
    emitter.segments[segment_idx].flags = use_arithmetic ? 3u : 0u;
}

inline void j2k_classic_tier1_split_mq_byte_token_emit_store(
    device J2kClassicTier1SymbolPlanCounters *out,
    thread J2kClassicTier1SplitMqByteRawTokenEmitter &emitter
) {
    if (emitter.mq_writer.failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 25u;
    } else if (emitter.raw_writer.failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 24u;
    } else if (emitter.segment_failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 22u;
    }
   j2k_classic_symbol_plan_store(out, emitter.counters);
}

inline void j2k_classic_tier1_token_reader_init(
    thread J2kClassicTier1TokenBitReader &reader,
    device const uchar *data,
    uint capacity_bits
) {
    reader.data = data;
    reader.bit_pos = 0u;
    reader.bit_capacity = capacity_bits;
    reader.failed = 0u;
}

inline void j2k_classic_tier1_token_reader_seek(
    thread J2kClassicTier1TokenBitReader &reader,
    uint bit_pos
) {
    if (bit_pos > reader.bit_capacity) {
        reader.failed = 1u;
        return;
    }
    reader.bit_pos = bit_pos;
}

inline uint j2k_classic_tier1_token_reader_read_bits(
    thread J2kClassicTier1TokenBitReader &reader,
    uint count
) {
    if (count > 32u || reader.bit_pos > reader.bit_capacity ||
        count > (reader.bit_capacity - reader.bit_pos)) {
        reader.failed = 1u;
        return 0u;
    }
    uint value = 0u;
    for (uint idx = 0u; idx < count; ++idx) {
        const uint byte_idx = reader.bit_pos >> 3u;
        const uint shift = 7u - (reader.bit_pos & 7u);
        value = (value << 1u) | ((uint(reader.data[byte_idx]) >> shift) & 1u);
        reader.bit_pos += 1u;
    }
    return value;
}

inline void j2k_pack_classic_tier1_tokens_bypass_u16_32_impl(
    J2kClassicEncodeParams params,
    J2kClassicTier1SymbolPlanCounters counters,
    device const uchar *token_data,
    device const J2kClassicTier1TokenSegment *token_segments,
    uint token_capacity_bytes,
    uint token_segment_capacity,
    device uchar *out,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments
) {
   j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 30u, 0u, 0u, 0u, 0u);

    if (params.style_flags != J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) {
       j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_UNSUPPORTED,
            2u,
            0u,
            counters.coding_passes,
            counters.missing_bit_planes,
            0u
        );
        return;
    }
    if (counters.code != J2K_ENCODE_STATUS_OK) {
       j2k_set_encode_status(
            status,
            counters.code,
            counters.detail,
            0u,
            counters.coding_passes,
            counters.missing_bit_planes,
            0u
        );
        return;
    }
    if (counters.segment_count > token_segment_capacity ||
        counters.segment_count > params.segment_capacity) {
       j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_FAIL,
            31u,
            0u,
            counters.coding_passes,
            counters.missing_bit_planes,
            0u
        );
        return;
    }

    const uint token_capacity_bits = token_capacity_bytes * 8u;
    thread J2kClassicTier1TokenBitReader reader;
   j2k_classic_tier1_token_reader_init(reader, token_data, token_capacity_bits);
    thread uchar contexts[19];
    reset_contexts(contexts);

    uint data_cursor = 0u;
    uint segment_count = 0u;
    for (uint segment_idx = 0u; segment_idx < counters.segment_count; ++segment_idx) {
        const J2kClassicTier1TokenSegment token_segment = token_segments[segment_idx];
        const uint start_pass = token_segment.pass_range & 0xFFFFu;
        const uint end_pass = token_segment.pass_range >> 16u;
        const bool use_arithmetic = (token_segment.flags & 1u) != 0u;
        if ((token_segment.flags & ~1u) != 0u ||
            start_pass > end_pass ||
            end_pass > counters.coding_passes) {
           j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                32u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        if (token_segment.token_bit_offset > token_capacity_bits ||
            token_segment.token_bit_count > (token_capacity_bits - token_segment.token_bit_offset)) {
           j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                33u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        if (data_cursor > params.output_capacity) {
           j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                34u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
       j2k_classic_tier1_token_reader_seek(reader, token_segment.token_bit_offset);
        if (reader.failed != 0u) {
           j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                35u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }

        uint segment_len = 0u;
        if (use_arithmetic) {
            if ((token_segment.token_bit_count % 6u) != 0u) {
               j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    36u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            thread J2kMqEncoder encoder;
           j2k_mq_init(encoder, out + data_cursor, params.output_capacity - data_cursor);
            const uint symbol_count = token_segment.token_bit_count / 6u;
            for (uint symbol_idx = 0u; symbol_idx < symbol_count; ++symbol_idx) {
                const uint token = j2k_classic_tier1_token_reader_read_bits(reader, 6u);
                if (reader.failed != 0u) {
                   j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        37u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
                const uint ctx_label = token & 0x1Fu;
                if (ctx_label >= 19u) {
                   j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        38u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
               j2k_mq_encode(encoder, contexts, ctx_label, (token >> 5u) & 1u);
                if (encoder.failed != 0u) {
                   j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        39u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
            }
            segment_len = j2k_classic_finish_arithmetic_segment(encoder);
            if (encoder.failed != 0u) {
               j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    40u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
        } else {
            thread J2kRawBitWriter writer;
           j2k_raw_writer_init(writer, out + data_cursor, params.output_capacity - data_cursor);
            for (uint bit_idx = 0u; bit_idx < token_segment.token_bit_count; ++bit_idx) {
                const uint bit = j2k_classic_tier1_token_reader_read_bits(reader, 1u);
                if (reader.failed != 0u) {
                   j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        41u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
               j2k_raw_writer_write_bit(writer, bit);
                if (writer.failed != 0u) {
                   j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        42u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
            }
           j2k_raw_writer_finish(writer);
            if (writer.failed != 0u) {
               j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    43u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            segment_len = writer.len;
        }

        if (segment_len > (params.output_capacity - data_cursor)) {
           j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                44u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        if (!j2k_classic_push_segment(
                segments,
                params.segment_capacity,
                segment_count,
                data_cursor,
                segment_len,
                start_pass,
                end_pass,
                use_arithmetic)) {
           j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                45u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        data_cursor += segment_len;
    }

   j2k_set_encode_status(
        status,
        J2K_ENCODE_STATUS_OK,
        0u,
        data_cursor,
        counters.coding_passes,
        counters.missing_bit_planes,
        segment_count
    );
}

inline void j2k_pack_classic_tier1_split_tokens_bypass_u16_32_impl(
    J2kClassicEncodeParams params,
    J2kClassicTier1SymbolPlanCounters counters,
    device const uchar *mq_token_data,
    device const uchar *raw_token_data,
    device const J2kClassicTier1TokenSegment *token_segments,
    uint mq_token_capacity_bytes,
    uint raw_token_capacity_bytes,
    uint token_segment_capacity,
    device uchar *out,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments
) {
   j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 70u, 0u, 0u, 0u, 0u);

    if (params.style_flags != J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) {
       j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_UNSUPPORTED,
            2u,
            0u,
            counters.coding_passes,
            counters.missing_bit_planes,
            0u
        );
        return;
    }
    if (counters.code != J2K_ENCODE_STATUS_OK) {
       j2k_set_encode_status(
            status,
            counters.code,
            counters.detail,
            0u,
            counters.coding_passes,
            counters.missing_bit_planes,
            0u
        );
        return;
    }
    if (counters.segment_count > token_segment_capacity ||
        counters.segment_count > params.segment_capacity) {
       j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_FAIL,
            71u,
            0u,
            counters.coding_passes,
            counters.missing_bit_planes,
            0u
        );
        return;
    }

    const uint mq_token_capacity_bits = mq_token_capacity_bytes * 8u;
    const uint raw_token_capacity_bits = raw_token_capacity_bytes * 8u;
    thread J2kClassicTier1TokenBitReader mq_reader;
    thread J2kClassicTier1TokenBitReader raw_reader;
   j2k_classic_tier1_token_reader_init(mq_reader, mq_token_data, mq_token_capacity_bits);
   j2k_classic_tier1_token_reader_init(raw_reader, raw_token_data, raw_token_capacity_bits);
    thread uchar contexts[19];
    reset_contexts(contexts);

    uint data_cursor = 0u;
    uint segment_count = 0u;
    for (uint segment_idx = 0u; segment_idx < counters.segment_count; ++segment_idx) {
        const J2kClassicTier1TokenSegment token_segment = token_segments[segment_idx];
        const uint start_pass = token_segment.pass_range & 0xFFFFu;
        const uint end_pass = token_segment.pass_range >> 16u;
        const bool use_arithmetic = (token_segment.flags & 1u) != 0u;
        const bool mq_tokens_are_bytes = (token_segment.flags & 2u) != 0u;
        const uint token_capacity_bits =
            use_arithmetic ? mq_token_capacity_bits : raw_token_capacity_bits;
        if ((token_segment.flags & ~3u) != 0u ||
            (!use_arithmetic && mq_tokens_are_bytes) ||
            start_pass > end_pass ||
            end_pass > counters.coding_passes) {
           j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                72u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        if (token_segment.token_bit_offset > token_capacity_bits ||
            token_segment.token_bit_count > (token_capacity_bits - token_segment.token_bit_offset)) {
           j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                73u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        if (data_cursor > params.output_capacity) {
           j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                74u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }

        uint segment_len = 0u;
        if (use_arithmetic) {
            if (!mq_tokens_are_bytes && (token_segment.token_bit_count % 6u) != 0u) {
               j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    75u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            if (mq_tokens_are_bytes &&
                ((token_segment.token_bit_offset | token_segment.token_bit_count) & 7u) != 0u) {
               j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    75u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            if (!mq_tokens_are_bytes) {
               j2k_classic_tier1_token_reader_seek(mq_reader, token_segment.token_bit_offset);
            }
            if (!mq_tokens_are_bytes && mq_reader.failed != 0u) {
               j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    76u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            thread J2kMqEncoder encoder;
           j2k_mq_init(encoder, out + data_cursor, params.output_capacity - data_cursor);
            const uint symbol_count =
                mq_tokens_are_bytes ? (token_segment.token_bit_count >> 3u) :
                (token_segment.token_bit_count / 6u);
            const uint mq_byte_offset = token_segment.token_bit_offset >> 3u;
            for (uint symbol_idx = 0u; symbol_idx < symbol_count; ++symbol_idx) {
                const uint token = mq_tokens_are_bytes
                    ? uint(mq_token_data[mq_byte_offset + symbol_idx])
                    : j2k_classic_tier1_token_reader_read_bits(mq_reader, 6u);
                if (!mq_tokens_are_bytes && mq_reader.failed != 0u) {
                   j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        77u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
                const uint ctx_label = token & 0x1Fu;
                if (ctx_label >= 19u) {
                   j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        78u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
               j2k_mq_encode(encoder, contexts, ctx_label, (token >> 5u) & 1u);
                if (encoder.failed != 0u) {
                   j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        79u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
            }
            segment_len = j2k_classic_finish_arithmetic_segment(encoder);
            if (encoder.failed != 0u) {
               j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    80u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
        } else {
           j2k_classic_tier1_token_reader_seek(raw_reader, token_segment.token_bit_offset);
            if (raw_reader.failed != 0u) {
               j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    81u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            thread J2kRawBitWriter writer;
           j2k_raw_writer_init(writer, out + data_cursor, params.output_capacity - data_cursor);
            for (uint bit_idx = 0u; bit_idx < token_segment.token_bit_count; ++bit_idx) {
                const uint bit = j2k_classic_tier1_token_reader_read_bits(raw_reader, 1u);
                if (raw_reader.failed != 0u) {
                   j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        82u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
               j2k_raw_writer_write_bit(writer, bit);
                if (writer.failed != 0u) {
                   j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        83u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
            }
           j2k_raw_writer_finish(writer);
            if (writer.failed != 0u) {
               j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    84u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            segment_len = writer.len;
        }

        if (segment_len > (params.output_capacity - data_cursor)) {
           j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                85u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        if (!j2k_classic_push_segment(
                segments,
                params.segment_capacity,
                segment_count,
                data_cursor,
                segment_len,
                start_pass,
                end_pass,
                use_arithmetic)) {
           j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                86u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        data_cursor += segment_len;
    }

   j2k_set_encode_status(
        status,
        J2K_ENCODE_STATUS_OK,
        0u,
        data_cursor,
        counters.coding_passes,
        counters.missing_bit_planes,
        segment_count
    );
}
