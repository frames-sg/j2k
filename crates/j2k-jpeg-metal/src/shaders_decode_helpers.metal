inline bool jpeg_decode_idct_deposit_region_block_or_skip(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    constant PreparedHuffman &dc_table,
    constant PreparedHuffman &ac_table,
    constant ushort *quant,
    thread int &prev_dc,
    device JpegDecodeStatus *status,
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint origin_x,
    uint origin_y,
    uint block_x,
    uint block_y,
    uint block_width,
    uint block_height,
    thread short coeffs[64],
    thread uchar pixels[64]
) {
    if (!block_intersects_rect(
        block_x,
        block_y,
        block_width,
        block_height,
        origin_x,
        origin_y,
        width,
        height
    )) {
        return decode_block_skip(br, bytes, len, dc_table, ac_table, prev_dc, status);
    }

    bool dc_only = false;
    if (!decode_block(br, bytes, len, dc_table, ac_table, quant, prev_dc, status, coeffs, dc_only)) {
        return false;
    }
    idct_block(coeffs, dc_only, pixels);
    deposit_block_region(plane, stride, width, height, origin_x, origin_y, block_x, block_y, pixels);
    return true;
}

inline bool jpeg_decode_deposit_scaled_region_block_or_skip(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    constant PreparedHuffman &dc_table,
    constant PreparedHuffman &ac_table,
    constant ushort *quant,
    thread int &prev_dc,
    device JpegDecodeStatus *status,
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint origin_x,
    uint origin_y,
    uint block_x,
    uint block_y,
    uint block_width,
    uint block_height,
    uint scale_shift,
    thread short coeffs[64]
) {
    if (!block_intersects_rect(
        block_x,
        block_y,
        block_width,
        block_height,
        origin_x,
        origin_y,
        width,
        height
    )) {
        return decode_block_skip(br, bytes, len, dc_table, ac_table, prev_dc, status);
    }

    bool dc_only = false;
    if (!decode_block(br, bytes, len, dc_table, ac_table, quant, prev_dc, status, coeffs, dc_only)) {
        return false;
    }
    deposit_scaled_block_region(
        plane,
        stride,
        width,
        height,
        origin_x,
        origin_y,
        block_x,
        block_y,
        scale_shift,
        coeffs,
        dc_only
    );
    return true;
}

inline bool jpeg_decode_deposit_scaled_block(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    constant PreparedHuffman &dc_table,
    constant PreparedHuffman &ac_table,
    constant ushort *quant,
    thread int &prev_dc,
    device JpegDecodeStatus *status,
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint block_x,
    uint block_y,
    uint scale_shift,
    thread short coeffs[64]
) {
    bool dc_only = false;
    if (!decode_block(br, bytes, len, dc_table, ac_table, quant, prev_dc, status, coeffs, dc_only)) {
        return false;
    }
    deposit_scaled_block(plane, stride, width, height, block_x, block_y, scale_shift, coeffs, dc_only);
    return true;
}

inline void jpeg_decode_clear_meta_quad(device uint *meta, uint base) {
    for (uint index = 0u; index < 4u; ++index) { meta[base + index] = 0u; }
}

inline uint jpeg_clamped_extent(uint origin, uint span, uint limit) {
    return min(span, limit - min(origin, limit));
}

inline bool jpeg_skip_h2v2_boundary_repair_row(uint row, uint row_count, uint mcu_rows, bool has_top_edge, bool has_bottom_edge) {
    return mcu_rows > 1u && ((has_top_edge && row == 0u) || (has_bottom_edge && row + 1u == row_count));
}

inline uchar h2v1_boundary_left_from_samples(uchar left, uchar right) { return uchar((3u * uint(left) + uint(right) + 2u) >> 2); }

inline uchar h2v1_boundary_right_from_samples(uchar left, uchar right) { return uchar((3u * uint(right) + uint(left) + 1u) >> 2); }

inline void jpeg_write_ycbcr_rgba(
    texture2d<float, access::write> out,
    uint2 pos,
    uchar y_value,
    uchar cb_value,
    uchar cr_value,
    uint alpha
) {
    out.write(rgba_float_ycbcr(y_value, cb_value, cr_value, alpha), pos);
}

inline void jpeg_write_h2v2_boundary_pair(
    texture2d<float, access::write> out,
    uint2 left_pos,
    uint2 right_pos,
    uchar left_y,
    uchar right_y,
    uint left_cb_sum,
    uint left_cr_sum,
    uint right_cb_sum,
    uint right_cr_sum,
    uint alpha
) {
    const uchar left_cb = h2v2_boundary_left_from_sums(left_cb_sum, right_cb_sum);
    const uchar left_cr = h2v2_boundary_left_from_sums(left_cr_sum, right_cr_sum);
    const uchar right_cb = h2v2_boundary_right_from_sums(left_cb_sum, right_cb_sum);
    const uchar right_cr = h2v2_boundary_right_from_sums(left_cr_sum, right_cr_sum);
    jpeg_write_ycbcr_rgba(out, left_pos, left_y, left_cb, left_cr, alpha);
    jpeg_write_ycbcr_rgba(out, right_pos, right_y, right_cb, right_cr, alpha);
}
