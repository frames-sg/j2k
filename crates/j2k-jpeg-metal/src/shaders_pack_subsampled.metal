kernel void jpeg_pack_420(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint y_idx = gid.y * params.width + gid.x;
    if (params.out_format == OUT_GRAY) {
        out[gid.y * params.out_stride + gid.x] = y_plane[y_idx];
        return;
    }

    uchar cb;
    uchar cr;
    jpeg_sample_420_chroma(
        cb_plane,
        cr_plane,
        params.chroma_width,
        params.chroma_height,
        gid.x,
        gid.y,
        cb,
        cr
    );

    uint out_idx = gid.y * params.out_stride + gid.x * (params.out_format == OUT_RGB ? 3u : 4u);
    store_rgb_ycbcr(out, out_idx, y_plane[y_idx], cb, cr);
    if (params.out_format == OUT_RGBA) {
        out[out_idx + 3] = uchar(params.alpha);
    }
}

kernel void jpeg_pack_420_rgb(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint y_idx = gid.y * params.width + gid.x;
    uchar cb;
    uchar cr;
    jpeg_sample_420_chroma(
        cb_plane,
        cr_plane,
        params.chroma_width,
        params.chroma_height,
        gid.x,
        gid.y,
        cb,
        cr
    );

    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    store_rgb_ycbcr(out, out_idx, y_plane[y_idx], cb, cr);
}

kernel void jpeg_pack_420_rgb_batch(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420BatchParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint x0 = gid.x * 2u;
    const uint y0 = gid.y * 2u;
    if (x0 >= params.width || y0 >= params.height || gid.z >= params.tile_count) {
        return;
    }

    const uint y_plane_base = gid.z * params.width * params.height;
    const uint chroma_plane_base = gid.z * params.chroma_width * params.chroma_height;
    device const uchar *tile_y_plane = y_plane + y_plane_base;
    device const uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device const uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    const uint x1 = x0 + 1u;
    const uint out_base = gid.z * params.out_stride * params.height;

    const uint chroma_y = min(y0 / 2u, params.chroma_height - 1u);
    const uint near_y = (y0 & 1u) == 0u
        ? (chroma_y == 0u ? 0u : chroma_y - 1u)
        : min(chroma_y + 1u, params.chroma_height - 1u);
    device const uchar *curr_cb = tile_cb_plane + chroma_y * params.chroma_width;
    device const uchar *near_cb = tile_cb_plane + near_y * params.chroma_width;
    device const uchar *curr_cr = tile_cr_plane + chroma_y * params.chroma_width;
    device const uchar *near_cr = tile_cr_plane + near_y * params.chroma_width;

    uchar cb0;
    uchar cb1;
    uchar cr0;
    uchar cr1;
    h2v2_sample_even_pair(near_cb, curr_cb, params.chroma_width, x0, cb0, cb1);
    h2v2_sample_even_pair(near_cr, curr_cr, params.chroma_width, x0, cr0, cr1);

    const uint y_idx0 = y0 * params.width + x0;
    const uint out_idx0 = out_base + y0 * params.out_stride + x0 * 3u;
    store_rgb_ycbcr(out, out_idx0, tile_y_plane[y_idx0], cb0, cr0);
    if (x1 < params.width) {
        store_rgb_ycbcr(out, out_idx0 + 3u, tile_y_plane[y_idx0 + 1u], cb1, cr1);
    }

    const uint y1 = y0 + 1u;
    if (y1 >= params.height) {
        return;
    }

    const uint chroma_y1 = min(y1 / 2u, params.chroma_height - 1u);
    const uint near_y1 = (y1 & 1u) == 0u
        ? (chroma_y1 == 0u ? 0u : chroma_y1 - 1u)
        : min(chroma_y1 + 1u, params.chroma_height - 1u);
    device const uchar *curr_cb1 = tile_cb_plane + chroma_y1 * params.chroma_width;
    device const uchar *near_cb1 = tile_cb_plane + near_y1 * params.chroma_width;
    device const uchar *curr_cr1 = tile_cr_plane + chroma_y1 * params.chroma_width;
    device const uchar *near_cr1 = tile_cr_plane + near_y1 * params.chroma_width;

    h2v2_sample_even_pair(near_cb1, curr_cb1, params.chroma_width, x0, cb0, cb1);
    h2v2_sample_even_pair(near_cr1, curr_cr1, params.chroma_width, x0, cr0, cr1);

    const uint y_idx1 = y1 * params.width + x0;
    const uint out_idx1 = out_base + y1 * params.out_stride + x0 * 3u;
    store_rgb_ycbcr(out, out_idx1, tile_y_plane[y_idx1], cb0, cr0);
    if (x1 < params.width) {
        store_rgb_ycbcr(out, out_idx1 + 3u, tile_y_plane[y_idx1 + 1u], cb1, cr1);
    }
}

kernel void jpeg_pack_420_rgba(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint y_idx = gid.y * params.width + gid.x;
    uchar cb;
    uchar cr;
    jpeg_sample_420_chroma(
        cb_plane,
        cr_plane,
        params.chroma_width,
        params.chroma_height,
        gid.x,
        gid.y,
        cb,
        cr
    );

    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    store_rgba_ycbcr(out, out_idx, y_plane[y_idx], cb, cr, params.alpha);
}

kernel void jpeg_pack_420_rgba_texture(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    constant JpegTexturePackBatchParams &params [[buffer(3)]],
    texture2d<float, access::write> out [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint x0 = gid.x * 2u;
    const uint y0 = gid.y * 2u;
    if (x0 >= params.width || y0 >= params.height) {
        return;
    }

    const uint y_plane_base = params.tile_index * params.width * params.height;
    const uint chroma_plane_base = params.tile_index * params.chroma_width * params.chroma_height;
    device const uchar *tile_y_plane = y_plane + y_plane_base;
    device const uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device const uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    const uint chroma_y = min(y0 / 2u, params.chroma_height - 1u);
    const uint near_y = (y0 & 1u) == 0u
        ? (chroma_y == 0u ? 0u : chroma_y - 1u)
        : min(chroma_y + 1u, params.chroma_height - 1u);
    device const uchar *curr_cb = tile_cb_plane + chroma_y * params.chroma_width;
    device const uchar *near_cb = tile_cb_plane + near_y * params.chroma_width;
    device const uchar *curr_cr = tile_cr_plane + chroma_y * params.chroma_width;
    device const uchar *near_cr = tile_cr_plane + near_y * params.chroma_width;

    uchar cb0;
    uchar cb1;
    uchar cr0;
    uchar cr1;
    h2v2_sample_even_pair(near_cb, curr_cb, params.chroma_width, x0, cb0, cb1);
    h2v2_sample_even_pair(near_cr, curr_cr, params.chroma_width, x0, cr0, cr1);

    const uint y_idx0 = y0 * params.width + x0;
    out.write(
        rgba_float_ycbcr(tile_y_plane[y_idx0], cb0, cr0, params.alpha),
        uint2(x0, y0)
    );
    const uint x1 = x0 + 1u;
    if (x1 < params.width) {
        out.write(
            rgba_float_ycbcr(tile_y_plane[y_idx0 + 1u], cb1, cr1, params.alpha),
            uint2(x1, y0)
        );
    }

    const uint y1 = y0 + 1u;
    if (y1 >= params.height) {
        return;
    }

    const uint chroma_y1 = min(y1 / 2u, params.chroma_height - 1u);
    const uint near_y1 = (y1 & 1u) == 0u
        ? (chroma_y1 == 0u ? 0u : chroma_y1 - 1u)
        : min(chroma_y1 + 1u, params.chroma_height - 1u);
    device const uchar *curr_cb1 = tile_cb_plane + chroma_y1 * params.chroma_width;
    device const uchar *near_cb1 = tile_cb_plane + near_y1 * params.chroma_width;
    device const uchar *curr_cr1 = tile_cr_plane + chroma_y1 * params.chroma_width;
    device const uchar *near_cr1 = tile_cr_plane + near_y1 * params.chroma_width;

    h2v2_sample_even_pair(near_cb1, curr_cb1, params.chroma_width, x0, cb0, cb1);
    h2v2_sample_even_pair(near_cr1, curr_cr1, params.chroma_width, x0, cr0, cr1);

    const uint y_idx1 = y1 * params.width + x0;
    out.write(
        rgba_float_ycbcr(tile_y_plane[y_idx1], cb0, cr0, params.alpha),
        uint2(x0, y1)
    );
    if (x1 < params.width) {
        out.write(
            rgba_float_ycbcr(tile_y_plane[y_idx1 + 1u], cb1, cr1, params.alpha),
            uint2(x1, y1)
        );
    }
}

kernel void jpeg_pack_422_rgb(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint y_idx = gid.y * params.width + gid.x;
    uchar cb;
    uchar cr;
    jpeg_sample_422_chroma(
        cb_plane,
        cr_plane,
        params.chroma_width,
        params.chroma_height,
        gid.x,
        gid.y,
        cb,
        cr
    );

    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    store_rgb_ycbcr(out, out_idx, y_plane[y_idx], cb, cr);
}

kernel void jpeg_pack_422_rgb_batch(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420BatchParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint x0 = gid.x * 2u;
    if (x0 >= params.width || gid.y >= params.height || gid.z >= params.tile_count) {
        return;
    }

    const uint y_plane_base = gid.z * params.width * params.height;
    const uint chroma_plane_base = gid.z * params.chroma_width * params.chroma_height;
    device const uchar *tile_y_plane = y_plane + y_plane_base;
    device const uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device const uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    const uint x1 = x0 + 1u;
    const uint y_idx = gid.y * params.width + x0;
    const uint chroma_y = min(gid.y, params.chroma_height - 1u);
    device const uchar *curr_cb = tile_cb_plane + chroma_y * params.chroma_width;
    device const uchar *curr_cr = tile_cr_plane + chroma_y * params.chroma_width;

    uchar cb0;
    uchar cb1;
    uchar cr0;
    uchar cr1;
    h2v1_sample_even_pair(curr_cb, params.chroma_width, x0, cb0, cb1);
    h2v1_sample_even_pair(curr_cr, params.chroma_width, x0, cr0, cr1);

    const uint out_base = gid.z * params.out_stride * params.height;
    const uint out_idx = out_base + gid.y * params.out_stride + x0 * 3u;
    store_rgb_ycbcr(out, out_idx, tile_y_plane[y_idx], cb0, cr0);
    if (x1 < params.width) {
        store_rgb_ycbcr(out, out_idx + 3u, tile_y_plane[y_idx + 1u], cb1, cr1);
    }
}

kernel void jpeg_pack_422_rgba_texture(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    constant JpegTexturePackBatchParams &params [[buffer(3)]],
    texture2d<float, access::write> out [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint x0 = gid.x * 2u;
    if (x0 >= params.width || gid.y >= params.height) {
        return;
    }

    const uint y_plane_base = params.tile_index * params.width * params.height;
    const uint chroma_plane_base = params.tile_index * params.chroma_width * params.chroma_height;
    device const uchar *tile_y_plane = y_plane + y_plane_base;
    device const uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device const uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    const uint y_idx = gid.y * params.width + x0;
    const uint chroma_y = min(gid.y, params.chroma_height - 1u);
    device const uchar *curr_cb = tile_cb_plane + chroma_y * params.chroma_width;
    device const uchar *curr_cr = tile_cr_plane + chroma_y * params.chroma_width;

    uchar cb0;
    uchar cb1;
    uchar cr0;
    uchar cr1;
    h2v1_sample_even_pair(curr_cb, params.chroma_width, x0, cb0, cb1);
    h2v1_sample_even_pair(curr_cr, params.chroma_width, x0, cr0, cr1);

    out.write(
        rgba_float_ycbcr(tile_y_plane[y_idx], cb0, cr0, params.alpha),
        uint2(x0, gid.y)
    );
    const uint x1 = x0 + 1u;
    if (x1 < params.width) {
        out.write(
            rgba_float_ycbcr(tile_y_plane[y_idx + 1u], cb1, cr1, params.alpha),
            uint2(x1, gid.y)
        );
    }
}

kernel void jpeg_pack_422_rgba(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint y_idx = gid.y * params.width + gid.x;
    uchar cb;
    uchar cr;
    jpeg_sample_422_chroma(
        cb_plane,
        cr_plane,
        params.chroma_width,
        params.chroma_height,
        gid.x,
        gid.y,
        cb,
        cr
    );

    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    store_rgba_ycbcr(out, out_idx, y_plane[y_idx], cb, cr, params.alpha);
}

kernel void jpeg_pack_422_windowed(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420WindowedPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_idx = src_y * params.src_width + src_x;
    if (params.out_format == OUT_GRAY) {
        out[gid.y * params.out_stride + gid.x] = y_plane[y_idx];
        return;
    }

    uchar cb;
    uchar cr;
    jpeg_sample_422_chroma(
        cb_plane,
        cr_plane,
        params.chroma_width,
        params.chroma_height,
        src_x,
        src_y,
        cb,
        cr
    );

    uint out_idx = gid.y * params.out_stride + gid.x * (params.out_format == OUT_RGB ? 3u : 4u);
    store_rgb_ycbcr(out, out_idx, y_plane[y_idx], cb, cr);
    if (params.out_format == OUT_RGBA) {
        out[out_idx + 3] = uchar(params.alpha);
    }
}

kernel void jpeg_pack_422_windowed_rgb(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420WindowedPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_idx = src_y * params.src_width + src_x;
    uchar cb;
    uchar cr;
    jpeg_sample_422_chroma(
        cb_plane,
        cr_plane,
        params.chroma_width,
        params.chroma_height,
        src_x,
        src_y,
        cb,
        cr
    );

    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    store_rgb_ycbcr(out, out_idx, y_plane[y_idx], cb, cr);
}

kernel void jpeg_pack_422_windowed_rgb_batch(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegWindowedPackBatchParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.tile_count) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_plane_base = gid.z * params.src_width * params.src_height;
    const uint chroma_plane_base = gid.z * params.chroma_width * params.chroma_height;
    device const uchar *tile_y_plane = y_plane + y_plane_base;
    device const uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device const uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    const uint y_idx = src_y * params.src_width + src_x;
    uchar cb;
    uchar cr;
    jpeg_sample_422_chroma(
        tile_cb_plane,
        tile_cr_plane,
        params.chroma_width,
        params.chroma_height,
        src_x,
        src_y,
        cb,
        cr
    );

    const uint out_base = gid.z * params.out_stride * params.height;
    const uint out_idx = out_base + gid.y * params.out_stride + gid.x * 3u;
    store_rgb_ycbcr(out, out_idx, tile_y_plane[y_idx], cb, cr);
}

kernel void jpeg_pack_422_windowed_rgba_texture(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    constant JpegWindowedTexturePackBatchParams &params [[buffer(3)]],
    texture2d<float, access::write> out [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_plane_base = params.tile_index * params.src_width * params.src_height;
    const uint chroma_plane_base = params.tile_index * params.chroma_width * params.chroma_height;
    device const uchar *tile_y_plane = y_plane + y_plane_base;
    device const uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device const uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    const uint y_idx = src_y * params.src_width + src_x;
    uchar cb;
    uchar cr;
    jpeg_sample_422_chroma(
        tile_cb_plane,
        tile_cr_plane,
        params.chroma_width,
        params.chroma_height,
        src_x,
        src_y,
        cb,
        cr
    );
    out.write(rgba_float_ycbcr(tile_y_plane[y_idx], cb, cr, params.alpha), gid);
}

kernel void jpeg_pack_422_windowed_rgba(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420WindowedPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_idx = src_y * params.src_width + src_x;
    uchar cb;
    uchar cr;
    jpeg_sample_422_chroma(
        cb_plane,
        cr_plane,
        params.chroma_width,
        params.chroma_height,
        src_x,
        src_y,
        cb,
        cr
    );

    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    store_rgba_ycbcr(out, out_idx, y_plane[y_idx], cb, cr, params.alpha);
}

kernel void jpeg_pack_420_windowed(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420WindowedPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_idx = src_y * params.src_width + src_x;
    if (params.out_format == OUT_GRAY) {
        out[gid.y * params.out_stride + gid.x] = y_plane[y_idx];
        return;
    }

    uchar cb;
    uchar cr;
    jpeg_sample_420_chroma(
        cb_plane,
        cr_plane,
        params.chroma_width,
        params.chroma_height,
        src_x,
        src_y,
        cb,
        cr
    );

    uint out_idx = gid.y * params.out_stride + gid.x * (params.out_format == OUT_RGB ? 3u : 4u);
    store_rgb_ycbcr(out, out_idx, y_plane[y_idx], cb, cr);
    if (params.out_format == OUT_RGBA) {
        out[out_idx + 3] = uchar(params.alpha);
    }
}

kernel void jpeg_pack_420_windowed_rgb(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420WindowedPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_idx = src_y * params.src_width + src_x;
    uchar cb;
    uchar cr;
    jpeg_sample_420_chroma(
        cb_plane,
        cr_plane,
        params.chroma_width,
        params.chroma_height,
        src_x,
        src_y,
        cb,
        cr
    );

    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    store_rgb_ycbcr(out, out_idx, y_plane[y_idx], cb, cr);
}

kernel void jpeg_pack_420_windowed_rgb_batch(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegWindowedPackBatchParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.tile_count) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_plane_base = gid.z * params.src_width * params.src_height;
    const uint chroma_plane_base = gid.z * params.chroma_width * params.chroma_height;
    device const uchar *tile_y_plane = y_plane + y_plane_base;
    device const uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device const uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    const uint y_idx = src_y * params.src_width + src_x;
    uchar cb;
    uchar cr;
    jpeg_sample_420_chroma(
        tile_cb_plane,
        tile_cr_plane,
        params.chroma_width,
        params.chroma_height,
        src_x,
        src_y,
        cb,
        cr
    );

    const uint out_base = gid.z * params.out_stride * params.height;
    const uint out_idx = out_base + gid.y * params.out_stride + gid.x * 3u;
    store_rgb_ycbcr(out, out_idx, tile_y_plane[y_idx], cb, cr);
}

kernel void jpeg_pack_420_windowed_rgba_texture(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    constant JpegWindowedTexturePackBatchParams &params [[buffer(3)]],
    texture2d<float, access::write> out [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_plane_base = params.tile_index * params.src_width * params.src_height;
    const uint chroma_plane_base = params.tile_index * params.chroma_width * params.chroma_height;
    device const uchar *tile_y_plane = y_plane + y_plane_base;
    device const uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device const uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    const uint y_idx = src_y * params.src_width + src_x;
    uchar cb;
    uchar cr;
    jpeg_sample_420_chroma(
        tile_cb_plane,
        tile_cr_plane,
        params.chroma_width,
        params.chroma_height,
        src_x,
        src_y,
        cb,
        cr
    );
    out.write(rgba_float_ycbcr(tile_y_plane[y_idx], cb, cr, params.alpha), gid);
}

kernel void jpeg_copy_rgb8_to_rgba_texture(
    device const uchar *rgb [[buffer(0)]],
    constant JpegRgb8ToRgbaTextureParams &params [[buffer(1)]],
    texture2d<float, access::write> out [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.in_stride + gid.x * 3u;
    out.write(float4(
        float(rgb[idx]) / 255.0f,
        float(rgb[idx + 1u]) / 255.0f,
        float(rgb[idx + 2u]) / 255.0f,
        float(params.alpha) / 255.0f
    ), gid);
}

kernel void jpeg_pack_420_windowed_rgba(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420WindowedPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_idx = src_y * params.src_width + src_x;
    uchar cb;
    uchar cr;
    jpeg_sample_420_chroma(
        cb_plane,
        cr_plane,
        params.chroma_width,
        params.chroma_height,
        src_x,
        src_y,
        cb,
        cr
    );

    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    store_rgba_ycbcr(out, out_idx, y_plane[y_idx], cb, cr, params.alpha);
}
