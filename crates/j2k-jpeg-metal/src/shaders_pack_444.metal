kernel void jpeg_pack(
    device const uchar *plane0 [[buffer(0)]],
    device const uchar *plane1 [[buffer(1)]],
    device const uchar *plane2 [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    uint out_idx = gid.y * params.out_stride;

    if (params.out_format == OUT_GRAY) {
        out_idx += gid.x;
        if (params.mode == MODE_GRAY || params.mode == MODE_YCBCR) {
            out[out_idx] = plane0[idx];
            return;
        }

        const uint r = plane0[idx];
        const uint g = plane1[idx];
        const uint b = plane2[idx];
        out[out_idx] = uchar((77u * r + 150u * g + 29u * b + 128u) >> 8);
        return;
    }

    out_idx += gid.x * (params.out_format == OUT_RGB ? 3u : 4u);

    if (params.mode == MODE_GRAY) {
        const uchar gray = plane0[idx];
        out[out_idx] = gray;
        out[out_idx + 1] = gray;
        out[out_idx + 2] = gray;
    } else if (params.mode == MODE_RGB) {
        out[out_idx] = plane0[idx];
        out[out_idx + 1] = plane1[idx];
        out[out_idx + 2] = plane2[idx];
    } else {
        store_rgb_ycbcr(out, out_idx, plane0[idx], plane1[idx], plane2[idx]);
    }

    if (params.out_format == OUT_RGBA) {
        out[out_idx + 3] = uchar(params.alpha);
    }
}

kernel void jpeg_pack_444_rgb_batch(
    device const uchar *plane0 [[buffer(0)]],
    device const uchar *plane1 [[buffer(1)]],
    device const uchar *plane2 [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegWindowedPackBatchParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.tile_count) {
        return;
    }

    const uint plane_len = params.src_width * params.src_height;
    const uint plane_base = gid.z * plane_len;
    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint idx = plane_base + src_y * params.src_width + src_x;
    const uint out_base = gid.z * params.out_stride * params.height;
    const uint out_idx =
        out_base + gid.y * params.out_stride + gid.x * (params.out_format == OUT_RGB ? 3u : 4u);

    if (params.mode == MODE_GRAY) {
        const uchar gray = plane0[idx];
        out[out_idx] = gray;
        out[out_idx + 1] = gray;
        out[out_idx + 2] = gray;
    } else if (params.mode == MODE_RGB) {
        out[out_idx] = plane0[idx];
        out[out_idx + 1] = plane1[idx];
        out[out_idx + 2] = plane2[idx];
    } else {
        store_rgb_ycbcr(out, out_idx, plane0[idx], plane1[idx], plane2[idx]);
    }

    if (params.out_format == OUT_RGBA) {
        out[out_idx + 3] = uchar(params.alpha);
    }
}

kernel void jpeg_pack_444_rgba_texture(
    device const uchar *plane0 [[buffer(0)]],
    device const uchar *plane1 [[buffer(1)]],
    device const uchar *plane2 [[buffer(2)]],
    constant JpegTexturePackBatchParams &params [[buffer(3)]],
    texture2d<float, access::write> out [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint plane_len = params.width * params.height;
    const uint plane_base = params.tile_index * plane_len;
    const uint idx = plane_base + gid.y * params.width + gid.x;

    if (params.mode == MODE_GRAY) {
        const uchar gray = plane0[idx];
        out.write(rgba_float_direct(gray, gray, gray, params.alpha), gid);
    } else if (params.mode == MODE_RGB) {
        out.write(
            rgba_float_direct(plane0[idx], plane1[idx], plane2[idx], params.alpha),
            gid
        );
    } else {
        out.write(rgba_float_ycbcr(plane0[idx], plane1[idx], plane2[idx], params.alpha), gid);
    }
}
