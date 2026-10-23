// SPDX-License-Identifier: MIT OR Apache-2.0

// The last inverse wavelet level of a grayscale plane fused with its final
// store: each synthesized sample inside the store window is shifted, rounded,
// and packed exactly as the `j2k_store_component_gray_*` kernels do, and
// written straight to the output instead of to a float plane the store reads
// back.

struct J2kStoreGrayU8 {
    static uchar pack(float value, float max_value, float u8_scale, float) {
        return scale_to_u8(value, max_value, u8_scale);
    }
};

struct J2kStoreGrayU16 {
    static ushort pack(float value, float max_value, float, float u16_scale) {
        return pack_to_u16(value, max_value, u16_scale);
    }
};

struct J2kStoreGrayI16 {
    static short pack(float value, float max_value, float, float) {
        return j2k_pack_native_i16(value, max_value);
    }
};

// Signed sources of at most eight bits written to Gray16, as
// `j2k_store_component_gray_i8_widened_u16` does.
struct J2kStoreGrayI8WidenedU16 {
    static ushort pack(float value, float max_value, float, float) {
        return j2k_pack_signed_narrow_u16(value, max_value);
    }
};

template<typename Store, typename Sample>
struct J2kIdwtGrayStoreWriter {
    device Sample *out;
    J2kGrayStoreParams params;

    void write(uint x, uint y, float value) const {
        // Coordinates before the window wrap and fail the bound checks.
        const uint gx = x - params.source_x;
        const uint gy = y - params.source_y;
        if (gx >= params.copy_width || gy >= params.copy_height) {
            return;
        }
        out[params.output_item_offset + (params.output_y + gy) * params.output_stride +
            params.output_x + gx] =
            Store::pack(
                j2k_shift_centered_sample(value, params.addend, true),
                params.max_value,
                params.u8_scale,
                params.u16_scale
            );
    }
};

#define J2K_IDWT_GRAY_STORE_KERNELS(STORE, SAMPLE, SUFFIX)                                   \
kernel void j2k_idwt_reversible53_fused_2d_store_##SUFFIX(                                  \
    device const float *ll [[buffer(0)]],                                                     \
    device const float *hl [[buffer(1)]],                                                     \
    device const float *lh [[buffer(2)]],                                                     \
    device const float *hh [[buffer(3)]],                                                     \
    device SAMPLE *out [[buffer(4)]],                                                         \
    constant J2kIdwtSingleDecompositionParams &params [[buffer(5)]],                          \
    constant J2kGrayStoreParams &store [[buffer(6)]],                                         \
    uint2 group [[threadgroup_position_in_grid]],                                             \
    uint2 local [[thread_position_in_threadgroup]]                                            \
) {                                                                                           \
    threadgroup float tile[(J2K_IDWT53_2D_TILE_H + 2u * J2K_IDWT53_HALO) * J2K_IDWT53_2D_STRIDE]; \
    const J2kGrayStoreParams store_params = store;                                            \
    idwt53_fused_2d_core(                                                                     \
        ll, hl, lh, hh, J2kIdwtGrayStoreWriter<STORE, SAMPLE>{out, store_params}, params,     \
        group, local, tile                                                                    \
    );                                                                                        \
}                                                                                             \
kernel void j2k_idwt_irreversible97_fused_2d_store_##SUFFIX(                                \
    device const float *ll [[buffer(0)]],                                                     \
    device const float *hl [[buffer(1)]],                                                     \
    device const float *lh [[buffer(2)]],                                                     \
    device const float *hh [[buffer(3)]],                                                     \
    device SAMPLE *out [[buffer(4)]],                                                         \
    constant J2kIdwtSingleDecompositionParams &params [[buffer(5)]],                          \
    constant J2kIdwt97LiftSteps &horizontal [[buffer(6)]],                                    \
    constant J2kIdwt97LiftSteps &vertical [[buffer(7)]],                                      \
    constant J2kGrayStoreParams &store [[buffer(8)]],                                         \
    uint2 group [[threadgroup_position_in_grid]],                                             \
    uint2 local [[thread_position_in_threadgroup]]                                            \
) {                                                                                           \
    threadgroup float tile[(J2K_IDWT97_2D_TILE_H + 2u * J2K_IDWT97_HALO) * J2K_IDWT97_2D_STRIDE]; \
    const J2kGrayStoreParams store_params = store;                                            \
    idwt97_fused_2d_core(                                                                     \
        ll, hl, lh, hh, J2kIdwtGrayStoreWriter<STORE, SAMPLE>{out, store_params}, params,     \
        horizontal, vertical, group, local, tile                                              \
    );                                                                                        \
}

J2K_IDWT_GRAY_STORE_KERNELS(J2kStoreGrayU8, uchar, u8)
J2K_IDWT_GRAY_STORE_KERNELS(J2kStoreGrayU16, ushort, u16)
J2K_IDWT_GRAY_STORE_KERNELS(J2kStoreGrayI16, short, i16)
J2K_IDWT_GRAY_STORE_KERNELS(J2kStoreGrayI8WidenedU16, ushort, i8_widened_u16)

// A stacked batch's last level written straight into its dense grayscale
// group: the component store's shift, then the group pack's shift and pack,
// as `j2k_store_component_repeated` and `j2k_store_component_repeated_gray_*`
// apply them one after the other.
struct J2kStackedGrayStoreParams {
    uint source_x;
    uint source_y;
    uint copy_width;
    uint copy_height;
    uint output_stride;
    uint image_stride;
    float addend;
    uint round_centered;
    float max_value;
    float u8_scale;
    float u16_scale;
    uint reserved;
};

template<typename Store, typename Sample>
struct J2kIdwtStackedGrayStoreWriter {
    device Sample *out;
    J2kStackedGrayStoreParams params;

    void write(uint x, uint y, float value) const {
        const uint gx = x - params.source_x;
        const uint gy = y - params.source_y;
        if (gx >= params.copy_width || gy >= params.copy_height) {
            return;
        }
        const float stored =
            j2k_shift_centered_sample(value, params.addend, params.round_centered != 0u);
        out[gy * params.output_stride + gx] = Store::pack(
            j2k_shift_centered_sample(stored, 0.0f, true),
            params.max_value,
            params.u8_scale,
            params.u16_scale
        );
    }
};

#define J2K_IDWT_STACKED_GRAY_STORE_KERNELS(STORE, SAMPLE, SUFFIX)                           \
kernel void j2k_idwt_reversible53_fused_2d_batched_gray_store_##SUFFIX(                      \
    device const float *ll [[buffer(0)]],                                                     \
    device const float *hl [[buffer(1)]],                                                     \
    device const float *lh [[buffer(2)]],                                                     \
    device const float *hh [[buffer(3)]],                                                     \
    device SAMPLE *out [[buffer(4)]],                                                         \
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(5)]],                  \
    constant J2kStackedGrayStoreParams &store [[buffer(6)]],                                  \
    uint3 group [[threadgroup_position_in_grid]],                                             \
    uint3 local [[thread_position_in_threadgroup]]                                            \
) {                                                                                           \
    threadgroup float tile[(J2K_IDWT53_2D_TILE_H + 2u * J2K_IDWT53_HALO) * J2K_IDWT53_2D_STRIDE]; \
    if (group.z >= params.batch_count) {                                                      \
        return;                                                                               \
    }                                                                                         \
    const J2kStackedGrayStoreParams store_params = store;                                     \
    idwt53_fused_2d_core(                                                                     \
        ll + group.z * params.ll_instance_stride,                                             \
        hl + group.z * params.hl_instance_stride,                                             \
        lh + group.z * params.lh_instance_stride,                                             \
        hh + group.z * params.hh_instance_stride,                                             \
        J2kIdwtStackedGrayStoreWriter<STORE, SAMPLE>{                                         \
            out + ulong(group.z) * store_params.image_stride, store_params},                  \
        params, group.xy, local.xy, tile                                                      \
    );                                                                                        \
}                                                                                             \
kernel void j2k_idwt_irreversible97_fused_2d_batched_gray_store_##SUFFIX(                    \
    device const float *ll [[buffer(0)]],                                                     \
    device const float *hl [[buffer(1)]],                                                     \
    device const float *lh [[buffer(2)]],                                                     \
    device const float *hh [[buffer(3)]],                                                     \
    device SAMPLE *out [[buffer(4)]],                                                         \
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(5)]],                  \
    constant J2kIdwt97LiftSteps &horizontal [[buffer(6)]],                                    \
    constant J2kIdwt97LiftSteps &vertical [[buffer(7)]],                                      \
    constant J2kStackedGrayStoreParams &store [[buffer(8)]],                                  \
    uint3 group [[threadgroup_position_in_grid]],                                             \
    uint3 local [[thread_position_in_threadgroup]]                                            \
) {                                                                                           \
    threadgroup float tile[(J2K_IDWT97_2D_TILE_H + 2u * J2K_IDWT97_HALO) * J2K_IDWT97_2D_STRIDE]; \
    if (group.z >= params.batch_count) {                                                      \
        return;                                                                               \
    }                                                                                         \
    const J2kStackedGrayStoreParams store_params = store;                                     \
    idwt97_fused_2d_core(                                                                     \
        ll + group.z * params.ll_instance_stride,                                             \
        hl + group.z * params.hl_instance_stride,                                             \
        lh + group.z * params.lh_instance_stride,                                             \
        hh + group.z * params.hh_instance_stride,                                             \
        J2kIdwtStackedGrayStoreWriter<STORE, SAMPLE>{                                         \
            out + ulong(group.z) * store_params.image_stride, store_params},                  \
        params, horizontal, vertical, group.xy, local.xy, tile                                \
    );                                                                                        \
}

J2K_IDWT_STACKED_GRAY_STORE_KERNELS(J2kStoreGrayU8, uchar, u8)
J2K_IDWT_STACKED_GRAY_STORE_KERNELS(J2kStoreGrayU16, ushort, u16)
J2K_IDWT_STACKED_GRAY_STORE_KERNELS(J2kStoreGrayI16, short, i16)

// The last 5/3 or 9/7 level of three colour components fused with
// `j2k_pack_mct_rgb8`: each component's tile is synthesized in turn into the
// shared tile, the first two are kept in registers, and every pixel is packed
// from its three samples, so the float planes are never written or read back.
// Thread `local` owns rows `ty0 + local.y + 8k` and columns `tx0 + 2 local.x + j`.
template<uint ROWS>
inline void idwt_keep_tile(
    thread float (&kept)[ROWS][2],
    threadgroup const float *tile,
    uint stride,
    J2kIdwtTile t,
    uint2 local
) {
    for (uint k = 0u; k < ROWS; ++k) {
        const uint y = t.ty0 + local.y + k * J2K_IDWT53_2D_ROWS;
        for (uint j = 0u; j < 2u; ++j) {
            const uint x = t.tx0 + 2u * local.x + j;
            kept[k][j] = (y < t.ty1 && x < t.tx1) ? tile[(y - t.ly0) * stride + x - t.lx0] : 0.0f;
        }
    }
}

template<uint ROWS>
inline void idwt_pack_tile_mct_rgb8(
    thread const float (&first)[ROWS][2],
    thread const float (&second)[ROWS][2],
    threadgroup const float *tile,
    uint stride,
    J2kIdwtTile t,
    uint2 local,
    constant J2kMctRgb8PackParams &pack,
    device uchar *out
) {
    for (uint k = 0u; k < ROWS; ++k) {
        const uint y = t.ty0 + local.y + k * J2K_IDWT53_2D_ROWS;
        for (uint j = 0u; j < 2u; ++j) {
            const uint x = t.tx0 + 2u * local.x + j;
            if (y < t.ty1 && x < t.tx1) {
                j2k_store_mct_rgb8_pixel(
                    first[k][j], second[k][j], tile[(y - t.ly0) * stride + x - t.lx0], pack, out,
                    y * pack.out_stride + x * 3u
                );
            }
        }
    }
}

#define J2K_IDWT_MCT_RGB8_BANDS                                                                \
    device const float *ll0 [[buffer(0)]], device const float *hl0 [[buffer(1)]],               \
    device const float *lh0 [[buffer(2)]], device const float *hh0 [[buffer(3)]],               \
    device const float *ll1 [[buffer(4)]], device const float *hl1 [[buffer(5)]],               \
    device const float *lh1 [[buffer(6)]], device const float *hh1 [[buffer(7)]],               \
    device const float *ll2 [[buffer(8)]], device const float *hl2 [[buffer(9)]],               \
    device const float *lh2 [[buffer(10)]], device const float *hh2 [[buffer(11)]],             \
    device uchar *out [[buffer(12)]],                                                          \
    constant J2kMctRgb8PackParams &pack [[buffer(13)]],                                        \
    constant J2kIdwtSingleDecompositionParams &params0 [[buffer(14)]],                         \
    constant J2kIdwtSingleDecompositionParams &params1 [[buffer(15)]],                         \
    constant J2kIdwtSingleDecompositionParams &params2 [[buffer(16)]]

kernel void j2k_idwt_reversible53_fused_2d_mct_rgb8(
    J2K_IDWT_MCT_RGB8_BANDS,
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]
) {
    threadgroup float tile[(J2K_IDWT53_2D_TILE_H + 2u * J2K_IDWT53_HALO) * J2K_IDWT53_2D_STRIDE];
    float first[8][2];
    float second[8][2];
    const J2kIdwtTile t = idwt53_fused_2d_tile(ll0, hl0, lh0, hh0, params0, group, local, tile);
    if (!t.valid) {
        return;
    }
    idwt_keep_tile<8>(first, tile, J2K_IDWT53_2D_STRIDE, t, local);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    idwt53_fused_2d_tile(ll1, hl1, lh1, hh1, params1, group, local, tile);
    idwt_keep_tile<8>(second, tile, J2K_IDWT53_2D_STRIDE, t, local);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    idwt53_fused_2d_tile(ll2, hl2, lh2, hh2, params2, group, local, tile);
    idwt_pack_tile_mct_rgb8<8>(first, second, tile, J2K_IDWT53_2D_STRIDE, t, local, pack, out);
}

kernel void j2k_idwt_irreversible97_fused_2d_mct_rgb8(
    J2K_IDWT_MCT_RGB8_BANDS,
    constant J2kIdwt97LiftSteps &horizontal0 [[buffer(17)]],
    constant J2kIdwt97LiftSteps &vertical0 [[buffer(18)]],
    constant J2kIdwt97LiftSteps &horizontal1 [[buffer(19)]],
    constant J2kIdwt97LiftSteps &vertical1 [[buffer(20)]],
    constant J2kIdwt97LiftSteps &horizontal2 [[buffer(21)]],
    constant J2kIdwt97LiftSteps &vertical2 [[buffer(22)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]
) {
    threadgroup float tile[(J2K_IDWT97_2D_TILE_H + 2u * J2K_IDWT97_HALO) * J2K_IDWT97_2D_STRIDE];
    float first[7][2];
    float second[7][2];
    const J2kIdwtTile t = idwt97_fused_2d_tile(
        ll0, hl0, lh0, hh0, params0, horizontal0, vertical0, group, local, tile
    );
    if (!t.valid) {
        return;
    }
    idwt_keep_tile<7>(first, tile, J2K_IDWT97_2D_STRIDE, t, local);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    idwt97_fused_2d_tile(ll1, hl1, lh1, hh1, params1, horizontal1, vertical1, group, local, tile);
    idwt_keep_tile<7>(second, tile, J2K_IDWT97_2D_STRIDE, t, local);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    idwt97_fused_2d_tile(ll2, hl2, lh2, hh2, params2, horizontal2, vertical2, group, local, tile);
    idwt_pack_tile_mct_rgb8<7>(first, second, tile, J2K_IDWT97_2D_STRIDE, t, local, pack, out);
}
