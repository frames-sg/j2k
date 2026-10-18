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
        const short sample = j2k_pack_native_i16(value, max_value);
        return ushort(uchar(char(sample))) * 257u;
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
