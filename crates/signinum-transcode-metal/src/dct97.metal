// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

using namespace metal;

struct Dct97ProjectionParams {
    uint width;
    uint height;
    uint block_cols;
    uint band_width;
    uint band_height;
};

struct Dct97BatchProjectionParams {
    uint width;
    uint height;
    uint block_cols;
    uint blocks_per_item;
    uint band_width;
    uint band_height;
    uint output_stride;
};

struct Dct97IdctRowLiftParams {
    uint width;
    uint height;
    uint block_cols;
    uint blocks_per_item;
    uint low_width;
    uint high_width;
};

struct Dct97ColumnLiftParams {
    uint height;
    uint low_width;
    uint high_width;
    uint low_height;
    uint high_height;
    uint row_low_stride;
    uint row_high_stride;
    uint ll_stride;
    uint hl_stride;
    uint lh_stride;
    uint hh_stride;
};

struct Dct97QuantizeCodeblocksParams {
    uint band_width;
    uint band_height;
    uint output_stride;
    uint code_block_width;
    uint code_block_height;
    float inv_delta;
};

struct Reversible53ProjectionParams {
    uint width;
    uint height;
    uint block_cols;
    uint blocks_per_item;
    uint band_width;
    uint band_height;
    uint output_stride;
    uint vertical_low;
    uint horizontal_low;
};

struct Dct97SparseRow {
    uint offset;
    uint count;
};

struct Dct97WeightTap {
    uint sample_idx;
    float weight;
};

#define DCT97_STAGED_MAX_AXIS 1024
#define DCT97_ROWS_PER_GROUP 2
#define DCT97_COLUMNS_PER_GROUP 4
#define DCT97_THREADS_PER_GROUP 256

constant float DCT97_ALPHA = -1.586134342059924f;
constant float DCT97_BETA = -0.052980118572961f;
constant float DCT97_GAMMA = 0.882911075530934f;
constant float DCT97_DELTA = 0.443506852043971f;
constant float DCT97_KAPPA = 1.230174104914001f;
constant float DCT97_INV_KAPPA = 1.0f / DCT97_KAPPA;

static inline float dct97_idct_sample(
    device const float *blocks,
    device const float *idct_basis,
    constant Dct97IdctRowLiftParams &params,
    uint item_idx,
    uint x,
    uint y
) {
    const uint block_x = x / 8u;
    const uint block_y = y / 8u;
    const uint local_x = x % 8u;
    const uint local_y = y % 8u;
    const uint block_base =
        (item_idx * params.blocks_per_item + block_y * params.block_cols + block_x) * 64u;

    float sample = 0.0f;
    for (uint freq_y = 0; freq_y < 8u; ++freq_y) {
        const float y_basis = idct_basis[local_y * 8u + freq_y];
        for (uint freq_x = 0; freq_x < 8u; ++freq_x) {
            const float coefficient = blocks[block_base + freq_y * 8u + freq_x];
            sample += coefficient * y_basis * idct_basis[local_x * 8u + freq_x];
        }
    }
    return sample;
}

static inline void dct97_forward_lift_in_threadgroup(
    threadgroup float *data,
    uint n,
    uint thread_idx,
    uint threads_per_group
) {
    if (n < 2u) {
        return;
    }

    const uint last_even = ((n % 2u) == 0u) ? n - 2u : n - 1u;

    for (uint i = 1u + thread_idx * 2u; i < n; i += threads_per_group * 2u) {
        const float left = data[i - 1u];
        const float right = (i + 1u < n) ? data[i + 1u] : data[last_even];
        data[i] += DCT97_ALPHA * (left + right);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = thread_idx * 2u; i < n; i += threads_per_group * 2u) {
        const float left = (i > 0u) ? data[i - 1u] : data[1u];
        const float right = (i + 1u < n) ? data[i + 1u] : left;
        data[i] += DCT97_BETA * (left + right);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 1u + thread_idx * 2u; i < n; i += threads_per_group * 2u) {
        const float left = data[i - 1u];
        const float right = (i + 1u < n) ? data[i + 1u] : data[last_even];
        data[i] += DCT97_GAMMA * (left + right);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = thread_idx * 2u; i < n; i += threads_per_group * 2u) {
        const float left = (i > 0u) ? data[i - 1u] : data[1u];
        const float right = (i + 1u < n) ? data[i + 1u] : left;
        data[i] += DCT97_DELTA * (left + right);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void dct97_idct_row_lift_batch(
    device const float *blocks [[buffer(0)]],
    device const float *idct_basis [[buffer(1)]],
    device float *row_low [[buffer(2)]],
    device float *row_high [[buffer(3)]],
    constant Dct97IdctRowLiftParams &params [[buffer(4)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    threadgroup float rows[DCT97_ROWS_PER_GROUP][DCT97_STAGED_MAX_AXIS];

    const uint row_group = group_id.x;
    const uint item_idx = group_id.y;
    if (params.width > DCT97_STAGED_MAX_AXIS) {
        return;
    }

    for (uint row_offset = 0u; row_offset < DCT97_ROWS_PER_GROUP; ++row_offset) {
        const uint y = row_group * DCT97_ROWS_PER_GROUP + row_offset;
        if (y >= params.height) {
            continue;
        }
        for (uint x = thread_idx; x < params.width; x += DCT97_THREADS_PER_GROUP) {
            rows[row_offset][x] = dct97_idct_sample(blocks, idct_basis, params, item_idx, x, y);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row_offset = 0u; row_offset < DCT97_ROWS_PER_GROUP; ++row_offset) {
        const uint y = row_group * DCT97_ROWS_PER_GROUP + row_offset;
        if (y < params.height) {
            dct97_forward_lift_in_threadgroup(
                rows[row_offset], params.width, thread_idx, DCT97_THREADS_PER_GROUP);
        }
    }

    for (uint row_offset = 0u; row_offset < DCT97_ROWS_PER_GROUP; ++row_offset) {
        const uint y = row_group * DCT97_ROWS_PER_GROUP + row_offset;
        if (y >= params.height) {
            continue;
        }
        const uint low_base = item_idx * params.height * params.low_width + y * params.low_width;
        const uint high_base = item_idx * params.height * params.high_width + y * params.high_width;
        for (uint low_x = thread_idx; low_x < params.low_width; low_x += DCT97_THREADS_PER_GROUP) {
            row_low[low_base + low_x] = rows[row_offset][low_x * 2u] * DCT97_INV_KAPPA;
        }
        for (uint high_x = thread_idx; high_x < params.high_width; high_x += DCT97_THREADS_PER_GROUP) {
            row_high[high_base + high_x] = rows[row_offset][high_x * 2u + 1u] * DCT97_KAPPA;
        }
    }
}

kernel void dct97_column_lift_batch(
    device const float *row_low [[buffer(0)]],
    device const float *row_high [[buffer(1)]],
    device float *ll [[buffer(2)]],
    device float *hl [[buffer(3)]],
    device float *lh [[buffer(4)]],
    device float *hh [[buffer(5)]],
    constant Dct97ColumnLiftParams &params [[buffer(6)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    threadgroup float columns[DCT97_COLUMNS_PER_GROUP][DCT97_STAGED_MAX_AXIS];

    const uint column_group = group_id.x;
    const uint item_idx = group_id.y;
    const bool horizontal_low = group_id.z == 0u;
    const uint band_width = horizontal_low ? params.low_width : params.high_width;
    if (params.height > DCT97_STAGED_MAX_AXIS) {
        return;
    }

    for (uint column_offset = 0u; column_offset < DCT97_COLUMNS_PER_GROUP; ++column_offset) {
        const uint x = column_group * DCT97_COLUMNS_PER_GROUP + column_offset;
        if (x >= band_width) {
            continue;
        }
        const uint source_stride = horizontal_low ? params.row_low_stride : params.row_high_stride;
        device const float *source = horizontal_low ? row_low : row_high;
        const uint source_width = band_width;
        const uint source_base = item_idx * source_stride + x;
        for (uint y = thread_idx; y < params.height; y += DCT97_THREADS_PER_GROUP) {
            columns[column_offset][y] = source[source_base + y * source_width];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint column_offset = 0u; column_offset < DCT97_COLUMNS_PER_GROUP; ++column_offset) {
        const uint x = column_group * DCT97_COLUMNS_PER_GROUP + column_offset;
        if (x < band_width) {
            dct97_forward_lift_in_threadgroup(
                columns[column_offset], params.height, thread_idx, DCT97_THREADS_PER_GROUP);
        }
    }

    for (uint column_offset = 0u; column_offset < DCT97_COLUMNS_PER_GROUP; ++column_offset) {
        const uint x = column_group * DCT97_COLUMNS_PER_GROUP + column_offset;
        if (x >= band_width) {
            continue;
        }
        for (uint low_y = thread_idx; low_y < params.low_height; low_y += DCT97_THREADS_PER_GROUP) {
            const float value = columns[column_offset][low_y * 2u] * DCT97_INV_KAPPA;
            if (horizontal_low) {
                ll[item_idx * params.ll_stride + low_y * params.low_width + x] = value;
            } else {
                hl[item_idx * params.hl_stride + low_y * params.high_width + x] = value;
            }
        }
        for (uint high_y = thread_idx; high_y < params.high_height; high_y += DCT97_THREADS_PER_GROUP) {
            const float value = columns[column_offset][high_y * 2u + 1u] * DCT97_KAPPA;
            if (horizontal_low) {
                lh[item_idx * params.lh_stride + high_y * params.low_width + x] = value;
            } else {
                hh[item_idx * params.hh_stride + high_y * params.high_width + x] = value;
            }
        }
    }
}

kernel void dct97_quantize_codeblocks_batch(
    device const float *band [[buffer(0)]],
    device int *output [[buffer(1)]],
    constant Dct97QuantizeCodeblocksParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint x = gid.x;
    const uint y = gid.y;
    const uint item_idx = gid.z;
    if (x >= params.band_width || y >= params.band_height) {
        return;
    }

    const float value =
        band[item_idx * params.band_width * params.band_height + y * params.band_width + x];
    const int sign = value < 0.0f ? -1 : 1;
    const int magnitude = int(floor(fabs(value) * params.inv_delta));

    const uint cbx = x / params.code_block_width;
    const uint cby = y / params.code_block_height;
    const uint local_x = x - cbx * params.code_block_width;
    const uint local_y = y - cby * params.code_block_height;
    const uint block_x0 = cbx * params.code_block_width;
    const uint block_y0 = cby * params.code_block_height;
    const uint block_width = min(params.code_block_width, params.band_width - block_x0);
    const uint block_height = min(params.code_block_height, params.band_height - block_y0);
    const uint item_base = item_idx * params.output_stride;
    const uint codeblock_row_base = cby * params.code_block_height * params.band_width;
    const uint codeblock_base = codeblock_row_base + cbx * params.code_block_width * block_height;
    const uint block_offset = local_y * block_width + local_x;

    output[item_base + codeblock_base + block_offset] = sign * magnitude;
}

kernel void dct97_project_band(
    device const float *blocks [[buffer(0)]],
    device const Dct97SparseRow *x_rows [[buffer(1)]],
    device const Dct97WeightTap *x_taps [[buffer(2)]],
    device const Dct97SparseRow *y_rows [[buffer(3)]],
    device const Dct97WeightTap *y_taps [[buffer(4)]],
    device const float *idct_basis [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant Dct97ProjectionParams &params [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.band_width || gid.y >= params.band_height) {
        return;
    }

    const Dct97SparseRow x_row = x_rows[gid.x];
    const Dct97SparseRow y_row = y_rows[gid.y];
    float value = 0.0f;
    for (uint y_tap_idx = 0; y_tap_idx < y_row.count; ++y_tap_idx) {
        const Dct97WeightTap y_tap = y_taps[y_row.offset + y_tap_idx];
        const uint sample_y = y_tap.sample_idx;
        const float y_weight = y_tap.weight;
        const uint block_y = sample_y / 8u;
        const uint local_y = sample_y % 8u;

        for (uint x_tap_idx = 0; x_tap_idx < x_row.count; ++x_tap_idx) {
            const Dct97WeightTap x_tap = x_taps[x_row.offset + x_tap_idx];
            const uint sample_x = x_tap.sample_idx;
            const float x_weight = x_tap.weight;
            const uint block_x = sample_x / 8u;
            const uint local_x = sample_x % 8u;
            const uint block_base = (block_y * params.block_cols + block_x) * 64u;
            const float sample_weight = y_weight * x_weight;

            for (uint freq_y = 0; freq_y < 8u; ++freq_y) {
                const float y_basis = idct_basis[local_y * 8u + freq_y];
                for (uint freq_x = 0; freq_x < 8u; ++freq_x) {
                    const float coefficient = blocks[block_base + freq_y * 8u + freq_x];
                    const float x_basis = idct_basis[local_x * 8u + freq_x];
                    value += sample_weight * y_basis * x_basis * coefficient;
                }
            }
        }
    }

    output[gid.y * params.band_width + gid.x] = value;
}

kernel void dct97_project_band_batch(
    device const float *blocks [[buffer(0)]],
    device const Dct97SparseRow *x_rows [[buffer(1)]],
    device const Dct97WeightTap *x_taps [[buffer(2)]],
    device const Dct97SparseRow *y_rows [[buffer(3)]],
    device const Dct97WeightTap *y_taps [[buffer(4)]],
    device const float *idct_basis [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant Dct97BatchProjectionParams &params [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.band_width || gid.y >= params.band_height) {
        return;
    }

    const Dct97SparseRow x_row = x_rows[gid.x];
    const Dct97SparseRow y_row = y_rows[gid.y];
    const uint item_base = gid.z * params.blocks_per_item;
    float value = 0.0f;
    for (uint y_tap_idx = 0; y_tap_idx < y_row.count; ++y_tap_idx) {
        const Dct97WeightTap y_tap = y_taps[y_row.offset + y_tap_idx];
        const uint sample_y = y_tap.sample_idx;
        const float y_weight = y_tap.weight;
        const uint block_y = sample_y / 8u;
        const uint local_y = sample_y % 8u;

        for (uint x_tap_idx = 0; x_tap_idx < x_row.count; ++x_tap_idx) {
            const Dct97WeightTap x_tap = x_taps[x_row.offset + x_tap_idx];
            const uint sample_x = x_tap.sample_idx;
            const float x_weight = x_tap.weight;
            const uint block_x = sample_x / 8u;
            const uint local_x = sample_x % 8u;
            const uint block_base = (item_base + block_y * params.block_cols + block_x) * 64u;
            const float sample_weight = y_weight * x_weight;

            for (uint freq_y = 0; freq_y < 8u; ++freq_y) {
                const float y_basis = idct_basis[local_y * 8u + freq_y];
                for (uint freq_x = 0; freq_x < 8u; ++freq_x) {
                    const float coefficient = blocks[block_base + freq_y * 8u + freq_x];
                    const float x_basis = idct_basis[local_x * 8u + freq_x];
                    value += sample_weight * y_basis * x_basis * coefficient;
                }
            }
        }
    }

    output[gid.z * params.output_stride + gid.y * params.band_width + gid.x] = value;
}

static inline int floor_div_i32(int numerator, int denominator) {
    const int quotient = numerator / denominator;
    const int remainder = numerator % denominator;
    return (remainder < 0) ? quotient - 1 : quotient;
}

static inline int reversible53_sample(
    device const int *blocks,
    uint block_cols,
    uint blocks_per_item,
    uint item_idx,
    uint x,
    uint y
) {
    const uint block_x = x / 8u;
    const uint block_y = y / 8u;
    const uint local_x = x % 8u;
    const uint local_y = y % 8u;
    const uint item_block_base = item_idx * blocks_per_item;
    const uint block_base = (item_block_base + block_y * block_cols + block_x) * 64u;
    return blocks[block_base + local_y * 8u + local_x];
}

static inline int reversible53_vertical_high(
    device const int *blocks,
    constant Reversible53ProjectionParams &params,
    uint item_idx,
    uint x,
    uint high_idx
) {
    const uint odd_idx = high_idx * 2u + 1u;
    const int current = reversible53_sample(
        blocks, params.block_cols, params.blocks_per_item, item_idx, x, odd_idx);
    const int left = reversible53_sample(
        blocks, params.block_cols, params.blocks_per_item, item_idx, x, odd_idx - 1u);
    if ((params.height % 2u) == 0u && odd_idx + 1u == params.height) {
        return current - left;
    }

    const uint right_idx = (odd_idx + 1u < params.height) ? odd_idx + 1u : params.height - 1u;
    const int right = reversible53_sample(
        blocks, params.block_cols, params.blocks_per_item, item_idx, x, right_idx);
    return current - floor_div_i32(left + right, 2);
}

static inline int reversible53_vertical_low(
    device const int *blocks,
    constant Reversible53ProjectionParams &params,
    uint item_idx,
    uint x,
    uint low_idx
) {
    const uint even_idx = low_idx * 2u;
    const int current = reversible53_sample(
        blocks, params.block_cols, params.blocks_per_item, item_idx, x, even_idx);
    if (params.height < 2u) {
        return current;
    }

    if ((params.height % 2u) == 0u) {
        const int right = reversible53_vertical_high(blocks, params, item_idx, x, low_idx);
        if (low_idx == 0u) {
            return current + floor_div_i32(right + 1, 2);
        }
        const int left = reversible53_vertical_high(blocks, params, item_idx, x, low_idx - 1u);
        return current + floor_div_i32(left + right + 2, 4);
    }

    const uint high_len = params.height / 2u;
    if (high_len == 0u) {
        return current;
    }
    const int left = low_idx > 0u
        ? reversible53_vertical_high(blocks, params, item_idx, x, low_idx - 1u)
        : reversible53_vertical_high(blocks, params, item_idx, x, 0u);
    const int right = low_idx < high_len
        ? reversible53_vertical_high(blocks, params, item_idx, x, low_idx)
        : left;
    return current + floor_div_i32(left + right + 2, 4);
}

static inline int reversible53_vertical_value(
    device const int *blocks,
    constant Reversible53ProjectionParams &params,
    uint item_idx,
    uint x,
    uint output_y
) {
    return params.vertical_low != 0u
        ? reversible53_vertical_low(blocks, params, item_idx, x, output_y)
        : reversible53_vertical_high(blocks, params, item_idx, x, output_y);
}

static inline int reversible53_horizontal_high(
    device const int *blocks,
    constant Reversible53ProjectionParams &params,
    uint item_idx,
    uint high_idx,
    uint output_y
) {
    const uint odd_idx = high_idx * 2u + 1u;
    const int current = reversible53_vertical_value(blocks, params, item_idx, odd_idx, output_y);
    const int left = reversible53_vertical_value(blocks, params, item_idx, odd_idx - 1u, output_y);
    if ((params.width % 2u) == 0u && odd_idx + 1u == params.width) {
        return current - left;
    }

    const uint right_idx = (odd_idx + 1u < params.width) ? odd_idx + 1u : params.width - 1u;
    const int right = reversible53_vertical_value(blocks, params, item_idx, right_idx, output_y);
    return current - floor_div_i32(left + right, 2);
}

static inline int reversible53_horizontal_low(
    device const int *blocks,
    constant Reversible53ProjectionParams &params,
    uint item_idx,
    uint low_idx,
    uint output_y
) {
    const uint even_idx = low_idx * 2u;
    const int current = reversible53_vertical_value(blocks, params, item_idx, even_idx, output_y);
    if (params.width < 2u) {
        return current;
    }

    if ((params.width % 2u) == 0u) {
        const int right = reversible53_horizontal_high(
            blocks, params, item_idx, low_idx, output_y);
        if (low_idx == 0u) {
            return current + floor_div_i32(right + 1, 2);
        }
        const int left = reversible53_horizontal_high(
            blocks, params, item_idx, low_idx - 1u, output_y);
        return current + floor_div_i32(left + right + 2, 4);
    }

    const uint high_len = params.width / 2u;
    if (high_len == 0u) {
        return current;
    }
    const int left = low_idx > 0u
        ? reversible53_horizontal_high(blocks, params, item_idx, low_idx - 1u, output_y)
        : reversible53_horizontal_high(blocks, params, item_idx, 0u, output_y);
    const int right = low_idx < high_len
        ? reversible53_horizontal_high(blocks, params, item_idx, low_idx, output_y)
        : left;
    return current + floor_div_i32(left + right + 2, 4);
}

kernel void reversible53_project_band(
    device const int *blocks [[buffer(0)]],
    device int *output [[buffer(1)]],
    constant Reversible53ProjectionParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.band_width || gid.y >= params.band_height) {
        return;
    }

    const int value = params.horizontal_low != 0u
        ? reversible53_horizontal_low(blocks, params, gid.z, gid.x, gid.y)
        : reversible53_horizontal_high(blocks, params, gid.z, gid.x, gid.y);
    output[gid.z * params.output_stride + gid.y * params.band_width + gid.x] = value;
}
