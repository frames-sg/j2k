// SPDX-License-Identifier: Apache-2.0
//
// CUDA kernels for signinum-transcode-cuda: direct DCT-coefficient-domain
// JPEG -> HTJ2K transform stages. These port the signinum-transcode SCALAR
// ORACLE faithfully (they are the parity reference), not a re-derivation:
//
//   reversible 5/3:  idct_islow (jidctint, 16-bit fixed point) -> -128 level
//                    shift -> separable reversible integer 5/3 lifting with
//                    Euclidean floor division, deinterleaved into LL/HL/LH/HH.
//
// Compiled with --fmad=false (see build.rs) so the irreversible 9/7 path (added
// later) keeps bit-identical f32 operation ordering vs the scalar reference.
//
// Integer arithmetic mirrors the scalar oracle's `i32` (Wrapping) math; valid
// JPEG coefficients do not overflow i32 in the islow algorithm by construction.

typedef short i16;
typedef int i32;
typedef unsigned int u32;

// ---- jidctint fixed-point constants (match idct/scalar.rs) -----------------
#define CONST_BITS 13
#define PASS1_BITS 2
#define FIX_0_298631336 2446
#define FIX_0_390180644 3196
#define FIX_0_541196100 4433
#define FIX_0_765366865 6270
#define FIX_0_899976223 7373
#define FIX_1_175875602 9633
#define FIX_1_501321110 12299
#define FIX_1_847759065 15137
#define FIX_1_961570560 16069
#define FIX_2_053119869 16819
#define FIX_2_562915447 20995
#define FIX_3_072711026 25172

// Euclidean floor division by a positive divisor (matches i32::div_euclid).
__device__ __forceinline__ i32 floor_div_pos(i32 a, i32 d) {
    i32 q = a / d;
    i32 r = a - q * d;
    if (r < 0) {
        q -= 1;
    }
    return q;
}

// One full 8x8 ISLOW inverse DCT (full path; the scalar fast paths are pure
// optimizations and are mathematically identical, so the full path is
// bit-exact for every input). Output: signed samples already level-shifted by
// -128 (i.e. clamp(idct+128,0,255) - 128), matching idct_blocks_to_signed_samples.
__device__ void idct_islow_signed(const i16* in, i32* out64) {
    i32 work[64];
    // Column pass.
    for (int col = 0; col < 8; ++col) {
        i32 p0 = (i32)in[col];
        i32 p1 = (i32)in[col + 8];
        i32 p2 = (i32)in[col + 16];
        i32 p3 = (i32)in[col + 24];
        i32 p4 = (i32)in[col + 32];
        i32 p5 = (i32)in[col + 40];
        i32 p6 = (i32)in[col + 48];
        i32 p7 = (i32)in[col + 56];

        i32 z2 = p2;
        i32 z3 = p6;
        i32 z1 = (z2 + z3) * FIX_0_541196100;
        i32 tmp2 = z1 + z3 * (-FIX_1_847759065);
        i32 tmp3 = z1 + z2 * FIX_0_765366865;

        z2 = p0;
        z3 = p4;
        i32 tmp0 = (z2 + z3) << CONST_BITS;
        i32 tmp1 = (z2 - z3) << CONST_BITS;

        i32 tmp10 = tmp0 + tmp3;
        i32 tmp13 = tmp0 - tmp3;
        i32 tmp11 = tmp1 + tmp2;
        i32 tmp12 = tmp1 - tmp2;

        tmp0 = p7;
        tmp1 = p5;
        tmp2 = p3;
        tmp3 = p1;

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        i32 z4 = tmp1 + tmp3;
        i32 z5 = (z3 + z4) * FIX_1_175875602;

        tmp0 = tmp0 * FIX_0_298631336;
        tmp1 = tmp1 * FIX_2_053119869;
        tmp2 = tmp2 * FIX_3_072711026;
        tmp3 = tmp3 * FIX_1_501321110;
        z1 = z1 * (-FIX_0_899976223);
        z2 = z2 * (-FIX_2_562915447);
        z3 = z3 * (-FIX_1_961570560);
        z4 = z4 * (-FIX_0_390180644);

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        const int shift = CONST_BITS - PASS1_BITS;
        const i32 rounding = (i32)1 << (shift - 1);
        work[col]      = (tmp10 + tmp3 + rounding) >> shift;
        work[col + 56] = (tmp10 - tmp3 + rounding) >> shift;
        work[col + 8]  = (tmp11 + tmp2 + rounding) >> shift;
        work[col + 48] = (tmp11 - tmp2 + rounding) >> shift;
        work[col + 16] = (tmp12 + tmp1 + rounding) >> shift;
        work[col + 40] = (tmp12 - tmp1 + rounding) >> shift;
        work[col + 24] = (tmp13 + tmp0 + rounding) >> shift;
        work[col + 32] = (tmp13 - tmp0 + rounding) >> shift;
    }
    // Row pass.
    for (int row = 0; row < 8; ++row) {
        const int base = row * 8;
        i32 p0 = work[base];
        i32 p1 = work[base + 1];
        i32 p2 = work[base + 2];
        i32 p3 = work[base + 3];
        i32 p4 = work[base + 4];
        i32 p5 = work[base + 5];
        i32 p6 = work[base + 6];
        i32 p7 = work[base + 7];

        const int shift = CONST_BITS + PASS1_BITS + 3;
        const i32 rounding = (i32)1 << (shift - 1);

        i32 z2 = p2;
        i32 z3 = p6;
        i32 z1 = (z2 + z3) * FIX_0_541196100;
        i32 tmp2 = z1 + z3 * (-FIX_1_847759065);
        i32 tmp3 = z1 + z2 * FIX_0_765366865;

        i32 tmp0 = (p0 + p4) << CONST_BITS;
        i32 tmp1 = (p0 - p4) << CONST_BITS;

        i32 tmp10 = tmp0 + tmp3;
        i32 tmp13 = tmp0 - tmp3;
        i32 tmp11 = tmp1 + tmp2;
        i32 tmp12 = tmp1 - tmp2;

        tmp0 = p7;
        tmp1 = p5;
        tmp2 = p3;
        tmp3 = p1;

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        i32 z4 = tmp1 + tmp3;
        i32 z5 = (z3 + z4) * FIX_1_175875602;

        tmp0 = tmp0 * FIX_0_298631336;
        tmp1 = tmp1 * FIX_2_053119869;
        tmp2 = tmp2 * FIX_3_072711026;
        tmp3 = tmp3 * FIX_1_501321110;
        z1 = z1 * (-FIX_0_899976223);
        z2 = z2 * (-FIX_2_562915447);
        z3 = z3 * (-FIX_1_961570560);
        z4 = z4 * (-FIX_0_390180644);

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        i32 r0 = (tmp10 + tmp3 + rounding) >> shift;
        i32 r7 = (tmp10 - tmp3 + rounding) >> shift;
        i32 r1 = (tmp11 + tmp2 + rounding) >> shift;
        i32 r6 = (tmp11 - tmp2 + rounding) >> shift;
        i32 r2 = (tmp12 + tmp1 + rounding) >> shift;
        i32 r5 = (tmp12 - tmp1 + rounding) >> shift;
        i32 r3 = (tmp13 + tmp0 + rounding) >> shift;
        i32 r4 = (tmp13 - tmp0 + rounding) >> shift;

        // clamp(value+128, 0, 255) then subtract 128 -> signed sample in [-128,127].
        i32 vals[8] = {r0, r1, r2, r3, r4, r5, r6, r7};
        for (int k = 0; k < 8; ++k) {
            i32 lv = vals[k] + 128;
            if (lv < 0) lv = 0; else if (lv > 255) lv = 255;
            out64[base + k] = lv - 128;
        }
    }
}

// Kernel 1: one thread per 8x8 DCT block -> block-local signed sample plane.
extern "C" __global__ void transcode_reversible53_idct(
    const i16* blocks,  // block_count * 64, natural order
    i32* samples,       // block_count * 64, block-local signed samples
    u32 block_count) {
    const u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= block_count) {
        return;
    }
    idct_islow_signed(blocks + (size_t)idx * 64, samples + (size_t)idx * 64);
}

// Block-local sample fetch matching component_sample_i32 (x<width, y<height).
__device__ __forceinline__ i32 sample_at(
    const i32* samples, int block_cols, int x, int y) {
    const int block_idx = (y >> 3) * block_cols + (x >> 3);
    const int local_idx = (y & 7) * 8 + (x & 7);
    return samples[(size_t)block_idx * 64 + local_idx];
}

// vertical_high_53_i32_at
__device__ i32 vertical_high(
    const i32* samples, int block_cols, int height, int x, int high_idx) {
    const int odd_idx = high_idx * 2 + 1;
    const i32 current = sample_at(samples, block_cols, x, odd_idx);
    const i32 left = sample_at(samples, block_cols, x, odd_idx - 1);
    if ((height % 2 == 0) && (odd_idx + 1 == height)) {
        return current - left;
    }
    const int right_idx = (odd_idx + 1 < height) ? (odd_idx + 1) : (height - 1);
    const i32 right = sample_at(samples, block_cols, x, right_idx);
    return current - floor_div_pos(left + right, 2);
}

// vertical_low_53_i32_at
__device__ i32 vertical_low(
    const i32* samples, int block_cols, int height, int x, int low_idx) {
    const int even_idx = low_idx * 2;
    const i32 current = sample_at(samples, block_cols, x, even_idx);
    if (height < 2) {
        return current;
    }
    if (height % 2 == 0) {
        const i32 right = vertical_high(samples, block_cols, height, x, low_idx);
        if (low_idx == 0) {
            return current + floor_div_pos(right + 1, 2);
        }
        const i32 left = vertical_high(samples, block_cols, height, x, low_idx - 1);
        return current + floor_div_pos(left + right + 2, 4);
    }
    const int high_len = height / 2;
    if (high_len == 0) {
        return current;
    }
    const i32 left = vertical_high(samples, block_cols, height, x, (low_idx > 0) ? (low_idx - 1) : 0);
    const i32 right = (low_idx < high_len)
        ? vertical_high(samples, block_cols, height, x, low_idx)
        : left;
    return current + floor_div_pos(left + right + 2, 4);
}

// Kernel 2: vertical 5/3 low band. One thread per (x, low row).
extern "C" __global__ void transcode_reversible53_vertical_low(
    const i32* samples, int block_cols, int width, int height,
    i32* v_low, int low_height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int yl = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || yl >= low_height) {
        return;
    }
    v_low[(size_t)yl * width + x] = vertical_low(samples, block_cols, height, x, yl);
}

// Kernel 3: vertical 5/3 high band. One thread per (x, high row).
extern "C" __global__ void transcode_reversible53_vertical_high(
    const i32* samples, int block_cols, int width, int height,
    i32* v_high, int high_height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int yh = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || yh >= high_height) {
        return;
    }
    v_high[(size_t)yh * width + x] = vertical_high(samples, block_cols, height, x, yh);
}

// reversible_lift_53_i32 applied in place to a contiguous row of `n` i32.
__device__ void reversible_lift_row(i32* v, int n) {
    if (n < 2) {
        return;
    }
    if (n % 2 == 0) {
        for (int i = 1; i < n - 1; i += 2) {
            v[i] -= floor_div_pos(v[i - 1] + v[i + 1], 2);
        }
        v[n - 1] -= v[n - 2];
        v[0] += floor_div_pos(v[1] + 1, 2);
        for (int i = 2; i < n; i += 2) {
            v[i] += floor_div_pos(v[i - 1] + v[i + 1] + 2, 4);
        }
        return;
    }
    const int last_even = n - 1;
    for (int i = 1; i < n; i += 2) {
        const i32 right = (i + 1 < n) ? v[i + 1] : v[last_even];
        v[i] -= floor_div_pos(v[i - 1] + right, 2);
    }
    for (int i = 0; i < n; i += 2) {
        const i32 left = (i > 0) ? v[i - 1] : v[1];
        const i32 right = (i + 1 < n) ? v[i + 1] : left;
        v[i] += floor_div_pos(left + right + 2, 4);
    }
}

// Kernel 4: horizontal 5/3 lift of each vertically-low row, in place, then
// deinterleave even -> LL, odd -> HL. One thread per low row.
extern "C" __global__ void transcode_reversible53_horizontal_low(
    i32* v_low, int width, int low_height, int low_width, int high_width,
    i32* ll, i32* hl) {
    const int yl = blockIdx.x * blockDim.x + threadIdx.x;
    if (yl >= low_height) {
        return;
    }
    i32* row = v_low + (size_t)yl * width;
    reversible_lift_row(row, width);
    for (int i = 0; i < low_width; ++i) {
        ll[(size_t)yl * low_width + i] = row[i * 2];
    }
    for (int i = 0; i < high_width; ++i) {
        hl[(size_t)yl * high_width + i] = row[i * 2 + 1];
    }
}

// Kernel 5: horizontal 5/3 lift of each vertically-high row -> LH/HH.
extern "C" __global__ void transcode_reversible53_horizontal_high(
    i32* v_high, int width, int high_height, int low_width, int high_width,
    i32* lh, i32* hh) {
    const int yh = blockIdx.x * blockDim.x + threadIdx.x;
    if (yh >= high_height) {
        return;
    }
    i32* row = v_high + (size_t)yh * width;
    reversible_lift_row(row, width);
    for (int i = 0; i < low_width; ++i) {
        lh[(size_t)yh * low_width + i] = row[i * 2];
    }
    for (int i = 0; i < high_width; ++i) {
        hh[(size_t)yh * high_width + i] = row[i * 2 + 1];
    }
}
