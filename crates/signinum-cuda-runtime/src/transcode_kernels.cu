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

// ===========================================================================
// Irreversible 9/7 path: faithful f32 port of the signinum-transcode scalar
// oracle (dct97_2d.rs). Separable float IDCT (idct8x8_sample) into a spatial
// plane, then the separable single-level 9/7 transform (forward_lift_97):
// rows -> {row_low, row_high}; columns of row_low -> {LL, LH}; columns of
// row_high -> {HL, HH}. Device math is f32 (Metal uses f32 here too); parity
// is asserted within the 2e-2 band tolerance. --fmad=false keeps the lift
// operation ordering close to the scalar reference.
// ===========================================================================

typedef float f32;

#define J2K_PI 3.14159265358979323846
#define DWT97_ALPHA (-1.586134342059924f)
#define DWT97_BETA  (-0.052980118572961f)
#define DWT97_GAMMA (0.882911075530934f)
#define DWT97_DELTA (0.443506852043971f)
#define DWT97_KAPPA (1.230174104914001f)
#define DWT97_INV_KAPPA (1.0f / DWT97_KAPPA)

// idct8_basis(sample_idx, freq): scale * cos((s+0.5)*f*pi/8), scale = sqrt(1/8)
// for freq 0 else sqrt(2/8). Matches idct8_basis_uncached in dct97_2d.rs.
// Precomputed to avoid per-sample sqrtf/cosf in the CUDA hot path.
__device__ __constant__ f32 DWT97_IDCT8_BASIS[64] = {
    0.353553391f, 0.49039264f, 0.461939766f, 0.415734806f, 0.353553391f, 0.277785117f, 0.191341716f, 0.097545161f,
    0.353553391f, 0.415734806f, 0.191341716f, -0.097545161f, -0.353553391f, -0.49039264f, -0.461939766f, -0.277785117f,
    0.353553391f, 0.277785117f, -0.191341716f, -0.49039264f, -0.353553391f, 0.097545161f, 0.461939766f, 0.415734806f,
    0.353553391f, 0.097545161f, -0.461939766f, -0.277785117f, 0.353553391f, 0.415734806f, -0.191341716f, -0.49039264f,
    0.353553391f, -0.097545161f, -0.461939766f, 0.277785117f, 0.353553391f, -0.415734806f, -0.191341716f, 0.49039264f,
    0.353553391f, -0.277785117f, -0.191341716f, 0.49039264f, -0.353553391f, -0.097545161f, 0.461939766f, -0.415734806f,
    0.353553391f, -0.415734806f, 0.191341716f, 0.097545161f, -0.353553391f, 0.49039264f, -0.461939766f, 0.277785117f,
    0.353553391f, -0.49039264f, 0.461939766f, -0.415734806f, 0.353553391f, -0.277785117f, 0.191341716f, -0.097545161f
};

__device__ __forceinline__ f32 idct8_basis(int sample_idx, int freq) {
    return DWT97_IDCT8_BASIS[sample_idx * 8 + freq];
}

// One IDCT sample, matching idct8x8_sample's accumulation order
// (freq_y outer, freq_x inner; coeff * y_basis * x_basis).
__device__ f32 idct8x8_sample(const f32* block, int local_x, int local_y) {
    f32 sample = 0.0f;
    // transcode_dwt97_idct_unroll_guard: the IDCT loops have fixed 8x8 bounds.
#pragma unroll
    for (int freq_y = 0; freq_y < 8; ++freq_y) {
        const f32 y_basis = idct8_basis(local_y, freq_y);
        const f32* brow = block + freq_y * 8;
#pragma unroll
        for (int freq_x = 0; freq_x < 8; ++freq_x) {
            sample += brow[freq_x] * y_basis * idct8_basis(local_x, freq_x);
        }
    }
    return sample;
}

// Same accumulation order as idct8x8_sample, with exact i16->f32 conversion
// performed in-kernel to avoid host-side expansion and halve HtoD traffic.
__device__ f32 idct8x8_sample_i16(const i16* block, int local_x, int local_y) {
    f32 sample = 0.0f;
#pragma unroll
    for (int freq_y = 0; freq_y < 8; ++freq_y) {
        const f32 y_basis = idct8_basis(local_y, freq_y);
        const i16* brow = block + freq_y * 8;
#pragma unroll
        for (int freq_x = 0; freq_x < 8; ++freq_x) {
            sample += ((f32)brow[freq_x]) * y_basis * idct8_basis(local_x, freq_x);
        }
    }
    return sample;
}

// forward_lift_97 applied with element stride `s` (1 for rows, band width for
// columns), in place over `n` logical samples. Matches forward_lift_97.
__device__ void forward_lift_97(f32* d, int n, int s) {
    if (n < 2) {
        return;
    }
    const int last_even = (n % 2 == 0) ? (n - 2) : (n - 1);
    for (int i = 1; i < n; i += 2) {
        const f32 left = d[(i - 1) * s];
        const f32 right = (i + 1 < n) ? d[(i + 1) * s] : d[last_even * s];
        d[i * s] += DWT97_ALPHA * (left + right);
    }
    for (int i = 0; i < n; i += 2) {
        const f32 left = (i > 0) ? d[(i - 1) * s] : d[1 * s];
        const f32 right = (i + 1 < n) ? d[(i + 1) * s] : left;
        d[i * s] += DWT97_BETA * (left + right);
    }
    for (int i = 1; i < n; i += 2) {
        const f32 left = d[(i - 1) * s];
        const f32 right = (i + 1 < n) ? d[(i + 1) * s] : d[last_even * s];
        d[i * s] += DWT97_GAMMA * (left + right);
    }
    for (int i = 0; i < n; i += 2) {
        const f32 left = (i > 0) ? d[(i - 1) * s] : d[1 * s];
        const f32 right = (i + 1 < n) ? d[(i + 1) * s] : left;
        d[i * s] += DWT97_DELTA * (left + right);
    }
    for (int i = 0; i < n; i += 2) {
        d[i * s] *= DWT97_INV_KAPPA;
    }
    for (int i = 1; i < n; i += 2) {
        d[i * s] *= DWT97_KAPPA;
    }
}

// Kernel: separable float IDCT into a width*height spatial plane.
// One thread per (x, y).
extern "C" __global__ void transcode_dwt97_idct(
    const f32* blocks, int block_cols, int width, int height, f32* spatial) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int block_idx = (y >> 3) * block_cols + (x >> 3);
    const f32* block = blocks + (size_t)block_idx * 64;
    spatial[(size_t)y * width + x] = idct8x8_sample(block, x & 7, y & 7);
}

// Kernel: horizontal 9/7 lift per row, then split even -> row_low, odd ->
// row_high. One thread per row.
extern "C" __global__ void transcode_dwt97_row_lift(
    f32* spatial, int width, int height, int low_width, int high_width,
    f32* row_low, f32* row_high) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) {
        return;
    }
    f32* row = spatial + (size_t)y * width;
    forward_lift_97(row, width, 1);
    for (int i = 0; i < low_width; ++i) {
        row_low[(size_t)y * low_width + i] = row[i * 2];
    }
    for (int i = 0; i < high_width; ++i) {
        row_high[(size_t)y * high_width + i] = row[i * 2 + 1];
    }
}

// Kernel: vertical 9/7 lift per column (strided) of a band buffer, then split
// even rows -> low_out, odd rows -> high_out. Used for row_low -> {LL, LH} and
// row_high -> {HL, HH}. One thread per column. `band_width` is the column count
// (= the in-place stride).
extern "C" __global__ void transcode_dwt97_column_lift(
    f32* rows, int band_width, int height, f32* low_out, f32* high_out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= band_width) {
        return;
    }
    forward_lift_97(rows + x, height, band_width);
    for (int i = 0; i < height; ++i) {
        const f32 value = rows[(size_t)i * band_width + x];
        if ((i & 1) == 0) {
            low_out[(size_t)(i / 2) * band_width + x] = value;
        } else {
            high_out[(size_t)(i / 2) * band_width + x] = value;
        }
    }
}

// ===========================================================================
// Same-geometry 9/7 batch: per-item replicas of the three single-job kernels
// above, selecting the item from a grid dimension and offsetting every buffer
// by the item's uniform per-item stride. Output is bit-identical to running the
// single-job kernels once per item (same f32 op ordering, same --fmad=false).
// ===========================================================================

// Kernel: batched separable float IDCT. Grid (x, y, item); thread per sample.
// `blocks_per_item` = block_cols * block_rows; spatial item stride = width*height.
extern "C" __global__ void transcode_dwt97_idct_batch(
    const f32* blocks, int block_cols, int width, int height,
    int blocks_per_item, f32* spatial) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int item = blockIdx.z;
    if (x >= width || y >= height) {
        return;
    }
    const f32* item_blocks = blocks + (size_t)item * blocks_per_item * 64;
    const int block_idx = (y >> 3) * block_cols + (x >> 3);
    const f32* block = item_blocks + (size_t)block_idx * 64;
    spatial[((size_t)item * height + y) * width + x] = idct8x8_sample(block, x & 7, y & 7);
}

// Same as transcode_dwt97_idct_batch, but input coefficients stay i16 on the
// host/device boundary and are widened to f32 inside the IDCT sample loop.
extern "C" __global__ void transcode_dwt97_idct_i16_batch(
    const i16* blocks, int block_cols, int width, int height,
    int blocks_per_item, f32* spatial) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int item = blockIdx.z;
    if (x >= width || y >= height) {
        return;
    }
    const i16* item_blocks = blocks + (size_t)item * blocks_per_item * 64;
    const int block_idx = (y >> 3) * block_cols + (x >> 3);
    const i16* block = item_blocks + (size_t)block_idx * 64;
    spatial[((size_t)item * height + y) * width + x] = idct8x8_sample_i16(block, x & 7, y & 7);
}

// Kernel: batched horizontal 9/7 row lift + split. Grid (row, item); thread per row.
extern "C" __global__ void transcode_dwt97_row_lift_batch(
    f32* spatial, int width, int height, int low_width, int high_width,
    f32* row_low, f32* row_high) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int item = blockIdx.y;
    if (y >= height) {
        return;
    }
    f32* item_spatial = spatial + (size_t)item * width * height;
    f32* item_row_low = row_low + (size_t)item * height * low_width;
    f32* item_row_high = row_high + (size_t)item * height * high_width;
    f32* row = item_spatial + (size_t)y * width;
    forward_lift_97(row, width, 1);
    for (int i = 0; i < low_width; ++i) {
        item_row_low[(size_t)y * low_width + i] = row[i * 2];
    }
    for (int i = 0; i < high_width; ++i) {
        item_row_high[(size_t)y * high_width + i] = row[i * 2 + 1];
    }
}

#define DWT97_ROW_LIFT_MAX_WIDTH 1024
#define DWT97_ROW_LIFT_ROWS_PER_BLOCK 4

// Cooperative row lift for common tile widths. Each block handles up to four
// rows, staging them in shared memory so each lifting phase can run in parallel
// across row positions while preserving the scalar phase order.
extern "C" __global__ void transcode_dwt97_row_lift_batch_coop(
    const f32* spatial, int width, int height, int low_width, int high_width,
    f32* row_low, f32* row_high) {
    __shared__ f32 rows[DWT97_ROW_LIFT_ROWS_PER_BLOCK][DWT97_ROW_LIFT_MAX_WIDTH];

    const int row_lane = threadIdx.y;
    const int tid = threadIdx.x;
    const int y = blockIdx.x * DWT97_ROW_LIFT_ROWS_PER_BLOCK + row_lane;
    const int item = blockIdx.y;
    const bool valid = y < height && width <= DWT97_ROW_LIFT_MAX_WIDTH;

    const f32* item_spatial = spatial + (size_t)item * width * height;
    f32* item_row_low = row_low + (size_t)item * height * low_width;
    f32* item_row_high = row_high + (size_t)item * height * high_width;
    const f32* source = item_spatial + (size_t)y * width;
    f32* row = rows[row_lane];

    if (valid) {
        for (int i = tid; i < width; i += blockDim.x) {
            row[i] = source[i];
        }
    }
    __syncthreads();

    if (width >= 2 && width <= DWT97_ROW_LIFT_MAX_WIDTH) {
        const int last_even = (width % 2 == 0) ? (width - 2) : (width - 1);
        if (valid) {
            for (int i = tid * 2 + 1; i < width; i += blockDim.x * 2) {
                const f32 left = row[i - 1];
                const f32 right = (i + 1 < width) ? row[i + 1] : row[last_even];
                row[i] += DWT97_ALPHA * (left + right);
            }
        }
        __syncthreads();

        if (valid) {
            for (int i = tid * 2; i < width; i += blockDim.x * 2) {
                const f32 left = (i > 0) ? row[i - 1] : row[1];
                const f32 right = (i + 1 < width) ? row[i + 1] : left;
                row[i] += DWT97_BETA * (left + right);
            }
        }
        __syncthreads();

        if (valid) {
            for (int i = tid * 2 + 1; i < width; i += blockDim.x * 2) {
                const f32 left = row[i - 1];
                const f32 right = (i + 1 < width) ? row[i + 1] : row[last_even];
                row[i] += DWT97_GAMMA * (left + right);
            }
        }
        __syncthreads();

        if (valid) {
            for (int i = tid * 2; i < width; i += blockDim.x * 2) {
                const f32 left = (i > 0) ? row[i - 1] : row[1];
                const f32 right = (i + 1 < width) ? row[i + 1] : left;
                row[i] += DWT97_DELTA * (left + right);
            }
        }
        __syncthreads();

        if (valid) {
            for (int i = tid * 2; i < width; i += blockDim.x * 2) {
                row[i] *= DWT97_INV_KAPPA;
            }
            for (int i = tid * 2 + 1; i < width; i += blockDim.x * 2) {
                row[i] *= DWT97_KAPPA;
            }
        }
        __syncthreads();
    }

    if (valid) {
        for (int i = tid; i < low_width; i += blockDim.x) {
            item_row_low[(size_t)y * low_width + i] = row[i * 2];
        }
        for (int i = tid; i < high_width; i += blockDim.x) {
            item_row_high[(size_t)y * high_width + i] = row[i * 2 + 1];
        }
    }
}

// Kernel: batched vertical 9/7 column lift + split. Grid (column, item); thread
// per column. `low_height`/`high_height` give the per-item output band strides.
extern "C" __global__ void transcode_dwt97_column_lift_batch(
    f32* rows, int band_width, int height, int low_height, int high_height,
    f32* low_out, f32* high_out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int item = blockIdx.y;
    if (x >= band_width) {
        return;
    }
    f32* item_rows = rows + (size_t)item * height * band_width;
    f32* item_low = low_out + (size_t)item * low_height * band_width;
    f32* item_high = high_out + (size_t)item * high_height * band_width;
    forward_lift_97(item_rows + x, height, band_width);
    for (int i = 0; i < height; ++i) {
        const f32 value = item_rows[(size_t)i * band_width + x];
        if ((i & 1) == 0) {
            item_low[(size_t)(i / 2) * band_width + x] = value;
        } else {
            item_high[(size_t)(i / 2) * band_width + x] = value;
        }
    }
}

__device__ __forceinline__ i32 quantize_dwt97_deadzone(f32 value, f32 inv_delta) {
    const int sign = (value < 0.0f) ? -1 : 1;
    const int magnitude = (int)floorf(fabsf(value) * inv_delta);
    return sign * magnitude;
}

__device__ __forceinline__ size_t dwt97_codeblock_major_offset(
    int x, int y, int width, int height, int cb_width, int cb_height) {
    if (cb_width == 64 && cb_height == 64) {
        const int cbx = x >> 6;
        const int cby = y >> 6;
        const int local_x = x & 63;
        const int local_y = y & 63;
        const int block_width = min(64, width - (cbx << 6));
        const int block_height = min(64, height - (cby << 6));
        return (size_t)cby * 64 * width
            + (size_t)cbx * 64 * block_height
            + (size_t)local_y * block_width + local_x;
    }
    const int cbx = x / cb_width;
    const int cby = y / cb_height;
    const int local_x = x - cbx * cb_width;
    const int local_y = y - cby * cb_height;
    const int block_width = min(cb_width, width - cbx * cb_width);
    const int block_height = min(cb_height, height - cby * cb_height);
    return (size_t)cby * cb_height * width
        + (size_t)cbx * cb_width * block_height
        + (size_t)local_y * block_width + local_x;
}

// Kernel: deadzone-quantize one 9/7 band into code-block-major i32 layout for
// one batch, mirroring the signinum-transcode shared code-block oracle and
// Metal's dct97_quantize_codeblocks_batch. Launched once per subband (its own
// width, height, and inv_delta). Grid (x, y, item); thread per band sample.
//
//   q = sign(value) * floor(|value| * inv_delta), sign(0) = +1
//
// Output offset for (cbx, cby, local): code-block-major (outer cby, inner cbx),
// each block stored row-major; per-item stride = width*height (coeffs preserved).
extern "C" __global__ void transcode_dwt97_quantize_codeblocks(
    const f32* band, i32* output, int width, int height,
    int cb_width, int cb_height, f32 inv_delta) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int item = blockIdx.z;
    if (x >= width || y >= height) {
        return;
    }
    const size_t item_stride = (size_t)width * height;
    const f32 value = band[(size_t)item * item_stride + (size_t)y * width + x];
    const size_t offset =
        dwt97_codeblock_major_offset(x, y, width, height, cb_width, cb_height);
    output[(size_t)item * item_stride + offset] =
        quantize_dwt97_deadzone(value, inv_delta);
}

// Fused resident HT path stage: vertical 9/7 column lift, even/odd row split,
// and deadzone quantization directly into code-block-major i32 output. This
// keeps the same forward_lift_97 global-memory update order as
// transcode_dwt97_column_lift_batch, then quantizes the f32 value that would
// otherwise have been stored in the intermediate LL/LH/HL/HH band buffer.
extern "C" __global__ void transcode_dwt97_column_lift_quantize_codeblocks_batch(
    f32* rows, int band_width, int height, int low_height, int high_height,
    i32* low_out, i32* high_out, int cb_width, int cb_height,
    f32 inv_delta_low, f32 inv_delta_high) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int item = blockIdx.y;
    if (x >= band_width) {
        return;
    }

    f32* item_rows = rows + (size_t)item * height * band_width;
    i32* item_low = low_out + (size_t)item * low_height * band_width;
    i32* item_high = high_out + (size_t)item * high_height * band_width;

    forward_lift_97(item_rows + x, height, band_width);
    for (int i = 0; i < height; ++i) {
        const f32 value = item_rows[(size_t)i * band_width + x];
        if ((i & 1) == 0) {
            const int y = i / 2;
            const size_t offset = dwt97_codeblock_major_offset(
                x, y, band_width, low_height, cb_width, cb_height);
            item_low[offset] = quantize_dwt97_deadzone(value, inv_delta_low);
        } else {
            const int y = i / 2;
            const size_t offset = dwt97_codeblock_major_offset(
                x, y, band_width, high_height, cb_width, cb_height);
            item_high[offset] = quantize_dwt97_deadzone(value, inv_delta_high);
        }
    }
}
