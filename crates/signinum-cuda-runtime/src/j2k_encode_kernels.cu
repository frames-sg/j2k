#include <math.h>

static constexpr float J2K_FDWT97_ALPHA = -1.5861343f;
static constexpr float J2K_FDWT97_BETA = -0.052980117f;
static constexpr float J2K_FDWT97_GAMMA = 0.8829111f;
static constexpr float J2K_FDWT97_DELTA = 0.44350687f;
static constexpr float J2K_FDWT97_KAPPA = 1.2301741f;
static constexpr float J2K_FDWT97_INV_KAPPA = 1.0f / J2K_FDWT97_KAPPA;

extern "C" __global__ void signinum_j2k_deinterleave_to_f32(
    const unsigned char *pixels,
    float *components,
    unsigned long long num_pixels,
    unsigned int num_components,
    unsigned int bit_depth,
    unsigned int is_signed
) {
    const unsigned long long idx =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_pixels || num_components == 0u || num_components > 4u) {
        return;
    }

    const unsigned int bytes_per_sample = bit_depth <= 8u ? 1u : 2u;
    const float unsigned_offset =
        is_signed != 0u ? 0.0f : float(1u << (bit_depth - 1u));
    const unsigned long long pixel_base =
        idx * static_cast<unsigned long long>(num_components) * bytes_per_sample;
    for (unsigned int component = 0u; component < num_components; ++component) {
        const unsigned long long sample_base =
            pixel_base + static_cast<unsigned long long>(component) * bytes_per_sample;
        float sample;
        if (bit_depth <= 8u) {
            const unsigned char raw = pixels[sample_base];
            sample = is_signed != 0u
                ? float(static_cast<signed char>(raw))
                : float(raw) - unsigned_offset;
        } else {
            const unsigned short raw =
                static_cast<unsigned short>(pixels[sample_base])
                | (static_cast<unsigned short>(pixels[sample_base + 1u]) << 8u);
            sample = is_signed != 0u
                ? float(static_cast<short>(raw))
                : float(raw) - unsigned_offset;
        }
        components[static_cast<unsigned long long>(component) * num_pixels + idx] = sample;
    }
}

extern "C" __global__ void signinum_j2k_forward_rct(
    float *plane0,
    float *plane1,
    float *plane2,
    unsigned long long len
) {
    const unsigned long long idx =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    const float r = plane0[idx];
    const float g = plane1[idx];
    const float b = plane2[idx];
    plane0[idx] = floorf((r + 2.0f * g + b) * 0.25f);
    plane1[idx] = b - g;
    plane2[idx] = r - g;
}

extern "C" __global__ void signinum_j2k_forward_ict(
    float *plane0,
    float *plane1,
    float *plane2,
    unsigned long long len
) {
    const unsigned long long idx =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    const float r = plane0[idx];
    const float g = plane1[idx];
    const float b = plane2[idx];
    plane0[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    plane1[idx] = -0.16875f * r - 0.33126f * g + 0.5f * b;
    plane2[idx] = 0.5f * r - 0.41869f * g - 0.08131f * b;
}

__device__ float signinum_j2k_fdwt53_predict_row(
    const float *src,
    unsigned int row_base,
    unsigned int width,
    unsigned int high_index
) {
    const unsigned int odd = high_index * 2u + 1u;
    const unsigned int last_even = (width % 2u == 0u) ? width - 2u : width - 1u;
    const float left = src[row_base + odd - 1u];
    const float right = (odd + 1u < width) ? src[row_base + odd + 1u] : src[row_base + last_even];
    return src[row_base + odd] - floorf((left + right) * 0.5f);
}

__device__ float signinum_j2k_fdwt53_predict_col(
    const float *src,
    unsigned int x,
    unsigned int full_width,
    unsigned int height,
    unsigned int high_index
) {
    const unsigned int odd = high_index * 2u + 1u;
    const unsigned int last_even = (height % 2u == 0u) ? height - 2u : height - 1u;
    const float top = src[(odd - 1u) * full_width + x];
    const float bottom = (odd + 1u < height)
        ? src[(odd + 1u) * full_width + x]
        : src[last_even * full_width + x];
    return src[odd * full_width + x] - floorf((top + bottom) * 0.5f);
}

extern "C" __global__ void signinum_j2k_forward_dwt53_horizontal(
    const float *src,
    float *dst,
    unsigned int full_width,
    unsigned int current_width,
    unsigned int current_height,
    unsigned int low_width
) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= current_width || y >= current_height) {
        return;
    }

    const unsigned int row_base = y * full_width;
    if (x < low_width) {
        const unsigned int even = x * 2u;
        const float left = x > 0u
            ? signinum_j2k_fdwt53_predict_row(src, row_base, current_width, x - 1u)
            : signinum_j2k_fdwt53_predict_row(src, row_base, current_width, 0u);
        const float right = even + 1u < current_width
            ? signinum_j2k_fdwt53_predict_row(src, row_base, current_width, x)
            : left;
        dst[row_base + x] = src[row_base + even] + floorf((left + right) * 0.25f + 0.5f);
        return;
    }

    dst[row_base + x] = signinum_j2k_fdwt53_predict_row(
        src,
        row_base,
        current_width,
        x - low_width
    );
}

extern "C" __global__ void signinum_j2k_forward_dwt53_vertical(
    const float *src,
    float *dst,
    unsigned int full_width,
    unsigned int current_width,
    unsigned int current_height,
    unsigned int low_height
) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= current_width || y >= current_height) {
        return;
    }

    if (y < low_height) {
        const unsigned int even = y * 2u;
        const float top = y > 0u
            ? signinum_j2k_fdwt53_predict_col(src, x, full_width, current_height, y - 1u)
            : signinum_j2k_fdwt53_predict_col(src, x, full_width, current_height, 0u);
        const float bottom = even + 1u < current_height
            ? signinum_j2k_fdwt53_predict_col(src, x, full_width, current_height, y)
            : top;
        dst[y * full_width + x] =
            src[even * full_width + x] + floorf((top + bottom) * 0.25f + 0.5f);
        return;
    }

    dst[y * full_width + x] = signinum_j2k_fdwt53_predict_col(
        src,
        x,
        full_width,
        current_height,
        y - low_height
    );
}

__device__ float signinum_j2k_fdwt97_high1_row(
    const float *src,
    unsigned int row_base,
    unsigned int width,
    unsigned int high_index
) {
    const unsigned int odd = high_index * 2u + 1u;
    const unsigned int last_even = (width % 2u == 0u) ? width - 2u : width - 1u;
    const float left = src[row_base + odd - 1u];
    const float right = (odd + 1u < width) ? src[row_base + odd + 1u] : src[row_base + last_even];
    return src[row_base + odd] + J2K_FDWT97_ALPHA * (left + right);
}

__device__ float signinum_j2k_fdwt97_low1_row(
    const float *src,
    unsigned int row_base,
    unsigned int width,
    unsigned int low_index
) {
    const unsigned int even = low_index * 2u;
    const float left = low_index > 0u
        ? signinum_j2k_fdwt97_high1_row(src, row_base, width, low_index - 1u)
        : signinum_j2k_fdwt97_high1_row(src, row_base, width, 0u);
    const float right = even + 1u < width
        ? signinum_j2k_fdwt97_high1_row(src, row_base, width, low_index)
        : left;
    return src[row_base + even] + J2K_FDWT97_BETA * (left + right);
}

__device__ float signinum_j2k_fdwt97_high2_row(
    const float *src,
    unsigned int row_base,
    unsigned int width,
    unsigned int high_index
) {
    const unsigned int odd = high_index * 2u + 1u;
    const unsigned int last_even = (width % 2u == 0u) ? width - 2u : width - 1u;
    const unsigned int last_low = last_even / 2u;
    const float left = signinum_j2k_fdwt97_low1_row(src, row_base, width, high_index);
    const float right = (odd + 1u < width)
        ? signinum_j2k_fdwt97_low1_row(src, row_base, width, high_index + 1u)
        : signinum_j2k_fdwt97_low1_row(src, row_base, width, last_low);
    return signinum_j2k_fdwt97_high1_row(src, row_base, width, high_index)
        + J2K_FDWT97_GAMMA * (left + right);
}

__device__ float signinum_j2k_fdwt97_low2_row(
    const float *src,
    unsigned int row_base,
    unsigned int width,
    unsigned int low_index
) {
    const unsigned int even = low_index * 2u;
    const float left = low_index > 0u
        ? signinum_j2k_fdwt97_high2_row(src, row_base, width, low_index - 1u)
        : signinum_j2k_fdwt97_high2_row(src, row_base, width, 0u);
    const float right = even + 1u < width
        ? signinum_j2k_fdwt97_high2_row(src, row_base, width, low_index)
        : left;
    return signinum_j2k_fdwt97_low1_row(src, row_base, width, low_index)
        + J2K_FDWT97_DELTA * (left + right);
}

extern "C" __global__ void signinum_j2k_forward_dwt97_horizontal(
    const float *src,
    float *dst,
    unsigned int full_width,
    unsigned int current_width,
    unsigned int current_height,
    unsigned int low_width
) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= current_width || y >= current_height) {
        return;
    }

    const unsigned int row_base = y * full_width;
    if (x < low_width) {
        dst[row_base + x] = signinum_j2k_fdwt97_low2_row(src, row_base, current_width, x)
            * J2K_FDWT97_INV_KAPPA;
        return;
    }

    dst[row_base + x] = signinum_j2k_fdwt97_high2_row(
        src,
        row_base,
        current_width,
        x - low_width
    ) * J2K_FDWT97_KAPPA;
}

__device__ float signinum_j2k_fdwt97_high1_col(
    const float *src,
    unsigned int x,
    unsigned int full_width,
    unsigned int height,
    unsigned int high_index
) {
    const unsigned int odd = high_index * 2u + 1u;
    const unsigned int last_even = (height % 2u == 0u) ? height - 2u : height - 1u;
    const float top = src[(odd - 1u) * full_width + x];
    const float bottom = (odd + 1u < height)
        ? src[(odd + 1u) * full_width + x]
        : src[last_even * full_width + x];
    return src[odd * full_width + x] + J2K_FDWT97_ALPHA * (top + bottom);
}

__device__ float signinum_j2k_fdwt97_low1_col(
    const float *src,
    unsigned int x,
    unsigned int full_width,
    unsigned int height,
    unsigned int low_index
) {
    const unsigned int even = low_index * 2u;
    const float top = low_index > 0u
        ? signinum_j2k_fdwt97_high1_col(src, x, full_width, height, low_index - 1u)
        : signinum_j2k_fdwt97_high1_col(src, x, full_width, height, 0u);
    const float bottom = even + 1u < height
        ? signinum_j2k_fdwt97_high1_col(src, x, full_width, height, low_index)
        : top;
    return src[even * full_width + x] + J2K_FDWT97_BETA * (top + bottom);
}

__device__ float signinum_j2k_fdwt97_high2_col(
    const float *src,
    unsigned int x,
    unsigned int full_width,
    unsigned int height,
    unsigned int high_index
) {
    const unsigned int odd = high_index * 2u + 1u;
    const unsigned int last_even = (height % 2u == 0u) ? height - 2u : height - 1u;
    const unsigned int last_low = last_even / 2u;
    const float top = signinum_j2k_fdwt97_low1_col(src, x, full_width, height, high_index);
    const float bottom = (odd + 1u < height)
        ? signinum_j2k_fdwt97_low1_col(src, x, full_width, height, high_index + 1u)
        : signinum_j2k_fdwt97_low1_col(src, x, full_width, height, last_low);
    return signinum_j2k_fdwt97_high1_col(src, x, full_width, height, high_index)
        + J2K_FDWT97_GAMMA * (top + bottom);
}

__device__ float signinum_j2k_fdwt97_low2_col(
    const float *src,
    unsigned int x,
    unsigned int full_width,
    unsigned int height,
    unsigned int low_index
) {
    const unsigned int even = low_index * 2u;
    const float top = low_index > 0u
        ? signinum_j2k_fdwt97_high2_col(src, x, full_width, height, low_index - 1u)
        : signinum_j2k_fdwt97_high2_col(src, x, full_width, height, 0u);
    const float bottom = even + 1u < height
        ? signinum_j2k_fdwt97_high2_col(src, x, full_width, height, low_index)
        : top;
    return signinum_j2k_fdwt97_low1_col(src, x, full_width, height, low_index)
        + J2K_FDWT97_DELTA * (top + bottom);
}

extern "C" __global__ void signinum_j2k_forward_dwt97_vertical(
    const float *src,
    float *dst,
    unsigned int full_width,
    unsigned int current_width,
    unsigned int current_height,
    unsigned int low_height
) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= current_width || y >= current_height) {
        return;
    }

    if (y < low_height) {
        dst[y * full_width + x] =
            signinum_j2k_fdwt97_low2_col(src, x, full_width, current_height, y)
            * J2K_FDWT97_INV_KAPPA;
        return;
    }

    dst[y * full_width + x] = signinum_j2k_fdwt97_high2_col(
        src,
        x,
        full_width,
        current_height,
        y - low_height
    ) * J2K_FDWT97_KAPPA;
}

__device__ int signinum_j2k_quantize_sample(
    float sample,
    unsigned int step_exponent,
    unsigned int step_mantissa,
    unsigned int range_bits,
    unsigned int reversible
) {
    if (reversible != 0u) {
        const float rounded = sample >= 0.0f ? floorf(sample + 0.5f) : -floorf(-sample + 0.5f);
        return int(rounded);
    }

    const int exponent = int(range_bits) - int(step_exponent);
    const float base = ldexpf(1.0f, exponent);
    const float delta = base * (1.0f + float(step_mantissa) / 2048.0f);
    if (delta <= 0.0f) {
        return 0;
    }

    const int sign = sample < 0.0f ? -1 : 1;
    const int magnitude = int(floorf(fabsf(sample) / delta));
    return sign * magnitude;
}

extern "C" __global__ void signinum_j2k_quantize_subband(
    const float *samples,
    int *coefficients,
    unsigned long long len,
    unsigned int step_exponent,
    unsigned int step_mantissa,
    unsigned int range_bits,
    unsigned int reversible
) {
    const unsigned long long idx =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    coefficients[idx] = signinum_j2k_quantize_sample(
        samples[idx],
        step_exponent,
        step_mantissa,
        range_bits,
        reversible
    );
}

extern "C" __global__ void signinum_j2k_quantize_subband_strided(
    const float *samples,
    int *coefficients,
    unsigned int x0,
    unsigned int y0,
    unsigned int width,
    unsigned int height,
    unsigned int stride,
    unsigned int step_exponent,
    unsigned int step_mantissa,
    unsigned int range_bits,
    unsigned int reversible
) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const unsigned long long source_index =
        static_cast<unsigned long long>(y0 + y) * stride + x0 + x;
    const unsigned long long output_index =
        static_cast<unsigned long long>(y) * width + x;
    coefficients[output_index] = signinum_j2k_quantize_sample(
        samples[source_index],
        step_exponent,
        step_mantissa,
        range_bits,
        reversible
    );
}
