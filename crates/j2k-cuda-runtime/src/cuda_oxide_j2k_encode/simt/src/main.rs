use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

const J2K_FDWT97_ALPHA: f32 = -1.5861343;
const J2K_FDWT97_BETA: f32 = -0.052980117;
const J2K_FDWT97_GAMMA: f32 = 0.8829111;
const J2K_FDWT97_DELTA: f32 = 0.44350687;
const J2K_FDWT97_KAPPA: f32 = 1.2301741;
const J2K_FDWT97_INV_KAPPA: f32 = 1.0 / J2K_FDWT97_KAPPA;

#[inline(always)]
fn load_u8(ptr: *const u8, index: u64) -> u8 {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn load_f32(ptr: *const f32, index: u32) -> f32 {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn load_f32_u64(ptr: *const f32, index: u64) -> f32 {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn store_f32(ptr: *mut f32, index: u32, value: f32) {
    unsafe {
        *ptr.add(index as usize) = value;
    }
}

#[inline(always)]
fn store_f32_u64(ptr: *mut f32, index: u64, value: f32) {
    unsafe {
        *ptr.add(index as usize) = value;
    }
}

#[inline(always)]
fn store_i32(ptr: *mut i32, index: u64, value: i32) {
    unsafe {
        *ptr.add(index as usize) = value;
    }
}

#[inline(always)]
fn floor_f32(value: f32) -> f32 {
    // f32::floor routes through libdevice in cuda-oxide, which emits NVVM IR
    // instead of the PTX loaded by this runtime path.
    let truncated = value as i32 as f32;
    if truncated > value {
        truncated - 1.0
    } else {
        truncated
    }
}

#[inline(always)]
fn abs_f32(value: f32) -> f32 {
    if value < 0.0 { -value } else { value }
}

#[inline(always)]
fn j2k_fdwt53_predict_row(src: *const f32, row_base: u32, width: u32, high_index: u32) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if width % 2 == 0 { width - 2 } else { width - 1 };
    let left = load_f32(src, row_base + odd - 1);
    let right = if odd + 1 < width {
        load_f32(src, row_base + odd + 1)
    } else {
        load_f32(src, row_base + last_even)
    };
    load_f32(src, row_base + odd) - floor_f32((left + right) * 0.5)
}

#[inline(always)]
fn j2k_fdwt53_predict_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    high_index: u32,
) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if height % 2 == 0 { height - 2 } else { height - 1 };
    let top = load_f32(src, (odd - 1) * full_width + x);
    let bottom = if odd + 1 < height {
        load_f32(src, (odd + 1) * full_width + x)
    } else {
        load_f32(src, last_even * full_width + x)
    };
    load_f32(src, odd * full_width + x) - floor_f32((top + bottom) * 0.5)
}

#[inline(always)]
fn j2k_fdwt97_high1_row(src: *const f32, row_base: u32, width: u32, high_index: u32) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if width % 2 == 0 { width - 2 } else { width - 1 };
    let left = load_f32(src, row_base + odd - 1);
    let right = if odd + 1 < width {
        load_f32(src, row_base + odd + 1)
    } else {
        load_f32(src, row_base + last_even)
    };
    load_f32(src, row_base + odd) + J2K_FDWT97_ALPHA * (left + right)
}

#[inline(always)]
fn j2k_fdwt97_low1_row(src: *const f32, row_base: u32, width: u32, low_index: u32) -> f32 {
    let even = low_index * 2;
    let left = if low_index > 0 {
        j2k_fdwt97_high1_row(src, row_base, width, low_index - 1)
    } else {
        j2k_fdwt97_high1_row(src, row_base, width, 0)
    };
    let right = if even + 1 < width {
        j2k_fdwt97_high1_row(src, row_base, width, low_index)
    } else {
        left
    };
    load_f32(src, row_base + even) + J2K_FDWT97_BETA * (left + right)
}

#[inline(always)]
fn j2k_fdwt97_high2_row(src: *const f32, row_base: u32, width: u32, high_index: u32) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if width % 2 == 0 { width - 2 } else { width - 1 };
    let last_low = last_even / 2;
    let left = j2k_fdwt97_low1_row(src, row_base, width, high_index);
    let right = if odd + 1 < width {
        j2k_fdwt97_low1_row(src, row_base, width, high_index + 1)
    } else {
        j2k_fdwt97_low1_row(src, row_base, width, last_low)
    };
    j2k_fdwt97_high1_row(src, row_base, width, high_index)
        + J2K_FDWT97_GAMMA * (left + right)
}

#[inline(always)]
fn j2k_fdwt97_low2_row(src: *const f32, row_base: u32, width: u32, low_index: u32) -> f32 {
    let even = low_index * 2;
    let left = if low_index > 0 {
        j2k_fdwt97_high2_row(src, row_base, width, low_index - 1)
    } else {
        j2k_fdwt97_high2_row(src, row_base, width, 0)
    };
    let right = if even + 1 < width {
        j2k_fdwt97_high2_row(src, row_base, width, low_index)
    } else {
        left
    };
    j2k_fdwt97_low1_row(src, row_base, width, low_index) + J2K_FDWT97_DELTA * (left + right)
}

#[inline(always)]
fn j2k_fdwt97_high1_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    high_index: u32,
) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if height % 2 == 0 { height - 2 } else { height - 1 };
    let top = load_f32(src, (odd - 1) * full_width + x);
    let bottom = if odd + 1 < height {
        load_f32(src, (odd + 1) * full_width + x)
    } else {
        load_f32(src, last_even * full_width + x)
    };
    load_f32(src, odd * full_width + x) + J2K_FDWT97_ALPHA * (top + bottom)
}

#[inline(always)]
fn j2k_fdwt97_low1_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    low_index: u32,
) -> f32 {
    let even = low_index * 2;
    let top = if low_index > 0 {
        j2k_fdwt97_high1_col(src, x, full_width, height, low_index - 1)
    } else {
        j2k_fdwt97_high1_col(src, x, full_width, height, 0)
    };
    let bottom = if even + 1 < height {
        j2k_fdwt97_high1_col(src, x, full_width, height, low_index)
    } else {
        top
    };
    load_f32(src, even * full_width + x) + J2K_FDWT97_BETA * (top + bottom)
}

#[inline(always)]
fn j2k_fdwt97_high2_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    high_index: u32,
) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if height % 2 == 0 { height - 2 } else { height - 1 };
    let last_low = last_even / 2;
    let top = j2k_fdwt97_low1_col(src, x, full_width, height, high_index);
    let bottom = if odd + 1 < height {
        j2k_fdwt97_low1_col(src, x, full_width, height, high_index + 1)
    } else {
        j2k_fdwt97_low1_col(src, x, full_width, height, last_low)
    };
    j2k_fdwt97_high1_col(src, x, full_width, height, high_index)
        + J2K_FDWT97_GAMMA * (top + bottom)
}

#[inline(always)]
fn j2k_fdwt97_low2_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    low_index: u32,
) -> f32 {
    let even = low_index * 2;
    let top = if low_index > 0 {
        j2k_fdwt97_high2_col(src, x, full_width, height, low_index - 1)
    } else {
        j2k_fdwt97_high2_col(src, x, full_width, height, 0)
    };
    let bottom = if even + 1 < height {
        j2k_fdwt97_high2_col(src, x, full_width, height, low_index)
    } else {
        top
    };
    j2k_fdwt97_low1_col(src, x, full_width, height, low_index) + J2K_FDWT97_DELTA * (top + bottom)
}

#[inline(always)]
fn ldexp_one_f32(exponent: i32) -> f32 {
    if exponent < -149 {
        0.0
    } else if exponent < -126 {
        f32::from_bits(1_u32 << ((exponent + 149) as u32))
    } else if exponent <= 127 {
        f32::from_bits(((exponent + 127) as u32) << 23)
    } else {
        f32::INFINITY
    }
}

#[inline(always)]
fn j2k_quantize_sample(
    sample: f32,
    step_exponent: u32,
    step_mantissa: u32,
    range_bits: u32,
    reversible: u32,
) -> i32 {
    if reversible != 0 {
        let rounded = if sample >= 0.0 {
            floor_f32(sample + 0.5)
        } else {
            -floor_f32(-sample + 0.5)
        };
        return rounded as i32;
    }

    let exponent = range_bits as i32 - step_exponent as i32;
    let base = ldexp_one_f32(exponent);
    let delta = base * (1.0 + step_mantissa as f32 / 2048.0);
    if delta <= 0.0 {
        return 0;
    }

    let sign = if sample < 0.0 { -1 } else { 1 };
    let magnitude = floor_f32(abs_f32(sample) / delta) as i32;
    sign * magnitude
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn j2k_deinterleave_to_f32(
        pixels: *const u8,
        components: *mut f32,
        num_pixels: u64,
        num_components: u32,
        bit_depth: u32,
        is_signed: u32,
    ) {
        let idx = thread::index_1d().get() as u64;
        if idx >= num_pixels || num_components == 0 || num_components > 4 {
            return;
        }

        let bytes_per_sample = if bit_depth <= 8 { 1_u32 } else { 2_u32 };
        let unsigned_offset = if is_signed != 0 {
            0.0
        } else {
            (1_u32 << (bit_depth - 1)) as f32
        };
        let pixel_base = idx * num_components as u64 * bytes_per_sample as u64;
        let mut component = 0_u32;
        while component < num_components {
            let sample_base = pixel_base + component as u64 * bytes_per_sample as u64;
            let sample = if bit_depth <= 8 {
                let raw = load_u8(pixels, sample_base);
                if is_signed != 0 {
                    (raw as i8) as f32
                } else {
                    raw as f32 - unsigned_offset
                }
            } else {
                let raw = load_u8(pixels, sample_base) as u16
                    | ((load_u8(pixels, sample_base + 1) as u16) << 8);
                if is_signed != 0 {
                    (raw as i16) as f32
                } else {
                    raw as f32 - unsigned_offset
                }
            };
            store_f32_u64(components, component as u64 * num_pixels + idx, sample);
            component += 1;
        }
    }

    #[kernel]
    pub unsafe fn j2k_deinterleave_strided_to_f32(
        pixels: *const u8,
        components: *mut f32,
        width: u64,
        height: u64,
        byte_offset: u64,
        pitch_bytes: u64,
        num_components: u32,
        bit_depth: u32,
        is_signed: u32,
    ) {
        let idx = thread::index_1d().get() as u64;
        let num_pixels = width * height;
        if idx >= num_pixels || num_components == 0 || num_components > 4 {
            return;
        }

        let bytes_per_sample = if bit_depth <= 8 { 1_u32 } else { 2_u32 };
        let unsigned_offset = if is_signed != 0 {
            0.0
        } else {
            (1_u32 << (bit_depth - 1)) as f32
        };
        let y = idx / width;
        let x = idx - y * width;
        let pixel_base =
            byte_offset + y * pitch_bytes + x * num_components as u64 * bytes_per_sample as u64;
        let mut component = 0_u32;
        while component < num_components {
            let sample_base = pixel_base + component as u64 * bytes_per_sample as u64;
            let sample = if bit_depth <= 8 {
                let raw = load_u8(pixels, sample_base);
                if is_signed != 0 {
                    (raw as i8) as f32
                } else {
                    raw as f32 - unsigned_offset
                }
            } else {
                let raw = load_u8(pixels, sample_base) as u16
                    | ((load_u8(pixels, sample_base + 1) as u16) << 8);
                if is_signed != 0 {
                    (raw as i16) as f32
                } else {
                    raw as f32 - unsigned_offset
                }
            };
            store_f32_u64(components, component as u64 * num_pixels + idx, sample);
            component += 1;
        }
    }

    #[kernel]
    pub unsafe fn j2k_forward_rct(
        plane0: *mut f32,
        plane1: *mut f32,
        plane2: *mut f32,
        len: u64,
    ) {
        let idx = thread::index_1d().get() as u64;
        if idx >= len {
            return;
        }

        let r = load_f32_u64(plane0.cast_const(), idx);
        let g = load_f32_u64(plane1.cast_const(), idx);
        let b = load_f32_u64(plane2.cast_const(), idx);
        store_f32_u64(plane0, idx, floor_f32((r + 2.0 * g + b) * 0.25));
        store_f32_u64(plane1, idx, b - g);
        store_f32_u64(plane2, idx, r - g);
    }

    #[kernel]
    pub unsafe fn j2k_forward_ict(
        plane0: *mut f32,
        plane1: *mut f32,
        plane2: *mut f32,
        len: u64,
    ) {
        let idx = thread::index_1d().get() as u64;
        if idx >= len {
            return;
        }

        let r = load_f32_u64(plane0.cast_const(), idx);
        let g = load_f32_u64(plane1.cast_const(), idx);
        let b = load_f32_u64(plane2.cast_const(), idx);
        store_f32_u64(plane0, idx, 0.299 * r + 0.587 * g + 0.114 * b);
        store_f32_u64(plane1, idx, -0.16875 * r - 0.33126 * g + 0.5 * b);
        store_f32_u64(plane2, idx, 0.5 * r - 0.41869 * g - 0.08131 * b);
    }

    #[kernel]
    pub unsafe fn j2k_forward_dwt53_horizontal(
        src: *const f32,
        dst: *mut f32,
        full_width: u32,
        current_width: u32,
        current_height: u32,
        low_width: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= current_width || y >= current_height {
            return;
        }

        let row_base = y * full_width;
        if x < low_width {
            let even = x * 2;
            let left = if x > 0 {
                j2k_fdwt53_predict_row(src, row_base, current_width, x - 1)
            } else {
                j2k_fdwt53_predict_row(src, row_base, current_width, 0)
            };
            let right = if even + 1 < current_width {
                j2k_fdwt53_predict_row(src, row_base, current_width, x)
            } else {
                left
            };
            let value = load_f32(src, row_base + even) + floor_f32((left + right) * 0.25 + 0.5);
            store_f32(dst, row_base + x, value);
            return;
        }

        let value = j2k_fdwt53_predict_row(src, row_base, current_width, x - low_width);
        store_f32(dst, row_base + x, value);
    }

    #[kernel]
    pub unsafe fn j2k_forward_dwt53_vertical(
        src: *const f32,
        dst: *mut f32,
        full_width: u32,
        current_width: u32,
        current_height: u32,
        low_height: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= current_width || y >= current_height {
            return;
        }

        if y < low_height {
            let even = y * 2;
            let top = if y > 0 {
                j2k_fdwt53_predict_col(src, x, full_width, current_height, y - 1)
            } else {
                j2k_fdwt53_predict_col(src, x, full_width, current_height, 0)
            };
            let bottom = if even + 1 < current_height {
                j2k_fdwt53_predict_col(src, x, full_width, current_height, y)
            } else {
                top
            };
            let value =
                load_f32(src, even * full_width + x) + floor_f32((top + bottom) * 0.25 + 0.5);
            store_f32(dst, y * full_width + x, value);
            return;
        }

        let value = j2k_fdwt53_predict_col(src, x, full_width, current_height, y - low_height);
        store_f32(dst, y * full_width + x, value);
    }

    #[kernel]
    pub unsafe fn j2k_forward_dwt97_horizontal(
        src: *const f32,
        dst: *mut f32,
        full_width: u32,
        current_width: u32,
        current_height: u32,
        low_width: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= current_width || y >= current_height {
            return;
        }

        let row_base = y * full_width;
        let value = if x < low_width {
            j2k_fdwt97_low2_row(src, row_base, current_width, x) * J2K_FDWT97_INV_KAPPA
        } else {
            j2k_fdwt97_high2_row(src, row_base, current_width, x - low_width) * J2K_FDWT97_KAPPA
        };
        store_f32(dst, row_base + x, value);
    }

    #[kernel]
    pub unsafe fn j2k_forward_dwt97_vertical(
        src: *const f32,
        dst: *mut f32,
        full_width: u32,
        current_width: u32,
        current_height: u32,
        low_height: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= current_width || y >= current_height {
            return;
        }

        let value = if y < low_height {
            j2k_fdwt97_low2_col(src, x, full_width, current_height, y) * J2K_FDWT97_INV_KAPPA
        } else {
            j2k_fdwt97_high2_col(src, x, full_width, current_height, y - low_height)
                * J2K_FDWT97_KAPPA
        };
        store_f32(dst, y * full_width + x, value);
    }

    #[kernel]
    pub unsafe fn j2k_quantize_subband(
        samples: *const f32,
        coefficients: *mut i32,
        len: u64,
        step_exponent: u32,
        step_mantissa: u32,
        range_bits: u32,
        reversible: u32,
    ) {
        let idx = thread::index_1d().get() as u64;
        if idx >= len {
            return;
        }

        let coefficient = j2k_quantize_sample(
            load_f32_u64(samples, idx),
            step_exponent,
            step_mantissa,
            range_bits,
            reversible,
        );
        store_i32(coefficients, idx, coefficient);
    }

    #[kernel]
    pub unsafe fn j2k_quantize_subband_strided(
        samples: *const f32,
        coefficients: *mut i32,
        x0: u32,
        y0: u32,
        width: u32,
        height: u32,
        stride: u32,
        step_exponent: u32,
        step_mantissa: u32,
        range_bits: u32,
        reversible: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= width || y >= height {
            return;
        }

        let source_index = (y0 + y) as u64 * stride as u64 + (x0 + x) as u64;
        let output_index = y as u64 * width as u64 + x as u64;
        let coefficient = j2k_quantize_sample(
            load_f32_u64(samples, source_index),
            step_exponent,
            step_mantissa,
            range_bits,
            reversible,
        );
        store_i32(coefficients, output_index, coefficient);
    }
}

fn main() {}
