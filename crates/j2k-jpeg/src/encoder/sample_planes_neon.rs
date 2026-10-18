// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact `AArch64` NEON RGB-to-YCbCr conversion for the CPU encoder.

use alloc::vec::Vec;
use core::arch::aarch64::{
    uint16x4_t, uint16x8_t, uint8x16_t, uint8x8_t, vadd_s16, vaddq_u16, vcombine_u16, vcombine_u8,
    vdup_n_s16, vext_u8, vget_high_u16, vget_high_u8, vget_lane_u16, vget_lane_u32, vget_lane_u8,
    vget_low_u16, vget_low_u8, vld1q_u8, vld3q_u8, vmlal_n_s16, vmlsl_n_s16, vmovl_u8, vmull_n_s16,
    vpaddlq_u8, vqmovn_u16, vqtbl1q_u8, vreinterpret_s16_u16, vreinterpret_u16_s16,
    vreinterpret_u16_u8, vreinterpret_u32_u8, vrshrn_n_s32, vshll_n_s16, vshrn_n_u16, vst1_u8,
    vst1q_u8, vsub_s16,
};

use super::PrecomputedChromaLayout;

const LANES: usize = 16;
const RGB_BYTES: usize = LANES * 3;
const COMPACT_TAIL_SHUFFLE_INDICES: [[u8; LANES]; LANES] = compact_tail_shuffle_indices();

#[expect(
    clippy::cast_possible_truncation,
    reason = "tail shuffle indices are clamped to the sixteen input lanes"
)]
const fn compact_tail_shuffle_indices() -> [[u8; LANES]; LANES] {
    let mut indices = [[0u8; LANES]; LANES];
    let mut tail = 1;
    while tail < LANES {
        let delta = LANES - tail;
        let mut lane = 0;
        while lane < LANES {
            let source = delta + lane;
            indices[tail][lane] = if source < LANES {
                source as u8
            } else {
                (LANES - 1) as u8
            };
            lane += 1;
        }
        tail += 1;
    }
    indices
}

pub(super) fn rgb_to_ycbcr_planes(
    neon: fearless_simd::Neon,
    data: &[u8],
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    rgb_to_ycbcr_planes_kernel(neon, data, y_plane, cb_plane, cr_plane);
}

#[inline(never)]
pub(super) fn sample_422_block(
    neon: fearless_simd::Neon,
    plane: &[u8],
    width: usize,
    origin_x: usize,
    origin_y: usize,
    out: &mut [u8; 64],
) {
    let start = origin_y
        .checked_mul(width)
        .and_then(|row| row.checked_add(origin_x))
        .expect("4:2:2 block start fits usize");
    let span_len = 7usize
        .checked_mul(width)
        .and_then(|rows| rows.checked_add(16))
        .expect("4:2:2 block span fits usize");
    let end = start
        .checked_add(span_len)
        .expect("4:2:2 block end fits usize");
    let source = plane
        .get(start..end)
        .expect("4:2:2 block source span is present");
    sample_422_block_kernel(neon, source, width, out);
}

pub(super) fn rgb_to_ycbcr_block_planes(
    neon: fearless_simd::Neon,
    data: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    rgb_to_ycbcr_block_planes_kernel(neon, data, width, y_plane, cb_plane, cr_plane);
}

pub(super) fn rgb_to_ycbcr_422_block_planes(
    neon: fearless_simd::Neon,
    data: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    rgb_to_ycbcr_422_block_planes_kernel(neon, data, width, y_plane, cb_plane, cr_plane);
}

pub(super) fn rgb_to_ycbcr_420_block_planes(
    neon: fearless_simd::Neon,
    data: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    rgb_to_ycbcr_420_block_planes_kernel(neon, data, width, y_plane, cb_plane, cr_plane);
}

pub(super) fn rgb_to_ycbcr_compact_block_planes(
    neon: fearless_simd::Neon,
    data: &[u8],
    width: usize,
    height: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    rgb_to_ycbcr_compact_block_planes_kernel(
        neon, data, width, height, y_plane, cb_plane, cr_plane,
    );
}

/// Fill one compact chroma row's short horizontal padding.
///
/// # Safety
/// `output` must be valid for writes of `count` bytes, and `count` must be in
/// `1..=7`.
#[expect(
    clippy::inline_always,
    reason = "short chroma padding stores remain folded into the measured compact-plane tail"
)]
#[inline(always)]
unsafe fn fill_compact_chroma_padding(output: *mut u8, value: u8, count: usize) {
    debug_assert!((1..8).contains(&count));
    if count >= 4 {
        let repeated = u32::from_ne_bytes([value; 4]);
        // SAFETY: both unaligned four-byte stores stay within the promised
        // `count`-byte region. Their overlap, if any, writes the same value.
        unsafe {
            output.cast::<u32>().write_unaligned(repeated);
            output
                .add(count - 4)
                .cast::<u32>()
                .write_unaligned(repeated);
        }
    } else if count >= 2 {
        let repeated = u16::from_ne_bytes([value; 2]);
        // SAFETY: both unaligned two-byte stores stay within the promised
        // `count`-byte region. Their overlap, if any, writes the same value.
        unsafe {
            output.cast::<u16>().write_unaligned(repeated);
            output
                .add(count - 2)
                .cast::<u16>()
                .write_unaligned(repeated);
        }
    } else {
        // SAFETY: the one-byte region promised by the caller is writable.
        unsafe { output.write(value) };
    }
}

/// Store a nonempty prefix of one eight-byte vector.
///
/// # Safety
/// `output` must be valid for writes of `count` bytes, and `count` must be in
/// `1..=8`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_u8_prefix(output: *mut u8, mut values: uint8x8_t, count: usize) {
    debug_assert!((1..=8).contains(&count));
    if count == 8 {
        // SAFETY: the caller promises a complete eight-byte output region.
        unsafe { vst1_u8(output, values) };
        return;
    }

    let mut written = 0;
    let mut remaining = count;
    if remaining >= 4 {
        let packed = vget_lane_u32::<0>(vreinterpret_u32_u8(values));
        // SAFETY: four bytes remain in the caller-provided output prefix;
        // unaligned storage is permitted by the compact tile layout.
        unsafe { output.add(written).cast::<u32>().write_unaligned(packed) };
        values = vext_u8::<4>(values, values);
        written += 4;
        remaining -= 4;
    }
    if remaining >= 2 {
        let packed = vget_lane_u16::<0>(vreinterpret_u16_u8(values));
        // SAFETY: two bytes remain in the caller-provided output prefix;
        // unaligned storage is permitted by the compact tile layout.
        unsafe { output.add(written).cast::<u16>().write_unaligned(packed) };
        values = vext_u8::<2>(values, values);
        written += 2;
        remaining -= 2;
    }
    if remaining != 0 {
        // SAFETY: one byte remains in the caller-provided output prefix.
        unsafe { output.add(written).write(vget_lane_u8::<0>(values)) };
    }
}

/// Store the real-pixel prefix of an aligned compact luma tail.
///
/// # Safety
/// `output` must reserve the complete compact `width` by `height` plane, and
/// `x..x+count` on row `y` must be the final real source columns.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_compact_luma_tail(
    output: *mut u8,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    values: uint8x16_t,
    count: usize,
) {
    debug_assert_eq!(x % LANES, 0);
    debug_assert!((1..LANES).contains(&count));
    debug_assert_eq!(x + count, width);
    debug_assert!(y < height);

    let low_count = count.min(8);
    let low_offset = super::compact_plane_sample_offset(width, height, x, y);
    // SAFETY: the first prefix stays within one compact tile row.
    unsafe { store_u8_prefix(output.add(low_offset), vget_low_u8(values), low_count) };
    if count > 8 {
        let high_offset = super::compact_plane_sample_offset(width, height, x + 8, y);
        // SAFETY: the second prefix stays within the final compact tile row.
        unsafe { store_u8_prefix(output.add(high_offset), vget_high_u8(values), count - 8) };
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "compact subsampled conversion carries source geometry and three output planes"
)]
pub(super) fn rgb_to_ycbcr_compact_subsampled_planes(
    neon: fearless_simd::Neon,
    data: &[u8],
    width: usize,
    height: usize,
    layout: PrecomputedChromaLayout,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    assert_eq!(layout.x_scale, 2);
    assert!(matches!(layout.y_scale, 1 | 2));
    assert_eq!(layout.width % 8, 0);
    assert_eq!(layout.height % 8, 0);
    assert!(layout.width >= width.div_ceil(layout.x_scale));
    assert!(layout.height >= height.div_ceil(layout.y_scale));
    rgb_to_ycbcr_compact_subsampled_planes_kernel(
        neon,
        data,
        width,
        height,
        layout.width,
        layout.height,
        layout.y_scale,
        y_plane,
        cb_plane,
        cr_plane,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "the fused conversion owns three output planes and their validated geometry"
)]
pub(super) fn rgb_to_ycbcr_subsampled_planes(
    neon: fearless_simd::Neon,
    data: &[u8],
    width: usize,
    height: usize,
    layout: PrecomputedChromaLayout,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    assert_eq!(layout.x_scale, 2);
    assert!(matches!(layout.y_scale, 1 | 2));
    assert!(layout.width >= width.div_ceil(layout.x_scale));
    assert!(layout.height >= height.div_ceil(layout.y_scale));
    rgb_to_ycbcr_subsampled_planes_kernel(
        neon,
        data,
        width,
        height,
        layout.width,
        layout.height,
        layout.y_scale,
        y_plane,
        cb_plane,
        cr_plane,
    );
}

fearless_simd::kernel! {
    fn rgb_to_ycbcr_planes_kernel(
        neon: Neon,
        data: &[u8],
        y_plane: &mut Vec<u8>,
        cb_plane: &mut Vec<u8>,
        cr_plane: &mut Vec<u8>,
    ) {
        rgb_to_ycbcr_planes_neon(data, y_plane, cb_plane, cr_plane);
    }
}

fearless_simd::kernel! {
    fn rgb_to_ycbcr_compact_block_planes_kernel(
        neon: Neon,
        data: &[u8],
        width: usize,
        height: usize,
        y_plane: &mut Vec<u8>,
        cb_plane: &mut Vec<u8>,
        cr_plane: &mut Vec<u8>,
    ) {
        rgb_to_ycbcr_compact_block_planes_neon(
            data, width, height, y_plane, cb_plane, cr_plane,
        );
    }
}

fearless_simd::kernel! {
    #[expect(
        clippy::too_many_arguments,
        reason = "the fused compact kernel carries validated source geometry and three output planes"
    )]
    fn rgb_to_ycbcr_compact_subsampled_planes_kernel(
        neon: Neon,
        data: &[u8],
        width: usize,
        height: usize,
        chroma_width: usize,
        chroma_height: usize,
        y_scale: usize,
        y_plane: &mut Vec<u8>,
        cb_plane: &mut Vec<u8>,
        cr_plane: &mut Vec<u8>,
    ) {
        rgb_to_ycbcr_compact_subsampled_planes_neon(
            data,
            width,
            height,
            chroma_width,
            chroma_height,
            y_scale,
            y_plane,
            cb_plane,
            cr_plane,
        );
    }
}

fearless_simd::kernel! {
    fn sample_422_block_kernel(
        neon: Neon,
        source: &[u8],
        width: usize,
        out: &mut [u8; 64],
    ) {
        sample_422_block_neon(source, width, out);
    }
}

fearless_simd::kernel! {
    fn rgb_to_ycbcr_block_planes_kernel(
        neon: Neon,
        data: &[u8],
        width: usize,
        y_plane: &mut Vec<u8>,
        cb_plane: &mut Vec<u8>,
        cr_plane: &mut Vec<u8>,
    ) {
        rgb_to_ycbcr_block_planes_neon(data, width, y_plane, cb_plane, cr_plane);
    }
}

fearless_simd::kernel! {
    fn rgb_to_ycbcr_422_block_planes_kernel(
        neon: Neon,
        data: &[u8],
        width: usize,
        y_plane: &mut Vec<u8>,
        cb_plane: &mut Vec<u8>,
        cr_plane: &mut Vec<u8>,
    ) {
        rgb_to_ycbcr_422_block_planes_neon(data, width, y_plane, cb_plane, cr_plane);
    }
}

fearless_simd::kernel! {
    fn rgb_to_ycbcr_420_block_planes_kernel(
        neon: Neon,
        data: &[u8],
        width: usize,
        y_plane: &mut Vec<u8>,
        cb_plane: &mut Vec<u8>,
        cr_plane: &mut Vec<u8>,
    ) {
        rgb_to_ycbcr_420_block_planes_neon(data, width, y_plane, cb_plane, cr_plane);
    }
}

fearless_simd::kernel! {
    #[expect(
        clippy::too_many_arguments,
        reason = "the fused subsampled kernel carries validated source geometry and three output planes"
    )]
    fn rgb_to_ycbcr_subsampled_planes_kernel(
        neon: Neon,
        data: &[u8],
        width: usize,
        height: usize,
        chroma_width: usize,
        chroma_height: usize,
        y_scale: usize,
        y_plane: &mut Vec<u8>,
        cb_plane: &mut Vec<u8>,
        cr_plane: &mut Vec<u8>,
    ) {
        rgb_to_ycbcr_subsampled_planes_neon(
            data,
            width,
            height,
            chroma_width,
            chroma_height,
            y_scale,
            y_plane,
            cb_plane,
            cr_plane,
        );
    }
}

#[target_feature(enable = "neon")]
fn sample_422_block_neon(source: &[u8], width: usize, out: &mut [u8; 64]) {
    for y in 0..8 {
        // SAFETY: the safe wrapper validates one contiguous span that covers
        // each complete sixteen-byte source row.
        let values = unsafe { vld1q_u8(source.as_ptr().add(y * width)) };
        // SAFETY: each store initializes one distinct eight-byte output row.
        unsafe { vst1_u8(out.as_mut_ptr().add(y * 8), average_chroma_422(values)) };
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the fused conversion owns three output planes and their validated geometry"
)]
#[target_feature(enable = "neon")]
fn rgb_to_ycbcr_subsampled_planes_neon(
    data: &[u8],
    width: usize,
    height: usize,
    chroma_width: usize,
    chroma_height: usize,
    y_scale: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    let y_required = width * height;
    let chroma_required = chroma_width * chroma_height;
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    assert!(y_plane.spare_capacity_mut().len() >= y_required);
    assert!(cb_plane.spare_capacity_mut().len() >= chroma_required);
    assert!(cr_plane.spare_capacity_mut().len() >= chroma_required);
    let y_out = y_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let cb_out = cb_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let cr_out = cr_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let logical_chroma_height = height.div_ceil(y_scale);
    for cy in 0..logical_chroma_height {
        let top_y = cy * y_scale;
        let bottom_y = (top_y + y_scale - 1).min(height - 1);
        match y_scale {
            // SAFETY: plane construction reserved every output byte once.
            1 => unsafe {
                convert_subsampled_row::<1>(
                    data,
                    width,
                    top_y,
                    bottom_y,
                    chroma_width,
                    true,
                    y_out,
                    cb_out.add(cy * chroma_width),
                    cr_out.add(cy * chroma_width),
                );
            },
            // SAFETY: plane construction reserved every output byte once.
            2 => unsafe {
                convert_subsampled_row::<2>(
                    data,
                    width,
                    top_y,
                    bottom_y,
                    chroma_width,
                    true,
                    y_out,
                    cb_out.add(cy * chroma_width),
                    cr_out.add(cy * chroma_width),
                );
            },
            _ => unreachable!("canonical chroma vertical scale is one or two"),
        }
    }
    if logical_chroma_height < chroma_height {
        let edge = logical_chroma_height * chroma_width;
        match y_scale {
            // SAFETY: the padded edge row is within the reserved chroma plane.
            1 => unsafe {
                convert_subsampled_row::<1>(
                    data,
                    width,
                    height - 1,
                    height - 1,
                    chroma_width,
                    false,
                    y_out,
                    cb_out.add(edge),
                    cr_out.add(edge),
                );
            },
            // SAFETY: the padded edge row is within the reserved chroma plane.
            2 => unsafe {
                convert_subsampled_row::<2>(
                    data,
                    width,
                    height - 1,
                    height - 1,
                    chroma_width,
                    false,
                    y_out,
                    cb_out.add(edge),
                    cr_out.add(edge),
                );
            },
            _ => unreachable!("canonical chroma vertical scale is one or two"),
        }
        for cy in logical_chroma_height + 1..chroma_height {
            // SAFETY: the source edge row is initialized, and each destination
            // is a distinct writable row in the reserved chroma allocations.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    cb_out.add(edge),
                    cb_out.add(cy * chroma_width),
                    chroma_width,
                );
                core::ptr::copy_nonoverlapping(
                    cr_out.add(edge),
                    cr_out.add(cy * chroma_width),
                    chroma_width,
                );
            }
        }
    }
    // SAFETY: the row conversions and copies initialize every exposed byte.
    unsafe {
        y_plane.set_len(y_len + y_required);
        cb_plane.set_len(cb_len + chroma_required);
        cr_plane.set_len(cr_len + chroma_required);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
    reason = "one row-pair conversion carries three pointers and emits bounded floor averages of u8 chroma samples"
)]
#[target_feature(enable = "neon")]
unsafe fn convert_subsampled_row<const Y_SCALE: usize>(
    data: &[u8],
    width: usize,
    top_y: usize,
    bottom_y: usize,
    chroma_width: usize,
    write_y: bool,
    y_out: *mut u8,
    cb_out: *mut u8,
    cr_out: *mut u8,
) {
    let chunks = width / LANES;
    for chunk in 0..chunks {
        let x = chunk * LANES;
        let top = data[top_y * width * 3 + x * 3..]
            .first_chunk::<RGB_BYTES>()
            .expect("complete RGB vector row");
        let (top_y_values, top_cb, top_cr) = convert_16(top);
        let (cb, cr) = if Y_SCALE == 2 {
            let bottom = data[bottom_y * width * 3 + x * 3..]
                .first_chunk::<RGB_BYTES>()
                .expect("complete RGB vector row");
            let (bottom_y_values, bottom_cb, bottom_cr) = convert_16(bottom);
            if write_y && bottom_y != top_y {
                // SAFETY: the caller reserves the complete Y plane.
                unsafe { vst1q_u8(y_out.add(bottom_y * width + x), bottom_y_values) };
            }
            (
                average_chroma_420(top_cb, bottom_cb),
                average_chroma_420(top_cr, bottom_cr),
            )
        } else {
            (average_chroma_422(top_cb), average_chroma_422(top_cr))
        };
        if write_y {
            // SAFETY: the caller reserves the complete Y plane.
            unsafe { vst1q_u8(y_out.add(top_y * width + x), top_y_values) };
        }
        // SAFETY: each chunk initializes eight distinct chroma samples.
        unsafe {
            vst1_u8(cb_out.add(chunk * 8), cb);
            vst1_u8(cr_out.add(chunk * 8), cr);
        }
    }

    let logical_chroma_width = width.div_ceil(2);
    for cx in chunks * 8..chroma_width {
        let left_x = (cx * 2).min(width - 1);
        let right_x = (left_x + 1).min(width - 1);
        let top_left = super::rgb_at(data, width, left_x, top_y);
        let top_right = super::rgb_at(data, width, right_x, top_y);
        let (y00, cb00, cr00) = super::rgb_to_ycbcr(top_left[0], top_left[1], top_left[2]);
        let (y01, cb01, cr01) = super::rgb_to_ycbcr(top_right[0], top_right[1], top_right[2]);
        let (cb_sum, cr_sum, count) = if Y_SCALE == 2 {
            let bottom_left = super::rgb_at(data, width, left_x, bottom_y);
            let bottom_right = super::rgb_at(data, width, right_x, bottom_y);
            let (y10, cb10, cr10) =
                super::rgb_to_ycbcr(bottom_left[0], bottom_left[1], bottom_left[2]);
            let (y11, cb11, cr11) =
                super::rgb_to_ycbcr(bottom_right[0], bottom_right[1], bottom_right[2]);
            if write_y && cx < logical_chroma_width && bottom_y != top_y {
                // SAFETY: logical tail coordinates are within the Y plane.
                unsafe {
                    y_out.add(bottom_y * width + left_x).write(y10);
                    if right_x != left_x {
                        y_out.add(bottom_y * width + right_x).write(y11);
                    }
                }
            }
            (
                u16::from(cb00) + u16::from(cb01) + u16::from(cb10) + u16::from(cb11),
                u16::from(cr00) + u16::from(cr01) + u16::from(cr10) + u16::from(cr11),
                4,
            )
        } else {
            (
                u16::from(cb00) + u16::from(cb01),
                u16::from(cr00) + u16::from(cr01),
                2,
            )
        };
        if write_y && cx < logical_chroma_width {
            // SAFETY: logical tail coordinates are within the Y plane.
            unsafe {
                y_out.add(top_y * width + left_x).write(y00);
                if right_x != left_x {
                    y_out.add(top_y * width + right_x).write(y01);
                }
            }
        }
        // SAFETY: cx is within the reserved chroma row.
        unsafe {
            cb_out.add(cx).write((cb_sum / count) as u8);
            cr_out.add(cx).write((cr_sum / count) as u8);
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn average_chroma_422(values: uint8x16_t) -> uint8x8_t {
    vshrn_n_u16::<1>(vpaddlq_u8(values))
}

#[inline]
#[target_feature(enable = "neon")]
fn average_chroma_420(top: uint8x16_t, bottom: uint8x16_t) -> uint8x8_t {
    vshrn_n_u16::<2>(vaddq_u16(vpaddlq_u8(top), vpaddlq_u8(bottom)))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_compact_16(
    output: *mut u8,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    values: uint8x16_t,
) {
    debug_assert_eq!(x % 16, 0);
    debug_assert!(x + 16 <= width && y < height);
    if y < height / 8 * 8 {
        let first = y / 8 * 8 * width + x / 8 * 64 + y % 8 * 8;
        // SAFETY: two adjacent complete compact tiles own these distinct rows.
        unsafe {
            vst1_u8(output.add(first), vget_low_u8(values));
            vst1_u8(output.add(first + 64), vget_high_u8(values));
        }
    } else {
        // SAFETY: incomplete bottom rows remain contiguous and row-major.
        unsafe { vst1q_u8(output.add(y * width + x), values) };
    }
}

#[target_feature(enable = "neon")]
fn rgb_to_ycbcr_compact_block_planes_neon(
    data: &[u8],
    width: usize,
    height: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    let required = width
        .checked_mul(height)
        .expect("compact 4:4:4 plane length fits usize");
    assert_eq!(data.len(), required * 3);
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    assert!(y_plane.spare_capacity_mut().len() >= required);
    assert!(cb_plane.spare_capacity_mut().len() >= required);
    assert!(cr_plane.spare_capacity_mut().len() >= required);
    let y_out = y_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let cb_out = cb_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let cr_out = cr_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let chunks = width / LANES;
    for y in 0..height {
        for chunk in 0..chunks {
            let x = chunk * LANES;
            let rgb = data[y * width * 3 + x * 3..]
                .first_chunk::<RGB_BYTES>()
                .expect("complete compact RGB vector row");
            let (luma, cb, cr) = convert_16(rgb);
            // SAFETY: the capacity checks cover the exact compact planes, and
            // each source row/chunk initializes two unique tile rows.
            unsafe {
                store_compact_16(y_out, width, height, x, y, luma);
                store_compact_16(cb_out, width, height, x, y, cb);
                store_compact_16(cr_out, width, height, x, y, cr);
            }
        }
        for x in chunks * LANES..width {
            let rgb = super::rgb_at(data, width, x, y);
            let (luma, cb, cr) = super::rgb_to_ycbcr(rgb[0], rgb[1], rgb[2]);
            let offset = super::compact_plane_sample_offset(width, height, x, y);
            // SAFETY: tail pixels occupy unique positions after the vector tiles.
            unsafe {
                y_out.add(offset).write(luma);
                cb_out.add(offset).write(cb);
                cr_out.add(offset).write(cr);
            }
        }
    }
    // SAFETY: vector and scalar tail stores initialize every compact position.
    unsafe {
        y_plane.set_len(y_len + required);
        cb_plane.set_len(cb_len + required);
        cr_plane.set_len(cr_len + required);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::manual_is_multiple_of,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the measured compact subsampled kernel keeps explicit edge parity, bounded averages, and one linear dataflow"
)]
#[target_feature(enable = "neon")]
fn rgb_to_ycbcr_compact_subsampled_planes_neon(
    data: &[u8],
    width: usize,
    height: usize,
    chroma_width: usize,
    chroma_height: usize,
    y_scale: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    assert!(matches!(y_scale, 1 | 2));
    assert_eq!(chroma_width % 8, 0);
    assert_eq!(chroma_height % 8, 0);
    let y_required = width * height;
    let chroma_required = chroma_width * chroma_height;
    assert_eq!(data.len(), y_required * 3);
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    assert!(y_plane.spare_capacity_mut().len() >= y_required);
    assert!(cb_plane.spare_capacity_mut().len() >= chroma_required);
    assert!(cr_plane.spare_capacity_mut().len() >= chroma_required);
    let y_out = y_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let cb_out = cb_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let cr_out = cr_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let logical_chroma_width = width.div_ceil(2);
    let logical_chroma_height = height.div_ceil(y_scale);
    let chunks = width / LANES;
    let tail = width % LANES;
    // A lone final pixel needs only one clamped scalar pair; converting a
    // complete vector costs more than that small tail.
    let vector_tail = width >= LANES && tail >= 2;
    // An even width ends with one complete two-column average, while padded
    // columns repeat only the final source column. Compute that edge once.
    let distinct_edge_column = usize::from(width % 2 == 0 && logical_chroma_width < chroma_width);
    let computed_chroma_width = logical_chroma_width + distinct_edge_column;
    // An even-height 4:2:0 image ends with one complete two-row average, while
    // its padded rows repeat only the final source row. Compute that distinct
    // edge row once; every other padded row can reuse it.
    let distinct_edge_row =
        usize::from(y_scale == 2 && height % 2 == 0 && logical_chroma_height < chroma_height);
    let computed_chroma_height = logical_chroma_height + distinct_edge_row;
    for cy in 0..computed_chroma_height {
        let top_y = (cy * y_scale).min(height - 1);
        let bottom_y = (top_y + y_scale - 1).min(height - 1);
        let write_y = cy < logical_chroma_height;
        for chunk in 0..chunks {
            let x = chunk * LANES;
            let top = data[top_y * width * 3 + x * 3..]
                .first_chunk::<RGB_BYTES>()
                .expect("complete compact RGB vector row");
            let (top_luma, top_cb, top_cr) = convert_16(top);
            let (cb, cr) = if y_scale == 2 {
                let bottom = data[bottom_y * width * 3 + x * 3..]
                    .first_chunk::<RGB_BYTES>()
                    .expect("complete compact RGB vector row");
                let (bottom_luma, bottom_cb, bottom_cr) = convert_16(bottom);
                if write_y && bottom_y != top_y {
                    // SAFETY: every actual bottom-row pixel has one compact slot.
                    unsafe { store_compact_16(y_out, width, height, x, bottom_y, bottom_luma) };
                }
                (
                    average_chroma_420(top_cb, bottom_cb),
                    average_chroma_420(top_cr, bottom_cr),
                )
            } else {
                (average_chroma_422(top_cb), average_chroma_422(top_cr))
            };
            if write_y {
                // SAFETY: every actual top-row pixel has one compact slot.
                unsafe { store_compact_16(y_out, width, height, x, top_y, top_luma) };
            }
            let chroma_x = chunk * 8;
            let chroma_offset =
                super::block_plane_sample_offset(chroma_width, chroma_height, chroma_x, cy);
            // SAFETY: each vector chunk writes one distinct row in a chroma tile.
            unsafe {
                vst1_u8(cb_out.add(chroma_offset), cb);
                vst1_u8(cr_out.add(chroma_offset), cr);
            }
        }
        if vector_tail {
            let input_x = width - LANES;
            let output_x = chunks * LANES;
            // SAFETY: `tail` is in 1..=15, so this fixed table row contains
            // sixteen initialized byte indices.
            let shuffle = unsafe { vld1q_u8(COMPACT_TAIL_SHUFFLE_INDICES[tail].as_ptr()) };
            let top = data[top_y * width * 3 + input_x * 3..]
                .first_chunk::<RGB_BYTES>()
                .expect("complete final compact RGB vector row");
            let (top_luma, top_cb, top_cr) = convert_16(top);
            let top_luma = vqtbl1q_u8(top_luma, shuffle);
            let top_cb = vqtbl1q_u8(top_cb, shuffle);
            let top_cr = vqtbl1q_u8(top_cr, shuffle);
            let (cb, cr) = if y_scale == 2 {
                let bottom = data[bottom_y * width * 3 + input_x * 3..]
                    .first_chunk::<RGB_BYTES>()
                    .expect("complete final compact RGB vector row");
                let (bottom_luma, bottom_cb, bottom_cr) = convert_16(bottom);
                let bottom_luma = vqtbl1q_u8(bottom_luma, shuffle);
                let bottom_cb = vqtbl1q_u8(bottom_cb, shuffle);
                let bottom_cr = vqtbl1q_u8(bottom_cr, shuffle);
                if write_y && bottom_y != top_y {
                    // SAFETY: only the real final-row pixels are stored.
                    unsafe {
                        store_compact_luma_tail(
                            y_out,
                            width,
                            height,
                            output_x,
                            bottom_y,
                            bottom_luma,
                            tail,
                        );
                    };
                }
                (
                    average_chroma_420(top_cb, bottom_cb),
                    average_chroma_420(top_cr, bottom_cr),
                )
            } else {
                (average_chroma_422(top_cb), average_chroma_422(top_cr))
            };
            if write_y {
                // SAFETY: only the real final-row pixels are stored.
                unsafe {
                    store_compact_luma_tail(y_out, width, height, output_x, top_y, top_luma, tail);
                };
            }
            let chroma_x = chunks * 8;
            let chroma_offset =
                super::block_plane_sample_offset(chroma_width, chroma_height, chroma_x, cy);
            // SAFETY: the clamped vector fills the complete final chroma tile
            // row, including the retained horizontal edge padding.
            unsafe {
                vst1_u8(cb_out.add(chroma_offset), cb);
                vst1_u8(cr_out.add(chroma_offset), cr);
            }
            continue;
        }
        for cx in chunks * 8..computed_chroma_width {
            let left_x = (cx * 2).min(width - 1);
            let right_x = (left_x + 1).min(width - 1);
            let top_left = super::rgb_at(data, width, left_x, top_y);
            let top_right = super::rgb_at(data, width, right_x, top_y);
            let (y00, cb00, cr00) = super::rgb_to_ycbcr(top_left[0], top_left[1], top_left[2]);
            let (y01, cb01, cr01) = super::rgb_to_ycbcr(top_right[0], top_right[1], top_right[2]);
            let (cb_sum, cr_sum, count) = if y_scale == 2 {
                let bottom_left = super::rgb_at(data, width, left_x, bottom_y);
                let bottom_right = super::rgb_at(data, width, right_x, bottom_y);
                let (y10, cb10, cr10) =
                    super::rgb_to_ycbcr(bottom_left[0], bottom_left[1], bottom_left[2]);
                let (y11, cb11, cr11) =
                    super::rgb_to_ycbcr(bottom_right[0], bottom_right[1], bottom_right[2]);
                if write_y && cx < logical_chroma_width && bottom_y != top_y {
                    // SAFETY: scalar tail coordinates are real source pixels.
                    unsafe {
                        y_out
                            .add(super::compact_plane_sample_offset(
                                width, height, left_x, bottom_y,
                            ))
                            .write(y10);
                        if right_x != left_x {
                            y_out
                                .add(super::compact_plane_sample_offset(
                                    width, height, right_x, bottom_y,
                                ))
                                .write(y11);
                        }
                    }
                }
                (
                    u16::from(cb00) + u16::from(cb01) + u16::from(cb10) + u16::from(cb11),
                    u16::from(cr00) + u16::from(cr01) + u16::from(cr10) + u16::from(cr11),
                    4,
                )
            } else {
                (
                    u16::from(cb00) + u16::from(cb01),
                    u16::from(cr00) + u16::from(cr01),
                    2,
                )
            };
            if write_y && cx < logical_chroma_width {
                // SAFETY: scalar tail coordinates are real source pixels.
                unsafe {
                    y_out
                        .add(super::compact_plane_sample_offset(
                            width, height, left_x, top_y,
                        ))
                        .write(y00);
                    if right_x != left_x {
                        y_out
                            .add(super::compact_plane_sample_offset(
                                width, height, right_x, top_y,
                            ))
                            .write(y01);
                    }
                }
            }
            let chroma_offset =
                super::block_plane_sample_offset(chroma_width, chroma_height, cx, cy);
            // SAFETY: every computed scalar chroma coordinate has one reserved tile slot.
            unsafe {
                cb_out.add(chroma_offset).write((cb_sum / count) as u8);
                cr_out.add(chroma_offset).write((cr_sum / count) as u8);
            }
        }
        if computed_chroma_width < chroma_width {
            let edge_offset = super::block_plane_sample_offset(
                chroma_width,
                chroma_height,
                computed_chroma_width - 1,
                cy,
            );
            let padding_offset = super::block_plane_sample_offset(
                chroma_width,
                chroma_height,
                computed_chroma_width,
                cy,
            );
            let padding = chroma_width - computed_chroma_width;
            debug_assert!(padding < 8);
            // SAFETY: horizontal padding remains in the final tile row. Its
            // clamped samples equal the initialized final computed edge.
            unsafe {
                let edge_cb = cb_out.add(edge_offset).read();
                let edge_cr = cr_out.add(edge_offset).read();
                fill_compact_chroma_padding(cb_out.add(padding_offset), edge_cb, padding);
                fill_compact_chroma_padding(cr_out.add(padding_offset), edge_cr, padding);
            }
        }
    }
    if computed_chroma_height < chroma_height {
        let source_y = computed_chroma_height - 1;
        for cy in computed_chroma_height..chroma_height {
            for chroma_x in (0..chroma_width).step_by(8) {
                let source_offset = super::block_plane_sample_offset(
                    chroma_width,
                    chroma_height,
                    chroma_x,
                    source_y,
                );
                let destination_offset =
                    super::block_plane_sample_offset(chroma_width, chroma_height, chroma_x, cy);
                // SAFETY: the source is an initialized eight-byte chroma tile
                // row, and the distinct destination row is within reserved capacity.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        cb_out.add(source_offset),
                        cb_out.add(destination_offset),
                        8,
                    );
                    core::ptr::copy_nonoverlapping(
                        cr_out.add(source_offset),
                        cr_out.add(destination_offset),
                        8,
                    );
                }
            }
        }
    }
    // SAFETY: logical conversion plus horizontal fills and tile-row copies
    // initialize each real compact Y sample and every padded chroma sample.
    unsafe {
        y_plane.set_len(y_len + y_required);
        cb_plane.set_len(cb_len + chroma_required);
        cr_plane.set_len(cr_len + chroma_required);
    }
}

#[target_feature(enable = "neon")]
fn rgb_to_ycbcr_planes_neon(
    data: &[u8],
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    let (chunks, tail) = data.as_chunks::<RGB_BYTES>();
    let initialized = chunks.len() * LANES;
    let required = initialized + tail.as_chunks::<3>().0.len();
    // Component-plane construction reserves the complete image before this
    // kernel. Validate the available storage once, before writing any bytes.
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    let y_spare = y_plane.spare_capacity_mut();
    let cb_spare = cb_plane.spare_capacity_mut();
    let cr_spare = cr_plane.spare_capacity_mut();
    assert!(y_spare.len() >= required);
    assert!(cb_spare.len() >= required);
    assert!(cr_spare.len() >= required);
    for (index, rgb) in chunks.iter().enumerate() {
        let (y, cb, cr) = convert_16(rgb);
        let offset = index * LANES;
        // SAFETY: every complete input chunk produces sixteen output bytes.
        // The capacity checks above cover all chunks, and these three exclusive
        // spare-capacity slices refer to distinct allocations. The stores
        // initialize every byte exposed by set_len below.
        unsafe {
            vst1q_u8(y_spare.as_mut_ptr().add(offset).cast(), y);
            vst1q_u8(cb_spare.as_mut_ptr().add(offset).cast(), cb);
            vst1q_u8(cr_spare.as_mut_ptr().add(offset).cast(), cr);
        }
    }
    // SAFETY: each byte in these prefixes was initialized by the stores above;
    // each old length plus initialized is bounded by its vector capacity.
    unsafe {
        y_plane.set_len(y_len + initialized);
        cb_plane.set_len(cb_len + initialized);
        cr_plane.set_len(cr_len + initialized);
    }
    for rgb in tail.as_chunks::<3>().0 {
        let (y, cb, cr) = super::rgb_to_ycbcr(rgb[0], rgb[1], rgb[2]);
        y_plane.push(y);
        cb_plane.push(cb);
        cr_plane.push(cr);
    }
}

#[target_feature(enable = "neon")]
fn rgb_to_ycbcr_block_planes_neon(
    data: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    assert_ne!(width, 0);
    assert_eq!(width % 16, 0);
    let row_bytes = width
        .checked_mul(3)
        .expect("4:4:4 block-plane row bytes fit usize");
    let block_row_bytes = row_bytes
        .checked_mul(8)
        .expect("4:4:4 block-plane row group fits usize");
    assert_eq!(data.len() % block_row_bytes, 0);
    let required = data.len() / 3;
    let blocks_per_row = width / 8;
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    assert!(y_plane.spare_capacity_mut().len() >= required);
    assert!(cb_plane.spare_capacity_mut().len() >= required);
    assert!(cr_plane.spare_capacity_mut().len() >= required);
    let y_out = y_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let cb_out = cb_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let cr_out = cr_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    for (block_row, rows) in data.chunks_exact(block_row_bytes).enumerate() {
        for (row_in_block, row) in rows.chunks_exact(row_bytes).enumerate() {
            for (pair_x, rgb) in row.as_chunks::<RGB_BYTES>().0.iter().enumerate() {
                let (y, cb, cr) = convert_16(rgb);
                let offset = (block_row * blocks_per_row + pair_x * 2) * 64 + row_in_block * 8;
                // SAFETY: aligned 4:4:4 geometry pairs adjacent complete
                // blocks. Each low/high store initializes a distinct tile row
                // within every reserved output plane.
                unsafe {
                    vst1_u8(y_out.add(offset), vget_low_u8(y));
                    vst1_u8(y_out.add(offset + 64), vget_high_u8(y));
                    vst1_u8(cb_out.add(offset), vget_low_u8(cb));
                    vst1_u8(cb_out.add(offset + 64), vget_high_u8(cb));
                    vst1_u8(cr_out.add(offset), vget_low_u8(cr));
                    vst1_u8(cr_out.add(offset + 64), vget_high_u8(cr));
                }
            }
        }
    }
    // SAFETY: every byte in each complete block-major plane was initialized
    // exactly once by the paired stores above.
    unsafe {
        y_plane.set_len(y_len + required);
        cb_plane.set_len(cb_len + required);
        cr_plane.set_len(cr_len + required);
    }
}

#[target_feature(enable = "neon")]
fn rgb_to_ycbcr_422_block_planes_neon(
    data: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    assert_ne!(width, 0);
    assert_eq!(width % 16, 0);
    let row_bytes = width
        .checked_mul(3)
        .expect("4:2:2 block-plane row bytes fit usize");
    let block_row_bytes = row_bytes
        .checked_mul(8)
        .expect("4:2:2 block-plane row group fits usize");
    assert_eq!(data.len() % block_row_bytes, 0);
    let y_required = data.len() / 3;
    let chroma_required = y_required / 2;
    let mcus_per_row = width / 16;
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    assert!(y_plane.spare_capacity_mut().len() >= y_required);
    assert!(cb_plane.spare_capacity_mut().len() >= chroma_required);
    assert!(cr_plane.spare_capacity_mut().len() >= chroma_required);
    let y_out = y_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let cb_out = cb_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let cr_out = cr_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    for (block_row, rows) in data.chunks_exact(block_row_bytes).enumerate() {
        for (row_in_block, row) in rows.chunks_exact(row_bytes).enumerate() {
            for (mcu_x, rgb) in row.as_chunks::<RGB_BYTES>().0.iter().enumerate() {
                let (y, cb, cr) = convert_16(rgb);
                let mcu = block_row * mcus_per_row + mcu_x;
                let y_offset = mcu * 128 + row_in_block * 8;
                let chroma_offset = mcu * 64 + row_in_block * 8;
                // SAFETY: each aligned 4:2:2 MCU owns two complete Y blocks
                // and one block in each chroma plane. These four stores write
                // one distinct row in those blocks.
                unsafe {
                    vst1_u8(y_out.add(y_offset), vget_low_u8(y));
                    vst1_u8(y_out.add(y_offset + 64), vget_high_u8(y));
                    vst1_u8(cb_out.add(chroma_offset), average_chroma_422(cb));
                    vst1_u8(cr_out.add(chroma_offset), average_chroma_422(cr));
                }
            }
        }
    }
    // SAFETY: every aligned MCU initializes its two Y blocks and both chroma
    // blocks completely, covering each reserved prefix exactly once.
    unsafe {
        y_plane.set_len(y_len + y_required);
        cb_plane.set_len(cb_len + chroma_required);
        cr_plane.set_len(cr_len + chroma_required);
    }
}

#[target_feature(enable = "neon")]
fn rgb_to_ycbcr_420_block_planes_neon(
    data: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    assert_ne!(width, 0);
    assert_eq!(width % 16, 0);
    let row_bytes = width
        .checked_mul(3)
        .expect("4:2:0 block-plane row bytes fit usize");
    let block_row_bytes = row_bytes
        .checked_mul(16)
        .expect("4:2:0 block-plane row group fits usize");
    assert_eq!(data.len() % block_row_bytes, 0);
    let y_required = data.len() / 3;
    let chroma_required = y_required / 4;
    let mcus_per_row = width / 16;
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    assert!(y_plane.spare_capacity_mut().len() >= y_required);
    assert!(cb_plane.spare_capacity_mut().len() >= chroma_required);
    assert!(cr_plane.spare_capacity_mut().len() >= chroma_required);
    let y_out = y_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let cb_out = cb_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let cr_out = cr_plane.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    for (block_row, rows) in data.chunks_exact(block_row_bytes).enumerate() {
        for (row_pair, rows) in rows.chunks_exact(row_bytes * 2).enumerate() {
            let (top, bottom) = rows.split_at(row_bytes);
            for (mcu_x, (top_rgb, bottom_rgb)) in top
                .as_chunks::<RGB_BYTES>()
                .0
                .iter()
                .zip(bottom.as_chunks::<RGB_BYTES>().0.iter())
                .enumerate()
            {
                let (top_y, top_cb, top_cr) = convert_16(top_rgb);
                let (bottom_y, bottom_cb, bottom_cr) = convert_16(bottom_rgb);
                let mcu = block_row * mcus_per_row + mcu_x;
                let y_offset = mcu * 256 + (row_pair / 4) * 128 + (row_pair % 4) * 16;
                let chroma_offset = mcu * 64 + row_pair * 8;
                // SAFETY: each aligned 4:2:0 MCU owns four complete Y blocks
                // and one block in each chroma plane. These six stores write
                // one distinct row in those blocks.
                unsafe {
                    vst1_u8(y_out.add(y_offset), vget_low_u8(top_y));
                    vst1_u8(y_out.add(y_offset + 64), vget_high_u8(top_y));
                    vst1_u8(y_out.add(y_offset + 8), vget_low_u8(bottom_y));
                    vst1_u8(y_out.add(y_offset + 72), vget_high_u8(bottom_y));
                    vst1_u8(
                        cb_out.add(chroma_offset),
                        average_chroma_420(top_cb, bottom_cb),
                    );
                    vst1_u8(
                        cr_out.add(chroma_offset),
                        average_chroma_420(top_cr, bottom_cr),
                    );
                }
            }
        }
    }
    // SAFETY: every aligned MCU initializes its four Y blocks and both chroma
    // blocks completely, covering each reserved prefix exactly once.
    unsafe {
        y_plane.set_len(y_len + y_required);
        cb_plane.set_len(cb_len + chroma_required);
        cr_plane.set_len(cr_len + chroma_required);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn convert_16(rgb: &[u8; RGB_BYTES]) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    // SAFETY: the fixed-size input proves 48 readable bytes, AArch64 vector
    // loads support unaligned addresses, and this runs inside a Neon kernel.
    let pixels = unsafe { vld3q_u8(rgb.as_ptr()) };
    let r_l = vmovl_u8(vget_low_u8(pixels.0));
    let r_h = vmovl_u8(vget_high_u8(pixels.0));
    let g_l = vmovl_u8(vget_low_u8(pixels.1));
    let g_h = vmovl_u8(vget_high_u8(pixels.1));
    let b_l = vmovl_u8(vget_low_u8(pixels.2));
    let b_h = vmovl_u8(vget_high_u8(pixels.2));

    let (y_l, cb_l, cr_l) = convert_8(r_l, g_l, b_l);
    let (y_h, cb_h, cr_h) = convert_8(r_h, g_h, b_h);
    (
        combine_u8(y_l, y_h),
        combine_u8(cb_l, cb_h),
        combine_u8(cr_l, cr_h),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn convert_8(r: uint16x8_t, g: uint16x8_t, b: uint16x8_t) -> (uint16x8_t, uint16x8_t, uint16x8_t) {
    let (y_l, cb_l, cr_l) = convert_4(vget_low_u16(r), vget_low_u16(g), vget_low_u16(b));
    let (y_h, cb_h, cr_h) = convert_4(vget_high_u16(r), vget_high_u16(g), vget_high_u16(b));
    (
        vcombine_u16(y_l, y_h),
        vcombine_u16(cb_l, cb_h),
        vcombine_u16(cr_l, cr_h),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn convert_4(r: uint16x4_t, g: uint16x4_t, b: uint16x4_t) -> (uint16x4_t, uint16x4_t, uint16x4_t) {
    // Differences stay in -255..=255, and all products and sums fit i32.
    // RSHRN adds 32768 before the signed shift, preserving the scalar rounding.
    // These factored expressions equal the original numerators: after adding
    // G or 128, Y is in 0..=255 and chroma in 0..=256. The existing unsigned
    // saturating narrow therefore still clamps exactly the chroma value 256.
    let r = vreinterpret_s16_u16(r);
    let g = vreinterpret_s16_u16(g);
    let b = vreinterpret_s16_u16(b);
    let red_difference = vsub_s16(r, g);
    let blue_difference = vsub_s16(b, g);

    let mut y = vmull_n_s16(red_difference, 19_595);
    y = vmlal_n_s16(y, blue_difference, 7_471);
    let y = vadd_s16(vrshrn_n_s32::<16>(y), g);

    let mut cb = vshll_n_s16::<15>(blue_difference);
    cb = vmlsl_n_s16(cb, red_difference, 11_059);
    let cb = vadd_s16(vrshrn_n_s32::<16>(cb), vdup_n_s16(128));

    let mut cr = vshll_n_s16::<15>(red_difference);
    cr = vmlsl_n_s16(cr, blue_difference, 5_329);
    let cr = vadd_s16(vrshrn_n_s32::<16>(cr), vdup_n_s16(128));

    (
        vreinterpret_u16_s16(y),
        vreinterpret_u16_s16(cb),
        vreinterpret_u16_s16(cr),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn combine_u8(low: uint16x8_t, high: uint16x8_t) -> uint8x16_t {
    vcombine_u8(vqmovn_u16(low), vqmovn_u16(high))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::baseline_encode_tables;
    use crate::encoder::{JpegBackend, JpegEncodeOptions, JpegSubsampling};

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one SIMD parity matrix covers complete vectors, tails, and every plane layout"
    )]
    fn neon_rgb_to_ycbcr_matches_scalar_across_chunks_and_tail() {
        let Some(neon) = fearless_simd::Level::new().as_neon() else {
            return;
        };
        for pixels in [1usize, 15, 16, 17, 31, 32, 33, 257] {
            let mut data = (0..pixels * 3)
                .map(|index| index.to_le_bytes()[0].wrapping_mul(73).wrapping_add(19))
                .collect::<Vec<_>>();
            let extremes = [
                [0, 0, 0],
                [255, 255, 255],
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ];
            for (pixel, extreme) in data.as_chunks_mut::<3>().0.iter_mut().zip(extremes) {
                *pixel = extreme;
            }
            let mut expected_y = Vec::new();
            let mut expected_cb = Vec::new();
            let mut expected_cr = Vec::new();
            super::super::fill_rgb_to_ycbcr_scalar(
                &data,
                &mut expected_y,
                &mut expected_cb,
                &mut expected_cr,
            );
            let mut actual_y = Vec::with_capacity(pixels);
            let mut actual_cb = Vec::with_capacity(pixels);
            let mut actual_cr = Vec::with_capacity(pixels);
            rgb_to_ycbcr_planes(neon, &data, &mut actual_y, &mut actual_cb, &mut actual_cr);
            assert_eq!(actual_y, expected_y, "Y pixels={pixels}");
            assert_eq!(actual_cb, expected_cb, "Cb pixels={pixels}");
            assert_eq!(actual_cr, expected_cr, "Cr pixels={pixels}");
        }

        for subsampling in [JpegSubsampling::Ybr422, JpegSubsampling::Ybr420] {
            let sampling = baseline_encode_tables(JpegEncodeOptions {
                quality: 90,
                subsampling,
                restart_interval: None,
                backend: JpegBackend::Cpu,
            })
            .unwrap()
            .sampling;
            for (width, height) in [
                (31usize, 18usize),
                (32, 16),
                (32usize, 18usize),
                (33, 18),
                (34, 17),
                (34, 18),
                (17, 18),
                (18, 17),
            ] {
                let Some(layout) = super::super::precomputed_chroma_layout(
                    u32::try_from(width).expect("test width fits u32"),
                    u32::try_from(height).expect("test height fits u32"),
                    sampling,
                    1,
                ) else {
                    continue;
                };
                let data = (0..width * height * 3)
                    .map(|index| index.to_le_bytes()[0].wrapping_mul(73).wrapping_add(19))
                    .collect::<Vec<_>>();
                let chroma_len = layout.len().expect("validated chroma layout");
                let mut expected_y = Vec::with_capacity(width * height);
                let mut expected_cb = Vec::with_capacity(chroma_len);
                let mut expected_cr = Vec::with_capacity(chroma_len);
                super::super::fill_rgb_to_ycbcr_subsampled_scalar(
                    &data,
                    width,
                    height,
                    layout,
                    &mut expected_y,
                    &mut expected_cb,
                    &mut expected_cr,
                );
                let mut actual_y = Vec::with_capacity(width * height);
                let mut actual_cb = Vec::with_capacity(chroma_len);
                let mut actual_cr = Vec::with_capacity(chroma_len);
                rgb_to_ycbcr_subsampled_planes(
                    neon,
                    &data,
                    width,
                    height,
                    layout,
                    &mut actual_y,
                    &mut actual_cb,
                    &mut actual_cr,
                );
                assert_eq!(actual_y, expected_y, "Y {subsampling:?} {width}x{height}");
                assert_eq!(
                    actual_cb, expected_cb,
                    "Cb {subsampling:?} {width}x{height}"
                );
                assert_eq!(
                    actual_cr, expected_cr,
                    "Cr {subsampling:?} {width}x{height}"
                );
            }
        }
    }
}
