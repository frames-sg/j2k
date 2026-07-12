// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{ResolvedPreparedComponentPlan, StripeBuffer};
use crate::backend::Backend;
use crate::entropy::block::{
    decode_block_for_1x1_idct, decode_block_for_reduced_idct, BlockActivity, CoefficientBlock,
    ReducedIdctCoefficients,
};
use crate::error::JpegError;
use crate::idct::downscale;
use crate::info::DownscaleFactor;
use crate::internal::bit_reader::BitReader;
use core::ptr;

/// Release-mode precondition for the raw-pointer writes in [`deposit_block`]
/// and [`deposit_dc_block`]: the stripe plane must carry the stride and
/// capacity [`StripeBuffer::resize_for`] establishes for
/// `stripe_mcus_per_row` MCUs of `h`×`v` `block_size`-pixel blocks. Asserted
/// once per stripe row so the per-block deposits stay bounds-check free in
/// the hot loop.
pub(super) fn assert_stripe_deposit_capacity(
    stripe: &StripeBuffer,
    plane_idx: usize,
    h: u32,
    v: u32,
    stripe_mcus_per_row: u32,
    block_size: u32,
) {
    let needed_cols = stripe_mcus_per_row as usize * h as usize * block_size as usize;
    let needed_rows = v as usize * block_size as usize;
    let stride = stripe.plane_strides[plane_idx];
    let len = stripe.planes[plane_idx].len();
    assert!(
        stride >= needed_cols
            && stride
                .checked_mul(needed_rows)
                .is_some_and(|needed| len >= needed),
        "stripe plane {plane_idx} ({len} bytes, stride {stride}) is not sized for \
         {stripe_mcus_per_row} MCUs of {h}x{v} blocks of {block_size}px; \
         StripeBuffer::resize_for must run first"
    );
}

pub(super) fn deposit_block(plane: &mut [u8], stride: usize, x: u32, y: u32, block: &[u8; 64]) {
    let dst_row_start = (y as usize) * stride + (x as usize);
    debug_assert!(x as usize + 8 <= stride);
    debug_assert!(plane.len() >= dst_row_start + stride.saturating_mul(7) + 8);
    // SAFETY: `plane`/`stride` come from a StripeBuffer sized by `resize_for`,
    // and the caller's per-stripe `assert_stripe_deposit_capacity` check
    // guarantees in release builds that every in-stripe (x, y) block start
    // leaves room for 8 rows of 8 bytes at this stride.
    // SAFETY: Destination pointers are derived from validated row starts and row widths.
    let mut dst = unsafe { plane.as_mut_ptr().add(dst_row_start) };
    let mut src = block.as_ptr();
    for _ in 0..8 {
        // SAFETY: Destination pointers are derived from validated row starts and row widths.
        unsafe {
            ptr::copy_nonoverlapping(src, dst, 8);
            dst = dst.add(stride);
            src = src.add(8);
        }
    }
}

pub(super) fn deposit_dc_block(plane: &mut [u8], stride: usize, x: u32, y: u32, pixel: u8) {
    let dst_row_start = (y as usize) * stride + (x as usize);
    debug_assert!(x as usize + 8 <= stride);
    debug_assert!(plane.len() >= dst_row_start + stride.saturating_mul(7) + 8);
    // SAFETY: same contract as `deposit_block` — upheld by `resize_for` plus
    // the caller's per-stripe `assert_stripe_deposit_capacity` check.
    // SAFETY: Destination pointers are derived from validated row starts and row widths.
    let mut dst = unsafe { plane.as_mut_ptr().add(dst_row_start) };
    for _ in 0..8 {
        // SAFETY: Destination pointers are derived from validated row starts and row widths.
        unsafe {
            ptr::write_bytes(dst, pixel, 8);
            dst = dst.add(stride);
        }
    }
}

#[inline]
#[expect(
    clippy::needless_pass_by_value,
    reason = "compact activity, backend, and plane-target descriptors stay register-passed on the IDCT hot path"
)]
pub(super) fn idct_deposit_fast_tile_block(
    activity: BlockActivity,
    backend: Backend,
    coeff: &CoefficientBlock,
    pixels: &mut [u8; 64],
    target: PlaneBlockTarget<'_>,
) {
    match activity {
        BlockActivity::DcOnly => {
            let pixel = crate::idct::idct_islow_dc_only_pixel(coeff.dc_coeff());
            deposit_dc_block(target.plane, target.stride, target.x, target.y, pixel);
        }
        BlockActivity::BottomHalfZero => {
            backend.idct_bottom_half_zero(coeff.coefficients(), pixels);
            deposit_block(target.plane, target.stride, target.x, target.y, pixels);
        }
        BlockActivity::General => {
            backend.idct(coeff.coefficients(), pixels);
            deposit_block(target.plane, target.stride, target.x, target.y, pixels);
        }
    }
}

pub(super) fn deposit_block_4x4(plane: &mut [u8], stride: usize, x: u32, y: u32, block: &[u8; 16]) {
    let x = x as usize;
    let y = y as usize;
    for by in 0..4 {
        let dst_start = (y + by) * stride + x;
        plane[dst_start..dst_start + 4].copy_from_slice(&block[by * 4..by * 4 + 4]);
    }
}

pub(super) fn deposit_block_2x2(plane: &mut [u8], stride: usize, x: u32, y: u32, block: [u8; 4]) {
    let x = x as usize;
    let y = y as usize;
    let top = y * stride + x;
    let bottom = top + stride;
    plane[top] = block[0];
    plane[top + 1] = block[1];
    plane[bottom] = block[2];
    plane[bottom + 1] = block[3];
}

pub(super) fn deposit_block_1x1(plane: &mut [u8], stride: usize, x: u32, y: u32, pixel: u8) {
    let dst = (y as usize) * stride + (x as usize);
    plane[dst] = pixel;
}

pub(super) struct EntropyBlockState<'a, 'b> {
    pub(super) br: &'a mut BitReader<'b>,
    pub(super) prev_dc: &'a mut i32,
    pub(super) coeff: &'a mut CoefficientBlock,
}

pub(super) struct ReducedIdctScratch<'a> {
    pub(super) pixels_4x4: &'a mut [u8; 16],
    pub(super) pixels_2x2: &'a mut [u8; 4],
}

pub(super) struct PlaneBlockTarget<'a> {
    pub(super) plane: &'a mut [u8],
    pub(super) stride: usize,
    pub(super) x: u32,
    pub(super) y: u32,
}

#[derive(Clone, Copy)]
pub(super) struct FastTile420Components<'a> {
    pub(super) y: ResolvedPreparedComponentPlan<'a>,
    pub(super) cb: ResolvedPreparedComponentPlan<'a>,
    pub(super) cr: ResolvedPreparedComponentPlan<'a>,
}

pub(super) struct FastTile420DcState<'a> {
    pub(super) y: &'a mut i32,
    pub(super) cb: &'a mut i32,
    pub(super) cr: &'a mut i32,
}

pub(super) struct FastTile420EntropyState<'a, 'b> {
    pub(super) br: &'a mut BitReader<'b>,
    pub(super) dc: FastTile420DcState<'a>,
    pub(super) coeff: &'a mut CoefficientBlock,
}

#[derive(Clone, Copy)]
pub(super) struct FastTile420Window {
    pub(super) mcus_per_row: u32,
    pub(super) stripe_mcu_start: u32,
    pub(super) stripe_mcus_per_row: u32,
}

impl FastTile420Window {
    #[inline]
    pub(super) fn stripe_mcu_end(self) -> u32 {
        self.stripe_mcu_start + self.stripe_mcus_per_row
    }

    #[inline]
    pub(super) fn contains_mcu(self, mx: u32) -> bool {
        mx >= self.stripe_mcu_start && mx < self.stripe_mcu_end()
    }

    #[inline]
    pub(super) fn local_mcu_x(self, mx: u32) -> u32 {
        mx - self.stripe_mcu_start
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "entropy state, scratch, and plane target are compact borrowing descriptors consumed as one hot-path operation"
)]
pub(super) fn decode_scaled_block_to_plane(
    comp: ResolvedPreparedComponentPlan<'_>,
    downscale: DownscaleFactor,
    state: EntropyBlockState<'_, '_>,
    scratch: ReducedIdctScratch<'_>,
    target: PlaneBlockTarget<'_>,
) -> Result<(), JpegError> {
    let keep = match downscale {
        DownscaleFactor::Full => unreachable!("scaled block path excludes full-size decode"),
        DownscaleFactor::Half => ReducedIdctCoefficients::Half,
        DownscaleFactor::Quarter => ReducedIdctCoefficients::Quarter,
        DownscaleFactor::Eighth => {
            decode_block_for_1x1_idct(
                state.br,
                comp.dc_table,
                comp.ac_table,
                state.prev_dc,
                comp.quant,
                state.coeff,
            )?;
            let pixel = downscale::idct_islow_1x1(state.coeff.coefficients());
            deposit_block_1x1(target.plane, target.stride, target.x, target.y, pixel);
            return Ok(());
        }
    };
    let dc_only = decode_block_for_reduced_idct(
        state.br,
        comp.dc_table,
        comp.ac_table,
        state.prev_dc,
        comp.quant,
        state.coeff,
        keep,
    )?;
    match downscale {
        DownscaleFactor::Full => unreachable!("scaled block path excludes full-size decode"),
        DownscaleFactor::Half => {
            if dc_only {
                downscale::idct_islow_4x4_dc_only(state.coeff.dc_coeff(), scratch.pixels_4x4);
            } else {
                downscale::idct_islow_4x4(state.coeff.coefficients(), scratch.pixels_4x4);
            }
            deposit_block_4x4(
                target.plane,
                target.stride,
                target.x,
                target.y,
                scratch.pixels_4x4,
            );
        }
        DownscaleFactor::Quarter => {
            if dc_only {
                downscale::idct_islow_2x2_dc_only(state.coeff.dc_coeff(), scratch.pixels_2x2);
            } else {
                downscale::idct_islow_2x2(state.coeff.coefficients(), scratch.pixels_2x2);
            }
            deposit_block_2x2(
                target.plane,
                target.stride,
                target.x,
                target.y,
                *scratch.pixels_2x2,
            );
        }
        DownscaleFactor::Eighth => {
            let pixel = downscale::idct_islow_1x1(state.coeff.coefficients());
            deposit_block_1x1(target.plane, target.stride, target.x, target.y, pixel);
        }
    }
    Ok(())
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "entropy state and plane target are compact borrowing descriptors used as one reduced-IDCT operation"
)]
pub(super) fn decode_quarter_block_to_plane(
    comp: ResolvedPreparedComponentPlan<'_>,
    pixels_2x2: &mut [u8; 4],
    state: EntropyBlockState<'_, '_>,
    target: PlaneBlockTarget<'_>,
) -> Result<(), JpegError> {
    let dc_only = decode_block_for_reduced_idct(
        state.br,
        comp.dc_table,
        comp.ac_table,
        state.prev_dc,
        comp.quant,
        state.coeff,
        ReducedIdctCoefficients::Quarter,
    )?;
    if dc_only {
        downscale::idct_islow_2x2_dc_only(state.coeff.dc_coeff(), pixels_2x2);
    } else {
        downscale::idct_islow_2x2(state.coeff.coefficients(), pixels_2x2);
    }
    deposit_block_2x2(target.plane, target.stride, target.x, target.y, *pixels_2x2);
    Ok(())
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "entropy state and plane target are compact borrowing descriptors used as one reduced-IDCT operation"
)]
pub(super) fn decode_eighth_block_to_plane(
    comp: ResolvedPreparedComponentPlan<'_>,
    state: EntropyBlockState<'_, '_>,
    target: PlaneBlockTarget<'_>,
) -> Result<(), JpegError> {
    decode_block_for_1x1_idct(
        state.br,
        comp.dc_table,
        comp.ac_table,
        state.prev_dc,
        comp.quant,
        state.coeff,
    )?;
    let pixel = downscale::idct_islow_1x1(state.coeff.coefficients());
    deposit_block_1x1(target.plane, target.stride, target.x, target.y, pixel);
    Ok(())
}
