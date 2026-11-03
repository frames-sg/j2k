// SPDX-License-Identifier: MIT OR Apache-2.0

use super::layout::McuColumns;
use super::restart::RestartState;
use super::{PreparedDecodePlan, ResolvedPreparedComponentPlan, StripeBuffer};
use crate::backend::Backend;
use crate::entropy::block::{skip_block, BlockActivity, CoefficientBlock};
use crate::error::JpegError;
use crate::internal::bit_reader::BitReader;
use core::ptr;

/// Release-mode precondition for the raw-pointer writes in [`deposit_block`]
/// and [`deposit_dc_block`]: the stripe plane must carry the stride and
/// capacity [`StripeBuffer::resize_for`] establishes for
/// `stripe_mcus_per_row` MCUs of `h`×`v` `block_size`-pixel blocks below
/// `halo_rows` context rows (see [`super::StripeLayout::for_sampling_with_halo`]).
/// Asserted once per stripe row so the per-block deposits stay bounds-check
/// free in the hot loop.
pub(super) fn assert_stripe_deposit_capacity(
    stripe: &StripeBuffer,
    plane_idx: usize,
    h: u32,
    v: u32,
    stripe_mcus_per_row: u32,
    block_size: u32,
    halo_rows: u32,
) {
    let needed_cols = stripe_mcus_per_row as usize * h as usize * block_size as usize;
    let needed_rows = v as usize * block_size as usize + halo_rows as usize;
    let stride = stripe.plane_strides[plane_idx];
    let len = stripe.planes[plane_idx].len();
    assert!(
        stride >= needed_cols
            && stride
                .checked_mul(needed_rows)
                .is_some_and(|needed| len >= needed),
        "stripe plane {plane_idx} ({len} bytes, stride {stride}) is not sized for \
         {stripe_mcus_per_row} MCUs of {h}x{v} blocks of {block_size}px below \
         {halo_rows} halo rows; StripeBuffer::resize_for must run first"
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
        BlockActivity::TopLeft2x2 => {
            backend.idct_top_left_2x2(coeff.coefficients(), pixels);
            deposit_block(target.plane, target.stride, target.x, target.y, pixels);
        }
        BlockActivity::TopLeftQuadrant => {
            backend.idct_top_left_quadrant(coeff.coefficients(), pixels);
            deposit_block(target.plane, target.stride, target.x, target.y, pixels);
        }
        BlockActivity::RightHalfZero => {
            backend.idct_right_half_zero(coeff.coefficients(), pixels);
            deposit_block(target.plane, target.stride, target.x, target.y, pixels);
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

pub(super) struct PlaneBlockTarget<'a> {
    pub(super) plane: &'a mut [u8],
    pub(super) stride: usize,
    pub(super) x: u32,
    pub(super) y: u32,
}

/// The three coded components of a fast-route scan in output order (Y, Cb,
/// Cr, or R, G, B for identity RGB), with tables resolved for the hot loops.
#[derive(Clone, Copy)]
pub(super) struct FastTile420Components<'a, const N: usize> {
    pub(super) y: ResolvedPreparedComponentPlan<'a, N>,
    pub(super) cb: ResolvedPreparedComponentPlan<'a, N>,
    pub(super) cr: ResolvedPreparedComponentPlan<'a, N>,
}

impl<'a, const N: usize> FastTile420Components<'a, N> {
    pub(super) fn resolve(
        plan: &'a PreparedDecodePlan,
        wide_huffman: bool,
    ) -> Result<Self, JpegError> {
        let component = |index| {
            plan.resolved_component_with_entries::<N>(index)
                .map(|component| component.with_wide_huffman(wide_huffman))
        };
        Ok(Self {
            y: component(0)?,
            cb: component(1)?,
            cr: component(2)?,
        })
    }

    /// Entropy-skip one MCU of `LUMA_BLOCKS` luma blocks and one block of each
    /// chroma component, for the seek to a route's first decoded MCU. The row
    /// kernels skip MCUs outside their column window with direct
    /// [`skip_block`] calls instead: this helper, inlined there, spills their
    /// register-resident DC predictors and reader (+1–3 % decode time).
    // ROI seeks invoke this leaf once per skipped MCU; forced inlining keeps
    // its Huffman skips in the caller's hot loop without adding a
    // branch-and-call boundary.
    #[expect(
        clippy::inline_always,
        reason = "ROI seek hot path keeps the MCU's Huffman skips inside the caller loop"
    )]
    #[inline(always)]
    pub(super) fn skip_mcu<const LUMA_BLOCKS: u32>(
        self,
        br: &mut BitReader<'_>,
        y_dc: &mut i32,
        cb_dc: &mut i32,
        cr_dc: &mut i32,
    ) -> Result<(), JpegError> {
        for _ in 0..LUMA_BLOCKS {
            skip_block(br, self.y.dc_table, self.y.ac_table, y_dc)?;
        }
        skip_block(br, self.cb.dc_table, self.cb.ac_table, cb_dc)?;
        skip_block(br, self.cr.dc_table, self.cr.ac_table, cr_dc)?;
        Ok(())
    }
}

/// The first three of an entropy cursor's DC predictors, which the
/// three-component fast routes keep in a fixed array.
pub(super) fn three_component_dc(prev_dc: &mut [i32]) -> Result<&mut [i32; 3], JpegError> {
    prev_dc
        .first_chunk_mut::<3>()
        .ok_or(JpegError::InternalInvariant {
            reason: "fast sequential decode requires three DC predictors",
        })
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
    pub(super) restart: &'a mut RestartState,
}

/// Register-friendly copy of a [`FastTile420EntropyState`]'s reader and DC
/// predictors. Behind `&mut` fields the accumulator round-trips through
/// memory on every Huffman symbol; decoding a row against this local (with
/// the row kernel inlined, so it never escapes) lets LLVM keep it in
/// registers. Callers restore it even when the row fails.
pub(super) struct LocalEntropy<'b> {
    br: BitReader<'b>,
    dc: [i32; 3],
}

impl<'b> LocalEntropy<'b> {
    /// Entropy state that decodes against these locals and `outer`'s other
    /// fields.
    #[expect(
        clippy::inline_always,
        reason = "must inline so the borrowed locals never escape the row frame"
    )]
    #[inline(always)]
    pub(super) fn state<'a>(
        &'a mut self,
        outer: &'a mut FastTile420EntropyState<'_, 'b>,
    ) -> FastTile420EntropyState<'a, 'b> {
        let [y, cb, cr] = &mut self.dc;
        FastTile420EntropyState {
            br: &mut self.br,
            dc: FastTile420DcState { y, cb, cr },
            coeff: &mut *outer.coeff,
            restart: &mut *outer.restart,
        }
    }
}

impl<'b> FastTile420EntropyState<'_, 'b> {
    #[expect(
        clippy::inline_always,
        reason = "must inline so the copied reader stays register-resident"
    )]
    #[inline(always)]
    pub(super) fn take_local(&self) -> LocalEntropy<'b> {
        LocalEntropy {
            br: self.br.clone(),
            dc: [*self.dc.y, *self.dc.cb, *self.dc.cr],
        }
    }

    #[expect(
        clippy::inline_always,
        reason = "must inline so the copied reader stays register-resident"
    )]
    #[inline(always)]
    pub(super) fn restore_local(&mut self, local: LocalEntropy<'b>) {
        *self.br = local.br;
        [*self.dc.y, *self.dc.cb, *self.dc.cr] = local.dc;
    }
}

#[derive(Clone, Copy)]
pub(super) struct FastTile420Window {
    pub(super) columns: McuColumns,
    /// Whether the final coded MCU's right luma blocks intersect the image.
    pub(super) right_luma: bool,
    /// Whether this row's bottom luma blocks intersect the image.
    pub(super) bottom_luma: bool,
}
