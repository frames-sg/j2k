// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast sequential RGB output and horizontal MCU rows for full-size 4:4:4
//! (YCbCr or identity RGB) and 4:2:2 scans.
//!
//! Whole images and regions, with or without restart intervals, decode one
//! MCU row of the requested MCU columns into a single stripe and convert it
//! to the output right away. Neither shape needs vertical upsampling context,
//! so the stripe has no halo row; 4:2:2's horizontal context comes from the
//! decoded window's margin columns.

use super::dct::sequential_mcu_size;
use super::deposit::{
    assert_stripe_deposit_capacity, deposit_block, FastRowState, FastTile420Components,
    LocalEntropy,
};
use super::emit::FastRgbRowsEmitter;
use super::layout::{last_mcu_row_for_rect, McuColumns};
use super::restart::finish_scan;
use super::{FastRgbRequest, HuffmanPrimaryWidth, PreparedDecodePlan, StripeBuffer};
use crate::adapter::JpegFastPacketFamily;
use crate::backend::Backend;
use crate::entropy::block::{
    decode_block_with_activity, skip_block, BlockActivity, CoefficientBlock,
};
use crate::entropy::huffman::{FULL_PRIMARY_ENTRIES, NARROW_PRIMARY_ENTRIES};
use crate::error::{JpegError, Warning};
use crate::internal::scratch::ScratchPool;
use crate::output::{InterleavedRgbWriter, OutputWriter};
use alloc::vec::Vec;

/// Decode `request.roi` of a 4:4:4 (`LUMA_H` 1) or 4:2:2 (`LUMA_H` 2) scan
/// into `writer`, whose row 0 is `request.roi.y`, entering the scan at
/// `request.checkpoint` or a restart marker when either is nearer than the
/// scan start.
pub(super) fn decode_scan_fast_rgb_rows<
    const LUMA_H: u32,
    W: OutputWriter + InterleavedRgbWriter,
>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    request: FastRgbRequest<'_>,
) -> Result<Vec<Warning>, JpegError> {
    match plan.huffman_primary_width {
        HuffmanPrimaryWidth::Narrow8 => {
            decode_scan_fast_rgb_rows_with_entries::<NARROW_PRIMARY_ENTRIES, LUMA_H, W>(
                plan, backend, scan_bytes, pool, writer, request,
            )
        }
        HuffmanPrimaryWidth::Full12 => {
            decode_scan_fast_rgb_rows_with_entries::<FULL_PRIMARY_ENTRIES, LUMA_H, W>(
                plan, backend, scan_bytes, pool, writer, request,
            )
        }
    }
}

fn decode_scan_fast_rgb_rows_with_entries<
    const N: usize,
    const LUMA_H: u32,
    W: OutputWriter + InterleavedRgbWriter,
>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    request: FastRgbRequest<'_>,
) -> Result<Vec<Warning>, JpegError> {
    debug_assert!(plan.matches_fast_422_shape() || plan.matches_fast_444_shape());
    debug_assert_eq!(u32::from(plan.sampling.max_h), LUMA_H);

    let roi = request.roi;
    let (columns, crop) = request.columns(plan);
    let first_row = JpegFastPacketFamily::Fast444.first_decode_row(plan, roi);
    let row_end = last_mcu_row_for_rect(roi, sequential_mcu_size(plan).1, columns.mcu_rows) + 1;
    let emitter = FastRgbRowsEmitter {
        plan,
        backend,
        roi,
        crop,
    };
    // 4:2:2 upsamples a chroma row into row buffers before conversion.
    let upsample_width = if LUMA_H == 2 {
        emitter.crop.row_width
    } else {
        0
    };
    pool.prepare_for_streaming(
        plan,
        columns.stripe_mcus_per_row,
        0,
        upsample_width,
        plan.scratch_bytes,
    )?;

    let (components, mut cursor, mut dc) = FastTile420Components::<N>::enter_scan::<LUMA_H>(
        plan,
        scan_bytes,
        request.checkpoint,
        columns.mcu_position(0, first_row),
    )?;
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let ScratchPool {
        stripe_a: stripe,
        ycbcr_generic_rows: upsampled,
        ..
    } = pool;
    let row_context = FastRgbMcuRowContext {
        components,
        backend,
        columns,
    };

    {
        let mut row_state = FastRowState {
            br: &mut cursor.br,
            dc: &mut dc,
            coeff: &mut coeff,
            restart: &mut cursor.restart,
        };
        for my in first_row..row_end {
            decode_mcu_row_fast_rgb::<N, LUMA_H>(
                &row_context,
                &mut row_state,
                &mut pixels,
                my,
                stripe,
            )?;
            emitter.emit_stripe_rgb_rows(stripe, my, upsampled, writer)?;
        }
    }

    finish_scan(&mut cursor.br, row_end == columns.mcu_rows)
}

pub(super) struct FastRgbMcuRowContext<'a, const N: usize> {
    pub(super) components: FastTile420Components<'a, N>,
    pub(super) backend: Backend,
    pub(super) columns: McuColumns,
}

pub(super) fn decode_mcu_row_fast_rgb<const N: usize, const LUMA_H: u32>(
    context: &FastRgbMcuRowContext<'_, N>,
    state: &mut FastRowState<'_, '_>,
    pixels: &mut [u8; 64],
    mcu_y: u32,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    // The kernel takes the caller's `state` beside the local copy, and the
    // caller's pixel block. Through a state rebuilt around the local, as the
    // 4:2:0 kernel decodes, the coefficient-block and restart pointers stay
    // live for the whole row and this kernel spills (aarch64: +7 %
    // instructions, +7-30 % stack accesses); a kernel-local pixel block made
    // x86 4:4:4 decodes 2-4 % slower.
    let mut local = state.take_local();
    let result = decode_mcu_row_fast_rgb_local::<N, LUMA_H>(
        context, state, &mut local, pixels, mcu_y, stripe,
    );
    state.restore_local(local);
    result
}

#[expect(
    clippy::too_many_lines,
    reason = "the horizontal MCU kernel keeps luma and chroma block decodes and deposition in sampling order"
)]
#[expect(
    clippy::inline_always,
    reason = "inlining into the row wrapper is what keeps the local reader out of memory"
)]
#[inline(always)]
fn decode_mcu_row_fast_rgb_local<const N: usize, const LUMA_H: u32>(
    context: &FastRgbMcuRowContext<'_, N>,
    state: &mut FastRowState<'_, '_>,
    local: &mut LocalEntropy<'_>,
    pixels: &mut [u8; 64],
    mcu_y: u32,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    let FastRgbMcuRowContext {
        components,
        backend,
        columns,
    } = *context;
    let LocalEntropy {
        br,
        dc: [y_dc, cb_dc, cr_dc],
    } = local;
    for plane_idx in 0..3 {
        let h = if plane_idx == 0 { LUMA_H } else { 1 };
        assert_stripe_deposit_capacity(stripe, plane_idx, h, 1, columns.stripe_mcus_per_row, 8, 0);
    }
    for mx in 0..columns.mcus_per_row {
        if state
            .restart
            .begin_mcu(br, columns.mcu_position(mx, mcu_y))?
        {
            *y_dc = 0;
            *cb_dc = 0;
            *cr_dc = 0;
        }

        if !columns.contains(mx) {
            // Direct skips keep the DC predictors in registers; see
            // `FastTile420Components::skip_mcu`.
            for _ in 0..LUMA_H {
                skip_block(br, components.y.dc_table, components.y.ac_table, y_dc)?;
            }
            skip_block(br, components.cb.dc_table, components.cb.ac_table, cb_dc)?;
            skip_block(br, components.cr.dc_table, components.cr.ac_table, cr_dc)?;
            state.restart.end_mcu();
            continue;
        }
        let local_mx = columns.local(mx);

        for vx in 0..LUMA_H {
            let block_x = (local_mx * LUMA_H + vx) * 8;

            let y_activity = decode_block_with_activity(
                br,
                components.y.dc_table,
                components.y.ac_table,
                y_dc,
                components.y.quant,
                state.coeff,
            )?;
            match y_activity {
                BlockActivity::DcOnly => {
                    crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), pixels);
                }
                BlockActivity::TopLeft2x2 => {
                    backend.idct_top_left_2x2(state.coeff.coefficients(), pixels);
                }
                BlockActivity::TopLeftQuadrant => {
                    backend.idct_top_left_quadrant(state.coeff.coefficients(), pixels);
                }
                BlockActivity::RightHalfZero => {
                    backend.idct_right_half_zero(state.coeff.coefficients(), pixels);
                }
                BlockActivity::BottomHalfZero => {
                    backend.idct_bottom_half_zero(state.coeff.coefficients(), pixels);
                }
                BlockActivity::General => backend.idct(state.coeff.coefficients(), pixels),
            }
            deposit_block(
                &mut stripe.planes[0],
                stripe.plane_strides[0],
                block_x,
                0,
                pixels,
            );
        }

        let block_x = local_mx * 8;

        let cb_activity = decode_block_with_activity(
            br,
            components.cb.dc_table,
            components.cb.ac_table,
            cb_dc,
            components.cb.quant,
            state.coeff,
        )?;
        match cb_activity {
            BlockActivity::DcOnly => {
                crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), pixels);
            }
            BlockActivity::TopLeft2x2 => {
                backend.idct_top_left_2x2(state.coeff.coefficients(), pixels);
            }
            BlockActivity::TopLeftQuadrant => {
                backend.idct_top_left_quadrant(state.coeff.coefficients(), pixels);
            }
            BlockActivity::RightHalfZero => {
                backend.idct_right_half_zero(state.coeff.coefficients(), pixels);
            }
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(state.coeff.coefficients(), pixels);
            }
            BlockActivity::General => backend.idct(state.coeff.coefficients(), pixels),
        }
        deposit_block(
            &mut stripe.planes[1],
            stripe.plane_strides[1],
            block_x,
            0,
            pixels,
        );

        let cr_activity = decode_block_with_activity(
            br,
            components.cr.dc_table,
            components.cr.ac_table,
            cr_dc,
            components.cr.quant,
            state.coeff,
        )?;
        match cr_activity {
            BlockActivity::DcOnly => {
                crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), pixels);
            }
            BlockActivity::TopLeft2x2 => {
                backend.idct_top_left_2x2(state.coeff.coefficients(), pixels);
            }
            BlockActivity::TopLeftQuadrant => {
                backend.idct_top_left_quadrant(state.coeff.coefficients(), pixels);
            }
            BlockActivity::RightHalfZero => {
                backend.idct_right_half_zero(state.coeff.coefficients(), pixels);
            }
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(state.coeff.coefficients(), pixels);
            }
            BlockActivity::General => backend.idct(state.coeff.coefficients(), pixels),
        }
        deposit_block(
            &mut stripe.planes[2],
            stripe.plane_strides[2],
            block_x,
            0,
            pixels,
        );

        state.restart.end_mcu();
    }

    Ok(())
}
