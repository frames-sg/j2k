// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast sequential RGB output and horizontal MCU rows for full-size 4:4:4
//! (YCbCr or identity RGB) and 4:2:2 scans.
//!
//! Whole images and regions, with or without restart intervals, decode one
//! MCU row of the requested MCU columns into a single stripe and convert it
//! to the output right away. Neither shape needs vertical upsampling context,
//! so the stripe has no halo row; 4:2:2's horizontal context comes from the
//! decoded window's margin columns.

use super::dct::sequential_mcu_grid;
use super::deposit::{
    assert_stripe_deposit_capacity, deposit_block, three_component_dc, FastTile420Components,
};
use super::emit::FastRgbRowsEmitter;
use super::layout::{last_mcu_row_for_rect, stripe_region_layout, CropWindow, McuColumns};
use super::restart::{enter_scan_at_mcu_with, finish_scan, RestartState};
use super::{FastRgbRequest, FastRgbRoute, HuffmanPrimaryWidth, PreparedDecodePlan, StripeBuffer};
use crate::backend::Backend;
use crate::entropy::block::{
    decode_block_with_activity, skip_block, BlockActivity, CoefficientBlock,
};
use crate::entropy::huffman::{FULL_PRIMARY_ENTRIES, NARROW_PRIMARY_ENTRIES};
use crate::error::{JpegError, Warning};
use crate::info::DownscaleFactor;
use crate::internal::bit_reader::BitReader;
use crate::internal::scratch::ScratchPool;
use crate::output::{InterleavedRgbWriter, OutputWriter};
use alloc::vec::Vec;

/// Decode `request.roi` of a 4:4:4 or 4:2:2 scan into `writer`, whose row 0
/// is `request.roi.y`, entering the scan at `request.checkpoint` or a restart
/// marker when either is nearer than the scan start.
pub(super) fn decode_scan_fast_rgb_rows<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    request: FastRgbRequest<'_>,
) -> Result<Vec<Warning>, JpegError> {
    match (plan.huffman_primary_width, plan.sampling.max_h) {
        (HuffmanPrimaryWidth::Narrow8, 2) => {
            decode_scan_fast_rgb_rows_with_entries::<NARROW_PRIMARY_ENTRIES, 2, W>(
                plan, backend, scan_bytes, pool, writer, request,
            )
        }
        (HuffmanPrimaryWidth::Narrow8, _) => {
            decode_scan_fast_rgb_rows_with_entries::<NARROW_PRIMARY_ENTRIES, 1, W>(
                plan, backend, scan_bytes, pool, writer, request,
            )
        }
        (HuffmanPrimaryWidth::Full12, 2) => {
            decode_scan_fast_rgb_rows_with_entries::<FULL_PRIMARY_ENTRIES, 2, W>(
                plan, backend, scan_bytes, pool, writer, request,
            )
        }
        (HuffmanPrimaryWidth::Full12, _) => {
            decode_scan_fast_rgb_rows_with_entries::<FULL_PRIMARY_ENTRIES, 1, W>(
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
    let region = stripe_region_layout(plan, DownscaleFactor::Full, roi);
    let columns = McuColumns::new(sequential_mcu_grid(plan), region);
    let first_row = FastRgbRoute::Rows.first_decode_row(plan, roi);
    let row_end = last_mcu_row_for_rect(roi, 8, columns.mcu_rows) + 1;
    let emitter = FastRgbRowsEmitter {
        plan,
        backend,
        roi,
        crop: CropWindow::new(region, roi),
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

    let components =
        FastTile420Components::<N>::resolve(plan, plan.prefers_wide_huffman(scan_bytes.len()))?;
    let mut cursor = enter_scan_at_mcu_with(
        scan_bytes,
        plan.restart_interval.unwrap_or(0),
        request.checkpoint,
        columns.mcu_position(0, first_row),
        |br, dc| {
            let [y_dc, cb_dc, cr_dc] = three_component_dc(dc)?;
            components.skip_mcu::<LUMA_H>(br, y_dc, cb_dc, cr_dc)
        },
    )?;
    let mut dc = *three_component_dc(&mut cursor.prev_dc)?;
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
        let mut row_state = FastRgbMcuRowState {
            br: &mut cursor.br,
            dc: &mut dc,
            coeff: &mut coeff,
            pixels: &mut pixels,
            restart: &mut cursor.restart,
        };
        for my in first_row..row_end {
            decode_mcu_row_fast_rgb::<N, LUMA_H>(&row_context, &mut row_state, my, stripe)?;
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

pub(super) struct FastRgbMcuRowState<'a, 'b> {
    pub(super) br: &'a mut BitReader<'b>,
    pub(super) dc: &'a mut [i32; 3],
    pub(super) coeff: &'a mut CoefficientBlock,
    pub(super) pixels: &'a mut [u8; 64],
    pub(super) restart: &'a mut RestartState,
}

pub(super) fn decode_mcu_row_fast_rgb<const N: usize, const LUMA_H: u32>(
    context: &FastRgbMcuRowContext<'_, N>,
    state: &mut FastRgbMcuRowState<'_, '_>,
    mcu_y: u32,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    // Decode against locals so the bit accumulator and DC predictors live in
    // registers for the whole row instead of round-tripping through `state`
    // on every symbol. They are written back even when the row fails.
    let mut br = state.br.clone();
    let mut dc = *state.dc;
    let result =
        decode_mcu_row_fast_rgb_local::<N, LUMA_H>(context, state, &mut br, &mut dc, mcu_y, stripe);
    *state.br = br;
    *state.dc = dc;
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
    state: &mut FastRgbMcuRowState<'_, '_>,
    br: &mut BitReader<'_>,
    dc: &mut [i32; 3],
    mcu_y: u32,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    let FastRgbMcuRowContext {
        components,
        backend,
        columns,
    } = *context;
    let [y_dc, cb_dc, cr_dc] = dc;
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
                    crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), state.pixels);
                }
                BlockActivity::TopLeft2x2 => {
                    backend.idct_top_left_2x2(state.coeff.coefficients(), state.pixels);
                }
                BlockActivity::TopLeftQuadrant => {
                    backend.idct_top_left_quadrant(state.coeff.coefficients(), state.pixels);
                }
                BlockActivity::RightHalfZero => {
                    backend.idct_right_half_zero(state.coeff.coefficients(), state.pixels);
                }
                BlockActivity::BottomHalfZero => {
                    backend.idct_bottom_half_zero(state.coeff.coefficients(), state.pixels);
                }
                BlockActivity::General => backend.idct(state.coeff.coefficients(), state.pixels),
            }
            deposit_block(
                &mut stripe.planes[0],
                stripe.plane_strides[0],
                block_x,
                0,
                state.pixels,
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
                crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), state.pixels);
            }
            BlockActivity::TopLeft2x2 => {
                backend.idct_top_left_2x2(state.coeff.coefficients(), state.pixels);
            }
            BlockActivity::TopLeftQuadrant => {
                backend.idct_top_left_quadrant(state.coeff.coefficients(), state.pixels);
            }
            BlockActivity::RightHalfZero => {
                backend.idct_right_half_zero(state.coeff.coefficients(), state.pixels);
            }
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(state.coeff.coefficients(), state.pixels);
            }
            BlockActivity::General => backend.idct(state.coeff.coefficients(), state.pixels),
        }
        deposit_block(
            &mut stripe.planes[1],
            stripe.plane_strides[1],
            block_x,
            0,
            state.pixels,
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
                crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), state.pixels);
            }
            BlockActivity::TopLeft2x2 => {
                backend.idct_top_left_2x2(state.coeff.coefficients(), state.pixels);
            }
            BlockActivity::TopLeftQuadrant => {
                backend.idct_top_left_quadrant(state.coeff.coefficients(), state.pixels);
            }
            BlockActivity::RightHalfZero => {
                backend.idct_right_half_zero(state.coeff.coefficients(), state.pixels);
            }
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(state.coeff.coefficients(), state.pixels);
            }
            BlockActivity::General => backend.idct(state.coeff.coefficients(), state.pixels),
        }
        deposit_block(
            &mut stripe.planes[2],
            stripe.plane_strides[2],
            block_x,
            0,
            state.pixels,
        );

        state.restart.end_mcu();
    }

    Ok(())
}
