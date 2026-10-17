// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast sequential RGB output and horizontal MCU rows for YCbCr scans.

use super::deposit::{assert_stripe_deposit_capacity, deposit_block};
use super::emit::emit_stripe_rgb_444;
use super::restart::{consume_restart_marker_if_due, finish_scan, McuPosition};
use super::{HuffmanPrimaryWidth, PreparedDecodePlan, ResolvedPreparedComponentPlan, StripeBuffer};
use crate::backend::Backend;
use crate::entropy::block::{decode_block_with_activity, BlockActivity, CoefficientBlock};
use crate::entropy::huffman::{FULL_PRIMARY_ENTRIES, NARROW_PRIMARY_ENTRIES};
use crate::error::{JpegError, Warning};
use crate::info::DownscaleFactor;
use crate::internal::bit_reader::BitReader;
use crate::internal::scratch::ScratchPool;
use crate::output::{InterleavedRgbWriter, OutputWriter};
use alloc::vec::Vec;

fn fast_rgb444_components<const N: usize>(
    plan: &PreparedDecodePlan,
    wide_huffman: bool,
) -> Result<
    (
        ResolvedPreparedComponentPlan<'_, N>,
        ResolvedPreparedComponentPlan<'_, N>,
        ResolvedPreparedComponentPlan<'_, N>,
    ),
    JpegError,
> {
    debug_assert!(plan.matches_fast_rgb444_shape());
    Ok((
        plan.resolved_component_with_entries::<N>(0)?
            .with_wide_huffman(wide_huffman),
        plan.resolved_component_with_entries::<N>(1)?
            .with_wide_huffman(wide_huffman),
        plan.resolved_component_with_entries::<N>(2)?
            .with_wide_huffman(wide_huffman),
    ))
}

pub(crate) fn decode_scan_fast_rgb_444<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
) -> Result<Vec<Warning>, JpegError> {
    match plan.huffman_primary_width {
        HuffmanPrimaryWidth::Narrow8 => decode_scan_fast_rgb_444_with_entries::<
            NARROW_PRIMARY_ENTRIES,
            W,
        >(plan, backend, scan_bytes, pool, writer),
        HuffmanPrimaryWidth::Full12 => decode_scan_fast_rgb_444_with_entries::<
            FULL_PRIMARY_ENTRIES,
            W,
        >(plan, backend, scan_bytes, pool, writer),
    }
}

fn decode_scan_fast_rgb_444_with_entries<const N: usize, W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
) -> Result<Vec<Warning>, JpegError> {
    debug_assert!(plan.matches_fast_rgb444_shape());

    let (width, height) = plan.dimensions;
    let mcus_per_row = width.div_ceil(8);
    let mcu_rows = height.div_ceil(8);

    pool.prepare_for(
        plan,
        plan.sampling,
        mcus_per_row,
        DownscaleFactor::Full.output_block_size(),
        plan.scratch_bytes,
    )?;

    let mut br = BitReader::new(scan_bytes);
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let (y_comp, cb_comp, cr_comp) =
        fast_rgb444_components::<N>(plan, plan.prefers_wide_huffman(scan_bytes.len()))?;
    let mut y_dc = 0i32;
    let mut cb_dc = 0i32;
    let mut cr_dc = 0i32;
    let restart = plan.restart_interval.unwrap_or(0);
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;
    let stripe = &mut pool.stripe_a;
    let row_context = FastRgbMcuRowContext {
        y_comp,
        cb_comp,
        cr_comp,
        backend,
        mcus_per_row,
        mcu_rows,
        restart,
    };

    {
        let mut row_state = FastRgbMcuRowState {
            br: &mut br,
            y_dc: &mut y_dc,
            cb_dc: &mut cb_dc,
            cr_dc: &mut cr_dc,
            coeff: &mut coeff,
            pixels: &mut pixels,
            mcus_since_restart: &mut mcus_since_restart,
            expected_rst: &mut expected_rst,
        };
        for my in 0..mcu_rows {
            decode_mcu_row_fast_rgb::<N, 1>(&row_context, &mut row_state, my, stripe)?;
            emit_stripe_rgb_444(plan, backend, stripe, my, writer)?;
        }
    }

    finish_scan(&mut br, true)
}

pub(super) struct FastRgbMcuRowContext<'a, const N: usize> {
    pub(super) y_comp: ResolvedPreparedComponentPlan<'a, N>,
    pub(super) cb_comp: ResolvedPreparedComponentPlan<'a, N>,
    pub(super) cr_comp: ResolvedPreparedComponentPlan<'a, N>,
    pub(super) backend: Backend,
    pub(super) mcus_per_row: u32,
    pub(super) mcu_rows: u32,
    pub(super) restart: u16,
}

pub(super) struct FastRgbMcuRowState<'a, 'b> {
    pub(super) br: &'a mut BitReader<'b>,
    pub(super) y_dc: &'a mut i32,
    pub(super) cb_dc: &'a mut i32,
    pub(super) cr_dc: &'a mut i32,
    pub(super) coeff: &'a mut CoefficientBlock,
    pub(super) pixels: &'a mut [u8; 64],
    pub(super) mcus_since_restart: &'a mut u32,
    pub(super) expected_rst: &'a mut u8,
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
    let mut dc = [*state.y_dc, *state.cb_dc, *state.cr_dc];
    let result =
        decode_mcu_row_fast_rgb_local::<N, LUMA_H>(context, state, &mut br, &mut dc, mcu_y, stripe);
    *state.br = br;
    [*state.y_dc, *state.cb_dc, *state.cr_dc] = dc;
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
    let [y_dc, cb_dc, cr_dc] = dc;
    for plane_idx in 0..3 {
        let h = if plane_idx == 0 { LUMA_H } else { 1 };
        assert_stripe_deposit_capacity(stripe, plane_idx, h, 1, context.mcus_per_row, 8);
    }
    for mx in 0..context.mcus_per_row {
        if consume_restart_marker_if_due(
            br,
            context.restart,
            *state.mcus_since_restart,
            state.expected_rst,
            McuPosition {
                current: mcu_y * context.mcus_per_row + mx,
                total: context.mcu_rows * context.mcus_per_row,
            },
        )? {
            *y_dc = 0;
            *cb_dc = 0;
            *cr_dc = 0;
            *state.mcus_since_restart = 0;
        }

        for vx in 0..LUMA_H {
            let block_x = (mx * LUMA_H + vx) * 8;

            let y_activity = decode_block_with_activity(
                br,
                context.y_comp.dc_table,
                context.y_comp.ac_table,
                y_dc,
                context.y_comp.quant,
                state.coeff,
            )?;
            match y_activity {
                BlockActivity::DcOnly => {
                    crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), state.pixels);
                }
                BlockActivity::TopLeft2x2 => context
                    .backend
                    .idct_top_left_2x2(state.coeff.coefficients(), state.pixels),
                BlockActivity::TopLeftQuadrant => context
                    .backend
                    .idct_top_left_quadrant(state.coeff.coefficients(), state.pixels),
                BlockActivity::RightHalfZero => context
                    .backend
                    .idct_right_half_zero(state.coeff.coefficients(), state.pixels),
                BlockActivity::BottomHalfZero => {
                    context
                        .backend
                        .idct_bottom_half_zero(state.coeff.coefficients(), state.pixels);
                }
                BlockActivity::General => context
                    .backend
                    .idct(state.coeff.coefficients(), state.pixels),
            }
            deposit_block(
                &mut stripe.planes[0],
                stripe.plane_strides[0],
                block_x,
                0,
                state.pixels,
            );
        }

        let block_x = mx * 8;

        let cb_activity = decode_block_with_activity(
            br,
            context.cb_comp.dc_table,
            context.cb_comp.ac_table,
            cb_dc,
            context.cb_comp.quant,
            state.coeff,
        )?;
        match cb_activity {
            BlockActivity::DcOnly => {
                crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), state.pixels);
            }
            BlockActivity::TopLeft2x2 => context
                .backend
                .idct_top_left_2x2(state.coeff.coefficients(), state.pixels),
            BlockActivity::TopLeftQuadrant => context
                .backend
                .idct_top_left_quadrant(state.coeff.coefficients(), state.pixels),
            BlockActivity::RightHalfZero => context
                .backend
                .idct_right_half_zero(state.coeff.coefficients(), state.pixels),
            BlockActivity::BottomHalfZero => {
                context
                    .backend
                    .idct_bottom_half_zero(state.coeff.coefficients(), state.pixels);
            }
            BlockActivity::General => context
                .backend
                .idct(state.coeff.coefficients(), state.pixels),
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
            context.cr_comp.dc_table,
            context.cr_comp.ac_table,
            cr_dc,
            context.cr_comp.quant,
            state.coeff,
        )?;
        match cr_activity {
            BlockActivity::DcOnly => {
                crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), state.pixels);
            }
            BlockActivity::TopLeft2x2 => context
                .backend
                .idct_top_left_2x2(state.coeff.coefficients(), state.pixels),
            BlockActivity::TopLeftQuadrant => context
                .backend
                .idct_top_left_quadrant(state.coeff.coefficients(), state.pixels),
            BlockActivity::RightHalfZero => context
                .backend
                .idct_right_half_zero(state.coeff.coefficients(), state.pixels),
            BlockActivity::BottomHalfZero => {
                context
                    .backend
                    .idct_bottom_half_zero(state.coeff.coefficients(), state.pixels);
            }
            BlockActivity::General => context
                .backend
                .idct(state.coeff.coefficients(), state.pixels),
        }
        deposit_block(
            &mut stripe.planes[2],
            stripe.plane_strides[2],
            block_x,
            0,
            state.pixels,
        );

        *state.mcus_since_restart += 1;
    }

    Ok(())
}
