// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast sequential RGB output and horizontal MCU rows for full-size 4:4:4
//! (YCbCr or identity RGB) and 4:2:2 scans.
//!
//! Whole images and regions, with or without restart intervals, decode one
//! MCU row of the requested MCU columns into a single stripe and convert it
//! to the output right away. Neither shape needs vertical upsampling context,
//! so the stripe has no halo row; 4:2:2's horizontal context comes from the
//! decoded window's margin columns.

use super::deposit::{assert_stripe_deposit_capacity, deposit_block};
use super::emit::{emit_stripe_rgb_v1, StripeOutputWindow};
use super::layout::{
    fast_tile_region_first_decode_mcu, last_mcu_row_for_rect, stripe_region_layout,
};
use super::restart::{
    consume_restart_marker_if_due, finish_scan, scan_position_before_mcu, skip_to_mcu_with,
    McuPosition, McuSkipState, McuSkipTarget, ScanPosition,
};
use super::{HuffmanPrimaryWidth, PreparedDecodePlan, ResolvedPreparedComponentPlan, StripeBuffer};
use crate::backend::Backend;
use crate::entropy::block::{
    decode_block_with_activity, skip_block, BlockActivity, CoefficientBlock,
};
use crate::entropy::huffman::{FULL_PRIMARY_ENTRIES, NARROW_PRIMARY_ENTRIES};
use crate::error::{JpegError, Warning};
use crate::info::{DownscaleFactor, Rect};
use crate::internal::bit_reader::BitReader;
use crate::internal::checkpoint::DeviceCheckpoint;
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
    debug_assert!(plan.matches_fast_rgb_v1_shape());
    Ok((
        plan.resolved_component_with_entries::<N>(0)?
            .with_wide_huffman(wide_huffman),
        plan.resolved_component_with_entries::<N>(1)?
            .with_wide_huffman(wide_huffman),
        plan.resolved_component_with_entries::<N>(2)?
            .with_wide_huffman(wide_huffman),
    ))
}

pub(crate) fn decode_scan_fast_rgb_v1<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
) -> Result<Vec<Warning>, JpegError> {
    decode_scan_fast_rgb_v1_region(
        plan,
        backend,
        scan_bytes,
        pool,
        writer,
        Rect::full(plan.dimensions),
        None,
    )
}

/// Decode `roi` of a 4:4:4 or 4:2:2 scan into `writer`, whose row 0 is
/// `roi.y`, entering the scan at `checkpoint` or a restart marker when either
/// is nearer than the scan start.
pub(crate) fn decode_scan_fast_rgb_v1_region<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    roi: Rect,
    checkpoint: Option<&DeviceCheckpoint>,
) -> Result<Vec<Warning>, JpegError> {
    match (plan.huffman_primary_width, plan.sampling.max_h) {
        (HuffmanPrimaryWidth::Narrow8, 2) => {
            decode_scan_fast_rgb_v1_with_entries::<NARROW_PRIMARY_ENTRIES, 2, W>(
                plan, backend, scan_bytes, pool, writer, roi, checkpoint,
            )
        }
        (HuffmanPrimaryWidth::Narrow8, _) => {
            decode_scan_fast_rgb_v1_with_entries::<NARROW_PRIMARY_ENTRIES, 1, W>(
                plan, backend, scan_bytes, pool, writer, roi, checkpoint,
            )
        }
        (HuffmanPrimaryWidth::Full12, 2) => {
            decode_scan_fast_rgb_v1_with_entries::<FULL_PRIMARY_ENTRIES, 2, W>(
                plan, backend, scan_bytes, pool, writer, roi, checkpoint,
            )
        }
        (HuffmanPrimaryWidth::Full12, _) => {
            decode_scan_fast_rgb_v1_with_entries::<FULL_PRIMARY_ENTRIES, 1, W>(
                plan, backend, scan_bytes, pool, writer, roi, checkpoint,
            )
        }
    }
}

fn decode_scan_fast_rgb_v1_with_entries<
    const N: usize,
    const LUMA_H: u32,
    W: OutputWriter + InterleavedRgbWriter,
>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    roi: Rect,
    checkpoint: Option<&DeviceCheckpoint>,
) -> Result<Vec<Warning>, JpegError> {
    debug_assert!(plan.matches_fast_rgb_v1_shape());
    debug_assert_eq!(u32::from(plan.sampling.max_h), LUMA_H);

    let (width, height) = plan.dimensions;
    let mcus_per_row = width.div_ceil(8 * LUMA_H);
    let mcu_rows = height.div_ceil(8);
    let region = stripe_region_layout(plan, DownscaleFactor::Full, roi);
    let first_mcu = fast_tile_region_first_decode_mcu(plan, roi, DownscaleFactor::Full);
    let first_row = first_mcu / mcus_per_row;
    let row_end = last_mcu_row_for_rect(roi, 8, mcu_rows) + 1;
    let window = StripeOutputWindow {
        roi,
        crop_start: (roi.x - region.source_x0) as usize,
        row_width: region.source_width_usize(),
    };
    // 4:2:2 upsamples a chroma row into row buffers before conversion.
    let upsample_width = if LUMA_H == 2 { window.row_width } else { 0 };
    pool.prepare_for_streaming(
        plan,
        region.stripe_mcus_per_row,
        0,
        upsample_width,
        plan.scratch_bytes,
    )?;

    let (y_comp, cb_comp, cr_comp) =
        fast_rgb444_components::<N>(plan, plan.prefers_wide_huffman(scan_bytes.len()))?;
    let restart = plan.restart_interval.unwrap_or(0);
    let ScanPosition {
        mut br,
        prev_dc,
        mcu_index: mut current_mcu,
        mut expected_rst,
    } = scan_position_before_mcu(scan_bytes, restart, checkpoint, first_mcu);
    let mut dc = [prev_dc[0], prev_dc[1], prev_dc[2]];
    let mut mcus_since_restart = 0u32;
    skip_to_mcu_with(
        McuSkipTarget {
            target_mcu: first_mcu,
            total_mcus: mcus_per_row * mcu_rows,
            restart,
            wide_huffman: plan.prefers_wide_huffman(scan_bytes.len()),
        },
        &mut McuSkipState {
            br: &mut br,
            prev_dc: &mut dc,
            current_mcu: &mut current_mcu,
            mcus_since_restart: &mut mcus_since_restart,
            expected_rst: &mut expected_rst,
        },
        |br, dc| {
            for ((comp, blocks), dc) in [(y_comp, LUMA_H), (cb_comp, 1), (cr_comp, 1)]
                .into_iter()
                .zip(dc)
            {
                for _ in 0..blocks {
                    skip_block(br, comp.dc_table, comp.ac_table, dc)?;
                }
            }
            Ok(())
        },
    )?;
    let [mut y_dc, mut cb_dc, mut cr_dc] = dc;
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let ScratchPool {
        stripe_a: stripe,
        ycbcr_generic_rows: upsampled,
        ..
    } = pool;
    let row_context = FastRgbMcuRowContext {
        y_comp,
        cb_comp,
        cr_comp,
        backend,
        mcus_per_row,
        mcu_rows,
        restart,
        stripe_mcu_start: region.stripe_mcu_start,
        stripe_mcus_per_row: region.stripe_mcus_per_row,
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
        for my in first_row..row_end {
            decode_mcu_row_fast_rgb::<N, LUMA_H>(&row_context, &mut row_state, my, stripe)?;
            emit_stripe_rgb_v1(plan, backend, stripe, my, window, upsampled, writer)?;
        }
    }

    finish_scan(&mut br, row_end == mcu_rows)
}

pub(super) struct FastRgbMcuRowContext<'a, const N: usize> {
    pub(super) y_comp: ResolvedPreparedComponentPlan<'a, N>,
    pub(super) cb_comp: ResolvedPreparedComponentPlan<'a, N>,
    pub(super) cr_comp: ResolvedPreparedComponentPlan<'a, N>,
    pub(super) backend: Backend,
    pub(super) mcus_per_row: u32,
    pub(super) mcu_rows: u32,
    pub(super) restart: u16,
    /// Decoded MCU columns; the others are entropy-skipped.
    pub(super) stripe_mcu_start: u32,
    pub(super) stripe_mcus_per_row: u32,
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
        assert_stripe_deposit_capacity(stripe, plane_idx, h, 1, context.stripe_mcus_per_row, 8, 0);
    }
    let window_end = context.stripe_mcu_start + context.stripe_mcus_per_row;
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

        if mx < context.stripe_mcu_start || mx >= window_end {
            for _ in 0..LUMA_H {
                skip_block(br, context.y_comp.dc_table, context.y_comp.ac_table, y_dc)?;
            }
            skip_block(
                br,
                context.cb_comp.dc_table,
                context.cb_comp.ac_table,
                cb_dc,
            )?;
            skip_block(
                br,
                context.cr_comp.dc_table,
                context.cr_comp.ac_table,
                cr_dc,
            )?;
            *state.mcus_since_restart += 1;
            continue;
        }
        let local_mx = mx - context.stripe_mcu_start;

        for vx in 0..LUMA_H {
            let block_x = (local_mx * LUMA_H + vx) * 8;

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

        let block_x = local_mx * 8;

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
