// SPDX-License-Identifier: MIT OR Apache-2.0

//! MCU-row entropy decode and component-plane deposit.

use super::super::deposit::{
    assert_stripe_deposit_capacity, deposit_block, deposit_block_1x1, deposit_block_2x2,
    deposit_block_4x4,
};
use super::super::layout::{component_block_intersects_rect, ComponentBlockPosition};
use super::super::restart::{consume_restart_marker_if_due, McuPosition};
use super::super::{PreparedDecodePlan, StripeBuffer};
use crate::backend::Backend;
use crate::color::scaled_sampling::ScaledSampling;
use crate::entropy::block::{
    decode_block_with_activity, skip_block, BlockActivity, CoefficientBlock,
};
use crate::error::JpegError;
use crate::idct::downscale;
use crate::info::{DownscaleFactor, Rect};
use crate::internal::bit_reader::BitReader;

pub(super) struct McuRowContext<'a> {
    pub(super) plan: &'a PreparedDecodePlan,
    pub(super) backend: Backend,
    pub(super) downscale: DownscaleFactor,
    /// Per-component IDCT sizes; planes are laid out for its effective
    /// sampling.
    pub(super) scaled: &'a ScaledSampling,
    pub(super) output_rect: Rect,
    pub(super) full_output_rect: bool,
    pub(super) stripe_mcu_start: u32,
    pub(super) stripe_mcus_per_row: u32,
    pub(super) mcus_per_row: u32,
    pub(super) mcu_rows: u32,
    pub(super) restart: u16,
}

pub(super) struct McuRowState<'a, 'b> {
    pub(super) br: &'a mut BitReader<'b>,
    pub(super) prev_dc: &'a mut [i32],
    pub(super) coeff: &'a mut CoefficientBlock,
    pub(super) pixels: &'a mut [u8; 64],
    pub(super) mcus_since_restart: &'a mut u32,
    pub(super) expected_rst: &'a mut u8,
}

pub(super) fn decode_mcu_row(
    context: &McuRowContext<'_>,
    state: &mut McuRowState<'_, '_>,
    mcu_y: u32,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    // Decode against locals so the bit accumulator and DC predictors live in
    // registers for the whole row instead of round-tripping through `state`
    // on every symbol. They are written back even when the row fails.
    let mut prev_dc = [0i32; 4];
    let Some(local_dc) = prev_dc.get_mut(..state.prev_dc.len()) else {
        return decode_mcu_row_local(context, state, mcu_y, stripe);
    };
    local_dc.copy_from_slice(state.prev_dc);
    let mut br = state.br.clone();
    let result = decode_mcu_row_local(
        context,
        &mut McuRowState {
            br: &mut br,
            prev_dc: &mut *local_dc,
            coeff: &mut *state.coeff,
            pixels: &mut *state.pixels,
            mcus_since_restart: &mut *state.mcus_since_restart,
            expected_rst: &mut *state.expected_rst,
        },
        mcu_y,
        stripe,
    );
    *state.br = br;
    state.prev_dc.copy_from_slice(local_dc);
    result
}

#[expect(
    clippy::too_many_lines,
    reason = "the MCU kernel traverses component sampling factors while preserving entropy, predictor, IDCT, and deposit order"
)]
#[expect(
    clippy::inline_always,
    reason = "inlining into the row wrapper is what keeps the local reader out of memory"
)]
#[inline(always)]
fn decode_mcu_row_local(
    context: &McuRowContext<'_>,
    state: &mut McuRowState<'_, '_>,
    mcu_y: u32,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    let stripe_mcu_end = context.stripe_mcu_start + context.stripe_mcus_per_row;
    for comp in &context.plan.components {
        assert_stripe_deposit_capacity(
            stripe,
            comp.output_index,
            u32::from(comp.h),
            u32::from(comp.v),
            context.stripe_mcus_per_row,
            context.scaled.component(comp.output_index).idct_size,
        );
    }
    let mut pixels_4x4 = [0u8; 16];
    let mut pixels_2x2 = [0u8; 4];
    // Resolve immutable Huffman handles outside the MCU loop, including the
    // generic grayscale, restart, and subsampled paths.
    let mut resolved = [context.plan.resolved_component(0)?; 4];
    if context.plan.components.len() > resolved.len() {
        return Err(JpegError::InternalInvariant {
            reason: "sequential decode supports at most four components",
        });
    }
    for (tables, comp) in resolved.iter_mut().zip(&context.plan.components) {
        *tables = context.plan.resolve_component(comp)?;
    }
    for mx in 0..context.mcus_per_row {
        if consume_restart_marker_if_due(
            state.br,
            context.restart,
            *state.mcus_since_restart,
            state.expected_rst,
            McuPosition {
                current: mcu_y * context.mcus_per_row + mx,
                total: context.mcu_rows * context.mcus_per_row,
            },
        )? {
            state.prev_dc.fill(0);
            *state.mcus_since_restart = 0;
        }

        for (comp, tables) in context.plan.components.iter().zip(&resolved) {
            let plane_idx = comp.output_index;
            let dc_table = tables.dc_table;
            let ac_table = tables.ac_table;
            let in_region = mx >= context.stripe_mcu_start && mx < stripe_mcu_end;
            // libjpeg-turbo may decode a subsampled component with a larger
            // reduced IDCT than the scale's own block size.
            let block_size = context.scaled.component(plane_idx).idct_size;
            let local_mcu_x0_px =
                mx.saturating_sub(context.stripe_mcu_start) * u32::from(comp.h) * block_size;
            for vy in 0..u32::from(comp.v) {
                for vx in 0..u32::from(comp.h) {
                    let should_output = in_region
                        && (context.full_output_rect
                            || component_block_intersects_rect(
                                context.plan,
                                comp,
                                context.downscale,
                                ComponentBlockPosition {
                                    mcu_x: mx,
                                    mcu_y,
                                    block_x: vx,
                                    block_y: vy,
                                },
                                context.output_rect,
                            ));
                    if !should_output {
                        skip_block(state.br, dc_table, ac_table, &mut state.prev_dc[plane_idx])?;
                        continue;
                    }

                    let activity = decode_block_with_activity(
                        state.br,
                        dc_table,
                        ac_table,
                        &mut state.prev_dc[plane_idx],
                        &comp.quant,
                        state.coeff,
                    )?;
                    let block_x = local_mcu_x0_px + vx * block_size;
                    let block_y = vy * block_size;
                    let plane = &mut stripe.planes[plane_idx];
                    let stride = stripe.plane_strides[plane_idx];
                    match (block_size, activity) {
                        (8, BlockActivity::DcOnly) => {
                            crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), state.pixels);
                            deposit_block(plane, stride, block_x, block_y, state.pixels);
                        }
                        (8, BlockActivity::BottomHalfZero) => {
                            context
                                .backend
                                .idct_bottom_half_zero(state.coeff.coefficients(), state.pixels);
                            deposit_block(plane, stride, block_x, block_y, state.pixels);
                        }
                        (8, BlockActivity::General) => {
                            context
                                .backend
                                .idct(state.coeff.coefficients(), state.pixels);
                            deposit_block(plane, stride, block_x, block_y, state.pixels);
                        }
                        (4, BlockActivity::DcOnly) => {
                            downscale::idct_islow_4x4_dc_only(
                                state.coeff.dc_coeff(),
                                &mut pixels_4x4,
                            );
                            deposit_block_4x4(plane, stride, block_x, block_y, &pixels_4x4);
                        }
                        (4, _) => {
                            downscale::idct_islow_4x4(state.coeff.coefficients(), &mut pixels_4x4);
                            deposit_block_4x4(plane, stride, block_x, block_y, &pixels_4x4);
                        }
                        (2, BlockActivity::DcOnly) => {
                            downscale::idct_islow_2x2_dc_only(
                                state.coeff.dc_coeff(),
                                &mut pixels_2x2,
                            );
                            deposit_block_2x2(plane, stride, block_x, block_y, pixels_2x2);
                        }
                        (2, _) => {
                            downscale::idct_islow_2x2(state.coeff.coefficients(), &mut pixels_2x2);
                            deposit_block_2x2(plane, stride, block_x, block_y, pixels_2x2);
                        }
                        _ => {
                            debug_assert_eq!(block_size, 1, "IDCT sizes are 8, 4, 2 or 1");
                            let pixel = downscale::idct_islow_1x1(state.coeff.coefficients());
                            deposit_block_1x1(plane, stride, block_x, block_y, pixel);
                        }
                    }
                }
            }
        }
        *state.mcus_since_restart += 1;
    }

    Ok(())
}
