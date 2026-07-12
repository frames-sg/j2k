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

#[expect(
    clippy::too_many_lines,
    reason = "the MCU kernel traverses component sampling factors while preserving entropy, predictor, IDCT, and deposit order"
)]
pub(super) fn decode_mcu_row(
    context: &McuRowContext<'_>,
    state: &mut McuRowState<'_, '_>,
    mcu_y: u32,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    let stripe_mcu_end = context.stripe_mcu_start + context.stripe_mcus_per_row;
    let block_size = context.downscale.output_block_size();
    for comp in &context.plan.components {
        assert_stripe_deposit_capacity(
            stripe,
            comp.output_index,
            u32::from(comp.h),
            u32::from(comp.v),
            context.stripe_mcus_per_row,
            block_size,
        );
    }
    let mut pixels_4x4 = [0u8; 16];
    let mut pixels_2x2 = [0u8; 4];
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

        for comp in &context.plan.components {
            let plane_idx = comp.output_index;
            let dc_table = context.plan.dc_table(comp)?;
            let ac_table = context.plan.ac_table(comp)?;
            let in_region = mx >= context.stripe_mcu_start && mx < stripe_mcu_end;
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
                    match context.downscale {
                        DownscaleFactor::Full => {
                            match activity {
                                BlockActivity::DcOnly => {
                                    crate::idct::idct_islow_dc_only(
                                        state.coeff.dc_coeff(),
                                        state.pixels,
                                    );
                                }
                                BlockActivity::BottomHalfZero => {
                                    context.backend.idct_bottom_half_zero(
                                        state.coeff.coefficients(),
                                        state.pixels,
                                    );
                                }
                                BlockActivity::General => {
                                    context
                                        .backend
                                        .idct(state.coeff.coefficients(), state.pixels);
                                }
                            }
                            deposit_block(
                                &mut stripe.planes[plane_idx],
                                stripe.plane_strides[plane_idx],
                                block_x,
                                block_y,
                                state.pixels,
                            );
                        }
                        DownscaleFactor::Half => {
                            if activity == BlockActivity::DcOnly {
                                downscale::idct_islow_4x4_dc_only(
                                    state.coeff.dc_coeff(),
                                    &mut pixels_4x4,
                                );
                            } else {
                                downscale::idct_islow_4x4(
                                    state.coeff.coefficients(),
                                    &mut pixels_4x4,
                                );
                            }
                            deposit_block_4x4(
                                &mut stripe.planes[plane_idx],
                                stripe.plane_strides[plane_idx],
                                block_x,
                                block_y,
                                &pixels_4x4,
                            );
                        }
                        DownscaleFactor::Quarter => {
                            if activity == BlockActivity::DcOnly {
                                downscale::idct_islow_2x2_dc_only(
                                    state.coeff.dc_coeff(),
                                    &mut pixels_2x2,
                                );
                            } else {
                                downscale::idct_islow_2x2(
                                    state.coeff.coefficients(),
                                    &mut pixels_2x2,
                                );
                            }
                            deposit_block_2x2(
                                &mut stripe.planes[plane_idx],
                                stripe.plane_strides[plane_idx],
                                block_x,
                                block_y,
                                pixels_2x2,
                            );
                        }
                        DownscaleFactor::Eighth => {
                            let pixel = downscale::idct_islow_1x1(state.coeff.coefficients());
                            deposit_block_1x1(
                                &mut stripe.planes[plane_idx],
                                stripe.plane_strides[plane_idx],
                                block_x,
                                block_y,
                                pixel,
                            );
                        }
                    }
                }
            }
        }
        *state.mcus_since_restart += 1;
    }

    Ok(())
}
