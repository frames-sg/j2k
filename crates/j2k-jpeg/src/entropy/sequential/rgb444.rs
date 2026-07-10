// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast sequential RGB output route for 4:4:4 YCbCr scans.

use super::deposit::{assert_stripe_deposit_capacity, deposit_block};
use super::emit::emit_stripe_rgb_444;
use super::restart::{consume_restart_marker_if_due, McuPosition};
use super::{PreparedComponentPlan, PreparedDecodePlan, StripeBuffer};
use crate::backend::Backend;
use crate::entropy::block::{decode_block_with_activity, BlockActivity, CoefficientBlock};
use crate::error::{JpegError, Warning};
use crate::info::DownscaleFactor;
use crate::internal::bit_reader::BitReader;
use crate::internal::scratch::ScratchPool;
use crate::output::{InterleavedRgbWriter, OutputWriter};
use alloc::vec::Vec;

fn fast_rgb444_components(
    plan: &PreparedDecodePlan,
) -> (
    &PreparedComponentPlan,
    &PreparedComponentPlan,
    &PreparedComponentPlan,
) {
    debug_assert!(plan.matches_fast_rgb444_shape());
    (
        &plan.components[0],
        &plan.components[1],
        &plan.components[2],
    )
}

pub(crate) fn decode_scan_fast_rgb_444<W: OutputWriter + InterleavedRgbWriter>(
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
        mcus_per_row,
        DownscaleFactor::Full.output_block_size(),
    );

    let mut br = BitReader::new(scan_bytes);
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let (y_comp, cb_comp, cr_comp) = fast_rgb444_components(plan);
    let mut y_dc = 0i32;
    let mut cb_dc = 0i32;
    let mut cr_dc = 0i32;
    let restart = plan.restart_interval.unwrap_or(0);
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;
    let stripe = &mut pool.stripe_a;
    let row_context = FastRgb444McuRowContext {
        y_comp,
        cb_comp,
        cr_comp,
        backend,
        mcus_per_row,
        mcu_rows,
        restart,
    };

    {
        let mut row_state = FastRgb444McuRowState {
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
            decode_mcu_row_fast_rgb_444(&row_context, &mut row_state, my, stripe)?;
            emit_stripe_rgb_444(plan, backend, stripe, my, writer)?;
        }
    }

    let mut warnings = Vec::new();
    match br.take_marker() {
        Some(0xD9) => Ok(warnings),
        Some(found) => Err(JpegError::UnexpectedMarker {
            offset: br.position().saturating_sub(2),
            expected: crate::error::MarkerKind::Eoi,
            found,
        }),
        None => {
            warnings.push(Warning::MissingEoi);
            Ok(warnings)
        }
    }
}

struct FastRgb444McuRowContext<'a> {
    y_comp: &'a PreparedComponentPlan,
    cb_comp: &'a PreparedComponentPlan,
    cr_comp: &'a PreparedComponentPlan,
    backend: Backend,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart: u16,
}

struct FastRgb444McuRowState<'a, 'b> {
    br: &'a mut BitReader<'b>,
    y_dc: &'a mut i32,
    cb_dc: &'a mut i32,
    cr_dc: &'a mut i32,
    coeff: &'a mut CoefficientBlock,
    pixels: &'a mut [u8; 64],
    mcus_since_restart: &'a mut u32,
    expected_rst: &'a mut u8,
}

#[expect(
    clippy::too_many_lines,
    reason = "the 4:4:4 MCU kernel keeps three block decodes and direct RGB row deposition in sampling order"
)]
fn decode_mcu_row_fast_rgb_444(
    context: &FastRgb444McuRowContext<'_>,
    state: &mut FastRgb444McuRowState<'_, '_>,
    mcu_y: u32,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    for plane_idx in 0..3 {
        assert_stripe_deposit_capacity(stripe, plane_idx, 1, 1, context.mcus_per_row, 8);
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
            *state.y_dc = 0;
            *state.cb_dc = 0;
            *state.cr_dc = 0;
            *state.mcus_since_restart = 0;
        }

        let block_x = mx * 8;

        let y_activity = decode_block_with_activity(
            state.br,
            &context.y_comp.dc_table,
            &context.y_comp.ac_table,
            state.y_dc,
            context.y_comp.quant.as_ref(),
            state.coeff,
        )?;
        match y_activity {
            BlockActivity::DcOnly => {
                crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), state.pixels);
            }
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

        let cb_activity = decode_block_with_activity(
            state.br,
            &context.cb_comp.dc_table,
            &context.cb_comp.ac_table,
            state.cb_dc,
            context.cb_comp.quant.as_ref(),
            state.coeff,
        )?;
        match cb_activity {
            BlockActivity::DcOnly => {
                crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), state.pixels);
            }
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
            state.br,
            &context.cr_comp.dc_table,
            &context.cr_comp.ac_table,
            state.cr_dc,
            context.cr_comp.quant.as_ref(),
            state.coeff,
        )?;
        match cr_activity {
            BlockActivity::DcOnly => {
                crate::idct::idct_islow_dc_only(state.coeff.dc_coeff(), state.pixels);
            }
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
