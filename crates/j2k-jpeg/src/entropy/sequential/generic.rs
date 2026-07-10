// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic baseline sequential scan drivers and MCU-row entropy decode.

use super::deposit::{
    assert_stripe_deposit_capacity, deposit_block, deposit_block_1x1, deposit_block_2x2,
    deposit_block_4x4,
};
use super::emit::{emit_stripe, emit_stripe_rgb, StripeEmit};
use super::layout::{
    component_block_intersects_rect, decode_mcu_row_end_for_rect, expanded_output_rect,
    fast420_decode_mcu_row_end, fast420_first_decode_mcu_row, first_decode_mcu_row_for_rect,
    is_ycbcr_420, last_mcu_row_for_rect, mcu_row_intersects_rect, scaled_dimensions,
    stripe_region_layout, ComponentBlockPosition,
};
use super::restart::{
    consume_restart_marker_if_due, finish_scan, restart_seek_for_mcu, skip_to_mcu, McuPosition,
    McuSkipState, McuSkipTarget,
};
use super::{OutputScratch, PreparedDecodePlan, RgbOutputScratch, StripeBuffer};
use crate::backend::Backend;
use crate::entropy::block::{
    decode_block_with_activity, skip_block, BlockActivity, CoefficientBlock,
};
use crate::error::{JpegError, Warning};
use crate::idct::downscale;
use crate::info::{ColorSpace, DownscaleFactor, Rect};
use crate::internal::bit_reader::BitReader;
use crate::internal::scratch::ScratchPool;
use crate::output::{InterleavedRgbWriter, OutputWriter};
use alloc::vec::Vec;

pub(crate) fn decode_scan_baseline<W: OutputWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    downscale: DownscaleFactor,
    output_rect: Rect,
) -> Result<Vec<Warning>, JpegError> {
    let (width, height) = scaled_dimensions(plan.dimensions, downscale);
    let max_h = plan.sampling.max_h as u32;
    let max_v = plan.sampling.max_v as u32;
    let block_size = downscale.output_block_size();
    let mcu_width_px = block_size * max_h;
    let mcu_height_px = block_size * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);

    let region_layout = stripe_region_layout(plan, downscale, output_rect);

    pool.prepare_for(plan, region_layout.stripe_mcus_per_row, block_size);

    let mut br = BitReader::new(scan_bytes);
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let ScratchPool {
        prev_dc,
        stripe_a,
        stripe_b,
        stripe_c,
        ycbcr_420_rows,
        ycbcr_generic_rows,
        rgb_generic_rows,
        ..
    } = pool;
    let mut output_scratch = match plan.color_space {
        ColorSpace::Grayscale => OutputScratch::Grayscale,
        ColorSpace::YCbCr if is_ycbcr_420(plan) => OutputScratch::YCbCr420(ycbcr_420_rows),
        ColorSpace::YCbCr => OutputScratch::YCbCrGeneric(ycbcr_generic_rows),
        ColorSpace::Rgb => OutputScratch::RgbGeneric(rgb_generic_rows),
        ColorSpace::Cmyk | ColorSpace::Ycck => OutputScratch::RgbGeneric(rgb_generic_rows),
    };
    let mut prev_stripe: &mut StripeBuffer = stripe_a;
    let mut curr_stripe: &mut StripeBuffer = stripe_b;
    let mut next_stripe: &mut StripeBuffer = stripe_c;

    let restart = plan.restart_interval.unwrap_or(0);
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;
    let expanded_rect = expanded_output_rect(output_rect, width, height);
    let full_output_rect = expanded_rect == Rect::full((width, height));
    let first_decode_mcu_row =
        first_decode_mcu_row_for_rect(full_output_rect, expanded_rect, mcu_height_px);
    let decode_mcu_row_end =
        decode_mcu_row_end_for_rect(full_output_rect, expanded_rect, mcu_height_px, mcu_rows);
    let last_output_mcu_row = last_mcu_row_for_rect(expanded_rect, mcu_height_px, mcu_rows);
    let total_mcus = mcu_rows * mcus_per_row;
    let first_decode_mcu = first_decode_mcu_row * mcus_per_row;
    let mut current_mcu = 0u32;
    if let Some(seek) = restart_seek_for_mcu(scan_bytes, restart, first_decode_mcu) {
        br = BitReader::new(&scan_bytes[seek.scan_offset..]);
        current_mcu = seek.mcu_index;
        expected_rst = seek.expected_rst;
    }
    skip_to_mcu(
        plan,
        McuSkipTarget {
            target_mcu: first_decode_mcu,
            total_mcus,
            restart,
        },
        &mut McuSkipState {
            br: &mut br,
            prev_dc,
            current_mcu: &mut current_mcu,
            mcus_since_restart: &mut mcus_since_restart,
            expected_rst: &mut expected_rst,
        },
    )?;

    let row_context = McuRowContext {
        plan,
        backend,
        downscale,
        output_rect: expanded_rect,
        full_output_rect,
        stripe_mcu_start: region_layout.stripe_mcu_start,
        stripe_mcus_per_row: region_layout.stripe_mcus_per_row,
        mcus_per_row,
        mcu_rows,
        restart,
    };
    let mut has_prev = false;
    {
        let mut row_state = McuRowState {
            br: &mut br,
            prev_dc,
            coeff: &mut coeff,
            pixels: &mut pixels,
            mcus_since_restart: &mut mcus_since_restart,
            expected_rst: &mut expected_rst,
        };

        decode_mcu_row(
            &row_context,
            &mut row_state,
            first_decode_mcu_row,
            curr_stripe,
        )?;

        for my in first_decode_mcu_row + 1..decode_mcu_row_end {
            decode_mcu_row(&row_context, &mut row_state, my, next_stripe)?;
            if full_output_rect || mcu_row_intersects_rect(my - 1, mcu_height_px, expanded_rect) {
                emit_stripe(
                    plan,
                    writer,
                    &mut output_scratch,
                    StripeEmit {
                        prev: has_prev.then_some(&*prev_stripe),
                        curr: curr_stripe,
                        next: Some(&*next_stripe),
                        stripe_index: my - 1,
                        source_width: region_layout.source_width_usize(),
                        downscale,
                    },
                )?;
            }
            core::mem::swap(&mut prev_stripe, &mut curr_stripe);
            core::mem::swap(&mut curr_stripe, &mut next_stripe);
            has_prev = true;
        }
    }

    let curr_mcu_row = decode_mcu_row_end - 1;
    if curr_mcu_row <= last_output_mcu_row
        && (full_output_rect || mcu_row_intersects_rect(curr_mcu_row, mcu_height_px, expanded_rect))
    {
        emit_stripe(
            plan,
            writer,
            &mut output_scratch,
            StripeEmit {
                prev: has_prev.then_some(&*prev_stripe),
                curr: curr_stripe,
                next: None,
                stripe_index: curr_mcu_row,
                source_width: region_layout.source_width_usize(),
                downscale,
            },
        )?;
    }
    finish_scan(&mut br, decode_mcu_row_end == mcu_rows)
}

pub(crate) fn decode_scan_baseline_rgb<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    downscale: DownscaleFactor,
    output_rect: Rect,
) -> Result<Vec<Warning>, JpegError> {
    let (width, height) = scaled_dimensions(plan.dimensions, downscale);
    let max_h = plan.sampling.max_h as u32;
    let max_v = plan.sampling.max_v as u32;
    let block_size = downscale.output_block_size();
    let mcu_width_px = block_size * max_h;
    let mcu_height_px = block_size * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);

    let region_layout = stripe_region_layout(plan, downscale, output_rect);

    pool.prepare_for(plan, region_layout.stripe_mcus_per_row, block_size);

    let mut br = BitReader::new(scan_bytes);
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let ScratchPool {
        prev_dc,
        stripe_a,
        stripe_b,
        stripe_c,
        ycbcr_generic_rows,
        rgb_generic_rows,
        ..
    } = pool;
    let mut output_scratch = match plan.color_space {
        ColorSpace::Grayscale => RgbOutputScratch::None,
        ColorSpace::YCbCr if is_ycbcr_420(plan) => RgbOutputScratch::YCbCr420,
        ColorSpace::YCbCr => RgbOutputScratch::YCbCrGeneric(ycbcr_generic_rows),
        ColorSpace::Rgb => RgbOutputScratch::RgbGeneric(rgb_generic_rows),
        ColorSpace::Cmyk | ColorSpace::Ycck => RgbOutputScratch::RgbGeneric(rgb_generic_rows),
    };
    let mut prev_stripe: &mut StripeBuffer = stripe_a;
    let mut curr_stripe: &mut StripeBuffer = stripe_b;
    let mut next_stripe: &mut StripeBuffer = stripe_c;

    let restart = plan.restart_interval.unwrap_or(0);
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;
    let expanded_rect = expanded_output_rect(output_rect, width, height);
    let full_output_rect = expanded_rect == Rect::full((width, height));
    let use_420_context_window = !full_output_rect && is_ycbcr_420(plan);
    let emit_rect = if use_420_context_window {
        output_rect
    } else {
        expanded_rect
    };
    let first_decode_mcu_row = if use_420_context_window {
        fast420_first_decode_mcu_row(output_rect, mcu_height_px)
    } else {
        first_decode_mcu_row_for_rect(full_output_rect, expanded_rect, mcu_height_px)
    };
    let decode_mcu_row_end = if use_420_context_window {
        fast420_decode_mcu_row_end(output_rect, mcu_height_px, mcu_rows)
    } else {
        decode_mcu_row_end_for_rect(full_output_rect, expanded_rect, mcu_height_px, mcu_rows)
    };
    let last_output_mcu_row = last_mcu_row_for_rect(emit_rect, mcu_height_px, mcu_rows);
    let total_mcus = mcu_rows * mcus_per_row;
    let first_decode_mcu = first_decode_mcu_row * mcus_per_row;
    let mut current_mcu = 0u32;
    if let Some(seek) = restart_seek_for_mcu(scan_bytes, restart, first_decode_mcu) {
        br = BitReader::new(&scan_bytes[seek.scan_offset..]);
        current_mcu = seek.mcu_index;
        expected_rst = seek.expected_rst;
    }
    skip_to_mcu(
        plan,
        McuSkipTarget {
            target_mcu: first_decode_mcu,
            total_mcus,
            restart,
        },
        &mut McuSkipState {
            br: &mut br,
            prev_dc,
            current_mcu: &mut current_mcu,
            mcus_since_restart: &mut mcus_since_restart,
            expected_rst: &mut expected_rst,
        },
    )?;

    let row_context = McuRowContext {
        plan,
        backend,
        downscale,
        output_rect: expanded_rect,
        full_output_rect,
        stripe_mcu_start: region_layout.stripe_mcu_start,
        stripe_mcus_per_row: region_layout.stripe_mcus_per_row,
        mcus_per_row,
        mcu_rows,
        restart,
    };
    let mut has_prev = false;
    {
        let mut row_state = McuRowState {
            br: &mut br,
            prev_dc,
            coeff: &mut coeff,
            pixels: &mut pixels,
            mcus_since_restart: &mut mcus_since_restart,
            expected_rst: &mut expected_rst,
        };

        decode_mcu_row(
            &row_context,
            &mut row_state,
            first_decode_mcu_row,
            curr_stripe,
        )?;

        for my in first_decode_mcu_row + 1..decode_mcu_row_end {
            decode_mcu_row(&row_context, &mut row_state, my, next_stripe)?;
            if full_output_rect || mcu_row_intersects_rect(my - 1, mcu_height_px, emit_rect) {
                emit_stripe_rgb(
                    plan,
                    backend,
                    writer,
                    &mut output_scratch,
                    StripeEmit {
                        prev: has_prev.then_some(&*prev_stripe),
                        curr: curr_stripe,
                        next: Some(&*next_stripe),
                        stripe_index: my - 1,
                        source_width: region_layout.source_width_usize(),
                        downscale,
                    },
                )?;
            }
            core::mem::swap(&mut prev_stripe, &mut curr_stripe);
            core::mem::swap(&mut curr_stripe, &mut next_stripe);
            has_prev = true;
        }
    }

    let curr_mcu_row = decode_mcu_row_end - 1;
    if curr_mcu_row <= last_output_mcu_row
        && (full_output_rect || mcu_row_intersects_rect(curr_mcu_row, mcu_height_px, emit_rect))
    {
        emit_stripe_rgb(
            plan,
            backend,
            writer,
            &mut output_scratch,
            StripeEmit {
                prev: has_prev.then_some(&*prev_stripe),
                curr: curr_stripe,
                next: None,
                stripe_index: curr_mcu_row,
                source_width: region_layout.source_width_usize(),
                downscale,
            },
        )?;
    }
    finish_scan(&mut br, decode_mcu_row_end == mcu_rows)
}

struct McuRowContext<'a> {
    plan: &'a PreparedDecodePlan,
    backend: Backend,
    downscale: DownscaleFactor,
    output_rect: Rect,
    full_output_rect: bool,
    stripe_mcu_start: u32,
    stripe_mcus_per_row: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart: u16,
}

struct McuRowState<'a, 'b> {
    br: &'a mut BitReader<'b>,
    prev_dc: &'a mut [i32],
    coeff: &'a mut CoefficientBlock,
    pixels: &'a mut [u8; 64],
    mcus_since_restart: &'a mut u32,
    expected_rst: &'a mut u8,
}

fn decode_mcu_row(
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
            let in_region = mx >= context.stripe_mcu_start && mx < stripe_mcu_end;
            let local_mcu_x0_px =
                mx.saturating_sub(context.stripe_mcu_start) * u32::from(comp.h) * block_size;
            for vy in 0..comp.v as u32 {
                for vx in 0..comp.h as u32 {
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
                        skip_block(
                            state.br,
                            &comp.dc_table,
                            &comp.ac_table,
                            &mut state.prev_dc[plane_idx],
                        )?;
                        continue;
                    }

                    let activity = decode_block_with_activity(
                        state.br,
                        &comp.dc_table,
                        &comp.ac_table,
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
