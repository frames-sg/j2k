// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy and IDCT row kernels for the fast 4:2:0 sequential route.

use super::super::deposit::{
    assert_stripe_deposit_capacity, decode_eighth_block_to_plane, decode_quarter_block_to_plane,
    decode_scaled_block_to_plane, idct_deposit_fast_tile_block, EntropyBlockState,
    FastTile420Components, FastTile420EntropyState, FastTile420Window, PlaneBlockTarget,
    ReducedIdctScratch,
};
use super::super::profile::Fast420Profiler;
use super::super::{PreparedComponentPlan, StripeBuffer};
use crate::backend::Backend;
use crate::entropy::block::{decode_block_with_activity, skip_block};
use crate::error::JpegError;
use crate::info::DownscaleFactor;
use crate::internal::bit_reader::BitReader;

// ROI seeks invoke this leaf once per skipped MCU; forced inlining keeps its six Huffman skips
// in the caller's hot loop without adding a branch-and-call boundary.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) fn skip_mcu_fast_tile_420(
    y_comp: &PreparedComponentPlan,
    cb_comp: &PreparedComponentPlan,
    cr_comp: &PreparedComponentPlan,
    br: &mut BitReader<'_>,
    y_dc: &mut i32,
    cb_dc: &mut i32,
    cr_dc: &mut i32,
) -> Result<(), JpegError> {
    for _ in 0..4 {
        skip_block(br, &y_comp.dc_table, &y_comp.ac_table, y_dc)?;
    }
    skip_block(br, &cb_comp.dc_table, &cb_comp.ac_table, cb_dc)?;
    skip_block(br, &cr_comp.dc_table, &cr_comp.ac_table, cr_dc)?;
    Ok(())
}

pub(super) fn decode_mcu_row_fast_tile_420_scaled(
    components: FastTile420Components<'_>,
    state: &mut FastTile420EntropyState<'_, '_>,
    downscale: DownscaleFactor,
    scratch: ReducedIdctScratch<'_>,
    window: FastTile420Window,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    if downscale == DownscaleFactor::Quarter {
        return decode_mcu_row_fast_tile_420_quarter(
            components,
            state,
            scratch.pixels_2x2,
            window,
            stripe,
        );
    }
    if downscale == DownscaleFactor::Eighth {
        return decode_mcu_row_fast_tile_420_eighth(components, state, window, stripe);
    }

    let block_size = downscale.output_block_size();
    let y_stride = stripe.plane_strides[0];
    let cb_stride = stripe.plane_strides[1];
    let cr_stride = stripe.plane_strides[2];

    for mx in 0..window.mcus_per_row {
        if !window.contains_mcu(mx) {
            skip_mcu_fast_tile_420(
                components.y,
                components.cb,
                components.cr,
                &mut *state.br,
                &mut *state.dc.y,
                &mut *state.dc.cb,
                &mut *state.dc.cr,
            )?;
            continue;
        }

        let local_mx = window.local_mcu_x(mx);
        let y_x = local_mx * 2 * block_size;
        let c_x = local_mx * block_size;
        decode_scaled_block_to_plane(
            components.y,
            downscale,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.y,
                coeff: &mut *state.coeff,
            },
            ReducedIdctScratch {
                pixels_4x4: &mut *scratch.pixels_4x4,
                pixels_2x2: &mut *scratch.pixels_2x2,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: y_stride,
                x: y_x,
                y: 0,
            },
        )?;
        decode_scaled_block_to_plane(
            components.y,
            downscale,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.y,
                coeff: &mut *state.coeff,
            },
            ReducedIdctScratch {
                pixels_4x4: &mut *scratch.pixels_4x4,
                pixels_2x2: &mut *scratch.pixels_2x2,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: y_stride,
                x: y_x + block_size,
                y: 0,
            },
        )?;
        decode_scaled_block_to_plane(
            components.y,
            downscale,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.y,
                coeff: &mut *state.coeff,
            },
            ReducedIdctScratch {
                pixels_4x4: &mut *scratch.pixels_4x4,
                pixels_2x2: &mut *scratch.pixels_2x2,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: y_stride,
                x: y_x,
                y: block_size,
            },
        )?;
        decode_scaled_block_to_plane(
            components.y,
            downscale,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.y,
                coeff: &mut *state.coeff,
            },
            ReducedIdctScratch {
                pixels_4x4: &mut *scratch.pixels_4x4,
                pixels_2x2: &mut *scratch.pixels_2x2,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: y_stride,
                x: y_x + block_size,
                y: block_size,
            },
        )?;
        decode_scaled_block_to_plane(
            components.cb,
            downscale,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.cb,
                coeff: &mut *state.coeff,
            },
            ReducedIdctScratch {
                pixels_4x4: &mut *scratch.pixels_4x4,
                pixels_2x2: &mut *scratch.pixels_2x2,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[1],
                stride: cb_stride,
                x: c_x,
                y: 0,
            },
        )?;
        decode_scaled_block_to_plane(
            components.cr,
            downscale,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.cr,
                coeff: &mut *state.coeff,
            },
            ReducedIdctScratch {
                pixels_4x4: &mut *scratch.pixels_4x4,
                pixels_2x2: &mut *scratch.pixels_2x2,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[2],
                stride: cr_stride,
                x: c_x,
                y: 0,
            },
        )?;
    }
    Ok(())
}

pub(in crate::entropy::sequential) fn decode_mcu_row_fast_tile_420(
    components: FastTile420Components<'_>,
    backend: Backend,
    state: &mut FastTile420EntropyState<'_, '_>,
    pixels: &mut [u8; 64],
    window: FastTile420Window,
    stripe: &mut StripeBuffer,
    profiler: &mut impl Fast420Profiler,
) -> Result<(), JpegError> {
    assert_stripe_deposit_capacity(stripe, 0, 2, 2, window.stripe_mcus_per_row, 8);
    assert_stripe_deposit_capacity(stripe, 1, 1, 1, window.stripe_mcus_per_row, 8);
    assert_stripe_deposit_capacity(stripe, 2, 1, 1, window.stripe_mcus_per_row, 8);
    for mx in 0..window.mcus_per_row {
        if !window.contains_mcu(mx) {
            for _ in 0..4 {
                skip_block(
                    &mut *state.br,
                    &components.y.dc_table,
                    &components.y.ac_table,
                    &mut *state.dc.y,
                )?;
            }
            skip_block(
                &mut *state.br,
                &components.cb.dc_table,
                &components.cb.ac_table,
                &mut *state.dc.cb,
            )?;
            skip_block(
                &mut *state.br,
                &components.cr.dc_table,
                &components.cr.ac_table,
                &mut *state.dc.cr,
            )?;
            continue;
        }

        let local_mx = window.local_mcu_x(mx);
        let y_x = local_mx * 16;
        let c_x = local_mx * 8;

        let y0_activity = decode_block_with_activity(
            &mut *state.br,
            &components.y.dc_table,
            &components.y.ac_table,
            &mut *state.dc.y,
            components.y.quant.as_ref(),
            &mut *state.coeff,
        )?;
        profiler.record_activity(y0_activity);
        idct_deposit_fast_tile_block(
            y0_activity,
            backend,
            &*state.coeff,
            pixels,
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: stripe.plane_strides[0],
                x: y_x,
                y: 0,
            },
        );

        let y1_activity = decode_block_with_activity(
            &mut *state.br,
            &components.y.dc_table,
            &components.y.ac_table,
            &mut *state.dc.y,
            components.y.quant.as_ref(),
            &mut *state.coeff,
        )?;
        profiler.record_activity(y1_activity);
        idct_deposit_fast_tile_block(
            y1_activity,
            backend,
            &*state.coeff,
            pixels,
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: stripe.plane_strides[0],
                x: y_x + 8,
                y: 0,
            },
        );

        let y2_activity = decode_block_with_activity(
            &mut *state.br,
            &components.y.dc_table,
            &components.y.ac_table,
            &mut *state.dc.y,
            components.y.quant.as_ref(),
            &mut *state.coeff,
        )?;
        profiler.record_activity(y2_activity);
        idct_deposit_fast_tile_block(
            y2_activity,
            backend,
            &*state.coeff,
            pixels,
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: stripe.plane_strides[0],
                x: y_x,
                y: 8,
            },
        );

        let y3_activity = decode_block_with_activity(
            &mut *state.br,
            &components.y.dc_table,
            &components.y.ac_table,
            &mut *state.dc.y,
            components.y.quant.as_ref(),
            &mut *state.coeff,
        )?;
        profiler.record_activity(y3_activity);
        idct_deposit_fast_tile_block(
            y3_activity,
            backend,
            &*state.coeff,
            pixels,
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: stripe.plane_strides[0],
                x: y_x + 8,
                y: 8,
            },
        );

        let cb_activity = decode_block_with_activity(
            &mut *state.br,
            &components.cb.dc_table,
            &components.cb.ac_table,
            &mut *state.dc.cb,
            components.cb.quant.as_ref(),
            &mut *state.coeff,
        )?;
        profiler.record_activity(cb_activity);
        idct_deposit_fast_tile_block(
            cb_activity,
            backend,
            &*state.coeff,
            pixels,
            PlaneBlockTarget {
                plane: &mut stripe.planes[1],
                stride: stripe.plane_strides[1],
                x: c_x,
                y: 0,
            },
        );

        let cr_activity = decode_block_with_activity(
            &mut *state.br,
            &components.cr.dc_table,
            &components.cr.ac_table,
            &mut *state.dc.cr,
            components.cr.quant.as_ref(),
            &mut *state.coeff,
        )?;
        profiler.record_activity(cr_activity);
        idct_deposit_fast_tile_block(
            cr_activity,
            backend,
            &*state.coeff,
            pixels,
            PlaneBlockTarget {
                plane: &mut stripe.planes[2],
                stride: stripe.plane_strides[2],
                x: c_x,
                y: 0,
            },
        );
    }

    Ok(())
}

fn decode_mcu_row_fast_tile_420_eighth(
    components: FastTile420Components<'_>,
    state: &mut FastTile420EntropyState<'_, '_>,
    window: FastTile420Window,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    const BLOCK_SIZE: u32 = 1;
    let y_stride = stripe.plane_strides[0];
    let cb_stride = stripe.plane_strides[1];
    let cr_stride = stripe.plane_strides[2];

    for mx in 0..window.mcus_per_row {
        if !window.contains_mcu(mx) {
            skip_mcu_fast_tile_420(
                components.y,
                components.cb,
                components.cr,
                &mut *state.br,
                &mut *state.dc.y,
                &mut *state.dc.cb,
                &mut *state.dc.cr,
            )?;
            continue;
        }

        let local_mx = window.local_mcu_x(mx);
        let y_x = local_mx * 2 * BLOCK_SIZE;
        let c_x = local_mx * BLOCK_SIZE;
        decode_eighth_block_to_plane(
            components.y,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.y,
                coeff: &mut *state.coeff,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: y_stride,
                x: y_x,
                y: 0,
            },
        )?;
        decode_eighth_block_to_plane(
            components.y,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.y,
                coeff: &mut *state.coeff,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: y_stride,
                x: y_x + BLOCK_SIZE,
                y: 0,
            },
        )?;
        decode_eighth_block_to_plane(
            components.y,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.y,
                coeff: &mut *state.coeff,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: y_stride,
                x: y_x,
                y: BLOCK_SIZE,
            },
        )?;
        decode_eighth_block_to_plane(
            components.y,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.y,
                coeff: &mut *state.coeff,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: y_stride,
                x: y_x + BLOCK_SIZE,
                y: BLOCK_SIZE,
            },
        )?;
        decode_eighth_block_to_plane(
            components.cb,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.cb,
                coeff: &mut *state.coeff,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[1],
                stride: cb_stride,
                x: c_x,
                y: 0,
            },
        )?;
        decode_eighth_block_to_plane(
            components.cr,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.cr,
                coeff: &mut *state.coeff,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[2],
                stride: cr_stride,
                x: c_x,
                y: 0,
            },
        )?;
    }

    Ok(())
}

fn decode_mcu_row_fast_tile_420_quarter(
    components: FastTile420Components<'_>,
    state: &mut FastTile420EntropyState<'_, '_>,
    pixels_2x2: &mut [u8; 4],
    window: FastTile420Window,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    const BLOCK_SIZE: u32 = 2;
    let y_stride = stripe.plane_strides[0];
    let cb_stride = stripe.plane_strides[1];
    let cr_stride = stripe.plane_strides[2];

    for mx in 0..window.mcus_per_row {
        if !window.contains_mcu(mx) {
            skip_mcu_fast_tile_420(
                components.y,
                components.cb,
                components.cr,
                &mut *state.br,
                &mut *state.dc.y,
                &mut *state.dc.cb,
                &mut *state.dc.cr,
            )?;
            continue;
        }

        let local_mx = window.local_mcu_x(mx);
        let y_x = local_mx * 2 * BLOCK_SIZE;
        let c_x = local_mx * BLOCK_SIZE;
        decode_quarter_block_to_plane(
            components.y,
            pixels_2x2,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.y,
                coeff: &mut *state.coeff,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: y_stride,
                x: y_x,
                y: 0,
            },
        )?;
        decode_quarter_block_to_plane(
            components.y,
            pixels_2x2,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.y,
                coeff: &mut *state.coeff,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: y_stride,
                x: y_x + BLOCK_SIZE,
                y: 0,
            },
        )?;
        decode_quarter_block_to_plane(
            components.y,
            pixels_2x2,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.y,
                coeff: &mut *state.coeff,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: y_stride,
                x: y_x,
                y: BLOCK_SIZE,
            },
        )?;
        decode_quarter_block_to_plane(
            components.y,
            pixels_2x2,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.y,
                coeff: &mut *state.coeff,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: y_stride,
                x: y_x + BLOCK_SIZE,
                y: BLOCK_SIZE,
            },
        )?;
        decode_quarter_block_to_plane(
            components.cb,
            pixels_2x2,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.cb,
                coeff: &mut *state.coeff,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[1],
                stride: cb_stride,
                x: c_x,
                y: 0,
            },
        )?;
        decode_quarter_block_to_plane(
            components.cr,
            pixels_2x2,
            EntropyBlockState {
                br: &mut *state.br,
                prev_dc: &mut *state.dc.cr,
                coeff: &mut *state.coeff,
            },
            PlaneBlockTarget {
                plane: &mut stripe.planes[2],
                stride: cr_stride,
                x: c_x,
                y: 0,
            },
        )?;
    }

    Ok(())
}
