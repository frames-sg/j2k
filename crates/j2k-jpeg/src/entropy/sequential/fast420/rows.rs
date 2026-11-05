// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy and IDCT row kernels for the fast 4:2:0 sequential route.

use super::super::deposit::{
    assert_stripe_deposit_capacity, idct_deposit_fast_tile_block, FastRowState,
    FastTile420Components, FastTile420Window, PlaneBlockTarget,
};
use super::super::profile::Fast420Profiler;
use super::super::StripeBuffer;
use super::FAST420_HALO_ROWS as HALO_ROWS;
use crate::backend::Backend;
use crate::entropy::block::{decode_block_with_activity, skip_block};
use crate::error::JpegError;

/// Decode one MCU row into `stripe`, below its halo row.
pub(in crate::entropy::sequential) fn decode_mcu_row_fast_tile_420<const N: usize>(
    components: FastTile420Components<'_, N>,
    backend: Backend,
    state: &mut FastRowState<'_, '_>,
    window: FastTile420Window,
    mcu_y: u32,
    stripe: &mut StripeBuffer,
    profiler: &mut impl Fast420Profiler,
) -> Result<(), JpegError> {
    let clip_padding = !(window.right_luma && window.bottom_luma);
    match (clip_padding, state.restart.interval != 0) {
        (false, false) => decode_mcu_row_fast_tile_420_specialized::<N, false, false>(
            components, backend, state, window, mcu_y, stripe, profiler,
        ),
        (false, true) => decode_mcu_row_fast_tile_420_specialized::<N, false, true>(
            components, backend, state, window, mcu_y, stripe, profiler,
        ),
        (true, false) => decode_mcu_row_fast_tile_420_specialized::<N, true, false>(
            components, backend, state, window, mcu_y, stripe, profiler,
        ),
        (true, true) => decode_mcu_row_fast_tile_420_specialized::<N, true, true>(
            components, backend, state, window, mcu_y, stripe, profiler,
        ),
    }
}

fn decode_mcu_row_fast_tile_420_specialized<
    const N: usize,
    const CLIP_PADDING: bool,
    const RESTART: bool,
>(
    components: FastTile420Components<'_, N>,
    backend: Backend,
    state: &mut FastRowState<'_, '_>,
    window: FastTile420Window,
    mcu_y: u32,
    stripe: &mut StripeBuffer,
    profiler: &mut impl Fast420Profiler,
) -> Result<(), JpegError> {
    let mut local = state.take_local();
    let result = decode_mcu_row_fast_tile_420_local::<N, CLIP_PADDING, RESTART>(
        components,
        backend,
        &mut local.state(state),
        window,
        mcu_y,
        stripe,
        profiler,
    );
    state.restore_local(local);
    result
}

#[expect(
    clippy::too_many_lines,
    reason = "the MCU-row kernel keeps six block decodes and plane deposits in JPEG sampling order"
)]
#[expect(
    clippy::inline_always,
    reason = "inlining into the row wrapper is what keeps the local reader out of memory"
)]
#[inline(always)]
fn decode_mcu_row_fast_tile_420_local<
    const N: usize,
    const CLIP_PADDING: bool,
    const RESTART: bool,
>(
    components: FastTile420Components<'_, N>,
    backend: Backend,
    state: &mut FastRowState<'_, '_>,
    window: FastTile420Window,
    mcu_y: u32,
    stripe: &mut StripeBuffer,
    profiler: &mut impl Fast420Profiler,
) -> Result<(), JpegError> {
    let columns = window.columns;
    let br = &mut *state.br;
    let [y_dc, cb_dc, cr_dc] = &mut *state.dc;
    let coeff = &mut *state.coeff;
    let restart = &mut *state.restart;
    for (plane_idx, h, v) in [(0, 2, 2), (1, 1, 1), (2, 1, 1)] {
        assert_stripe_deposit_capacity(
            stripe,
            plane_idx,
            h,
            v,
            columns.stripe_mcus_per_row,
            8,
            HALO_ROWS,
        );
    }
    let mut pixels = [0u8; 64];
    for mx in 0..columns.mcus_per_row {
        if RESTART && restart.begin_mcu(&mut *br, columns.mcu_position(mx, mcu_y))? {
            *y_dc = 0;
            *cb_dc = 0;
            *cr_dc = 0;
        }
        if !columns.contains(mx) {
            // Direct skips keep the DC predictors in registers; see
            // `FastTile420Components::skip_mcu`.
            for _ in 0..4 {
                skip_block(
                    &mut *br,
                    components.y.dc_table,
                    components.y.ac_table,
                    &mut *y_dc,
                )?;
            }
            skip_block(
                &mut *br,
                components.cb.dc_table,
                components.cb.ac_table,
                &mut *cb_dc,
            )?;
            skip_block(
                &mut *br,
                components.cr.dc_table,
                components.cr.ac_table,
                &mut *cr_dc,
            )?;
            if RESTART {
                restart.end_mcu();
            }
            continue;
        }

        let local_mx = columns.local(mx);
        let y_x = local_mx * 16;
        let c_x = local_mx * 8;
        let emit_right = !CLIP_PADDING || window.right_luma || mx + 1 < columns.mcus_per_row;
        let emit_bottom = !CLIP_PADDING || window.bottom_luma;

        let y0_activity = decode_block_with_activity(
            &mut *br,
            components.y.dc_table,
            components.y.ac_table,
            &mut *y_dc,
            components.y.quant,
            coeff,
        )?;
        profiler.record_activity(y0_activity);
        idct_deposit_fast_tile_block(
            y0_activity,
            backend,
            coeff,
            &mut pixels,
            PlaneBlockTarget {
                plane: &mut stripe.planes[0],
                stride: stripe.plane_strides[0],
                x: y_x,
                y: HALO_ROWS,
            },
        );

        let y1_activity = decode_block_with_activity(
            &mut *br,
            components.y.dc_table,
            components.y.ac_table,
            &mut *y_dc,
            components.y.quant,
            coeff,
        )?;
        profiler.record_activity(y1_activity);
        if emit_right {
            idct_deposit_fast_tile_block(
                y1_activity,
                backend,
                coeff,
                &mut pixels,
                PlaneBlockTarget {
                    plane: &mut stripe.planes[0],
                    stride: stripe.plane_strides[0],
                    x: y_x + 8,
                    y: HALO_ROWS,
                },
            );
        }

        let y2_activity = decode_block_with_activity(
            &mut *br,
            components.y.dc_table,
            components.y.ac_table,
            &mut *y_dc,
            components.y.quant,
            coeff,
        )?;
        profiler.record_activity(y2_activity);
        if emit_bottom {
            idct_deposit_fast_tile_block(
                y2_activity,
                backend,
                coeff,
                &mut pixels,
                PlaneBlockTarget {
                    plane: &mut stripe.planes[0],
                    stride: stripe.plane_strides[0],
                    x: y_x,
                    y: HALO_ROWS + 8,
                },
            );
        }

        let y3_activity = decode_block_with_activity(
            &mut *br,
            components.y.dc_table,
            components.y.ac_table,
            &mut *y_dc,
            components.y.quant,
            coeff,
        )?;
        profiler.record_activity(y3_activity);
        if emit_right && emit_bottom {
            idct_deposit_fast_tile_block(
                y3_activity,
                backend,
                coeff,
                &mut pixels,
                PlaneBlockTarget {
                    plane: &mut stripe.planes[0],
                    stride: stripe.plane_strides[0],
                    x: y_x + 8,
                    y: HALO_ROWS + 8,
                },
            );
        }

        let cb_activity = decode_block_with_activity(
            &mut *br,
            components.cb.dc_table,
            components.cb.ac_table,
            &mut *cb_dc,
            components.cb.quant,
            coeff,
        )?;
        profiler.record_activity(cb_activity);
        idct_deposit_fast_tile_block(
            cb_activity,
            backend,
            coeff,
            &mut pixels,
            PlaneBlockTarget {
                plane: &mut stripe.planes[1],
                stride: stripe.plane_strides[1],
                x: c_x,
                y: HALO_ROWS,
            },
        );

        let cr_activity = decode_block_with_activity(
            &mut *br,
            components.cr.dc_table,
            components.cr.ac_table,
            &mut *cr_dc,
            components.cr.quant,
            coeff,
        )?;
        profiler.record_activity(cr_activity);
        idct_deposit_fast_tile_block(
            cr_activity,
            backend,
            coeff,
            &mut pixels,
            PlaneBlockTarget {
                plane: &mut stripe.planes[2],
                stride: stripe.plane_strides[2],
                x: c_x,
                y: HALO_ROWS,
            },
        );

        if RESTART {
            restart.end_mcu();
        }
    }

    Ok(())
}
