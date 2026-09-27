// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy and IDCT row kernels for the fast 4:2:0 sequential route.

use super::super::deposit::{
    assert_stripe_deposit_capacity, idct_deposit_fast_tile_block, FastTile420Components,
    FastTile420EntropyState, FastTile420Window, PlaneBlockTarget,
};
use super::super::profile::Fast420Profiler;
use super::super::{ResolvedPreparedComponentPlan, StripeBuffer};
use crate::backend::Backend;
use crate::entropy::block::{decode_block_with_activity, skip_block};
use crate::error::JpegError;
use crate::internal::bit_reader::BitReader;

// ROI seeks invoke this leaf once per skipped MCU; forced inlining keeps its six Huffman skips
// in the caller's hot loop without adding a branch-and-call boundary.
#[expect(
    clippy::inline_always,
    reason = "ROI seek hot path keeps six Huffman skips inside the caller loop"
)]
#[inline(always)]
pub(super) fn skip_mcu_fast_tile_420(
    y_comp: ResolvedPreparedComponentPlan<'_>,
    cb_comp: ResolvedPreparedComponentPlan<'_>,
    cr_comp: ResolvedPreparedComponentPlan<'_>,
    br: &mut BitReader<'_>,
    y_dc: &mut i32,
    cb_dc: &mut i32,
    cr_dc: &mut i32,
) -> Result<(), JpegError> {
    for _ in 0..4 {
        skip_block(br, y_comp.dc_table, y_comp.ac_table, y_dc)?;
    }
    skip_block(br, cb_comp.dc_table, cb_comp.ac_table, cb_dc)?;
    skip_block(br, cr_comp.dc_table, cr_comp.ac_table, cr_dc)?;
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
    let mut local = state.take_local();
    let result = decode_mcu_row_fast_tile_420_local(
        components,
        backend,
        &mut local.state(&mut *state.coeff),
        pixels,
        window,
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
fn decode_mcu_row_fast_tile_420_local(
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
                    components.y.dc_table,
                    components.y.ac_table,
                    &mut *state.dc.y,
                )?;
            }
            skip_block(
                &mut *state.br,
                components.cb.dc_table,
                components.cb.ac_table,
                &mut *state.dc.cb,
            )?;
            skip_block(
                &mut *state.br,
                components.cr.dc_table,
                components.cr.ac_table,
                &mut *state.dc.cr,
            )?;
            continue;
        }

        let local_mx = window.local_mcu_x(mx);
        let y_x = local_mx * 16;
        let c_x = local_mx * 8;

        let y0_activity = decode_block_with_activity(
            &mut *state.br,
            components.y.dc_table,
            components.y.ac_table,
            &mut *state.dc.y,
            components.y.quant,
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
            components.y.dc_table,
            components.y.ac_table,
            &mut *state.dc.y,
            components.y.quant,
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
            components.y.dc_table,
            components.y.ac_table,
            &mut *state.dc.y,
            components.y.quant,
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
            components.y.dc_table,
            components.y.ac_table,
            &mut *state.dc.y,
            components.y.quant,
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
            components.cb.dc_table,
            components.cb.ac_table,
            &mut *state.dc.cb,
            components.cb.quant,
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
            components.cr.dc_table,
            components.cr.ac_table,
            &mut *state.dc.cr,
            components.cr.quant,
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
