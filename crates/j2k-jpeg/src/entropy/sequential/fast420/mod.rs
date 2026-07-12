// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast 4:2:0 sequential scan drivers, ROI planning, and scaled routing.

use super::deposit::{
    FastTile420Components, FastTile420DcState, FastTile420EntropyState, FastTile420Window,
    ReducedIdctScratch,
};
use super::emit::{
    emit_stripe_rgb, emit_stripe_rgb_420_region, Fast420RegionStripe, StripeEmit, StripeNeighbors,
};
use super::layout::{
    fast420_decode_mcu_row_end, fast420_first_decode_mcu_row, last_mcu_row_for_rect,
    mcu_row_intersects_rect, scaled_dimensions, Fast420RegionLayout,
};
use super::profile::{Fast420ScanProfiler, NoopFast420Profiler, NoopFast420ScanProfile};
use super::restart::{finish_scan, reader_from_checkpoint};
use super::{PreparedDecodePlan, ResolvedPreparedComponentPlan, RgbOutputScratch, StripeBuffer};
use crate::backend::Backend;
#[cfg(feature = "bench-internals")]
use crate::bench_support::BenchFast420Profile;
use crate::entropy::block::CoefficientBlock;
use crate::error::{JpegError, Warning};
use crate::info::{DownscaleFactor, Rect};
use crate::internal::bit_reader::BitReader;
use crate::internal::checkpoint::DeviceCheckpoint;
use crate::internal::scratch::ScratchPool;
use crate::output::{InterleavedRgbWriter, OutputWriter};
use alloc::vec::Vec;

mod rows;

pub(super) use self::rows::decode_mcu_row_fast_tile_420;
use self::rows::{decode_mcu_row_fast_tile_420_scaled, skip_mcu_fast_tile_420};

fn fast_tile_components(
    plan: &PreparedDecodePlan,
) -> Result<
    (
        ResolvedPreparedComponentPlan<'_>,
        ResolvedPreparedComponentPlan<'_>,
        ResolvedPreparedComponentPlan<'_>,
    ),
    JpegError,
> {
    debug_assert!(plan.matches_fast_tile_shape());
    Ok((
        plan.resolved_component(0)?,
        plan.resolved_component(1)?,
        plan.resolved_component(2)?,
    ))
}

pub(crate) fn decode_scan_fast_tile_rgb<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
) -> Result<Vec<Warning>, JpegError> {
    let mut profile = NoopFast420ScanProfile::default();
    decode_scan_fast_tile_rgb_impl(plan, backend, scan_bytes, pool, writer, &mut profile)
}

#[expect(
    clippy::too_many_lines,
    reason = "the fused 4:2:0 scan loop keeps predictor, restart, stripe, and writer state together for hot-path codegen"
)]
fn decode_scan_fast_tile_rgb_impl<W, P>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    profile: &mut P,
) -> Result<Vec<Warning>, JpegError>
where
    W: OutputWriter + InterleavedRgbWriter,
    P: Fast420ScanProfiler,
{
    debug_assert!(plan.matches_fast_tile_shape());

    let (width, height) = plan.dimensions;
    let max_h = u32::from(plan.sampling.max_h);
    let max_v = u32::from(plan.sampling.max_v);
    let mcu_width_px = 8 * max_h;
    let mcu_height_px = 8 * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);

    pool.prepare_for(
        plan,
        mcus_per_row,
        DownscaleFactor::Full.output_block_size(),
        plan.scratch_bytes,
    )?;

    let mut br = BitReader::new(scan_bytes);
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let (y_comp, cb_comp, cr_comp) = fast_tile_components(plan)?;
    let components = FastTile420Components {
        y: y_comp,
        cb: cb_comp,
        cr: cr_comp,
    };
    let window = FastTile420Window {
        mcus_per_row,
        stripe_mcu_start: 0,
        stripe_mcus_per_row: mcus_per_row,
    };
    let mut y_dc = 0i32;
    let mut cb_dc = 0i32;
    let mut cr_dc = 0i32;

    let ScratchPool {
        stripe_a,
        stripe_b,
        stripe_c,
        ..
    } = pool;
    let mut prev_stripe: &mut StripeBuffer = stripe_a;
    let mut curr_stripe: &mut StripeBuffer = stripe_b;
    let mut next_stripe: &mut StripeBuffer = stripe_c;
    let mut output_scratch = RgbOutputScratch::YCbCr420;

    let mcu_timer = profile.begin_mcu_decode();
    decode_mcu_row_fast_tile_420(
        components,
        backend,
        &mut FastTile420EntropyState {
            br: &mut br,
            dc: FastTile420DcState {
                y: &mut y_dc,
                cb: &mut cb_dc,
                cr: &mut cr_dc,
            },
            coeff: &mut coeff,
        },
        &mut pixels,
        window,
        curr_stripe,
        profile.activity_profiler(),
    )?;
    profile.finish_mcu_decode(mcu_timer);

    let mut has_prev = false;
    for my in 1..mcu_rows {
        let mcu_timer = profile.begin_mcu_decode();
        decode_mcu_row_fast_tile_420(
            components,
            backend,
            &mut FastTile420EntropyState {
                br: &mut br,
                dc: FastTile420DcState {
                    y: &mut y_dc,
                    cb: &mut cb_dc,
                    cr: &mut cr_dc,
                },
                coeff: &mut coeff,
            },
            &mut pixels,
            window,
            next_stripe,
            profile.activity_profiler(),
        )?;
        profile.finish_mcu_decode(mcu_timer);

        let emit_timer = profile.begin_rgb_emit();
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
                source_width: width as usize,
                downscale: DownscaleFactor::Full,
            },
        )?;
        profile.finish_rgb_emit(emit_timer);
        core::mem::swap(&mut prev_stripe, &mut curr_stripe);
        core::mem::swap(&mut curr_stripe, &mut next_stripe);
        has_prev = true;
    }

    let emit_timer = profile.begin_rgb_emit();
    emit_stripe_rgb(
        plan,
        backend,
        writer,
        &mut output_scratch,
        StripeEmit {
            prev: has_prev.then_some(&*prev_stripe),
            curr: curr_stripe,
            next: None,
            stripe_index: mcu_rows - 1,
            source_width: width as usize,
            downscale: DownscaleFactor::Full,
        },
    )?;
    profile.finish_rgb_emit(emit_timer);

    let finish_timer = profile.begin_finish_scan();
    let result = finish_scan(&mut br, true);
    profile.finish_finish_scan(finish_timer);
    result
}

#[cfg(feature = "bench-internals")]
pub(crate) fn decode_scan_fast_tile_rgb_profiled<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    profile: &mut BenchFast420Profile,
) -> Result<Vec<Warning>, JpegError> {
    decode_scan_fast_tile_rgb_impl(plan, backend, scan_bytes, pool, writer, profile)
}

#[expect(
    clippy::too_many_lines,
    reason = "the fused region loop keeps ROI seek, restart, stripe, and writer state together for hot-path codegen"
)]
pub(crate) fn decode_scan_fast_tile_rgb_region<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    roi: Rect,
    checkpoint: Option<&DeviceCheckpoint>,
) -> Result<Vec<Warning>, JpegError> {
    debug_assert!(plan.matches_fast_tile_shape());

    let (width, height) = plan.dimensions;
    let max_h = u32::from(plan.sampling.max_h);
    let max_v = u32::from(plan.sampling.max_v);
    let mcu_width_px = 8 * max_h;
    let mcu_height_px = 8 * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);
    let first_decode_mcu_row = fast420_first_decode_mcu_row(roi, mcu_height_px);
    let decode_mcu_row_end = fast420_decode_mcu_row_end(roi, mcu_height_px, mcu_rows);
    let last_output_mcu_row = last_mcu_row_for_rect(roi, mcu_height_px, mcu_rows);

    let region_layout = Fast420RegionLayout::new(width as usize, roi);

    let mut crop_rows = pool.take_sink_rows(
        region_layout.row_width().saturating_mul(3),
        plan.scratch_bytes,
    )?;
    if let Err(error) = pool.prepare_for(
        plan,
        region_layout.stripe_mcus_per_row,
        DownscaleFactor::Full.output_block_size(),
        plan.scratch_bytes,
    ) {
        pool.restore_sink_rows(crop_rows);
        return Err(error);
    }
    let result = (|| {
        let mut coeff = CoefficientBlock::default();
        let mut pixels = [0u8; 64];
        let (y_comp, cb_comp, cr_comp) = fast_tile_components(plan)?;
        let components = FastTile420Components {
            y: y_comp,
            cb: cb_comp,
            cr: cr_comp,
        };
        let window = FastTile420Window {
            mcus_per_row,
            stripe_mcu_start: region_layout.stripe_mcu_start,
            stripe_mcus_per_row: region_layout.stripe_mcus_per_row,
        };
        let target_mcu = first_decode_mcu_row * mcus_per_row;
        let (mut br, prev_dc, start_mcu) =
            reader_from_checkpoint(scan_bytes, checkpoint, target_mcu);
        let mut y_dc = prev_dc[0];
        let mut cb_dc = prev_dc[1];
        let mut cr_dc = prev_dc[2];
        for _ in start_mcu..target_mcu {
            skip_mcu_fast_tile_420(
                y_comp, cb_comp, cr_comp, &mut br, &mut y_dc, &mut cb_dc, &mut cr_dc,
            )?;
        }

        let ScratchPool {
            stripe_a,
            stripe_b,
            stripe_c,
            ..
        } = pool;
        let mut prev_stripe: &mut StripeBuffer = stripe_a;
        let mut curr_stripe: &mut StripeBuffer = stripe_b;
        let mut next_stripe: &mut StripeBuffer = stripe_c;
        let mut profiler = NoopFast420Profiler;

        decode_mcu_row_fast_tile_420(
            components,
            backend,
            &mut FastTile420EntropyState {
                br: &mut br,
                dc: FastTile420DcState {
                    y: &mut y_dc,
                    cb: &mut cb_dc,
                    cr: &mut cr_dc,
                },
                coeff: &mut coeff,
            },
            &mut pixels,
            window,
            curr_stripe,
            &mut profiler,
        )?;

        let mut has_prev = false;
        for my in first_decode_mcu_row + 1..decode_mcu_row_end {
            decode_mcu_row_fast_tile_420(
                components,
                backend,
                &mut FastTile420EntropyState {
                    br: &mut br,
                    dc: FastTile420DcState {
                        y: &mut y_dc,
                        cb: &mut cb_dc,
                        cr: &mut cr_dc,
                    },
                    coeff: &mut coeff,
                },
                &mut pixels,
                window,
                next_stripe,
                &mut profiler,
            )?;
            if mcu_row_intersects_rect(my - 1, mcu_height_px, roi) {
                emit_stripe_rgb_420_region(
                    plan,
                    backend,
                    writer,
                    Fast420RegionStripe {
                        neighbors: StripeNeighbors {
                            prev: has_prev.then_some(&*prev_stripe),
                            curr: curr_stripe,
                            next: Some(&*next_stripe),
                        },
                        stripe_index: my - 1,
                        roi,
                        region_layout,
                        crop_rows: &mut crop_rows,
                        downscale: DownscaleFactor::Full,
                    },
                )?;
            }
            core::mem::swap(&mut prev_stripe, &mut curr_stripe);
            core::mem::swap(&mut curr_stripe, &mut next_stripe);
            has_prev = true;
        }

        let curr_mcu_row = decode_mcu_row_end - 1;
        if curr_mcu_row <= last_output_mcu_row
            && mcu_row_intersects_rect(curr_mcu_row, mcu_height_px, roi)
        {
            emit_stripe_rgb_420_region(
                plan,
                backend,
                writer,
                Fast420RegionStripe {
                    neighbors: StripeNeighbors {
                        prev: has_prev.then_some(&*prev_stripe),
                        curr: curr_stripe,
                        next: None,
                    },
                    stripe_index: curr_mcu_row,
                    roi,
                    region_layout,
                    crop_rows: &mut crop_rows,
                    downscale: DownscaleFactor::Full,
                },
            )?;
        }
        finish_scan(&mut br, decode_mcu_row_end == mcu_rows)
    })();
    pool.restore_sink_rows(crop_rows);
    result
}

#[derive(Clone, Copy)]
pub(crate) struct FastTileRegionScaledRequest<'a> {
    pub(crate) roi: Rect,
    pub(crate) downscale: DownscaleFactor,
    pub(crate) checkpoint: Option<&'a DeviceCheckpoint>,
}

#[expect(
    clippy::too_many_lines,
    reason = "the fused scaled-region loop keeps ROI seek, reduced IDCT, restart, and writer state in decode order"
)]
pub(crate) fn decode_scan_fast_tile_rgb_region_scaled<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    request: FastTileRegionScaledRequest<'_>,
) -> Result<Vec<Warning>, JpegError> {
    let FastTileRegionScaledRequest {
        roi,
        downscale,
        checkpoint,
    } = request;
    debug_assert!(plan.matches_fast_tile_shape());
    debug_assert!(downscale != DownscaleFactor::Full);

    let (width, height) = scaled_dimensions(plan.dimensions, downscale);
    let max_h = u32::from(plan.sampling.max_h);
    let max_v = u32::from(plan.sampling.max_v);
    let block_size = downscale.output_block_size();
    let mcu_width_px = block_size * max_h;
    let mcu_height_px = block_size * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);
    let first_decode_mcu_row = fast420_first_decode_mcu_row(roi, mcu_height_px);
    let decode_mcu_row_end = fast420_decode_mcu_row_end(roi, mcu_height_px, mcu_rows);
    let last_output_mcu_row = last_mcu_row_for_rect(roi, mcu_height_px, mcu_rows);

    let region_layout = Fast420RegionLayout::new_for_mcu_width(width as usize, roi, mcu_width_px);

    let mut crop_rows = pool.take_sink_rows(
        region_layout.row_width().saturating_mul(3),
        plan.scratch_bytes,
    )?;
    if let Err(error) = pool.prepare_for(
        plan,
        region_layout.stripe_mcus_per_row,
        block_size,
        plan.scratch_bytes,
    ) {
        pool.restore_sink_rows(crop_rows);
        return Err(error);
    }
    let result = (|| {
        let mut coeff = CoefficientBlock::default();
        let ScratchPool {
            prev_dc,
            stripe_a,
            stripe_b,
            stripe_c,
            ..
        } = pool;
        let mut pixels_4x4 = [0u8; 16];
        let mut pixels_2x2 = [0u8; 4];
        let (y_dc_slice, rest_dc) = prev_dc.split_at_mut(1);
        let (cb_dc_slice, cr_dc_slice) = rest_dc.split_at_mut(1);
        let y_dc = &mut y_dc_slice[0];
        let cb_dc = &mut cb_dc_slice[0];
        let cr_dc = &mut cr_dc_slice[0];
        let target_mcu = first_decode_mcu_row * mcus_per_row;
        let (mut br, checkpoint_dc, start_mcu) =
            reader_from_checkpoint(scan_bytes, checkpoint, target_mcu);
        *y_dc = checkpoint_dc[0];
        *cb_dc = checkpoint_dc[1];
        *cr_dc = checkpoint_dc[2];
        let (y, cb, cr) = fast_tile_components(plan)?;
        let components = FastTile420Components { y, cb, cr };
        let window = FastTile420Window {
            mcus_per_row,
            stripe_mcu_start: region_layout.stripe_mcu_start,
            stripe_mcus_per_row: region_layout.stripe_mcus_per_row,
        };
        for _ in start_mcu..target_mcu {
            skip_mcu_fast_tile_420(
                components.y,
                components.cb,
                components.cr,
                &mut br,
                &mut *y_dc,
                &mut *cb_dc,
                &mut *cr_dc,
            )?;
        }

        let mut prev_stripe: &mut StripeBuffer = stripe_a;
        let mut curr_stripe: &mut StripeBuffer = stripe_b;
        let mut next_stripe: &mut StripeBuffer = stripe_c;

        decode_mcu_row_fast_tile_420_scaled(
            components,
            &mut FastTile420EntropyState {
                br: &mut br,
                dc: FastTile420DcState {
                    y: &mut *y_dc,
                    cb: &mut *cb_dc,
                    cr: &mut *cr_dc,
                },
                coeff: &mut coeff,
            },
            downscale,
            ReducedIdctScratch {
                pixels_4x4: &mut pixels_4x4,
                pixels_2x2: &mut pixels_2x2,
            },
            window,
            curr_stripe,
        )?;

        let mut has_prev = false;
        for my in first_decode_mcu_row + 1..decode_mcu_row_end {
            decode_mcu_row_fast_tile_420_scaled(
                components,
                &mut FastTile420EntropyState {
                    br: &mut br,
                    dc: FastTile420DcState {
                        y: &mut *y_dc,
                        cb: &mut *cb_dc,
                        cr: &mut *cr_dc,
                    },
                    coeff: &mut coeff,
                },
                downscale,
                ReducedIdctScratch {
                    pixels_4x4: &mut pixels_4x4,
                    pixels_2x2: &mut pixels_2x2,
                },
                window,
                next_stripe,
            )?;
            if mcu_row_intersects_rect(my - 1, mcu_height_px, roi) {
                emit_stripe_rgb_420_region(
                    plan,
                    backend,
                    writer,
                    Fast420RegionStripe {
                        neighbors: StripeNeighbors {
                            prev: has_prev.then_some(&*prev_stripe),
                            curr: curr_stripe,
                            next: Some(&*next_stripe),
                        },
                        stripe_index: my - 1,
                        roi,
                        region_layout,
                        crop_rows: &mut crop_rows,
                        downscale,
                    },
                )?;
            }
            core::mem::swap(&mut prev_stripe, &mut curr_stripe);
            core::mem::swap(&mut curr_stripe, &mut next_stripe);
            has_prev = true;
        }

        let curr_mcu_row = decode_mcu_row_end - 1;
        if curr_mcu_row <= last_output_mcu_row
            && mcu_row_intersects_rect(curr_mcu_row, mcu_height_px, roi)
        {
            emit_stripe_rgb_420_region(
                plan,
                backend,
                writer,
                Fast420RegionStripe {
                    neighbors: StripeNeighbors {
                        prev: has_prev.then_some(&*prev_stripe),
                        curr: curr_stripe,
                        next: None,
                    },
                    stripe_index: curr_mcu_row,
                    roi,
                    region_layout,
                    crop_rows: &mut crop_rows,
                    downscale,
                },
            )?;
        }
        finish_scan(&mut br, decode_mcu_row_end == mcu_rows)
    })();
    pool.restore_sink_rows(crop_rows);
    result
}
