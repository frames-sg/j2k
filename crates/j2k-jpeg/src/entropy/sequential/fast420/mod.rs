// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast 4:2:0 sequential scan driver (full-size only; scaled 4:2:0 decodes
//! chroma with a larger IDCT on the generic route).
//!
//! Whole images and regions, with or without restart intervals, decode one
//! MCU row at a time into a single stripe and convert it to RGB right away;
//! see [`emit`] for the row window.

use super::deposit::{
    FastTile420Components, FastTile420DcState, FastTile420EntropyState, FastTile420Window,
};
use super::layout::{
    fast420_decode_mcu_row_end, fast420_first_decode_mcu_row, uses_fancy_420_emit,
    Fast420RegionLayout,
};
use super::profile::{Fast420ScanProfiler, NoopFast420ScanProfile};
use super::restart::{
    finish_scan, scan_position_before_mcu, skip_to_mcu_with, McuSkipState, McuSkipTarget,
    ScanPosition,
};
use super::{HuffmanPrimaryWidth, PreparedDecodePlan, StripeBuffer};
use crate::backend::Backend;
#[cfg(feature = "bench-internals")]
use crate::bench_support::BenchFast420Profile;
use crate::color::scaled_sampling::ScaledSampling;
use crate::entropy::block::CoefficientBlock;
use crate::entropy::huffman::{FULL_PRIMARY_ENTRIES, NARROW_PRIMARY_ENTRIES};
use crate::error::{JpegError, Warning};
use crate::info::{DownscaleFactor, Rect};
use crate::internal::checkpoint::DeviceCheckpoint;
use crate::internal::scratch::ScratchPool;
use crate::output::{InterleavedRgbWriter, OutputWriter};
use alloc::vec::Vec;

mod emit;
mod rows;

use self::emit::{emit_pass_rows, Fast420RowPass};
pub(super) use self::rows::decode_mcu_row_fast_tile_420;
use self::rows::skip_mcu_fast_tile_420;

/// Luma pixels per 4:2:0 MCU in each direction.
const MCU_PX: u32 = 16;

/// Rows of context the streaming stripe keeps above each plane's MCU-row
/// samples: the previous MCU row's last two rows, so every pass emits whole
/// row pairs that share one chroma row.
pub(super) const FAST420_HALO_ROWS: u32 = 2;

fn fast_tile_components<const N: usize>(
    plan: &PreparedDecodePlan,
    wide_huffman: bool,
) -> Result<FastTile420Components<'_, N>, JpegError> {
    debug_assert!(plan.matches_fast_tile_shape());
    Ok(FastTile420Components {
        y: plan
            .resolved_component_with_entries::<N>(0)?
            .with_wide_huffman(wide_huffman),
        cb: plan
            .resolved_component_with_entries::<N>(1)?
            .with_wide_huffman(wide_huffman),
        cr: plan
            .resolved_component_with_entries::<N>(2)?
            .with_wide_huffman(wide_huffman),
    })
}

/// The part of the image a fast 4:2:0 scan writes, and an optional entropy
/// checkpoint to start nearer to it.
#[derive(Clone, Copy)]
struct Fast420Request<'a> {
    roi: Rect,
    checkpoint: Option<&'a DeviceCheckpoint>,
}

pub(crate) fn decode_scan_fast_tile_rgb<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
) -> Result<Vec<Warning>, JpegError> {
    let request = Fast420Request {
        roi: Rect::full(plan.dimensions),
        checkpoint: None,
    };
    let mut profile = NoopFast420ScanProfile::default();
    decode_scan_fast_tile_rgb_request(
        plan,
        backend,
        scan_bytes,
        pool,
        writer,
        request,
        &mut profile,
    )
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
    let request = Fast420Request {
        roi: Rect::full(plan.dimensions),
        checkpoint: None,
    };
    decode_scan_fast_tile_rgb_request(plan, backend, scan_bytes, pool, writer, request, profile)
}

pub(crate) fn decode_scan_fast_tile_rgb_region<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    roi: Rect,
    checkpoint: Option<&DeviceCheckpoint>,
) -> Result<Vec<Warning>, JpegError> {
    let request = Fast420Request { roi, checkpoint };
    let mut profile = NoopFast420ScanProfile::default();
    decode_scan_fast_tile_rgb_request(
        plan,
        backend,
        scan_bytes,
        pool,
        writer,
        request,
        &mut profile,
    )
}

fn decode_scan_fast_tile_rgb_request<W, P>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    request: Fast420Request<'_>,
    profile: &mut P,
) -> Result<Vec<Warning>, JpegError>
where
    W: OutputWriter + InterleavedRgbWriter,
    P: Fast420ScanProfiler,
{
    let scan = Fast420Scan {
        plan,
        backend,
        scan_bytes,
        request,
    };
    match plan.huffman_primary_width {
        HuffmanPrimaryWidth::Narrow8 => {
            scan.decode::<NARROW_PRIMARY_ENTRIES, W, P>(pool, writer, profile)
        }
        HuffmanPrimaryWidth::Full12 => {
            scan.decode::<FULL_PRIMARY_ENTRIES, W, P>(pool, writer, profile)
        }
    }
}

#[derive(Clone, Copy)]
struct Fast420Scan<'a, 'r> {
    plan: &'a PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &'a [u8],
    request: Fast420Request<'r>,
}

/// Geometry of the MCU rows a scan decodes.
#[derive(Clone, Copy)]
struct Fast420Rows {
    mcus_per_row: u32,
    mcu_rows: u32,
    first_row: u32,
    row_end: u32,
    window: FastTile420Window,
    layout: Fast420RegionLayout,
    roi: Rect,
    image_height: u32,
}

impl Fast420Rows {
    fn window(self, mcu_y: u32) -> FastTile420Window {
        FastTile420Window {
            mcu_y,
            bottom_luma: self.image_height.saturating_sub(mcu_y * MCU_PX) > 8,
            ..self.window
        }
    }

    fn pass(self, mcu_y: u32) -> Result<Fast420RowPass, JpegError> {
        Fast420RowPass::new(
            self.roi,
            self.image_height,
            mcu_y,
            mcu_y > self.first_row,
            mcu_y + 1 == self.mcu_rows,
        )
    }
}

impl Fast420Scan<'_, '_> {
    fn rows(self) -> Fast420Rows {
        let (width, height) = self.plan.dimensions;
        let roi = self.request.roi;
        let mcus_per_row = width.div_ceil(MCU_PX);
        let mcu_rows = height.div_ceil(MCU_PX);
        let layout = Fast420RegionLayout::new(width as usize, roi);
        Fast420Rows {
            mcus_per_row,
            mcu_rows,
            first_row: fast420_first_decode_mcu_row(roi, MCU_PX),
            row_end: fast420_decode_mcu_row_end(roi, MCU_PX, mcu_rows, FAST420_HALO_ROWS),
            window: FastTile420Window {
                mcu_y: 0,
                mcus_per_row,
                mcu_rows,
                restart: self.plan.restart_interval.unwrap_or(0),
                stripe_mcu_start: layout.stripe_mcu_start,
                stripe_mcus_per_row: layout.stripe_mcus_per_row,
                // Whether the final coded MCU's right luma blocks reach into
                // the image; the IDCT skips them otherwise.
                right_luma: width % MCU_PX == 0 || width % MCU_PX > 8,
                bottom_luma: true,
            },
            layout,
            roi,
            image_height: height,
        }
    }

    fn decode<const N: usize, W, P>(
        self,
        pool: &mut ScratchPool,
        writer: &mut W,
        profile: &mut P,
    ) -> Result<Vec<Warning>, JpegError>
    where
        W: OutputWriter + InterleavedRgbWriter,
        P: Fast420ScanProfiler,
    {
        let plan = self.plan;
        debug_assert!(plan.matches_fast_tile_shape());
        debug_assert!(uses_fancy_420_emit(
            plan,
            &ScaledSampling::new(plan.sampling, DownscaleFactor::Full, plan.dimensions)
        ));
        let rows = self.rows();
        pool.prepare_for_streaming(
            plan,
            rows.window.stripe_mcus_per_row,
            FAST420_HALO_ROWS as usize,
            0,
            plan.scratch_bytes,
        )?;

        let components =
            fast_tile_components::<N>(plan, plan.prefers_wide_huffman(self.scan_bytes.len()))?;
        let first_mcu = rows.first_row * rows.mcus_per_row;
        let ScanPosition {
            mut br,
            prev_dc,
            mcu_index: mut current_mcu,
            mut expected_rst,
        } = scan_position_before_mcu(
            self.scan_bytes,
            rows.window.restart,
            self.request.checkpoint,
            first_mcu,
        );
        let mut dc = [prev_dc[0], prev_dc[1], prev_dc[2]];
        let mut mcus_since_restart = 0u32;
        skip_to_mcu_with(
            McuSkipTarget {
                target_mcu: first_mcu,
                total_mcus: rows.mcus_per_row * rows.mcu_rows,
                restart: rows.window.restart,
                wide_huffman: plan.prefers_wide_huffman(self.scan_bytes.len()),
            },
            &mut McuSkipState {
                br: &mut br,
                prev_dc: &mut dc,
                current_mcu: &mut current_mcu,
                mcus_since_restart: &mut mcus_since_restart,
                expected_rst: &mut expected_rst,
            },
            |br, dc| {
                let [y_dc, cb_dc, cr_dc] = dc else {
                    return Err(JpegError::InternalInvariant {
                        reason: "4:2:0 seek requires three DC predictors",
                    });
                };
                skip_mcu_fast_tile_420(
                    components.y,
                    components.cb,
                    components.cr,
                    br,
                    y_dc,
                    cb_dc,
                    cr_dc,
                )
            },
        )?;
        let [mut y_dc, mut cb_dc, mut cr_dc] = dc;
        let mut coeff = CoefficientBlock::default();
        let mut state = FastTile420EntropyState {
            br: &mut br,
            dc: FastTile420DcState {
                y: &mut y_dc,
                cb: &mut cb_dc,
                cr: &mut cr_dc,
            },
            coeff: &mut coeff,
            mcus_since_restart: &mut mcus_since_restart,
            expected_rst: &mut expected_rst,
        };

        self.decode_rows(components, &mut state, &mut pool.stripe_a, writer, profile)?;

        let finish_timer = profile.begin_finish_scan();
        let result = finish_scan(&mut br, rows.row_end == rows.mcu_rows);
        profile.finish_finish_scan(finish_timer);
        result
    }

    fn decode_rows<const N: usize, W, P>(
        self,
        components: FastTile420Components<'_, N>,
        state: &mut FastTile420EntropyState<'_, '_>,
        stripe: &mut StripeBuffer,
        writer: &mut W,
        profile: &mut P,
    ) -> Result<(), JpegError>
    where
        W: InterleavedRgbWriter,
        P: Fast420ScanProfiler,
    {
        let rows = self.rows();
        for mcu_y in rows.first_row..rows.row_end {
            let pass = rows.pass(mcu_y)?;
            let mcu_timer = profile.begin_mcu_decode();
            decode_mcu_row_fast_tile_420(
                components,
                self.backend,
                state,
                rows.window(mcu_y),
                stripe,
                profile.activity_profiler(),
            )?;
            profile.finish_mcu_decode(mcu_timer);

            let emit_timer = profile.begin_rgb_emit();
            emit_pass_rows(self.backend, rows.layout, pass, stripe, writer)?;
            profile.finish_rgb_emit(emit_timer);
            if mcu_y + 1 < rows.row_end {
                stripe.carry_last_rows_to_halo(FAST420_HALO_ROWS as usize);
            }
        }
        Ok(())
    }
}
