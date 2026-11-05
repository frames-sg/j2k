// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast 4:2:0 sequential scan driver (full-size only; scaled 4:2:0 decodes
//! chroma with a larger IDCT on the generic route).
//!
//! Whole images and regions, with or without restart intervals, decode one
//! MCU row at a time into a single stripe and convert it to RGB right away;
//! see [`emit`] for the row window.

use super::dct::sequential_mcu_size;
use super::deposit::{FastRowState, FastTile420Components, FastTile420Window};
use super::layout::{fast420_decode_mcu_row_end, uses_fancy_420_emit, CropWindow};
use super::profile::{Fast420ScanProfiler, NoopFast420ScanProfile};
use super::restart::finish_scan;
use super::{FastRgbRequest, HuffmanPrimaryWidth, PreparedDecodePlan, StripeBuffer};
use crate::adapter::JpegFastPacketFamily;
use crate::backend::Backend;
#[cfg(feature = "bench-internals")]
use crate::bench_support::BenchFast420Profile;
use crate::color::scaled_sampling::ScaledSampling;
use crate::entropy::block::CoefficientBlock;
use crate::entropy::huffman::{FULL_PRIMARY_ENTRIES, NARROW_PRIMARY_ENTRIES};
use crate::error::{JpegError, Warning};
use crate::info::{DownscaleFactor, Rect};
use crate::internal::scratch::ScratchPool;
use crate::output::{InterleavedRgbWriter, OutputWriter};
use alloc::vec::Vec;

mod emit;
mod rows;

use self::emit::{emit_pass_rows, Fast420RowPass};
pub(super) use self::rows::decode_mcu_row_fast_tile_420;

/// Luma pixels per 4:2:0 MCU in each direction.
const MCU_PX: u32 = 16;

/// Rows of context the streaming stripe keeps above each plane's MCU-row
/// samples: the previous MCU row's last two rows, so every pass emits whole
/// row pairs that share one chroma row.
pub(super) const FAST420_HALO_ROWS: u32 = 2;

/// Decode `request.roi` of a fancy 4:2:0 scan into `writer`, whose row 0 is
/// `request.roi.y`.
pub(super) fn decode_scan_fast_tile_rgb<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    request: FastRgbRequest<'_>,
) -> Result<Vec<Warning>, JpegError> {
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
    let request = FastRgbRequest {
        roi: Rect::full(plan.dimensions),
        checkpoint: None,
    };
    decode_scan_fast_tile_rgb_request(plan, backend, scan_bytes, pool, writer, request, profile)
}

fn decode_scan_fast_tile_rgb_request<W, P>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    request: FastRgbRequest<'_>,
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
    request: FastRgbRequest<'r>,
}

/// Geometry of the MCU rows a scan decodes.
#[derive(Clone, Copy)]
struct Fast420Rows {
    first_row: u32,
    row_end: u32,
    /// Decoded MCU columns; `bottom_luma` is set per row by [`Self::window`].
    window: FastTile420Window,
    crop: CropWindow,
    image_height: u32,
}

impl Fast420Rows {
    fn window(self, mcu_y: u32) -> FastTile420Window {
        FastTile420Window {
            bottom_luma: self.image_height.saturating_sub(mcu_y * MCU_PX) > 8,
            ..self.window
        }
    }

    fn pass(self, roi: Rect, mcu_y: u32) -> Result<Fast420RowPass, JpegError> {
        Fast420RowPass::new(
            roi,
            self.image_height,
            mcu_y,
            mcu_y > self.first_row,
            mcu_y + 1 == self.window.columns.mcu_rows,
        )
    }
}

impl Fast420Scan<'_, '_> {
    fn rows(self) -> Fast420Rows {
        let (width, height) = self.plan.dimensions;
        let roi = self.request.roi;
        let (columns, crop) = self.request.columns(self.plan);
        let (_, mcu_height_px) = sequential_mcu_size(self.plan);
        Fast420Rows {
            first_row: JpegFastPacketFamily::Fast420.first_decode_row(self.plan, roi),
            row_end: fast420_decode_mcu_row_end(
                roi,
                mcu_height_px,
                columns.mcu_rows,
                FAST420_HALO_ROWS,
            ),
            window: FastTile420Window {
                columns,
                // Whether the final coded MCU's right luma blocks reach into
                // the image; the IDCT skips them otherwise.
                right_luma: width % MCU_PX == 0 || width % MCU_PX > 8,
                bottom_luma: true,
            },
            crop,
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
        debug_assert!(plan.matches_fast_420_shape());
        debug_assert!(uses_fancy_420_emit(
            plan,
            &ScaledSampling::new(plan.sampling, DownscaleFactor::Full, plan.dimensions)
        ));
        let rows = self.rows();
        let columns = rows.window.columns;
        pool.prepare_for_streaming(
            plan,
            columns.stripe_mcus_per_row,
            FAST420_HALO_ROWS as usize,
            0,
            plan.scratch_bytes,
        )?;

        let (components, mut cursor, mut dc) = FastTile420Components::<N>::enter_scan::<4>(
            plan,
            self.scan_bytes,
            self.request.checkpoint,
            columns.mcu_position(0, rows.first_row),
        )?;
        let mut coeff = CoefficientBlock::default();
        let mut state = FastRowState {
            br: &mut cursor.br,
            dc: &mut dc,
            coeff: &mut coeff,
            restart: &mut cursor.restart,
        };

        self.decode_rows(components, &mut state, &mut pool.stripe_a, writer, profile)?;

        let finish_timer = profile.begin_finish_scan();
        let result = finish_scan(&mut cursor.br, rows.row_end == columns.mcu_rows);
        profile.finish_finish_scan(finish_timer);
        result
    }

    fn decode_rows<const N: usize, W, P>(
        self,
        components: FastTile420Components<'_, N>,
        state: &mut FastRowState<'_, '_>,
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
            let pass = rows.pass(self.request.roi, mcu_y)?;
            let mcu_timer = profile.begin_mcu_decode();
            decode_mcu_row_fast_tile_420(
                components,
                self.backend,
                state,
                rows.window(mcu_y),
                mcu_y,
                stripe,
                profile.activity_profiler(),
            )?;
            profile.finish_mcu_decode(mcu_timer);

            let emit_timer = profile.begin_rgb_emit();
            emit_pass_rows(self.backend, rows.crop, pass, stripe, writer)?;
            profile.finish_rgb_emit(emit_timer);
            if mcu_y + 1 < rows.row_end {
                stripe.carry_last_rows_to_halo(FAST420_HALO_ROWS as usize);
            }
        }
        Ok(())
    }
}
