// SPDX-License-Identifier: MIT OR Apache-2.0

//! Baseline sequential scan decoder. Iterates MCUs, decodes blocks, runs the
//! IDCT, and pipes rows through an [`OutputWriter`] with chroma upsample and
//! color conversion.

use crate::backend::Backend;
#[cfg(feature = "bench-internals")]
use crate::bench_support::BenchFast420Profile;
use crate::entropy::block::{
    decode_block_dequantized_into, decode_block_quantized_and_dequantized_with_activity,
    decode_block_with_activity, skip_block, BlockActivity, CoefficientBlock,
};
use crate::entropy::huffman::HuffmanTable;
use crate::error::{JpegError, Warning};
use crate::idct::downscale;
use crate::info::{ColorSpace, DownscaleFactor, Rect, SamplingFactors};
use crate::internal::bit_reader::BitReader;
use crate::internal::checkpoint::DeviceCheckpoint;
use crate::internal::scratch::{RgbGenericRows, ScratchPool, YCbCr420Rows, YCbCrGenericRows};
use crate::output::{InterleavedRgbWriter, OutputWriter};
use alloc::sync::Arc;
use alloc::vec::Vec;

mod deposit;
mod emit;
mod layout;
mod profile;
mod restart;

#[cfg(test)]
use self::deposit::deposit_dc_block;
use self::deposit::{
    assert_stripe_deposit_capacity, decode_eighth_block_to_plane, decode_quarter_block_to_plane,
    decode_scaled_block_to_plane, deposit_block, deposit_block_1x1, deposit_block_2x2,
    deposit_block_4x4, idct_deposit_fast_tile_block, EntropyBlockState, FastTile420Components,
    FastTile420DcState, FastTile420EntropyState, FastTile420Window, PlaneBlockTarget,
    ReducedIdctScratch,
};
#[cfg(test)]
use self::emit::{component_row_triplet, should_use_direct_420_crop};
use self::emit::{
    emit_stripe, emit_stripe_rgb, emit_stripe_rgb_420_region, emit_stripe_rgb_444,
    Fast420RegionStripe, StripeEmit, StripeNeighbors,
};
use self::layout::{
    component_block_intersects_rect, decode_mcu_row_end_for_rect, expanded_output_rect,
    fast420_decode_mcu_row_end, fast420_first_decode_mcu_row, first_decode_mcu_row_for_rect,
    is_ycbcr_420, last_mcu_row_for_rect, mcu_row_intersects_rect, scaled_dimensions,
    ComponentBlockPosition, Fast420RegionLayout,
};
pub(crate) use self::layout::{fast_tile_region_first_decode_mcu, stripe_region_layout};
use self::profile::{
    Fast420Profiler, Fast420ScanProfiler, NoopFast420Profiler, NoopFast420ScanProfile,
};
pub(crate) use self::restart::finish_scan;
use self::restart::{
    reader_from_checkpoint, restart_seek_for_mcu, skip_to_mcu, McuSkipState, McuSkipTarget,
};

/// Per-component decode context. One entry per component declared in the
/// SOF, in scan order.
#[derive(Debug, Clone)]
pub(crate) struct PreparedComponentPlan {
    pub(crate) h: u8,
    pub(crate) v: u8,
    pub(crate) output_index: usize,
    pub(crate) quant: Arc<[u16; 64]>,
    pub(crate) dc_table: Arc<HuffmanTable>,
    pub(crate) ac_table: Arc<HuffmanTable>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedDecodePlan {
    pub(crate) components: Vec<PreparedComponentPlan>,
    pub(crate) sampling: SamplingFactors,
    pub(crate) color_space: ColorSpace,
    pub(crate) restart_interval: Option<u16>,
    pub(crate) dimensions: (u32, u32),
    pub(crate) scan_offset: usize,
    pub(crate) scratch_bytes: usize,
}

impl PreparedDecodePlan {
    pub(crate) fn matches_fast_tile_shape(&self) -> bool {
        self.restart_interval.is_none()
            && is_ycbcr_420(self)
            && self.components.len() == 3
            && self.components[0].output_index == 0
            && self.components[0].h == 2
            && self.components[0].v == 2
            && self.components[1].output_index == 1
            && self.components[1].h == 1
            && self.components[1].v == 1
            && self.components[2].output_index == 2
            && self.components[2].h == 1
            && self.components[2].v == 1
    }

    pub(crate) fn matches_fast_rgb444_shape(&self) -> bool {
        self.color_space == ColorSpace::YCbCr
            && self.components.len() == 3
            && self.components[0].output_index == 0
            && self.components[0].h == 1
            && self.components[0].v == 1
            && self.components[1].output_index == 1
            && self.components[1].h == 1
            && self.components[1].v == 1
            && self.components[2].output_index == 2
            && self.components[2].h == 1
            && self.components[2].v == 1
    }

    pub(crate) fn matches_fast_rgb422_shape(&self) -> bool {
        self.color_space == ColorSpace::YCbCr
            && self.components.len() == 3
            && self.components[0].output_index == 0
            && self.components[0].h == 2
            && self.components[0].v == 1
            && self.components[1].output_index == 1
            && self.components[1].h == 1
            && self.components[1].v == 1
            && self.components[2].output_index == 2
            && self.components[2].h == 1
            && self.components[2].v == 1
    }
}

enum OutputScratch<'a> {
    Grayscale,
    YCbCr420(&'a mut YCbCr420Rows),
    YCbCrGeneric(&'a mut YCbCrGenericRows),
    RgbGeneric(&'a mut RgbGenericRows),
}

enum RgbOutputScratch<'a> {
    None,
    YCbCr420,
    YCbCrGeneric(&'a mut YCbCrGenericRows),
    RgbGeneric(&'a mut RgbGenericRows),
}

#[derive(Debug, Default)]
pub(crate) struct StripeBuffer {
    pub(crate) planes: Vec<Vec<u8>>,
    pub(crate) plane_strides: Vec<usize>,
    pub(crate) plane_rows: Vec<usize>,
}

#[derive(Clone, Copy)]
struct StripePlane<'a> {
    data: &'a [u8],
    stride: usize,
    rows: usize,
}

impl StripeBuffer {
    /// Grow each plane's backing Vec to the size required by `plan` and
    /// `mcus_per_row`. Never shrinks the allocation — a monotonic
    /// tile-batch workload pays the allocation cost exactly once.
    pub(crate) fn resize_for(
        &mut self,
        plan: &PreparedDecodePlan,
        mcus_per_row: u32,
        block_size: u32,
    ) {
        let n = plan.sampling.len();
        self.planes.resize_with(n, Vec::new);
        self.plane_strides.resize(n, 0);
        self.plane_rows.resize(n, 0);
        for (i, (h, v)) in plan.sampling.iter().enumerate() {
            let cols = (mcus_per_row as usize) * (h as usize) * (block_size as usize);
            let rows = (v as usize) * (block_size as usize);
            let bytes = cols * rows;
            if self.planes[i].len() < bytes {
                self.planes[i].resize(bytes, 0);
            }
            self.plane_strides[i] = cols;
            self.plane_rows[i] = rows;
        }
    }

    fn row_count(&self, plane_idx: usize) -> usize {
        self.plane_rows[plane_idx]
    }

    fn row(&self, plane_idx: usize, row: usize) -> &[u8] {
        let stride = self.plane_strides[plane_idx];
        let start = row * stride;
        &self.planes[plane_idx][start..start + stride]
    }

    fn plane(&self, plane_idx: usize) -> StripePlane<'_> {
        StripePlane {
            data: &self.planes[plane_idx],
            stride: self.plane_strides[plane_idx],
            rows: self.plane_rows[plane_idx],
        }
    }
}

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
    let max_h = plan.sampling.max_h as u32;
    let max_v = plan.sampling.max_v as u32;
    let mcu_width_px = 8 * max_h;
    let mcu_height_px = 8 * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);

    pool.prepare_for(
        plan,
        mcus_per_row,
        DownscaleFactor::Full.output_block_size(),
    );

    let mut br = BitReader::new(scan_bytes);
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let (y_comp, cb_comp, cr_comp) = fast_tile_components(plan);
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
    let result = finish_fast_tile_scan(&mut br);
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

fn finish_fast_tile_scan(br: &mut BitReader<'_>) -> Result<Vec<Warning>, JpegError> {
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

pub(crate) fn decode_scan_dct_blocks(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    retain_quantized_blocks: bool,
) -> Result<DecodedDctBlocks, JpegError> {
    let max_h = u32::from(plan.sampling.max_h);
    let max_v = u32::from(plan.sampling.max_v);
    let mcu_width_px = 8 * max_h;
    let mcu_height_px = 8 * max_v;
    let mcus_per_row = plan.dimensions.0.div_ceil(mcu_width_px);
    let mcu_rows = plan.dimensions.1.div_ceil(mcu_height_px);
    let component_count = plan.sampling.len();

    let mut block_cols_by_component = vec![0_u32; component_count];
    let mut block_rows_by_component = vec![0_u32; component_count];
    for component in &plan.components {
        block_cols_by_component[component.output_index] = mcus_per_row * u32::from(component.h);
        block_rows_by_component[component.output_index] = mcu_rows * u32::from(component.v);
    }

    let mut quantized_blocks = block_cols_by_component
        .iter()
        .zip(block_rows_by_component.iter())
        .map(|(&cols, &rows)| {
            if retain_quantized_blocks {
                vec![[0_i16; 64]; (cols * rows) as usize]
            } else {
                Vec::new()
            }
        })
        .collect::<Vec<_>>();
    let mut dequantized_blocks = block_cols_by_component
        .iter()
        .zip(block_rows_by_component.iter())
        .map(|(&cols, &rows)| vec![[0_i16; 64]; (cols * rows) as usize])
        .collect::<Vec<_>>();

    let mut br = BitReader::new(scan_bytes);
    let mut prev_dc = vec![0_i32; component_count];
    let mut quantized_coeff = CoefficientBlock::default();
    let mut dequantized_coeff = CoefficientBlock::default();
    let restart = plan.restart_interval.unwrap_or(0);
    let mut mcus_since_restart = 0_u32;
    let mut expected_rst = 0_u8;
    let total_mcus = mcu_rows * mcus_per_row;

    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcus_per_row {
            let current_mcu = mcu_y * mcus_per_row + mcu_x;
            if restart > 0 && mcus_since_restart == u32::from(restart) {
                let _ = br.ensure_bits(1);
                let marker = br.take_marker().ok_or(JpegError::UnexpectedEoi {
                    mcu_at: current_mcu,
                    mcu_total: total_mcus,
                })?;
                let expected = 0xD0 | expected_rst;
                if marker != expected {
                    return Err(JpegError::RestartMismatch {
                        offset: br.position(),
                        expected: expected_rst,
                        found: marker,
                    });
                }
                expected_rst = (expected_rst + 1) & 0x07;
                br.reset_at_restart();
                prev_dc.fill(0);
                mcus_since_restart = 0;
            }

            for component in &plan.components {
                let plane_idx = component.output_index;
                let block_cols = block_cols_by_component[plane_idx];
                for vy in 0..u32::from(component.v) {
                    for vx in 0..u32::from(component.h) {
                        let block_x = mcu_x * u32::from(component.h) + vx;
                        let block_y = mcu_y * u32::from(component.v) + vy;
                        let block_idx = (block_y * block_cols + block_x) as usize;
                        if retain_quantized_blocks {
                            decode_block_quantized_and_dequantized_with_activity(
                                &mut br,
                                &component.dc_table,
                                &component.ac_table,
                                &mut prev_dc[plane_idx],
                                &component.quant,
                                &mut quantized_coeff,
                                &mut dequantized_coeff,
                            )?;
                            quantized_blocks[plane_idx][block_idx] =
                                *quantized_coeff.coefficients();
                        } else {
                            decode_block_dequantized_into(
                                &mut br,
                                &component.dc_table,
                                &component.ac_table,
                                &mut prev_dc[plane_idx],
                                &component.quant,
                                &mut dequantized_blocks[plane_idx][block_idx],
                            )?;
                        }
                        if retain_quantized_blocks {
                            dequantized_blocks[plane_idx][block_idx] =
                                *dequantized_coeff.coefficients();
                        }
                    }
                }
            }

            mcus_since_restart += 1;
        }
    }

    finish_scan(&mut br, true)?;
    Ok(DecodedDctBlocks {
        quantized: quantized_blocks,
        dequantized: dequantized_blocks,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedDctBlocks {
    pub(crate) quantized: Vec<Vec<[i16; 64]>>,
    pub(crate) dequantized: Vec<Vec<[i16; 64]>>,
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
    debug_assert!(plan.matches_fast_tile_shape());

    let (width, height) = plan.dimensions;
    let max_h = plan.sampling.max_h as u32;
    let max_v = plan.sampling.max_v as u32;
    let mcu_width_px = 8 * max_h;
    let mcu_height_px = 8 * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);
    let first_decode_mcu_row = fast420_first_decode_mcu_row(roi, mcu_height_px);
    let decode_mcu_row_end = fast420_decode_mcu_row_end(roi, mcu_height_px, mcu_rows);
    let last_output_mcu_row = last_mcu_row_for_rect(roi, mcu_height_px, mcu_rows);

    let region_layout = Fast420RegionLayout::new(width as usize, roi);

    pool.prepare_for(
        plan,
        region_layout.stripe_mcus_per_row,
        DownscaleFactor::Full.output_block_size(),
    );

    let mut crop_rows = pool.take_sink_rows(region_layout.row_width());
    let result = (|| {
        let mut coeff = CoefficientBlock::default();
        let mut pixels = [0u8; 64];
        let (y_comp, cb_comp, cr_comp) = fast_tile_components(plan);
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
    let max_h = plan.sampling.max_h as u32;
    let max_v = plan.sampling.max_v as u32;
    let block_size = downscale.output_block_size();
    let mcu_width_px = block_size * max_h;
    let mcu_height_px = block_size * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);
    let first_decode_mcu_row = fast420_first_decode_mcu_row(roi, mcu_height_px);
    let decode_mcu_row_end = fast420_decode_mcu_row_end(roi, mcu_height_px, mcu_rows);
    let last_output_mcu_row = last_mcu_row_for_rect(roi, mcu_height_px, mcu_rows);

    let region_layout = Fast420RegionLayout::new_for_mcu_width(width as usize, roi, mcu_width_px);

    pool.prepare_for(plan, region_layout.stripe_mcus_per_row, block_size);

    let mut crop_rows = pool.take_sink_rows(region_layout.row_width());
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
        let components = FastTile420Components {
            y: &plan.components[0],
            cb: &plan.components[1],
            cr: &plan.components[2],
        };
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
        if context.restart > 0 && *state.mcus_since_restart == u32::from(context.restart) {
            let _ = state.br.ensure_bits(1);
            let marker = state.br.take_marker().ok_or(JpegError::UnexpectedEoi {
                mcu_at: mcu_y * context.mcus_per_row + mx,
                mcu_total: context.mcu_rows * context.mcus_per_row,
            })?;
            let expected = 0xD0 | *state.expected_rst;
            if marker != expected {
                return Err(JpegError::RestartMismatch {
                    offset: state.br.position(),
                    expected: *state.expected_rst,
                    found: marker,
                });
            }
            *state.expected_rst = (*state.expected_rst + 1) & 0x07;
            state.br.reset_at_restart();
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

fn fast_tile_components(
    plan: &PreparedDecodePlan,
) -> (
    &PreparedComponentPlan,
    &PreparedComponentPlan,
    &PreparedComponentPlan,
) {
    debug_assert!(plan.matches_fast_tile_shape());
    (
        &plan.components[0],
        &plan.components[1],
        &plan.components[2],
    )
}

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
        if context.restart > 0 && *state.mcus_since_restart == u32::from(context.restart) {
            let _ = state.br.ensure_bits(1);
            let marker = state.br.take_marker().ok_or(JpegError::UnexpectedEoi {
                mcu_at: mcu_y * context.mcus_per_row + mx,
                mcu_total: context.mcu_rows * context.mcus_per_row,
            })?;
            let expected = 0xD0 | *state.expected_rst;
            if marker != expected {
                return Err(JpegError::RestartMismatch {
                    offset: state.br.position(),
                    expected: *state.expected_rst,
                    found: marker,
                });
            }
            *state.expected_rst = (*state.expected_rst + 1) & 0x07;
            state.br.reset_at_restart();
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

#[allow(clippy::inline_always)]
#[inline(always)]
fn skip_mcu_fast_tile_420(
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

fn decode_mcu_row_fast_tile_420_scaled(
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

fn decode_mcu_row_fast_tile_420(
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

#[cfg(test)]
mod tests;
