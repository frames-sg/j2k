// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public [`Decoder`] entry points.

use crate::backend::Backend;
use crate::context::DecoderContext;
use crate::entropy::block::{decode_block_with_activity, BlockActivity, CoefficientBlock};
use crate::entropy::huffman::HuffmanTable;
use crate::entropy::progressive::{
    decode_progressive, decode_progressive_dct_blocks, PreparedProgressiveComponentPlan,
    PreparedProgressivePlan, PreparedProgressiveScan, PreparedProgressiveScanComponent,
};
use crate::entropy::sequential::{
    decode_scan_baseline, decode_scan_baseline_rgb, decode_scan_fast_rgb_444,
    decode_scan_fast_tile_rgb, decode_scan_fast_tile_rgb_region,
    decode_scan_fast_tile_rgb_region_scaled, fast_tile_region_first_decode_mcu, finish_scan,
    stripe_region_layout, FastTileRegionScaledRequest, PreparedComponentPlan, PreparedDecodePlan,
};
use crate::entropy::ZIGZAG;
use crate::error::{JpegError, MarkerKind, Warning};
use crate::info::{
    ColorSpace, DecodeOptions, DownscaleFactor, Info, OutputFormat, Rect, RestartIndex,
    RestartSegment, SofKind,
};
use crate::internal::bit_reader::BitReader;
use crate::internal::checkpoint::{checkpoint_before_mcu, CpuCheckpointCache, DeviceCheckpoint};
use crate::internal::scratch::ScratchPool;
use crate::lossless::{lossless_predict, LosslessSample};
use crate::output::{
    validate_buffer, Gray8Writer, InterleavedRgbWriter, OutputWriter, Rgb8Writer, Rgba8Writer,
};
use crate::parse::header::{parse_info, ParsedHeader};
use crate::parse::tables::{HuffmanValues, RawHuffmanTable};
use crate::profile::{duration_us_string, emit_jpeg_profile_row, jpeg_profile_stages_enabled};
use crate::segment::PreparedJpeg;
use crate::JpegCodec;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;
pub use j2k_core::TileBatchOptions;
use j2k_core::{
    CompressedTransferSyntax, DecodeOutcome as CoreDecodeOutcome, DecodeRowsError,
    DecoderContext as CoreDecoderContext, Downscale, ImageCodec, ImageDecode, ImageDecodeRows,
    PixelFormat, RowSink, TileBatchDecode,
};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const DEFAULT_MAX_DECODE_BYTES: usize = 512 * 1024 * 1024;
const CPU_ROI_CHECKPOINT_CADENCE_MCUS: u32 = 1024;
const CPU_ROI_CHECKPOINT_MIN_TARGET_MCUS: u32 = 4096;

std::thread_local! {
    static DEFAULT_SCRATCH: RefCell<ScratchPool> = RefCell::new(ScratchPool::new());
    static DEFAULT_CONTEXT: RefCell<DecoderContext> = RefCell::new(DecoderContext::new());
}

mod view;
pub use self::view::JpegView;
mod output_format;
use self::output_format::{
    allocate_output_buffer, checked_output_geometry, downscale_profile_name, jpeg_downscale,
    output_format_from_parts, output_format_profile_name, scaled_dimensions, scaled_rect_covering,
};
mod extended12;
use self::extended12::{
    decode_extended12_block_pixels, decode_extended12_color_planes,
    decode_extended12_four_component_planes, dequantize_progressive12_block,
    extended12_color_sampling, extended12_four_component_sampling, lossless_color_sampling,
    progressive_color_component_indices, progressive_color_sampling,
    progressive_four_component_sampling, render_progressive12_color_planes,
    render_progressive12_four_component_planes, upsample_h2v1_sample_at, upsample_h2v2_rows_at,
    validate_extended12_color444_plan, validate_extended12_four_component444_plan,
    write_extended12_block_region, write_extended12_color420_planes_region,
    write_extended12_color422_planes_region, write_extended12_four_component_block_region,
    write_extended12_four_component_planes_region, write_extended12_rgb_block_region,
    Extended12ColorSampling, Extended12Output, Extended12RgbProjection, Extended12WriteRegion,
};
mod lossless_helpers;
use self::lossless_helpers::{
    decode_lossless_color_sample, decode_lossless_sampled_color_mcu, emit_decode_scan_profile,
    lossless_predictor_gray_rows, lossless_predictor_value, lossless_predictor_value_u16,
    restart_index_for_stream, validate_lossless_color_plan, write_lossless_color16_sampled_output,
    write_lossless_color8_sampled_output, Extended12RestartTracker, LosslessColorIntoSample,
    LosslessColorPlanes, LosslessColorRowSample, LosslessRestartTracker,
    LosslessSampledColorPlanesMut, LosslessSampledMcu,
};
mod color_convert;
use self::color_convert::{
    convert_ycbcr16_to_rgb16_in_place, convert_ycbcr8_to_rgb8_in_place, copy_gray16_scaled_rect,
    copy_gray8_scaled_rect, copy_rgb16_to_rgba16, copy_ycbcr16_row_to_rgb16,
    copy_ycbcr8_row_to_rgb8, merged_warnings,
};
mod core_traits;
use self::core_traits::{CroppedWriter, ProgressiveDownscaleWriter};
mod lossless_region;
use self::lossless_region::{LosslessRgbRegionFallback, LosslessRgbaAlpha};
mod scratch;
use self::scratch::{
    checked_scratch_len, checked_usize_product, compute_decode_scratch_bytes,
    compute_extended12_planes_scratch_bytes, compute_lossless_scratch_bytes,
};
mod sink_writer;
pub(crate) use self::sink_writer::SinkWriter;

/// Non-fatal outcome of a successful decode. See spec Section 2.
///
/// `DecodeOutcome` lives on `decoder.rs` rather than `info.rs` because it
/// carries `Warning` values from `error.rs`, and moving it into `info` would
/// create a `info → error` cycle (see `info.rs` header note).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOutcome {
    /// The rectangle actually written to the output buffer. For `decode_into`
    /// this is always `Rect::full(info.dimensions)`; later milestones add
    /// `decode_region_into` which can return a narrower rect.
    pub decoded: Rect,
    /// Warnings emitted during parse or decode. Empty when the stream is
    /// syntactically clean and every capability was exercised without fallback.
    pub warnings: Vec<Warning>,
}

impl From<DecodeOutcome> for CoreDecodeOutcome<Warning> {
    fn from(outcome: DecodeOutcome) -> Self {
        Self {
            decoded: outcome.decoded.into(),
            warnings: outcome.warnings,
        }
    }
}

/// Owned-output JPEG decode request.
///
/// This consolidates the full-image, region, and downscale axes for callers
/// that want a freshly allocated tightly packed output buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeRequest {
    /// Requested output pixel format.
    pub fmt: PixelFormat,
    /// Optional source-image region to decode.
    pub region: Option<Rect>,
    /// Requested decoder downscale.
    pub scale: Downscale,
}

impl DecodeRequest {
    /// Full-image decode at native scale.
    pub const fn full(fmt: PixelFormat) -> Self {
        Self {
            fmt,
            region: None,
            scale: Downscale::None,
        }
    }

    /// Full-image decode with downscale.
    pub const fn scaled(fmt: PixelFormat, scale: Downscale) -> Self {
        Self {
            fmt,
            region: None,
            scale,
        }
    }

    /// Region decode at native scale.
    pub const fn region(fmt: PixelFormat, region: Rect) -> Self {
        Self {
            fmt,
            region: Some(region),
            scale: Downscale::None,
        }
    }

    /// Region decode with downscale.
    pub const fn region_scaled(fmt: PixelFormat, region: Rect, scale: Downscale) -> Self {
        Self {
            fmt,
            region: Some(region),
            scale,
        }
    }
}

/// One tile decode request for [`decode_tiles_into`].
pub type TileDecodeJob<'i, 'o> = j2k_core::TileDecodeJob<'i, 'o>;

/// Caller-owned output target for one context-reused tile decode helper.
pub struct TileDecodeOutput<'o> {
    /// Caller-owned output buffer.
    pub out: &'o mut [u8],
    /// Distance in bytes between output rows.
    pub stride: usize,
    /// Requested output pixel format.
    pub fmt: PixelFormat,
}

/// One decode request for a JPEG tile already normalized by
/// [`prepare_tiff_jpeg_tile`](crate::prepare_tiff_jpeg_tile).
pub struct PreparedJpegTileJob<'i, 'o> {
    /// Decode-ready prepared JPEG bytes.
    pub input: PreparedJpeg<'i>,
    /// Caller-owned RGB8 output buffer for this tile.
    pub out: &'o mut [u8],
    /// Distance in bytes between output rows.
    pub stride: usize,
    /// Per-job JPEG decode options.
    pub options: DecodeOptions,
}

/// Result for one successful prepared JPEG tile decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedTile {
    /// Tile dimensions reported by the prepared JPEG header.
    pub dimensions: (u32, u32),
    /// Rectangle written into the output buffer.
    pub decoded: Rect,
    /// Non-fatal warnings emitted during parse or decode.
    pub warnings: Vec<Warning>,
}

/// One scaled tile decode request for [`decode_tiles_scaled_into`].
pub type TileScaledDecodeJob<'i, 'o> = j2k_core::TileScaledDecodeJob<'i, 'o>;

/// One ROI+scaled tile decode request for
/// [`decode_tiles_region_scaled_into`].
pub type TileRegionScaledDecodeJob<'i, 'o> = j2k_core::TileRegionScaledDecodeJob<'i, 'o>;

/// Error returned by [`decode_tiles_into`], annotated with the failing tile
/// index from the caller's input order.
pub type TileBatchError = j2k_core::TileBatchError<JpegError>;

/// Receives decoded component rows before they are packed into the final
/// interleaved pixel format.
pub trait ComponentRowWriter {
    /// Receive one grayscale row.
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError>;

    /// Receive one full-width Y/Cb/Cr row.
    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError>;

    /// Receive one full-width planar RGB row.
    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError>;
}

/// A borrowed view of a JPEG stream ready to decode. Constructed via
/// [`Decoder::new`] or [`Decoder::from_view`]. `Decoder<'a>: Send + Sync`.
#[derive(Debug)]
pub struct Decoder<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) info: Info,
    pub(crate) warnings: Arc<[Warning]>,
    pub(crate) backend: Backend,
    pub(crate) plan: PreparedDecodePlan,
    pub(crate) progressive_plan: Option<PreparedProgressivePlan>,
    lossless_plan: Option<PreparedLosslessPlan>,
    pub(crate) cpu_entropy_checkpoints: Mutex<CpuCheckpointCache>,
}

#[derive(Debug, Clone)]
struct PreparedLosslessPlan {
    predictor: u8,
    bit_depth: u8,
    dc_table: Arc<HuffmanTable>,
    dimensions: (u32, u32),
    scan_offset: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LosslessColorSampling {
    S444,
    S422,
    S420,
}

impl<'a> Decoder<'a> {
    /// Parse the headers without decoding pixels. The parser walks headers up
    /// to the first SOS and then performs a lightweight marker scan so
    /// `Info::scan_count` reflects all scans in the file.
    ///
    /// # Errors
    /// Returns any structural, unsupported-SOF, or sanity-check error
    /// encountered before the Start-of-Scan marker. See [`JpegError`].
    pub fn inspect(input: &'a [u8]) -> Result<Info, JpegError> {
        let info = parse_info(input)?;
        Ok(info)
    }

    fn from_bytes_with_options(input: &'a [u8], options: DecodeOptions) -> Result<Self, JpegError> {
        let view = JpegView::parse_with_options(input, options)?;
        DEFAULT_CONTEXT.with(|ctx| Self::from_view_in_context(view, &mut ctx.borrow_mut()))
    }

    /// Build a decoder ready for `decode_into`. Parses the full header, pre-
    /// builds every referenced Huffman table, and validates that the stream is
    /// one of the SOFs this release implements.
    ///
    /// # Errors
    /// - Any parse error encountered before SOS (see [`Self::inspect`]).
    /// - [`JpegError::NotImplemented`] for SOFs that parse but are not yet
    ///   decodable for the requested shape (for example Progressive12 or
    ///   unsupported Lossless predictors).
    /// - [`JpegError::MissingHuffmanTable`] if the scan references a DC/AC
    ///   table slot that was never defined by a DHT segment.
    pub fn new(input: &'a [u8]) -> Result<Self, JpegError> {
        Self::from_bytes_with_options(input, DecodeOptions::default())
    }

    /// Build a decoder from a previously parsed [`JpegView`].
    pub fn from_view(view: JpegView<'a>) -> Result<Self, JpegError> {
        DEFAULT_CONTEXT.with(|ctx| Self::from_view_in_context(view, &mut ctx.borrow_mut()))
    }

    /// Build a decoder from a previously parsed [`JpegView`], reusing shared
    /// compiled DHT/DQT state from `ctx` when table contents repeat.
    pub fn from_view_in_context(
        view: JpegView<'a>,
        ctx: &mut DecoderContext,
    ) -> Result<Self, JpegError> {
        let JpegView {
            bytes,
            header,
            info,
            options,
        } = view;
        let backend = Backend::detect();
        let (info, warnings, plan, progressive_plan, lossless_plan) = if matches!(
            info.sof_kind,
            SofKind::Progressive8 | SofKind::Progressive12
        ) {
            let progressive_plan = Self::build_progressive_plan(&header, &info, ctx)?;
            let plan = Self::build_progressive_host_output_plan(&header, &info, ctx)?;
            (
                info,
                Arc::<[Warning]>::from(header.warnings.as_slice()),
                plan,
                Some(progressive_plan),
                None,
            )
        } else if info.sof_kind == SofKind::Lossless {
            let (plan, lossless_plan) = Self::build_lossless_plan(&header, &info, ctx)?;
            (
                info,
                Arc::<[Warning]>::from(header.warnings.as_slice()),
                plan,
                None,
                Some(lossless_plan),
            )
        } else if options == DecodeOptions::default() {
            if let Some(scan_offset) = header.sos_offset {
                let header_prefix = &bytes[..scan_offset];
                let (info, warnings, plan) = ctx.resolve_decode_plan(header_prefix, |ctx| {
                    let plan = Self::build_prepared_plan(&header, &info, ctx)?;
                    Ok((
                        info.clone(),
                        Arc::<[Warning]>::from(header.warnings.as_slice()),
                        plan,
                    ))
                })?;
                (info, warnings, plan, None, None)
            } else {
                let plan = Self::build_prepared_plan(&header, &info, ctx)?;
                (
                    info,
                    Arc::<[Warning]>::from(header.warnings.as_slice()),
                    plan,
                    None,
                    None,
                )
            }
        } else {
            let plan = Self::build_prepared_plan(&header, &info, ctx)?;
            (
                info,
                Arc::<[Warning]>::from(header.warnings.as_slice()),
                plan,
                None,
                None,
            )
        };
        Ok(Self {
            bytes,
            info,
            warnings,
            backend,
            plan,
            progressive_plan,
            lossless_plan,
            cpu_entropy_checkpoints: Mutex::new(CpuCheckpointCache::default()),
        })
    }

    fn build_prepared_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
    ) -> Result<PreparedDecodePlan, JpegError> {
        match info.sof_kind {
            SofKind::Baseline8 | SofKind::Extended8 | SofKind::Extended12 => {}
            other => return Err(JpegError::NotImplemented { sof: other }),
        }
        if info.sof_kind == SofKind::Extended12
            && !matches!(
                info.color_space,
                ColorSpace::Grayscale
                    | ColorSpace::YCbCr
                    | ColorSpace::Rgb
                    | ColorSpace::Cmyk
                    | ColorSpace::Ycck
            )
        {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        match info.color_space {
            ColorSpace::Grayscale
            | ColorSpace::YCbCr
            | ColorSpace::Rgb
            | ColorSpace::Cmyk
            | ColorSpace::Ycck => {}
        }

        let mut dc_tables: [Option<Arc<HuffmanTable>>; 4] = Default::default();
        let mut ac_tables: [Option<Arc<HuffmanTable>>; 4] = Default::default();
        let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
            marker: MarkerKind::Sos,
        })?;
        if header.scan_count != 1 {
            return Err(JpegError::InvalidSequentialScanCount {
                sof: info.sof_kind,
                count: header.scan_count,
            });
        }
        validate_leading_component_sampling(header, info)?;
        // Every component must declare H,V in 1..=4 per T.81 §B.2.2, and max_h
        // must actually divide every component's H (same for V). Malformed
        // streams can set H=0 (div-by-zero in upsample ratio), non-divisors
        // (arbitrary ratios M2 handles), or ratios that don't produce planes
        // that cover the image width.
        for (h, v) in header.sampling.iter() {
            if h == 0 || v == 0 || h > 4 || v > 4 {
                return Err(JpegError::NotImplemented { sof: info.sof_kind });
            }
            if !header.sampling.max_h.is_multiple_of(h) || !header.sampling.max_v.is_multiple_of(v)
            {
                return Err(JpegError::NotImplemented { sof: info.sof_kind });
            }
        }
        for comp in &scan.components {
            let di = comp.dc_table as usize;
            let ai = comp.ac_table as usize;
            if dc_tables[di].is_none() {
                let raw = header.huffman_tables.dc[di].as_ref().ok_or(
                    JpegError::MissingHuffmanTable {
                        component: comp.id,
                        class: 0,
                        id: comp.dc_table,
                    },
                )?;
                dc_tables[di] = Some(ctx.resolve_huffman_table(raw)?);
            }
            if ac_tables[ai].is_none() {
                let raw = header.huffman_tables.ac[ai].as_ref().ok_or(
                    JpegError::MissingHuffmanTable {
                        component: comp.id,
                        class: 1,
                        id: comp.ac_table,
                    },
                )?;
                ac_tables[ai] = Some(ctx.resolve_huffman_table(raw)?);
            }
        }

        build_decode_plan(header, info, &dc_tables, &ac_tables, ctx)
    }

    fn build_lossless_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
    ) -> Result<(PreparedDecodePlan, PreparedLosslessPlan), JpegError> {
        if info.sof_kind != SofKind::Lossless {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        if header.scan_count != 1 {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
            marker: MarkerKind::Sos,
        })?;
        if !(1..=7).contains(&scan.ss) {
            return Err(JpegError::UnsupportedPredictor { predictor: scan.ss });
        }
        if scan.se != 0 || scan.ah != 0 || scan.al != 0 {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        let expected_components = match (info.color_space, info.bit_depth) {
            (ColorSpace::Grayscale, 8 | 16) => 1,
            (ColorSpace::Rgb, 8 | 16) => 3,
            (ColorSpace::YCbCr, 8 | 16) => 3,
            _ => return Err(JpegError::NotImplemented { sof: info.sof_kind }),
        };
        if scan.components.len() != expected_components {
            return Err(JpegError::UnsupportedComponentCount {
                count: scan.components.len() as u8,
            });
        }
        let empty_raw = RawHuffmanTable {
            bits: [0; 16],
            values: HuffmanValues::default(),
        };
        let empty_huffman = ctx.resolve_huffman_table(&empty_raw)?;
        let mut components = Vec::with_capacity(scan.components.len());
        let mut first_dc_table = None;
        for scan_component in scan.components.iter().copied() {
            let component_index = find_component_index(&header.component_ids, scan_component.id)
                .ok_or(JpegError::UnknownScanComponent {
                    offset: header.sos_offset.unwrap_or_default(),
                    component: scan_component.id,
                })?;
            let (h, v) =
                header
                    .sampling
                    .component(component_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })?;
            let raw_dc = header.huffman_tables.dc[scan_component.dc_table as usize]
                .as_ref()
                .ok_or(JpegError::MissingHuffmanTable {
                    component: scan_component.id,
                    class: 0,
                    id: scan_component.dc_table,
                })?;
            let dc_table = ctx.resolve_huffman_table(raw_dc)?;
            first_dc_table.get_or_insert_with(|| Arc::clone(&dc_table));
            components.push(PreparedComponentPlan {
                h,
                v,
                output_index: component_index,
                quant: ctx.resolve_quant_table([1; 64]),
                dc_table,
                ac_table: Arc::clone(&empty_huffman),
            });
        }
        if matches!(info.color_space, ColorSpace::Rgb | ColorSpace::YCbCr)
            && lossless_color_sampling(info).is_none()
        {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        let plan = PreparedDecodePlan {
            components,
            sampling: info.sampling,
            color_space: info.color_space,
            restart_interval: header.restart_interval,
            dimensions: info.dimensions,
            scan_offset: header.sos_offset.ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?,
            scratch_bytes: compute_lossless_scratch_bytes(info, DEFAULT_MAX_DECODE_BYTES)?,
        };
        let lossless = PreparedLosslessPlan {
            predictor: scan.ss,
            bit_depth: info.bit_depth,
            dc_table: first_dc_table.ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?,
            dimensions: info.dimensions,
            scan_offset: plan.scan_offset,
        };
        Ok((plan, lossless))
    }

    fn build_progressive_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
    ) -> Result<PreparedProgressivePlan, JpegError> {
        if !matches!(
            info.sof_kind,
            SofKind::Progressive8 | SofKind::Progressive12
        ) {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        match (info.sof_kind, info.color_space) {
            (
                SofKind::Progressive8 | SofKind::Progressive12,
                ColorSpace::Grayscale | ColorSpace::YCbCr | ColorSpace::Rgb,
            ) => {}
            (SofKind::Progressive12, ColorSpace::Cmyk | ColorSpace::Ycck) => {}
            (_, color_space) => return Err(JpegError::UnsupportedColorSpace { color_space }),
        }
        validate_sampling_factors(header, info)?;
        if header.progressive_scans.is_empty() {
            return Err(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            });
        }

        let max_h = u32::from(header.sampling.max_h);
        let max_v = u32::from(header.sampling.max_v);
        let mcu_cols = info.dimensions.0.div_ceil(8 * max_h);
        let mcu_rows = info.dimensions.1.div_ceil(8 * max_v);
        let mut components = Vec::with_capacity(header.component_ids.len());
        for (output_index, &id) in header.component_ids.iter().enumerate() {
            let (h, v) =
                header
                    .sampling
                    .component(output_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })?;
            let quant_id =
                *header
                    .quant_table_ids
                    .get(output_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })? as usize;
            let quant = *header
                .quant_tables
                .entries
                .get(quant_id)
                .and_then(|q| q.as_ref())
                .ok_or(JpegError::MissingQuantTable {
                    component: id,
                    table_id: quant_id as u8,
                })?;
            components.push(PreparedProgressiveComponentPlan {
                h,
                v,
                output_index,
                quant: ctx.resolve_quant_table(quant),
                block_cols: mcu_cols * u32::from(h),
                block_rows: mcu_rows * u32::from(v),
                sample_width: info
                    .dimensions
                    .0
                    .saturating_mul(u32::from(h))
                    .div_ceil(max_h),
                sample_height: info
                    .dimensions
                    .1
                    .saturating_mul(u32::from(v))
                    .div_ceil(max_v),
            });
        }

        let mut scans = Vec::with_capacity(header.progressive_scans.len());
        for parsed in &header.progressive_scans {
            let mut scan_components = Vec::with_capacity(parsed.scan.components.len());
            for component in &parsed.scan.components {
                let component_index = find_component_index(&header.component_ids, component.id)
                    .ok_or(JpegError::UnknownScanComponent {
                        offset: parsed.entropy_offset,
                        component: component.id,
                    })?;
                let quant_id = *header.quant_table_ids.get(component_index).ok_or(
                    JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    },
                )?;
                let _ = parsed
                    .quant_tables
                    .entries
                    .get(quant_id as usize)
                    .and_then(|q| q.as_ref())
                    .ok_or(JpegError::MissingQuantTable {
                        component: component.id,
                        table_id: quant_id,
                    })?;
                let dc_table = if parsed.scan.ss == 0 {
                    Some(resolve_progressive_huffman(
                        ctx,
                        &parsed.huffman_tables.dc,
                        component.id,
                        0,
                        component.dc_table,
                    )?)
                } else {
                    None
                };
                let ac_table = if parsed.scan.ss > 0 {
                    Some(resolve_progressive_huffman(
                        ctx,
                        &parsed.huffman_tables.ac,
                        component.id,
                        1,
                        component.ac_table,
                    )?)
                } else {
                    None
                };
                scan_components.push(PreparedProgressiveScanComponent {
                    component_index,
                    dc_table,
                    ac_table,
                });
            }
            scans.push(PreparedProgressiveScan {
                components: scan_components,
                ss: parsed.scan.ss,
                se: parsed.scan.se,
                ah: parsed.scan.ah,
                al: parsed.scan.al,
                entropy_offset: parsed.entropy_offset,
                restart_interval: parsed.restart_interval,
            });
        }

        let scratch_bytes =
            compute_progressive_scratch_bytes(&components, info.dimensions.0 as usize)?;
        Ok(PreparedProgressivePlan {
            components,
            scans,
            sampling: info.sampling,
            color_space: info.color_space,
            dimensions: info.dimensions,
            mcu_cols,
            mcu_rows,
            scratch_bytes,
        })
    }

    fn build_progressive_host_output_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
    ) -> Result<PreparedDecodePlan, JpegError> {
        let empty_raw = RawHuffmanTable {
            bits: [0; 16],
            values: HuffmanValues::default(),
        };
        let empty_huffman = ctx.resolve_huffman_table(&empty_raw)?;
        let mut components = Vec::with_capacity(header.component_ids.len());
        for (output_index, &id) in header.component_ids.iter().enumerate() {
            let (h, v) =
                header
                    .sampling
                    .component(output_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })?;
            let quant_id =
                *header
                    .quant_table_ids
                    .get(output_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })? as usize;
            let quant = *header
                .quant_tables
                .entries
                .get(quant_id)
                .and_then(|q| q.as_ref())
                .ok_or(JpegError::MissingQuantTable {
                    component: id,
                    table_id: quant_id as u8,
                })?;
            components.push(PreparedComponentPlan {
                h,
                v,
                output_index,
                quant: ctx.resolve_quant_table(quant),
                dc_table: Arc::clone(&empty_huffman),
                ac_table: Arc::clone(&empty_huffman),
            });
        }
        Ok(PreparedDecodePlan {
            components,
            sampling: info.sampling,
            color_space: info.color_space,
            restart_interval: header.restart_interval,
            dimensions: info.dimensions,
            scan_offset: header.sos_offset.ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?,
            scratch_bytes: compute_decode_scratch_bytes(
                info.dimensions,
                info.sampling,
                DEFAULT_MAX_DECODE_BYTES,
            )?,
        })
    }

    /// The parsed header as a public [`Info`].
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Build a restart-marker byte-offset index for the first scan.
    ///
    /// Offsets are absolute byte positions in the original JPEG byte slice.
    /// Returns `Ok(None)` when the stream has no non-zero DRI marker.
    pub(crate) fn restart_index(&self) -> Result<Option<RestartIndex>, JpegError> {
        restart_index_for_stream(
            self.bytes,
            Some(self.plan.scan_offset),
            &self.info,
            self.plan.restart_interval,
        )
    }

    /// Decode the full image into the caller's buffer.
    ///
    /// # Errors
    /// - [`JpegError::OutputBufferTooSmall`] or [`JpegError::InvalidStride`]
    ///   if the provided buffer/stride cannot hold the image at `fmt`.
    /// - [`JpegError::NotImplemented`] if `fmt` requests a raw output the
    ///   current release does not emit (e.g. `RawYCbCr8`).
    /// - Any entropy- or structural-decode error from the scan walker.
    pub fn decode_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH
            .with(|pool| self.decode_into_with_scratch(&mut pool.borrow_mut(), out, stride, fmt))
    }

    /// Decode into a freshly allocated tightly packed buffer using a request
    /// object instead of a method-name cross-product.
    pub fn decode_request(
        &self,
        request: DecodeRequest,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        DEFAULT_SCRATCH
            .with(|pool| self.decode_request_with_scratch(&mut pool.borrow_mut(), request))
    }

    /// Decode the full image into the caller's buffer using the core
    /// `PixelFormat` + `Downscale` contract.
    pub fn decode_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_scaled_into_with_scratch(&mut pool.borrow_mut(), out, stride, fmt, scale)
        })
    }

    fn decode_request_with_scratch(
        &self,
        pool: &mut ScratchPool,
        request: DecodeRequest,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        let legacy = output_format_from_parts(self.info.sof_kind, request.fmt, request.scale)?;
        let (stride, len) = if let Some(roi) = request.region {
            let scaled_roi = scaled_rect_covering(roi, legacy.downscale())?;
            checked_output_geometry(scaled_roi.w, scaled_roi.h, legacy.bytes_per_pixel())?
        } else {
            let (width, height) = scaled_dimensions(self.info.dimensions, legacy.downscale());
            checked_output_geometry(width, height, legacy.bytes_per_pixel())?
        };
        let mut out = allocate_output_buffer(len);
        let outcome = if let Some(roi) = request.region {
            self.decode_region_scaled_into_with_scratch(
                pool,
                &mut out,
                stride,
                request.fmt,
                roi,
                request.scale,
            )?
        } else {
            self.decode_scaled_into_with_scratch(
                pool,
                &mut out,
                stride,
                request.fmt,
                request.scale,
            )?
        };
        Ok((out, outcome))
    }

    /// Decode the full image into the caller's buffer, reusing the supplied
    /// [`ScratchPool`]. On a long-running tile batch this eliminates the
    /// per-tile allocation of stripe buffers, the DC predictor, and the
    /// chroma upsample rows — the realistic WSI reader shape. The first
    /// call against a fresh pool does the allocation; subsequent calls at
    /// the same-or-smaller shape reuse the underlying `Vec`s.
    ///
    /// # Errors
    /// Identical to [`Self::decode_into`].
    pub fn decode_into_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_scaled_into_with_scratch(pool, out, stride, fmt, Downscale::None)
    }

    fn decode_lossless_output_format_region_scaled(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: OutputFormat,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Option<Result<DecodeOutcome, JpegError>> {
        self.lossless_plan.as_ref()?;
        let result = match fmt {
            OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => {
                LosslessRgbRegionFallback::for_color_space_8(self.info.color_space)
                    .decode_rgb_region_scaled_into(self, out, stride, roi, downscale)
            }
            OutputFormat::Rgba8 { alpha } | OutputFormat::Rgba8Scaled { alpha, .. } => {
                LosslessRgbRegionFallback::for_color_space_8(self.info.color_space)
                    .decode_rgba_region_scaled_into(
                        self,
                        out,
                        stride,
                        roi,
                        downscale,
                        LosslessRgbaAlpha::U8(alpha),
                    )
            }
            OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => {
                self.decode_lossless_gray8_region_scaled_into(out, stride, roi, downscale)
            }
            OutputFormat::Gray16 | OutputFormat::Gray16Scaled { .. } => {
                self.decode_lossless_gray16_region_scaled_into(out, stride, roi, downscale)
            }
            OutputFormat::Rgb16 | OutputFormat::Rgb16Scaled { .. } => {
                LosslessRgbRegionFallback::for_color_space_16(self.info.color_space)
                    .decode_rgb_region_scaled_into(self, out, stride, roi, downscale)
            }
            OutputFormat::Rgba16 { alpha } | OutputFormat::Rgba16Scaled { alpha, .. } => {
                LosslessRgbRegionFallback::for_color_space_16(self.info.color_space)
                    .decode_rgba_region_scaled_into(
                        self,
                        out,
                        stride,
                        roi,
                        downscale,
                        LosslessRgbaAlpha::U16(alpha),
                    )
            }
        };
        Some(result)
    }

    fn decode_into_output_format_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: OutputFormat,
    ) -> Result<DecodeOutcome, JpegError> {
        let profile_enabled = jpeg_profile_stages_enabled();
        let total_start = profile_enabled.then(Instant::now);
        let downscale = fmt.downscale();
        let (w, h) = scaled_dimensions(self.info.dimensions, downscale);
        let scratch_bytes = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        let bpp = fmt.bytes_per_pixel();
        validate_buffer(out, stride, w, h, bpp)?;
        let full_roi = Rect::full(self.info.dimensions);
        if let Some(result) =
            self.decode_lossless_output_format_region_scaled(out, stride, fmt, full_roi, downscale)
        {
            return result;
        }
        let decode_start = profile_enabled.then(Instant::now);
        let result = match fmt {
            OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => {
                let mut writer = Rgb8Writer::new_with_backend(out, stride, w, self.backend);
                self.decode_rgb_with_writer(pool, &mut writer, downscale, full_roi)
            }
            OutputFormat::Rgba8 { alpha } | OutputFormat::Rgba8Scaled { alpha, .. } => {
                let mut writer = Rgba8Writer::new_with_backend(out, stride, w, alpha, self.backend);
                self.decode_with_writer(pool, &mut writer, downscale, full_roi)
            }
            OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => {
                let mut writer = Gray8Writer::new(out, stride, w);
                self.decode_with_writer(pool, &mut writer, downscale, full_roi)
            }
            OutputFormat::Gray16 => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_gray16_region_scaled_into(
                        out, stride, full_roi, downscale,
                    );
                }
                self.decode_extended12_gray16_into(out, stride)
            }
            OutputFormat::Gray16Scaled { .. } => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_gray16_region_scaled_into(
                        out, stride, full_roi, downscale,
                    );
                }
                self.decode_extended12_gray16_region_scaled_into(out, stride, full_roi, downscale)
            }
            OutputFormat::Rgb16 => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_rgb16_region_scaled_into(
                        out, stride, full_roi, downscale,
                    );
                }
                self.decode_extended12_rgb16_into(out, stride)
            }
            OutputFormat::Rgb16Scaled { .. } => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_rgb16_region_scaled_into(
                        out, stride, full_roi, downscale,
                    );
                }
                self.decode_extended12_rgb16_region_scaled_into(out, stride, full_roi, downscale)
            }
            OutputFormat::Rgba16 { alpha } | OutputFormat::Rgba16Scaled { alpha, .. } => {
                if matches!(
                    self.info.sof_kind,
                    SofKind::Extended12 | SofKind::Progressive12
                ) {
                    return self.decode_12bit_rgba16_region_scaled_into(
                        out, stride, full_roi, downscale, alpha,
                    );
                }
                Err(JpegError::NotImplemented {
                    sof: self.info.sof_kind,
                })
            }
        };
        if let (Some(total_start), Some(decode_start), Ok(outcome)) =
            (total_start, decode_start, &result)
        {
            let source_width_s = self.info.dimensions.0.to_string();
            let source_height_s = self.info.dimensions.1.to_string();
            let output_width_s = w.to_string();
            let output_height_s = h.to_string();
            let stride_s = stride.to_string();
            let bpp_s = bpp.to_string();
            let output_bytes_s = stride.saturating_mul(h as usize).to_string();
            let scratch_bytes_s = scratch_bytes.to_string();
            let warning_count_s = outcome.warnings.len().to_string();
            let decode_us = duration_us_string(decode_start.elapsed());
            let total_us = duration_us_string(total_start.elapsed());
            emit_jpeg_profile_row(
                "decode",
                "cpu",
                &[
                    ("mode", "full"),
                    ("fmt", output_format_profile_name(fmt)),
                    ("downscale", downscale_profile_name(downscale)),
                    ("source_width", source_width_s.as_str()),
                    ("source_height", source_height_s.as_str()),
                    ("output_width", output_width_s.as_str()),
                    ("output_height", output_height_s.as_str()),
                    ("stride", stride_s.as_str()),
                    ("bpp", bpp_s.as_str()),
                    ("scratch_bytes", scratch_bytes_s.as_str()),
                    ("output_bytes", output_bytes_s.as_str()),
                    ("decode_us", decode_us.as_str()),
                    ("total_us", total_us.as_str()),
                    ("warnings", warning_count_s.as_str()),
                ],
            );
        }
        result
    }

    /// [`Self::decode_scaled_into`] with caller-owned scratch.
    pub fn decode_scaled_into_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_into_output_format_with_scratch(
            pool,
            out,
            stride,
            output_format_from_parts(self.info.sof_kind, fmt, scale)?,
        )
    }

    /// Decode the full image into rows delivered to `sink`.
    ///
    /// DCT-backed and 8-bit lossless color paths emit interleaved RGB8 rows.
    /// Lossless 16-bit grayscale SOF3 emits little-endian Gray16 rows, and
    /// supported lossless 16-bit color SOF3 emits little-endian Rgb16 rows.
    pub fn decode_rows<S>(&self, sink: &mut S) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        DEFAULT_SCRATCH.with(|pool| self.decode_rows_with_scratch(&mut pool.borrow_mut(), sink))
    }

    /// [`Self::decode_rows`] with caller-owned scratch. See
    /// [`Self::decode_into_with_scratch`] for the reuse contract.
    pub fn decode_rows_with_scratch<S>(
        &self,
        pool: &mut ScratchPool,
        sink: &mut S,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        if self.lossless_plan.is_some() {
            return self.decode_lossless_rows_with_scratch(pool, sink);
        }
        let width = self.info.dimensions.0 as usize;
        let rows = pool.take_sink_rows(width);
        let mut writer = SinkWriter::new(sink, rows, self.backend);
        let result = self.decode_rgb_with_writer(
            pool,
            &mut writer,
            DownscaleFactor::Full,
            Rect::full(self.info.dimensions),
        );
        pool.restore_sink_rows(writer.into_rows());
        result
    }

    fn decode_lossless_rows_with_scratch<S>(
        &self,
        pool: &mut ScratchPool,
        sink: &mut S,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if !(1..=7).contains(&plan.predictor) {
            return Err(JpegError::UnsupportedPredictor {
                predictor: plan.predictor,
            });
        }

        let width = self.info.dimensions.0 as usize;
        match (self.info.color_space, plan.bit_depth) {
            (ColorSpace::Grayscale, 8) => {
                let mut rows = pool.take_sink_rows(width);
                let result = self.decode_lossless_gray8_rows(
                    sink,
                    &mut pool.lossless_prev_row,
                    &mut pool.lossless_curr_row,
                    &mut rows.top_row,
                );
                pool.restore_sink_rows(rows);
                result
            }
            (ColorSpace::Grayscale, 16) => self.decode_lossless_gray16_rows(
                sink,
                &mut pool.lossless_prev_row,
                &mut pool.lossless_curr_row,
            ),
            (ColorSpace::Rgb, 8) => self.decode_lossless_color8_rows(
                sink,
                &mut pool.lossless_prev_row,
                &mut pool.lossless_curr_row,
                None,
                ColorSpace::Rgb,
            ),
            (ColorSpace::YCbCr, 8) => {
                let mut rows = pool.take_sink_rows(width);
                let result = self.decode_lossless_color8_rows(
                    sink,
                    &mut pool.lossless_prev_row,
                    &mut pool.lossless_curr_row,
                    Some(&mut rows.top_row),
                    ColorSpace::YCbCr,
                );
                pool.restore_sink_rows(rows);
                result
            }
            (ColorSpace::Rgb, 16) => self.decode_lossless_color16_rows(
                sink,
                &mut pool.lossless_prev_row,
                &mut pool.lossless_curr_row,
                None,
                ColorSpace::Rgb,
            ),
            (ColorSpace::YCbCr, 16) => {
                let mut rows = pool.take_sink_rows(width);
                rows.top_row.resize(width.saturating_mul(6), 0);
                let result = self.decode_lossless_color16_rows(
                    sink,
                    &mut pool.lossless_prev_row,
                    &mut pool.lossless_curr_row,
                    Some(&mut rows.top_row),
                    ColorSpace::YCbCr,
                );
                pool.restore_sink_rows(rows);
                result
            }
            (_, depth) if depth != 8 && depth != 16 => {
                Err(JpegError::UnsupportedBitDepth { depth })
            }
            _ => Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            }),
        }
    }

    /// Decode the full image into component rows.
    pub fn decode_component_rows_with_scratch<W>(
        &self,
        pool: &mut ScratchPool,
        writer: &mut W,
    ) -> Result<DecodeOutcome, JpegError>
    where
        W: ComponentRowWriter,
    {
        self.decode_region_component_rows_with_scratch(
            pool,
            writer,
            Rect::full(self.info.dimensions),
            Downscale::None,
        )
    }

    /// Decode `roi` into component rows, optionally at a reduced scale.
    pub fn decode_region_component_rows_with_scratch<W>(
        &self,
        pool: &mut ScratchPool,
        mut writer: &mut W,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError>
    where
        W: ComponentRowWriter,
    {
        if !roi.is_within(self.info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: roi,
                width: self.info.dimensions.0,
                height: self.info.dimensions.1,
            });
        }

        let downscale = jpeg_downscale(scale);
        let scaled_roi = scaled_rect_covering(roi, downscale)?;

        if roi == Rect::full(self.info.dimensions) {
            self.decode_with_writer(pool, &mut writer, downscale, roi)
        } else {
            let (source_x0, source_width) =
                self.source_window_for_output_rect(downscale, scaled_roi);
            let mut cropped = CroppedWriter::new(writer, scaled_roi, source_x0, source_width);
            self.decode_with_writer(pool, &mut cropped, downscale, roi)
        }
    }

    /// Decode a rectangular region of the image into the caller's buffer.
    ///
    /// `roi` is expressed in source-image coordinates. If `fmt` requests a
    /// downscaled output, the written pixels cover the corresponding bounding
    /// box in the scaled image grid.
    pub fn decode_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_region_into_with_scratch(&mut pool.borrow_mut(), out, stride, fmt, roi)
        })
    }

    /// [`Self::decode_region_into`] with caller-owned scratch.
    pub(crate) fn decode_region_into_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_region_scaled_into_with_scratch(pool, out, stride, fmt, roi, Downscale::None)
    }

    fn decode_region_into_output_format_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: OutputFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        let profile_enabled = jpeg_profile_stages_enabled();
        let total_start = profile_enabled.then(Instant::now);
        if !roi.is_within(self.info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: roi,
                width: self.info.dimensions.0,
                height: self.info.dimensions.1,
            });
        }

        if roi == Rect::full(self.info.dimensions) {
            return self.decode_into_output_format_with_scratch(pool, out, stride, fmt);
        }

        let downscale = fmt.downscale();
        let scaled_roi = scaled_rect_covering(roi, downscale)?;
        let scratch_bytes = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        validate_buffer(
            out,
            stride,
            scaled_roi.w,
            scaled_roi.h,
            fmt.bytes_per_pixel(),
        )?;
        if let Some(result) =
            self.decode_lossless_output_format_region_scaled(out, stride, fmt, roi, downscale)
        {
            return result;
        }

        let decode_start = profile_enabled.then(Instant::now);
        let result = match fmt {
            OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => {
                if fmt == OutputFormat::Rgb8
                    && downscale == DownscaleFactor::Full
                    && self.progressive_plan.is_none()
                    && self.plan.matches_fast_tile_shape()
                {
                    let mut writer =
                        Rgb8Writer::new_with_backend(out, stride, scaled_roi.w, self.backend);
                    let scan_bytes = &self.bytes[self.plan.scan_offset..];
                    let checkpoint = self.checkpoint_for_mcu(
                        scan_bytes,
                        fast_tile_region_first_decode_mcu(&self.plan, roi, DownscaleFactor::Full),
                    )?;
                    let scan_warnings = decode_scan_fast_tile_rgb_region(
                        &self.plan,
                        self.backend,
                        scan_bytes,
                        pool,
                        &mut writer,
                        roi,
                        checkpoint.as_ref(),
                    )?;
                    Ok(DecodeOutcome {
                        decoded: roi,
                        warnings: merged_warnings(&self.warnings, scan_warnings),
                    })
                } else if matches!(fmt, OutputFormat::Rgb8Scaled { .. })
                    && self.progressive_plan.is_none()
                    && self.plan.matches_fast_tile_shape()
                {
                    let mut writer =
                        Rgb8Writer::new_with_backend(out, stride, scaled_roi.w, self.backend);
                    let scan_bytes = &self.bytes[self.plan.scan_offset..];
                    let checkpoint = self.checkpoint_for_mcu(
                        scan_bytes,
                        fast_tile_region_first_decode_mcu(&self.plan, scaled_roi, downscale),
                    )?;
                    let scan_warnings = decode_scan_fast_tile_rgb_region_scaled(
                        &self.plan,
                        self.backend,
                        scan_bytes,
                        pool,
                        &mut writer,
                        FastTileRegionScaledRequest {
                            roi: scaled_roi,
                            downscale,
                            checkpoint: checkpoint.as_ref(),
                        },
                    )?;
                    Ok(DecodeOutcome {
                        decoded: scaled_roi,
                        warnings: merged_warnings(&self.warnings, scan_warnings),
                    })
                } else {
                    let base =
                        Rgb8Writer::new_with_backend(out, stride, scaled_roi.w, self.backend);
                    let (source_x0, source_width) =
                        self.source_window_for_output_rect(downscale, scaled_roi);
                    let mut writer = CroppedWriter::new(base, scaled_roi, source_x0, source_width);
                    self.decode_rgb_with_writer(pool, &mut writer, downscale, roi)
                }
            }
            OutputFormat::Rgba8 { alpha } | OutputFormat::Rgba8Scaled { alpha, .. } => {
                let base =
                    Rgba8Writer::new_with_backend(out, stride, scaled_roi.w, alpha, self.backend);
                let (source_x0, source_width) =
                    self.source_window_for_output_rect(downscale, scaled_roi);
                let mut writer = CroppedWriter::new(base, scaled_roi, source_x0, source_width);
                self.decode_with_writer(pool, &mut writer, downscale, roi)
            }
            OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => {
                let base = Gray8Writer::new(out, stride, scaled_roi.w);
                let (source_x0, source_width) =
                    self.source_window_for_output_rect(downscale, scaled_roi);
                let mut writer = CroppedWriter::new(base, scaled_roi, source_x0, source_width);
                self.decode_with_writer(pool, &mut writer, downscale, roi)
            }
            OutputFormat::Gray16 => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_gray16_region_scaled_into(
                        out, stride, roi, downscale,
                    );
                }
                self.decode_extended12_gray16_region_into(out, stride, roi)
            }
            OutputFormat::Gray16Scaled { .. } => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_gray16_region_scaled_into(
                        out, stride, roi, downscale,
                    );
                }
                self.decode_extended12_gray16_region_scaled_into(out, stride, roi, downscale)
            }
            OutputFormat::Rgb16 => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_rgb16_region_scaled_into(
                        out, stride, roi, downscale,
                    );
                }
                self.decode_extended12_rgb16_region_into(out, stride, roi)
            }
            OutputFormat::Rgb16Scaled { .. } => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_rgb16_region_scaled_into(
                        out, stride, roi, downscale,
                    );
                }
                self.decode_extended12_rgb16_region_scaled_into(out, stride, roi, downscale)
            }
            OutputFormat::Rgba16 { alpha } | OutputFormat::Rgba16Scaled { alpha, .. } => {
                if matches!(
                    self.info.sof_kind,
                    SofKind::Extended12 | SofKind::Progressive12
                ) {
                    return self.decode_12bit_rgba16_region_scaled_into(
                        out, stride, roi, downscale, alpha,
                    );
                }
                Err(JpegError::NotImplemented {
                    sof: self.info.sof_kind,
                })
            }
        };
        if let (Some(total_start), Some(decode_start), Ok(outcome)) =
            (total_start, decode_start, &result)
        {
            let source_width_s = self.info.dimensions.0.to_string();
            let source_height_s = self.info.dimensions.1.to_string();
            let roi_x_s = roi.x.to_string();
            let roi_y_s = roi.y.to_string();
            let roi_w_s = roi.w.to_string();
            let roi_h_s = roi.h.to_string();
            let output_width_s = scaled_roi.w.to_string();
            let output_height_s = scaled_roi.h.to_string();
            let stride_s = stride.to_string();
            let bpp_s = fmt.bytes_per_pixel().to_string();
            let output_bytes_s = stride.saturating_mul(scaled_roi.h as usize).to_string();
            let scratch_bytes_s = scratch_bytes.to_string();
            let warning_count_s = outcome.warnings.len().to_string();
            let decode_us = duration_us_string(decode_start.elapsed());
            let total_us = duration_us_string(total_start.elapsed());
            let mode = if downscale == DownscaleFactor::Full {
                "region"
            } else {
                "region_scaled"
            };
            emit_jpeg_profile_row(
                "decode",
                "cpu",
                &[
                    ("mode", mode),
                    ("fmt", output_format_profile_name(fmt)),
                    ("downscale", downscale_profile_name(downscale)),
                    ("source_width", source_width_s.as_str()),
                    ("source_height", source_height_s.as_str()),
                    ("roi_x", roi_x_s.as_str()),
                    ("roi_y", roi_y_s.as_str()),
                    ("roi_w", roi_w_s.as_str()),
                    ("roi_h", roi_h_s.as_str()),
                    ("output_width", output_width_s.as_str()),
                    ("output_height", output_height_s.as_str()),
                    ("stride", stride_s.as_str()),
                    ("bpp", bpp_s.as_str()),
                    ("scratch_bytes", scratch_bytes_s.as_str()),
                    ("output_bytes", output_bytes_s.as_str()),
                    ("decode_us", decode_us.as_str()),
                    ("total_us", total_us.as_str()),
                    ("warnings", warning_count_s.as_str()),
                ],
            );
        }
        result
    }

    /// Decode `roi` into the caller's buffer using the core `PixelFormat` +
    /// `Downscale` contract.
    pub fn decode_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_region_scaled_into_with_scratch(
                &mut pool.borrow_mut(),
                out,
                stride,
                fmt,
                roi,
                scale,
            )
        })
    }

    /// [`Self::decode_region_scaled_into`] with caller-owned scratch.
    pub fn decode_region_scaled_into_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_region_into_output_format_with_scratch(
            pool,
            out,
            stride,
            output_format_from_parts(self.info.sof_kind, fmt, scale)?,
            roi,
        )
    }
}

/// One-shot parse-plus-decode of an independent JPEG tile into the caller's
/// buffer, reusing a pre-allocated [`ScratchPool`]. This is the primitive
/// WSI tile-batch readers want: one function call per tile, with all
/// heap state external.
///
/// Parallelism is the caller's responsibility for this primitive. For
/// production batch decode, use [`decode_tiles_into`].
///
/// # Example
///
/// ```no_run
/// use j2k_jpeg::{decode_tile_into, PixelFormat, ScratchPool};
///
/// let bytes: &[u8] = &[];
/// let mut out = vec![0u8; 256 * 256 * 3];
/// let mut pool = ScratchPool::new();
/// decode_tile_into(bytes, &mut pool, &mut out, 256 * 3, PixelFormat::Rgb8)?;
/// # Ok::<(), j2k_jpeg::JpegError>(())
/// ```
///
/// # Errors
/// Forwarded from [`Decoder::new`] (parse) and
/// [`Decoder::decode_into_with_scratch`] (decode).
pub fn decode_tile_into(
    bytes: &[u8],
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<DecodeOutcome, JpegError> {
    DEFAULT_CONTEXT.with(|ctx| {
        decode_tile_into_in_context(bytes, &mut ctx.borrow_mut(), pool, out, stride, fmt)
    })
}

/// One-shot parse-plus-decode of an independent JPEG tile into the caller's
/// buffer, reusing both caller-owned [`DecoderContext`] and caller-owned
/// [`ScratchPool`].
#[doc(hidden)]
pub fn decode_tile_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<DecodeOutcome, JpegError> {
    decode_tile_into_in_context_with_options(
        bytes,
        ctx,
        pool,
        out,
        stride,
        fmt,
        DecodeOptions::default(),
    )
}

/// One-shot parse-plus-decode of an independent JPEG tile into the caller's
/// buffer, reusing both caller-owned [`DecoderContext`] and caller-owned
/// [`ScratchPool`], with explicit JPEG decode options.
#[doc(hidden)]
pub fn decode_tile_into_in_context_with_options(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    options: DecodeOptions,
) -> Result<DecodeOutcome, JpegError> {
    let dec = Decoder::from_view_in_context(JpegView::parse_with_options(bytes, options)?, ctx)?;
    dec.decode_into_with_scratch(pool, out, stride, fmt)
}

pub(crate) fn decode_prepared_jpeg_tile_rgb8_in_context(
    input: &PreparedJpeg<'_>,
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    options: DecodeOptions,
) -> Result<DecodedTile, JpegError> {
    let view = JpegView::parse_with_options(input.as_bytes(), options)?;
    let dimensions = view.info().dimensions;
    let dec = Decoder::from_view_in_context(view, ctx)?;
    let outcome = dec.decode_into_with_scratch(pool, out, stride, PixelFormat::Rgb8)?;
    Ok(DecodedTile {
        dimensions,
        decoded: outcome.decoded,
        warnings: outcome.warnings,
    })
}

/// Decode prepared TIFF/WSI JPEG tiles into caller-owned RGB8 output buffers.
///
/// Returned results preserve the caller's input order. Each job carries its
/// own [`DecodeOptions`], allowing container metadata to resolve RGB/YCbCr
/// interpretation independently per tile.
#[must_use]
pub fn decode_prepared_jpeg_tiles_rgb8(
    jobs: &mut [PreparedJpegTileJob<'_, '_>],
) -> Vec<Result<DecodedTile, JpegError>> {
    crate::JpegBatchSession::new_one_shot(TileBatchOptions::default())
        .decode_prepared_jpeg_tiles_rgb8(jobs)
}

/// Decode independent JPEG tiles into caller-owned output buffers using a
/// scoped CPU worker pool.
///
/// Each worker owns one [`DecoderContext`] and one [`ScratchPool`], so repeated
/// tiles reuse parsed table state and heap scratch within that worker without
/// sharing mutable decoder state across threads. Returned outcomes preserve
/// the caller's input order.
///
/// # Errors
/// Returns [`TileBatchError`] with the first failing tile index in input order.
pub fn decode_tiles_into(
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    decode_tiles_into_with_options(jobs, fmt, DecodeOptions::default(), options)
}

/// Decode independent JPEG tiles into caller-owned output buffers using a
/// scoped CPU worker pool and explicit JPEG decode options.
///
/// Use this variant when container metadata has already resolved ambiguous
/// three-component JPEG data to RGB or YCbCr via [`DecodeOptions`].
///
/// # Errors
/// Returns [`TileBatchError`] with the first failing tile index in input order.
#[doc(hidden)]
pub fn decode_tiles_into_with_options(
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    decode_options: DecodeOptions,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    crate::JpegBatchSession::new_one_shot(options).decode_tiles_into_with_options(
        jobs,
        fmt,
        decode_options,
    )
}

/// Decode independent JPEG tiles at reduced resolution into caller-owned
/// output buffers using a scoped CPU worker pool.
///
/// Each worker owns one [`DecoderContext`] and one [`ScratchPool`], so repeated
/// tiles reuse parsed table state and heap scratch within that worker without
/// sharing mutable decoder state across threads. Returned outcomes preserve
/// the caller's input order.
///
/// # Errors
/// Returns [`TileBatchError`] with the first failing tile index in input order.
pub fn decode_tiles_scaled_into(
    jobs: &mut [TileScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    decode_tiles_scaled_into_with_options(jobs, fmt, DecodeOptions::default(), options)
}

/// Decode independent JPEG tiles at reduced resolution into caller-owned
/// output buffers using a scoped CPU worker pool and explicit JPEG decode
/// options.
///
/// # Errors
/// Returns [`TileBatchError`] with the first failing tile index in input order.
#[doc(hidden)]
pub fn decode_tiles_scaled_into_with_options(
    jobs: &mut [TileScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    decode_options: DecodeOptions,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    crate::JpegBatchSession::new_one_shot(options).decode_tiles_scaled_into_with_options(
        jobs,
        fmt,
        decode_options,
    )
}

/// Decode independent JPEG tile regions at reduced resolution into
/// caller-owned output buffers using a scoped CPU worker pool.
///
/// Each worker owns one [`DecoderContext`] and one [`ScratchPool`], so repeated
/// tiles reuse parsed table state and heap scratch within that worker without
/// sharing mutable decoder state across threads. Returned outcomes preserve
/// the caller's input order.
///
/// # Errors
/// Returns [`TileBatchError`] with the first failing tile index in input order.
pub fn decode_tiles_region_scaled_into(
    jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    decode_tiles_region_scaled_into_with_options(jobs, fmt, DecodeOptions::default(), options)
}

/// Decode independent JPEG tile regions at reduced resolution into
/// caller-owned output buffers using a scoped CPU worker pool and explicit JPEG
/// decode options.
///
/// # Errors
/// Returns [`TileBatchError`] with the first failing tile index in input order.
#[doc(hidden)]
pub fn decode_tiles_region_scaled_into_with_options(
    jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    decode_options: DecodeOptions,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    crate::JpegBatchSession::new_one_shot(options).decode_tiles_region_scaled_into_with_options(
        jobs,
        fmt,
        decode_options,
    )
}

/// One-shot parse-plus-region-decode of an independent JPEG tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`ScratchPool`].
#[doc(hidden)]
pub fn decode_tile_region_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    output: TileDecodeOutput<'_>,
    roi: Rect,
) -> Result<DecodeOutcome, JpegError> {
    decode_tile_region_into_in_context_with_options(
        bytes,
        ctx,
        pool,
        output,
        roi,
        DecodeOptions::default(),
    )
}

/// One-shot parse-plus-region-decode of an independent JPEG tile into the
/// caller's buffer, reusing caller-owned state and explicit JPEG decode
/// options.
#[doc(hidden)]
pub fn decode_tile_region_into_in_context_with_options(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    output: TileDecodeOutput<'_>,
    roi: Rect,
    options: DecodeOptions,
) -> Result<DecodeOutcome, JpegError> {
    let TileDecodeOutput { out, stride, fmt } = output;
    let dec = Decoder::from_view_in_context(JpegView::parse_with_options(bytes, options)?, ctx)?;
    dec.decode_region_into_with_scratch(pool, out, stride, fmt, roi)
}

/// One-shot parse-plus-scaled-decode of an independent JPEG tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`ScratchPool`].
#[doc(hidden)]
pub fn decode_tile_scaled_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    output: TileDecodeOutput<'_>,
    scale: Downscale,
) -> Result<DecodeOutcome, JpegError> {
    decode_tile_scaled_into_in_context_with_options(
        bytes,
        ctx,
        pool,
        output,
        scale,
        DecodeOptions::default(),
    )
}

/// One-shot parse-plus-scaled-decode of an independent JPEG tile into the
/// caller's buffer, reusing caller-owned state and explicit JPEG decode
/// options.
#[doc(hidden)]
pub fn decode_tile_scaled_into_in_context_with_options(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    output: TileDecodeOutput<'_>,
    scale: Downscale,
    options: DecodeOptions,
) -> Result<DecodeOutcome, JpegError> {
    let TileDecodeOutput { out, stride, fmt } = output;
    let dec = Decoder::from_view_in_context(JpegView::parse_with_options(bytes, options)?, ctx)?;
    dec.decode_scaled_into_with_scratch(pool, out, stride, fmt, scale)
}

/// One-shot parse-plus-region-scaled-decode of an independent JPEG tile into
/// the caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`ScratchPool`].
#[doc(hidden)]
pub fn decode_tile_region_scaled_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    output: TileDecodeOutput<'_>,
    roi: Rect,
    scale: Downscale,
) -> Result<DecodeOutcome, JpegError> {
    decode_tile_region_scaled_into_in_context_with_options(
        bytes,
        ctx,
        pool,
        output,
        roi,
        scale,
        DecodeOptions::default(),
    )
}

/// One-shot parse-plus-region-scaled-decode of an independent JPEG tile into
/// the caller's buffer, reusing caller-owned state and explicit JPEG decode
/// options.
#[doc(hidden)]
pub fn decode_tile_region_scaled_into_in_context_with_options(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    output: TileDecodeOutput<'_>,
    roi: Rect,
    scale: Downscale,
    options: DecodeOptions,
) -> Result<DecodeOutcome, JpegError> {
    let TileDecodeOutput { out, stride, fmt } = output;
    let dec = Decoder::from_view_in_context(JpegView::parse_with_options(bytes, options)?, ctx)?;
    dec.decode_region_scaled_into_with_scratch(pool, out, stride, fmt, roi, scale)
}

impl Decoder<'_> {
    fn decode_scratch_bytes(&self, cap: usize) -> Result<usize, JpegError> {
        let scratch_bytes = self
            .progressive_plan
            .as_ref()
            .map_or(self.plan.scratch_bytes, |plan| plan.scratch_bytes);
        if scratch_bytes > cap {
            return Err(JpegError::MemoryCapExceeded {
                requested: scratch_bytes,
                cap,
            });
        }
        Ok(scratch_bytes)
    }

    fn checkpoint_for_mcu(
        &self,
        scan_bytes: &[u8],
        target_mcu: u32,
    ) -> Result<Option<DeviceCheckpoint>, JpegError> {
        if self.plan.restart_interval.is_some() || target_mcu < CPU_ROI_CHECKPOINT_MIN_TARGET_MCUS {
            return Ok(None);
        }

        let mut cache =
            self.cpu_entropy_checkpoints
                .lock()
                .map_err(|_| JpegError::InternalInvariant {
                    reason: "CPU entropy checkpoint cache mutex poisoned",
                })?;
        checkpoint_before_mcu(
            &self.plan,
            scan_bytes,
            CPU_ROI_CHECKPOINT_CADENCE_MCUS,
            target_mcu,
            &mut cache,
        )
    }

    fn source_window_for_output_rect(
        &self,
        downscale: DownscaleFactor,
        output_rect: Rect,
    ) -> (u32, u32) {
        if self.progressive_plan.is_some() {
            return (0, scaled_dimensions(self.info.dimensions, downscale).0);
        }
        let layout = stripe_region_layout(&self.plan, downscale, output_rect);
        (layout.source_x0, layout.source_width)
    }

    fn decode_with_writer<W: OutputWriter>(
        &self,
        pool: &mut ScratchPool,
        writer: &mut W,
        downscale: DownscaleFactor,
        decoded: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        let _ = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        let profile_enabled = jpeg_profile_stages_enabled();
        if let Some(plan) = &self.progressive_plan {
            let scan_start = profile_enabled.then(Instant::now);
            let scan_warnings = if downscale == DownscaleFactor::Full {
                decode_progressive(plan, self.backend, self.bytes, writer)?
            } else {
                let mut scaled =
                    ProgressiveDownscaleWriter::new(writer, downscale, self.info.dimensions);
                decode_progressive(plan, self.backend, self.bytes, &mut scaled)?
            };
            if let Some(start) = scan_start {
                emit_decode_scan_profile(
                    "progressive",
                    self.info.dimensions,
                    decoded,
                    downscale,
                    start.elapsed(),
                );
            }
            return Ok(DecodeOutcome {
                decoded,
                warnings: merged_warnings(&self.warnings, scan_warnings),
            });
        }
        let output_rect = scaled_rect_covering(decoded, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let scan_start = profile_enabled.then(Instant::now);
        let scan_warnings = decode_scan_baseline(
            &self.plan,
            self.backend,
            scan_bytes,
            pool,
            writer,
            downscale,
            output_rect,
        )?;
        if let Some(start) = scan_start {
            emit_decode_scan_profile(
                "baseline",
                self.info.dimensions,
                decoded,
                downscale,
                start.elapsed(),
            );
        }
        Ok(DecodeOutcome {
            decoded,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_rgb_with_writer<W: OutputWriter + InterleavedRgbWriter>(
        &self,
        pool: &mut ScratchPool,
        writer: &mut W,
        downscale: DownscaleFactor,
        decoded: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        let _ = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        let profile_enabled = jpeg_profile_stages_enabled();
        if let Some(plan) = &self.progressive_plan {
            let scan_start = profile_enabled.then(Instant::now);
            let scan_warnings = if downscale == DownscaleFactor::Full {
                decode_progressive(plan, self.backend, self.bytes, writer)?
            } else {
                let mut scaled =
                    ProgressiveDownscaleWriter::new(writer, downscale, self.info.dimensions);
                decode_progressive(plan, self.backend, self.bytes, &mut scaled)?
            };
            if let Some(start) = scan_start {
                emit_decode_scan_profile(
                    "progressive_rgb",
                    self.info.dimensions,
                    decoded,
                    downscale,
                    start.elapsed(),
                );
            }
            return Ok(DecodeOutcome {
                decoded,
                warnings: merged_warnings(&self.warnings, scan_warnings),
            });
        }
        let output_rect = scaled_rect_covering(decoded, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let scan_start = profile_enabled.then(Instant::now);
        let (scan_path, scan_warnings) =
            if downscale == DownscaleFactor::Full && self.plan.matches_fast_tile_shape() {
                (
                    "fast420_rgb",
                    decode_scan_fast_tile_rgb(&self.plan, self.backend, scan_bytes, pool, writer)?,
                )
            } else if downscale == DownscaleFactor::Full
                && decoded == Rect::full(self.info.dimensions)
                && self.plan.matches_fast_rgb444_shape()
            {
                (
                    "fast444_rgb",
                    decode_scan_fast_rgb_444(&self.plan, self.backend, scan_bytes, pool, writer)?,
                )
            } else {
                (
                    "baseline_rgb",
                    decode_scan_baseline_rgb(
                        &self.plan,
                        self.backend,
                        scan_bytes,
                        pool,
                        writer,
                        downscale,
                        output_rect,
                    )?,
                )
            };
        if let Some(start) = scan_start {
            emit_decode_scan_profile(
                scan_path,
                self.info.dimensions,
                decoded,
                downscale,
                start.elapsed(),
            );
        }
        Ok(DecodeOutcome {
            decoded,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_lossless_gray8_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if plan.bit_depth != 8 {
            return Err(JpegError::UnsupportedBitDepth {
                depth: plan.bit_depth,
            });
        }
        if !(1..=7).contains(&plan.predictor) {
            return Err(JpegError::UnsupportedPredictor {
                predictor: plan.predictor,
            });
        }

        let (width, height) = plan.dimensions;
        let scan_bytes = &self.bytes[plan.scan_offset..];
        let mut br = BitReader::new(scan_bytes);
        let total_samples = width.saturating_mul(height);
        let mut restart_tracker =
            LosslessRestartTracker::new(self.plan.restart_interval, total_samples);
        for y in 0..height as usize {
            for x in 0..width as usize {
                let sample_index = y as u32 * width + x as u32;
                let restart_first_sample = restart_tracker.begin_unit(&mut br, sample_index)?;
                let predictor = if restart_first_sample {
                    128
                } else {
                    lossless_predictor_value(plan.predictor, out, stride, x, y)
                };
                let diff = plan.dc_table.decode_fast_dc(&mut br)?;
                let sample = <u8 as LosslessSample>::from_i32(predictor + diff)?;
                out[y * stride + x] = sample;
                restart_tracker.finish_unit();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_lossless_gray8_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        if roi == Rect::full(self.info.dimensions) && downscale == DownscaleFactor::Full {
            return self.decode_lossless_gray8_into(out, stride);
        }

        let (width, height) = self.info.dimensions;
        let full_stride = width as usize;
        let mut full =
            allocate_output_buffer(checked_scratch_len(&[full_stride, height as usize])?);
        let mut outcome = self.decode_lossless_gray8_into(&mut full, full_stride)?;
        let output_rect = scaled_rect_covering(roi, downscale)?;
        copy_gray8_scaled_rect(
            &full,
            (width, height),
            output_rect,
            downscale.denominator(),
            out,
            stride,
        );
        outcome.decoded = roi;
        Ok(outcome)
    }

    fn decode_lossless_gray_rows<P, S>(
        &self,
        sink: &mut S,
        prev_row: &mut Vec<u8>,
        curr_row: &mut Vec<u8>,
        mut emit_row: impl FnMut(&mut S, u32, &[u8]) -> Result<(), JpegError>,
    ) -> Result<DecodeOutcome, JpegError>
    where
        P: LosslessSample,
        S: RowSink<u8, Error = JpegError>,
    {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if plan.bit_depth != P::BIT_DEPTH {
            return Err(JpegError::UnsupportedBitDepth {
                depth: plan.bit_depth,
            });
        }
        if self.info.color_space != ColorSpace::Grayscale {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }
        if !(1..=7).contains(&plan.predictor) {
            return Err(JpegError::UnsupportedPredictor {
                predictor: plan.predictor,
            });
        }

        let (width, height) = plan.dimensions;
        let width = width as usize;
        let row_len = width.saturating_mul(P::BYTES);
        prev_row.resize(row_len, 0);
        curr_row.resize(row_len, 0);

        let scan_bytes = &self.bytes[plan.scan_offset..];
        let mut br = BitReader::new(scan_bytes);
        let total_samples = plan.dimensions.0.saturating_mul(height);
        let mut restart_tracker =
            LosslessRestartTracker::new(self.plan.restart_interval, total_samples);
        for y in 0..height as usize {
            for x in 0..width {
                let sample_index = y as u32 * plan.dimensions.0 + x as u32;
                let restart_first_sample = restart_tracker.begin_unit(&mut br, sample_index)?;
                let predictor = if restart_first_sample {
                    P::RESTART_PREDICTOR
                } else {
                    lossless_predictor_gray_rows::<P>(plan.predictor, curr_row, prev_row, x, y)
                };
                let diff = plan.dc_table.decode_fast_dc(&mut br)?;
                let sample = P::from_i32(predictor + diff)?;
                sample.write_le(&mut curr_row[x * P::BYTES..]);
                restart_tracker.finish_unit();
            }
            emit_row(sink, y as u32, &curr_row[..row_len])?;
            core::mem::swap(prev_row, curr_row);
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_lossless_gray8_rows<S>(
        &self,
        sink: &mut S,
        prev_row: &mut Vec<u8>,
        curr_row: &mut Vec<u8>,
        rgb_row: &mut [u8],
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        self.decode_lossless_gray_rows::<u8, S>(sink, prev_row, curr_row, |sink, y, gray_row| {
            let rgb_len = gray_row.len().saturating_mul(3);
            if rgb_row.len() < rgb_len {
                return Err(JpegError::OutputBufferTooSmall {
                    required: rgb_len,
                    provided: rgb_row.len(),
                });
            }
            for (pixel, &sample) in rgb_row[..rgb_len].chunks_exact_mut(3).zip(gray_row.iter()) {
                pixel.copy_from_slice(&[sample, sample, sample]);
            }
            sink.write_row(y, &rgb_row[..rgb_len])
        })
    }

    fn decode_lossless_rgb8_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color8_output_into(out, stride, ColorSpace::Rgb)
    }

    fn decode_lossless_color8_output_into(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        match lossless_color_sampling(&self.info) {
            Some(LosslessColorSampling::S444) => {
                let outcome =
                    self.decode_lossless_color8_components_into(out, stride, color_space)?;
                if color_space == ColorSpace::YCbCr {
                    convert_ycbcr8_to_rgb8_in_place(out, stride, self.info.dimensions);
                }
                Ok(outcome)
            }
            Some(LosslessColorSampling::S422 | LosslessColorSampling::S420) => {
                self.decode_lossless_color8_sampled_into(out, stride, color_space)
            }
            None => Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            }),
        }
    }

    fn decode_lossless_color_components_into<P>(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError>
    where
        P: LosslessSample,
    {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        validate_lossless_color_plan::<P>(plan, &self.plan, &self.info, color_space)?;

        let (width, height) = plan.dimensions;
        let scan_bytes = &self.bytes[plan.scan_offset..];
        let mut br = BitReader::new(scan_bytes);
        let total_pixels = width.saturating_mul(height);
        let mut restart_tracker =
            LosslessRestartTracker::new(self.plan.restart_interval, total_pixels);
        for y in 0..height as usize {
            for x in 0..width as usize {
                let pixel_index = y as u32 * width + x as u32;
                let restart_first_pixel = restart_tracker.begin_unit(&mut br, pixel_index)?;
                decode_lossless_color_sample::<P, _>(
                    &mut br,
                    &self.plan.components,
                    plan.predictor,
                    restart_first_pixel,
                    &mut LosslessColorIntoSample {
                        out: &mut *out,
                        stride,
                        x,
                        y,
                    },
                )?;
                restart_tracker.finish_unit();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_lossless_color8_components_into(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color_components_into::<u8>(out, stride, color_space)
    }

    fn decode_lossless_color_sampled_into<P>(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
        write_output: impl FnOnce(
            &mut [u8],
            usize,
            ColorSpace,
            LosslessColorSampling,
            (usize, usize),
            LosslessColorPlanes<'_, P>,
        ),
    ) -> Result<DecodeOutcome, JpegError>
    where
        P: LosslessSample,
    {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        let sampling = lossless_color_sampling(&self.info).ok_or(JpegError::NotImplemented {
            sof: self.info.sof_kind,
        })?;
        if !matches!(
            sampling,
            LosslessColorSampling::S422 | LosslessColorSampling::S420
        ) {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }
        validate_lossless_color_plan::<P>(plan, &self.plan, &self.info, color_space)?;

        let (width, height) = plan.dimensions;
        let width = width as usize;
        let height = height as usize;
        let chroma_width = width.div_ceil(self.info.sampling.max_h as usize);
        let chroma_height = height.div_ceil(self.info.sampling.max_v as usize);
        let mut c0 = vec![P::default(); width * height];
        let mut c1 = vec![P::default(); chroma_width * chroma_height];
        let mut c2 = vec![P::default(); chroma_width * chroma_height];
        let mut planes = LosslessSampledColorPlanesMut {
            c0: &mut c0,
            c1: &mut c1,
            c2: &mut c2,
            dimensions: (width, height),
            chroma_dimensions: (chroma_width, chroma_height),
        };

        let scan_bytes = &self.bytes[plan.scan_offset..];
        let mut br = BitReader::new(scan_bytes);
        let total_mcus = (chroma_width * chroma_height) as u32;
        let mut restart_tracker =
            LosslessRestartTracker::new(self.plan.restart_interval, total_mcus);
        for mcu_y in 0..chroma_height {
            for mcu_x in 0..chroma_width {
                let mcu_index = (mcu_y * chroma_width + mcu_x) as u32;
                let restart_first_mcu = restart_tracker.begin_unit(&mut br, mcu_index)?;
                decode_lossless_sampled_color_mcu::<P>(
                    &mut br,
                    &self.plan.components,
                    plan.predictor,
                    LosslessSampledMcu {
                        x: mcu_x,
                        y: mcu_y,
                        restart_first_mcu,
                    },
                    &mut planes,
                )?;
                restart_tracker.finish_unit();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        let LosslessSampledColorPlanesMut { c0, c1, c2, .. } = planes;
        write_output(
            out,
            stride,
            color_space,
            sampling,
            (width, height),
            LosslessColorPlanes { c0, c1, c2 },
        );
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_lossless_color8_sampled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color_sampled_into::<u8>(
            out,
            stride,
            color_space,
            write_lossless_color8_sampled_output,
        )
    }

    fn decode_lossless_ycbcr8_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color8_output_into(out, stride, ColorSpace::YCbCr)
    }

    fn decode_lossless_color_rows<P, S>(
        &self,
        sink: &mut S,
        prev_row: &mut Vec<u8>,
        curr_row: &mut Vec<u8>,
        conversion_row: Option<&mut [u8]>,
        color_space: ColorSpace,
        convert_row: impl Fn(&[u8], &mut [u8]),
    ) -> Result<DecodeOutcome, JpegError>
    where
        P: LosslessSample,
        S: RowSink<u8, Error = JpegError>,
    {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        validate_lossless_color_plan::<P>(plan, &self.plan, &self.info, color_space)?;

        let (width, height) = plan.dimensions;
        let width = width as usize;
        let row_len = width.saturating_mul(3 * P::BYTES);
        prev_row.resize(row_len, 0);
        curr_row.resize(row_len, 0);

        let scan_bytes = &self.bytes[plan.scan_offset..];
        let mut br = BitReader::new(scan_bytes);
        let total_pixels = plan.dimensions.0.saturating_mul(height);
        let mut restart_tracker =
            LosslessRestartTracker::new(self.plan.restart_interval, total_pixels);
        let mut conversion_row = conversion_row;
        for y in 0..height as usize {
            for x in 0..width {
                let pixel_index = y as u32 * plan.dimensions.0 + x as u32;
                let restart_first_pixel = restart_tracker.begin_unit(&mut br, pixel_index)?;
                decode_lossless_color_sample::<P, _>(
                    &mut br,
                    &self.plan.components,
                    plan.predictor,
                    restart_first_pixel,
                    &mut LosslessColorRowSample {
                        curr_row: &mut *curr_row,
                        prev_row: &*prev_row,
                        x,
                        y,
                    },
                )?;
                restart_tracker.finish_unit();
            }
            let row = if color_space == ColorSpace::YCbCr {
                let row = conversion_row
                    .as_deref_mut()
                    .ok_or(JpegError::OutputBufferTooSmall {
                        required: row_len,
                        provided: 0,
                    })?;
                if row.len() < row_len {
                    return Err(JpegError::OutputBufferTooSmall {
                        required: row_len,
                        provided: row.len(),
                    });
                }
                convert_row(&curr_row[..row_len], &mut row[..row_len]);
                &row[..row_len]
            } else {
                &curr_row[..row_len]
            };
            sink.write_row(y as u32, row)?;
            core::mem::swap(prev_row, curr_row);
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_lossless_color8_rows<S>(
        &self,
        sink: &mut S,
        prev_row: &mut Vec<u8>,
        curr_row: &mut Vec<u8>,
        conversion_row: Option<&mut [u8]>,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        self.decode_lossless_color_rows::<u8, S>(
            sink,
            prev_row,
            curr_row,
            conversion_row,
            color_space,
            copy_ycbcr8_row_to_rgb8,
        )
    }

    fn decode_lossless_gray16_rows<S>(
        &self,
        sink: &mut S,
        prev_row: &mut Vec<u8>,
        curr_row: &mut Vec<u8>,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        self.decode_lossless_gray_rows::<u16, S>(sink, prev_row, curr_row, |sink, y, row| {
            sink.write_row(y, row)
        })
    }

    fn decode_lossless_color16_rows<S>(
        &self,
        sink: &mut S,
        prev_row: &mut Vec<u8>,
        curr_row: &mut Vec<u8>,
        conversion_row: Option<&mut [u8]>,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        self.decode_lossless_color_rows::<u16, S>(
            sink,
            prev_row,
            curr_row,
            conversion_row,
            color_space,
            copy_ycbcr16_row_to_rgb16,
        )
    }

    fn decode_lossless_rgb16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color16_output_into(out, stride, ColorSpace::Rgb)
    }

    fn decode_lossless_color16_output_into(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        match lossless_color_sampling(&self.info) {
            Some(LosslessColorSampling::S444) => {
                let outcome =
                    self.decode_lossless_color16_components_into(out, stride, color_space)?;
                if color_space == ColorSpace::YCbCr {
                    convert_ycbcr16_to_rgb16_in_place(out, stride, self.info.dimensions);
                }
                Ok(outcome)
            }
            Some(LosslessColorSampling::S422 | LosslessColorSampling::S420) => {
                self.decode_lossless_color16_sampled_into(out, stride, color_space)
            }
            None => Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            }),
        }
    }

    fn decode_lossless_color16_components_into(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color_components_into::<u16>(out, stride, color_space)
    }

    fn decode_lossless_color16_sampled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color_sampled_into::<u16>(
            out,
            stride,
            color_space,
            write_lossless_color16_sampled_output,
        )
    }

    fn decode_lossless_ycbcr16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color16_output_into(out, stride, ColorSpace::YCbCr)
    }

    fn decode_12bit_rgba16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        alpha: u16,
    ) -> Result<DecodeOutcome, JpegError> {
        let output_rect = scaled_rect_covering(roi, downscale)?;
        let rgb_stride = output_rect.w as usize * 6;
        let mut rgb =
            allocate_output_buffer(checked_scratch_len(&[rgb_stride, output_rect.h as usize])?);
        let outcome = if self.info.sof_kind == SofKind::Progressive12 {
            self.decode_progressive12_rgb16_region_scaled_into(&mut rgb, rgb_stride, roi, downscale)
        } else {
            self.decode_extended12_rgb16_region_scaled_into(&mut rgb, rgb_stride, roi, downscale)
        }?;
        copy_rgb16_to_rgba16(
            &rgb,
            rgb_stride,
            output_rect.w,
            output_rect.h,
            out,
            stride,
            alpha,
        );
        Ok(outcome)
    }

    fn decode_lossless_gray16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if plan.bit_depth != 16 {
            return Err(JpegError::UnsupportedBitDepth {
                depth: plan.bit_depth,
            });
        }
        if !(1..=7).contains(&plan.predictor) {
            return Err(JpegError::UnsupportedPredictor {
                predictor: plan.predictor,
            });
        }

        let (width, height) = plan.dimensions;
        let scan_bytes = &self.bytes[plan.scan_offset..];
        let mut br = BitReader::new(scan_bytes);
        let total_samples = width.saturating_mul(height);
        let mut restart_tracker =
            LosslessRestartTracker::new(self.plan.restart_interval, total_samples);
        for y in 0..height as usize {
            for x in 0..width as usize {
                let sample_index = y as u32 * width + x as u32;
                let restart_first_sample = restart_tracker.begin_unit(&mut br, sample_index)?;
                let predictor = if restart_first_sample {
                    32768
                } else {
                    lossless_predictor_value_u16(plan.predictor, out, stride, x, y)
                };
                let diff = plan.dc_table.decode_fast_dc(&mut br)?;
                let sample = <u16 as LosslessSample>::from_i32(predictor + diff)?;
                let offset = y * stride + x * 2;
                sample.write_le(&mut out[offset..offset + 2]);
                restart_tracker.finish_unit();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_lossless_gray16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        if roi == Rect::full(self.info.dimensions) && downscale == DownscaleFactor::Full {
            return self.decode_lossless_gray16_into(out, stride);
        }

        let (width, height) = self.info.dimensions;
        let full_stride = width as usize * 2;
        let mut full =
            allocate_output_buffer(checked_scratch_len(&[full_stride, height as usize])?);
        let mut outcome = self.decode_lossless_gray16_into(&mut full, full_stride)?;
        let output_rect = scaled_rect_covering(roi, downscale)?;
        copy_gray16_scaled_rect(
            &full,
            (width, height),
            output_rect,
            downscale.denominator(),
            out,
            stride,
        );
        outcome.decoded = roi;
        Ok(outcome)
    }

    fn decode_progressive12_gray16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_progressive12_region_into(out, stride, roi, downscale, Extended12Output::Gray16)
    }

    fn decode_progressive12_rgb16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_progressive12_region_into(out, stride, roi, downscale, Extended12Output::Rgb16)
    }

    fn decode_progressive12_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        output: Extended12Output,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self
            .progressive_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if self.info.sof_kind != SofKind::Progressive12
            || !matches!(
                self.info.color_space,
                ColorSpace::Grayscale
                    | ColorSpace::YCbCr
                    | ColorSpace::Rgb
                    | ColorSpace::Cmyk
                    | ColorSpace::Ycck
            )
        {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }
        if !roi.is_within(self.info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: roi,
                width: self.info.dimensions.0,
                height: self.info.dimensions.1,
            });
        }
        if matches!(output, Extended12Output::Rgb16) {
            match self.info.color_space {
                ColorSpace::Rgb => {
                    let sampling = progressive_color_sampling(plan, self.info.sof_kind)?;
                    return match sampling {
                        Extended12ColorSampling::S444 => self
                            .decode_progressive12_color444_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                Extended12RgbProjection::Identity,
                            ),
                        Extended12ColorSampling::S422 | Extended12ColorSampling::S420 => self
                            .decode_progressive12_color_subsampled_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                sampling,
                                Extended12RgbProjection::Identity,
                            ),
                    };
                }
                ColorSpace::YCbCr => {
                    let sampling = progressive_color_sampling(plan, self.info.sof_kind)?;
                    return match sampling {
                        Extended12ColorSampling::S444 => self
                            .decode_progressive12_color444_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                Extended12RgbProjection::YCbCr,
                            ),
                        Extended12ColorSampling::S422 | Extended12ColorSampling::S420 => self
                            .decode_progressive12_color_subsampled_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                sampling,
                                Extended12RgbProjection::YCbCr,
                            ),
                    };
                }
                ColorSpace::Cmyk | ColorSpace::Ycck => {
                    let sampling = progressive_four_component_sampling(plan, self.info.sof_kind)?;
                    return self.decode_progressive12_four_component_region_into(
                        out, stride, roi, downscale, sampling,
                    );
                }
                ColorSpace::Grayscale => {}
            }
        }
        if self.info.color_space != ColorSpace::Grayscale || plan.components.len() != 1 {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let dct_blocks = decode_progressive_dct_blocks(plan, self.bytes)?;
        let component = &plan.components[0];
        let component_coeffs = &dct_blocks.quantized[0];
        let (width, height) = self.info.dimensions;
        let mut dequant = [0i16; 64];
        let mut pixels = [0u16; 64];
        let write_region = Extended12WriteRegion {
            output_rect,
            dimensions: (width, height),
            downscale,
            output,
        };

        for block_y in 0..component.block_rows as usize {
            for block_x in 0..component.block_cols as usize {
                let block_index = block_y * component.block_cols as usize + block_x;
                dequantize_progressive12_block(
                    &component_coeffs[block_index],
                    &component.quant,
                    &mut dequant,
                );
                if dequant[1..].iter().all(|&coeff| coeff == 0) {
                    pixels.fill(crate::idct::idct_islow_12bit_dc_only_sample(dequant[0]));
                } else {
                    crate::idct::idct_islow_12bit(&dequant, &mut pixels);
                }
                write_extended12_block_region(
                    out,
                    stride,
                    write_region,
                    ((block_x as u32) * 8, (block_y as u32) * 8),
                    &pixels,
                );
            }
        }

        Ok(DecodeOutcome {
            decoded: roi,
            warnings: self.warnings.to_vec(),
        })
    }

    fn decode_progressive12_color444_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        projection: Extended12RgbProjection,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self
            .progressive_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if plan.components.len() != 3
            || plan.sampling.max_h != 1
            || plan.sampling.max_v != 1
            || plan
                .components
                .iter()
                .any(|component| component.h != 1 || component.v != 1 || component.output_index > 2)
        {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let dct_blocks = decode_progressive_dct_blocks(plan, self.bytes)?;
        let (width, height) = self.info.dimensions;
        let component_indices = progressive_color_component_indices(plan)?;
        let block_cols = plan.components[component_indices[0]].block_cols as usize;
        let block_rows = plan.components[component_indices[0]].block_rows as usize;
        let mut dequant = [[0i16; 64]; 3];
        let mut pixels = [[0u16; 64]; 3];
        let write_region = Extended12WriteRegion {
            output_rect,
            dimensions: (width, height),
            downscale,
            output: Extended12Output::Rgb16,
        };

        for block_y in 0..block_rows {
            for block_x in 0..block_cols {
                for output_index in 0..3 {
                    let component_index = component_indices[output_index];
                    let component = &plan.components[component_index];
                    let component_coeffs = &dct_blocks.quantized[component_index];
                    let block_index = block_y * component.block_cols as usize + block_x;
                    dequantize_progressive12_block(
                        &component_coeffs[block_index],
                        &component.quant,
                        &mut dequant[output_index],
                    );
                    if dequant[output_index][1..].iter().all(|&coeff| coeff == 0) {
                        pixels[output_index].fill(crate::idct::idct_islow_12bit_dc_only_sample(
                            dequant[output_index][0],
                        ));
                    } else {
                        crate::idct::idct_islow_12bit(
                            &dequant[output_index],
                            &mut pixels[output_index],
                        );
                    }
                }
                write_extended12_rgb_block_region(
                    out,
                    stride,
                    write_region,
                    projection,
                    ((block_x as u32) * 8, (block_y as u32) * 8),
                    &pixels,
                );
            }
        }

        Ok(DecodeOutcome {
            decoded: roi,
            warnings: self.warnings.to_vec(),
        })
    }

    fn decode_progressive12_color_subsampled_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        sampling: Extended12ColorSampling,
        projection: Extended12RgbProjection,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self
            .progressive_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        debug_assert!(matches!(
            sampling,
            Extended12ColorSampling::S422 | Extended12ColorSampling::S420
        ));
        if progressive_color_sampling(plan, self.info.sof_kind)? != sampling {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let dct_blocks = decode_progressive_dct_blocks(plan, self.bytes)?;
        let planes = render_progressive12_color_planes(plan, &dct_blocks.quantized)?;
        let write_region = Extended12WriteRegion {
            output_rect,
            dimensions: self.info.dimensions,
            downscale,
            output: Extended12Output::Rgb16,
        };
        match sampling {
            Extended12ColorSampling::S444 => unreachable!("4:4:4 path is handled directly"),
            Extended12ColorSampling::S422 => write_extended12_color422_planes_region(
                out,
                stride,
                write_region,
                projection,
                &planes,
            ),
            Extended12ColorSampling::S420 => write_extended12_color420_planes_region(
                out,
                stride,
                write_region,
                projection,
                &planes,
            ),
        }

        Ok(DecodeOutcome {
            decoded: roi,
            warnings: self.warnings.to_vec(),
        })
    }

    fn decode_progressive12_four_component_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        sampling: Extended12ColorSampling,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self
            .progressive_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if progressive_four_component_sampling(plan, self.info.sof_kind)? != sampling {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let dct_blocks = decode_progressive_dct_blocks(plan, self.bytes)?;
        let planes = render_progressive12_four_component_planes(plan, &dct_blocks.quantized)?;
        write_extended12_four_component_planes_region(
            out,
            stride,
            Extended12WriteRegion {
                output_rect,
                dimensions: self.info.dimensions,
                downscale,
                output: Extended12Output::Rgb16,
            },
            self.info.color_space,
            sampling,
            &planes,
        );

        Ok(DecodeOutcome {
            decoded: roi,
            warnings: self.warnings.to_vec(),
        })
    }

    fn decode_extended12_gray16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_extended12_region_into(
            out,
            stride,
            Rect::full(self.info.dimensions),
            DownscaleFactor::Full,
            Extended12Output::Gray16,
        )
    }

    fn decode_extended12_gray16_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_extended12_region_into(
            out,
            stride,
            roi,
            DownscaleFactor::Full,
            Extended12Output::Gray16,
        )
    }

    fn decode_extended12_gray16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_extended12_region_into(out, stride, roi, downscale, Extended12Output::Gray16)
    }

    fn decode_extended12_rgb16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_extended12_region_into(
            out,
            stride,
            Rect::full(self.info.dimensions),
            DownscaleFactor::Full,
            Extended12Output::Rgb16,
        )
    }

    fn decode_extended12_rgb16_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_extended12_region_into(
            out,
            stride,
            roi,
            DownscaleFactor::Full,
            Extended12Output::Rgb16,
        )
    }

    fn decode_extended12_rgb16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_extended12_region_into(out, stride, roi, downscale, Extended12Output::Rgb16)
    }

    fn decode_extended12_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        output: Extended12Output,
    ) -> Result<DecodeOutcome, JpegError> {
        if self.info.sof_kind != SofKind::Extended12 {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }
        if !roi.is_within(self.info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: roi,
                width: self.info.dimensions.0,
                height: self.info.dimensions.1,
            });
        }
        if matches!(output, Extended12Output::Rgb16) {
            match self.info.color_space {
                ColorSpace::Rgb => {
                    let sampling = extended12_color_sampling(&self.plan, self.info.sof_kind)?;
                    return match sampling {
                        Extended12ColorSampling::S444 => self
                            .decode_extended12_color444_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                Extended12RgbProjection::Identity,
                            ),
                        Extended12ColorSampling::S422 | Extended12ColorSampling::S420 => self
                            .decode_extended12_color_subsampled_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                sampling,
                                Extended12RgbProjection::Identity,
                            ),
                    };
                }
                ColorSpace::YCbCr => {
                    let sampling = extended12_color_sampling(&self.plan, self.info.sof_kind)?;
                    return match sampling {
                        Extended12ColorSampling::S444 => self
                            .decode_extended12_color444_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                Extended12RgbProjection::YCbCr,
                            ),
                        Extended12ColorSampling::S422 | Extended12ColorSampling::S420 => self
                            .decode_extended12_color_subsampled_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                sampling,
                                Extended12RgbProjection::YCbCr,
                            ),
                    };
                }
                ColorSpace::Cmyk | ColorSpace::Ycck => {
                    let sampling =
                        extended12_four_component_sampling(&self.plan, self.info.sof_kind)?;
                    return match sampling {
                        Extended12ColorSampling::S444 => self
                            .decode_extended12_four_component444_region_into(
                                out, stride, roi, downscale,
                            ),
                        Extended12ColorSampling::S422 | Extended12ColorSampling::S420 => self
                            .decode_extended12_four_component_subsampled_region_into(
                                out, stride, roi, downscale, sampling,
                            ),
                    };
                }
                ColorSpace::Grayscale => {}
            }
        }
        if self.info.color_space != ColorSpace::Grayscale || self.plan.components.len() != 1 {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let component = &self.plan.components[0];
        let (width, height) = self.info.dimensions;
        let mcu_cols = width.div_ceil(8);
        let mcu_rows = height.div_ceil(8);
        let mut br = BitReader::new(scan_bytes);
        let mut prev_dc = 0i32;
        let mut coeff = CoefficientBlock::default();
        let mut pixels = [0u16; 64];
        let total_mcus = mcu_cols * mcu_rows;
        let mut restart_tracker =
            Extended12RestartTracker::new(self.plan.restart_interval, total_mcus);
        let write_region = Extended12WriteRegion {
            output_rect,
            dimensions: (width, height),
            downscale,
            output,
        };

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                let current_mcu = mcu_y * mcu_cols + mcu_x;
                if restart_tracker.begin_mcu(&mut br, current_mcu)? {
                    prev_dc = 0;
                }
                decode_extended12_block_pixels(
                    &mut br,
                    component,
                    &mut prev_dc,
                    &mut coeff,
                    &mut pixels,
                )?;
                write_extended12_block_region(
                    out,
                    stride,
                    write_region,
                    (mcu_x * 8, mcu_y * 8),
                    &pixels,
                );
                restart_tracker.finish_mcu();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: roi,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_extended12_color444_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        projection: Extended12RgbProjection,
    ) -> Result<DecodeOutcome, JpegError> {
        validate_extended12_color444_plan(&self.plan, self.info.sof_kind)?;

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let (width, height) = self.info.dimensions;
        let mcu_cols = width.div_ceil(8);
        let mcu_rows = height.div_ceil(8);
        let mut br = BitReader::new(scan_bytes);
        let mut prev_dc = [0i32; 3];
        let mut coeffs: [CoefficientBlock; 3] =
            core::array::from_fn(|_| CoefficientBlock::default());
        let mut pixels = [[0u16; 64]; 3];
        let total_mcus = mcu_cols * mcu_rows;
        let mut restart_tracker =
            Extended12RestartTracker::new(self.plan.restart_interval, total_mcus);
        let write_region = Extended12WriteRegion {
            output_rect,
            dimensions: (width, height),
            downscale,
            output: Extended12Output::Rgb16,
        };

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                let current_mcu = mcu_y * mcu_cols + mcu_x;
                if restart_tracker.begin_mcu(&mut br, current_mcu)? {
                    prev_dc.fill(0);
                }
                for component in &self.plan.components {
                    let output_index = component.output_index;
                    decode_extended12_block_pixels(
                        &mut br,
                        component,
                        &mut prev_dc[output_index],
                        &mut coeffs[output_index],
                        &mut pixels[output_index],
                    )?;
                }
                write_extended12_rgb_block_region(
                    out,
                    stride,
                    write_region,
                    projection,
                    (mcu_x * 8, mcu_y * 8),
                    &pixels,
                );
                restart_tracker.finish_mcu();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: roi,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_extended12_color_subsampled_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        sampling: Extended12ColorSampling,
        projection: Extended12RgbProjection,
    ) -> Result<DecodeOutcome, JpegError> {
        debug_assert!(matches!(
            sampling,
            Extended12ColorSampling::S422 | Extended12ColorSampling::S420
        ));
        if extended12_color_sampling(&self.plan, self.info.sof_kind)? != sampling {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let (planes, scan_warnings) =
            decode_extended12_color_planes(&self.plan, scan_bytes, self.info.sof_kind)?;
        let write_region = Extended12WriteRegion {
            output_rect,
            dimensions: self.info.dimensions,
            downscale,
            output: Extended12Output::Rgb16,
        };
        match sampling {
            Extended12ColorSampling::S444 => unreachable!("4:4:4 path is handled directly"),
            Extended12ColorSampling::S422 => write_extended12_color422_planes_region(
                out,
                stride,
                write_region,
                projection,
                &planes,
            ),
            Extended12ColorSampling::S420 => write_extended12_color420_planes_region(
                out,
                stride,
                write_region,
                projection,
                &planes,
            ),
        }

        Ok(DecodeOutcome {
            decoded: roi,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_extended12_four_component444_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        validate_extended12_four_component444_plan(&self.plan, self.info.sof_kind)?;

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let (width, height) = self.info.dimensions;
        let mcu_cols = width.div_ceil(8);
        let mcu_rows = height.div_ceil(8);
        let mut br = BitReader::new(scan_bytes);
        let mut prev_dc = [0i32; 4];
        let mut coeffs: [CoefficientBlock; 4] =
            core::array::from_fn(|_| CoefficientBlock::default());
        let mut pixels = [[0u16; 64]; 4];
        let total_mcus = mcu_cols * mcu_rows;
        let mut restart_tracker =
            Extended12RestartTracker::new(self.plan.restart_interval, total_mcus);
        let write_region = Extended12WriteRegion {
            output_rect,
            dimensions: (width, height),
            downscale,
            output: Extended12Output::Rgb16,
        };

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                let current_mcu = mcu_y * mcu_cols + mcu_x;
                if restart_tracker.begin_mcu(&mut br, current_mcu)? {
                    prev_dc.fill(0);
                }
                for component in &self.plan.components {
                    let output_index = component.output_index;
                    decode_extended12_block_pixels(
                        &mut br,
                        component,
                        &mut prev_dc[output_index],
                        &mut coeffs[output_index],
                        &mut pixels[output_index],
                    )?;
                }
                write_extended12_four_component_block_region(
                    out,
                    stride,
                    write_region,
                    self.info.color_space,
                    (mcu_x * 8, mcu_y * 8),
                    &pixels,
                );
                restart_tracker.finish_mcu();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: roi,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_extended12_four_component_subsampled_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        sampling: Extended12ColorSampling,
    ) -> Result<DecodeOutcome, JpegError> {
        debug_assert!(matches!(
            sampling,
            Extended12ColorSampling::S422 | Extended12ColorSampling::S420
        ));
        if extended12_four_component_sampling(&self.plan, self.info.sof_kind)? != sampling {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let (planes, scan_warnings) =
            decode_extended12_four_component_planes(&self.plan, scan_bytes, self.info.sof_kind)?;
        write_extended12_four_component_planes_region(
            out,
            stride,
            Extended12WriteRegion {
                output_rect,
                dimensions: self.info.dimensions,
                downscale,
                output: Extended12Output::Rgb16,
            },
            self.info.color_space,
            sampling,
            &planes,
        );

        Ok(DecodeOutcome {
            decoded: roi,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }
}

fn build_decode_plan(
    header: &ParsedHeader,
    info: &Info,
    dc_tables: &[Option<Arc<HuffmanTable>>; 4],
    ac_tables: &[Option<Arc<HuffmanTable>>; 4],
    ctx: &mut DecoderContext,
) -> Result<PreparedDecodePlan, JpegError> {
    let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sos,
    })?;
    let scan_offset = header.sos_offset.ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sos,
    })?;

    let mut components = Vec::with_capacity(scan.components.len());
    for scan_comp in scan.components.iter().copied() {
        let output_index = find_component_index(&header.component_ids, scan_comp.id).ok_or(
            JpegError::UnknownScanComponent {
                offset: scan_offset,
                component: scan_comp.id,
            },
        )?;
        let (h, v) = header
            .sampling
            .component(output_index)
            .ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sof,
            })?;
        let quant_id =
            *header
                .quant_table_ids
                .get(output_index)
                .ok_or(JpegError::MissingMarker {
                    marker: MarkerKind::Sof,
                })? as usize;
        let quant = *header
            .quant_tables
            .entries
            .get(quant_id)
            .and_then(|q| q.as_ref())
            .ok_or(JpegError::MissingQuantTable {
                component: scan_comp.id,
                table_id: quant_id as u8,
            })?;
        let dc_table = dc_tables[scan_comp.dc_table as usize].as_ref().ok_or(
            JpegError::MissingHuffmanTable {
                component: scan_comp.id,
                class: 0,
                id: scan_comp.dc_table,
            },
        )?;
        let ac_table = ac_tables[scan_comp.ac_table as usize].as_ref().ok_or(
            JpegError::MissingHuffmanTable {
                component: scan_comp.id,
                class: 1,
                id: scan_comp.ac_table,
            },
        )?;
        components.push(PreparedComponentPlan {
            h,
            v,
            output_index,
            quant: ctx.resolve_quant_table(quant),
            dc_table: Arc::clone(dc_table),
            ac_table: Arc::clone(ac_table),
        });
    }

    let mut scratch_bytes =
        compute_decode_scratch_bytes(info.dimensions, info.sampling, DEFAULT_MAX_DECODE_BYTES)?;
    if info.sof_kind == SofKind::Extended12 {
        // The sequential 12-bit paths render through full-frame u16 component
        // planes, which dwarf the stripe-based estimate above.
        scratch_bytes = scratch_bytes.max(compute_extended12_planes_scratch_bytes(
            &components,
            info.dimensions,
            info.sampling,
            DEFAULT_MAX_DECODE_BYTES,
        )?);
    }

    Ok(PreparedDecodePlan {
        components,
        sampling: info.sampling,
        color_space: info.color_space,
        restart_interval: header.restart_interval,
        dimensions: info.dimensions,
        scan_offset,
        scratch_bytes,
    })
}

fn validate_sampling_factors(header: &ParsedHeader, info: &Info) -> Result<(), JpegError> {
    validate_leading_component_sampling(header, info)?;
    for (h, v) in header.sampling.iter() {
        if h == 0 || v == 0 || h > 4 || v > 4 {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        if !header.sampling.max_h.is_multiple_of(h) || !header.sampling.max_v.is_multiple_of(v) {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
    }
    Ok(())
}

fn validate_leading_component_sampling(
    header: &ParsedHeader,
    info: &Info,
) -> Result<(), JpegError> {
    if !matches!(info.color_space, ColorSpace::YCbCr) {
        return Ok(());
    }
    if let Some((h, v)) = header.sampling.component(0) {
        if h != header.sampling.max_h || v != header.sampling.max_v {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
    }
    Ok(())
}

fn resolve_progressive_huffman(
    ctx: &mut DecoderContext,
    tables: &[Option<RawHuffmanTable>; 4],
    component: u8,
    class: u8,
    id: u8,
) -> Result<Arc<HuffmanTable>, JpegError> {
    let raw = tables
        .get(id as usize)
        .and_then(|table| table.as_ref())
        .ok_or(JpegError::MissingHuffmanTable {
            component,
            class,
            id,
        })?;
    ctx.resolve_huffman_table(raw)
}

fn compute_progressive_scratch_bytes(
    components: &[PreparedProgressiveComponentPlan],
    output_width: usize,
) -> Result<usize, JpegError> {
    let cap = DEFAULT_MAX_DECODE_BYTES;
    let mut total = 0usize;
    for component in components {
        let blocks = checked_usize_product(
            &[component.block_cols as usize, component.block_rows as usize],
            cap,
        )?;
        let coeffs = checked_usize_product(&[blocks, 64, core::mem::size_of::<i32>()], cap)?;
        total = total
            .checked_add(coeffs)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;

        let plane = checked_usize_product(
            &[
                component.block_cols as usize,
                component.block_rows as usize,
                64,
            ],
            cap,
        )?;
        total = total
            .checked_add(plane)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
        if total > cap {
            return Err(JpegError::MemoryCapExceeded {
                requested: total,
                cap,
            });
        }
    }
    total =
        total
            .checked_add(output_width.saturating_mul(3))
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    if total > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: total,
            cap,
        });
    }
    Ok(total)
}

fn find_component_index(component_ids: &[u8], id: u8) -> Option<usize> {
    component_ids
        .iter()
        .position(|&component_id| component_id == id)
}

#[cfg(test)]
include!("decoder_tests.rs");
