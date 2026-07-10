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
use self::extended12::{lossless_color_sampling, upsample_h2v1_sample_at, upsample_h2v2_rows_at};
mod lossless_helpers;
use self::lossless_helpers::{
    decode_lossless_color_sample, decode_lossless_sampled_color_mcu, emit_decode_scan_profile,
    lossless_predictor_gray_rows, lossless_predictor_value, lossless_predictor_value_u16,
    restart_index_for_stream, validate_lossless_color_plan, write_lossless_color16_sampled_output,
    write_lossless_color8_sampled_output, LosslessColorIntoSample, LosslessColorPlanes,
    LosslessColorRowSample, LosslessRestartTracker, LosslessSampledColorPlanesMut,
    LosslessSampledMcu,
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
mod plan;
use self::plan::find_component_index;
mod routing;
mod rows;
mod sequential;
mod tile;
pub(crate) use self::tile::decode_prepared_jpeg_tile_rgb8_in_context;
pub use self::tile::{
    decode_prepared_jpeg_tiles_rgb8, decode_tile_into, decode_tile_into_in_context,
    decode_tile_into_in_context_with_options, decode_tile_region_into_in_context,
    decode_tile_region_into_in_context_with_options, decode_tile_region_scaled_into_in_context,
    decode_tile_region_scaled_into_in_context_with_options, decode_tile_scaled_into_in_context,
    decode_tile_scaled_into_in_context_with_options, decode_tiles_into,
    decode_tiles_into_with_options, decode_tiles_region_scaled_into,
    decode_tiles_region_scaled_into_with_options, decode_tiles_scaled_into,
    decode_tiles_scaled_into_with_options,
};
mod lossless_render;

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
}

#[cfg(test)]
include!("decoder_tests.rs");
