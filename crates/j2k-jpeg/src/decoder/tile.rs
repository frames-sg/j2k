// SPDX-License-Identifier: MIT OR Apache-2.0

//! One-shot tile and batch facade functions.

use super::{
    DecodeOptions, DecodeOutcome, DecodedTile, Decoder, DecoderContext, Downscale, JpegError,
    JpegView, PixelFormat, PreparedJpeg, PreparedJpegTileJob, Rect, ScratchPool, TileBatchError,
    TileBatchOptions, TileDecodeJob, TileDecodeOutput, TileRegionScaledDecodeJob,
    TileScaledDecodeJob, Vec, DEFAULT_CONTEXT,
};

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
