// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared codec-trait adapters and tile-batch entry points.

use super::{J2kCodec, J2kDecoder, J2kRowDecodeOptions, J2kView};
use crate::{context::J2kContext, decode::J2kDecodeWarning, scratch::J2kScratchPool, J2kError};
use j2k_core::{
    DecodeRowsError, DecoderContext, Downscale, ImageCodec, ImageDecode, ImageDecodeRows, Info,
    PixelFormat, Rect, RowSink, TileBatchDecode, TileRegionScaledDecodeJob,
};

#[doc(hidden)]
impl ImageCodec for J2kDecoder<'_> {
    type Error = J2kError;
    type Warning = J2kDecodeWarning;
    type Pool = J2kScratchPool;
}

#[doc(hidden)]
impl<'a> ImageDecode<'a> for J2kDecoder<'a> {
    type View = J2kView<'a>;

    fn inspect(input: &'a [u8]) -> Result<Info, Self::Error> {
        Self::inspect(input)
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        J2kView::parse(input)
    }

    fn from_view(view: Self::View) -> Result<Self, Self::Error> {
        Self::from_view(view)
    }

    fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_into(self, out, stride, fmt)
    }

    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_into_with_scratch(self, pool, out, stride, fmt)
    }

    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_region_into(self, pool, out, stride, fmt, roi)
    }

    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_scaled_into(self, pool, out, stride, fmt, scale)
    }

    fn decode_region_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_region_scaled_into(self, pool, out, stride, fmt, roi, scale)
    }
}

#[doc(hidden)]
impl<'a> ImageDecodeRows<'a, u8> for J2kDecoder<'a> {
    fn decode_rows<R: RowSink<u8>>(
        &mut self,
        sink: &mut R,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>>
    {
        self.decode_rows_u8_bounded(sink, J2kRowDecodeOptions::default())
    }
}

#[doc(hidden)]
impl<'a> ImageDecodeRows<'a, u16> for J2kDecoder<'a> {
    fn decode_rows<R: RowSink<u16>>(
        &mut self,
        sink: &mut R,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>>
    {
        self.decode_rows_u16_bounded(sink, J2kRowDecodeOptions::default())
    }
}

#[doc(hidden)]
impl ImageCodec for J2kCodec {
    type Error = J2kError;
    type Warning = J2kDecodeWarning;
    type Pool = J2kScratchPool;
}

#[doc(hidden)]
impl TileBatchDecode for J2kCodec {
    type Context = J2kContext;

    fn decode_tile(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        let mut decoder = J2kDecoder::new(input)?;
        decoder.set_cpu_decode_parallelism(ctx.codec().cpu_decode_parallelism());
        decoder.decode_into_with_scratch(pool, out, stride, fmt)
    }

    fn decode_tile_region(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        let mut decoder = J2kDecoder::new(input)?;
        decoder.set_cpu_decode_parallelism(ctx.codec().cpu_decode_parallelism());
        decoder.decode_region_into(pool, out, stride, fmt, roi)
    }

    fn decode_tile_scaled(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        let mut decoder = J2kDecoder::new(input)?;
        decoder.set_cpu_decode_parallelism(ctx.codec().cpu_decode_parallelism());
        decoder.decode_scaled_into(pool, out, stride, fmt, scale)
    }

    fn decode_tile_region_scaled(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        fmt: PixelFormat,
        job: TileRegionScaledDecodeJob<'_, '_>,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        let TileRegionScaledDecodeJob {
            input,
            out,
            stride,
            roi,
            scale,
        } = job;
        let mut decoder = J2kDecoder::new(input)?;
        decoder.set_cpu_decode_parallelism(ctx.codec().cpu_decode_parallelism());
        decoder.decode_region_scaled_into(pool, out, stride, fmt, roi, scale)
    }
}
