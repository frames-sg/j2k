// SPDX-License-Identifier: Apache-2.0

use crate::{
    decode::{
        decode_full_frame, decode_region_not_implemented, decode_scaled_not_implemented,
        inspect_info_via_backend, J2kDecodeOutcome,
    },
    parse::parse_info,
    scratch::J2kScratchPool,
    J2kError,
};
use slidecodec_core::{
    Downscale, ImageCodec, ImageDecode, Info, PixelFormat, Rect,
};
use core::convert::Infallible;

#[derive(Debug)]
pub struct J2kView<'a> {
    bytes: &'a [u8],
    info: Info,
}

impl<'a> J2kView<'a> {
    pub fn parse(input: &'a [u8]) -> Result<Self, J2kError> {
        let info = parse_info(input).or_else(|_| inspect_info_via_backend(input))?;
        Ok(Self { bytes: input, info })
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

#[derive(Debug)]
pub struct J2kDecoder<'a> {
    bytes: &'a [u8],
    info: Info,
}

impl<'a> J2kDecoder<'a> {
    pub fn inspect(input: &'a [u8]) -> Result<Info, J2kError> {
        parse_info(input)
    }

    pub fn new(input: &'a [u8]) -> Result<Self, J2kError> {
        Self::from_view(J2kView::parse(input)?)
    }

    pub fn from_view(view: J2kView<'a>) -> Result<Self, J2kError> {
        Ok(Self {
            bytes: view.bytes,
            info: view.info,
        })
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        self.decode_into_with_scratch(&mut J2kScratchPool::new(), out, stride, fmt)
    }

    pub fn decode_into_with_scratch(
        &mut self,
        _pool: &mut J2kScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        decode_full_frame(self.bytes, out, stride, fmt)
    }

    pub fn decode_region_into(
        &mut self,
        _pool: &mut J2kScratchPool,
        _out: &mut [u8],
        _stride: usize,
        _fmt: PixelFormat,
        _roi: Rect,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        decode_region_not_implemented()
    }

    pub fn decode_scaled_into(
        &mut self,
        _pool: &mut J2kScratchPool,
        _out: &mut [u8],
        _stride: usize,
        _fmt: PixelFormat,
        _scale: Downscale,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        decode_scaled_not_implemented()
    }
}

impl ImageCodec for J2kDecoder<'_> {
    type Error = J2kError;
    type Warning = Infallible;
    type Pool = J2kScratchPool;
}

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
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_into(self, out, stride, fmt)
    }

    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_into_with_scratch(self, pool, out, stride, fmt)
    }

    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_region_into(self, pool, out, stride, fmt, roi)
    }

    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_scaled_into(self, pool, out, stride, fmt, scale)
    }
}
