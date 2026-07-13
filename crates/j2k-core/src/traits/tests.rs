// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CodecError, CpuBackedImageDecode, DecodeOutcome, Downscale, ImageCodec, ImageDecode, Info,
    PixelFormat, Rect,
};
use crate::{Colorspace, ScratchPool};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("fake decode error: {0}")]
struct FakeError(&'static str);

impl CodecError for FakeError {
    fn is_truncated(&self) -> bool {
        false
    }

    fn is_not_implemented(&self) -> bool {
        false
    }

    fn is_unsupported(&self) -> bool {
        false
    }

    fn is_buffer_error(&self) -> bool {
        false
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct FakePool {
    observed_stride: usize,
}

impl ScratchPool for FakePool {
    fn bytes_allocated(&self) -> usize {
        self.observed_stride
    }

    fn reset(&mut self) {
        self.observed_stride = 0;
    }
}

#[derive(Debug)]
struct FakeCpuDecoder<'a> {
    input: &'a [u8],
}

impl ImageCodec for FakeCpuDecoder<'_> {
    type Error = FakeError;
    type Warning = &'static str;
    type Pool = FakePool;
}

impl<'a> ImageDecode<'a> for FakeCpuDecoder<'a> {
    type View = &'a [u8];

    fn inspect(input: &'a [u8]) -> Result<Info, Self::Error> {
        Ok(Info {
            dimensions: (
                u32::try_from(input.len()).map_err(|_| FakeError("input length"))?,
                1,
            ),
            components: u16::from(input.first().copied().unwrap_or_default()),
            colorspace: Colorspace::SGray,
            bit_depth: 8,
            tile_layout: None,
            coded_unit_layout: None,
            restart_interval: None,
            resolution_levels: 1,
        })
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(input)
    }

    fn from_view(view: Self::View) -> Result<Self, Self::Error> {
        Ok(Self { input: view })
    }

    fn decode_into(
        &mut self,
        _out: &mut [u8],
        _stride: usize,
        _fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Err(FakeError("unused decode path"))
    }

    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        pool.observed_stride = stride;
        let first = out.first_mut().ok_or(FakeError("empty output"))?;
        *first = self.input.first().copied().unwrap_or_default();
        let second = out.get_mut(1).ok_or(FakeError("short output"))?;
        *second = u8::try_from(fmt.bytes_per_pixel()).map_err(|_| FakeError("pixel size"))?;
        Ok(DecodeOutcome::new(
            Rect {
                x: 1,
                y: 2,
                w: 3,
                h: 4,
            },
            alloc::vec!["cpu warning"],
        ))
    }

    fn decode_region_into(
        &mut self,
        _pool: &mut Self::Pool,
        _out: &mut [u8],
        _stride: usize,
        _fmt: PixelFormat,
        _roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Err(FakeError("unused region path"))
    }

    fn decode_scaled_into(
        &mut self,
        _pool: &mut Self::Pool,
        _out: &mut [u8],
        _stride: usize,
        _fmt: PixelFormat,
        _scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Err(FakeError("unused scaled path"))
    }

    fn decode_region_scaled_into(
        &mut self,
        _pool: &mut Self::Pool,
        _out: &mut [u8],
        _stride: usize,
        _fmt: PixelFormat,
        _roi: Rect,
        _scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Err(FakeError("unused region-scaled path"))
    }
}

#[derive(Debug)]
struct FakeAdapter<'a> {
    cpu: FakeCpuDecoder<'a>,
}

impl ImageCodec for FakeAdapter<'_> {
    type Error = FakeError;
    type Warning = &'static str;
    type Pool = FakePool;
}

impl<'a> CpuBackedImageDecode<'a> for FakeAdapter<'a> {
    type Cpu = FakeCpuDecoder<'a>;
    type View = &'a [u8];

    fn inspect_cpu(input: &'a [u8]) -> Result<Info, Self::Error> {
        FakeCpuDecoder::inspect(input)
    }

    fn parse_cpu(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        FakeCpuDecoder::parse(input)
    }

    fn from_cpu_view(view: Self::View) -> Result<Self, Self::Error> {
        Ok(Self {
            cpu: FakeCpuDecoder::from_view(view)?,
        })
    }

    fn cpu_decoder_mut(&mut self) -> &mut Self::Cpu {
        &mut self.cpu
    }

    fn map_cpu_outcome(
        outcome: DecodeOutcome<<Self::Cpu as ImageCodec>::Warning>,
    ) -> DecodeOutcome<Self::Warning> {
        DecodeOutcome::new(outcome.decoded, alloc::vec!["adapter warning"])
    }
}

#[test]
fn cpu_backed_static_defaults_preserve_input_and_view() {
    let input = [5, 8, 13];

    let info = <FakeAdapter<'_> as ImageDecode<'_>>::inspect(&input).expect("inspect delegates");
    assert_eq!(info.dimensions, (3, 1));
    assert_eq!(info.components, 5);

    let view = <FakeAdapter<'_> as ImageDecode<'_>>::parse(&input).expect("parse delegates");
    assert!(core::ptr::eq(view.as_ptr(), input.as_ptr()));
    let adapter =
        <FakeAdapter<'_> as ImageDecode<'_>>::from_view(view).expect("from_view delegates");
    assert!(core::ptr::eq(adapter.cpu.input.as_ptr(), input.as_ptr()));
}

#[test]
fn cpu_backed_scratch_default_preserves_arguments_and_maps_outcome() {
    let input = [21, 34];
    let mut adapter =
        <FakeAdapter<'_> as ImageDecode<'_>>::from_view(&input).expect("construct fake adapter");
    let mut pool = FakePool::default();
    let mut output = [0; 4];

    let outcome = <FakeAdapter<'_> as ImageDecode<'_>>::decode_into_with_scratch(
        &mut adapter,
        &mut pool,
        &mut output,
        19,
        PixelFormat::Rgba16,
    )
    .expect("scratch decode delegates");

    assert_eq!(pool.observed_stride, 19);
    assert_eq!(output, [21, 8, 0, 0]);
    assert_eq!(
        outcome.decoded,
        Rect {
            x: 1,
            y: 2,
            w: 3,
            h: 4
        }
    );
    assert_eq!(outcome.warnings, ["adapter warning"]);
}
