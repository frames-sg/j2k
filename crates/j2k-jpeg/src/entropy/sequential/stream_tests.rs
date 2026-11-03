// SPDX-License-Identifier: MIT OR Apache-2.0

//! The streaming fast routes (4:2:0, 4:2:2, and 4:4:4 coded as YCbCr or
//! identity RGB) against the stripe-at-a-time generic route, which shares no
//! row windowing or emission code with them, across partial edge MCUs,
//! restart intervals, writer kinds, and regions.

use super::decode_scan_baseline_rgb;
use crate::info::{DownscaleFactor, Rect};
use crate::internal::scratch::ScratchPool;
use crate::output::Rgb8Writer;
use crate::{
    encode_jpeg_baseline, ColorTransform, DecodeOptions, Decoder, JpegBackend, JpegEncodeOptions,
    JpegError, JpegSamples, JpegSubsampling, JpegView, PixelFormat,
};
use alloc::vec;
use alloc::vec::Vec;
use j2k_core::RowSink;

/// Gradients plus per-pixel noise, so neighbouring chroma samples differ and
/// every fancy-upsampling tap matters.
pub(crate) fn noisy_rgb8(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3u32 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let ramp = (x * (7 + channel * 5) + y * (3 + channel * 11)) & 0xFF;
                let noise = state & 0x3F;
                pixels.push(u8::try_from((ramp + noise) & 0xFF).expect("masked to a byte"));
            }
        }
    }
    pixels
}

/// Coded layouts the streaming routes decode.
#[derive(Clone, Copy, Debug)]
enum Shape {
    Ycbcr420,
    Ycbcr422,
    Ycbcr444,
    /// 4:4:4 whose three components are decoded as identity RGB.
    Rgb444,
}

const SHAPES: [Shape; 4] = [
    Shape::Ycbcr420,
    Shape::Ycbcr422,
    Shape::Ycbcr444,
    Shape::Rgb444,
];

fn encode(shape: Shape, width: u32, height: u32, restart_interval: Option<u16>) -> Vec<u8> {
    let pixels = noisy_rgb8(width, height, width * 131 + height);
    encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &pixels,
            width,
            height,
        },
        JpegEncodeOptions {
            quality: 92,
            subsampling: match shape {
                Shape::Ycbcr420 => JpegSubsampling::Ybr420,
                Shape::Ycbcr422 => JpegSubsampling::Ybr422,
                Shape::Ycbcr444 | Shape::Rgb444 => JpegSubsampling::Ybr444,
            },
            restart_interval,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fixture")
    .data
}

/// A decoder of `bytes` that takes the shape's streaming route.
fn decoder(bytes: &[u8], shape: Shape) -> Decoder<'_> {
    let dec = match shape {
        Shape::Ycbcr420 | Shape::Ycbcr422 | Shape::Ycbcr444 => Decoder::new(bytes),
        Shape::Rgb444 => Decoder::from_view(
            JpegView::parse_with_options(
                bytes,
                DecodeOptions::default().with_color_transform(ColorTransform::ForceRgb),
            )
            .expect("fixture parses"),
        ),
    }
    .expect("fixture decoder");
    let streams = match shape {
        Shape::Ycbcr420 => dec.plan.matches_fast_tile_shape(),
        Shape::Ycbcr422 => dec.plan.matches_fast_rgb422_shape(),
        Shape::Ycbcr444 | Shape::Rgb444 => dec.plan.matches_fast_rgb444_shape(),
    };
    assert!(streams, "{shape:?} takes its streaming route");
    dec
}

/// Stripe-at-a-time reference: the generic baseline route.
fn generic_rgb(dec: &Decoder<'_>) -> Vec<u8> {
    let (width, height) = dec.info.dimensions;
    let stride = width as usize * 3;
    let mut out = vec![0u8; stride * height as usize];
    let mut writer = Rgb8Writer::new(&mut out, stride, width);
    decode_scan_baseline_rgb(
        &dec.plan,
        dec.backend,
        &dec.bytes[dec.plan.scan_offset..],
        &mut ScratchPool::new(),
        &mut writer,
        DownscaleFactor::Full,
        Rect::full(dec.info.dimensions),
    )
    .expect("generic route decodes");
    out
}

/// The streaming route writing straight into an RGB8 buffer.
fn streamed_rgb8(dec: &Decoder<'_>) -> Vec<u8> {
    let (width, height) = dec.info.dimensions;
    let stride = width as usize * 3;
    let mut out = vec![0u8; stride * height as usize];
    dec.decode_into(&mut out, stride, PixelFormat::Rgb8)
        .expect("streaming route decodes");
    out
}

struct CollectRows {
    stride: usize,
    out: Vec<u8>,
    next_row: u32,
}

impl RowSink<u8> for CollectRows {
    type Error = JpegError;

    fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), JpegError> {
        assert_eq!(y, self.next_row, "rows reach the sink in order");
        self.next_row += 1;
        let start = y as usize * self.stride;
        self.out[start..start + row.len()].copy_from_slice(row);
        Ok(())
    }
}

/// The streaming route through a row sink.
fn streamed_rows_rgb(dec: &Decoder<'_>) -> Vec<u8> {
    let (width, height) = dec.info.dimensions;
    let stride = width as usize * 3;
    let mut sink = CollectRows {
        stride,
        out: vec![0u8; stride * height as usize],
        next_row: 0,
    };
    dec.decode_rows(&mut sink).expect("row decode");
    assert_eq!(sink.next_row, height);
    sink.out
}

fn crop_rgb(full: &[u8], width: u32, roi: Rect) -> Vec<u8> {
    let stride = width as usize * 3;
    let mut out = Vec::with_capacity(roi.w as usize * roi.h as usize * 3);
    for y in roi.y..roi.y + roi.h {
        let start = y as usize * stride + roi.x as usize * 3;
        out.extend_from_slice(&full[start..start + roi.w as usize * 3]);
    }
    out
}

const WIDTHS: [u32; 12] = [5, 7, 9, 15, 16, 17, 24, 31, 33, 47, 65, 100];
const HEIGHTS: [u32; 11] = [1, 2, 7, 8, 9, 15, 16, 17, 31, 33, 40];

#[test]
fn streamed_full_decodes_match_the_generic_route_at_every_edge_phase() {
    for shape in SHAPES {
        for restart_interval in [None, Some(1), Some(3)] {
            for width in WIDTHS {
                for height in HEIGHTS {
                    let bytes = encode(shape, width, height, restart_interval);
                    let dec = decoder(&bytes, shape);
                    assert_eq!(dec.plan.restart_interval, restart_interval);
                    let expected = generic_rgb(&dec);
                    let case = format!("{shape:?} {width}x{height} restart {restart_interval:?}");
                    assert!(streamed_rgb8(&dec) == expected, "RGB8 stream {case}");
                    assert!(streamed_rows_rgb(&dec) == expected, "row stream {case}");
                }
            }
        }
    }
}

#[test]
fn streamed_rgba_rows_match_the_generic_rgb_route() {
    for shape in SHAPES {
        for (width, height, restart_interval) in [(33, 17, None), (100, 40, Some(2))] {
            let bytes = encode(shape, width, height, restart_interval);
            let dec = decoder(&bytes, shape);
            let expected = generic_rgb(&dec);
            let stride = width as usize * 4;
            let mut rgba = vec![0u8; stride * height as usize];
            dec.decode_into(&mut rgba, stride, PixelFormat::Rgba8)
                .expect("RGBA decode");
            for (rgb, rgba) in expected
                .as_chunks::<3>()
                .0
                .iter()
                .zip(rgba.as_chunks::<4>().0)
            {
                assert_eq!(rgb, &rgba[..3]);
                assert_eq!(rgba[3], u8::MAX);
            }
        }
    }
}

#[test]
fn streamed_regions_at_odd_offsets_match_the_generic_full_decode() {
    for shape in SHAPES {
        for (width, height, restart_interval) in
            [(100, 40, None), (65, 33, Some(2)), (47, 31, Some(5))]
        {
            let bytes = encode(shape, width, height, restart_interval);
            let dec = decoder(&bytes, shape);
            let full = generic_rgb(&dec);
            let xs = [0, 1, 7, 15, 16, 17, 31, width / 2 + 1, width - 3, width - 1];
            let ys = [0, 1, 14, 15, 16, 17, height / 2 + 1, height - 2, height - 1];
            for &x in &xs {
                for &y in &ys {
                    for (w, h) in [(1, 1), (2, 3), (13, 9), (width - x, height - y)] {
                        let roi = Rect {
                            x,
                            y,
                            w: w.min(width - x),
                            h: h.min(height - y),
                        };
                        let stride = roi.w as usize * 3;
                        let mut out = vec![0u8; stride * roi.h as usize];
                        dec.decode_region_into(&mut out, stride, PixelFormat::Rgb8, roi)
                            .expect("region decode");
                        assert!(
                            out == crop_rgb(&full, width, roi),
                            "{shape:?} region {roi:?} of {width}x{height} restart \
                             {restart_interval:?}"
                        );
                    }
                }
            }
        }
    }
}
