// SPDX-License-Identifier: MIT OR Apache-2.0

use super::classification::DecodeMode;
use super::null_sink::NullSink;
use j2k_jpeg::{
    decode_tiles_region_scaled_into, decode_tiles_scaled_into, DecodeRequest, Decoder, Downscale,
    PixelFormat, Rect, TileBatchOptions, TileRegionScaledDecodeJob, TileScaledDecodeJob,
};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace as ZuneColorSpace;
use zune_core::options::DecoderOptions;

const ZUNE_DIMENSION_LIMIT: usize = 1 << 20;

pub(crate) fn j2k_inspect(bytes: &[u8]) {
    let info = Decoder::inspect(bytes).expect("j2k inspect");
    std::hint::black_box(info);
}

pub(crate) fn jpeg_decoder_inspect(bytes: &[u8]) {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    decoder.read_info().expect("jpeg-decoder read_info");
    std::hint::black_box(decoder.info());
}

pub(crate) fn zune_inspect(bytes: &[u8]) {
    let options = DecoderOptions::new_fast()
        .set_max_width(ZUNE_DIMENSION_LIMIT)
        .set_max_height(ZUNE_DIMENSION_LIMIT);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder.decode_headers().expect("zune-jpeg decode_headers");
    std::hint::black_box(decoder.info());
}

pub(crate) fn j2k_decode(bytes: &[u8], mode: DecodeMode) {
    let dec = Decoder::new(bytes).expect("j2k decoder");
    let fmt = match mode {
        DecodeMode::Gray => PixelFormat::Gray8,
        DecodeMode::Rgb => PixelFormat::Rgb8,
    };
    let (out, _) = dec
        .decode_request(DecodeRequest::full(fmt))
        .expect("j2k decode");
    std::hint::black_box(out);
}

pub(crate) fn j2k_decode_rows(bytes: &[u8]) {
    let dec = Decoder::new(bytes).expect("j2k decoder");
    let mut sink = NullSink;
    dec.decode_rows(&mut sink).expect("j2k decode_rows");
}

pub(crate) fn j2k_decode_tile_batch_scaled(bytes: &[u8], batch_size: usize, factor: Downscale) {
    let info = Decoder::inspect(bytes).expect("j2k inspect");
    let out_width = info.dimensions.0.div_ceil(factor.denominator());
    let out_height = info.dimensions.1.div_ceil(factor.denominator());
    let stride = out_width as usize * 3;
    let len = stride * out_height as usize;
    let mut outputs = (0..batch_size).map(|_| vec![0u8; len]).collect::<Vec<_>>();
    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileScaledDecodeJob {
                input: bytes,
                out: out.as_mut_slice(),
                stride,
                scale: factor,
            })
            .collect::<Vec<_>>();
        decode_tiles_scaled_into(&mut jobs, PixelFormat::Rgb8, TileBatchOptions::default())
            .expect("j2k production scaled tile batch")
    };
    std::hint::black_box(outcomes);
    std::hint::black_box(outputs);
}

pub(crate) fn j2k_decode_tile_batch_region_scaled(
    bytes: &[u8],
    batch_size: usize,
    side: u32,
    factor: Downscale,
) {
    let info = Decoder::inspect(bytes).expect("j2k inspect");
    let roi = centered_roi(info.dimensions, side);
    let scaled = scaled_rect(roi, factor);
    let stride = scaled.w as usize * 3;
    let len = stride * scaled.h as usize;
    let mut outputs = (0..batch_size).map(|_| vec![0u8; len]).collect::<Vec<_>>();
    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileRegionScaledDecodeJob {
                input: bytes,
                out: out.as_mut_slice(),
                stride,
                roi: roi.into(),
                scale: factor,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8, TileBatchOptions::default())
            .expect("j2k production region-scaled tile batch")
    };
    std::hint::black_box(outcomes);
    std::hint::black_box(outputs);
}

pub(crate) fn j2k_decode_region(bytes: &[u8], side: u32) {
    let dec = Decoder::new(bytes).expect("j2k decoder");
    let roi = centered_roi(dec.info().dimensions, side);
    let (out, _) = dec
        .decode_request(DecodeRequest::region(PixelFormat::Rgb8, roi))
        .expect("j2k region decode");
    std::hint::black_box(out);
}

pub(crate) fn j2k_decode_scaled(bytes: &[u8], factor: Downscale) {
    let dec = Decoder::new(bytes).expect("j2k decoder");
    let (out, _) = dec
        .decode_request(DecodeRequest::scaled(PixelFormat::Rgb8, factor))
        .expect("j2k scaled decode");
    std::hint::black_box(out);
}

pub(crate) fn j2k_decode_region_scaled(bytes: &[u8], side: u32, factor: Downscale) {
    let dec = Decoder::new(bytes).expect("j2k decoder");
    let roi = centered_roi(dec.info().dimensions, side);
    let (out, _) = dec
        .decode_request(DecodeRequest::region_scaled(PixelFormat::Rgb8, roi, factor))
        .expect("j2k scaled region decode");
    std::hint::black_box(out);
}

pub(crate) fn jpeg_decoder_decode(bytes: &[u8]) {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let out = decoder.decode().expect("jpeg-decoder decode");
    std::hint::black_box(out);
}

pub(crate) fn jpeg_decoder_decode_region(bytes: &[u8], side: u32) {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let out = decoder.decode().expect("jpeg-decoder decode");
    let info = decoder.info().expect("jpeg-decoder info");
    let roi = centered_roi((info.width.into(), info.height.into()), side);
    let cropped = crop_rgb(&out, info.width as usize, roi);
    std::hint::black_box(cropped);
}

pub(crate) fn jpeg_decoder_decode_scaled(bytes: &[u8], factor: Downscale) {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let out = decoder.decode().expect("jpeg-decoder decode");
    let info = decoder.info().expect("jpeg-decoder info");
    let scaled = decimate_rgb(
        &out,
        info.width as usize,
        info.height as usize,
        factor.denominator() as usize,
    );
    std::hint::black_box(scaled);
}

pub(crate) fn jpeg_decoder_decode_region_scaled(bytes: &[u8], side: u32, factor: Downscale) {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let out = decoder.decode().expect("jpeg-decoder decode");
    let info = decoder.info().expect("jpeg-decoder info");
    let roi = centered_roi((info.width.into(), info.height.into()), side);
    let cropped = crop_rgb(&out, info.width as usize, roi);
    let scaled = decimate_rgb(
        &cropped,
        roi.w as usize,
        roi.h as usize,
        factor.denominator() as usize,
    );
    std::hint::black_box(scaled);
}

pub(crate) fn zune_decode(bytes: &[u8], mode: DecodeMode) {
    let colorspace = match mode {
        DecodeMode::Gray => ZuneColorSpace::Luma,
        DecodeMode::Rgb => ZuneColorSpace::RGB,
    };
    let options = DecoderOptions::new_fast()
        .set_max_width(ZUNE_DIMENSION_LIMIT)
        .set_max_height(ZUNE_DIMENSION_LIMIT)
        .jpeg_set_out_colorspace(colorspace);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    let out = decoder.decode().expect("zune-jpeg decode");
    std::hint::black_box(out);
}

pub(crate) fn zune_decode_region(bytes: &[u8], side: u32) {
    let mut decoder = zune_rgb_decoder(bytes);
    let out = decoder.decode().expect("zune-jpeg decode");
    let info = decoder.info().expect("zune-jpeg info");
    let roi = centered_roi((u32::from(info.width), u32::from(info.height)), side);
    let cropped = crop_rgb(&out, info.width.into(), roi);
    std::hint::black_box(cropped);
}

pub(crate) fn zune_decode_scaled(bytes: &[u8], factor: Downscale) {
    let mut decoder = zune_rgb_decoder(bytes);
    let out = decoder.decode().expect("zune-jpeg decode");
    let info = decoder.info().expect("zune-jpeg info");
    let scaled = decimate_rgb(
        &out,
        info.width.into(),
        info.height.into(),
        factor.denominator() as usize,
    );
    std::hint::black_box(scaled);
}

pub(crate) fn zune_decode_region_scaled(bytes: &[u8], side: u32, factor: Downscale) {
    let mut decoder = zune_rgb_decoder(bytes);
    let out = decoder.decode().expect("zune-jpeg decode");
    let info = decoder.info().expect("zune-jpeg info");
    let roi = centered_roi((u32::from(info.width), u32::from(info.height)), side);
    let cropped = crop_rgb(&out, info.width.into(), roi);
    let scaled = decimate_rgb(
        &cropped,
        roi.w as usize,
        roi.h as usize,
        factor.denominator() as usize,
    );
    std::hint::black_box(scaled);
}

pub(crate) fn jpeg_decoder_decode_batch_scaled(bytes: &[u8], batch_size: usize, factor: Downscale) {
    for _ in 0..batch_size {
        jpeg_decoder_decode_scaled(bytes, factor);
    }
}

pub(crate) fn jpeg_decoder_decode_batch_region_scaled(
    bytes: &[u8],
    batch_size: usize,
    side: u32,
    factor: Downscale,
) {
    for _ in 0..batch_size {
        jpeg_decoder_decode_region_scaled(bytes, side, factor);
    }
}

pub(crate) fn zune_decode_batch_scaled(bytes: &[u8], batch_size: usize, factor: Downscale) {
    for _ in 0..batch_size {
        zune_decode_scaled(bytes, factor);
    }
}

pub(crate) fn zune_decode_batch_region_scaled(
    bytes: &[u8],
    batch_size: usize,
    side: u32,
    factor: Downscale,
) {
    for _ in 0..batch_size {
        zune_decode_region_scaled(bytes, side, factor);
    }
}

pub(crate) fn centered_roi((width, height): (u32, u32), side: u32) -> Rect {
    let w = side.min(width);
    let h = side.min(height);
    Rect {
        x: (width - w) / 2,
        y: (height - h) / 2,
        w,
        h,
    }
}

pub(crate) fn scaled_rect(rect: Rect, factor: Downscale) -> Rect {
    let denom = factor.denominator();
    let x_end = rect.x + rect.w;
    let y_end = rect.y + rect.h;
    Rect {
        x: rect.x / denom,
        y: rect.y / denom,
        w: x_end.div_ceil(denom) - rect.x / denom,
        h: y_end.div_ceil(denom) - rect.y / denom,
    }
}

fn zune_rgb_decoder(bytes: &[u8]) -> zune_jpeg::JpegDecoder<ZCursor<&[u8]>> {
    zune_jpeg::JpegDecoder::new_with_options(
        ZCursor::new(bytes),
        DecoderOptions::new_fast()
            .set_max_width(ZUNE_DIMENSION_LIMIT)
            .set_max_height(ZUNE_DIMENSION_LIMIT)
            .jpeg_set_out_colorspace(ZuneColorSpace::RGB),
    )
}

fn crop_rgb(full: &[u8], width: usize, roi: Rect) -> Vec<u8> {
    let stride = width * 3;
    let mut out = vec![0u8; roi.w as usize * roi.h as usize * 3];
    for row in 0..roi.h as usize {
        let src_start = (roi.y as usize + row) * stride + roi.x as usize * 3;
        let src_end = src_start + roi.w as usize * 3;
        let dst_start = row * roi.w as usize * 3;
        out[dst_start..dst_start + roi.w as usize * 3].copy_from_slice(&full[src_start..src_end]);
    }
    out
}

fn decimate_rgb(full: &[u8], width: usize, height: usize, denom: usize) -> Vec<u8> {
    let out_width = width.div_ceil(denom);
    let out_height = height.div_ceil(denom);
    let mut out = vec![0u8; out_width * out_height * 3];
    for y in 0..out_height {
        let src_y = (y * denom).min(height.saturating_sub(1));
        for x in 0..out_width {
            let src_x = (x * denom).min(width.saturating_sub(1));
            let src = (src_y * width + src_x) * 3;
            let dst = (y * out_width + x) * 3;
            out[dst..dst + 3].copy_from_slice(&full[src..src + 3]);
        }
    }
    out
}
