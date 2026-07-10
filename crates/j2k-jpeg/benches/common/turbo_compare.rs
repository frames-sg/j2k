// SPDX-License-Identifier: MIT OR Apache-2.0

use super::classification::DecodeMode;
use super::libjpeg_turbo::TurboJpegDecoder;
use j2k_jpeg::{Downscale, Rect};

pub(crate) fn libjpeg_turbo_inspect(decoder: &mut TurboJpegDecoder, bytes: &[u8]) {
    let info = decoder.inspect(bytes).expect("libjpeg-turbo inspect");
    std::hint::black_box(info);
}

pub(crate) fn libjpeg_turbo_decode(decoder: &mut TurboJpegDecoder, bytes: &[u8], mode: DecodeMode) {
    let out = match mode {
        DecodeMode::Gray => decoder.decode_gray(bytes),
        DecodeMode::Rgb => decoder.decode_rgb(bytes),
    }
    .expect("libjpeg-turbo decode");
    std::hint::black_box(out);
}

pub(crate) fn libjpeg_turbo_decode_region(decoder: &mut TurboJpegDecoder, bytes: &[u8], roi: Rect) {
    let out = decoder
        .decode_region_rgb(bytes, roi)
        .expect("libjpeg-turbo region decode");
    std::hint::black_box(out);
}

pub(crate) fn libjpeg_turbo_decode_scaled(
    decoder: &mut TurboJpegDecoder,
    bytes: &[u8],
    factor: Downscale,
) {
    let out = decoder
        .decode_scaled_rgb(bytes, factor)
        .expect("libjpeg-turbo scaled decode");
    std::hint::black_box(out);
}

pub(crate) fn libjpeg_turbo_decode_region_scaled(
    decoder: &mut TurboJpegDecoder,
    bytes: &[u8],
    roi: Rect,
    factor: Downscale,
) {
    let out = decoder
        .decode_region_scaled_rgb(bytes, roi, factor)
        .expect("libjpeg-turbo scaled region decode");
    std::hint::black_box(out);
}

pub(crate) fn libjpeg_turbo_decode_batch_scaled(
    decoder: &mut TurboJpegDecoder,
    bytes: &[u8],
    batch_size: usize,
    factor: Downscale,
) {
    for _ in 0..batch_size {
        let out = decoder
            .decode_scaled_rgb(bytes, factor)
            .expect("libjpeg-turbo scaled decode");
        std::hint::black_box(out);
    }
}

pub(crate) fn libjpeg_turbo_decode_batch_region_scaled(
    decoder: &mut TurboJpegDecoder,
    bytes: &[u8],
    batch_size: usize,
    roi: Rect,
    factor: Downscale,
) {
    for _ in 0..batch_size {
        let out = decoder
            .decode_region_scaled_rgb(bytes, roi, factor)
            .expect("libjpeg-turbo scaled region decode");
        std::hint::black_box(out);
    }
}

pub(crate) struct TurboJpegBatchRgbOutputBuffers {
    decoder: TurboJpegDecoder,
    outputs: Vec<Vec<u8>>,
    stride: usize,
    dimensions: (usize, usize),
}

impl TurboJpegBatchRgbOutputBuffers {
    pub(crate) fn new(bytes: &[u8], batch_size: usize) -> Self {
        let mut decoder = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
        let info = decoder.prepare_rgb(bytes).expect("libjpeg-turbo prepare");
        let stride = info.width as usize * 3;
        let len = stride * info.height as usize;
        let mut this = Self {
            decoder,
            outputs: (0..batch_size).map(|_| vec![0u8; len]).collect(),
            stride,
            dimensions: (info.width as usize, info.height as usize),
        };
        this.run(bytes);
        this
    }

    pub(crate) fn run(&mut self, bytes: &[u8]) {
        for out in &mut self.outputs {
            self.decoder
                .decode_prepared_rgb_into(
                    bytes,
                    out,
                    self.stride,
                    self.dimensions.0,
                    self.dimensions.1,
                )
                .expect("libjpeg-turbo preallocated decode");
        }
        std::hint::black_box(&self.outputs);
    }
}
