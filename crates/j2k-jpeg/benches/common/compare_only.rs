// SPDX-License-Identifier: MIT OR Apache-2.0

use super::classification::DecodeMode;
use super::shared_drivers::{centered_roi, scaled_rect};
use j2k_jpeg::{
    decode_tiles_into, Decoder, Downscale, JpegBatchSession, JpegOutputBuffer, PixelFormat, Rect,
    ScratchPool, TileBatchOptions, TileDecodeJob, TileRegionScaledDecodeJob, TileScaledDecodeJob,
};

pub(crate) fn output_geometry(dec: &Decoder<'_>, mode: DecodeMode) -> (PixelFormat, usize, usize) {
    let (width, height) = dec.info().dimensions;
    match mode {
        DecodeMode::Gray => {
            let len = (width as usize) * (height as usize);
            (PixelFormat::Gray8, width as usize, len)
        }
        DecodeMode::Rgb => {
            let stride = (width as usize) * 3;
            let len = stride * (height as usize);
            (PixelFormat::Rgb8, stride, len)
        }
    }
}

pub(crate) fn j2k_decode_reused(
    dec: &Decoder<'_>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) {
    dec.decode_into(out, stride, fmt)
        .expect("j2k decode (reused)");
    std::hint::black_box(&*out);
}

pub(crate) fn j2k_decode_with_scratch(
    dec: &Decoder<'_>,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) {
    dec.decode_into_with_scratch(pool, out, stride, fmt)
        .expect("j2k decode (scratch)");
    std::hint::black_box(&*out);
}

pub(crate) struct J2KTileBatchRgbScratch {
    outputs: Vec<Vec<u8>>,
    stride: usize,
    options: TileBatchOptions,
}

impl J2KTileBatchRgbScratch {
    pub(crate) fn new(bytes: &[u8], batch_size: usize) -> Self {
        Self::new_with_options(bytes, batch_size, TileBatchOptions::default())
    }

    pub(crate) fn new_with_options(
        bytes: &[u8],
        batch_size: usize,
        options: TileBatchOptions,
    ) -> Self {
        let info = Decoder::inspect(bytes).expect("j2k inspect tile batch");
        let stride = info.dimensions.0 as usize * 3;
        let len = stride * info.dimensions.1 as usize;
        Self {
            outputs: (0..batch_size).map(|_| vec![0u8; len]).collect(),
            stride,
            options,
        }
    }

    pub(crate) fn run(&mut self, bytes: &[u8]) {
        let outcomes = {
            let mut jobs = self
                .outputs
                .iter_mut()
                .map(|out| TileDecodeJob {
                    input: bytes,
                    out: out.as_mut_slice(),
                    stride: self.stride,
                })
                .collect::<Vec<_>>();
            decode_tiles_into(&mut jobs, PixelFormat::Rgb8, self.options)
                .expect("j2k production tile batch")
        };
        std::hint::black_box(&outcomes);
        std::hint::black_box(&self.outputs);
    }
}

pub(crate) struct J2KTileBatchRgbSession {
    outputs: Vec<Vec<u8>>,
    stride: usize,
    session: JpegBatchSession,
}

impl J2KTileBatchRgbSession {
    pub(crate) fn new(bytes: &[u8], batch_size: usize) -> Self {
        let info = Decoder::inspect(bytes).expect("j2k inspect tile batch");
        let stride = info.dimensions.0 as usize * 3;
        let len = stride * info.dimensions.1 as usize;
        let mut this = Self {
            outputs: (0..batch_size).map(|_| vec![0u8; len]).collect(),
            stride,
            session: JpegBatchSession::default(),
        };
        this.run(bytes);
        this
    }

    pub(crate) fn run(&mut self, bytes: &[u8]) {
        let outcomes = {
            let mut jobs = self
                .outputs
                .iter_mut()
                .map(|out| TileDecodeJob {
                    input: bytes,
                    out: out.as_mut_slice(),
                    stride: self.stride,
                })
                .collect::<Vec<_>>();
            self.session
                .decode_tiles_into(&mut jobs, PixelFormat::Rgb8)
                .expect("j2k session tile batch")
        };
        std::hint::black_box(outcomes);
        std::hint::black_box(&self.outputs);
    }
}

pub(crate) struct J2KTileBatchRgbOutputBuffers {
    outputs: Vec<JpegOutputBuffer>,
    session: JpegBatchSession,
}

impl J2KTileBatchRgbOutputBuffers {
    pub(crate) fn new(bytes: &[u8], batch_size: usize) -> Self {
        let info = Decoder::inspect(bytes).expect("j2k inspect tile batch");
        let mut this = Self {
            outputs: (0..batch_size)
                .map(|_| {
                    JpegOutputBuffer::new(info.dimensions, PixelFormat::Rgb8)
                        .expect("JPEG output buffer")
                })
                .collect(),
            session: JpegBatchSession::default(),
        };
        this.run(bytes);
        this
    }

    pub(crate) fn run(&mut self, bytes: &[u8]) {
        let outcomes = {
            let mut jobs = self
                .outputs
                .iter_mut()
                .map(|out| {
                    let stride = out.stride();
                    TileDecodeJob {
                        input: bytes,
                        out: out.as_mut_slice(),
                        stride,
                    }
                })
                .collect::<Vec<_>>();
            self.session
                .decode_tiles_into(&mut jobs, PixelFormat::Rgb8)
                .expect("j2k session tile batch with output buffers")
        };
        std::hint::black_box(outcomes);
        std::hint::black_box(&self.outputs);
    }
}

pub(crate) struct J2KTileBatchScaledRgbSession {
    outputs: Vec<JpegOutputBuffer>,
    scale: Downscale,
    session: JpegBatchSession,
}

impl J2KTileBatchScaledRgbSession {
    pub(crate) fn new(bytes: &[u8], batch_size: usize, scale: Downscale) -> Self {
        let info = Decoder::inspect(bytes).expect("j2k inspect scaled tile batch");
        let dims = scaled_dims(info.dimensions, scale);
        let mut this = Self {
            outputs: (0..batch_size)
                .map(|_| JpegOutputBuffer::new(dims, PixelFormat::Rgb8).expect("output buffer"))
                .collect(),
            scale,
            session: JpegBatchSession::default(),
        };
        this.run(bytes);
        this
    }

    pub(crate) fn run(&mut self, bytes: &[u8]) {
        let outcomes = {
            let mut jobs = self
                .outputs
                .iter_mut()
                .map(|out| {
                    let stride = out.stride();
                    TileScaledDecodeJob {
                        input: bytes,
                        out: out.as_mut_slice(),
                        stride,
                        scale: self.scale,
                    }
                })
                .collect::<Vec<_>>();
            self.session
                .decode_tiles_scaled_into(&mut jobs, PixelFormat::Rgb8)
                .expect("j2k session scaled tile batch")
        };
        std::hint::black_box(outcomes);
        std::hint::black_box(&self.outputs);
    }
}

pub(crate) struct J2KTileBatchRegionScaledRgbSession {
    outputs: Vec<JpegOutputBuffer>,
    roi: Rect,
    scale: Downscale,
    session: JpegBatchSession,
}

impl J2KTileBatchRegionScaledRgbSession {
    pub(crate) fn new(bytes: &[u8], batch_size: usize, side: u32, scale: Downscale) -> Self {
        let info = Decoder::inspect(bytes).expect("j2k inspect region-scaled tile batch");
        let roi = centered_roi(info.dimensions, side);
        let scaled = scaled_rect(roi, scale);
        let dims = (scaled.w, scaled.h);
        let mut this = Self {
            outputs: (0..batch_size)
                .map(|_| JpegOutputBuffer::new(dims, PixelFormat::Rgb8).expect("output buffer"))
                .collect(),
            roi,
            scale,
            session: JpegBatchSession::default(),
        };
        this.run(bytes);
        this
    }

    pub(crate) fn run(&mut self, bytes: &[u8]) {
        let outcomes = {
            let mut jobs = self
                .outputs
                .iter_mut()
                .map(|out| {
                    let stride = out.stride();
                    TileRegionScaledDecodeJob {
                        input: bytes,
                        out: out.as_mut_slice(),
                        stride,
                        roi: self.roi.into(),
                        scale: self.scale,
                    }
                })
                .collect::<Vec<_>>();
            self.session
                .decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8)
                .expect("j2k session region-scaled tile batch")
        };
        std::hint::black_box(outcomes);
        std::hint::black_box(&self.outputs);
    }
}

fn scaled_dims((width, height): (u32, u32), factor: Downscale) -> (u32, u32) {
    let denom = factor.denominator();
    (width.div_ceil(denom), height.div_ceil(denom))
}
