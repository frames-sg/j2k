// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code)]

pub(crate) mod classification;
mod libjpeg_turbo;
pub(crate) mod report;

pub(crate) use self::classification::DecodeMode;
use self::classification::{classify_corpus_input, color_space_mode, CorpusInputClass};
pub(crate) use self::libjpeg_turbo::TurboJpegDecoder;
use j2k_core::tile_batch_worker_count;
use j2k_jpeg::{
    decode_tiles_into, decode_tiles_region_scaled_into, decode_tiles_scaled_into, Decoder,
    DecoderContext, Downscale, JpegBatchSession, JpegError, JpegOutputBuffer, PixelFormat, Rect,
    RowSink, ScratchPool, TileBatchOptions, TileDecodeJob, TileRegionScaledDecodeJob,
    TileScaledDecodeJob,
};
use j2k_test_support::{JPEG_BASELINE_420_16X16, JPEG_GRAYSCALE_8X8};
use std::path::{Path, PathBuf};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace as ZuneColorSpace;
use zune_core::options::DecoderOptions;

const ZUNE_DIMENSION_LIMIT: usize = 1 << 20;

#[derive(Clone)]
pub(crate) struct BenchInput {
    pub(crate) name: String,
    pub(crate) bytes: Vec<u8>,
    pub(crate) dimensions: (u32, u32),
    pub(crate) mode: DecodeMode,
    pub(crate) input_class: CorpusInputClass,
}

pub(crate) fn load_bench_inputs() -> Vec<BenchInput> {
    let mut inputs = vec![
        BenchInput {
            name: "repo/baseline_420_16x16".to_string(),
            bytes: JPEG_BASELINE_420_16X16.to_vec(),
            dimensions: (16, 16),
            mode: DecodeMode::Rgb,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
        BenchInput {
            name: "repo/grayscale_8x8".to_string(),
            bytes: JPEG_GRAYSCALE_8X8.to_vec(),
            dimensions: (8, 8),
            mode: DecodeMode::Gray,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
    ];

    let mut seen = inputs
        .iter()
        .map(|input| input.name.clone())
        .collect::<Vec<_>>();
    for path in j2k_test_support::paths_from_env("J2K_BENCH_INPUTS") {
        collect_jpegs(&path, &mut inputs, &mut seen);
    }

    inputs.sort_by(|a, b| a.name.cmp(&b.name));
    inputs
}

fn collect_jpegs(path: &Path, inputs: &mut Vec<BenchInput>, seen: &mut Vec<String>) {
    for path in j2k_test_support::collect_jpeg_paths(path) {
        push_jpeg(&path, inputs, seen);
    }
}

fn push_jpeg(path: &Path, inputs: &mut Vec<BenchInput>, seen: &mut Vec<String>) {
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let Ok(dec) = Decoder::new(&bytes) else {
        return;
    };
    let Some(mode) = color_space_mode(dec.info().color_space) else {
        return;
    };
    let dimensions = dec.info().dimensions;
    let input_class = classify_corpus_input(dimensions, mode);

    let name = relative_name(path);
    if seen.contains(&name) {
        return;
    }
    seen.push(name.clone());
    inputs.push(BenchInput {
        name,
        bytes,
        dimensions,
        mode,
        input_class,
    });
}

fn relative_name(path: &Path) -> String {
    let absolute = path.canonicalize().unwrap_or_else(|_| PathBuf::from(path));
    if let Some(prefix) = std::env::var_os("HOME") {
        let prefix = PathBuf::from(prefix);
        if let Ok(stripped) = absolute.strip_prefix(prefix) {
            return stripped.display().to_string();
        }
    }
    absolute.display().to_string()
}

pub(crate) fn j2k_inspect(bytes: &[u8]) {
    let info = Decoder::inspect(bytes).expect("j2k inspect");
    std::hint::black_box(info);
}

pub(crate) fn libjpeg_turbo_available() -> bool {
    libjpeg_turbo::is_available()
}

pub(crate) fn libjpeg_turbo_inspect(decoder: &mut TurboJpegDecoder, bytes: &[u8]) {
    let info = decoder.inspect(bytes).expect("libjpeg-turbo inspect");
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
    let (out, _) = dec.decode(fmt).expect("j2k decode");
    std::hint::black_box(out);
}

pub(crate) fn libjpeg_turbo_decode(decoder: &mut TurboJpegDecoder, bytes: &[u8], mode: DecodeMode) {
    let out = match mode {
        DecodeMode::Gray => decoder.decode_gray(bytes),
        DecodeMode::Rgb => decoder.decode_rgb(bytes),
    }
    .expect("libjpeg-turbo decode");
    std::hint::black_box(out);
}

/// Output dimensions (`stride`, `total length`, `PixelFormat`) for `mode`.
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

/// Reused-decoder driver: a pre-built `Decoder` decodes into a pre-allocated
/// buffer. Isolates pure decode cost from `Decoder::new` + output allocation —
/// the realistic WSI tile-batch shape. Competitor crates are not called from
/// this helper because neither `zune-jpeg` nor `jpeg-decoder` expose a reusable
/// decoder; fairness is preserved by keeping this group j2k-only.
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

/// Scratch-reuse driver: reuses both the pre-built `Decoder` and a
/// pre-allocated `ScratchPool`. The pool amortizes stripe-buffer and
/// DC-predictor allocations across many tiles — the shape every Phase 3
/// WSI benchmark is trying to surface.
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

#[derive(Default)]
struct NullSink;

impl RowSink<u8> for NullSink {
    type Error = JpegError;

    fn write_row(&mut self, _y: u32, _row: &[u8]) -> Result<(), JpegError> {
        Ok(())
    }
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
        let dims = {
            let scaled = scaled_rect(roi, scale);
            (scaled.w, scaled.h)
        };
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

pub(crate) fn j2k_decode_rows(bytes: &[u8]) {
    let dec = Decoder::new(bytes).expect("j2k decoder");
    let mut sink = NullSink;
    dec.decode_rows(&mut sink).expect("j2k decode_rows");
}

pub(crate) fn j2k_decode_tile_batch(bytes: &[u8], batch_size: usize) {
    let worker_count = j2k_tile_batch_worker_count(batch_size);
    if worker_count == 1 {
        j2k_decode_tile_batch_sequential(bytes, batch_size);
        return;
    }

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let base_tiles = batch_size / worker_count;
        let extra_tiles = batch_size % worker_count;
        for worker in 0..worker_count {
            let tile_count = base_tiles + usize::from(worker < extra_tiles);
            handles.push(scope.spawn(move || j2k_decode_tile_batch_worker(bytes, tile_count)));
        }
        for handle in handles {
            handle
                .join()
                .expect("j2k decode_tile worker panicked")
                .expect("j2k decode_tile batch");
        }
    });
}

pub(crate) fn j2k_decode_tile_batch_sequential(bytes: &[u8], batch_size: usize) {
    j2k_decode_tile_batch_worker(bytes, batch_size).expect("j2k decode_tile batch");
}

fn j2k_tile_batch_worker_count(batch_size: usize) -> usize {
    let configured = std::env::var("J2K_JPEG_BATCH_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .and_then(std::num::NonZeroUsize::new);
    tile_batch_worker_count(
        batch_size,
        TileBatchOptions {
            workers: configured,
        },
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get),
    )
}

fn j2k_decode_tile_batch_worker(bytes: &[u8], tile_count: usize) -> Result<(), JpegError> {
    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();
    let mut sink = NullSink;
    for _ in 0..tile_count {
        Decoder::decode_tile(bytes, &mut ctx, &mut pool, &mut sink)?;
    }
    Ok(())
}

pub(crate) fn libjpeg_turbo_decode_batch(
    decoder: &mut TurboJpegDecoder,
    bytes: &[u8],
    batch_size: usize,
) {
    for _ in 0..batch_size {
        let out = decoder.decode_rgb(bytes).expect("libjpeg-turbo decode");
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
        .decode_region(PixelFormat::Rgb8, roi)
        .expect("j2k region decode");
    std::hint::black_box(out);
}

pub(crate) fn libjpeg_turbo_decode_region(decoder: &mut TurboJpegDecoder, bytes: &[u8], roi: Rect) {
    let out = decoder
        .decode_region_rgb(bytes, roi)
        .expect("libjpeg-turbo region decode");
    std::hint::black_box(out);
}

pub(crate) fn j2k_decode_scaled(bytes: &[u8], factor: Downscale) {
    let dec = Decoder::new(bytes).expect("j2k decoder");
    let (out, _) = dec
        .decode_scaled(PixelFormat::Rgb8, factor)
        .expect("j2k scaled decode");
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

pub(crate) fn j2k_decode_region_scaled(bytes: &[u8], side: u32, factor: Downscale) {
    let dec = Decoder::new(bytes).expect("j2k decoder");
    let roi = centered_roi(dec.info().dimensions, side);
    let (out, _) = dec
        .decode_region_scaled(PixelFormat::Rgb8, roi, factor)
        .expect("j2k scaled region decode");
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
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
        ZCursor::new(bytes),
        DecoderOptions::new_fast()
            .set_max_width(ZUNE_DIMENSION_LIMIT)
            .set_max_height(ZUNE_DIMENSION_LIMIT)
            .jpeg_set_out_colorspace(ZuneColorSpace::RGB),
    );
    let out = decoder.decode().expect("zune-jpeg decode");
    let info = decoder.info().expect("zune-jpeg info");
    let roi = centered_roi((info.width as u32, info.height as u32), side);
    let cropped = crop_rgb(&out, info.width.into(), roi);
    std::hint::black_box(cropped);
}

pub(crate) fn zune_decode_scaled(bytes: &[u8], factor: Downscale) {
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
        ZCursor::new(bytes),
        DecoderOptions::new_fast()
            .set_max_width(ZUNE_DIMENSION_LIMIT)
            .set_max_height(ZUNE_DIMENSION_LIMIT)
            .jpeg_set_out_colorspace(ZuneColorSpace::RGB),
    );
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
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
        ZCursor::new(bytes),
        DecoderOptions::new_fast()
            .set_max_width(ZUNE_DIMENSION_LIMIT)
            .set_max_height(ZUNE_DIMENSION_LIMIT)
            .jpeg_set_out_colorspace(ZuneColorSpace::RGB),
    );
    let out = decoder.decode().expect("zune-jpeg decode");
    let info = decoder.info().expect("zune-jpeg info");
    let roi = centered_roi((info.width as u32, info.height as u32), side);
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

fn scaled_dims((width, height): (u32, u32), factor: Downscale) -> (u32, u32) {
    let denom = factor.denominator();
    (width.div_ceil(denom), height.div_ceil(denom))
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
