// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use j2k::{
    decode_tiles_region_scaled_into, encode_j2k_lossless, recode_j2k_to_htj2k_lossless,
    wrap_j2k_codestream, CpuDecodeParallelism, Downscale, EncodeBackendPreference, ImageDecodeRows,
    J2kBlockCodingMode, J2kCodec, J2kContext, J2kDecoder, J2kEncodeValidation, J2kFileWrapOptions,
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kScratchPool, J2kToHtj2kOptions, PixelFormat,
    Rect, RowSink, TileBatchDecode, TileBatchOptions, TileRegionScaledDecodeJob,
};
use j2k_test_support::{patterned_gray8, patterned_rgb8};

const TILE_SIDE: u32 = 128;
const ROI_SIDE: u32 = 64;
const HT_TILE_SIDE: u32 = 512;
const CPU_MATRIX_SIDE: u32 = 512;
const BATCH_SIZE: usize = 16;

fn bench_encode_options() -> J2kLosslessEncodeOptions {
    let mut options = J2kLosslessEncodeOptions::default();
    options.backend = EncodeBackendPreference::CpuOnly;
    options.validation = J2kEncodeValidation::External;
    options
}

fn ht_encode_options() -> J2kLosslessEncodeOptions {
    let mut options = bench_encode_options();
    options.block_coding_mode = J2kBlockCodingMode::HighThroughput;
    options
}

fn recode_options() -> J2kToHtj2kOptions {
    let mut options = J2kToHtj2kOptions::default();
    options.validation = J2kEncodeValidation::External;
    options
}

fn cpu_matrix_encode_options(
    block_coding_mode: J2kBlockCodingMode,
    validation: J2kEncodeValidation,
) -> J2kLosslessEncodeOptions {
    let mut options = J2kLosslessEncodeOptions::default();
    options.backend = EncodeBackendPreference::CpuOnly;
    options.validation = validation;
    options.block_coding_mode = block_coding_mode;
    options
}

fn encode_gray8_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_gray8(width, height);
    encode_gray8_codestream_from_pixels(width, height, &pixels, bench_encode_options())
}

fn encode_gray16_codestream(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 2);
    for y in 0..height {
        for x in 0..width {
            let sample = ((x * 257 + y * 911) & 0xffff) as u16;
            pixels.extend_from_slice(&sample.to_le_bytes());
        }
    }
    let samples = J2kLosslessSamples::new(&pixels, width, height, 1, 16, false)
        .expect("valid gray16 samples");
    encode_j2k_lossless(samples, &bench_encode_options())
        .expect("encode gray16 codestream")
        .codestream
}

fn encode_ht_gray8_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_gray8(width, height);
    encode_gray8_codestream_from_pixels(width, height, &pixels, ht_encode_options())
}

fn encode_ht_rgb8_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_rgb8(width, height);
    encode_rgb8_codestream_from_pixels(width, height, &pixels, ht_encode_options())
}

fn encode_gray8_codestream_from_pixels(
    width: u32,
    height: u32,
    pixels: &[u8],
    options: J2kLosslessEncodeOptions,
) -> Vec<u8> {
    let samples =
        J2kLosslessSamples::new(pixels, width, height, 1, 8, false).expect("valid gray8 samples");
    encode_j2k_lossless(samples, &options)
        .expect("encode gray8 codestream")
        .codestream
}

fn encode_rgb8_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_rgb8(width, height);
    encode_rgb8_codestream_from_pixels(width, height, &pixels, bench_encode_options())
}

fn encode_rgb8_codestream_with_levels(width: u32, height: u32, levels: u8) -> Vec<u8> {
    let pixels = patterned_rgb8(width, height);
    encode_rgb8_codestream_from_pixels(
        width,
        height,
        &pixels,
        bench_encode_options().with_max_decomposition_levels(Some(levels)),
    )
}

fn encode_rgb8_codestream_from_pixels(
    width: u32,
    height: u32,
    pixels: &[u8],
    options: J2kLosslessEncodeOptions,
) -> Vec<u8> {
    let samples =
        J2kLosslessSamples::new(pixels, width, height, 3, 8, false).expect("valid rgb8 samples");
    encode_j2k_lossless(samples, &options)
        .expect("encode rgb8 codestream")
        .codestream
}

fn bench_lossless_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("j2k_public_lossless_encode");

    let gray = patterned_gray8(TILE_SIDE, TILE_SIDE);
    let options = bench_encode_options();
    group.bench_function("gray8_128x128", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                std::hint::black_box(gray.as_slice()),
                TILE_SIDE,
                TILE_SIDE,
                1,
                8,
                false,
            )
            .expect("valid gray8 samples");
            let encoded = encode_j2k_lossless(samples, &options).expect("encode gray8");
            std::hint::black_box(encoded.codestream.len());
        });
    });

    let rgb = patterned_rgb8(TILE_SIDE, TILE_SIDE);
    group.bench_function("rgb8_128x128", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                std::hint::black_box(rgb.as_slice()),
                TILE_SIDE,
                TILE_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded = encode_j2k_lossless(samples, &options).expect("encode rgb8");
            std::hint::black_box(encoded.codestream.len());
        });
    });

    group.finish();
}

fn bench_inspect(c: &mut Criterion) {
    let codestream = encode_rgb8_codestream(TILE_SIDE, TILE_SIDE);

    let mut group = c.benchmark_group("j2k_public_inspect");
    group.bench_function("rgb8_128x128", |b| {
        b.iter(|| {
            let info =
                J2kDecoder::inspect(std::hint::black_box(codestream.as_slice())).expect("inspect");
            std::hint::black_box(info);
        });
    });
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let codestream = encode_rgb8_codestream(TILE_SIDE, TILE_SIDE);
    let ht_codestream = encode_ht_gray8_codestream(HT_TILE_SIDE, HT_TILE_SIDE);
    let mut group = c.benchmark_group("j2k_public_decode");

    let full_stride = TILE_SIDE as usize * 3;
    let mut full = vec![0u8; full_stride * TILE_SIDE as usize];
    group.bench_function("rgb8_full_128x128", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(std::hint::black_box(codestream.as_slice())).expect("rgb8 decoder");
            decoder
                .decode_into(&mut full, full_stride, PixelFormat::Rgb8)
                .expect("decode full rgb8");
            std::hint::black_box(&full);
        });
    });

    let roi = Rect {
        x: 32,
        y: 32,
        w: ROI_SIDE,
        h: ROI_SIDE,
    };
    let roi_stride = ROI_SIDE as usize * 3;
    let mut roi_out = vec![0u8; roi_stride * ROI_SIDE as usize];
    let mut pool = J2kScratchPool::new();
    group.bench_function("rgb8_roi_64x64", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(std::hint::black_box(codestream.as_slice())).expect("rgb8 decoder");
            decoder
                .decode_region_into(&mut pool, &mut roi_out, roi_stride, PixelFormat::Rgb8, roi)
                .expect("decode rgb8 roi");
            std::hint::black_box(&roi_out);
        });
    });

    let ht_stride = HT_TILE_SIDE as usize;
    let mut ht_out = vec![0u8; ht_stride * HT_TILE_SIDE as usize];
    group.bench_function("htj2k_gray8_full_512x512", |b| {
        b.iter(|| {
            let mut decoder = J2kDecoder::new(std::hint::black_box(ht_codestream.as_slice()))
                .expect("htj2k decoder");
            decoder
                .decode_into(&mut ht_out, ht_stride, PixelFormat::Gray8)
                .expect("decode full htj2k gray8");
            std::hint::black_box(&ht_out);
        });
    });

    group.finish();
}

fn bench_recode(c: &mut Criterion) {
    let classic = encode_rgb8_codestream(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
    let htj2k = encode_ht_rgb8_codestream(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
    let options = recode_options();
    let mut group = c.benchmark_group("j2k_public_recode");

    group.bench_function("classic_rgb8_512_to_htj2k_53_coefficients", |b| {
        b.iter(|| {
            let recoded =
                recode_j2k_to_htj2k_lossless(std::hint::black_box(classic.as_slice()), options)
                    .expect("coefficient-domain recode");
            std::hint::black_box(recoded.bytes.len());
        });
    });

    group.bench_function("raw_htj2k_rgb8_512_passthrough", |b| {
        b.iter(|| {
            let recoded =
                recode_j2k_to_htj2k_lossless(std::hint::black_box(htj2k.as_slice()), options)
                    .expect("HTJ2K passthrough");
            std::hint::black_box(recoded.bytes.len());
        });
    });

    group.finish();
}

fn bench_region_scaled(c: &mut Criterion) {
    let codestream = encode_rgb8_codestream_with_levels(TILE_SIDE, TILE_SIDE, 2);
    let roi = Rect {
        x: 32,
        y: 32,
        w: ROI_SIDE,
        h: ROI_SIDE,
    };
    let out_side = ROI_SIDE.div_ceil(Downscale::Quarter.denominator());
    let stride = out_side as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut out = vec![0u8; stride * out_side as usize];
    let mut pool = J2kScratchPool::new();

    let mut group = c.benchmark_group("j2k_public_decode_region_scaled");
    group.bench_function("rgb8_region_scaled_64x64_q4", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(std::hint::black_box(codestream.as_slice())).expect("rgb8 decoder");
            decoder
                .decode_region_scaled_into(
                    &mut pool,
                    &mut out,
                    stride,
                    PixelFormat::Rgb8,
                    roi,
                    Downscale::Quarter,
                )
                .expect("decode rgb8 region scaled");
            std::hint::black_box(&out);
        });
    });
    group.finish();
}

fn bench_scaled_reuse(c: &mut Criterion) {
    let codestream = encode_rgb8_codestream_with_levels(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE, 2);
    let scale = Downscale::Quarter;
    let out_side = CPU_MATRIX_SIDE.div_ceil(scale.denominator());
    let stride = out_side as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut setup_out = vec![0u8; stride * out_side as usize];
    let mut reused_out = vec![0u8; stride * out_side as usize];
    let mut reused_decoder = J2kDecoder::new(codestream.as_slice()).expect("reused scaled decoder");
    let mut reused_pool = J2kScratchPool::new();

    let mut group = c.benchmark_group("j2k_public_decode_scaled_reuse");
    group.bench_function("rgb8_512_q4_setup_inclusive", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(std::hint::black_box(codestream.as_slice())).expect("rgb8 decoder");
            let mut pool = J2kScratchPool::new();
            decoder
                .decode_scaled_into(&mut pool, &mut setup_out, stride, PixelFormat::Rgb8, scale)
                .expect("setup-inclusive scaled decode");
            std::hint::black_box(&setup_out);
        });
    });
    group.bench_function("rgb8_512_q4_decoder_setup_excluded", |b| {
        b.iter(|| {
            reused_decoder
                .decode_scaled_into(
                    &mut reused_pool,
                    &mut reused_out,
                    stride,
                    PixelFormat::Rgb8,
                    scale,
                )
                .expect("reused scaled decode");
            std::hint::black_box(&reused_out);
        });
    });
    group.finish();
}

fn bench_region_scaled_reuse(c: &mut Criterion) {
    let codestream = encode_rgb8_codestream_with_levels(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE, 2);
    let roi = Rect {
        x: 128,
        y: 128,
        w: 256,
        h: 256,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let stride = scaled.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut setup_out = vec![0u8; stride * scaled.h as usize];
    let mut reused_out = vec![0u8; stride * scaled.h as usize];
    let mut reused_decoder =
        J2kDecoder::new(codestream.as_slice()).expect("reused region-scaled decoder");
    let mut reused_pool = J2kScratchPool::new();

    let mut group = c.benchmark_group("j2k_public_decode_region_scaled_reuse");
    group.bench_function("rgb8_512_roi256_q4_setup_inclusive", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(std::hint::black_box(codestream.as_slice())).expect("rgb8 decoder");
            let mut pool = J2kScratchPool::new();
            decoder
                .decode_region_scaled_into(
                    &mut pool,
                    &mut setup_out,
                    stride,
                    PixelFormat::Rgb8,
                    roi,
                    scale,
                )
                .expect("setup-inclusive region scaled decode");
            std::hint::black_box(&setup_out);
        });
    });
    group.bench_function("rgb8_512_roi256_q4_decoder_setup_excluded", |b| {
        b.iter(|| {
            reused_decoder
                .decode_region_scaled_into(
                    &mut reused_pool,
                    &mut reused_out,
                    stride,
                    PixelFormat::Rgb8,
                    roi,
                    scale,
                )
                .expect("reused region scaled decode");
            std::hint::black_box(&reused_out);
        });
    });
    group.finish();
}

fn bench_mixed_scale_reuse(c: &mut Criterion) {
    let codestream = encode_rgb8_codestream_with_levels(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE, 2);
    let mut decoder = J2kDecoder::new(codestream.as_slice()).expect("mixed-scale decoder");
    let mut pool = J2kScratchPool::new();
    let half_side = CPU_MATRIX_SIDE.div_ceil(Downscale::Half.denominator());
    let half_stride = half_side as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut half_out = vec![0u8; half_stride * half_side as usize];
    let quarter_side = CPU_MATRIX_SIDE.div_ceil(Downscale::Quarter.denominator());
    let quarter_stride = quarter_side as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut quarter_out = vec![0u8; quarter_stride * quarter_side as usize];

    let mut group = c.benchmark_group("j2k_public_decode_mixed_scale_reuse");
    group.bench_function("rgb8_512_q4_q2_q4_single_decoder", |b| {
        b.iter(|| {
            decoder
                .decode_scaled_into(
                    &mut pool,
                    &mut quarter_out,
                    quarter_stride,
                    PixelFormat::Rgb8,
                    Downscale::Quarter,
                )
                .expect("quarter decode");
            decoder
                .decode_scaled_into(
                    &mut pool,
                    &mut half_out,
                    half_stride,
                    PixelFormat::Rgb8,
                    Downscale::Half,
                )
                .expect("half decode");
            decoder
                .decode_scaled_into(
                    &mut pool,
                    &mut quarter_out,
                    quarter_stride,
                    PixelFormat::Rgb8,
                    Downscale::Quarter,
                )
                .expect("quarter decode after half");
            std::hint::black_box((&quarter_out, &half_out));
        });
    });
    group.finish();
}

fn bench_rows(c: &mut Criterion) {
    let codestream = encode_gray8_codestream(TILE_SIDE, TILE_SIDE);
    let gray16_codestream = encode_gray16_codestream(TILE_SIDE, TILE_SIDE);
    let mut group = c.benchmark_group("j2k_public_decode_rows");
    group.bench_function("gray8_rows_128x128", |b| {
        b.iter(|| {
            let mut decoder = J2kDecoder::new(std::hint::black_box(codestream.as_slice()))
                .expect("gray8 decoder");
            let mut sink = VecRowSink::new(TILE_SIDE, TILE_SIDE);
            decoder.decode_rows(&mut sink).expect("decode gray8 rows");
            std::hint::black_box(sink.rows);
        });
    });
    group.bench_function("gray8_rows_128x128_reused_decoder", |b| {
        let mut decoder = J2kDecoder::new(codestream.as_slice()).expect("gray8 reused decoder");
        b.iter(|| {
            let mut sink = VecRowSink::new(TILE_SIDE, TILE_SIDE);
            decoder.decode_rows(&mut sink).expect("decode gray8 rows");
            std::hint::black_box(sink.rows);
        });
    });
    group.bench_function("gray16_rows_128x128_reused_decoder", |b| {
        let mut decoder =
            J2kDecoder::new(gray16_codestream.as_slice()).expect("gray16 reused decoder");
        b.iter(|| {
            let mut sink = VecRowSinkU16::new(TILE_SIDE, TILE_SIDE);
            <J2kDecoder<'_> as ImageDecodeRows<'_, u16>>::decode_rows(&mut decoder, &mut sink)
                .expect("decode gray16 rows");
            std::hint::black_box(sink.rows);
        });
    });
    group.finish();
}

fn bench_tile_batch(c: &mut Criterion) {
    let repeated = encode_gray8_codestream(TILE_SIDE, TILE_SIDE);
    let ht_repeated = encode_ht_gray8_codestream(TILE_SIDE, TILE_SIDE);
    let mut distinct = Vec::with_capacity(BATCH_SIZE);
    let mut ht_distinct = Vec::with_capacity(BATCH_SIZE);
    for idx in 0..BATCH_SIZE {
        let mut pixels = patterned_gray8(TILE_SIDE, TILE_SIDE);
        pixels[0] = pixels[0].wrapping_add(u8::try_from(idx).expect("batch index fits u8"));
        distinct.push(encode_gray8_codestream_from_pixels(
            TILE_SIDE,
            TILE_SIDE,
            &pixels,
            bench_encode_options(),
        ));
        ht_distinct.push(encode_gray8_codestream_from_pixels(
            TILE_SIDE,
            TILE_SIDE,
            &pixels,
            ht_encode_options(),
        ));
    }

    let stride = TILE_SIDE as usize;
    let mut out = vec![0u8; stride * TILE_SIDE as usize];
    let mut group = c.benchmark_group("j2k_public_tile_batch");

    group.bench_function("gray8_repeated_batch_16", |b| {
        b.iter(|| {
            let mut ctx = J2kContext::default();
            let mut pool = J2kScratchPool::new();
            for _ in 0..BATCH_SIZE {
                <J2kCodec as TileBatchDecode>::decode_tile(
                    &mut ctx,
                    &mut pool,
                    std::hint::black_box(repeated.as_slice()),
                    &mut out,
                    stride,
                    PixelFormat::Gray8,
                )
                .expect("decode repeated gray8 tile");
            }
            std::hint::black_box(&out);
        });
    });

    group.bench_function("htj2k_gray8_repeated_batch_16", |b| {
        b.iter(|| {
            let mut ctx = J2kContext::default();
            let mut pool = J2kScratchPool::new();
            for _ in 0..BATCH_SIZE {
                <J2kCodec as TileBatchDecode>::decode_tile(
                    &mut ctx,
                    &mut pool,
                    std::hint::black_box(ht_repeated.as_slice()),
                    &mut out,
                    stride,
                    PixelFormat::Gray8,
                )
                .expect("decode repeated htj2k gray8 tile");
            }
            std::hint::black_box(&out);
        });
    });

    group.bench_function("gray8_distinct_batch_16", |b| {
        b.iter(|| {
            let mut ctx = J2kContext::default();
            let mut pool = J2kScratchPool::new();
            for codestream in &distinct {
                <J2kCodec as TileBatchDecode>::decode_tile(
                    &mut ctx,
                    &mut pool,
                    std::hint::black_box(codestream.as_slice()),
                    &mut out,
                    stride,
                    PixelFormat::Gray8,
                )
                .expect("decode distinct gray8 tile");
            }
            std::hint::black_box(&out);
        });
    });

    group.bench_function("htj2k_gray8_distinct_batch_16", |b| {
        b.iter(|| {
            let mut ctx = J2kContext::default();
            let mut pool = J2kScratchPool::new();
            for codestream in &ht_distinct {
                <J2kCodec as TileBatchDecode>::decode_tile(
                    &mut ctx,
                    &mut pool,
                    std::hint::black_box(codestream.as_slice()),
                    &mut out,
                    stride,
                    PixelFormat::Gray8,
                )
                .expect("decode distinct htj2k gray8 tile");
            }
            std::hint::black_box(&out);
        });
    });

    group.finish();
}

#[expect(
    clippy::too_many_lines,
    reason = "the benchmark keeps one cohesive region-and-scale matrix in a single Criterion group"
)]
fn bench_tile_batch_region_scaled_rgb(c: &mut Criterion) {
    let repeated_classic = encode_rgb8_codestream(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
    let repeated_htj2k = encode_ht_rgb8_codestream(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
    let repeated_htj2k_jph =
        wrap_j2k_codestream(&repeated_htj2k, J2kFileWrapOptions::jph()).expect("wrap HTJ2K JPH");
    let repeated_htj2k_256 = encode_ht_rgb8_codestream(256, 256);
    let repeated_htj2k_256_jph =
        wrap_j2k_codestream(&repeated_htj2k_256, J2kFileWrapOptions::jph())
            .expect("wrap 256 HTJ2K JPH");
    let mut distinct_classic = Vec::with_capacity(BATCH_SIZE);
    let mut distinct_htj2k = Vec::with_capacity(BATCH_SIZE);
    for idx in 0..BATCH_SIZE {
        let mut pixels = patterned_rgb8(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
        pixels[0] = pixels[0].wrapping_add(u8::try_from(idx).expect("batch index fits u8"));
        distinct_classic.push(encode_rgb8_codestream_from_pixels(
            CPU_MATRIX_SIDE,
            CPU_MATRIX_SIDE,
            &pixels,
            bench_encode_options(),
        ));
        distinct_htj2k.push(encode_rgb8_codestream_from_pixels(
            CPU_MATRIX_SIDE,
            CPU_MATRIX_SIDE,
            &pixels,
            ht_encode_options(),
        ));
    }

    let roi = Rect {
        x: 128,
        y: 128,
        w: 256,
        h: 256,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let stride = scaled.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let output_len = stride * scaled.h as usize;
    let rgba_stride = scaled.w as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let rgba_output_len = rgba_stride * scaled.h as usize;

    let roi_256 = Rect {
        x: 64,
        y: 64,
        w: 128,
        h: 128,
    };
    let scaled_256 = roi_256.scaled_covering(scale);
    let stride_256 = scaled_256.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let output_len_256 = stride_256 * scaled_256.h as usize;

    let mut group = c.benchmark_group("j2k_public_tile_batch_region_scaled_rgb_q4");
    group.bench_function("classic_repeated_512_roi256_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; output_len]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileRegionScaledDecodeJob {
                    input: std::hint::black_box(repeated_classic.as_slice()),
                    out,
                    stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgb8,
                TileBatchOptions::default(),
            )
            .expect("decode repeated classic RGB ROI+scale batch");
            std::hint::black_box((outputs, outcomes));
        });
    });
    group.bench_function("classic_distinct_512_roi256_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; output_len]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .zip(distinct_classic.iter())
                .map(|(out, input)| TileRegionScaledDecodeJob {
                    input: std::hint::black_box(input.as_slice()),
                    out,
                    stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgb8,
                TileBatchOptions::default(),
            )
            .expect("decode distinct classic RGB ROI+scale batch");
            std::hint::black_box((outputs, outcomes));
        });
    });
    group.bench_function("htj2k_repeated_512_roi256_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; output_len]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileRegionScaledDecodeJob {
                    input: std::hint::black_box(repeated_htj2k.as_slice()),
                    out,
                    stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgb8,
                TileBatchOptions::default(),
            )
            .expect("decode repeated HTJ2K RGB ROI+scale batch");
            std::hint::black_box((outputs, outcomes));
        });
    });
    group.bench_function("htj2k_jph_rgb8_repeated_512_roi256_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; output_len]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileRegionScaledDecodeJob {
                    input: std::hint::black_box(repeated_htj2k_jph.as_slice()),
                    out,
                    stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgb8,
                TileBatchOptions::default(),
            )
            .expect("decode repeated HTJ2K JPH RGB ROI+scale batch");
            std::hint::black_box((outputs, outcomes));
        });
    });
    group.bench_function("htj2k_jph_rgba8_repeated_512_roi256_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; rgba_output_len]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileRegionScaledDecodeJob {
                    input: std::hint::black_box(repeated_htj2k_jph.as_slice()),
                    out,
                    stride: rgba_stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgba8,
                TileBatchOptions::default(),
            )
            .expect("decode repeated HTJ2K JPH RGBA ROI+scale batch");
            std::hint::black_box((outputs, outcomes));
        });
    });
    group.bench_function("htj2k_jph_rgb8_repeated_256_roi128_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; output_len_256]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileRegionScaledDecodeJob {
                    input: std::hint::black_box(repeated_htj2k_256_jph.as_slice()),
                    out,
                    stride: stride_256,
                    roi: roi_256,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgb8,
                TileBatchOptions::default(),
            )
            .expect("decode repeated 256 HTJ2K JPH RGB ROI+scale batch");
            std::hint::black_box((outputs, outcomes));
        });
    });
    group.bench_function("htj2k_distinct_512_roi256_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; output_len]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .zip(distinct_htj2k.iter())
                .map(|(out, input)| TileRegionScaledDecodeJob {
                    input: std::hint::black_box(input.as_slice()),
                    out,
                    stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgb8,
                TileBatchOptions::default(),
            )
            .expect("decode distinct HTJ2K RGB ROI+scale batch");
            std::hint::black_box((outputs, outcomes));
        });
    });
    group.finish();
}

fn bench_decode_gray_setup(c: &mut Criterion) {
    let codestream = encode_gray8_codestream(TILE_SIDE, TILE_SIDE);
    let stride = TILE_SIDE as usize;
    let mut out = vec![0u8; stride * TILE_SIDE as usize];

    let mut group = c.benchmark_group("j2k_public_decode_gray");
    group.bench_function("gray8_full_128x128", |b| {
        b.iter(|| {
            let mut decoder = J2kDecoder::new(std::hint::black_box(codestream.as_slice()))
                .expect("gray8 decoder");
            decoder
                .decode_into(&mut out, stride, PixelFormat::Gray8)
                .expect("decode full gray8");
            std::hint::black_box(&out);
        });
    });
    group.finish();
}

fn bench_cpu_encode_matrix(c: &mut Criterion) {
    let pixels = patterned_rgb8(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
    let classic_external =
        cpu_matrix_encode_options(J2kBlockCodingMode::Classic, J2kEncodeValidation::External);
    let htj2k_external = cpu_matrix_encode_options(
        J2kBlockCodingMode::HighThroughput,
        J2kEncodeValidation::External,
    );
    let classic_roundtrip = cpu_matrix_encode_options(
        J2kBlockCodingMode::Classic,
        J2kEncodeValidation::CpuRoundTrip,
    );

    let mut group = c.benchmark_group("j2k_public_cpu_encode_matrix");
    group.bench_function("rgb8_512_classic_external", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                std::hint::black_box(pixels.as_slice()),
                CPU_MATRIX_SIDE,
                CPU_MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                encode_j2k_lossless(samples, &classic_external).expect("classic CPU encode");
            std::hint::black_box(encoded.codestream.len());
        });
    });

    group.bench_function("rgb8_512_htj2k_external", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                std::hint::black_box(pixels.as_slice()),
                CPU_MATRIX_SIDE,
                CPU_MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded = encode_j2k_lossless(samples, &htj2k_external).expect("HTJ2K CPU encode");
            std::hint::black_box(encoded.codestream.len());
        });
    });

    group.bench_function("rgb8_512_classic_roundtrip", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                std::hint::black_box(pixels.as_slice()),
                CPU_MATRIX_SIDE,
                CPU_MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                encode_j2k_lossless(samples, &classic_roundtrip).expect("classic CPU encode");
            std::hint::black_box(encoded.codestream.len());
        });
    });
    group.finish();
}

fn bench_cpu_decode_matrix(c: &mut Criterion) {
    let pixels = patterned_gray8(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
    let classic_codestream = encode_gray8_codestream_from_pixels(
        CPU_MATRIX_SIDE,
        CPU_MATRIX_SIDE,
        &pixels,
        cpu_matrix_encode_options(J2kBlockCodingMode::Classic, J2kEncodeValidation::External),
    );
    let htj2k_codestream = encode_gray8_codestream_from_pixels(
        CPU_MATRIX_SIDE,
        CPU_MATRIX_SIDE,
        &pixels,
        cpu_matrix_encode_options(
            J2kBlockCodingMode::HighThroughput,
            J2kEncodeValidation::External,
        ),
    );
    let rgb_classic_codestream = encode_rgb8_codestream(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
    let rgb_htj2k_codestream = encode_ht_rgb8_codestream(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);

    let stride = CPU_MATRIX_SIDE as usize;
    let mut classic_out = vec![0u8; stride * CPU_MATRIX_SIDE as usize];
    let mut htj2k_out = vec![0u8; stride * CPU_MATRIX_SIDE as usize];
    let rgb_stride = CPU_MATRIX_SIDE as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut rgb_classic_out = vec![0u8; rgb_stride * CPU_MATRIX_SIDE as usize];
    let mut rgb_htj2k_out = vec![0u8; rgb_stride * CPU_MATRIX_SIDE as usize];

    let mut group = c.benchmark_group("j2k_public_cpu_decode_matrix");
    group.bench_function("gray8_512_classic_decode", |b| {
        b.iter(|| {
            let mut decoder = J2kDecoder::new(std::hint::black_box(classic_codestream.as_slice()))
                .expect("J2K decoder");
            decoder
                .decode_into(&mut classic_out, stride, PixelFormat::Gray8)
                .expect("decode classic gray8");
            std::hint::black_box(&classic_out);
        });
    });

    group.bench_function("gray8_512_htj2k_decode", |b| {
        b.iter(|| {
            let mut decoder = J2kDecoder::new(std::hint::black_box(htj2k_codestream.as_slice()))
                .expect("HTJ2K decoder");
            decoder
                .decode_into(&mut htj2k_out, stride, PixelFormat::Gray8)
                .expect("decode htj2k gray8");
            std::hint::black_box(&htj2k_out);
        });
    });

    group.bench_function("rgb8_512_classic_decode", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(std::hint::black_box(rgb_classic_codestream.as_slice()))
                    .expect("J2K decoder");
            decoder
                .decode_into(&mut rgb_classic_out, rgb_stride, PixelFormat::Rgb8)
                .expect("decode classic rgb8");
            std::hint::black_box(&rgb_classic_out);
        });
    });

    group.bench_function("rgb8_512_classic_decode_serial", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(std::hint::black_box(rgb_classic_codestream.as_slice()))
                    .expect("J2K decoder");
            decoder.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
            decoder
                .decode_into(&mut rgb_classic_out, rgb_stride, PixelFormat::Rgb8)
                .expect("decode serial classic rgb8");
            std::hint::black_box(&rgb_classic_out);
        });
    });

    group.bench_function("rgb8_512_htj2k_decode", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(std::hint::black_box(rgb_htj2k_codestream.as_slice()))
                    .expect("HTJ2K decoder");
            decoder
                .decode_into(&mut rgb_htj2k_out, rgb_stride, PixelFormat::Rgb8)
                .expect("decode htj2k rgb8");
            std::hint::black_box(&rgb_htj2k_out);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_lossless_encode,
    bench_inspect,
    bench_decode,
    bench_recode,
    bench_region_scaled,
    bench_scaled_reuse,
    bench_region_scaled_reuse,
    bench_mixed_scale_reuse,
    bench_rows,
    bench_tile_batch,
    bench_tile_batch_region_scaled_rgb,
    bench_decode_gray_setup,
    bench_cpu_encode_matrix,
    bench_cpu_decode_matrix
);
criterion_main!(benches);

struct VecRowSink {
    rows: Vec<u8>,
    width: usize,
}

impl VecRowSink {
    fn new(width: u32, height: u32) -> Self {
        Self {
            rows: vec![0; width as usize * height as usize],
            width: width as usize,
        }
    }
}

impl RowSink<u8> for VecRowSink {
    type Error = std::convert::Infallible;

    fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), Self::Error> {
        let start = y as usize * self.width;
        let end = start + row.len();
        self.rows[start..end].copy_from_slice(row);
        Ok(())
    }
}

struct VecRowSinkU16 {
    rows: Vec<u16>,
    width: usize,
}

impl VecRowSinkU16 {
    fn new(width: u32, height: u32) -> Self {
        Self {
            rows: vec![0; width as usize * height as usize],
            width: width as usize,
        }
    }
}

impl RowSink<u16> for VecRowSinkU16 {
    type Error = std::convert::Infallible;

    fn write_row(&mut self, y: u32, row: &[u16]) -> Result<(), Self::Error> {
        let start = y as usize * self.width;
        let end = start + row.len();
        self.rows[start..end].copy_from_slice(row);
        Ok(())
    }
}
