// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use signinum_jpeg::transcode::{extract_dct_blocks, DctExtractOptions};
use signinum_transcode::dct53_1d::{
    dct8_blocks_to_dwt53_float_linear, dct8_to_dwt53_float_linear, idct8_blocks_then_dwt53_float,
    idct8_then_dwt53_float,
};
use signinum_transcode::dct53_2d::{dct8x8_to_dwt53_float_linear, idct8x8_then_dwt53_float};
use signinum_transcode::dct53_multilevel::{
    dct8x8_to_dwt53_multilevel_float_linear, idct8x8_then_dwt53_multilevel_float,
};
use signinum_transcode::{jpeg_to_htj2k, JpegToHtj2kOptions};

fn bench_dct53(c: &mut Criterion) {
    let coeffs = [91.0, -36.0, 14.0, -9.0, 3.0, 22.0, -11.0, 4.0];

    let mut single_block = c.benchmark_group("dct53_1d_single_block_scalar");
    single_block.bench_function("direct_linear", |b| {
        b.iter(|| dct8_to_dwt53_float_linear(black_box(coeffs)));
    });
    single_block.bench_function("idct_then_dwt_reference", |b| {
        b.iter(|| idct8_then_dwt53_float(black_box(coeffs)));
    });
    single_block.finish();

    let blocks = pseudo_random_blocks(32);

    let mut multi_block = c.benchmark_group("dct53_1d_multi_block_scalar");
    multi_block.bench_function("direct_linear", |b| {
        b.iter(|| dct8_blocks_to_dwt53_float_linear(black_box(&blocks)));
    });
    multi_block.bench_function("idct_then_dwt_reference", |b| {
        b.iter(|| idct8_blocks_then_dwt53_float(black_box(&blocks)));
    });
    multi_block.finish();

    let block_2d = synthetic_8x8_block();

    let mut two_dimensional = c.benchmark_group("dct53_2d_single_level_scalar");
    two_dimensional.bench_function("direct_linear", |b| {
        b.iter(|| dct8x8_to_dwt53_float_linear(black_box(block_2d)));
    });
    two_dimensional.bench_function("idct_then_dwt_reference", |b| {
        b.iter(|| idct8x8_then_dwt53_float(black_box(block_2d)));
    });
    two_dimensional.finish();

    let mut multilevel = c.benchmark_group("dct53_multilevel_scalar");
    multilevel.bench_function("direct_level1_then_ll_recursion", |b| {
        b.iter(|| {
            dct8x8_to_dwt53_multilevel_float_linear(black_box(block_2d), black_box(2))
                .expect("valid decomposition levels");
        });
    });
    multilevel.bench_function("idct_then_dwt_reference", |b| {
        b.iter(|| {
            idct8x8_then_dwt53_multilevel_float(black_box(block_2d), black_box(2))
                .expect("valid decomposition levels");
        });
    });
    multilevel.finish();

    let jpeg_420 =
        include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_420_16x16.jpg");
    let jpeg_restart =
        include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_420_restart_32x16.jpg");

    let mut jpeg_extract = c.benchmark_group("jpeg_dct_extract");
    jpeg_extract.bench_function("baseline_420_16x16", |b| {
        b.iter(|| {
            extract_dct_blocks(black_box(jpeg_420), DctExtractOptions::default())
                .expect("extract baseline 4:2:0 DCT blocks");
        });
    });
    jpeg_extract.bench_function("baseline_420_restart_32x16", |b| {
        b.iter(|| {
            extract_dct_blocks(black_box(jpeg_restart), DctExtractOptions::default())
                .expect("extract restart-coded 4:2:0 DCT blocks");
        });
    });
    jpeg_extract.finish();

    let jpeg_gray = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");
    let transcode_options = JpegToHtj2kOptions::default();
    let mut jpeg_to_htj2k_group = c.benchmark_group("jpeg_to_htj2k_grayscale");
    jpeg_to_htj2k_group.bench_function("grayscale_8x8", |b| {
        b.iter(|| {
            jpeg_to_htj2k(black_box(jpeg_gray), black_box(&transcode_options))
                .expect("transcode grayscale JPEG to HTJ2K");
        });
    });
    jpeg_to_htj2k_group.finish();
}

fn pseudo_random_blocks(block_count: usize) -> Vec<[f64; 8]> {
    let mut state = 0x384f_921d_u32;
    (0..block_count)
        .map(|_| {
            let mut block = [0.0; 8];
            for coeff in &mut block {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let bounded = u16::try_from(state % 257).expect("modulo result fits u16");
                *coeff = f64::from(i32::from(bounded) - 128);
            }
            block
        })
        .collect()
}

fn synthetic_8x8_block() -> [[f64; 8]; 8] {
    let mut block = [[0.0; 8]; 8];
    block[0][0] = 512.0;
    block[0][1] = -31.0;
    block[1][0] = 27.0;
    block[2][3] = 9.0;
    block[7][7] = -6.0;
    block
}

criterion_group!(dct53_benches, bench_dct53);
criterion_main!(dct53_benches);
