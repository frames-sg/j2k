// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use j2k_jpeg::transcode::{extract_dct_blocks, DctExtractOptions};
use j2k_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};
use j2k_test_support::{
    JPEG_BASELINE_420_16X16, JPEG_BASELINE_420_RESTART_32X16, JPEG_BASELINE_422_16X8,
    JPEG_BASELINE_444_8X8, JPEG_GRAYSCALE_8X8,
};

#[allow(dead_code, unreachable_pub)]
#[path = "../tests/support/dct53_1d.rs"]
mod dct53_1d;
#[allow(clippy::large_types_passed_by_value, dead_code, unreachable_pub)]
#[path = "../src/dct53_2d.rs"]
mod dct53_2d;
#[allow(clippy::large_types_passed_by_value, dead_code, unreachable_pub)]
#[path = "../tests/support/dct53_multilevel.rs"]
mod dct53_multilevel;
#[allow(dead_code, unreachable_pub, unused_imports)]
#[path = "../src/dct_grid.rs"]
mod dct_grid;
#[allow(dead_code, unused_imports)]
#[path = "../src/reversible53.rs"]
mod reversible53;

pub use dct_grid::DctGridError;

use dct53_1d::{
    dct8_blocks_to_dwt53_float_linear, dct8_to_dwt53_float_linear, idct8_blocks_then_dwt53_float,
    idct8_then_dwt53_float,
};
use dct53_2d::{
    dct8x8_blocks_then_dwt53_float, dct8x8_blocks_to_dwt53_float_linear,
    dct8x8_blocks_to_dwt53_float_linear_with_scratch, dct8x8_to_dwt53_float_linear,
    idct8x8_then_dwt53_float, Dct53GridScratch,
};
use dct53_multilevel::{
    dct8x8_to_dwt53_multilevel_float_linear, idct8x8_then_dwt53_multilevel_float,
};
use j2k_transcode::dct97_2d::{
    dct8x8_blocks_then_dwt97_float, dct8x8_blocks_then_dwt97_float_with_scratch, Dct97GridScratch,
};
use j2k_transcode::{jpeg_to_htj2k, JpegToHtj2kOptions, JpegToHtj2kTranscoder};
use std::hint::black_box;

fn bench_dct53_math(c: &mut Criterion) {
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

    let grid_blocks = synthetic_8x8_grid_blocks(2, 2);
    let mut two_dimensional_grid = c.benchmark_group("dct53_2d_grid_scalar");
    two_dimensional_grid.bench_function("direct_linear_13x11", |b| {
        b.iter(|| {
            dct8x8_blocks_to_dwt53_float_linear(
                black_box(&grid_blocks),
                black_box(2),
                black_box(2),
                black_box(13),
                black_box(11),
            )
            .expect("valid DCT grid");
        });
    });
    let mut grid_scratch = Dct53GridScratch::default();
    two_dimensional_grid.bench_function("direct_linear_13x11_scratch_reuse", |b| {
        b.iter(|| {
            dct8x8_blocks_to_dwt53_float_linear_with_scratch(
                black_box(&grid_blocks),
                black_box(2),
                black_box(2),
                black_box(13),
                black_box(11),
                black_box(&mut grid_scratch),
            )
            .expect("valid DCT grid");
        });
    });
    two_dimensional_grid.bench_function("idct_then_dwt_reference_13x11", |b| {
        b.iter(|| {
            dct8x8_blocks_then_dwt53_float(
                black_box(&grid_blocks),
                black_box(2),
                black_box(2),
                black_box(13),
                black_box(11),
            )
            .expect("valid DCT grid");
        });
    });
    two_dimensional_grid.finish();

    bench_dct97_grid(c, &grid_blocks);

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
}

fn bench_dct97_grid(c: &mut Criterion, grid_blocks: &[[[f64; 8]; 8]]) {
    let mut two_dimensional_grid_97 = c.benchmark_group("dct97_2d_grid_scalar");
    let mut grid_97_scratch = Dct97GridScratch::default();
    two_dimensional_grid_97.bench_function("idct_then_dwt_reference_13x11", |b| {
        b.iter(|| {
            dct8x8_blocks_then_dwt97_float(
                black_box(grid_blocks),
                black_box(2),
                black_box(2),
                black_box(13),
                black_box(11),
            )
            .expect("valid DCT grid");
        });
    });
    two_dimensional_grid_97.bench_function("idct_then_dwt_reference_13x11_scratch_reuse", |b| {
        b.iter(|| {
            dct8x8_blocks_then_dwt97_float_with_scratch(
                black_box(grid_blocks),
                black_box(2),
                black_box(2),
                black_box(13),
                black_box(11),
                black_box(&mut grid_97_scratch),
            )
            .expect("valid DCT grid");
        });
    });
    two_dimensional_grid_97.finish();
}

fn bench_layout_candidates(c: &mut Criterion) {
    let block_cols = 8;
    let block_rows = 8;
    let blocks = synthetic_natural_i16_blocks(block_cols * block_rows);

    let mut layout = c.benchmark_group("dct53_layout_candidates");
    layout.bench_function("aos_8x8_f64", |b| {
        b.iter(|| pack_aos_8x8_f64(black_box(&blocks)));
    });
    layout.bench_function("row_window_packed_f64", |b| {
        b.iter(|| {
            pack_row_window_packed_f64(
                black_box(&blocks),
                black_box(block_cols),
                black_box(block_rows),
            );
        });
    });
    layout.bench_function("soa_coefficient_major_f64", |b| {
        b.iter(|| pack_soa_coefficient_major_f64(black_box(&blocks)));
    });
    layout.finish();
}

fn bench_jpeg_paths(c: &mut Criterion) {
    let jpeg_420 = JPEG_BASELINE_420_16X16;
    let jpeg_restart = JPEG_BASELINE_420_RESTART_32X16;

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

    let jpeg_gray = JPEG_GRAYSCALE_8X8;
    let jpeg_444 = JPEG_BASELINE_444_8X8;
    let jpeg_422 = JPEG_BASELINE_422_16X8;
    let jpeg_420 = JPEG_BASELINE_420_16X16;
    let transcode_options = JpegToHtj2kOptions::default();
    let transcode_97_options = JpegToHtj2kOptions::lossy_97();
    let mut jpeg_to_htj2k_group = c.benchmark_group("jpeg_to_htj2k");
    jpeg_to_htj2k_group.bench_function("grayscale_8x8", |b| {
        b.iter(|| {
            jpeg_to_htj2k(black_box(jpeg_gray), black_box(&transcode_options))
                .expect("transcode grayscale JPEG to HTJ2K");
        });
    });
    let mut stateful_transcoder = JpegToHtj2kTranscoder::default();
    jpeg_to_htj2k_group.bench_function("grayscale_8x8_stateful_reuse", |b| {
        b.iter(|| {
            stateful_transcoder
                .transcode(black_box(jpeg_gray), black_box(&transcode_options))
                .expect("stateful transcode grayscale JPEG to HTJ2K");
        });
    });
    let jpeg_gray_multiblock = grayscale_jpeg(13, 11);
    jpeg_to_htj2k_group.bench_function("grayscale_13x11", |b| {
        b.iter(|| {
            jpeg_to_htj2k(
                black_box(&jpeg_gray_multiblock),
                black_box(&transcode_options),
            )
            .expect("transcode multi-block grayscale JPEG to HTJ2K");
        });
    });
    jpeg_to_htj2k_group.bench_function("ycbcr_444_8x8", |b| {
        b.iter(|| {
            jpeg_to_htj2k(black_box(jpeg_444), black_box(&transcode_options))
                .expect("transcode 4:4:4 YCbCr JPEG to HTJ2K");
        });
    });
    jpeg_to_htj2k_group.bench_function("ycbcr_422_16x8", |b| {
        b.iter(|| {
            jpeg_to_htj2k(black_box(jpeg_422), black_box(&transcode_options))
                .expect("transcode 4:2:2 YCbCr JPEG to HTJ2K");
        });
    });
    jpeg_to_htj2k_group.bench_function("ycbcr_420_16x16", |b| {
        b.iter(|| {
            jpeg_to_htj2k(black_box(jpeg_420), black_box(&transcode_options))
                .expect("transcode 4:2:0 YCbCr JPEG to HTJ2K");
        });
    });
    jpeg_to_htj2k_group.bench_function("grayscale_8x8_float_direct_97", |b| {
        b.iter(|| {
            jpeg_to_htj2k(black_box(jpeg_gray), black_box(&transcode_97_options))
                .expect("transcode grayscale JPEG to 9/7 HTJ2K");
        });
    });
    jpeg_to_htj2k_group.bench_function("ycbcr_420_16x16_float_direct_97", |b| {
        b.iter(|| {
            jpeg_to_htj2k(black_box(jpeg_420), black_box(&transcode_97_options))
                .expect("transcode 4:2:0 YCbCr JPEG to 9/7 HTJ2K");
        });
    });
    jpeg_to_htj2k_group.finish();
}

fn bench_dct53(c: &mut Criterion) {
    bench_dct53_math(c);
    bench_layout_candidates(c);
    bench_jpeg_paths(c);
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

fn synthetic_8x8_grid_blocks(block_cols: usize, block_rows: usize) -> Vec<[[f64; 8]; 8]> {
    let mut blocks = Vec::with_capacity(block_cols * block_rows);
    for block_y in 0..block_rows {
        for block_x in 0..block_cols {
            let mut block = synthetic_8x8_block();
            block[0][0] += (block_x * 17 + block_y * 23) as f64;
            block[0][1] += block_x as f64;
            block[1][0] -= block_y as f64;
            blocks.push(block);
        }
    }
    blocks
}

fn synthetic_natural_i16_blocks(block_count: usize) -> Vec<[i16; 64]> {
    let mut state = 0x91af_b33d_u32;
    (0..block_count)
        .map(|_| {
            let mut block = [0; 64];
            for coefficient in &mut block {
                state = state.wrapping_mul(22_695_477).wrapping_add(1);
                let bounded = u16::try_from((state >> 8) % 2049).expect("modulo result fits u16");
                let signed = i32::from(bounded) - 1024;
                *coefficient = i16::try_from(signed).expect("bounded coefficient fits i16");
            }
            block
        })
        .collect()
}

fn pack_aos_8x8_f64(blocks: &[[i16; 64]]) -> Vec<[[f64; 8]; 8]> {
    blocks
        .iter()
        .map(|block| {
            let mut output = [[0.0; 8]; 8];
            for (idx, &coefficient) in block.iter().enumerate() {
                output[idx / 8][idx % 8] = f64::from(coefficient);
            }
            output
        })
        .collect()
}

fn pack_row_window_packed_f64(
    blocks: &[[i16; 64]],
    block_cols: usize,
    block_rows: usize,
) -> Vec<f64> {
    assert_eq!(blocks.len(), block_cols * block_rows);

    let mut output = Vec::with_capacity(blocks.len() * 64);
    for block_y in 0..block_rows {
        let row_start = block_y * block_cols;
        let row_blocks = &blocks[row_start..row_start + block_cols];
        for coefficient_y in 0..8 {
            for block in row_blocks {
                let coefficient_row = &block[coefficient_y * 8..coefficient_y * 8 + 8];
                output.extend(
                    coefficient_row
                        .iter()
                        .map(|&coefficient| f64::from(coefficient)),
                );
            }
        }
    }
    output
}

fn pack_soa_coefficient_major_f64(blocks: &[[i16; 64]]) -> Vec<f64> {
    let mut output = Vec::with_capacity(blocks.len() * 64);
    for coefficient_idx in 0..64 {
        output.extend(blocks.iter().map(|block| f64::from(block[coefficient_idx])));
    }
    output
}

fn grayscale_jpeg(width: u32, height: u32) -> Vec<u8> {
    let samples = patterned_gray(width, height);
    encode_jpeg_baseline(
        JpegSamples::Gray8 {
            data: &samples,
            width,
            height,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Gray,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode grayscale JPEG")
    .data
}

fn patterned_gray(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            out.push(((x * 7 + y * 11 + 19) & 0xff) as u8);
        }
    }
    out
}

criterion_group!(dct53_benches, bench_dct53);
criterion_main!(dct53_benches);
