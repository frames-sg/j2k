// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use signinum_transcode::accelerator::{DctGridToDwt97Job, DctToWaveletStageAccelerator};
use signinum_transcode::dct97_2d::{
    dct8x8_blocks_to_dwt97_float_linear_with_scratch, Dct97GridScratch,
};
use signinum_transcode_metal::{MetalDctToWaveletStageAccelerator, METAL_UNAVAILABLE};

fn bench_dct97_projection(c: &mut Criterion) {
    let blocks = structured_blocks(2, 2);
    let job = DctGridToDwt97Job {
        blocks: &blocks,
        block_cols: 2,
        block_rows: 2,
        width: 16,
        height: 16,
    };
    let mut group = c.benchmark_group("dct97_metal_projection");

    group.bench_function("scalar_16x16", |b| {
        let mut scratch = Dct97GridScratch::default();
        b.iter(|| {
            black_box(
                dct8x8_blocks_to_dwt97_float_linear_with_scratch(
                    black_box(&blocks),
                    2,
                    2,
                    16,
                    16,
                    &mut scratch,
                )
                .expect("scalar 9/7 projection accepts fixture grid"),
            );
        });
    });

    if explicit_metal_accepts(job) {
        group.bench_function("metal_explicit_16x16", |b| {
            let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
            b.iter(|| {
                black_box(
                    accelerator
                        .dct_grid_to_dwt97(black_box(job))
                        .expect("explicit Metal 9/7 projection succeeds")
                        .expect("explicit Metal handles benchmark job"),
                );
            });
        });
    } else {
        eprintln!("skipping metal_explicit_16x16 benchmark: {METAL_UNAVAILABLE}");
    }

    group.finish();
}

fn explicit_metal_accepts(job: DctGridToDwt97Job<'_>) -> bool {
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
    matches!(accelerator.dct_grid_to_dwt97(job), Ok(Some(_)))
}

fn structured_blocks(block_cols: usize, block_rows: usize) -> Vec<[[f64; 8]; 8]> {
    let mut blocks = Vec::with_capacity(block_cols * block_rows);
    for block_y in 0..block_rows {
        for block_x in 0..block_cols {
            let mut block = [[0.0; 8]; 8];
            block[0][0] = 384.0 + (block_x * 19 + block_y * 23) as f64;
            block[0][1] = -17.0 + block_x as f64;
            block[1][0] = 11.0 - block_y as f64;
            block[2][3] = 7.0;
            block[4][4] = -3.0;
            block[7][7] = 2.0;
            blocks.push(block);
        }
    }
    blocks
}

criterion_group!(dct97_metal_projection, bench_dct97_projection);
criterion_main!(dct97_metal_projection);
