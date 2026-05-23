// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use signinum_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};
use signinum_transcode::accelerator::{
    DctGridToDwt53Job, DctGridToDwt97Job, DctGridToReversibleDwt53Job,
    DctToWaveletStageAccelerator, RayonReversibleDwt53Accelerator,
};
use signinum_transcode::dct53_2d::{
    dct8x8_blocks_to_dwt53_float_linear_with_scratch, Dct53GridScratch,
};
use signinum_transcode::dct97_2d::{
    dct8x8_blocks_to_dwt97_float_linear_with_scratch, Dct97GridScratch,
};
use signinum_transcode::{JpegToHtj2kCoefficientPath, JpegToHtj2kOptions, JpegToHtj2kTranscoder};
use signinum_transcode_metal::{MetalDctToWaveletStageAccelerator, METAL_UNAVAILABLE};

const WSI_DIMS: [usize; 4] = [224, 512, 1024, 2048];
const REVERSIBLE_BATCH_SIZES: [usize; 5] = [1, 8, 32, 128, 512];
const MAX_REVERSIBLE_BATCH_SAMPLES: usize = 512 * 512 * 512;

const DIRECT_BENCH_MARKERS: [&str; 8] = [
    "scalar_224x224",
    "metal_explicit_224x224",
    "scalar_512x512",
    "metal_explicit_512x512",
    "scalar_1024x1024",
    "metal_explicit_1024x1024",
    "scalar_2048x2048",
    "metal_explicit_2048x2048",
];

const REVERSIBLE_BENCH_MARKERS: [&str; 8] = [
    "rayon_224x224",
    "metal_explicit_224x224",
    "rayon_512x512",
    "metal_explicit_512x512",
    "rayon_1024x1024",
    "metal_explicit_1024x1024",
    "rayon_2048x2048",
    "metal_explicit_2048x2048",
];

const REVERSIBLE_BATCH_BENCH_MARKERS: [&str; 11] = [
    "reversible_dct53_batch_metal_projection",
    "batch_1",
    "batch_8",
    "batch_32",
    "batch_128",
    "batch_512",
    "rayon_224x224_batch_1",
    "metal_explicit_224x224_batch_1",
    "rayon_512x512_batch_512",
    "rayon_1024x1024_batch_128",
    "rayon_2048x2048_batch_32",
];

const WSI_FIXTURES: [WsiFixtureSpec; 12] = [
    WsiFixtureSpec {
        name: "p3_like_ybr444_224",
        dim: 224,
        subsampling: JpegSubsampling::Ybr444,
        generator: rgb_p3_like_pattern,
    },
    WsiFixtureSpec {
        name: "p3_like_ybr444_512",
        dim: 512,
        subsampling: JpegSubsampling::Ybr444,
        generator: rgb_p3_like_pattern,
    },
    WsiFixtureSpec {
        name: "p3_like_ybr444_1024",
        dim: 1024,
        subsampling: JpegSubsampling::Ybr444,
        generator: rgb_p3_like_pattern,
    },
    WsiFixtureSpec {
        name: "p3_like_ybr444_2048",
        dim: 2048,
        subsampling: JpegSubsampling::Ybr444,
        generator: rgb_p3_like_pattern,
    },
    WsiFixtureSpec {
        name: "srgb_ybr420_224",
        dim: 224,
        subsampling: JpegSubsampling::Ybr420,
        generator: rgb_srgb_pattern,
    },
    WsiFixtureSpec {
        name: "srgb_ybr420_512",
        dim: 512,
        subsampling: JpegSubsampling::Ybr420,
        generator: rgb_srgb_pattern,
    },
    WsiFixtureSpec {
        name: "srgb_ybr420_1024",
        dim: 1024,
        subsampling: JpegSubsampling::Ybr420,
        generator: rgb_srgb_pattern,
    },
    WsiFixtureSpec {
        name: "srgb_ybr420_2048",
        dim: 2048,
        subsampling: JpegSubsampling::Ybr420,
        generator: rgb_srgb_pattern,
    },
    WsiFixtureSpec {
        name: "ycbcr_like_ybr420_224",
        dim: 224,
        subsampling: JpegSubsampling::Ybr420,
        generator: rgb_ycbcr_like_pattern,
    },
    WsiFixtureSpec {
        name: "ycbcr_like_ybr420_512",
        dim: 512,
        subsampling: JpegSubsampling::Ybr420,
        generator: rgb_ycbcr_like_pattern,
    },
    WsiFixtureSpec {
        name: "ycbcr_like_ybr420_1024",
        dim: 1024,
        subsampling: JpegSubsampling::Ybr420,
        generator: rgb_ycbcr_like_pattern,
    },
    WsiFixtureSpec {
        name: "ycbcr_like_ybr420_2048",
        dim: 2048,
        subsampling: JpegSubsampling::Ybr420,
        generator: rgb_ycbcr_like_pattern,
    },
];

fn bench_dct97_projection(c: &mut Criterion) {
    black_box(DIRECT_BENCH_MARKERS);
    let mut group = c.benchmark_group("dct97_metal_projection");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for dim in WSI_DIMS {
        let block_cols = dim / 8;
        let block_rows = dim / 8;
        let blocks = structured_blocks(block_cols, block_rows);
        let job = DctGridToDwt97Job {
            blocks: &blocks,
            block_cols,
            block_rows,
            width: dim,
            height: dim,
        };
        group.throughput(Throughput::Elements((dim * dim) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("scalar_{dim}x{dim}")),
            &job,
            |b, job| {
                let mut scratch = Dct97GridScratch::default();
                b.iter(|| {
                    black_box(
                        dct8x8_blocks_to_dwt97_float_linear_with_scratch(
                            black_box(job.blocks),
                            job.block_cols,
                            job.block_rows,
                            job.width,
                            job.height,
                            &mut scratch,
                        )
                        .expect("scalar 9/7 projection accepts fixture grid"),
                    );
                });
            },
        );

        if explicit_metal_accepts(job) {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("metal_explicit_{dim}x{dim}")),
                &job,
                |b, job| {
                    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
                    b.iter(|| {
                        black_box(
                            accelerator
                                .dct_grid_to_dwt97(black_box(*job))
                                .expect("explicit Metal 9/7 projection succeeds")
                                .expect("explicit Metal handles benchmark job"),
                        );
                    });
                },
            );
        } else {
            eprintln!("skipping metal_explicit_{dim}x{dim} benchmark: {METAL_UNAVAILABLE}");
        }
    }

    group.finish();
}

fn bench_dct53_projection(c: &mut Criterion) {
    black_box(DIRECT_BENCH_MARKERS);
    let mut group = c.benchmark_group("dct53_metal_projection");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for dim in WSI_DIMS {
        let block_cols = dim / 8;
        let block_rows = dim / 8;
        let blocks = structured_blocks(block_cols, block_rows);
        let job = DctGridToDwt53Job {
            blocks: &blocks,
            block_cols,
            block_rows,
            width: dim,
            height: dim,
        };
        group.throughput(Throughput::Elements((dim * dim) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("scalar_{dim}x{dim}")),
            &job,
            |b, job| {
                let mut scratch = Dct53GridScratch::default();
                b.iter(|| {
                    black_box(
                        dct8x8_blocks_to_dwt53_float_linear_with_scratch(
                            black_box(job.blocks),
                            job.block_cols,
                            job.block_rows,
                            job.width,
                            job.height,
                            &mut scratch,
                        )
                        .expect("scalar 5/3 projection accepts fixture grid"),
                    );
                });
            },
        );

        if explicit_metal_accepts_53(job) {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("metal_explicit_{dim}x{dim}")),
                &job,
                |b, job| {
                    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
                    b.iter(|| {
                        black_box(
                            accelerator
                                .dct_grid_to_dwt53(black_box(*job))
                                .expect("explicit Metal 5/3 projection succeeds")
                                .expect("explicit Metal handles benchmark job"),
                        );
                    });
                },
            );
        } else {
            eprintln!("skipping metal_explicit_{dim}x{dim} benchmark: {METAL_UNAVAILABLE}");
        }
    }

    group.finish();
}

fn bench_reversible_dct53_projection(c: &mut Criterion) {
    black_box(REVERSIBLE_BENCH_MARKERS);
    let mut group = c.benchmark_group("reversible_dct53_metal_projection");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for dim in WSI_DIMS {
        let block_cols = dim / 8;
        let block_rows = dim / 8;
        let blocks = structured_i16_blocks(block_cols, block_rows);
        let job = DctGridToReversibleDwt53Job {
            dequantized_blocks: &blocks,
            block_cols,
            block_rows,
            width: dim,
            height: dim,
        };
        group.throughput(Throughput::Elements((dim * dim) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("rayon_{dim}x{dim}")),
            &job,
            |b, job| {
                let mut accelerator = RayonReversibleDwt53Accelerator::default();
                b.iter(|| {
                    black_box(
                        accelerator
                            .dct_grid_to_reversible_dwt53(black_box(*job))
                            .expect("rayon reversible 5/3 projection succeeds")
                            .expect("rayon handles benchmark job"),
                    );
                });
            },
        );

        if explicit_metal_accepts_reversible_53(job) {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("metal_explicit_{dim}x{dim}")),
                &job,
                |b, job| {
                    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
                    b.iter(|| {
                        black_box(
                            accelerator
                                .dct_grid_to_reversible_dwt53(black_box(*job))
                                .expect("explicit Metal reversible 5/3 projection succeeds")
                                .expect("explicit Metal handles benchmark job"),
                        );
                    });
                },
            );
        } else {
            eprintln!("skipping metal_explicit_{dim}x{dim} benchmark: {METAL_UNAVAILABLE}");
        }
    }

    group.finish();
}

fn bench_reversible_dct53_batch_projection(c: &mut Criterion) {
    black_box(REVERSIBLE_BATCH_BENCH_MARKERS);
    let mut group = c.benchmark_group("reversible_dct53_batch_metal_projection");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for dim in WSI_DIMS {
        for batch_size in REVERSIBLE_BATCH_SIZES {
            let total_samples = dim.saturating_mul(dim).saturating_mul(batch_size);
            if total_samples > MAX_REVERSIBLE_BATCH_SAMPLES {
                continue;
            }

            let block_cols = dim / 8;
            let block_rows = dim / 8;
            let batch_blocks: Vec<_> = (0..batch_size)
                .map(|idx| {
                    let offset =
                        i16::try_from(idx.saturating_mul(3)).expect("benchmark offset fits i16");
                    structured_i16_blocks_with_offset(block_cols, block_rows, offset)
                })
                .collect();
            let jobs: Vec<_> = batch_blocks
                .iter()
                .map(|blocks| DctGridToReversibleDwt53Job {
                    dequantized_blocks: blocks,
                    block_cols,
                    block_rows,
                    width: dim,
                    height: dim,
                })
                .collect();

            group.throughput(Throughput::Elements(total_samples as u64));

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("rayon_{dim}x{dim}_batch_{batch_size}")),
                &jobs,
                |b, jobs| {
                    let mut accelerator = RayonReversibleDwt53Accelerator::default();
                    b.iter(|| {
                        let mut outputs = Vec::with_capacity(jobs.len());
                        for job in jobs {
                            outputs.push(
                                accelerator
                                    .dct_grid_to_reversible_dwt53(black_box(*job))
                                    .expect("rayon reversible 5/3 batch item succeeds")
                                    .expect("rayon handles reversible 5/3 batch item"),
                            );
                        }
                        black_box(outputs);
                    });
                },
            );

            if explicit_metal_accepts_reversible_53_batch(&jobs) {
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!(
                        "metal_explicit_{dim}x{dim}_batch_{batch_size}"
                    )),
                    &jobs,
                    |b, jobs| {
                        let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
                        b.iter(|| {
                            black_box(
                                accelerator
                                    .dct_grid_to_reversible_dwt53_batch(black_box(jobs))
                                    .expect("explicit Metal reversible 5/3 batch succeeds")
                                    .expect("explicit Metal handles benchmark batch"),
                            );
                        });
                    },
                );
            } else {
                eprintln!(
                    "skipping metal_explicit_{dim}x{dim}_batch_{batch_size} benchmark: \
                     {METAL_UNAVAILABLE}"
                );
            }
        }
    }

    group.finish();
}

fn bench_jpeg_to_htj2k_wsi(c: &mut Criterion) {
    let mut group = c.benchmark_group("jpeg_to_htj2k_wsi_97");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for spec in WSI_FIXTURES {
        let jpeg = encoded_fixture(spec);
        group.throughput(Throughput::Bytes(jpeg.len() as u64));

        group.bench_with_input(BenchmarkId::new(spec.name, "scalar"), &jpeg, |b, jpeg| {
            let mut transcoder = JpegToHtj2kTranscoder::default();
            let options = JpegToHtj2kOptions::lossy_97();
            b.iter(|| {
                black_box(
                    transcoder
                        .transcode(black_box(jpeg), &options)
                        .expect("scalar JPEG to HTJ2K 9/7 transcode succeeds"),
                );
            });
        });

        if metal_available() {
            group.bench_with_input(
                BenchmarkId::new(spec.name, "metal_explicit"),
                &jpeg,
                |b, jpeg| {
                    let mut transcoder = JpegToHtj2kTranscoder::default();
                    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
                    let options = JpegToHtj2kOptions::lossy_97();
                    b.iter(|| {
                        black_box(
                            transcoder
                                .transcode_with_accelerator(
                                    black_box(jpeg),
                                    &options,
                                    &mut accelerator,
                                )
                                .expect("Metal JPEG to HTJ2K 9/7 transcode succeeds"),
                        );
                    });
                },
            );
        } else {
            eprintln!(
                "skipping {}/metal_explicit benchmark: {METAL_UNAVAILABLE}",
                spec.name
            );
        }
    }

    group.finish();
}

fn bench_jpeg_to_htj2k_wsi_53(c: &mut Criterion) {
    let mut group = c.benchmark_group("jpeg_to_htj2k_wsi_53");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for spec in WSI_FIXTURES {
        let jpeg = encoded_fixture(spec);
        group.throughput(Throughput::Bytes(jpeg.len() as u64));

        group.bench_with_input(BenchmarkId::new(spec.name, "scalar"), &jpeg, |b, jpeg| {
            let mut transcoder = JpegToHtj2kTranscoder::default();
            let options = JpegToHtj2kOptions {
                coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear53,
                ..JpegToHtj2kOptions::lossless_53()
            };
            b.iter(|| {
                black_box(
                    transcoder
                        .transcode(black_box(jpeg), &options)
                        .expect("scalar JPEG to HTJ2K 5/3 transcode succeeds"),
                );
            });
        });

        if metal_available() {
            group.bench_with_input(
                BenchmarkId::new(spec.name, "metal_explicit"),
                &jpeg,
                |b, jpeg| {
                    let mut transcoder = JpegToHtj2kTranscoder::default();
                    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
                    let options = JpegToHtj2kOptions {
                        coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear53,
                        ..JpegToHtj2kOptions::lossless_53()
                    };
                    b.iter(|| {
                        black_box(
                            transcoder
                                .transcode_with_accelerator(
                                    black_box(jpeg),
                                    &options,
                                    &mut accelerator,
                                )
                                .expect("Metal JPEG to HTJ2K 5/3 transcode succeeds"),
                        );
                    });
                },
            );
        } else {
            eprintln!(
                "skipping {}/metal_explicit benchmark: {METAL_UNAVAILABLE}",
                spec.name
            );
        }
    }

    group.finish();
}

fn bench_jpeg_to_htj2k_wsi_integer_53(c: &mut Criterion) {
    let mut group = c.benchmark_group("jpeg_to_htj2k_wsi_integer_53");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for spec in WSI_FIXTURES {
        let jpeg = encoded_fixture(spec);
        group.throughput(Throughput::Bytes(jpeg.len() as u64));

        group.bench_with_input(BenchmarkId::new(spec.name, "scalar"), &jpeg, |b, jpeg| {
            let mut transcoder = JpegToHtj2kTranscoder::default();
            let options = JpegToHtj2kOptions::lossless_53();
            b.iter(|| {
                black_box(
                    transcoder
                        .transcode(black_box(jpeg), &options)
                        .expect("scalar JPEG to HTJ2K IntegerDirect53 transcode succeeds"),
                );
            });
        });

        group.bench_with_input(BenchmarkId::new(spec.name, "rayon"), &jpeg, |b, jpeg| {
            let mut transcoder = JpegToHtj2kTranscoder::default();
            let mut accelerator = RayonReversibleDwt53Accelerator::default();
            let options = JpegToHtj2kOptions::lossless_53();
            b.iter(|| {
                black_box(
                    transcoder
                        .transcode_with_accelerator(black_box(jpeg), &options, &mut accelerator)
                        .expect("rayon JPEG to HTJ2K IntegerDirect53 transcode succeeds"),
                );
            });
        });

        if metal_available() {
            group.bench_with_input(
                BenchmarkId::new(spec.name, "metal_explicit"),
                &jpeg,
                |b, jpeg| {
                    let mut transcoder = JpegToHtj2kTranscoder::default();
                    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
                    let options = JpegToHtj2kOptions::lossless_53();
                    b.iter(|| {
                        black_box(
                            transcoder
                                .transcode_with_accelerator(
                                    black_box(jpeg),
                                    &options,
                                    &mut accelerator,
                                )
                                .expect("Metal JPEG to HTJ2K IntegerDirect53 transcode succeeds"),
                        );
                    });
                },
            );
        } else {
            eprintln!(
                "skipping {}/metal_explicit benchmark: {METAL_UNAVAILABLE}",
                spec.name
            );
        }
    }

    group.finish();
}

#[derive(Clone, Copy)]
struct WsiFixtureSpec {
    name: &'static str,
    dim: usize,
    subsampling: JpegSubsampling,
    generator: fn(usize) -> Vec<u8>,
}

fn encoded_fixture(spec: WsiFixtureSpec) -> Vec<u8> {
    let rgb = (spec.generator)(spec.dim);
    encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: spec.dim as u32,
            height: spec.dim as u32,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: spec.subsampling,
            restart_interval: Some((spec.dim / 8) as u16),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode benchmark JPEG fixture")
    .data
}

fn metal_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        metal::Device::system_default().is_some()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

fn rgb_srgb_pattern(dim: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(dim * dim * 3);
    for y in 0..dim {
        for x in 0..dim {
            data.push(((x * 5 + y * 3 + 17) & 0xff) as u8);
            data.push(((x * 2 + y * 7 + 41) & 0xff) as u8);
            data.push(((x * 11 + y * 13 + 73) & 0xff) as u8);
        }
    }
    data
}

fn rgb_p3_like_pattern(dim: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(dim * dim * 3);
    for y in 0..dim {
        for x in 0..dim {
            let radial = ((x ^ y) & 0xff) as u8;
            data.push(radial.saturating_add(32));
            data.push(((x * 9 + y * 5 + 19) & 0xff) as u8);
            data.push(((x * 3 + y * 15 + 97) & 0xff) as u8);
        }
    }
    data
}

fn rgb_ycbcr_like_pattern(dim: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(dim * dim * 3);
    for y in 0..dim {
        for x in 0..dim {
            let y_sample = i32::from(((x * 3 + y * 2 + ((x / 31 + y / 17) * 23)) & 0xff) as u8);
            let cb = i32::from((((x / 8) * 9 + y * 2 + 96) & 0xff) as u8) - 128;
            let cr = i32::from(((x * 2 + (y / 8) * 11 + 160) & 0xff) as u8) - 128;
            let r = y_sample + ((91_881 * cr) >> 16);
            let g = y_sample - ((22_554 * cb + 46_802 * cr) >> 16);
            let b = y_sample + ((116_130 * cb) >> 16);
            data.push(clamp_u8(r));
            data.push(clamp_u8(g));
            data.push(clamp_u8(b));
        }
    }
    data
}

fn clamp_u8(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).expect("clamped value fits in u8")
}

fn explicit_metal_accepts(job: DctGridToDwt97Job<'_>) -> bool {
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
    matches!(accelerator.dct_grid_to_dwt97(job), Ok(Some(_)))
}

fn explicit_metal_accepts_53(job: DctGridToDwt53Job<'_>) -> bool {
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
    matches!(accelerator.dct_grid_to_dwt53(job), Ok(Some(_)))
}

fn explicit_metal_accepts_reversible_53(job: DctGridToReversibleDwt53Job<'_>) -> bool {
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
    matches!(accelerator.dct_grid_to_reversible_dwt53(job), Ok(Some(_)))
}

fn explicit_metal_accepts_reversible_53_batch(jobs: &[DctGridToReversibleDwt53Job<'_>]) -> bool {
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
    matches!(
        accelerator.dct_grid_to_reversible_dwt53_batch(jobs),
        Ok(Some(_))
    )
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

fn structured_i16_blocks(block_cols: usize, block_rows: usize) -> Vec<[i16; 64]> {
    structured_i16_blocks_with_offset(block_cols, block_rows, 0)
}

fn structured_i16_blocks_with_offset(
    block_cols: usize,
    block_rows: usize,
    base_offset: i16,
) -> Vec<[i16; 64]> {
    let mut blocks = Vec::with_capacity(block_cols * block_rows);
    for block_y in 0..block_rows {
        for block_x in 0..block_cols {
            let mut block = [0i16; 64];
            let block_offset =
                i16::try_from(block_x * 19 + block_y * 23).expect("fixture offset fits i16");
            let x_offset = i16::try_from(block_x).expect("fixture x offset fits i16");
            let y_offset = i16::try_from(block_y).expect("fixture y offset fits i16");
            block[0] = 384 + base_offset + block_offset;
            block[1] = -17 + x_offset;
            block[8] = 11 - y_offset;
            block[19] = 7;
            block[36] = -3;
            block[63] = 2;
            blocks.push(block);
        }
    }
    blocks
}

criterion_group!(dct53_metal_projection, bench_dct53_projection);
criterion_group!(dct97_metal_projection, bench_dct97_projection);
criterion_group!(jpeg_to_htj2k_wsi_53, bench_jpeg_to_htj2k_wsi_53);
criterion_group!(
    reversible_dct53_metal_projection,
    bench_reversible_dct53_projection
);
criterion_group!(
    reversible_dct53_batch_metal_projection,
    bench_reversible_dct53_batch_projection
);
criterion_group!(
    jpeg_to_htj2k_wsi_integer_53,
    bench_jpeg_to_htj2k_wsi_integer_53
);
criterion_group!(jpeg_to_htj2k_wsi_97, bench_jpeg_to_htj2k_wsi);
criterion_main!(
    dct53_metal_projection,
    dct97_metal_projection,
    reversible_dct53_metal_projection,
    reversible_dct53_batch_metal_projection,
    jpeg_to_htj2k_wsi_53,
    jpeg_to_htj2k_wsi_integer_53,
    jpeg_to_htj2k_wsi_97
);
