// SPDX-License-Identifier: Apache-2.0

mod common;

use std::num::NonZeroUsize;

use common::{
    centered_roi,
    classification::{should_bench_decode_rows_rgb, should_compare_full_frame, CorpusInputClass},
    jpeg_decoder_decode, jpeg_decoder_decode_batch_region_scaled, jpeg_decoder_decode_batch_scaled,
    jpeg_decoder_decode_region, jpeg_decoder_decode_region_scaled, jpeg_decoder_decode_scaled,
    jpeg_decoder_inspect, libjpeg_turbo_available, libjpeg_turbo_decode,
    libjpeg_turbo_decode_batch, libjpeg_turbo_decode_batch_region_scaled,
    libjpeg_turbo_decode_batch_scaled, libjpeg_turbo_decode_region,
    libjpeg_turbo_decode_region_scaled, libjpeg_turbo_decode_scaled, libjpeg_turbo_inspect,
    load_bench_inputs, output_geometry, signinum_decode, signinum_decode_region,
    signinum_decode_region_scaled, signinum_decode_reused, signinum_decode_rows,
    signinum_decode_scaled, signinum_decode_tile_batch_region_scaled,
    signinum_decode_tile_batch_scaled, signinum_decode_with_scratch, signinum_inspect, zune_decode,
    zune_decode_batch_region_scaled, zune_decode_batch_scaled, zune_decode_region,
    zune_decode_region_scaled, zune_decode_scaled, zune_inspect, DecodeMode,
    SigninumTileBatchRegionScaledRgbSession, SigninumTileBatchRgbOutputBuffers,
    SigninumTileBatchRgbScratch, SigninumTileBatchRgbSession, SigninumTileBatchScaledRgbSession,
    TurboJpegBatchRgbOutputBuffers, TurboJpegDecoder,
};
use criterion::{criterion_group, criterion_main, Criterion};
use signinum_jpeg::{Decoder, Downscale, ScratchPool, TileBatchOptions};

fn bench_compare(c: &mut Criterion) {
    let inputs = load_bench_inputs();

    let mut inspect = c.benchmark_group("inspect");
    for input in &inputs {
        inspect.bench_function(format!("signinum/{}", input.name), |b| {
            b.iter(|| signinum_inspect(&input.bytes));
        });
        inspect.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| jpeg_decoder_inspect(&input.bytes));
        });
        inspect.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_inspect(&input.bytes));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            inspect.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                b.iter(|| libjpeg_turbo_inspect(&mut turbo, &input.bytes));
            });
        }
    }
    inspect.finish();

    let mut decode_rgb = c.benchmark_group("decode_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && should_compare_full_frame(input.mode, input.input_class)
    }) {
        decode_rgb.bench_function(format!("signinum/{}", input.name), |b| {
            b.iter(|| signinum_decode(&input.bytes, DecodeMode::Rgb));
        });
        decode_rgb.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| jpeg_decoder_decode(&input.bytes));
        });
        decode_rgb.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_decode(&input.bytes, DecodeMode::Rgb));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            decode_rgb.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                b.iter(|| libjpeg_turbo_decode(&mut turbo, &input.bytes, DecodeMode::Rgb));
            });
        }
    }
    decode_rgb.finish();

    let mut decode_gray = c.benchmark_group("decode_gray");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Gray && should_compare_full_frame(input.mode, input.input_class)
    }) {
        decode_gray.bench_function(format!("signinum/{}", input.name), |b| {
            b.iter(|| signinum_decode(&input.bytes, DecodeMode::Gray));
        });
        decode_gray.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| jpeg_decoder_decode(&input.bytes));
        });
        decode_gray.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_decode(&input.bytes, DecodeMode::Gray));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            decode_gray.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                b.iter(|| libjpeg_turbo_decode(&mut turbo, &input.bytes, DecodeMode::Gray));
            });
        }
    }
    decode_gray.finish();

    let mut decode_reused_rgb = c.benchmark_group("decode_reused_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        let dec = Decoder::new(&input.bytes).expect("signinum decoder (reused-setup)");
        let (fmt, stride, len) = output_geometry(&dec, DecodeMode::Rgb);
        let mut out = vec![0u8; len];
        decode_reused_rgb.bench_function(format!("signinum_reused/{}", input.name), |b| {
            b.iter(|| signinum_decode_reused(&dec, &mut out, stride, fmt));
        });
    }
    decode_reused_rgb.finish();

    let mut decode_reused_gray = c.benchmark_group("decode_reused_gray");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Gray && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        let dec = Decoder::new(&input.bytes).expect("signinum decoder (reused-setup)");
        let (fmt, stride, len) = output_geometry(&dec, DecodeMode::Gray);
        let mut out = vec![0u8; len];
        decode_reused_gray.bench_function(format!("signinum_reused/{}", input.name), |b| {
            b.iter(|| signinum_decode_reused(&dec, &mut out, stride, fmt));
        });
    }
    decode_reused_gray.finish();

    let mut decode_scratch_rgb = c.benchmark_group("decode_scratch_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        let dec = Decoder::new(&input.bytes).expect("signinum decoder (scratch-setup)");
        let (fmt, stride, len) = output_geometry(&dec, DecodeMode::Rgb);
        let mut out = vec![0u8; len];
        let mut pool = ScratchPool::new();
        // Warm the pool once so iteration 1 pays zero allocation cost.
        signinum_decode_with_scratch(&dec, &mut pool, &mut out, stride, fmt);
        decode_scratch_rgb.bench_function(format!("signinum_scratch/{}", input.name), |b| {
            b.iter(|| signinum_decode_with_scratch(&dec, &mut pool, &mut out, stride, fmt));
        });
    }
    decode_scratch_rgb.finish();

    let mut decode_scratch_gray = c.benchmark_group("decode_scratch_gray");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Gray && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        let dec = Decoder::new(&input.bytes).expect("signinum decoder (scratch-setup)");
        let (fmt, stride, len) = output_geometry(&dec, DecodeMode::Gray);
        let mut out = vec![0u8; len];
        let mut pool = ScratchPool::new();
        signinum_decode_with_scratch(&dec, &mut pool, &mut out, stride, fmt);
        decode_scratch_gray.bench_function(format!("signinum_scratch/{}", input.name), |b| {
            b.iter(|| signinum_decode_with_scratch(&dec, &mut pool, &mut out, stride, fmt));
        });
    }
    decode_scratch_gray.finish();

    // CPU-first JPEG proving groups start here. These WSI-shaped benches are
    // the acceptance contract for Apple Silicon and other non-Metal hosts.
    let mut decode_rows_rgb = c.benchmark_group("decode_rows_rgb");
    for input in inputs
        .iter()
        .filter(|input| should_bench_decode_rows_rgb(input.mode, input.input_class))
    {
        decode_rows_rgb.bench_function(format!("signinum/{}", input.name), |b| {
            b.iter(|| signinum_decode_rows(&input.bytes));
        });
    }
    decode_rows_rgb.finish();

    let batch_size = wsi_tile_batch_size();
    let mut wsi_tile_batch_rgb = c.benchmark_group("wsi_tile_batch_rgb");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb) {
        let bytes = &input.bytes;
        let mut signinum_batch = SigninumTileBatchRgbScratch::new(bytes, batch_size);
        wsi_tile_batch_rgb.bench_function(format!("signinum/{}", input.name), move |b| {
            b.iter(|| signinum_batch.run(bytes));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            wsi_tile_batch_rgb.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                b.iter(|| libjpeg_turbo_decode_batch(&mut turbo, &input.bytes, batch_size));
            });
        }
    }
    wsi_tile_batch_rgb.finish();

    let mut wsi_tile_batch_session_rgb = c.benchmark_group("wsi_tile_batch_session_rgb");
    for batch_size in [16usize, 64, 256] {
        for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb) {
            let bytes = &input.bytes;
            let mut current_batch = SigninumTileBatchRgbScratch::new(bytes, batch_size);
            wsi_tile_batch_session_rgb.bench_function(
                format!("current_free_batch/{}/{}", batch_size, input.name),
                move |b| b.iter(|| current_batch.run(bytes)),
            );

            for workers in [1usize, 2, 4] {
                let bytes = &input.bytes;
                let options = TileBatchOptions {
                    workers: NonZeroUsize::new(workers),
                };
                let mut fixed_worker_batch =
                    SigninumTileBatchRgbScratch::new_with_options(bytes, batch_size, options);
                wsi_tile_batch_session_rgb.bench_function(
                    format!(
                        "current_free_batch_workers_{workers}/{}/{}",
                        batch_size, input.name
                    ),
                    move |b| b.iter(|| fixed_worker_batch.run(bytes)),
                );
            }

            let bytes = &input.bytes;
            let mut session_batch = SigninumTileBatchRgbSession::new(bytes, batch_size);
            wsi_tile_batch_session_rgb.bench_function(
                format!("warm_session/{}/{}", batch_size, input.name),
                move |b| b.iter(|| session_batch.run(bytes)),
            );

            let bytes = &input.bytes;
            let mut output_session = SigninumTileBatchRgbOutputBuffers::new(bytes, batch_size);
            wsi_tile_batch_session_rgb.bench_function(
                format!("warm_session_output_buffers/{}/{}", batch_size, input.name),
                move |b| b.iter(|| output_session.run(bytes)),
            );

            if libjpeg_turbo_available() {
                let bytes = &input.bytes;
                let mut turbo_output_session =
                    TurboJpegBatchRgbOutputBuffers::new(bytes, batch_size);
                wsi_tile_batch_session_rgb.bench_function(
                    format!("libjpeg-turbo_prealloc/{}/{}", batch_size, input.name),
                    move |b| b.iter(|| turbo_output_session.run(bytes)),
                );
            }

            let bytes = &input.bytes;
            wsi_tile_batch_session_rgb.bench_function(
                format!("current_scaled_q4/{}/{}", batch_size, input.name),
                move |b| {
                    b.iter(|| {
                        signinum_decode_tile_batch_scaled(bytes, batch_size, Downscale::Quarter);
                    });
                },
            );

            let bytes = &input.bytes;
            let mut scaled_session =
                SigninumTileBatchScaledRgbSession::new(bytes, batch_size, Downscale::Quarter);
            wsi_tile_batch_session_rgb.bench_function(
                format!("warm_session_scaled_q4/{}/{}", batch_size, input.name),
                move |b| b.iter(|| scaled_session.run(bytes)),
            );

            let bytes = &input.bytes;
            wsi_tile_batch_session_rgb.bench_function(
                format!("current_region_scaled_q4/{}/{}", batch_size, input.name),
                move |b| {
                    b.iter(|| {
                        signinum_decode_tile_batch_region_scaled(
                            bytes,
                            batch_size,
                            256,
                            Downscale::Quarter,
                        );
                    });
                },
            );

            let bytes = &input.bytes;
            let mut region_scaled_session = SigninumTileBatchRegionScaledRgbSession::new(
                bytes,
                batch_size,
                256,
                Downscale::Quarter,
            );
            wsi_tile_batch_session_rgb.bench_function(
                format!(
                    "warm_session_region_scaled_q4/{}/{}",
                    batch_size, input.name
                ),
                move |b| b.iter(|| region_scaled_session.run(bytes)),
            );
        }
    }
    wsi_tile_batch_session_rgb.finish();

    let mut wsi_region_rgb = c.benchmark_group("wsi_region_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        let roi = centered_roi(input.dimensions, 256);
        wsi_region_rgb.bench_function(format!("signinum/{}", input.name), |b| {
            b.iter(|| signinum_decode_region(&input.bytes, 256));
        });
        wsi_region_rgb.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| jpeg_decoder_decode_region(&input.bytes, 256));
        });
        wsi_region_rgb.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_decode_region(&input.bytes, 256));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            wsi_region_rgb.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                b.iter(|| libjpeg_turbo_decode_region(&mut turbo, &input.bytes, roi));
            });
        }
    }
    wsi_region_rgb.finish();

    for (group_name, scale) in [
        ("wsi_scaled_rgb_q4", Downscale::Quarter),
        ("wsi_scaled_rgb_q8", Downscale::Eighth),
    ] {
        let mut group = c.benchmark_group(group_name);
        for input in inputs.iter().filter(|input| {
            input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
        }) {
            group.bench_function(format!("signinum/{}", input.name), |b| {
                b.iter(|| signinum_decode_scaled(&input.bytes, scale));
            });
            group.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
                b.iter(|| jpeg_decoder_decode_scaled(&input.bytes, scale));
            });
            group.bench_function(format!("zune-jpeg/{}", input.name), |b| {
                b.iter(|| zune_decode_scaled(&input.bytes, scale));
            });
            if libjpeg_turbo_available() {
                let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
                group.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                    b.iter(|| {
                        libjpeg_turbo_decode_scaled(&mut turbo, &input.bytes, scale);
                    });
                });
            }
        }
        group.finish();
    }

    for (group_name, scale) in [
        ("wsi_region_scaled_rgb_q4", Downscale::Quarter),
        ("wsi_region_scaled_rgb_q8", Downscale::Eighth),
    ] {
        let mut group = c.benchmark_group(group_name);
        for input in inputs.iter().filter(|input| {
            input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
        }) {
            let roi = centered_roi(input.dimensions, 256);
            group.bench_function(format!("signinum/{}", input.name), |b| {
                b.iter(|| signinum_decode_region_scaled(&input.bytes, 256, scale));
            });
            group.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
                b.iter(|| {
                    jpeg_decoder_decode_region_scaled(&input.bytes, 256, scale);
                });
            });
            group.bench_function(format!("zune-jpeg/{}", input.name), |b| {
                b.iter(|| zune_decode_region_scaled(&input.bytes, 256, scale));
            });
            if libjpeg_turbo_available() {
                let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
                group.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                    b.iter(|| {
                        libjpeg_turbo_decode_region_scaled(&mut turbo, &input.bytes, roi, scale);
                    });
                });
            }
        }
        group.finish();
    }

    let mut wsi_tile_batch_scaled_rgb_q4 = c.benchmark_group("wsi_tile_batch_scaled_rgb_q4");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb) {
        wsi_tile_batch_scaled_rgb_q4.bench_function(format!("signinum/{}", input.name), |b| {
            b.iter(|| {
                signinum_decode_tile_batch_scaled(&input.bytes, 64, Downscale::Quarter);
            });
        });
        wsi_tile_batch_scaled_rgb_q4.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| jpeg_decoder_decode_batch_scaled(&input.bytes, 64, Downscale::Quarter));
        });
        wsi_tile_batch_scaled_rgb_q4.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_decode_batch_scaled(&input.bytes, 64, Downscale::Quarter));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            wsi_tile_batch_scaled_rgb_q4.bench_function(
                format!("libjpeg-turbo/{}", input.name),
                move |b| {
                    b.iter(|| {
                        libjpeg_turbo_decode_batch_scaled(
                            &mut turbo,
                            &input.bytes,
                            64,
                            Downscale::Quarter,
                        );
                    });
                },
            );
        }
    }
    wsi_tile_batch_scaled_rgb_q4.finish();

    let mut wsi_tile_batch_region_scaled_rgb_q4 =
        c.benchmark_group("wsi_tile_batch_region_scaled_rgb_q4");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb) {
        let roi = centered_roi(input.dimensions, 256);
        wsi_tile_batch_region_scaled_rgb_q4.bench_function(
            format!("signinum/{}", input.name),
            |b| {
                b.iter(|| {
                    signinum_decode_tile_batch_region_scaled(
                        &input.bytes,
                        64,
                        256,
                        Downscale::Quarter,
                    );
                });
            },
        );
        wsi_tile_batch_region_scaled_rgb_q4.bench_function(
            format!("jpeg-decoder/{}", input.name),
            |b| {
                b.iter(|| {
                    jpeg_decoder_decode_batch_region_scaled(
                        &input.bytes,
                        64,
                        256,
                        Downscale::Quarter,
                    );
                });
            },
        );
        wsi_tile_batch_region_scaled_rgb_q4.bench_function(
            format!("zune-jpeg/{}", input.name),
            |b| {
                b.iter(|| {
                    zune_decode_batch_region_scaled(&input.bytes, 64, 256, Downscale::Quarter);
                });
            },
        );
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            wsi_tile_batch_region_scaled_rgb_q4.bench_function(
                format!("libjpeg-turbo/{}", input.name),
                move |b| {
                    b.iter(|| {
                        libjpeg_turbo_decode_batch_region_scaled(
                            &mut turbo,
                            &input.bytes,
                            64,
                            roi,
                            Downscale::Quarter,
                        );
                    });
                },
            );
        }
    }
    wsi_tile_batch_region_scaled_rgb_q4.finish();
}

fn wsi_tile_batch_size() -> usize {
    std::env::var("SIGNINUM_JPEG_TILE_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(64)
}

criterion_group!(compare_benches, bench_compare);
criterion_main!(compare_benches);
