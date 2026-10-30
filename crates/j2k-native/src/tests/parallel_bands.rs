// SPDX-License-Identifier: MIT OR Apache-2.0

//! A parallel region decode on the streamed IDWT decodes its bands on the
//! worker pool ahead of the IDWT. It must match decoding Tier-1 over whole
//! windows bit for bit, decode and skip the same code-blocks, fail the same
//! way, store less, and finish when no pool thread is free to help.

use alloc::vec::Vec;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use super::banded_tier1::{cases, noisy_pixels, HEIGHT, TILE, WIDTH};
use crate::{encode, CpuDecodeParallelism, DecodeSettings, DecoderContext, EncodeOptions, Image};

struct RegionDecode {
    native: Vec<u8>,
    decoded_blocks: usize,
    skipped_blocks: usize,
    pool_components: usize,
    peak_coefficients: usize,
}

/// Decode `roi` with Auto parallelism on the current pool, in bands on the
/// pool unless `whole_windows`.
fn decode(image: &Image<'_>, roi: (u32, u32, u32, u32), whole_windows: bool) -> RegionDecode {
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(CpuDecodeParallelism::Auto);
    context.tile_decode_context.whole_window_tier1 = whole_windows;
    let native = image
        .decode_native_region_with_context(roi, &mut context)
        .expect("decode region")
        .data;
    let counters = context.tile_decode_context.debug_counters;
    RegionDecode {
        native,
        decoded_blocks: counters.decoded_code_blocks,
        skipped_blocks: counters.skipped_code_blocks,
        pool_components: counters.pool_band_components,
        peak_coefficients: counters.workspace_peak.bytes().coefficients,
    }
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build pool")
}

#[test]
fn pool_bands_match_whole_window_tier1_bit_for_bit() {
    for threads in [2, 4, 12] {
        let pool = pool(threads);
        for case in cases() {
            let pixels = noisy_pixels(case.components);
            let bytes = encode(
                &pixels,
                WIDTH,
                HEIGHT,
                u16::from(case.components),
                8,
                false,
                &case.options,
            )
            .expect("encode pool band fixture");
            let mut pool_components = 0;
            for reduction in [0_u8, 1, 2] {
                let image =
                    Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                        .expect("parse pool band fixture");
                let (width, height) = (image.width(), image.height());
                let tile = TILE >> reduction;
                let large = (
                    width / 20,
                    height / 20,
                    width - width / 10,
                    height - height / 10,
                );
                let regions = [
                    (0, 0, width, height),
                    large,
                    (width / 3, height / 3, width / 7, height / 9),
                    (tile + 1, tile + 2, tile - 3, tile - 4),
                    (tile - 5, tile - 3, 11, 7),
                    (0, 0, width, 9),
                    (0, height - 9, width, 9),
                    (width - 1, height - 1, 1, 1),
                ];
                for roi in regions {
                    let label = format!(
                        "{} threads={threads} reduction={reduction} roi={roi:?}",
                        case.label
                    );
                    let (whole, banded) =
                        pool.install(|| (decode(&image, roi, true), decode(&image, roi, false)));
                    assert_eq!(whole.pool_components, 0, "{label}");
                    assert_eq!(banded.native, whole.native, "{label}");
                    assert_eq!(banded.decoded_blocks, whole.decoded_blocks, "{label}");
                    assert_eq!(banded.skipped_blocks, whole.skipped_blocks, "{label}");
                    pool_components += banded.pool_components;
                    // A band of short code-blocks is a small part of the
                    // window.
                    if banded.pool_components > 0
                        && roi == large
                        && reduction == 0
                        && case.options.tile_size.is_none()
                        && case.options.code_block_height_exp <= 1
                    {
                        assert!(
                            banded.peak_coefficients < whole.peak_coefficients,
                            "{label}: coefficients {} vs {}",
                            banded.peak_coefficients,
                            whole.peak_coefficients
                        );
                    }
                }
            }
            // The route decodes HT blocks and classic blocks of one
            // arithmetic segment. The encoder terminates every pass when it
            // writes several layers, so those classic blocks keep whole
            // windows. Classic tiles with fewer rows of code-blocks than
            // threads do too, so only two threads reach every other case.
            let routed = case.options.use_ht_block_coding || case.options.num_layers <= 1;
            if !routed {
                assert_eq!(pool_components, 0, "{}", case.label);
            } else if threads == 2 {
                assert!(pool_components > 0, "{} never used the pool", case.label);
            }
        }
    }
}

fn rgb_fixture() -> Vec<u8> {
    let pixels = noisy_pixels(3);
    encode(
        &pixels,
        WIDTH,
        HEIGHT,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            num_decomposition_levels: 4,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode rgb fixture")
}

/// With the pool's other thread held by unrelated work, workers can't run
/// until the decode is over, so the IDWT must decode every band itself.
#[test]
fn pool_bands_finish_while_the_other_pool_thread_is_busy() {
    let bytes = rgb_fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse rgb fixture");
    let roi = (10, 12, 170, 130);
    let expected = pool(2).install(|| decode(&image, roi, true)).native;
    let pool = pool(2);
    let done = AtomicBool::new(false);
    let (started, running) = mpsc::channel();
    pool.in_place_scope(|scope| {
        scope.spawn(|_| {
            started.send(()).expect("signal start");
            while !done.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        });
        running.recv().expect("busy task started");
        let banded = pool.install(|| decode(&image, roi, false));
        done.store(true, Ordering::Release);
        assert!(banded.pool_components > 0);
        assert_eq!(banded.native, expected);
    });
}

/// Region decodes that run inside the pool, several at a time, each wait
/// only for bands another thread is decoding.
#[test]
fn concurrent_pool_band_decodes_inside_the_pool_finish() {
    use rayon::prelude::*;
    let bytes = rgb_fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse rgb fixture");
    let regions = [(0, 0, WIDTH, HEIGHT), (10, 12, 170, 130), (60, 40, 90, 70)];
    let expected: Vec<Vec<u8>> = regions
        .iter()
        .map(|roi| pool(2).install(|| decode(&image, *roi, true)).native)
        .collect();
    for threads in [2, 3] {
        let decoded: Vec<Vec<u8>> = pool(threads).install(|| {
            (0..12)
                .into_par_iter()
                .map(|index| decode(&image, regions[index % regions.len()], false).native)
                .collect()
        });
        for (index, native) in decoded.iter().enumerate() {
            assert_eq!(
                native,
                &expected[index % regions.len()],
                "threads={threads} decode={index}"
            );
        }
    }
}

/// Outcome of an Auto region decode on a two-thread pool: the pixels' hash,
/// or the error.
fn outcome(bytes: &[u8], strict: bool, roi: (u32, u32, u32, u32), whole_windows: bool) -> String {
    let settings = DecodeSettings {
        strict,
        ..DecodeSettings::default()
    };
    let image = match Image::new(bytes, &settings) {
        Ok(image) => image,
        Err(error) => return format!("parse {error:?}"),
    };
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(if whole_windows {
        CpuDecodeParallelism::Serial
    } else {
        CpuDecodeParallelism::Auto
    });
    context.tile_decode_context.whole_window_tier1 = whole_windows;
    match image.decode_region_with_context(roi, &mut context) {
        Ok(decoded) => {
            let hash = decoded
                .data
                .iter()
                .fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
                });
            format!("ok {hash:016x}")
        }
        Err(error) => format!("decode {error:?}"),
    }
}

#[test]
fn pool_bands_reject_corrupt_code_blocks_as_serial_whole_window_tier1_does() {
    const SIZE: u32 = 48;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let pixels: Vec<u8> = (0..SIZE * SIZE * 3).map(|_| (next() >> 56) as u8).collect();
    let pool = pool(2);
    for use_ht_block_coding in [false, true] {
        let bytes = encode(
            &pixels,
            SIZE,
            SIZE,
            3,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                code_block_width_exp: 0,
                code_block_height_exp: 0,
                num_layers: 2,
                use_ht_block_coding,
                ..EncodeOptions::default()
            },
        )
        .expect("encode corrupt-block fixture");
        let mut failures = 0;
        for iteration in 0..300 {
            let mut mutant = bytes.clone();
            for _ in 0..=next() % 4 {
                let span = u64::try_from(mutant.len() - 120).expect("fixture length");
                let at = 120 + usize::try_from(next() % span).expect("offset in the fixture");
                mutant[at] ^= 1 << (next() % 8);
            }
            for strict in [false, true] {
                let roi = [(0, 0, SIZE, SIZE), (30, 30, 18, 18), (5, 9, 13, 7)]
                    [usize::try_from(next() % 3).expect("region index")];
                let whole = outcome(&mutant, strict, roi, true);
                let banded = pool.install(|| outcome(&mutant, strict, roi, false));
                assert_eq!(
                    banded, whole,
                    "ht={use_ht_block_coding} iteration={iteration} strict={strict} roi={roi:?}"
                );
                failures += usize::from(whole.starts_with("decode"));
            }
        }
        assert!(
            failures > 20,
            "ht={use_ht_block_coding}: {failures} failures"
        );
    }
}

/// A band left running with no worker to finish it, as an earlier build's
/// fuzz once found, must not hang the decode: the component gives up the
/// pool and decodes over whole windows, with the same output.
#[test]
fn a_stranded_band_falls_back_to_whole_windows() {
    let bytes = rgb_fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse rgb fixture");
    let roi = (10, 12, 170, 130);
    let pool = pool(4);
    let expected = pool.install(|| decode(&image, roi, true)).native;
    let (native, counters) = pool.install(|| {
        let mut context = DecoderContext::default();
        context.set_cpu_decode_parallelism(CpuDecodeParallelism::Auto);
        context.tile_decode_context.strand_pool_band = true;
        let native = image
            .decode_native_region_with_context(roi, &mut context)
            .expect("decode with a stranded band")
            .data;
        (native, context.tile_decode_context.debug_counters)
    });
    assert_eq!(native, expected);
    assert_eq!(counters.pool_band_fallbacks, 1);
    assert_eq!(counters.pool_band_components, 0);
}
