// SPDX-License-Identifier: MIT OR Apache-2.0

//! A serial region decode on the streamed IDWT decodes each band of
//! code-block rows as the IDWT reaches it. It must match decoding Tier-1 over
//! whole windows first bit for bit, decode and skip the same code-blocks, and
//! store less.

use alloc::vec::Vec;

use crate::{
    encode, CpuDecodeParallelism, DecodeSettings, DecoderContext, EncodeOptions,
    EncodeProgressionOrder, Image,
};

pub(super) const WIDTH: u32 = 203;
pub(super) const HEIGHT: u32 = 157;
pub(super) const TILE: u32 = 64;

/// Deterministic noise, so every sub-band carries significant coefficients.
pub(super) fn noisy_pixels(components: u8) -> Vec<u8> {
    let mut state = 0x2545_f491_u32;
    (0..WIDTH * HEIGHT * u32::from(components))
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

struct RegionDecode {
    native: Vec<u8>,
    decoded_blocks: usize,
    skipped_blocks: usize,
    stored_coefficients: usize,
    peak_coefficients: usize,
}

fn decode(image: &Image<'_>, roi: (u32, u32, u32, u32), banded: bool) -> RegionDecode {
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
    context.tile_decode_context.whole_window_tier1 = !banded;
    let native = image
        .decode_native_region_with_context(roi, &mut context)
        .expect("decode banded Tier-1 region")
        .data;
    let counters = context.tile_decode_context.debug_counters;
    RegionDecode {
        native,
        decoded_blocks: counters.decoded_code_blocks,
        skipped_blocks: counters.skipped_code_blocks,
        stored_coefficients: counters.stored_coefficients,
        peak_coefficients: counters.workspace_peak.bytes().coefficients,
    }
}

pub(super) struct Case {
    pub(super) label: &'static str,
    pub(super) components: u8,
    pub(super) options: EncodeOptions,
}

pub(super) fn cases() -> Vec<Case> {
    let base = EncodeOptions {
        num_decomposition_levels: 4,
        ..EncodeOptions::default()
    };
    let mut cases = Vec::new();
    for (label, components, options) in [
        (
            "gray 5/3",
            1,
            EncodeOptions {
                reversible: true,
                ..base.clone()
            },
        ),
        (
            "rgb 9/7",
            3,
            EncodeOptions {
                reversible: false,
                ..base.clone()
            },
        ),
        (
            "rgb 5/3 tiled",
            3,
            EncodeOptions {
                reversible: true,
                tile_size: Some((TILE, TILE)),
                ..base.clone()
            },
        ),
        (
            "gray HT 9/7",
            1,
            EncodeOptions {
                reversible: false,
                use_ht_block_coding: true,
                ..base.clone()
            },
        ),
        (
            "rgb HT 5/3 tiled",
            3,
            EncodeOptions {
                reversible: true,
                use_ht_block_coding: true,
                tile_size: Some((TILE, TILE)),
                ..base.clone()
            },
        ),
        // 4x4 code-blocks in 8x8 precincts, three layers, RPCL without PLT.
        (
            "gray 4x4 blocks, precincts, layers",
            1,
            EncodeOptions {
                reversible: true,
                code_block_width_exp: 0,
                code_block_height_exp: 0,
                precinct_exponents: alloc::vec![(3, 3); 5],
                progression_order: EncodeProgressionOrder::Rpcl,
                num_layers: 3,
                ..base.clone()
            },
        ),
        // Wide, short code-blocks, LRCP with PLT.
        (
            "gray 32x8 blocks, PLT",
            1,
            EncodeOptions {
                reversible: false,
                code_block_width_exp: 3,
                code_block_height_exp: 1,
                progression_order: EncodeProgressionOrder::Lrcp,
                num_layers: 2,
                write_plt: true,
                ..base.clone()
            },
        ),
        // Narrow, tall code-blocks.
        (
            "rgb 8x64 blocks",
            3,
            EncodeOptions {
                reversible: true,
                code_block_width_exp: 1,
                code_block_height_exp: 4,
                ..base.clone()
            },
        ),
    ] {
        cases.push(Case {
            label,
            components,
            options,
        });
    }
    cases
}

#[test]
fn banded_tier1_matches_whole_window_tier1_bit_for_bit() {
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
        .expect("encode banded Tier-1 fixture");
        for reduction in [0_u8, 1, 2] {
            let image = Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                .expect("parse banded Tier-1 fixture");
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
                // The IDWT stops before the bands below a top strip, which
                // are decoded after it, and starts below the bands above a
                // bottom strip.
                (0, 0, width, 9),
                (0, height - 9, width, 9),
                (width - 1, height - 1, 1, 1),
            ];
            for roi in regions {
                let label = format!("{} reduction={reduction} roi={roi:?}", case.label);
                let whole = decode(&image, roi, false);
                let banded = decode(&image, roi, true);
                assert_eq!(banded.native, whole.native, "{label}");
                assert_eq!(banded.decoded_blocks, whole.decoded_blocks, "{label}");
                assert_eq!(banded.skipped_blocks, whole.skipped_blocks, "{label}");
                assert!(
                    banded.stored_coefficients <= whole.stored_coefficients,
                    "{label}: stored {} vs {}",
                    banded.stored_coefficients,
                    whole.stored_coefficients
                );
                // A band of short code-blocks is a small part of the window.
                if roi == large
                    && reduction == 0
                    && case.options.tile_size.is_none()
                    && case.options.code_block_height_exp <= 1
                {
                    assert!(
                        banded.peak_coefficients * 4 < whole.peak_coefficients,
                        "{label}: coefficients {} vs {}",
                        banded.peak_coefficients,
                        whole.peak_coefficients
                    );
                }
            }
        }
    }
}

/// Outcome of a serial region decode: the pixels' hash, or the error.
fn outcome(bytes: &[u8], strict: bool, roi: (u32, u32, u32, u32), banded: bool) -> String {
    let settings = DecodeSettings {
        strict,
        ..DecodeSettings::default()
    };
    let image = match Image::new(bytes, &settings) {
        Ok(image) => image,
        Err(error) => return format!("parse {error:?}"),
    };
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
    context.tile_decode_context.whole_window_tier1 = !banded;
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
fn banded_tier1_rejects_corrupt_code_blocks_as_whole_window_tier1_does() {
    // Small code-blocks give each component many bands, so two corrupt
    // blocks often fail in a different order than the whole-window path
    // decodes them, and the banded decode reruns to report its error.
    const SIZE: u32 = 48;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let pixels: Vec<u8> = (0..SIZE * SIZE * 3).map(|_| (next() >> 56) as u8).collect();
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
                let whole = outcome(&mutant, strict, roi, false);
                let banded = outcome(&mutant, strict, roi, true);
                assert_eq!(
                    banded, whole,
                    "ht={use_ht_block_coding} iteration={iteration} strict={strict} roi={roi:?}"
                );
                failures += usize::from(whole.starts_with("decode"));
            }
        }
        // The mutants must reach Tier-1 failures for the check to mean much.
        assert!(
            failures > 20,
            "ht={use_ht_block_coding}: {failures} failures"
        );
    }
}

#[test]
fn banded_tier1_keeps_decoding_when_the_row_index_does_not_fit() {
    let case = cases()
        .into_iter()
        .find(|case| case.label == "gray 4x4 blocks, precincts, layers")
        .expect("small-block case");
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
    .expect("encode small-block fixture");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
    let roi = (67, 52, 29, 17);
    let indexed = decode(&image, roi, true);

    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
    context.tile_decode_context.reject_band_index = true;
    let unindexed = image
        .decode_native_region_with_context(roi, &mut context)
        .expect("decode without row index")
        .data;
    let counters = context.tile_decode_context.debug_counters;
    assert_eq!(unindexed, indexed.native);
    assert_eq!(counters.decoded_code_blocks, indexed.decoded_blocks);
    assert_eq!(counters.skipped_code_blocks, indexed.skipped_blocks);
    assert_eq!(
        counters.workspace_peak.bytes().coefficients,
        indexed.peak_coefficients
    );
}
