// SPDX-License-Identifier: MIT OR Apache-2.0

//! An RGB8 decode whose one tile covers its three 8-bit components leaves
//! their last IDWT level to the pack, which reconstructs it a row at a time
//! with the inverse MCT. That must match the staged decode, which stores the
//! last level as planes first, bit for bit, at odd sizes and origins, for
//! both transforms, serial or parallel, and every other image must keep the
//! staged path.

use alloc::vec;
use alloc::vec::Vec;

use crate::{encode, CpuDecodeParallelism, DecodeSettings, DecoderContext, EncodeOptions, Image};

/// Deterministic noise, so every sub-band carries significant coefficients.
fn noisy_samples(len: u32) -> Vec<u8> {
    let mut state = 0x9e37_79b9_u32;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

struct Case {
    size: (u32, u32),
    components: u8,
    bit_depth: u8,
    levels: u8,
    tile: Option<(u32, u32)>,
    fused: bool,
}

const fn rgb(size: (u32, u32), levels: u8, fused: bool) -> Case {
    Case {
        size,
        components: 3,
        bit_depth: 8,
        levels,
        tile: None,
        fused,
    }
}

const CASES: [Case; 9] = [
    // Several parallel strips, odd widths and heights, and the fewest levels.
    rgb((520, 136), 5, true),
    // Coarse scratch can supply both serial and parallel pack rings.
    rgb((64, 2_048), 5, true),
    rgb((97, 88), 3, true),
    rgb((96, 75), 2, true),
    // A coefficient buffer that needs padding, a single level, and tiles.
    rgb((203, 157), 4, false),
    rgb((96, 75), 1, false),
    Case {
        tile: Some((128, 64)),
        ..rgb((256, 128), 4, false)
    },
    Case {
        components: 4,
        ..rgb((96, 80), 3, false)
    },
    Case {
        bit_depth: 12,
        ..rgb((96, 80), 3, false)
    },
];

struct Decoded {
    pixels: Vec<u8>,
    fused: bool,
    peak: usize,
}

fn decode(bytes: &[u8], staged: bool, parallelism: CpuDecodeParallelism) -> Decoded {
    let image = Image::new(bytes, &DecodeSettings::default()).expect("parse fused fixture");
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(parallelism);
    context.tile_decode_context.staged_rgb8_pack = staged;
    let pixels = image
        .decode_with_context(&mut context)
        .expect("decode fused fixture")
        .data;
    // A second decode reuses the first one's IDWT buffers, LL bands included.
    let mut into = vec![0; pixels.len()];
    crate::j2c::idwt::reset_zero_initialized_samples();
    image
        .decode_into(&mut into, &mut context)
        .expect("decode fused fixture into a buffer");
    assert_eq!(into, pixels);
    if parallelism == CpuDecodeParallelism::Serial {
        assert_eq!(crate::j2c::idwt::zero_initialized_samples(), 0);
    }
    let tile_ctx = &context.tile_decode_context;
    Decoded {
        pixels,
        // Only the fused pass keeps an LL band apart.
        fused: !tile_ctx.fused_ll_bands[0].is_empty(),
        peak: tile_ctx.debug_counters.workspace_peak.bytes().total(),
    }
}

fn assert_matches_staged(bytes: &[u8], fused: bool, label: &str) {
    for parallelism in [CpuDecodeParallelism::Auto, CpuDecodeParallelism::Serial] {
        let label = format!("{label} {parallelism:?}");
        let staged = decode(bytes, true, parallelism);
        let actual = decode(bytes, false, parallelism);
        assert!(!staged.fused, "{label}");
        assert_eq!(actual.fused, fused, "{label}");
        assert_eq!(actual.pixels, staged.pixels, "{label}");
        if fused {
            assert!(
                actual.peak < staged.peak,
                "{label}: peak {} vs {}",
                actual.peak,
                staged.peak
            );
        }
    }
}

#[test]
fn fused_last_level_matches_the_staged_decode() {
    for case in &CASES {
        let (width, height) = case.size;
        for reversible in [true, false] {
            let samples = noisy_samples(
                width * height * u32::from(case.components) * u32::from(case.bit_depth.div_ceil(8)),
            );
            let bytes = encode(
                &samples,
                width,
                height,
                u16::from(case.components),
                case.bit_depth,
                false,
                &EncodeOptions {
                    reversible,
                    num_decomposition_levels: case.levels,
                    tile_size: case.tile,
                    ..EncodeOptions::default()
                },
            )
            .expect("encode fused fixture");
            let label = format!(
                "{width}x{height} components={} bit_depth={} levels={} tile={:?} \
                 reversible={reversible}",
                case.components, case.bit_depth, case.levels, case.tile
            );
            assert_matches_staged(&bytes, case.fused, &label);
        }
    }
}

#[test]
fn fused_last_level_matches_at_an_odd_image_origin() {
    // See `fixtures/image_origin`.
    let fixtures: [(&str, &[u8]); 2] = [
        (
            "5/3 75x56 at (5, 3)",
            include_bytes!("../../fixtures/image_origin/openjpeg_rgb_75x56_origin5x3_53.j2k"),
        ),
        (
            "9/7 72x59 at (7, 1)",
            include_bytes!("../../fixtures/image_origin/openjpeg_rgb_72x59_origin7x1_97.j2k"),
        ),
    ];
    for (name, bytes) in fixtures {
        assert_matches_staged(bytes, true, name);
    }
}
