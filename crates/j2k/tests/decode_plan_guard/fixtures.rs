// SPDX-License-Identifier: MIT OR Apache-2.0

//! The images the decode guard decodes: small generated codestreams, each
//! for a geometry or coding whose route or working memory differs from the
//! others, and the two one-sample-edge-tile images of the test fixtures.

use j2k_core::PixelFormat;
use j2k_native::{encode, EncodeOptions, EncodeProgressionOrder};
use j2k_test_support::wrap_jp2_rgba_codestream;

/// Which requests the guard runs on a fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Extent {
    /// Every request at every budget.
    Every,
    /// An image large enough to change the route policy's choice: the full
    /// image and half resolution at every budget (and 1 MiB), a region and
    /// the native components.
    Large,
}

pub(super) struct Fixture {
    /// Stable label: components and precision, size, wavelet and levels,
    /// and what else sets it apart.
    pub(super) name: &'static str,
    pub(super) bytes: Vec<u8>,
    pub(super) width: u32,
    pub(super) height: u32,
    /// The 8- and 16-bit outputs the image is decoded to.
    pub(super) formats: [PixelFormat; 2],
    pub(super) extent: Extent,
    /// Also decode on the wide pool, where classic tiles of several
    /// components reconstruct each component on its own reader.
    pub(super) wide_pool: bool,
}

/// Samples of `bit_depth` bits in one to three little-endian bytes each:
/// noise when `noise_bits` is the precision, otherwise a ramp with that
/// many bits of noise, which codes in fewer passes.
fn samples(size: (u32, u32), components: u16, bit_depth: u8, noise_bits: u8) -> Vec<u8> {
    let bytes = usize::from(bit_depth).div_ceil(8);
    let (width, height) = (size.0 as usize, size.1 as usize);
    let mask = (1_u32 << bit_depth) - 1;
    let noise_mask = (1_u32 << noise_bits) - 1;
    let mut state = 0x2545_f491_u32 ^ (u32::from(components) << 8) ^ u32::from(bit_depth);
    let mut out = Vec::with_capacity(width * height * usize::from(components) * bytes);
    for y in 0..height {
        for x in 0..width {
            for c in 0..usize::from(components) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "only the ramp's bits below the sample depth are kept: the value is masked below"
                )]
                let ramp = ((13 * x + 29 * y + 37 * c) << bit_depth.saturating_sub(8)) as u32;
                let value = if noise_bits >= bit_depth {
                    state
                } else {
                    ramp.wrapping_add(state & noise_mask)
                };
                out.extend_from_slice(&(value & mask).to_le_bytes()[..bytes]);
            }
        }
    }
    out
}

fn codestream(
    size: (u32, u32),
    components: u16,
    bit_depth: u8,
    noise_bits: u8,
    options: &EncodeOptions,
) -> Vec<u8> {
    let samples = samples(size, components, bit_depth, noise_bits);
    encode(
        &samples, size.0, size.1, components, bit_depth, false, options,
    )
    .expect("encode guard fixture")
}

/// `codestream` with its image and tile grid moved `by` reference-grid
/// samples right and down (SIZ `Xsiz`, `Ysiz`, `XOsiz`, `YOsiz`, `XTOsiz`,
/// `YTOsiz` grow by `by`). With `by` a multiple of the tile size and of
/// `2^levels`, every tile keeps its samples, code-blocks and coded data.
fn moved(mut codestream: Vec<u8>, by: u32) -> Vec<u8> {
    let siz = codestream
        .windows(2)
        .position(|marker| marker == [0xff, 0x51])
        .expect("SIZ marker");
    for field in [6, 10, 14, 18, 30, 34] {
        let at = siz + field;
        let value = u32::from_be_bytes(
            codestream[at..at + 4]
                .try_into()
                .expect("four-byte SIZ field"),
        );
        codestream[at..at + 4].copy_from_slice(&(value + by).to_be_bytes());
    }
    codestream
}

/// The code-block style byte of the codestream's COD marker segment.
fn code_block_style(codestream: &[u8]) -> u8 {
    let cod = codestream
        .windows(2)
        .position(|marker| marker == [0xff, 0x52])
        .expect("COD marker");
    // Marker, Lcod, Scod, SGcod (4 bytes), then levels, code-block width
    // and height ahead of the style.
    codestream[cod + 12]
}

fn levels(levels: u8, reversible: bool) -> EncodeOptions {
    EncodeOptions {
        num_decomposition_levels: levels,
        reversible,
        ..EncodeOptions::default()
    }
}

fn small(
    name: &'static str,
    size: (u32, u32),
    formats: [PixelFormat; 2],
    bytes: Vec<u8>,
) -> Fixture {
    Fixture {
        name,
        bytes,
        width: size.0,
        height: size.1,
        formats,
        extent: Extent::Every,
        wide_pool: false,
    }
}

const GRAY: [PixelFormat; 2] = [PixelFormat::Gray8, PixelFormat::Gray16];
const RGB: [PixelFormat; 2] = [PixelFormat::Rgb8, PixelFormat::Rgb16];

/// Untiled images: precisions, both wavelets, both block coders, zero
/// levels, alpha in a JP2 box and more than four components.
fn untiled() -> Vec<Fixture> {
    let s = (97, 71);
    let ht = |levels_: u8, reversible| EncodeOptions {
        use_ht_block_coding: true,
        ..levels(levels_, reversible)
    };
    vec![
        small(
            "gray8-64x48-53-l3",
            (64, 48),
            GRAY,
            codestream((64, 48), 1, 8, 8, &levels(3, true)),
        ),
        small(
            "gray8-97x71-97-l0",
            s,
            GRAY,
            codestream(s, 1, 8, 8, &levels(0, false)),
        ),
        small(
            "gray12-97x71-53-l3",
            s,
            GRAY,
            codestream(s, 1, 12, 12, &levels(3, true)),
        ),
        small(
            "gray8-97x71-ht97-l3",
            s,
            GRAY,
            codestream(s, 1, 8, 8, &ht(3, false)),
        ),
        Fixture {
            wide_pool: true,
            ..small(
                "rgb8-97x71-97-l3",
                s,
                RGB,
                codestream(s, 3, 8, 8, &levels(3, false)),
            )
        },
        small(
            "rgb8-97x71-ht53-l3",
            s,
            RGB,
            codestream(s, 3, 8, 8, &ht(3, true)),
        ),
        small(
            "rgb8-64x48-53-l0",
            (64, 48),
            RGB,
            codestream((64, 48), 3, 8, 8, &levels(0, true)),
        ),
        small(
            "rgb16-97x71-53-l3",
            s,
            RGB,
            codestream(s, 3, 16, 16, &levels(3, true)),
        ),
        small(
            "rgba8-64x48-97-l2-jp2",
            (64, 48),
            RGB,
            wrap_jp2_rgba_codestream(&codestream((64, 48), 4, 8, 8, &levels(2, false)), 64, 48, 8),
        ),
        small(
            "five8-64x48-53-l2",
            (64, 48),
            RGB,
            codestream((64, 48), 5, 8, 8, &levels(2, true)),
        ),
    ]
}

/// Tiled images: partial edge tiles, one-sample edge tiles, image and tile
/// offsets.
fn tiled() -> Vec<Fixture> {
    let s = (97, 71);
    let tiles = |levels_: u8, reversible, tile| EncodeOptions {
        tile_size: Some(tile),
        ..levels(levels_, reversible)
    };
    let edge_gray: &[u8] = include_bytes!(
        "../../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_gray_8x6_tile8x5_53.j2k"
    );
    let edge_rgb: &[u8] = include_bytes!(
        "../../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_rgb_6x6_tile5x5_97.j2k"
    );
    vec![
        Fixture {
            wide_pool: true,
            ..small(
                "rgb8-97x71-53-l3-t40x24",
                s,
                RGB,
                codestream(s, 3, 8, 8, &tiles(3, true, (40, 24))),
            )
        },
        small(
            "gray16-97x71-97-l3-t40x24",
            s,
            GRAY,
            codestream(s, 1, 16, 16, &tiles(3, false, (40, 24))),
        ),
        // The encoder caps the levels at zero: a tile one sample wide has
        // no decomposition.
        small(
            "rgb8-65x33-53-l0-t32-edge1",
            (65, 33),
            RGB,
            codestream((65, 33), 3, 8, 8, &tiles(3, true, (32, 32))),
        ),
        small(
            "gray8-8x6-53-l2-t8x5-edge1",
            (8, 6),
            GRAY,
            edge_gray.to_vec(),
        ),
        small("rgb8-6x6-97-l2-t5x5-edge1", (6, 6), RGB, edge_rgb.to_vec()),
        small(
            "rgb8-97x71-53-l2-t64-offset64",
            s,
            RGB,
            moved(codestream(s, 3, 8, 8, &tiles(2, true, (64, 64))), 64),
        ),
        small(
            "gray8-97x71-97-l2-t64-offset64",
            s,
            GRAY,
            moved(codestream(s, 1, 8, 8, &tiles(2, false, (64, 64))), 64),
        ),
    ]
}

/// Layered codestreams: classic layers terminate every pass, so their code
/// blocks carry one segment per pass; HT layers split every code block's
/// payload over packets, and a packet limit over tile-parts.
fn layered() -> Vec<Fixture> {
    const TERMINATE_EACH_PASS: u8 = 0x04;
    const HIGH_THROUGHPUT: u8 = 0x40;
    let s = (97, 71);
    let classic = codestream(
        s,
        3,
        8,
        8,
        &EncodeOptions {
            num_layers: 3,
            ..levels(3, true)
        },
    );
    assert_eq!(code_block_style(&classic), TERMINATE_EACH_PASS);
    let ht = codestream(
        s,
        3,
        8,
        8,
        &EncodeOptions {
            use_ht_block_coding: true,
            num_layers: 3,
            progression_order: EncodeProgressionOrder::Rpcl,
            write_plt: true,
            tile_part_packet_limit: Some(4),
            ..levels(3, false)
        },
    );
    assert_eq!(code_block_style(&ht), HIGH_THROUGHPUT);
    vec![
        small("rgb8-97x71-53-l3-layers3-termall", s, RGB, classic),
        small("rgb8-97x71-ht97-l3-layers3-rpcl-plt-parts", s, RGB, ht),
    ]
}

/// Images large enough for the pool paths: the two finest levels of a
/// 512 x 512 9/7 image reconstruct as one pass on the pool, and five
/// 256 x 256 tiles decode at once on a pool of four threads.
fn large() -> Vec<Fixture> {
    let large = |name, size: (u32, u32), components, bit_depth, options: &EncodeOptions| {
        let formats = if components == 1 { GRAY } else { RGB };
        Fixture {
            extent: Extent::Large,
            ..small(
                name,
                size,
                formats,
                codestream(size, components, bit_depth, 2, options),
            )
        }
    };
    let tiles = EncodeOptions {
        tile_size: Some((256, 256)),
        ..levels(3, true)
    };
    vec![
        large(
            "gray8-512x512-97-l2-paired",
            (512, 512),
            1,
            8,
            &levels(2, false),
        ),
        large("rgb8-1280x256-53-l3-t256", (1280, 256), 3, 8, &tiles),
        large("gray16-1280x256-53-l3-t256", (1280, 256), 1, 16, &tiles),
    ]
}

/// Every fixture, in the order of the table.
pub(super) fn fixtures() -> Vec<Fixture> {
    let mut all = untiled();
    all.extend(tiled());
    all.extend(layered());
    all.extend(large());
    all
}

/// Code-block styles the encoder cannot write yet, so the guard matrix does
/// not generate them. The committed fixtures in
/// `crates/j2k-native/src/tests/multi_segment.rs` exercise both styles' routes
/// and bytes on the tile route, pool bands and serial fallback, but do not
/// provide heap-peak baselines.
pub(super) const COMPLEMENTARY_STYLES: &[&str] = &["BYPASS", "RESET"];
