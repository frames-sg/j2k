// SPDX-License-Identifier: MIT OR Apache-2.0

//! An 8-bit full-image decode whose component planes would pass the decode
//! budget, or for which streaming is the faster route, runs as a region
//! decode of the whole image that streams its rows (several tiles at once
//! when it has them and the pool to run them). A decoder whose planes budget
//! is fixed takes either route on demand, and the streamed route must write
//! the bytes the planes route writes, at every scale, stride and
//! parallelism, and fail where it fails.

use j2k::{CpuDecodeParallelism, J2kDecoder, J2kScratchPool};
use j2k_core::{Downscale, PixelFormat};
use j2k_native::{encode, EncodeOptions, EncodeProgressionOrder};
use j2k_test_support::{
    add_packet_length, edit_tile_parts, with_byte_after_last_packets, with_surplus_packet_lengths,
    wrap_jp2_rgba_codestream,
};

const WIDTH: u32 = 203;
const HEIGHT: u32 = 157;
const PADDING: u8 = 0xab;

fn noisy_pixels(components: u32) -> Vec<u8> {
    let mut state = 0x2545_f491_u32;
    (0..WIDTH * HEIGHT * components)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

fn codestream(components: u16, options: &EncodeOptions) -> Vec<u8> {
    let pixels = noisy_pixels(u32::from(components));
    encode(&pixels, WIDTH, HEIGHT, components, 8, false, options).expect("encode fixture")
}

/// The image at `scale` in a buffer padded past each row, or `None` when the
/// decode fails, from a decoder whose planes budget is `planes_budget` when
/// one is given.
fn decode(
    bytes: &[u8],
    planes_budget: Option<usize>,
    parallelism: CpuDecodeParallelism,
    fmt: PixelFormat,
    scale: Downscale,
    padding: usize,
) -> Option<Vec<u8>> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    decoder.set_cpu_decode_parallelism(parallelism);
    if let Some(bytes) = planes_budget {
        decoder.set_full_decode_planes_budget(bytes);
    }
    let (width, height) = decoder.info().dimensions;
    let divisor = scale.denominator();
    let (width, height) = (width.div_ceil(divisor), height.div_ceil(divisor));
    let row = width as usize * fmt.bytes_per_pixel();
    let stride = row + padding;
    let mut out = vec![PADDING; stride * height as usize];
    let pool = &mut J2kScratchPool::new();
    let result = match scale {
        Downscale::None => decoder.decode_into(&mut out, stride, fmt),
        _ => decoder.decode_scaled_into(pool, &mut out, stride, fmt, scale),
    };
    result.ok()?;
    for line in out.chunks(stride) {
        assert!(
            line[row..].iter().all(|&byte| byte == PADDING),
            "stride padding untouched"
        );
    }
    Some(out)
}

#[test]
fn streamed_full_image_matches_planes() {
    let base = EncodeOptions {
        num_decomposition_levels: 4,
        ..EncodeOptions::default()
    };
    let rgb = codestream(
        3,
        &EncodeOptions {
            reversible: false,
            ..base.clone()
        },
    );
    let rgb_tiled_ht = codestream(
        3,
        &EncodeOptions {
            reversible: true,
            tile_size: Some((64, 64)),
            use_ht_block_coding: true,
            ..base.clone()
        },
    );
    // Enough tiles for every worker of a large pool: decoded tiles at once.
    let rgb_many_tiles = codestream(
        3,
        &EncodeOptions {
            reversible: false,
            tile_size: Some((16, 16)),
            ..base.clone()
        },
    );
    let gray = codestream(
        1,
        &EncodeOptions {
            reversible: true,
            ..base.clone()
        },
    );
    let gray_tiled = codestream(
        1,
        &EncodeOptions {
            reversible: false,
            tile_size: Some((96, 64)),
            ..base.clone()
        },
    );
    let rgba = wrap_jp2_rgba_codestream(
        &codestream(
            4,
            &EncodeOptions {
                reversible: false,
                ..base
            },
        ),
        WIDTH,
        HEIGHT,
        8,
    );
    let (rgb_formats, gray_formats) = (
        &[PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Gray8][..],
        &[PixelFormat::Gray8, PixelFormat::Rgb8][..],
    );
    let fixtures = [
        ("rgb", rgb, rgb_formats),
        ("rgb tiled HT", rgb_tiled_ht, rgb_formats),
        ("rgb many tiles", rgb_many_tiles, rgb_formats),
        ("gray", gray, gray_formats),
        ("gray tiled", gray_tiled, gray_formats),
        ("rgba", rgba, rgb_formats),
    ];
    let mut decoded = 0;
    for (label, bytes, formats) in &fixtures {
        for &fmt in *formats {
            for scale in [Downscale::None, Downscale::Half, Downscale::Eighth] {
                for (parallelism, padding) in [
                    (CpuDecodeParallelism::Serial, 5),
                    (CpuDecodeParallelism::Auto, 0),
                ] {
                    let planes = decode(bytes, Some(usize::MAX), parallelism, fmt, scale, padding);
                    let streamed = decode(bytes, Some(0), parallelism, fmt, scale, padding);
                    let default = decode(bytes, None, parallelism, fmt, scale, padding);
                    assert!(
                        planes == streamed && planes == default,
                        "{label} {fmt:?} {scale:?} {parallelism:?}: planes {} streamed {} default {}",
                        planes.is_some(),
                        streamed.is_some(),
                        default.is_some()
                    );
                    decoded += usize::from(planes.is_some());
                }
            }
        }
    }
    // Gray to RGB and RGB to gray are rejected; the rest must decode.
    assert!(decoded >= 9 * 3 * 2, "{decoded} decodes succeeded");
}

/// Tiles whose coarse resolutions are empty while finer ones are not (a
/// one-sample tile row or column at an odd position, with two levels)
/// stream past a lowered budget like any other, and the streamed route
/// writes what the planes route writes.
#[test]
fn tiles_with_empty_coarse_levels_stream_as_the_planes_decode() {
    let gray: &[u8] = include_bytes!(
        "../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_gray_8x6_tile8x5_53.j2k"
    );
    let rgb: &[u8] = include_bytes!(
        "../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_rgb_6x6_tile5x5_97.j2k"
    );
    for (label, bytes, fmt) in [
        ("gray 5/3", gray, PixelFormat::Gray8),
        ("rgb 9/7", rgb, PixelFormat::Rgb8),
    ] {
        for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
            for scale in [Downscale::None, Downscale::Half] {
                let planes = decode(bytes, Some(usize::MAX), parallelism, fmt, scale, 0);
                let lowered = decode(bytes, Some(0), parallelism, fmt, scale, 0);
                let default = decode(bytes, None, parallelism, fmt, scale, 0);
                assert!(planes.is_some(), "{label} {parallelism:?} {scale:?}");
                assert_eq!(planes, lowered, "{label} {parallelism:?} {scale:?}");
                assert_eq!(planes, default, "{label} {parallelism:?} {scale:?}");
            }
        }
    }
}

/// A decoder that streamed one full-image decode decodes the next on the
/// planes, and the other way round, from the context it kept.
#[test]
fn routes_alternate_on_one_decoder() {
    let bytes = codestream(
        3,
        &EncodeOptions {
            num_decomposition_levels: 3,
            reversible: false,
            ..EncodeOptions::default()
        },
    );
    let fmt = PixelFormat::Rgb8;
    let stride = WIDTH as usize * 3;
    let expected = decode(
        &bytes,
        None,
        CpuDecodeParallelism::Auto,
        fmt,
        Downscale::None,
        0,
    )
    .expect("planes decode");
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    for budget in [usize::MAX, 0, usize::MAX, 0] {
        decoder.set_full_decode_planes_budget(budget);
        let mut out = vec![0; stride * HEIGHT as usize];
        decoder
            .decode_into(&mut out, stride, fmt)
            .unwrap_or_else(|error| panic!("budget {budget}: {error:?}"));
        assert!(out == expected, "budget {budget}");
    }
}

/// Data a tile-part holds after its tile's last packet is no packet's: bytes
/// after it, or PLT lengths past it, change no sample. Every route decodes
/// such a codestream as it decodes the codestream without them, on a pool of
/// one thread and of four (a half-resolution decode of the 512 x 512 RGB image
/// streams on 2 to 11 pool threads and keeps the planes on one).
#[test]
fn data_after_a_tiles_last_packet_decodes_as_without_it_on_every_route() {
    const SIDE: u32 = 512;
    let rgb_pixels: Vec<u8> = (0..SIDE * SIDE * 3)
        .map(|index| {
            let (pixel, component) = (index / 3, index % 3);
            ((pixel % SIDE) * (component + 1) + (pixel / SIDE) * 3).to_le_bytes()[0]
        })
        .collect();
    let rgb = encode(
        &rgb_pixels,
        SIDE,
        SIDE,
        3,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 3,
            reversible: false,
            ..EncodeOptions::default()
        },
    )
    .expect("encode 512 x 512 fixture");
    let tiled = EncodeOptions {
        num_decomposition_levels: 2,
        num_layers: 2,
        progression_order: EncodeProgressionOrder::Rpcl,
        tile_size: Some((64, 64)),
        write_plt: true,
        ..EncodeOptions::default()
    };
    let gray = codestream(1, &tiled);
    let rgb_ht = codestream(
        3,
        &EncodeOptions {
            use_ht_block_coding: true,
            ..tiled
        },
    );
    let rgb_formats = &[PixelFormat::Rgb8, PixelFormat::Rgb16][..];
    let cases = [
        (
            "rgb, byte after",
            &rgb,
            with_byte_after_last_packets(&rgb),
            rgb_formats,
        ),
        (
            "gray tiled PLT, byte after",
            &gray,
            with_byte_after_last_packets(&gray),
            &[PixelFormat::Gray8][..],
        ),
        (
            "gray tiled PLT, surplus lengths",
            &gray,
            with_surplus_packet_lengths(&gray),
            &[PixelFormat::Gray8][..],
        ),
        (
            "rgb tiled HT PLT, both",
            &rgb_ht,
            with_byte_after_last_packets(&with_surplus_packet_lengths(&rgb_ht)),
            rgb_formats,
        ),
    ];
    let pools = [1, 4].map(|threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
    });
    for (label, clean, edited, formats) in &cases {
        for &fmt in *formats {
            for scale in [Downscale::None, Downscale::Half] {
                let expected = decode(
                    clean,
                    Some(usize::MAX),
                    CpuDecodeParallelism::Serial,
                    fmt,
                    scale,
                    0,
                )
                .expect("clean fixture decodes");
                for (pool, threads) in pools.iter().zip([1, 4]) {
                    for budget in [Some(usize::MAX), Some(0), None] {
                        let decoded = pool.install(|| {
                            decode(edited, budget, CpuDecodeParallelism::Auto, fmt, scale, 0)
                        });
                        assert!(
                            decoded.as_ref() == Some(&expected),
                            "{label} {fmt:?} {scale:?} {threads} threads, planes budget {budget:?}: {}",
                            if decoded.is_some() { "other bytes" } else { "failed" }
                        );
                    }
                }
            }
        }
    }
}

/// A PLT length past the last packet of a tile-part while the tile's packets
/// go on in the next one disagrees with the packets: every route fails.
#[test]
fn surplus_packet_length_before_a_later_tile_part_fails_on_every_route() {
    let split = codestream(
        1,
        &EncodeOptions {
            num_decomposition_levels: 2,
            num_layers: 2,
            tile_part_packet_limit: Some(1),
            write_plt: true,
            ..EncodeOptions::default()
        },
    );
    let mut first = true;
    let edited = edit_tile_parts(&split, |part| {
        if std::mem::take(&mut first) {
            add_packet_length(part);
        }
    });
    let pools = [1, 4].map(|threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
    });
    for scale in [Downscale::None, Downscale::Half] {
        let fmt = PixelFormat::Gray8;
        assert!(decode(&split, None, CpuDecodeParallelism::Serial, fmt, scale, 0).is_some());
        for (pool, threads) in pools.iter().zip([1, 4]) {
            for budget in [Some(usize::MAX), Some(0), None] {
                let decoded = pool
                    .install(|| decode(&edited, budget, CpuDecodeParallelism::Auto, fmt, scale, 0));
                assert!(
                    decoded.is_none(),
                    "{scale:?} {threads} threads, planes budget {budget:?} decoded"
                );
            }
        }
    }
}

/// PLT lengths still bound a tile's last packet: with its non-empty bit
/// cleared, the packet's header ends after one byte, short of its signalled
/// length, and every route fails. Without PLT lengths nothing tells the
/// packet's unread bytes from data after the last packet.
#[test]
fn last_packet_shorter_than_its_plt_length_fails_on_every_route() {
    let one_packet = codestream(
        1,
        &EncodeOptions {
            num_decomposition_levels: 0,
            write_plt: true,
            ..EncodeOptions::default()
        },
    );
    let edited = edit_tile_parts(&one_packet, |part| {
        let sod = part
            .windows(2)
            .position(|window| window == [0xff, 0x93])
            .expect("SOD marker");
        assert_ne!(part[sod + 2] & 0x80, 0, "the packet is not empty");
        part[sod + 2] &= 0x7f;
    });
    let fmt = PixelFormat::Gray8;
    let clean = decode(
        &one_packet,
        None,
        CpuDecodeParallelism::Serial,
        fmt,
        Downscale::None,
        0,
    );
    assert!(clean.is_some());
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        for budget in [Some(usize::MAX), Some(0), None] {
            let decoded = pool.install(|| {
                decode(
                    &edited,
                    budget,
                    CpuDecodeParallelism::Auto,
                    fmt,
                    Downscale::None,
                    0,
                )
            });
            assert!(
                decoded.is_none(),
                "{threads} threads, planes budget {budget:?} decoded"
            );
        }
    }
}

/// The packets of a one-precinct-per-resolution tile in progression order,
/// as (resolution, layer) per packet, with `components` components.
fn packet_order(
    order: EncodeProgressionOrder,
    resolutions: usize,
    layers: usize,
    components: usize,
) -> Vec<(usize, usize)> {
    let mut packets = Vec::new();
    match order {
        EncodeProgressionOrder::Lrcp => {
            for layer in 0..layers {
                for resolution in 0..resolutions {
                    packets.extend((0..components).map(|_| (resolution, layer)));
                }
            }
        }
        EncodeProgressionOrder::Rlcp | EncodeProgressionOrder::Rpcl => {
            for resolution in 0..resolutions {
                for layer in 0..layers {
                    packets.extend((0..components).map(|_| (resolution, layer)));
                }
            }
        }
        EncodeProgressionOrder::Pcrl | EncodeProgressionOrder::Cprl => {
            for _ in 0..components {
                for resolution in 0..resolutions {
                    packets.extend((0..layers).map(|layer| (resolution, layer)));
                }
            }
        }
    }
    packets
}

/// One damage case: a codestream whose packets selected by `damaged` (from
/// (resolution, layer)) are overwritten with 0xff in every tile-part, coded
/// with `options` at three resolutions and two layers.
struct Damage {
    label: &'static str,
    components: u16,
    options: EncodeOptions,
    damaged: fn(usize, usize) -> bool,
    /// An edit of the damaged codestream's markers, packets unchanged.
    markers: fn(&[u8]) -> Vec<u8>,
    /// Whether a half-resolution decode reads none of the damaged packets.
    half_succeeds: bool,
}

fn same(codestream: &[u8]) -> Vec<u8> {
    codestream.to_vec()
}

/// `codestream` with a progression order change in its main header that
/// repeats its LRCP progression over every packet of three components.
fn with_lrcp_poc(codestream: &[u8]) -> Vec<u8> {
    let sot = codestream
        .windows(2)
        .position(|window| window == [0xff, 0x90])
        .expect("SOT marker");
    let poc = [0xff, 0x5f, 0, 9, 0, 0, 0, 2, 3, 3, 0];
    [&codestream[..sot], &poc[..], &codestream[sot..]].concat()
}

/// `codestream` without the PLT marker of its second tile-part.
fn without_second_plt(codestream: &[u8]) -> Vec<u8> {
    let mut part_index = 0;
    edit_tile_parts(codestream, |part| {
        part_index += 1;
        if part_index != 2 {
            return;
        }
        let plt = part
            .windows(2)
            .position(|window| window == [0xff, 0x58])
            .expect("PLT marker");
        let end = plt + 2 + usize::from(u16::from_be_bytes([part[plt + 2], part[plt + 3]]));
        part.drain(plt..end);
    })
}

fn damaged_codestream(damage: &Damage) -> (Vec<u8>, Vec<u8>) {
    let with_plt = EncodeOptions {
        write_plt: true,
        ..damage.options.clone()
    };
    let clean_plt = codestream(damage.components, &with_plt);
    let clean = codestream(damage.components, &damage.options);
    let packets = packet_order(
        damage.options.progression_order,
        3,
        2,
        usize::from(damage.components),
    );
    let ranges = j2k_test_support::packet_ranges(&clean_plt);
    assert_eq!(
        ranges.len() % packets.len(),
        0,
        "{}: whole tiles",
        damage.label
    );
    let selected: Vec<_> = ranges
        .iter()
        .enumerate()
        .filter(|&(index, _)| {
            let (resolution, layer) = packets[index % packets.len()];
            (damage.damaged)(resolution, layer)
        })
        .map(|(_, range)| range.clone())
        .collect();
    if damage.options.write_plt {
        let edited = j2k_test_support::with_packets_filled(&clean_plt, &selected, 0xff);
        return ((damage.markers)(&clean_plt), (damage.markers)(&edited));
    }
    // The same packets without PLT: the tile-part headers are shorter by the
    // PLT marker, the packet bytes are the same.
    assert_eq!(
        damage.options.tile_size, None,
        "{}: one tile-part",
        damage.label
    );
    let sod = |bytes: &[u8]| {
        bytes
            .windows(2)
            .position(|window| window == [0xff, 0x93])
            .expect("SOD marker")
    };
    let shift = sod(&clean_plt) - sod(&clean);
    assert_eq!(clean_plt[sod(&clean_plt)..], clean[sod(&clean)..]);
    let shifted: Vec<_> = selected
        .iter()
        .map(|range| range.start - shift..range.end - shift)
        .collect();
    let edited = j2k_test_support::with_packets_filled(&clean, &shifted, 0xff);
    ((damage.markers)(&clean), (damage.markers)(&edited))
}

/// Damage to the finest or the first layer's finest packets, in the
/// progressions and layouts a half-resolution decode skips them in.
fn finest_damage_cases() -> [Damage; 6] {
    let base = EncodeOptions {
        num_decomposition_levels: 2,
        num_layers: 2,
        ..EncodeOptions::default()
    };
    [
        Damage {
            label: "RPCL, PLT, finest packets",
            components: 3,
            options: EncodeOptions {
                progression_order: EncodeProgressionOrder::Rpcl,
                write_plt: true,
                ..base.clone()
            },
            damaged: finest,
            markers: same,
            half_succeeds: true,
        },
        Damage {
            label: "RPCL, finest packets",
            components: 1,
            options: EncodeOptions {
                progression_order: EncodeProgressionOrder::Rpcl,
                ..base.clone()
            },
            damaged: finest,
            markers: same,
            half_succeeds: true,
        },
        Damage {
            label: "RLCP, HTJ2K, finest packets",
            components: 3,
            options: EncodeOptions {
                progression_order: EncodeProgressionOrder::Rlcp,
                use_ht_block_coding: true,
                ..base.clone()
            },
            damaged: finest,
            markers: same,
            half_succeeds: true,
        },
        Damage {
            label: "RPCL, PLT, tiles, finest packets",
            components: 3,
            options: EncodeOptions {
                progression_order: EncodeProgressionOrder::Rpcl,
                tile_size: Some((64, 64)),
                write_plt: true,
                ..base.clone()
            },
            damaged: finest,
            markers: same,
            half_succeeds: true,
        },
        Damage {
            label: "LRCP, PLT, first layer's finest packets",
            components: 3,
            options: EncodeOptions {
                progression_order: EncodeProgressionOrder::Lrcp,
                write_plt: true,
                ..base.clone()
            },
            damaged: first_finest,
            markers: same,
            half_succeeds: true,
        },
        Damage {
            label: "CPRL, PLT, finest packets",
            components: 3,
            options: EncodeOptions {
                progression_order: EncodeProgressionOrder::Cprl,
                write_plt: true,
                ..base.clone()
            },
            damaged: finest,
            markers: same,
            half_succeeds: true,
        },
    ]
}

/// The other damage cases of
/// [`reduced_decodes_read_the_same_packets_on_every_route`]: packets a
/// half-resolution decode must parse or read, and more finest packets it
/// skips, with PCRL or several tile-parts per tile.
fn other_damage_cases() -> [Damage; 6] {
    let base = EncodeOptions {
        num_decomposition_levels: 2,
        num_layers: 2,
        ..EncodeOptions::default()
    };
    [
        Damage {
            label: "LRCP, first layer's finest packets, parsed to reach the second layer",
            components: 1,
            options: EncodeOptions {
                progression_order: EncodeProgressionOrder::Lrcp,
                ..base.clone()
            },
            damaged: first_finest,
            markers: same,
            half_succeeds: false,
        },
        Damage {
            label: "RPCL, PLT, coarsest packets",
            components: 3,
            options: EncodeOptions {
                progression_order: EncodeProgressionOrder::Rpcl,
                write_plt: true,
                ..base.clone()
            },
            damaged: coarsest,
            markers: same,
            half_succeeds: false,
        },
        Damage {
            label: "PCRL, PLT, finest packets",
            components: 3,
            options: EncodeOptions {
                progression_order: EncodeProgressionOrder::Pcrl,
                write_plt: true,
                ..base.clone()
            },
            damaged: finest,
            markers: same,
            half_succeeds: true,
        },
        Damage {
            label: "RPCL, PLT, two packets per tile-part, finest packets",
            components: 3,
            options: EncodeOptions {
                progression_order: EncodeProgressionOrder::Rpcl,
                tile_part_packet_limit: Some(2),
                write_plt: true,
                ..base.clone()
            },
            damaged: finest,
            markers: same,
            half_succeeds: true,
        },
        Damage {
            label: "LRCP, PLT, POC, first layer's finest packets, parsed to reach the second layer",
            components: 3,
            options: EncodeOptions {
                progression_order: EncodeProgressionOrder::Lrcp,
                write_plt: true,
                ..base.clone()
            },
            damaged: first_finest,
            markers: with_lrcp_poc,
            half_succeeds: false,
        },
        Damage {
            label: "LRCP, PLT in all but one tile-part, first layer's finest packets, parsed",
            components: 3,
            options: EncodeOptions {
                progression_order: EncodeProgressionOrder::Lrcp,
                tile_part_packet_limit: Some(3),
                write_plt: true,
                ..base.clone()
            },
            damaged: first_finest,
            markers: without_second_plt,
            half_succeeds: false,
        },
    ]
}

fn finest(resolution: usize, _layer: usize) -> bool {
    resolution == 2
}

fn first_finest(resolution: usize, layer: usize) -> bool {
    resolution == 2 && layer == 0
}

fn coarsest(resolution: usize, _layer: usize) -> bool {
    resolution == 0
}

/// A decode below full resolution reads the packets of the resolutions it
/// keeps, and of the finer ones only those it has to parse to find a later
/// packet it needs. Damage confined to the others leaves its output that of
/// the undamaged codestream on every route and pool size; damage in a
/// packet it needs or has to parse fails on every route, as the
/// full-resolution decode of every damaged codestream does.
#[test]
fn reduced_decodes_read_the_same_packets_on_every_route() {
    let cases = finest_damage_cases()
        .into_iter()
        .chain(other_damage_cases());
    let pools = [1, 4].map(|threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
    });
    for damage in cases {
        let (clean, edited) = damaged_codestream(&damage);
        let fmt = if damage.components == 1 {
            PixelFormat::Gray8
        } else {
            PixelFormat::Rgb8
        };
        let expected = decode(
            &clean,
            None,
            CpuDecodeParallelism::Serial,
            fmt,
            Downscale::Half,
            0,
        )
        .expect("clean half-resolution decode");
        for (pool, threads) in pools.iter().zip([1, 4]) {
            for budget in [Some(usize::MAX), Some(0), None] {
                let label = format!(
                    "{}, {threads} threads, planes budget {budget:?}",
                    damage.label
                );
                let half = pool.install(|| {
                    decode(
                        &edited,
                        budget,
                        CpuDecodeParallelism::Auto,
                        fmt,
                        Downscale::Half,
                        0,
                    )
                });
                if damage.half_succeeds {
                    assert!(half.as_ref() == Some(&expected), "{label}: half resolution");
                } else {
                    assert!(half.is_none(), "{label}: half resolution decoded");
                }
                let full = pool.install(|| {
                    decode(
                        &edited,
                        budget,
                        CpuDecodeParallelism::Auto,
                        fmt,
                        Downscale::None,
                        0,
                    )
                });
                assert!(full.is_none(), "{label}: full resolution decoded");
            }
        }
    }
}

/// A PLT length that ends a skipped packet inside the next packet the reduced
/// decode needs, or past the tile-part, disagrees with the packets: every
/// route fails rather than read a needed packet from the wrong byte.
#[test]
fn misplaced_plt_length_before_a_needed_packet_fails_on_every_route() {
    let clean = codestream(
        3,
        &EncodeOptions {
            num_decomposition_levels: 2,
            num_layers: 2,
            progression_order: EncodeProgressionOrder::Lrcp,
            write_plt: true,
            ..EncodeOptions::default()
        },
    );
    // LRCP, three components: packets 6..9 are the first layer's finest,
    // packet 9 the second layer's first, which a half-resolution decode reads.
    let one_byte_early = j2k_test_support::with_packet_lengths_edited(&clean, |lengths| {
        assert!(lengths[8] > 1, "a finest packet with a body");
        lengths[8] -= 1;
        lengths[9] += 1;
    });
    let past_the_tile_part =
        j2k_test_support::with_packet_lengths_edited(&clean, |lengths| lengths[8] += 1 << 20);
    let fmt = PixelFormat::Rgb8;
    assert!(decode(
        &clean,
        None,
        CpuDecodeParallelism::Serial,
        fmt,
        Downscale::Half,
        0
    )
    .is_some());
    for (label, edited) in [
        ("one byte early", one_byte_early),
        ("past the tile-part", past_the_tile_part),
    ] {
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            for budget in [Some(usize::MAX), Some(0), None] {
                let half = pool.install(|| {
                    decode(
                        &edited,
                        budget,
                        CpuDecodeParallelism::Auto,
                        fmt,
                        Downscale::Half,
                        0,
                    )
                });
                assert!(
                    half.is_none(),
                    "{label}, {threads} threads, planes budget {budget:?} decoded"
                );
            }
        }
    }
}

/// A tile-part cut inside its tile's last packet, a finest-resolution one,
/// fails at full resolution and decodes at half resolution as the uncut
/// codestream does, on every route, wherever the cut falls in that packet
/// (with EPH markers after the packet headers too).
#[test]
fn truncated_last_packet_spares_reduced_decodes_on_every_route() {
    for write_eph in [false, true] {
        let options = EncodeOptions {
            num_decomposition_levels: 2,
            progression_order: EncodeProgressionOrder::Rpcl,
            write_eph,
            write_plt: true,
            ..EncodeOptions::default()
        };
        let with_plt = codestream(1, &options);
        let last = j2k_test_support::packet_ranges(&with_plt)
            .pop()
            .expect("packets")
            .len();
        let clean = codestream(
            1,
            &EncodeOptions {
                write_plt: false,
                ..options
            },
        );
        let fmt = PixelFormat::Gray8;
        let expected = decode(
            &clean,
            None,
            CpuDecodeParallelism::Serial,
            fmt,
            Downscale::Half,
            0,
        )
        .expect("clean half-resolution decode");
        for cut in (1..last).step_by((last / 8).max(1)) {
            let edited = edit_tile_parts(&clean, |part| part.truncate(part.len() - cut));
            for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
                for budget in [Some(usize::MAX), Some(0), None] {
                    let label = format!("EPH {write_eph}, cut {cut} of {last}, {parallelism:?}, planes budget {budget:?}");
                    let half = decode(&edited, budget, parallelism, fmt, Downscale::Half, 0);
                    assert!(half.as_ref() == Some(&expected), "{label}: half resolution");
                    let full = decode(&edited, budget, parallelism, fmt, Downscale::None, 0);
                    assert!(full.is_none(), "{label}: full resolution decoded");
                }
            }
        }
    }
}
