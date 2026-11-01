// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Metal decodes read packets with the walk the CPU decodes use: data a
//! decode does not need is left unread on both, and packet data both must
//! read and reject fails on both with the same error class.

use std::ops::Range;

use super::*;
use j2k_native::{DecodeErrorClass, EncodeProgressionOrder};
use j2k_test_support::{
    add_packet_length, edit_tile_parts, packet_ranges, with_byte_after_last_packets,
    with_packets_filled, with_surplus_packet_lengths,
};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 80;
const LEVELS: u8 = 2;
const LAYERS: u8 = 2;

#[derive(Clone, Copy, Debug)]
struct Fixture {
    components: u16,
    ht: bool,
    tiled: bool,
}

const FIXTURES: [Fixture; 6] = [
    Fixture {
        components: 1,
        ht: false,
        tiled: false,
    },
    Fixture {
        components: 1,
        ht: true,
        tiled: false,
    },
    Fixture {
        components: 3,
        ht: false,
        tiled: false,
    },
    Fixture {
        components: 3,
        ht: true,
        tiled: false,
    },
    Fixture {
        components: 1,
        ht: false,
        tiled: true,
    },
    Fixture {
        components: 3,
        ht: true,
        tiled: true,
    },
];

impl Fixture {
    fn formats(self) -> &'static [PixelFormat] {
        if self.components == 1 {
            &[PixelFormat::Gray8, PixelFormat::Gray16]
        } else {
            &[PixelFormat::Rgb8, PixelFormat::Rgb16]
        }
    }

    /// Two layers over two decomposition levels, one precinct per
    /// resolution, with PLT lengths; tiled fixtures are four
    /// resolution-major 64 x 64 tiles, the others one layer-major tile.
    fn options(self) -> EncodeOptions {
        EncodeOptions {
            num_decomposition_levels: LEVELS,
            num_layers: LAYERS,
            use_ht_block_coding: self.ht,
            write_plt: true,
            tile_size: self.tiled.then_some((64, 64)),
            progression_order: if self.tiled {
                EncodeProgressionOrder::Rpcl
            } else {
                EncodeProgressionOrder::Lrcp
            },
            ..EncodeOptions::default()
        }
    }

    fn codestream(self, options: &EncodeOptions) -> Vec<u8> {
        self.codestream_of_noise(options, 0x2545_f491)
    }

    /// Another valid codestream of the same shape, other samples.
    fn companion(self, options: &EncodeOptions) -> Vec<u8> {
        self.codestream_of_noise(options, 0x9e37_79b9)
    }

    fn codestream_of_noise(self, options: &EncodeOptions, seed: u32) -> Vec<u8> {
        let mut state = seed;
        let pixels: Vec<u8> = (0..WIDTH * HEIGHT * u32::from(self.components))
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state.to_be_bytes()[0]
            })
            .collect();
        encode(&pixels, WIDTH, HEIGHT, self.components, 8, false, options).expect("encode fixture")
    }

    /// Indices, among one tile's packets, of the packets of `resolution`
    /// (of `layer` only, when given) in `progression`.
    fn packets_of(
        self,
        progression: EncodeProgressionOrder,
        resolution: usize,
        layer: Option<usize>,
    ) -> Vec<usize> {
        let (resolutions, components, layers) = (
            usize::from(LEVELS) + 1,
            usize::from(self.components),
            usize::from(LAYERS),
        );
        let mut indices = Vec::new();
        for component in 0..components {
            for l in (0..layers).filter(|&l| layer.is_none_or(|layer| layer == l)) {
                indices.push(match progression {
                    EncodeProgressionOrder::Rpcl => {
                        (resolution * components + component) * layers + l
                    }
                    EncodeProgressionOrder::Lrcp => {
                        (l * resolutions + resolution) * components + component
                    }
                    _ => unreachable!("fixtures are RPCL or LRCP"),
                });
            }
        }
        indices.sort_unstable();
        indices
    }
}

/// Where each tile-part's packet data starts, after its SOD marker.
fn tile_part_bodies(codestream: &[u8]) -> Vec<usize> {
    let segment_end = |at: usize| {
        at + 2 + usize::from(u16::from_be_bytes([codestream[at + 2], codestream[at + 3]]))
    };
    let mut bodies = Vec::new();
    let mut at = 2;
    while codestream[at..at + 2] != [0xff, 0x90] {
        at = segment_end(at);
    }
    while codestream[at..at + 2] == [0xff, 0x90] {
        let psot = u32::from_be_bytes(codestream[at + 6..at + 10].try_into().expect("Psot"));
        let end = at + usize::try_from(psot).expect("Psot fits");
        let mut marker = at + 12;
        while codestream[marker..marker + 2] != [0xff, 0x93] {
            marker = segment_end(marker);
        }
        bodies.push(marker + 2);
        at = end;
    }
    bodies
}

/// `codestream` (with PLT lengths) and `twin` (the same encode without them)
/// with the packets at `indices` of every tile filled with 0xFF.
fn damaged_packets(
    codestream: &[u8],
    twin: &[u8],
    packets_per_tile: usize,
    indices: &[usize],
) -> (Vec<u8>, Vec<u8>) {
    let ranges = packet_ranges(codestream);
    assert_eq!(ranges.len() % packets_per_tile, 0, "whole tiles of packets");
    let (bodies, twin_bodies) = (tile_part_bodies(codestream), tile_part_bodies(twin));
    assert_eq!(
        bodies.len(),
        ranges.len() / packets_per_tile,
        "one tile-part per tile"
    );
    let mut damaged = Vec::new();
    let mut twin_damaged = Vec::new();
    for (tile, packets) in ranges.chunks(packets_per_tile).enumerate() {
        for &index in indices {
            let range: Range<usize> = packets[index].clone();
            let shift = |at: usize| at - bodies[tile] + twin_bodies[tile];
            twin_damaged.push(shift(range.start)..shift(range.end));
            damaged.push(range);
        }
    }
    let twin_damaged_bytes = with_packets_filled(twin, &twin_damaged, 0xff);
    for (range, twin_range) in damaged.iter().zip(&twin_damaged) {
        assert_eq!(
            codestream[range.clone()],
            twin[twin_range.clone()],
            "the twin holds the same packet bytes"
        );
    }
    (
        with_packets_filled(codestream, &damaged, 0xff),
        twin_damaged_bytes,
    )
}

/// The CPU decode of `bytes` at `scale`, tightly packed.
fn cpu_decode(bytes: &[u8], fmt: PixelFormat, scale: Downscale) -> Result<Vec<u8>, j2k::J2kError> {
    let mut decoder = j2k::J2kDecoder::new(bytes)?;
    let (width, height) = decoder.info().dimensions;
    let divisor = scale.denominator();
    let stride = width.div_ceil(divisor) as usize * fmt.bytes_per_pixel();
    let mut out = vec![0_u8; stride * height.div_ceil(divisor) as usize];
    match scale {
        Downscale::None => decoder.decode_into(&mut out, stride, fmt)?,
        _ => decoder.decode_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            stride,
            fmt,
            scale,
        )?,
    };
    Ok(out)
}

fn request(fmt: PixelFormat, scale: Downscale) -> MetalDecodeRequest {
    match scale {
        Downscale::None => MetalDecodeRequest::full(fmt, BackendRequest::Metal),
        _ => MetalDecodeRequest::scaled(fmt, scale, BackendRequest::Metal),
    }
}

fn surface_bytes(surface: &Surface) -> Vec<u8> {
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    surface.as_bytes().expect("surface readback").into_owned()
}

/// A failed decode: its message and the class of the native decode error
/// behind it, if there is one.
#[derive(Debug)]
struct Failure {
    message: String,
    class: Option<DecodeErrorClass>,
}

impl Failure {
    fn of(error: &(dyn std::error::Error + 'static)) -> Self {
        let mut class = None;
        let mut next = Some(error);
        while let Some(current) = next {
            if let Some(native) = current.downcast_ref::<j2k_native::DecodeError>() {
                class = Some(native.classify());
                break;
            }
            next = current.source();
        }
        Self {
            message: error.to_string(),
            class,
        }
    }
}

/// `bytes` decoded at `scale` by each Metal entry point that consumes the
/// shared packet walk: an explicit request with and without a session, a
/// tile batch that also holds `other` (a distinct input that decodes, so the
/// batch takes its distinct-input route and fails only for `bytes`), and,
/// for 8-bit output, the batch decoder.
fn metal_outcomes(
    bytes: &[u8],
    other: &[u8],
    fmt: PixelFormat,
    scale: Downscale,
    session: &MetalBackendSession,
) -> Vec<(&'static str, Result<Vec<u8>, Failure>)> {
    let failure = |error: Error| Failure::of(&error);
    let mut outcomes = Vec::new();
    outcomes.push((
        "request",
        J2kDecoder::new(bytes)
            .and_then(|mut decoder| decoder.decode_request_to_device(request(fmt, scale)))
            .map(|surface| surface_bytes(&surface))
            .map_err(failure),
    ));
    outcomes.push((
        "session request",
        J2kDecoder::new(bytes)
            .and_then(|mut decoder| {
                decoder.decode_request_to_device_with_session(request(fmt, scale), session)
            })
            .map(|surface| surface_bytes(&surface))
            .map_err(failure),
    ));
    let mut batch = MetalTileBatch::with_capacity(2);
    for input in [bytes, other] {
        batch
            .push_tile_request(input, request(fmt, scale))
            .expect("queue tile batch request");
    }
    outcomes.push((
        "tile batch",
        batch
            .decode_all()
            .map(|surfaces| surface_bytes(&surfaces[0]))
            .map_err(failure),
    ));
    if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Rgb8) {
        let mut options = BatchDecodeOptions::default();
        options.layout = BatchLayout::Nhwc;
        let decode_request = match scale {
            Downscale::None => DecodeRequest::Full,
            _ => DecodeRequest::Reduced { scale },
        };
        let mut decoder =
            MetalBatchDecoder::system_default_with_options(options).expect("Metal batch decoder");
        let outcome =
            match decoder.decode_batch(vec![EncodedImage::new(Arc::from(bytes), decode_request)]) {
                Err(error) => Err(failure(error)),
                Ok(batch) => match (batch.errors(), batch.group_errors(), batch.groups()) {
                    ([error], [], []) => Err(Failure::of(error)),
                    ([], [error], []) => Err(Failure::of(error)),
                    ([], [], [group]) => Ok(surface_bytes(&group.surfaces()[0])),
                    _ => Err(Failure {
                        message: "an unexpected batch result shape".to_string(),
                        class: None,
                    }),
                },
            };
        outcomes.push(("batch decoder", outcome));
    }
    outcomes
}

/// What every route must give for one codestream and request.
enum Expected<'a> {
    /// These bytes.
    Decodes(&'a [u8]),
    /// A failure of one native error class.
    Fails,
}

/// Checks that the CPU decode and every Metal entry point give `expected`
/// for `bytes` at `scale`, and on failure fail with one native error class.
fn check(
    case: &str,
    (bytes, other): (&[u8], &[u8]),
    fmt: PixelFormat,
    scale: Downscale,
    expected: &Expected<'_>,
    session: &MetalBackendSession,
    failures: &mut Vec<String>,
) {
    let case = format!("{case} {fmt:?} {scale:?}");
    let cpu = cpu_decode(bytes, fmt, scale).map_err(|error| Failure::of(&error));
    let class = match (expected, &cpu) {
        (Expected::Decodes(expected), Ok(decoded)) if decoded == expected => None,
        (Expected::Decodes(_), Ok(_)) => {
            failures.push(format!("{case} CPU: other bytes"));
            None
        }
        (Expected::Decodes(_), Err(failure)) => {
            failures.push(format!("{case} CPU: {}", failure.message));
            None
        }
        (Expected::Fails, Ok(_)) => {
            failures.push(format!("{case} CPU: decoded"));
            None
        }
        (Expected::Fails, Err(failure)) => {
            if failure.class.is_none() {
                failures.push(format!("{case} CPU: no native cause: {}", failure.message));
            }
            failure.class
        }
    };
    for (route, outcome) in metal_outcomes(bytes, other, fmt, scale, session) {
        match (expected, outcome) {
            (Expected::Decodes(expected), Ok(decoded)) if decoded == *expected => {}
            (Expected::Decodes(_), Ok(_)) => failures.push(format!("{case} {route}: other bytes")),
            (Expected::Decodes(_), Err(failure)) => {
                failures.push(format!("{case} {route}: {}", failure.message));
            }
            (Expected::Fails, Ok(_)) => failures.push(format!("{case} {route}: decoded")),
            (Expected::Fails, Err(failure)) if class.is_some() && failure.class == class => {}
            (Expected::Fails, Err(failure)) => failures.push(format!(
                "{case} {route}: {} ({:?}), where the CPU reports {:?}",
                failure.message,
                failure.class,
                cpu.as_ref().err()
            )),
        }
    }
}

/// Data after the tile's last packet in each tile-part, bytes or PLT
/// lengths past it, is no packet's: every Metal entry point decodes such a
/// codestream, at full and half resolution, to the CPU decode of the
/// codestream without it.
#[test]
fn data_after_a_tiles_last_packet_decodes_on_metal_as_on_the_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut failures = Vec::new();
    for fixture in FIXTURES {
        let clean = fixture.codestream(&fixture.options());
        let companion = fixture.companion(&fixture.options());
        let edits = [
            ("clean", clean.clone()),
            ("byte after", with_byte_after_last_packets(&clean)),
            ("surplus lengths", with_surplus_packet_lengths(&clean)),
            (
                "both",
                with_byte_after_last_packets(&with_surplus_packet_lengths(&clean)),
            ),
        ];
        for &fmt in fixture.formats() {
            for scale in [Downscale::None, Downscale::Half] {
                let expected = cpu_decode(&clean, fmt, scale).expect("clean CPU decode");
                for (edit, edited) in &edits {
                    check(
                        &format!("{fixture:?} {edit}"),
                        (edited, &companion),
                        fmt,
                        scale,
                        &Expected::Decodes(&expected),
                        &session,
                        &mut failures,
                    );
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Packet data the walk must reject fails every Metal entry point with the
/// error class the CPU decode reports: a tile's last packet shorter than its
/// PLT length (at full resolution, where that packet is read), a PLT length
/// past a tile-part's packets while the tile's packets go on in the next
/// tile-part, and a damaged first packet, which every resolution needs.
#[test]
fn rejected_packet_data_fails_on_metal_as_on_the_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut failures = Vec::new();
    for fixture in FIXTURES {
        let clean = fixture.codestream(&fixture.options());
        // One layer: HT code-blocks put every pass in the first, which leaves
        // the last packet of a second layer empty.
        let mut short_last = fixture.codestream(&EncodeOptions {
            num_layers: 1,
            ..fixture.options()
        });
        let last = packet_ranges(&short_last).last().expect("packets").start;
        assert_ne!(
            short_last[last] & 0x80,
            0,
            "{fixture:?}: last packet is empty"
        );
        // Cleared, the packet's non-empty bit ends its header after one
        // byte, short of its PLT length.
        short_last[last] &= 0x7f;
        let mut cases = vec![
            ("short last packet", short_last, &[Downscale::None][..]),
            (
                "damaged first packet",
                with_packets_filled(&clean, &packet_ranges(&clean)[..1], 0xff),
                &[Downscale::None, Downscale::Half][..],
            ),
        ];
        if !fixture.tiled {
            let split = fixture.codestream(&EncodeOptions {
                tile_part_packet_limit: Some(1),
                ..fixture.options()
            });
            let mut first = true;
            let surplus_first = edit_tile_parts(&split, |part| {
                if std::mem::take(&mut first) {
                    add_packet_length(part);
                }
            });
            cases.push((
                "surplus length before a later tile-part",
                surplus_first,
                &[Downscale::None, Downscale::Half][..],
            ));
        }
        for (edit, edited, scales) in &cases {
            for &fmt in fixture.formats() {
                for &scale in *scales {
                    check(
                        &format!("{fixture:?} {edit}"),
                        (edited, &clean),
                        fmt,
                        scale,
                        &Expected::Fails,
                        &session,
                        &mut failures,
                    );
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A half-resolution decode reads only the packets it needs, on every
/// backend. Damage confined to finest-level packets after the last needed
/// one (resolution-major, with PLT lengths or without), or to finest-level
/// packets a PLT length skips (layer-major), decodes to the clean
/// codestream's half-resolution image on Metal as on the CPU. Damage in a
/// coarsest-level packet, or in a packet that must be parsed to reach a
/// later needed one, and any of it at full resolution, fails on both with
/// one error class.
#[test]
fn half_resolution_skips_damaged_finest_packets_on_metal_as_on_the_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal session");
    let finest = usize::from(LEVELS);
    let mut failures = Vec::new();
    for fixture in FIXTURES {
        let packets_per_tile =
            (usize::from(LEVELS) + 1) * usize::from(fixture.components) * usize::from(LAYERS);
        let progressions: &[EncodeProgressionOrder] = if fixture.tiled {
            &[EncodeProgressionOrder::Rpcl]
        } else {
            &[EncodeProgressionOrder::Rpcl, EncodeProgressionOrder::Lrcp]
        };
        for &progression in progressions {
            let options = EncodeOptions {
                progression_order: progression,
                ..fixture.options()
            };
            let clean = fixture.codestream(&options);
            let twin = fixture.codestream(&EncodeOptions {
                write_plt: false,
                ..options
            });
            // (damaged packets, half resolution decodes with PLT, without)
            let cases = match progression {
                EncodeProgressionOrder::Rpcl => [
                    (
                        "finest",
                        fixture.packets_of(progression, finest, None),
                        true,
                        true,
                    ),
                    (
                        "coarsest",
                        fixture.packets_of(progression, 0, None),
                        false,
                        false,
                    ),
                ],
                _ => [
                    (
                        "layer-0 finest",
                        fixture.packets_of(progression, finest, Some(0)),
                        true,
                        false,
                    ),
                    (
                        "layer-0 coarsest",
                        fixture.packets_of(progression, 0, Some(0)),
                        false,
                        false,
                    ),
                ],
            };
            for (damage, indices, plt_decodes, twin_decodes) in &cases {
                let (damaged, twin_damaged) =
                    damaged_packets(&clean, &twin, packets_per_tile, indices);
                for &fmt in fixture.formats() {
                    for (plt, original, edited, decodes) in [
                        ("PLT", &clean, &damaged, *plt_decodes),
                        ("no PLT", &twin, &twin_damaged, *twin_decodes),
                    ] {
                        let case = format!("{fixture:?} {progression:?} {plt} {damage} damaged");
                        let expected = cpu_decode(original, fmt, Downscale::Half)
                            .expect("clean half-resolution CPU decode");
                        for (scale, expected) in [
                            (Downscale::None, Expected::Fails),
                            (
                                Downscale::Half,
                                if decodes {
                                    Expected::Decodes(&expected)
                                } else {
                                    Expected::Fails
                                },
                            ),
                        ] {
                            check(
                                &case,
                                (edited, original),
                                fmt,
                                scale,
                                &expected,
                                &session,
                                &mut failures,
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A tile of one sample at an image corner has empty detail sub-bands, so its
/// full-resolution precinct holds no code-block and its packet, the
/// codestream's last, nothing a decode uses. With that packet's EPH marker
/// damaged, every Metal entry point decodes, at full and half resolution, to
/// the CPU decode of the undamaged codestream.
#[test]
fn packets_of_precincts_without_code_blocks_are_left_unread_on_metal_as_on_the_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let clean: &[u8] = include_bytes!(
        "../../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_gray_3x3_tile2x2_53_eph.j2k"
    );
    let mut damaged = clean.to_vec();
    let eph = damaged.len() - 4;
    assert_eq!(
        damaged[eph..],
        [0xff, 0x92, 0xff, 0xd9],
        "the last packet ends in EPH"
    );
    damaged[eph + 1] = 0;
    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut failures = Vec::new();
    for fmt in [PixelFormat::Gray8, PixelFormat::Gray16] {
        for scale in [Downscale::None, Downscale::Half] {
            let expected = cpu_decode(clean, fmt, scale).expect("clean CPU decode");
            check(
                "corner tile EPH damaged",
                (&damaged, clean),
                fmt,
                scale,
                &Expected::Decodes(&expected),
                &session,
                &mut failures,
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
