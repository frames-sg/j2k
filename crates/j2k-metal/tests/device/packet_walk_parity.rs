// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Metal decodes read packets with the walk the CPU decodes use: data a
//! decode does not need is left unread on both, and packet data both must
//! read and reject fails on both with the same error class.

use std::ops::Range;

use super::*;
use j2k_native::{DecodeErrorClass, EncodeProgressionOrder};
use j2k_test_support::{
    add_packet_length, edit_tile_parts, packed_packet_headers, packet_ranges,
    with_byte_after_last_packets, with_packet_lengths_edited, with_packets_filled,
    with_surplus_packet_lengths, with_tile_parts_interleaved,
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

/// A flat image of `WIDTH` x `HEIGHT` samples, each pixel `pixel`.
fn flat_codestream(pixel: &[u8], ht: bool, options: &EncodeOptions) -> Vec<u8> {
    let pixels = pixel.repeat((WIDTH * HEIGHT) as usize);
    let components = u16::try_from(pixel.len()).expect("component count");
    let options = EncodeOptions {
        use_ht_block_coding: ht,
        ..options.clone()
    };
    encode(&pixels, WIDTH, HEIGHT, components, 8, false, &options).expect("encode fixture")
}

/// Every packet holds at least the first byte of its header, so a zero PLT
/// length is no packet's. Skipping an unneeded packet by such a length would
/// read the next needed packet from the skipped one's bytes: here, in a flat
/// image whose finest packets are empty, the second component's finest packet
/// as the third component's coarsest, whose data a half-resolution decode
/// needs. Every Metal entry point fails on it, at full and half resolution,
/// with the error class the CPU decode reports, and decodes the codestream
/// with its true lengths as the CPU does.
#[test]
fn zero_plt_length_before_a_needed_packet_fails_on_metal_as_on_the_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = EncodeOptions {
        num_decomposition_levels: 1,
        progression_order: EncodeProgressionOrder::Cprl,
        write_plt: true,
        ..EncodeOptions::default()
    };
    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut failures = Vec::new();
    for ht in [false, true] {
        let clean = flat_codestream(&[200, 50, 120], ht, &options);
        let companion = flat_codestream(&[30, 160, 90], ht, &options);
        // CPRL, one layer, one precinct per resolution: each component's
        // coarsest packet, then its finest, which the flat image leaves empty.
        let lengths: Vec<usize> = packet_ranges(&clean).iter().map(Range::len).collect();
        assert_eq!(lengths.len(), 6, "HT {ht}: six packets");
        assert_eq!(
            [lengths[1], lengths[3], lengths[5]],
            [1, 1, 1],
            "HT {ht}: empty finest packets"
        );
        assert!(
            lengths[4] > 1,
            "HT {ht}: the third component's coarsest packet has data"
        );
        let zero = with_packet_lengths_edited(&clean, |lengths| {
            let total: usize = lengths.iter().sum();
            lengths[3] = 0;
            lengths[4] = 1;
            lengths[5] = total - lengths[..5].iter().sum::<usize>();
        });
        for fmt in [PixelFormat::Rgb8, PixelFormat::Rgb16] {
            for scale in [Downscale::None, Downscale::Half] {
                let expected = cpu_decode(&clean, fmt, scale).expect("clean CPU decode");
                let case = format!("flat CPRL HT {ht}");
                for (edit, edited, other, expected) in [
                    ("clean", &clean, &companion, Expected::Decodes(&expected)),
                    ("zero length", &zero, &clean, Expected::Fails),
                ] {
                    check(
                        &format!("{case} {edit}"),
                        (edited, other),
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
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// With EPH markers every packet holds at least three bytes, its header's
/// first byte and the marker, so PLT lengths of one and two are no packet's.
/// Skipping by them would misplace the next needed packet as a zero length
/// does: in a flat CPRL image with three resolutions, lengths of one and two
/// for the second component's finer packets make a quarter-resolution decode
/// read the second one as the third component's coarsest, needed, packet.
/// Every Metal entry point fails on it, at every resolution, with the error
/// class the CPU decode reports, and decodes the codestream with its true
/// lengths as the CPU does.
#[test]
fn plt_lengths_shorter_than_an_eph_packet_fail_on_metal_as_on_the_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = EncodeOptions {
        num_decomposition_levels: 2,
        progression_order: EncodeProgressionOrder::Cprl,
        write_plt: true,
        write_eph: true,
        ..EncodeOptions::default()
    };
    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut failures = Vec::new();
    for ht in [false, true] {
        let clean = flat_codestream(&[200, 50, 120], ht, &options);
        let companion = flat_codestream(&[30, 160, 90], ht, &options);
        // CPRL, one layer, one precinct per resolution: each component's
        // three packets, coarsest first; the finer ones are empty, a byte and
        // EPH.
        let lengths: Vec<usize> = packet_ranges(&clean).iter().map(Range::len).collect();
        assert_eq!(lengths.len(), 9, "HT {ht}: nine packets");
        for finer in [1, 2, 4, 5, 7, 8] {
            assert_eq!(lengths[finer], 3, "HT {ht}: empty packet {finer}");
        }
        assert!(
            lengths[6] > 3,
            "HT {ht}: the third component's coarsest packet has data"
        );
        let short = with_packet_lengths_edited(&clean, |lengths| {
            let total: usize = lengths.iter().sum();
            lengths[4] = 1;
            lengths[5] = 2;
            lengths[6] = 3;
            lengths[8] = total - lengths[..8].iter().sum::<usize>();
        });
        for fmt in [PixelFormat::Rgb8, PixelFormat::Rgb16] {
            for scale in [Downscale::None, Downscale::Half, Downscale::Quarter] {
                let expected = cpu_decode(&clean, fmt, scale).expect("clean CPU decode");
                let case = format!("flat CPRL EPH HT {ht}");
                for (edit, edited, other, expected) in [
                    ("clean", &clean, &companion, Expected::Decodes(&expected)),
                    ("short lengths", &short, &clean, Expected::Fails),
                ] {
                    check(
                        &format!("{case} {edit}"),
                        (edited, other),
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
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// With packet headers packed in PPM or PPT, PLT lengths hold only packet
/// bodies, so a reduced decode parses every header up to the last packet it
/// needs. In a flat image, whose packets after the first hold one-byte empty
/// headers, damage to the header of a finest packet that precedes a needed
/// one fails every Metal entry point at half resolution, and damage to the
/// last packet's header, after every packet a half-resolution decode needs,
/// decodes there to the CPU decode of the undamaged codestream; both fail at
/// full resolution, with the error class the CPU decode reports.
#[test]
fn damaged_packed_headers_fail_reduced_decodes_on_metal_as_on_the_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut failures = Vec::new();
    for (label, write_ppm, write_ppt) in [("PPM", true, false), ("PPT", false, true)] {
        for ht in [false, true] {
            let options = EncodeOptions {
                num_decomposition_levels: LEVELS,
                num_layers: LAYERS,
                // Every coding pass in the first layer, where HT code-blocks
                // put them without targets.
                quality_layer_byte_targets: if ht { Vec::new() } else { vec![1, u64::MAX] },
                progression_order: EncodeProgressionOrder::Lrcp,
                write_plt: true,
                write_ppm,
                write_ppt,
                ..EncodeOptions::default()
            };
            let clean = flat_codestream(&[90], ht, &options);
            let companion = flat_codestream(&[170], ht, &options);
            // LRCP, one component, one precinct per resolution: the first
            // layer's coarsest header holds the image, then five empty ones,
            // the first layer's two finer packets' and the second layer's
            // three.
            let headers = packed_packet_headers(&clean);
            assert_eq!(
                clean[headers.end - 5..headers.end],
                [0; 5],
                "{label} HT {ht}: empty headers"
            );
            let damaged_at = |from_end: usize| {
                let mut damaged = clean.clone();
                damaged[headers.end - from_end] = 0xff;
                damaged
            };
            // The first layer's finest header, before the second layer's
            // packets, and the second layer's finest header, the last.
            let (navigation, last) = (damaged_at(4), damaged_at(1));
            for fmt in [PixelFormat::Gray8, PixelFormat::Gray16] {
                let half = cpu_decode(&clean, fmt, Downscale::Half)
                    .expect("clean half-resolution CPU decode");
                let full = cpu_decode(&clean, fmt, Downscale::None).expect("clean CPU decode");
                for (edit, edited, other, scale, expected) in [
                    (
                        "clean",
                        &clean,
                        &companion,
                        Downscale::Half,
                        Expected::Decodes(&half),
                    ),
                    (
                        "clean",
                        &clean,
                        &companion,
                        Downscale::None,
                        Expected::Decodes(&full),
                    ),
                    (
                        "finest header before a needed packet damaged",
                        &navigation,
                        &clean,
                        Downscale::Half,
                        Expected::Fails,
                    ),
                    (
                        "finest header before a needed packet damaged",
                        &navigation,
                        &clean,
                        Downscale::None,
                        Expected::Fails,
                    ),
                    (
                        "last header damaged",
                        &last,
                        &clean,
                        Downscale::Half,
                        Expected::Decodes(&half),
                    ),
                    (
                        "last header damaged",
                        &last,
                        &clean,
                        Downscale::None,
                        Expected::Fails,
                    ),
                ] {
                    check(
                        &format!("flat {label} HT {ht} {edit}"),
                        (edited, other),
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
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Tile-parts of several tiles may take turns in the codestream, and a
/// tile's packets are complete across its tile-parts. With one RPCL packet
/// per tile-part, every Metal entry point decodes the interleaved codestream
/// to the CPU decode of the tile-major one; damage to every tile's finest
/// packets, in the last rounds of tile-parts, decodes at half resolution as
/// the undamaged codestream does, and damage to every tile's coarsest
/// packets, in the first rounds, fails; both fail at full resolution, with
/// the error class the CPU decode reports. The images are the tiled
/// fixtures, four tiles of at most 64 x 64, and a gray one of eight
/// 512 x 512 tiles.
#[test]
fn interleaved_tile_parts_spare_reduced_decodes_per_tile_on_metal_as_on_the_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal session");
    let finest = usize::from(LEVELS);
    // (image, codestream, formats, tiles, packets per tile, the packets of
    // each tile's finest and coarsest resolution)
    let mut images = Vec::new();
    for fixture in FIXTURES.into_iter().filter(|fixture| fixture.tiled) {
        let options = EncodeOptions {
            tile_part_packet_limit: Some(1),
            ..fixture.options()
        };
        images.push((
            format!("{fixture:?}"),
            fixture.codestream(&options),
            fixture.formats(),
            4,
            (usize::from(LEVELS) + 1) * usize::from(fixture.components) * usize::from(LAYERS),
            fixture.packets_of(EncodeProgressionOrder::Rpcl, finest, None),
            fixture.packets_of(EncodeProgressionOrder::Rpcl, 0, None),
        ));
    }
    let (width, height) = (2048, 1024);
    let mut state = 0x9e37_79b9_u32;
    let pixels: Vec<u8> = (0..width * height)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state.to_be_bytes()[0]
        })
        .collect();
    let options = EncodeOptions {
        num_decomposition_levels: LEVELS,
        progression_order: EncodeProgressionOrder::Rpcl,
        tile_size: Some((512, 512)),
        tile_part_packet_limit: Some(1),
        write_plt: true,
        ..EncodeOptions::default()
    };
    images.push((
        "2048 x 1024 gray, 512 x 512 tiles".to_string(),
        encode(&pixels, width, height, 1, 8, false, &options).expect("encode fixture"),
        &[PixelFormat::Gray8, PixelFormat::Gray16][..],
        8,
        usize::from(LEVELS) + 1,
        vec![finest],
        vec![0],
    ));
    let mut failures = Vec::new();
    for (image, clean, formats, tiles, packets_per_tile, finest_packets, coarsest_packets) in
        &images
    {
        let ranges = packet_ranges(clean);
        assert_eq!(
            ranges.len(),
            tiles * packets_per_tile,
            "{image}: {tiles} tiles"
        );
        let damaged = |indices: &[usize]| {
            let packets: Vec<Range<usize>> = ranges
                .chunks(*packets_per_tile)
                .flat_map(|tile| indices.iter().map(|&index| tile[index].clone()))
                .collect();
            with_tile_parts_interleaved(&with_packets_filled(clean, &packets, 0xff))
        };
        let interleaved = with_tile_parts_interleaved(clean);
        assert_ne!(&interleaved, clean, "{image}: tile-parts change places");
        let (finest_damaged, coarsest_damaged) =
            (damaged(finest_packets), damaged(coarsest_packets));
        for &fmt in *formats {
            for scale in [Downscale::None, Downscale::Half] {
                let expected = cpu_decode(clean, fmt, scale).expect("clean CPU decode");
                assert!(
                    cpu_decode(&interleaved, fmt, scale).is_ok_and(|decoded| decoded == expected),
                    "{image} {fmt:?} {scale:?}: interleaving keeps the image"
                );
                let finest_expected = match scale {
                    Downscale::None => Expected::Fails,
                    _ => Expected::Decodes(&expected),
                };
                for (edit, edited, expected) in [
                    ("clean", &interleaved, Expected::Decodes(&expected)),
                    ("finest damaged", &finest_damaged, finest_expected),
                    ("coarsest damaged", &coarsest_damaged, Expected::Fails),
                ] {
                    check(
                        &format!("{image} interleaved {edit}"),
                        (edited, clean),
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
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
