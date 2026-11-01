// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA decodes read a codestream's packets as the CPU decodes do. Data after
//! a tile's last packet, PLT lengths past it, and damage in packets that a
//! reduced-resolution decode does not read leave the samples of the clean
//! codestream; damage in a packet the decode must read or parse, in packed
//! packet headers it must parse, and PLT lengths it cannot skip by fail the
//! CUDA decode with the CPU decode's native error. Each case runs on the
//! single-image decode, the batch of images and the prepared batch, and with
//! tile-parts of several tiles interleaved, on the prepared batch.

#![cfg(feature = "cuda-runtime")]

use std::ops::Range;
use std::sync::Arc;

use j2k::{
    prepare_batch, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    DecodeRequest, EncodedImage,
};
use j2k_core::{BackendRequest, Downscale, ImageDecode, ImageDecodeDevice, PixelFormat};
use j2k_cuda::{
    CudaBatchDecoder, CudaSession, J2kDecoder, J2kScratchPool, Surface, SurfaceResidency,
};
use j2k_native::{
    encode, encode_htj2k, DecodeError, DecodeErrorClass, EncodeOptions, EncodeProgressionOrder,
};
use j2k_test_support::{
    add_packet_length, cuda_runtime_and_strict_oxide_gate, edit_tile_parts, packed_packet_headers,
    packet_ranges, with_byte_after_last_packets, with_packet_lengths_edited, with_packets_filled,
    with_sop_marker_fields, with_surplus_packet_lengths, with_tile_cod_disabling_sop_eph,
    with_tile_parts_interleaved,
};

const WIDTH: u32 = 70;
const HEIGHT: u32 = 52;
const LEVELS: u8 = 2;
const LAYERS: u8 = 2;

/// The sample layout and block coder of a fixture.
#[derive(Clone, Copy, Debug)]
struct Kind {
    components: u16,
    ht: bool,
}

const KINDS: [Kind; 4] = [
    Kind {
        components: 1,
        ht: false,
    },
    Kind {
        components: 1,
        ht: true,
    },
    Kind {
        components: 3,
        ht: false,
    },
    Kind {
        components: 3,
        ht: true,
    },
];

impl Kind {
    fn format(self) -> PixelFormat {
        if self.components == 1 {
            PixelFormat::Gray8
        } else {
            PixelFormat::Rgb8
        }
    }

    fn channels(self) -> usize {
        usize::from(self.components)
    }

    /// A one-tile reversible codestream of this kind with `LEVELS`
    /// decomposition levels and `LAYERS` quality layers, in `progression`.
    fn codestream(self, progression: EncodeProgressionOrder, write_plt: bool) -> Vec<u8> {
        self.encode(&EncodeOptions {
            progression_order: progression,
            write_plt,
            ..fixture_options()
        })
    }

    fn encode(self, options: &EncodeOptions) -> Vec<u8> {
        self.encode_sized(WIDTH, HEIGHT, options)
    }

    /// A `width` x `height` codestream of this kind.
    fn encode_sized(self, width: u32, height: u32, options: &EncodeOptions) -> Vec<u8> {
        let mut pixels = Vec::new();
        for y in 0..height {
            for x in 0..width {
                for c in 0..u32::from(self.components) {
                    let noise = (x.wrapping_mul(2_654_435_761) ^ (y * 40_503) ^ (c * 977)) >> 13;
                    pixels.push((x * 5 + y * 3 + c * 61 + noise % 37).to_le_bytes()[0]);
                }
            }
        }
        self.encode_pixels(&pixels, width, height, options)
    }

    /// A codestream of this kind whose every pixel is (200, 50, 120), or 200
    /// in gray: its detail sub-bands are empty.
    fn encode_flat(self, options: &EncodeOptions) -> Vec<u8> {
        let pixel = &[200, 50, 120][..self.channels()];
        let pixels = pixel.repeat((WIDTH * HEIGHT) as usize);
        self.encode_pixels(&pixels, WIDTH, HEIGHT, options)
    }

    fn encode_pixels(
        self,
        pixels: &[u8],
        width: u32,
        height: u32,
        options: &EncodeOptions,
    ) -> Vec<u8> {
        let encoder = if self.ht { encode_htj2k } else { encode };
        encoder(pixels, width, height, self.components, 8, false, options).expect("encode fixture")
    }

    /// Packets per resolution of a one-precinct tile.
    fn packets_per_resolution(self) -> usize {
        self.channels() * usize::from(LAYERS)
    }

    /// Packets of a one-precinct tile: resolutions 0 (the coarsest) to
    /// `LEVELS` (the finest).
    fn packets(self) -> usize {
        self.packets_per_resolution() * (usize::from(LEVELS) + 1)
    }

    /// Packets of a one-precinct tile in one layer.
    fn packets_per_layer(self) -> usize {
        self.packets() / usize::from(LAYERS)
    }
}

fn fixture_options() -> EncodeOptions {
    EncodeOptions {
        reversible: true,
        num_decomposition_levels: LEVELS,
        num_layers: LAYERS,
        ..EncodeOptions::default()
    }
}

/// The tiles of a case's codestream. Direct plans take one tile, so the
/// single-image CUDA decodes reject a tiled codestream before reading any
/// packet, as unsupported; the prepared batch decodes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tiles {
    One,
    Several,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expect {
    Decodes,
    Fails,
}

/// An edited codestream and what its full- and half-resolution decodes do.
struct Case {
    label: &'static str,
    clean: Vec<u8>,
    edited: Vec<u8>,
    full: Expect,
    half: Expect,
}

/// A decode's samples, or the native decoder error it failed with.
type Outcome = Result<Vec<u8>, String>;

/// Marks a native error that rejects the request rather than the codestream.
const UNSUPPORTED: &str = "unsupported: ";

fn failure(error: &(dyn std::error::Error + 'static)) -> String {
    let mut current = Some(error);
    while let Some(cause) = current {
        if let Some(native) = cause.downcast_ref::<DecodeError>() {
            let unsupported = matches!(native.classify(), DecodeErrorClass::Unsupported { .. });
            return format!("{}{native:?}", if unsupported { UNSUPPORTED } else { "" });
        }
        current = cause.source();
    }
    format!("no native decoder error: {error}")
}

/// The stride and length of a decode of `bytes` at `scale`.
fn layout(bytes: &[u8], kind: Kind, scale: Downscale) -> (usize, usize) {
    let (width, height) = J2kDecoder::inspect(bytes).expect("inspect").dimensions;
    let width = width.div_ceil(scale.denominator()) as usize;
    let height = height.div_ceil(scale.denominator()) as usize;
    let stride = width * kind.channels();
    (stride, stride * height)
}

fn cpu_decode(bytes: &[u8], kind: Kind, scale: Downscale) -> Outcome {
    let (stride, len) = layout(bytes, kind, scale);
    let mut out = vec![0_u8; len];
    let mut decoder = J2kDecoder::new(bytes).map_err(|error| failure(&error))?;
    let mut pool = J2kScratchPool::new();
    if scale == Downscale::None {
        decoder.decode_into_with_scratch(&mut pool, &mut out, stride, kind.format())
    } else {
        decoder.decode_scaled_into(&mut pool, &mut out, stride, kind.format(), scale)
    }
    .map_err(|error| failure(&error))?;
    Ok(out)
}

/// The samples of a surface a CUDA decode produced on the device.
fn device_samples(bytes: &[u8], surface: &Surface, kind: Kind, scale: Downscale) -> Vec<u8> {
    assert_eq!(
        surface.residency(),
        SurfaceResidency::CudaResidentDecode,
        "decoded on the device"
    );
    let (stride, len) = layout(bytes, kind, scale);
    let mut out = vec![0_u8; len];
    surface
        .download_into(&mut out, stride)
        .expect("download surface");
    out
}

fn cuda_decode(bytes: &[u8], kind: Kind, scale: Downscale) -> Outcome {
    let mut decoder = J2kDecoder::new(bytes).map_err(|error| failure(&error))?;
    let surface = if scale == Downscale::None {
        decoder.decode_to_device(kind.format(), BackendRequest::Cuda)
    } else {
        decoder.decode_scaled_to_device(kind.format(), scale, BackendRequest::Cuda)
    }
    .map_err(|error| failure(&error))?;
    Ok(device_samples(bytes, &surface, kind, scale))
}

fn cuda_batch_decode(bytes: &[u8], kind: Kind) -> Outcome {
    let mut session = CudaSession::default();
    let surfaces =
        J2kDecoder::decode_batch_to_device_with_session(&[bytes], kind.format(), &mut session)
            .map_err(|error| failure(&error))?;
    let [surface] = surfaces.as_slice() else {
        panic!("one surface per input");
    };
    Ok(device_samples(bytes, surface, kind, Downscale::None))
}

fn batch_options() -> BatchDecodeOptions {
    let mut options = BatchDecodeOptions::default();
    options.layout = BatchLayout::Nhwc;
    options
}

/// The CPU and CUDA decodes of a prepared batch of `bytes` alone.
fn prepared_batch_decodes(
    bytes: &[u8],
    scale: Downscale,
    cuda: &mut CudaBatchDecoder,
) -> (Outcome, Outcome) {
    let request = if scale == Downscale::None {
        DecodeRequest::Full
    } else {
        DecodeRequest::Reduced { scale }
    };
    let prepared = prepare_batch(
        vec![EncodedImage::new(Arc::from(bytes), request)],
        batch_options(),
    )
    .expect("prepare batch");
    let cpu = CpuBatchDecoder::new(batch_options())
        .decode_prepared(&prepared)
        .expect("CPU batch");
    let cpu = match (cpu.groups(), cpu.errors()) {
        ([group], []) => match group.samples() {
            CpuBatchSamples::U8(samples) => Ok(samples.clone()),
            other => panic!("8-bit samples, not {other:?}"),
        },
        ([], [error]) => Err(failure(&error.source)),
        _ => panic!("one image, one group or error"),
    };
    let cuda = match cuda.decode_prepared(&prepared) {
        Ok(result) => match (result.groups(), result.errors(), result.group_errors()) {
            ([group], [], []) => {
                Ok(Surface::download_batch_tight(group.surfaces()).expect("download batch"))
            }
            ([], [error], []) => Err(failure(&error.source)),
            // The group failed on the device.
            ([], [], [error]) => Err(failure(error.source())),
            _ => panic!("one image, one group or error"),
        },
        Err(error) => Err(failure(&error)),
    };
    (cpu, cuda)
}

/// How `outcome` differs from the CPU decode's `cpu`.
fn difference(outcome: &Outcome, cpu: &Outcome) -> String {
    match (outcome, cpu) {
        (Ok(samples), Ok(expected)) => {
            let differing = samples
                .iter()
                .zip(expected)
                .filter(|(sample, expected)| sample != expected)
                .count();
            format!(
                "{differing} of {} samples differ ({} expected)",
                samples.len(),
                expected.len()
            )
        }
        (Ok(_), Err(error)) => format!("decodes where the CPU fails with {error}"),
        (Err(error), Ok(_)) => format!("fails with {error} where the CPU decodes"),
        (Err(error), Err(expected)) => format!("fails with {error}, the CPU with {expected}"),
    }
}

/// Every way a decode of `case` departs from its expectation or from the CPU
/// decode of the same request, at full and half resolution.
fn check(case: &Case, kind: Kind, tiles: Tiles, prepared: &mut CudaBatchDecoder) -> Vec<String> {
    let scales = [(Downscale::None, case.full), (Downscale::Half, case.half)];
    check_at(case, kind, tiles, prepared, &scales)
}

/// [`check`] at each scale of `scales`, with what the decode there does.
fn check_at(
    case: &Case,
    kind: Kind,
    tiles: Tiles,
    prepared: &mut CudaBatchDecoder,
    scales: &[(Downscale, Expect)],
) -> Vec<String> {
    let mut mismatches = Vec::new();
    for &(scale, expect) in scales {
        let label = format!("{} {kind:?} {scale:?}", case.label);
        let cpu = cpu_decode(&case.edited, kind, scale);
        match expect {
            Expect::Decodes => {
                let clean = cpu_decode(&case.clean, kind, scale);
                if clean.is_err() || cpu != clean {
                    mismatches.push(format!(
                        "{label}: CPU {} of the clean codestream",
                        difference(&cpu, &clean)
                    ));
                }
            }
            Expect::Fails => {
                if cpu.is_ok() {
                    mismatches.push(format!("{label}: CPU decodes"));
                }
            }
        }
        let mut single_image = vec![("CUDA", cuda_decode(&case.edited, kind, scale))];
        if scale == Downscale::None {
            single_image.push(("CUDA batch", cuda_batch_decode(&case.edited, kind)));
        }
        let mut routes = Vec::new();
        match tiles {
            Tiles::One => routes.extend(single_image),
            Tiles::Several => {
                for (route, outcome) in single_image {
                    if !matches!(&outcome, Err(error) if error.starts_with(UNSUPPORTED)) {
                        mismatches.push(format!(
                            "{label}: {route} does not reject the tiled codestream: {}",
                            difference(&outcome, &cpu)
                        ));
                    }
                }
            }
        }
        let (cpu_prepared, cuda_prepared) = prepared_batch_decodes(&case.edited, scale, prepared);
        routes.push(("CPU prepared batch", cpu_prepared));
        routes.push(("CUDA prepared batch", cuda_prepared));
        for (route, outcome) in routes {
            if outcome != cpu {
                mismatches.push(format!("{label}: {route} {}", difference(&outcome, &cpu)));
            }
        }
    }
    mismatches
}

fn assert_parity(tiles: Tiles, cases: impl IntoIterator<Item = (Kind, Case)>) {
    let mut prepared = CudaBatchDecoder::with_options(batch_options());
    let mut mismatches = Vec::new();
    for (kind, case) in cases {
        mismatches.extend(check(&case, kind, tiles, &mut prepared));
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

fn check_all(cases: impl Fn(Kind) -> Vec<Case>) {
    assert_parity(
        Tiles::One,
        KINDS
            .into_iter()
            .flat_map(|kind| cases(kind).into_iter().map(move |case| (kind, case))),
    );
}

#[test]
fn cuda_decodes_honor_a_tile_cod_that_disables_sop_and_eph() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let cases = KINDS.into_iter().map(|kind| {
        let clean = kind.codestream(EncodeProgressionOrder::Lrcp, true);
        let case = Case {
            label: "tile COD disables main-header SOP and EPH",
            edited: with_tile_cod_disabling_sop_eph(&clean),
            clean,
            full: Expect::Decodes,
            half: Expect::Decodes,
        };
        (kind, case)
    });
    assert_parity(Tiles::One, cases);
}

#[test]
fn cuda_decodes_validate_sop_fields_only_in_packets_the_cpu_parses() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    check_all(|kind| {
        let clean = kind.encode(&EncodeOptions {
            progression_order: EncodeProgressionOrder::Lrcp,
            write_plt: true,
            write_sop: true,
            tile_part_packet_limit: Some(1),
            ..fixture_options()
        });
        let discarded_finest = usize::from(LEVELS) * kind.channels();
        vec![
            Case {
                label: "invalid Lsop in the first packet",
                edited: with_sop_marker_fields(&clean, 0, 3, 0),
                clean: clean.clone(),
                full: Expect::Fails,
                half: Expect::Fails,
            },
            Case {
                label: "invalid first Nsop",
                edited: with_sop_marker_fields(&clean, 0, 4, 1),
                clean: clean.clone(),
                full: Expect::Fails,
                half: Expect::Fails,
            },
            Case {
                label: "Nsop restarts in the second tile-part",
                edited: with_sop_marker_fields(&clean, 1, 4, 0),
                clean: clean.clone(),
                full: Expect::Fails,
                half: Expect::Fails,
            },
            Case {
                label: "invalid Nsop in a discarded finest packet",
                edited: with_sop_marker_fields(&clean, discarded_finest, 4, u16::MAX),
                clean,
                full: Expect::Fails,
                half: Expect::Decodes,
            },
        ]
    });
}

/// The position of the first `SOD` marker of a one-tile-part codestream.
fn sod(codestream: &[u8]) -> usize {
    codestream
        .windows(2)
        .position(|window| window == [0xff, 0x93])
        .expect("SOD marker")
}

/// The packet ranges of `plain`, the codestream `with_plt` encodes without its
/// PLT marker: the packets are the same bytes, after a shorter header.
fn packet_ranges_without_plt(with_plt: &[u8], plain: &[u8]) -> Vec<Range<usize>> {
    let shift = sod(with_plt) - sod(plain);
    packet_ranges(with_plt)
        .into_iter()
        .map(|range| {
            let moved = range.start - shift..range.end - shift;
            assert_eq!(plain[moved.clone()], with_plt[range], "same packet bytes");
            moved
        })
        .collect()
}

/// RPCL ranges of the packets of `resolution`, every component and layer.
fn rpcl_resolution(kind: Kind, ranges: &[Range<usize>], resolution: usize) -> Vec<Range<usize>> {
    let per_resolution = kind.packets_per_resolution();
    assert_eq!(ranges.len(), kind.packets(), "one precinct");
    ranges[resolution * per_resolution..][..per_resolution].to_vec()
}

/// LRCP ranges of the packets of `resolution` in the first layer.
fn lrcp_first_layer_resolution(
    kind: Kind,
    ranges: &[Range<usize>],
    resolution: usize,
) -> Vec<Range<usize>> {
    assert_eq!(ranges.len(), kind.packets(), "one precinct");
    ranges[resolution * kind.channels()..][..kind.channels()].to_vec()
}

#[test]
fn cuda_decodes_leave_data_after_a_tiles_last_packet_unread_as_cpu_decodes_do() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    check_all(|kind| {
        let lrcp_plt = kind.codestream(EncodeProgressionOrder::Lrcp, true);
        let rpcl = kind.codestream(EncodeProgressionOrder::Rpcl, false);
        let rpcl_plt = kind.codestream(EncodeProgressionOrder::Rpcl, true);
        vec![
            Case {
                label: "clean RPCL with PLT",
                clean: rpcl_plt.clone(),
                edited: rpcl_plt.clone(),
                full: Expect::Decodes,
                half: Expect::Decodes,
            },
            Case {
                label: "byte after the last packet, RPCL",
                edited: with_byte_after_last_packets(&rpcl),
                clean: rpcl,
                full: Expect::Decodes,
                half: Expect::Decodes,
            },
            Case {
                label: "PLT length past the last packet, LRCP",
                edited: with_surplus_packet_lengths(&lrcp_plt),
                clean: lrcp_plt,
                full: Expect::Decodes,
                half: Expect::Decodes,
            },
            Case {
                label: "byte and PLT length past the last packet, RPCL",
                edited: with_byte_after_last_packets(&with_surplus_packet_lengths(&rpcl_plt)),
                clean: rpcl_plt,
                full: Expect::Decodes,
                half: Expect::Decodes,
            },
        ]
    });
}

#[test]
fn cuda_decodes_fail_with_the_cpu_decodes_error_on_packets_they_must_read() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    check_all(|kind| {
        // A PLT length past a tile-part's last packet while the tile's
        // packets go on in the next tile-part.
        let split = kind.encode(&EncodeOptions {
            tile_part_packet_limit: Some(1),
            write_plt: true,
            ..fixture_options()
        });
        let mut first = true;
        let surplus_before_next = edit_tile_parts(&split, |part| {
            if std::mem::take(&mut first) {
                add_packet_length(part);
            }
        });
        // The tile's last packet, a finest-level one, with its non-empty bit
        // cleared: its header ends short of its PLT length. A reduced decode
        // does not read it.
        let one_layer = kind.encode(&EncodeOptions {
            progression_order: EncodeProgressionOrder::Rpcl,
            num_layers: 1,
            write_plt: true,
            ..fixture_options()
        });
        let last = packet_ranges(&one_layer).pop().expect("packets").start;
        let mut short_last = one_layer.clone();
        assert_ne!(short_last[last] & 0x80, 0, "the last packet is not empty");
        short_last[last] &= 0x7f;
        let rpcl_plt = kind.codestream(EncodeProgressionOrder::Rpcl, true);
        let first_packet = packet_ranges(&rpcl_plt)[..1].to_vec();
        vec![
            Case {
                label: "PLT length past a tile-part's last packet before the next",
                clean: split,
                edited: surplus_before_next,
                full: Expect::Fails,
                half: Expect::Fails,
            },
            Case {
                label: "last packet shorter than its PLT length",
                clean: one_layer,
                edited: short_last,
                full: Expect::Fails,
                half: Expect::Decodes,
            },
            Case {
                label: "first packet filled",
                edited: with_packets_filled(&rpcl_plt, &first_packet, 0xff),
                clean: rpcl_plt,
                full: Expect::Fails,
                half: Expect::Fails,
            },
        ]
    });
}

#[test]
fn cuda_reduced_decodes_read_only_the_packets_cpu_decodes_read() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    check_all(|kind| {
        let rpcl_plt = kind.codestream(EncodeProgressionOrder::Rpcl, true);
        let rpcl = kind.codestream(EncodeProgressionOrder::Rpcl, false);
        let lrcp_plt = kind.codestream(EncodeProgressionOrder::Lrcp, true);
        let lrcp = kind.codestream(EncodeProgressionOrder::Lrcp, false);
        let rpcl_plt_ranges = packet_ranges(&rpcl_plt);
        let rpcl_ranges = packet_ranges_without_plt(&rpcl_plt, &rpcl);
        let lrcp_plt_ranges = packet_ranges(&lrcp_plt);
        let lrcp_ranges = packet_ranges_without_plt(&lrcp_plt, &lrcp);
        let finest = usize::from(LEVELS);
        let filled = |codestream: &[u8], ranges: Vec<Range<usize>>| {
            with_packets_filled(codestream, &ranges, 0xff)
        };
        vec![
            Case {
                label: "finest-level packets filled, RPCL with PLT",
                edited: filled(&rpcl_plt, rpcl_resolution(kind, &rpcl_plt_ranges, finest)),
                clean: rpcl_plt.clone(),
                full: Expect::Fails,
                half: Expect::Decodes,
            },
            Case {
                label: "finest-level packets filled, RPCL",
                edited: filled(&rpcl, rpcl_resolution(kind, &rpcl_ranges, finest)),
                clean: rpcl.clone(),
                full: Expect::Fails,
                half: Expect::Decodes,
            },
            Case {
                label: "first-layer finest-level packets filled, LRCP with PLT",
                edited: filled(
                    &lrcp_plt,
                    lrcp_first_layer_resolution(kind, &lrcp_plt_ranges, finest),
                ),
                clean: lrcp_plt,
                full: Expect::Fails,
                half: Expect::Decodes,
            },
            Case {
                label: "first-layer finest-level packets filled, LRCP",
                edited: filled(
                    &lrcp,
                    lrcp_first_layer_resolution(kind, &lrcp_ranges, finest),
                ),
                clean: lrcp,
                full: Expect::Fails,
                half: Expect::Fails,
            },
            Case {
                label: "coarsest-level packets filled, RPCL with PLT",
                edited: filled(&rpcl_plt, rpcl_resolution(kind, &rpcl_plt_ranges, 0)),
                clean: rpcl_plt,
                full: Expect::Fails,
                half: Expect::Fails,
            },
            Case {
                label: "coarsest-level packets filled, RPCL",
                edited: filled(&rpcl, rpcl_resolution(kind, &rpcl_ranges, 0)),
                clean: rpcl,
                full: Expect::Fails,
                half: Expect::Fails,
            },
        ]
    });
}

/// A tile of one sample at an image corner has empty detail sub-bands, so its
/// full-resolution precinct holds no code-block, and its packet, the tile's
/// last, nothing a decode uses. With that packet's EPH marker damaged, the
/// image decodes as the clean one on the prepared batch, of tiled images, as
/// on the CPU.
#[test]
fn cuda_decodes_leave_packets_of_precincts_without_code_blocks_unread_as_cpu_decodes_do() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let clean = include_bytes!(
        "../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_gray_3x3_tile2x2_53_eph.j2k"
    )
    .to_vec();
    let mut edited = clean.clone();
    let eph = edited.len() - 4;
    assert_eq!(
        edited[eph..],
        [0xff, 0x92, 0xff, 0xd9],
        "the last packet ends in EPH"
    );
    edited[eph + 1] = 0;
    let kind = Kind {
        components: 1,
        ht: false,
    };
    assert_parity(
        Tiles::Several,
        [(
            kind,
            Case {
                label: "EPH of an empty precinct's packet damaged",
                clean,
                edited,
                full: Expect::Decodes,
                half: Expect::Decodes,
            },
        )],
    );
}

/// Every packet holds at least the first byte of its header, so a zero PLT
/// length is no packet's. In a flat image, whose finest packets are empty, a
/// zero length for a finest packet that a half-resolution decode skips, and a
/// length of one for the needed packet after it, would have that decode read
/// the needed packet from the skipped one's byte: in RGB, CPRL, the second
/// component's finest packet and the third component's coarsest; in gray,
/// LRCP with two layers, the first layer's finest packet and the second
/// layer's coarsest. The CUDA decodes fail as the CPU decodes do, at both
/// resolutions.
#[test]
fn cuda_decodes_reject_a_zero_plt_length_before_a_needed_packet_as_cpu_decodes_do() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    check_all(|kind| {
        let (progression, layers, skipped) = if kind.components == 1 {
            (EncodeProgressionOrder::Lrcp, 2, 1)
        } else {
            (EncodeProgressionOrder::Cprl, 1, 3)
        };
        let clean = kind.encode_flat(&EncodeOptions {
            num_decomposition_levels: 1,
            num_layers: layers,
            progression_order: progression,
            write_plt: true,
            ..fixture_options()
        });
        // One precinct per resolution: the packets alternate between the
        // coarsest and the finest resolution, whose packets are empty.
        let lengths: Vec<usize> = packet_ranges(&clean).iter().map(Range::len).collect();
        assert_eq!(
            lengths.len(),
            kind.channels() * 2 * usize::from(layers),
            "two resolutions"
        );
        assert!(
            lengths.iter().skip(1).step_by(2).all(|&length| length == 1),
            "empty finest packets: {lengths:?}"
        );
        let zero = with_packet_lengths_edited(&clean, |lengths| {
            let total: usize = lengths.iter().sum();
            lengths[skipped] = 0;
            lengths[skipped + 1] = 1;
            let last = lengths.len() - 1;
            lengths[last] = total - lengths[..last].iter().sum::<usize>();
        });
        vec![Case {
            label: "zero PLT length before a needed packet",
            clean,
            edited: zero,
            full: Expect::Fails,
            half: Expect::Fails,
        }]
    });
}

/// With packet headers packed in PPM or PPT, PLT lengths hold only packet
/// bodies, so a reduced decode parses every header up to the last packet it
/// needs. In a flat image, whose packets after each component's first hold
/// one-byte empty headers, a header damaged to include a code-block reads
/// that block's zero bit-plane count from the zero bytes after it and runs
/// out of packed headers: damage to the header of the first layer's last
/// finest packet, before the second layer's packets, fails the
/// half-resolution decodes; damage to the last packet's header, after every
/// packet they need, leaves them the undamaged image. Both fail at full
/// resolution.
#[test]
fn cuda_decodes_parse_damaged_packed_packet_headers_as_cpu_decodes_do() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    check_all(|kind| {
        let mut cases = Vec::new();
        for (labels, write_ppm, write_ppt) in [
            (
                [
                    "PPM: first layer's finest header damaged",
                    "PPM: last header damaged",
                ],
                true,
                false,
            ),
            (
                [
                    "PPT: first layer's finest header damaged",
                    "PPT: last header damaged",
                ],
                false,
                true,
            ),
        ] {
            let clean = kind.encode_flat(&EncodeOptions {
                // Every coding pass in the first layer: an HTJ2K block's
                // cleanup pass is there without targets.
                quality_layer_byte_targets: if kind.ht {
                    Vec::new()
                } else {
                    vec![1, u64::MAX]
                },
                progression_order: EncodeProgressionOrder::Lrcp,
                write_plt: true,
                write_ppm,
                write_ppt,
                ..fixture_options()
            });
            // LRCP, one precinct per resolution: the first layer's coarsest
            // headers hold the image, every later one is empty.
            let headers = packed_packet_headers(&clean);
            let empty = kind.packets() - kind.channels();
            assert!(
                clean[headers.end - empty..headers.end]
                    .iter()
                    .all(|&byte| byte == 0),
                "{}: empty headers",
                labels[0]
            );
            let damaged_at = |from_end: usize| {
                let mut damaged = clean.clone();
                // Not empty, its first code-block included.
                damaged[headers.end - from_end] = 0xc0;
                damaged
            };
            let [navigation, last] = labels;
            cases.push(Case {
                label: navigation,
                edited: damaged_at(kind.packets_per_layer() + 1),
                clean: clean.clone(),
                full: Expect::Fails,
                half: Expect::Fails,
            });
            cases.push(Case {
                label: last,
                edited: damaged_at(1),
                clean,
                full: Expect::Fails,
                half: Expect::Decodes,
            });
        }
        cases
    });
}

/// The cases of one interleaved tile layout: `width` x `height` pixels of
/// `kind` in `tiles` tiles of `tile` x `tile`, one RPCL packet per tile-part.
fn interleaved_cases(
    kind: Kind,
    (width, height, tile): (u32, u32, u32),
    tiles: usize,
    labels: [&'static str; 3],
) -> [(Kind, Case); 3] {
    let clean = kind.encode_sized(
        width,
        height,
        &EncodeOptions {
            progression_order: EncodeProgressionOrder::Rpcl,
            tile_size: Some((tile, tile)),
            tile_part_packet_limit: Some(1),
            write_plt: true,
            ..fixture_options()
        },
    );
    let ranges = packet_ranges(&clean);
    assert_eq!(
        ranges.len(),
        tiles * kind.packets(),
        "{}: one-precinct tiles",
        labels[0]
    );
    let resolution = |level: usize| -> Vec<Range<usize>> {
        ranges
            .iter()
            .enumerate()
            .filter(|(index, _)| index % kind.packets() / kind.packets_per_resolution() == level)
            .map(|(_, range)| range.clone())
            .collect()
    };
    let interleaved = with_tile_parts_interleaved(&clean);
    assert_ne!(
        interleaved, clean,
        "{}: tile-parts change places",
        labels[0]
    );
    let filled =
        |level| with_tile_parts_interleaved(&with_packets_filled(&clean, &resolution(level), 0xff));
    let finest = filled(usize::from(LEVELS));
    let coarsest = filled(0);
    let [moved, finest_filled, coarsest_filled] = labels;
    [
        Case {
            label: moved,
            clean: clean.clone(),
            edited: interleaved.clone(),
            full: Expect::Decodes,
            half: Expect::Decodes,
        },
        Case {
            label: finest_filled,
            clean: interleaved.clone(),
            edited: finest,
            full: Expect::Fails,
            half: Expect::Decodes,
        },
        Case {
            label: coarsest_filled,
            clean: interleaved,
            edited: coarsest,
            full: Expect::Fails,
            half: Expect::Fails,
        },
    ]
    .map(|case| (kind, case))
}

/// Tile-parts of several tiles may take turns in the codestream, and a
/// reduced decode completes each tile across its tile-parts. With one RPCL
/// packet per tile-part, interleaving keeps the image; damage to every tile's
/// finest packets, in the last rounds of tile-parts, leaves the
/// half-resolution decode the undamaged image, and damage to every tile's
/// coarsest packets, in the first rounds, fails it; both fail at full
/// resolution. Six 32 x 32 tiles of every kind, and eight 512 x 512 tiles of
/// a 2048 x 1024 gray image. Direct plans take one tile, so the single-image
/// CUDA decodes reject these codestreams as unsupported, and the prepared
/// batch decodes them on the device as the CPU does.
#[test]
fn cuda_prepared_batches_read_interleaved_tile_parts_as_cpu_decodes_do() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let small = KINDS.into_iter().flat_map(|kind| {
        interleaved_cases(
            kind,
            (WIDTH, HEIGHT, 32),
            6,
            [
                "32 x 32 tiles interleaved",
                "32 x 32 tiles interleaved, finest-level packets filled",
                "32 x 32 tiles interleaved, coarsest-level packets filled",
            ],
        )
    });
    let large = KINDS
        .into_iter()
        .filter(|kind| kind.components == 1)
        .flat_map(|kind| {
            interleaved_cases(
                kind,
                (2048, 1024, 512),
                8,
                [
                    "512 x 512 tiles interleaved",
                    "512 x 512 tiles interleaved, finest-level packets filled",
                    "512 x 512 tiles interleaved, coarsest-level packets filled",
                ],
            )
        });
    assert_parity(Tiles::Several, small.chain(large));
}

/// With EPH markers every packet holds at least three bytes, its header's
/// first byte and the marker, so PLT lengths of one and two are no packet's.
/// Skipping by them would misplace the next needed packet as a zero length
/// does. In a flat image with three resolutions, lengths of one, two and
/// three for two finer packets that a quarter-resolution decode skips and
/// the needed packet after them would have that decode read the needed one
/// from the second skipped packet's bytes: in RGB, CPRL, the second
/// component's finer packets and the third component's coarsest; in gray,
/// LRCP with two layers, the first layer's finer packets and the second
/// layer's coarsest. The CUDA decodes fail as the CPU decodes do, at full,
/// half and quarter resolution.
#[test]
fn cuda_decodes_reject_plt_lengths_shorter_than_an_eph_packet_as_cpu_decodes_do() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let mut prepared = CudaBatchDecoder::with_options(batch_options());
    let mut mismatches = Vec::new();
    for kind in KINDS {
        let (progression, layers, skipped) = if kind.components == 1 {
            (EncodeProgressionOrder::Lrcp, 2, 1)
        } else {
            (EncodeProgressionOrder::Cprl, 1, 4)
        };
        let clean = kind.encode_flat(&EncodeOptions {
            num_layers: layers,
            progression_order: progression,
            write_plt: true,
            write_eph: true,
            ..fixture_options()
        });
        // One precinct per resolution: the packets come in threes, coarsest
        // first; the finer ones are empty, a byte and EPH.
        let lengths: Vec<usize> = packet_ranges(&clean).iter().map(Range::len).collect();
        assert_eq!(
            lengths.len(),
            kind.channels() * 3 * usize::from(layers),
            "three resolutions"
        );
        assert!(
            lengths
                .iter()
                .enumerate()
                .all(|(index, &length)| index % 3 == 0 || length == 3),
            "empty finer packets: {lengths:?}"
        );
        let short = with_packet_lengths_edited(&clean, |lengths| {
            let total: usize = lengths.iter().sum();
            lengths[skipped..skipped + 3].copy_from_slice(&[1, 2, 3]);
            let last = lengths.len() - 1;
            lengths[last] = total - lengths[..last].iter().sum::<usize>();
        });
        let case = Case {
            label: "PLT lengths shorter than an EPH packet",
            clean,
            edited: short,
            full: Expect::Fails,
            half: Expect::Fails,
        };
        let scales = [
            (Downscale::None, Expect::Fails),
            (Downscale::Half, Expect::Fails),
            (Downscale::Quarter, Expect::Fails),
        ];
        mismatches.extend(check_at(&case, kind, Tiles::One, &mut prepared, &scales));
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// `codestream` with one guard bit fewer in each QCD and QCC marker of its
/// main header: every sub-band then codes one bit-plane fewer.
fn with_guard_bits_lowered(codestream: &[u8]) -> Vec<u8> {
    let mut edited = codestream.to_vec();
    let mut lowered = 0;
    let mut at = 2;
    while codestream[at..at + 2] != [0xff, 0x90] {
        // Sqcd follows Lqcd; Sqcc follows Lqcc and a one-byte Cqcc.
        let style = match codestream[at + 1] {
            0x5c => Some(at + 4),
            0x5d => Some(at + 5),
            _ => None,
        };
        if let Some(style) = style {
            assert_ne!(codestream[style] >> 5, 0, "a guard bit to remove");
            edited[style] -= 1 << 5;
            lowered += 1;
        }
        at += 2 + usize::from(u16::from_be_bytes([codestream[at + 2], codestream[at + 3]]));
    }
    assert_ne!(lowered, 0, "a QCD marker");
    edited
}

/// A strict decode rejects a code-block whose packet headers claim more
/// coding passes than its bit-planes allow. With the guard bits lowered by
/// one, every coded block of a lossless codestream has one bit-plane's
/// passes too many, a codestream error: the CUDA decodes of classic
/// code-blocks fail with the CPU decode's native error, at both resolutions.
#[test]
fn cuda_classic_decodes_reject_more_coding_passes_than_bit_planes_as_cpu_decodes_do() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let cases = KINDS.into_iter().filter(|kind| !kind.ht).map(|kind| {
        let clean = kind.codestream(EncodeProgressionOrder::Lrcp, false);
        let case = Case {
            label: "QCD guard bits lowered",
            edited: with_guard_bits_lowered(&clean),
            clean,
            full: Expect::Fails,
            half: Expect::Fails,
        };
        (kind, case)
    });
    assert_parity(Tiles::One, cases);
}
