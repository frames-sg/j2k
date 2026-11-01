// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA decodes read a codestream's packets as the CPU decodes do. Data after
//! a tile's last packet, PLT lengths past it, and damage in packets that a
//! reduced-resolution decode does not read leave the samples of the clean
//! codestream; damage in a packet the decode must read fails the CUDA decode
//! with the CPU decode's native error. Each case runs on the single-image
//! decode, the batch of images and the prepared batch.

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
    add_packet_length, cuda_runtime_and_strict_oxide_gate, edit_tile_parts, packet_ranges,
    with_byte_after_last_packets, with_packets_filled, with_surplus_packet_lengths,
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
        let mut pixels = Vec::new();
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                for c in 0..u32::from(self.components) {
                    let noise = (x.wrapping_mul(2_654_435_761) ^ (y * 40_503) ^ (c * 977)) >> 13;
                    pixels.push((x * 5 + y * 3 + c * 61 + noise % 37).to_le_bytes()[0]);
                }
            }
        }
        let encoder = if self.ht { encode_htj2k } else { encode };
        encoder(&pixels, WIDTH, HEIGHT, self.components, 8, false, options).expect("encode fixture")
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
        Ok(result) => match (result.groups(), result.errors()) {
            ([group], []) => {
                Ok(Surface::download_batch_tight(group.surfaces()).expect("download batch"))
            }
            ([], [error]) => Err(failure(&error.source)),
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
/// decode of the same request.
fn check(case: &Case, kind: Kind, tiles: Tiles, prepared: &mut CudaBatchDecoder) -> Vec<String> {
    let mut mismatches = Vec::new();
    for (scale, expect) in [(Downscale::None, case.full), (Downscale::Half, case.half)] {
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
