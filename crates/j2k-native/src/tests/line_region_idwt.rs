// SPDX-License-Identifier: MIT OR Apache-2.0

//! The row-streamed region IDWT behind `DecoderContext::set_line_region_idwt`
//! must match the whole-window region IDWT bit for bit, and must leave every
//! output it does not cover on the whole-window path.

use alloc::vec::Vec;

use crate::{
    encode, encode_component_planes_53, DecodeSettings, DecoderContext, EncodeComponentPlane,
    EncodeOptions, Image,
};

const WIDTH: u32 = 203;
const HEIGHT: u32 = 157;
const TILE: u32 = 64;

/// Deterministic noise, so every sub-band carries significant coefficients.
fn noisy_pixels(components: u8, bytes_per_sample: u8) -> Vec<u8> {
    let mut state = 0x6c07_8965_u32;
    (0..WIDTH * HEIGHT * u32::from(components) * u32::from(bytes_per_sample))
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

fn codestream(components: u8, bit_depth: u8, reversible: bool, tiled: bool) -> Vec<u8> {
    encode(
        &noisy_pixels(components, bit_depth.div_ceil(8)),
        WIDTH,
        HEIGHT,
        u16::from(components),
        bit_depth,
        false,
        &EncodeOptions {
            reversible,
            num_decomposition_levels: 4,
            tile_size: tiled.then_some((TILE, TILE)),
            ..EncodeOptions::default()
        },
    )
    .expect("encode line IDWT fixture")
}

struct Decoded {
    packed: Vec<u8>,
    native: Vec<u8>,
    peak_idwt: usize,
}

fn decode(image: &Image<'_>, roi: (u32, u32, u32, u32), lines: bool) -> Decoded {
    let mut context = DecoderContext::default();
    context.set_line_region_idwt(lines);
    let native = image
        .decode_native_region_with_context(roi, &mut context)
        .expect("decode line IDWT native region")
        .data;
    let peak_idwt = context
        .tile_decode_context
        .debug_counters
        .workspace_peak
        .bytes()
        .idwt;
    let packed = image
        .decode_region_with_context(roi, &mut context)
        .expect("decode line IDWT region")
        .data;
    Decoded {
        packed,
        native,
        peak_idwt,
    }
}

#[test]
fn line_region_idwt_matches_the_whole_window_idwt_bit_for_bit() {
    for (components, bit_depth) in [(1_u8, 8_u8), (3, 8), (1, 12)] {
        for reversible in [true, false] {
            for tiled in [false, true] {
                // 12-bit samples differ only in rounding, so one layout suffices.
                if bit_depth == 12 && tiled {
                    continue;
                }
                let bytes = codestream(components, bit_depth, reversible, tiled);
                for reduction in [0_u8, 1, 2] {
                    let image =
                        Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                            .expect("parse line IDWT fixture");
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
                        (
                            width / 3,
                            height / 4,
                            width - width / 3,
                            height - height / 4,
                        ),
                        (width / 3, height / 3, width / 7, height / 9),
                        // Inside one tile of the tiled fixture, and across four.
                        (tile + 1, tile + 2, tile - 3, tile - 4),
                        (tile - 5, tile - 3, 11, 7),
                        (width - 1, height - 1, 1, 1),
                    ];
                    for roi in regions {
                        let label = format!(
                            "components={components} bit_depth={bit_depth} \
                             reversible={reversible} tiled={tiled} reduction={reduction} \
                             roi={roi:?}"
                        );
                        let windows = decode(&image, roi, false);
                        let lines = decode(&image, roi, true);
                        assert_eq!(lines.packed, windows.packed, "{label}");
                        assert_eq!(lines.native, windows.native, "{label}");
                        if roi == large && !tiled && reduction == 0 {
                            assert!(
                                lines.peak_idwt * 4 < windows.peak_idwt,
                                "{label}: idwt {} vs {}",
                                lines.peak_idwt,
                                windows.peak_idwt
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn component_plane_outputs_keep_the_whole_window_idwt() {
    let bytes = codestream(3, 8, false, false);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse");
    let roi = (10, 10, 150, 120);
    let peak = |lines: bool| {
        let mut context = DecoderContext::default();
        context.set_line_region_idwt(lines);
        drop(
            image
                .decode_region_components_with_context(roi, &mut context)
                .expect("component planes"),
        );
        let planes_peak = context
            .tile_decode_context
            .debug_counters
            .workspace_peak
            .bytes()
            .idwt;
        drop(
            image
                .decode_region_components_for_integer_output_with_context(roi, &mut context)
                .expect("integer-output component planes"),
        );
        let integer_peak = context
            .tile_decode_context
            .debug_counters
            .workspace_peak
            .bytes()
            .idwt;
        (planes_peak, integer_peak)
    };
    assert_eq!(peak(true), peak(false));

    // The same request through a packed output does take the streamed path.
    let mut context = DecoderContext::default();
    context.set_line_region_idwt(true);
    image
        .decode_region_with_context(roi, &mut context)
        .expect("packed region");
    let packed_peak = context
        .tile_decode_context
        .debug_counters
        .workspace_peak
        .bytes()
        .idwt;
    assert!(packed_peak < peak(false).0);
}

fn jp2_box(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let len = u32::try_from(payload.len() + 8).expect("test box length fits u32");
    let mut out = len.to_be_bytes().to_vec();
    out.extend_from_slice(&kind);
    out.extend_from_slice(payload);
    out
}

/// A JP2 file around `codestream` with an enumerated `colr` box and any
/// further header boxes.
fn jp2(codestream: &[u8], components: u16, colorspace: &[u8], extra_boxes: &[Vec<u8>]) -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&HEIGHT.to_be_bytes());
    ihdr.extend_from_slice(&WIDTH.to_be_bytes());
    ihdr.extend_from_slice(&components.to_be_bytes());
    ihdr.extend_from_slice(&[7, 7, 0, 0]);
    let mut colr = vec![1, 0, 0];
    colr.extend_from_slice(colorspace);
    let mut jp2h = jp2_box(*b"ihdr", &ihdr);
    jp2h.extend_from_slice(&jp2_box(*b"colr", &colr));
    for extra in extra_boxes {
        jp2h.extend_from_slice(extra);
    }
    let mut out = jp2_box(*b"jP  ", &[0x0d, 0x0a, 0x87, 0x0a]);
    out.extend_from_slice(&jp2_box(*b"ftyp", b"jp2 \0\0\0\0jp2 "));
    out.extend_from_slice(&jp2_box(*b"jp2h", &jp2h));
    out.extend_from_slice(&jp2_box(*b"jp2c", codestream));
    out
}

fn enumerated(colorspace: u32) -> Vec<u8> {
    colorspace.to_be_bytes().to_vec()
}

/// Channel definitions that keep three colour channels in order.
fn cdef_box() -> Vec<u8> {
    let mut cdef = 3_u16.to_be_bytes().to_vec();
    for channel in 0_u16..3 {
        for value in [channel, 0, channel + 1] {
            cdef.extend_from_slice(&value.to_be_bytes());
        }
    }
    jp2_box(*b"cdef", &cdef)
}

#[test]
fn line_region_idwt_skips_images_with_colour_post_processing() {
    let rgb = codestream(3, 8, false, false);
    let gray = codestream(1, 8, true, false);
    let srgb = enumerated(16);
    let mut cielab = enumerated(14);
    cielab.extend_from_slice(&[0; 7 * 4]);
    // A two-entry RGB palette for the gray component.
    let mut pclr = 2_u16.to_be_bytes().to_vec();
    pclr.extend_from_slice(&[3, 7, 7, 7, 0, 0, 0, 255, 255, 255]);
    let mut cmap = Vec::new();
    for column in 0..3 {
        cmap.extend_from_slice(&[0, 0, 1, column]);
    }
    let palette = [jp2_box(*b"pclr", &pclr), jp2_box(*b"cmap", &cmap)];

    let cases = [
        ("raw codestream", rgb.clone(), true),
        ("sRGB", jp2(&rgb, 3, &srgb, &[]), true),
        ("sYCC", jp2(&rgb, 3, &enumerated(18), &[]), false),
        ("CIELab", jp2(&rgb, 3, &cielab, &[]), false),
        ("cdef", jp2(&rgb, 3, &srgb, &[cdef_box()]), false),
        ("palette", jp2(&gray, 1, &srgb, &palette), false),
    ];
    for (label, bytes, permitted) in cases {
        let image = Image::new(&bytes, &DecodeSettings::default()).expect(label);
        assert_eq!(image.permits_line_region_idwt(), permitted, "{label}");
    }

    // A skipped image decodes as it would without the opt-in.
    let bytes = jp2(&rgb, 3, &srgb, &[cdef_box()]);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("cdef image");
    let roi = (10, 10, 150, 120);
    let lines = decode(&image, roi, true);
    let windows = decode(&image, roi, false);
    assert_eq!(lines.packed, windows.packed);
    assert_eq!(lines.peak_idwt, windows.peak_idwt);
}

#[test]
fn prepared_region_decoder_keeps_the_whole_window_idwt() {
    let bytes = codestream(3, 8, false, false);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse");
    let peak = |lines: bool| {
        let mut context = DecoderContext::default();
        context.set_line_region_idwt(lines);
        let mut prepared = image
            .prepare_region_decoder_with_context(&mut context)
            .expect("prepare");
        drop(
            prepared
                .decode_region_components((10, 10, 150, 120))
                .expect("prepared region"),
        );
        drop(prepared);
        context
            .tile_decode_context
            .debug_counters
            .workspace_peak
            .bytes()
            .idwt
    };
    assert_eq!(peak(true), peak(false));
}

#[test]
fn line_region_idwt_reuses_its_rings_across_decodes() {
    let bytes = codestream(3, 8, false, false);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse");
    let roi = (10, 10, 150, 120);
    let mut context = DecoderContext::default();
    context.set_line_region_idwt(true);
    let first = image
        .decode_region_with_context(roi, &mut context)
        .expect("first decode")
        .data;
    let rings = context.tile_decode_context.idwt_line_rings.capacity();
    assert!(rings > 0);
    let second = image
        .decode_region_with_context(roi, &mut context)
        .expect("second decode")
        .data;
    assert_eq!(second, first);
    assert_eq!(
        context.tile_decode_context.idwt_line_rings.capacity(),
        rings
    );

    // Turning the opt-in off between calls returns to the whole-window path.
    context.set_line_region_idwt(false);
    let windows = image
        .decode_region_with_context(roi, &mut context)
        .expect("whole-window decode")
        .data;
    assert_eq!(windows, first);
}

#[test]
fn line_region_idwt_skips_subsampled_images() {
    let luma = noisy_pixels(1, 1);
    let chroma = luma
        .iter()
        .step_by(3)
        .copied()
        .take((WIDTH.div_ceil(2) * HEIGHT.div_ceil(2)) as usize)
        .collect::<Vec<_>>();
    let plane = |data, sampling| EncodeComponentPlane {
        data,
        x_rsiz: sampling,
        y_rsiz: sampling,
    };
    let bytes = encode_component_planes_53(
        &[plane(&luma, 1), plane(&chroma, 2), plane(&chroma, 2)],
        WIDTH,
        HEIGHT,
        8,
        false,
        &EncodeOptions {
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("encode subsampled fixture");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse");
    let roi = (10, 10, 150, 120);
    let lines = decode(&image, roi, true);
    let windows = decode(&image, roi, false);
    assert_eq!(lines.packed, windows.packed);
    assert_eq!(lines.peak_idwt, windows.peak_idwt);
}
