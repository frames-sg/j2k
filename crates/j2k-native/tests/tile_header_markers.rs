// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::Path;

use j2k_native::{
    encode, encode_with_roi_regions, CpuDecodeParallelism, DecodeSettings, DecoderContext,
    EncodeError, EncodeOptions, EncodeRoiRegion, Image, RawBitmap,
};

fn marker_offset(codestream: &[u8], marker: u8) -> usize {
    codestream
        .windows(2)
        .position(|window| window == [0xFF, marker])
        .unwrap_or_else(|| panic!("missing marker FF{marker:02X}"))
}

fn patterned_pixels(len: usize) -> Vec<u8> {
    (0..len)
        .map(|index| u8::try_from(index % 251).expect("pattern value fits u8"))
        .collect()
}

fn cod_marker_end(codestream: &[u8]) -> usize {
    let cod_offset = marker_offset(codestream, 0x52);
    let lcod = u16::from_be_bytes([codestream[cod_offset + 2], codestream[cod_offset + 3]]);
    cod_offset + 2 + usize::from(lcod)
}

fn insert_main_header_rgn(mut codestream: Vec<u8>, roi_shift: u8, style: u8) -> Vec<u8> {
    let rgn = [
        0xFF, 0x5E, // RGN
        0x00, 0x05,      // Lrgn
        0x00,      // Crgn
        style,     // Srgn
        roi_shift, // SPrgn
    ];
    codestream.splice(
        cod_marker_end(&codestream)..cod_marker_end(&codestream),
        rgn,
    );
    codestream
}

fn insert_tile_header_rgn(mut codestream: Vec<u8>, roi_shift: u8, style: u8) -> Vec<u8> {
    let sod_offset = marker_offset(&codestream, 0x93);
    let rgn = [
        0xFF, 0x5E, // RGN
        0x00, 0x05,      // Lrgn
        0x00,      // Crgn
        style,     // Srgn
        roi_shift, // SPrgn
    ];
    codestream.splice(sod_offset..sod_offset, rgn);

    let sot_offset = marker_offset(&codestream, 0x90);
    let psot = u32::from_be_bytes([
        codestream[sot_offset + 6],
        codestream[sot_offset + 7],
        codestream[sot_offset + 8],
        codestream[sot_offset + 9],
    ]);
    codestream[sot_offset + 6..sot_offset + 10]
        .copy_from_slice(&(psot + u32::try_from(rgn.len()).unwrap()).to_be_bytes());

    codestream
}

#[test]
fn tile_header_rgn_marker_with_zero_shift_is_a_noop() {
    let pixels = patterned_pixels(64 * 64);
    let codestream = encode(&pixels, 64, 64, 1, 8, false, &EncodeOptions::default())
        .expect("lossless fixture encode");
    let with_rgn = insert_tile_header_rgn(codestream, 0, 0);

    let decoded = Image::new(&with_rgn, &DecodeSettings::default())
        .expect("tile-header RGN codestream parses")
        .decode_native()
        .expect("tile-header RGN codestream decodes");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn encode_whole_component_roi_maxshift_roundtrips_and_writes_rgn() {
    let pixels = patterned_pixels(64 * 64);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        roi_component_shifts: vec![3],
        ..EncodeOptions::default()
    };

    let codestream =
        encode(&pixels, 64, 64, 1, 8, false, &options).expect("whole-component ROI fixture encode");
    let rgn = marker_offset(&codestream, 0x5E);

    assert_eq!(
        &codestream[rgn + 2..rgn + 7],
        &[0x00, 0x05, 0x00, 0x00, 0x03]
    );

    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("encoded RGN codestream parses")
        .decode_native()
        .expect("encoded RGN codestream decodes");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn encode_rectangular_roi_maxshift_roundtrips_and_writes_rgn() {
    let pixels = patterned_pixels(64 * 64);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let roi = [EncodeRoiRegion {
        component: 0,
        x: 8,
        y: 12,
        width: 24,
        height: 20,
        shift: 12,
    }];

    let codestream = encode_with_roi_regions(&pixels, 64, 64, 1, 8, false, &options, &roi)
        .expect("rectangular ROI fixture encode");
    let rgn = marker_offset(&codestream, 0x5E);

    assert_eq!(
        &codestream[rgn + 2..rgn + 7],
        &[0x00, 0x05, 0x00, 0x00, 0x0C]
    );

    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("encoded rectangular RGN codestream parses")
        .decode_native()
        .expect("encoded rectangular RGN codestream decodes");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn encode_rejects_ambiguous_whole_component_and_rectangular_roi() {
    let pixels = patterned_pixels(16 * 16);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        roi_component_shifts: vec![4],
        ..EncodeOptions::default()
    };
    let roi = [EncodeRoiRegion {
        component: 0,
        x: 0,
        y: 0,
        width: 8,
        height: 8,
        shift: 4,
    }];

    let err = encode_with_roi_regions(&pixels, 16, 16, 1, 8, false, &options, &roi)
        .expect_err("ambiguous ROI request must fail");

    assert_eq!(
        err,
        EncodeError::InvalidInput {
            what: "ROI region cannot be combined with whole-component ROI shift"
        }
    );
}

#[test]
fn tile_header_rgn_with_explicit_style_is_rejected() {
    let pixels = patterned_pixels(64 * 64);
    let codestream = encode(&pixels, 64, 64, 1, 8, false, &EncodeOptions::default())
        .expect("lossless fixture encode");
    let with_rgn = insert_tile_header_rgn(codestream, 7, 1);

    let err = Image::new(&with_rgn, &DecodeSettings::default())
        .expect("tile-header RGN codestream parses")
        .decode_native()
        .err()
        .expect("explicit ROI coding style is unsupported");

    assert_eq!(
        err.to_string(),
        "unsupported decoding feature: explicit ROI coding"
    );
}

#[test]
fn main_header_rgn_with_explicit_style_is_rejected() {
    let pixels = patterned_pixels(64 * 64);
    let codestream = encode(&pixels, 64, 64, 1, 8, false, &EncodeOptions::default())
        .expect("lossless fixture encode");
    let with_rgn = insert_main_header_rgn(codestream, 7, 1);

    let err = Image::new(&with_rgn, &DecodeSettings::default())
        .err()
        .expect("explicit ROI coding style is unsupported");

    assert_eq!(
        err.to_string(),
        "unsupported decoding feature: explicit ROI coding"
    );
}

#[test]
fn crafted_siz_with_absurd_tile_grid_is_rejected_without_allocating() {
    let pixels = patterned_pixels(16 * 16);
    let mut codestream = encode(&pixels, 16, 16, 1, 8, false, &EncodeOptions::default())
        .expect("lossless fixture encode");

    // Rewrite SIZ to a 15,300,000² reference grid of 1×1 tiles. The component
    // resolutions go to 255 so image_width stays at the 60,000 dimension cap,
    // but the tile grid implies ~2.3e14 tiles — overflowing num_tiles() and
    // driving the eager per-tile allocation in tile parsing.
    let siz = codestream
        .windows(2)
        .position(|w| w == [0xFF, 0x51])
        .expect("SIZ marker");
    codestream[siz + 6..siz + 10].copy_from_slice(&15_300_000u32.to_be_bytes()); // Xsiz
    codestream[siz + 10..siz + 14].copy_from_slice(&15_300_000u32.to_be_bytes()); // Ysiz
    codestream[siz + 22..siz + 26].copy_from_slice(&1u32.to_be_bytes()); // XTsiz
    codestream[siz + 26..siz + 30].copy_from_slice(&1u32.to_be_bytes()); // YTsiz
    codestream[siz + 41] = 255; // XRsiz (component 0)
    codestream[siz + 42] = 255; // YRsiz (component 0)

    let result =
        Image::new(&codestream, &DecodeSettings::default()).and_then(|image| image.decode_native());
    let err = result.err().expect("absurd tile grid must be rejected");
    assert_eq!(err.to_string(), "image has too many tiles");
}

#[test]
fn iso_p0_03_tile_header_roi_maxshift_matches_reference_when_available() {
    let Some(root) = std::env::var_os("J2K_ISO_CONFORMANCE_DIR") else {
        return;
    };
    let root = Path::new(&root);
    let codestream =
        std::fs::read(root.join("codestreams_profile0/p0_03.j2k")).expect("read p0_03 codestream");
    let reference = read_pgx(
        root.join("reference_class1_profile0/c1p0_03-0.pgx")
            .as_path(),
    );

    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("p0_03 parses")
        .decode_native()
        .expect("p0_03 decodes");

    assert_eq!(decoded.width, reference.width);
    assert_eq!(decoded.height, reference.height);
    assert_eq!(decoded.bit_depth, reference.bit_depth);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.bytes_per_sample, 1);

    let sign_shift = if reference.signed {
        1_i32 << (reference.bit_depth - 1)
    } else {
        0
    };
    let expected = reference
        .samples
        .iter()
        .map(|sample| u8::try_from(sample + sign_shift).expect("4-bit reference sample fits u8"))
        .collect::<Vec<_>>();
    assert_eq!(decoded.data, expected);
}

struct PgxImage {
    signed: bool,
    bit_depth: u8,
    width: u32,
    height: u32,
    samples: Vec<i32>,
}

fn read_pgx(path: &Path) -> PgxImage {
    let bytes = std::fs::read(path).expect("read PGX reference");
    let header_end = bytes
        .iter()
        .position(|byte| *byte == b'\n')
        .expect("PGX header terminator");
    let header = std::str::from_utf8(&bytes[..header_end]).expect("PGX header is UTF-8");
    let parts = header.split_whitespace().collect::<Vec<_>>();
    assert_eq!(parts.len(), 5);
    assert_eq!(parts[0], "PG");

    let big_endian = match parts[1] {
        "ML" => true,
        "LM" => false,
        endian => panic!("unsupported PGX byte order {endian}"),
    };
    let signed = parts[2].starts_with('-');
    let bit_depth = parts[2]
        .trim_start_matches(['+', '-'])
        .parse::<u8>()
        .expect("PGX bit depth");
    let width = parts[3].parse::<u32>().expect("PGX width");
    let height = parts[4].parse::<u32>().expect("PGX height");
    let bytes_per_sample = if bit_depth <= 8 { 1 } else { 2 };
    let payload = &bytes[header_end + 1..];
    assert_eq!(
        payload.len(),
        width as usize * height as usize * bytes_per_sample
    );

    let samples = if bytes_per_sample == 1 {
        payload
            .iter()
            .map(|byte| {
                if signed {
                    i32::from((*byte).cast_signed())
                } else {
                    i32::from(*byte)
                }
            })
            .collect()
    } else {
        payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| {
                let raw = if big_endian {
                    u16::from_be_bytes([chunk[0], chunk[1]])
                } else {
                    u16::from_le_bytes([chunk[0], chunk[1]])
                };
                if signed {
                    i32::from(raw.cast_signed())
                } else {
                    i32::from(raw)
                }
            })
            .collect()
    };

    PgxImage {
        signed,
        bit_depth,
        width,
        height,
        samples,
    }
}

// Coding style and quantization set in tile-part headers. A tile-part COC
// takes precedence over a tile-part COD, which takes precedence over a
// main-header COC, then the main-header COD; QCC and QCD likewise (T.800
// Annex A). Each test edits encoder output so that the precedence alone
// decides whether a decode reproduces the original image.

const COD: u8 = 0x52;
const COC: u8 = 0x53;
const QCD: u8 = 0x5C;
const QCC: u8 = 0x5D;
const SOT: u8 = 0x90;

#[derive(Clone, Copy, Debug)]
struct Style {
    reversible: bool,
    mct: bool,
    levels: u8,
}

const REVERSIBLE: Style = Style {
    reversible: true,
    mct: true,
    levels: 5,
};
const IRREVERSIBLE: Style = Style {
    reversible: false,
    mct: true,
    levels: 5,
};

fn textured_pixels(width: u32, height: u32, components: u16) -> Vec<u8> {
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            for c in 0..u32::from(components) {
                let wave = (x * 7 + y * 3 + c * 61) % 256;
                let noise = (x * y + c * 17).wrapping_mul(2_654_435_761) >> 27;
                pixels.push(u8::try_from((wave + noise) % 256).expect("masked sample"));
            }
        }
    }
    pixels
}

fn encode_fixture(
    (width, height): (u32, u32),
    components: u16,
    style: Style,
    tile_size: Option<(u32, u32)>,
) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: style.reversible,
        use_mct: style.mct,
        num_decomposition_levels: style.levels,
        tile_size,
        ..EncodeOptions::default()
    };
    let pixels = textured_pixels(width, height, components);
    encode(&pixels, width, height, components, 8, false, &options)
        .unwrap_or_else(|error| panic!("encode {components} components {style:?}: {error}"))
}

/// The length of the marker segment at `at`, marker included.
fn segment_len(codestream: &[u8], at: usize) -> usize {
    2 + usize::from(u16::from_be_bytes([codestream[at + 2], codestream[at + 3]]))
}

/// The offset and marker of each main-header segment after SOC.
fn main_header_segments(codestream: &[u8]) -> Vec<(usize, u8)> {
    let mut segments = Vec::new();
    let mut at = 2;
    while codestream[at + 1] != SOT {
        segments.push((at, codestream[at + 1]));
        at += segment_len(codestream, at);
    }
    segments
}

fn main_header_segment_offset(codestream: &[u8], marker: u8) -> usize {
    main_header_segments(codestream)
        .into_iter()
        .find_map(|(at, found)| (found == marker).then_some(at))
        .unwrap_or_else(|| panic!("missing main-header marker FF{marker:02X}"))
}

fn main_header_segment(codestream: &[u8], marker: u8) -> Vec<u8> {
    let at = main_header_segment_offset(codestream, marker);
    codestream[at..at + segment_len(codestream, at)].to_vec()
}

/// Every main-header COD, COC, QCD and QCC segment, in order.
fn main_header_coding_segments(codestream: &[u8]) -> Vec<u8> {
    main_header_segments(codestream)
        .into_iter()
        .filter(|&(_, marker)| matches!(marker, COD | COC | QCD | QCC))
        .flat_map(|(at, _)| codestream[at..at + segment_len(codestream, at)].to_vec())
        .collect()
}

fn insert_after_main_header_segment(codestream: &mut Vec<u8>, marker: u8, segment: &[u8]) {
    let at = main_header_segment_offset(codestream, marker);
    let end = at + segment_len(codestream, at);
    codestream.splice(end..end, segment.iter().copied());
}

/// The offset of each tile-part's SOT, following the Psot lengths.
fn tile_part_offsets(codestream: &[u8]) -> Vec<usize> {
    let &(last, _) = main_header_segments(codestream).last().expect("SIZ");
    let mut at = last + segment_len(codestream, last);
    let mut offsets = Vec::new();
    while codestream[at..at + 2] == [0xFF, SOT] {
        offsets.push(at);
        let psot = u32::from_be_bytes(codestream[at + 6..at + 10].try_into().expect("Psot"));
        assert_ne!(psot, 0, "fixture tile-parts carry their lengths");
        at += usize::try_from(psot).expect("Psot fits usize");
    }
    offsets
}

/// Insert `segments` into tile-part `index`'s header, right after its SOT.
fn insert_tile_part_segments(codestream: &mut Vec<u8>, index: usize, segments: &[u8]) {
    let sot = tile_part_offsets(codestream)[index];
    codestream.splice(sot + 12..sot + 12, segments.iter().copied());
    let psot = u32::from_be_bytes(codestream[sot + 6..sot + 10].try_into().expect("Psot"));
    let grown = psot + u32::try_from(segments.len()).expect("segments fit Psot");
    codestream[sot + 6..sot + 10].copy_from_slice(&grown.to_be_bytes());
}

/// Swap the `SPcod` or `SPcoc` wavelet of a COD or COC segment between 9/7 and
/// 5/3. COC components are one byte, as for fewer than 257 components.
fn swap_wavelet(segment: &mut [u8]) {
    let at = match segment[1] {
        COD => 13,
        COC => 10,
        marker => panic!("FF{marker:02X} has no wavelet"),
    };
    assert!(segment[at] <= 1, "fixture wavelets are 9/7 or 5/3");
    segment[at] ^= 1;
}

fn swap_main_header_wavelet(codestream: &mut [u8]) {
    let at = main_header_segment_offset(codestream, COD);
    swap_wavelet(&mut codestream[at..]);
}

/// A COC giving `component` the coding style of `cod`.
fn coc_from_cod(cod: &[u8], component: u8) -> Vec<u8> {
    let parameters = &cod[9..];
    let len = u16::try_from(4 + parameters.len()).expect("COC length");
    let mut coc = vec![0xFF, COC];
    coc.extend(len.to_be_bytes());
    coc.extend([component, cod[4] & 1]);
    coc.extend(parameters);
    coc
}

/// `qcd`, a scalar expounded QCD, with the top mantissa bit of every step
/// flipped. The exponents, and so the code-blocks' bit-planes, stay as they
/// were, so the codestream still decodes, to other samples.
fn with_other_mantissas(qcd: &[u8]) -> Vec<u8> {
    assert_eq!(qcd[4] & 0x1F, 2, "scalar expounded quantization");
    let mut other = qcd.to_vec();
    for step in other[5..].as_chunks_mut::<2>().0 {
        step[0] ^= 0x04;
    }
    other
}

/// A QCC giving `component` the quantization of `qcd`.
fn qcc_from_qcd(qcd: &[u8], component: u8) -> Vec<u8> {
    let len = u16::from_be_bytes([qcd[2], qcd[3]]) + 1;
    let mut qcc = vec![0xFF, QCC];
    qcc.extend(len.to_be_bytes());
    qcc.push(component);
    qcc.extend(&qcd[4..]);
    qcc
}

#[derive(Clone, Copy, Debug)]
struct DecodeMode {
    parallelism: CpuDecodeParallelism,
    half_resolution: bool,
    region: Option<(u32, u32, u32, u32)>,
    /// In a JP2 file with channel definitions, whose region decodes keep
    /// IDWT levels in windows rather than stream them a row at a time.
    jp2: bool,
}

/// Whole-image and region decodes, serial and parallel, at full and half
/// resolution, of the codestream and of a JP2 file holding it. The first
/// region crosses the vertical midline, the second lies right of it and the
/// third spans the image's width.
fn decode_modes(width: u32, height: u32) -> Vec<DecodeMode> {
    let mut modes = Vec::new();
    for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
        for half_resolution in [false, true] {
            let (w, h) = if half_resolution {
                (width.div_ceil(2), height.div_ceil(2))
            } else {
                (width, height)
            };
            for region in [
                None,
                Some((w / 4, h / 4, w / 2, h / 2)),
                Some((w / 2 + 3, 1, w / 2 - 3, h / 3)),
                Some((0, h - 5, w, 5)),
            ] {
                for jp2 in [false, true] {
                    modes.push(DecodeMode {
                        parallelism,
                        half_resolution,
                        region,
                        jp2,
                    });
                }
            }
        }
    }
    modes
}

/// `codestream`, of 8-bit unsigned samples, in a JP2 file that defines its
/// channels in codestream order.
fn jp2_with_channel_definitions(codestream: &[u8]) -> Vec<u8> {
    let jp2_box = |box_type: &[u8; 4], payload: &[u8]| {
        let len = u32::try_from(payload.len() + 8).expect("box length");
        [&len.to_be_bytes()[..], box_type, payload].concat()
    };
    let siz = main_header_segment(codestream, 0x51);
    let field = |at: usize| u32::from_be_bytes(siz[at..at + 4].try_into().expect("SIZ field"));
    let (width, height) = (field(6) - field(14), field(10) - field(18));
    let components = u16::from_be_bytes([siz[38], siz[39]]);
    let mut ihdr = [height.to_be_bytes(), width.to_be_bytes()].concat();
    ihdr.extend(components.to_be_bytes());
    ihdr.extend([7, 7, 0, 0]);
    let colour_space = if components == 3 { 16 } else { 17 };
    let mut cdef = components.to_be_bytes().to_vec();
    for channel in 0..components {
        cdef.extend(channel.to_be_bytes());
        cdef.extend(0_u16.to_be_bytes());
        cdef.extend((channel + 1).to_be_bytes());
    }
    let header = [
        jp2_box(b"ihdr", &ihdr),
        jp2_box(b"colr", &[1, 0, 0, 0, 0, 0, colour_space]),
        jp2_box(b"cdef", &cdef),
    ]
    .concat();
    [
        jp2_box(b"jP  ", &[0x0D, 0x0A, 0x87, 0x0A]),
        jp2_box(b"ftyp", b"jp2 \0\0\0\0jp2 "),
        jp2_box(b"jp2h", &header),
        jp2_box(b"jp2c", codestream),
    ]
    .concat()
}

fn decode_with_mode(codestream: &[u8], mode: DecodeMode) -> RawBitmap {
    let jp2;
    let codestream = if mode.jp2 {
        jp2 = jp2_with_channel_definitions(codestream);
        &jp2[..]
    } else {
        codestream
    };
    let full = Image::new(codestream, &DecodeSettings::default()).expect("parse codestream");
    let settings = DecodeSettings {
        target_resolution: mode
            .half_resolution
            .then(|| (full.width().div_ceil(2), full.height().div_ceil(2))),
        ..DecodeSettings::default()
    };
    let image = Image::new(codestream, &settings).expect("parse codestream");
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(mode.parallelism);
    match mode.region {
        None => image.decode_native_with_context(&mut context),
        Some(region) => image.decode_native_region_with_context(region, &mut context),
    }
    .unwrap_or_else(|error| panic!("decode {mode:?}: {error}"))
}

fn assert_decodes_match(patched: &[u8], original: &[u8], case: &str) {
    let image = Image::new(original, &DecodeSettings::default()).expect("parse original");
    for mode in decode_modes(image.width(), image.height()) {
        let expected = decode_with_mode(original, mode);
        let decoded = decode_with_mode(patched, mode);
        assert_eq!(
            (decoded.width, decoded.height, decoded.num_components),
            (expected.width, expected.height, expected.num_components),
            "{case} {mode:?}"
        );
        assert!(decoded.data == expected.data, "{case} {mode:?}");
    }
}

fn assert_decode_differs(patched: &[u8], original: &[u8], case: &str) {
    let mode = DecodeMode {
        parallelism: CpuDecodeParallelism::Serial,
        half_resolution: false,
        region: None,
        jp2: false,
    };
    assert!(
        decode_with_mode(patched, mode).data != decode_with_mode(original, mode).data,
        "{case}: the edit must change the image unless overridden"
    );
}

const SINGLE_TILE: (u32, u32) = (96, 64);

#[test]
fn tile_part_cod_overrides_the_main_header_wavelet() {
    for components in [1, 3] {
        for style in [REVERSIBLE, IRREVERSIBLE] {
            let case = format!("{components} components {style:?}");
            let original = encode_fixture(SINGLE_TILE, components, style, None);
            let mut patched = original.clone();
            swap_main_header_wavelet(&mut patched);
            assert_decode_differs(&patched, &original, &case);

            insert_tile_part_segments(&mut patched, 0, &main_header_segment(&original, COD));
            assert_decodes_match(&patched, &original, &case);
        }
    }
}

#[test]
fn tile_part_coc_overrides_a_main_header_coc() {
    for style in [REVERSIBLE, IRREVERSIBLE] {
        let style = Style {
            mct: false,
            ..style
        };
        let case = format!("{style:?}");
        let original = encode_fixture(SINGLE_TILE, 3, style, None);
        let restore = coc_from_cod(&main_header_segment(&original, COD), 1);
        let mut other = restore.clone();
        swap_wavelet(&mut other);
        let mut patched = original.clone();
        insert_after_main_header_segment(&mut patched, COD, &other);
        assert_decode_differs(&patched, &original, &case);

        insert_tile_part_segments(&mut patched, 0, &restore);
        assert_decodes_match(&patched, &original, &case);
    }
}

#[test]
fn tile_part_coc_takes_precedence_over_tile_part_cod_in_either_order() {
    for components in [1_u8, 3] {
        for coc_first in [true, false] {
            let case = format!("{components} components, COC first: {coc_first}");
            let original = encode_fixture(SINGLE_TILE, components.into(), REVERSIBLE, None);
            let cod = main_header_segment(&original, COD);
            let mut other_cod = cod.clone();
            swap_wavelet(&mut other_cod);
            let cocs = (0..components)
                .flat_map(|component| coc_from_cod(&cod, component))
                .collect::<Vec<_>>();
            let segments = if coc_first {
                [cocs, other_cod].concat()
            } else {
                [other_cod, cocs].concat()
            };
            let mut patched = original.clone();
            insert_tile_part_segments(&mut patched, 0, &segments);
            assert_decodes_match(&patched, &original, &case);
        }
    }
}

#[test]
fn tile_part_qcc_takes_precedence_over_tile_part_qcd_in_either_order() {
    let original = encode_fixture(SINGLE_TILE, 3, IRREVERSIBLE, None);
    let qcd = main_header_segment(&original, QCD);
    let other_qcd = with_other_mantissas(&qcd);
    let mut only_other = original.clone();
    insert_tile_part_segments(&mut only_other, 0, &other_qcd);
    assert_decode_differs(&only_other, &original, "tile-part QCD alone");

    let qccs = (0..3)
        .flat_map(|component| qcc_from_qcd(&qcd, component))
        .collect::<Vec<_>>();
    for qcc_first in [true, false] {
        let segments = if qcc_first {
            [qccs.clone(), other_qcd.clone()].concat()
        } else {
            [other_qcd.clone(), qccs.clone()].concat()
        };
        let mut patched = original.clone();
        insert_tile_part_segments(&mut patched, 0, &segments);
        assert_decodes_match(&patched, &original, &format!("QCC first: {qcc_first}"));
    }
}

const TWO_TILES: (u32, u32) = (256, 64);
const TILE: (u32, u32) = (128, 64);

/// Tile 0 of `left` under its main header, then tile 1 of `right` with
/// `right`'s main-header coding style and quantization in its tile-part
/// header.
fn splice_second_tile(left: &[u8], right: &[u8]) -> Vec<u8> {
    assert_eq!(
        main_header_segment(left, 0x51),
        main_header_segment(right, 0x51),
        "both encodes share one SIZ"
    );
    let (left_parts, right_parts) = (tile_part_offsets(left), tile_part_offsets(right));
    assert_eq!((left_parts.len(), right_parts.len()), (2, 2));
    let mut spliced = left[..left_parts[1]].to_vec();
    let start = right_parts[1];
    spliced.extend(&right[start..start + tile_part_len(right, start)]);
    spliced.extend([0xFF, 0xD9]);
    insert_tile_part_segments(&mut spliced, 1, &main_header_coding_segments(right));
    spliced
}

fn tile_part_len(codestream: &[u8], sot: usize) -> usize {
    let psot = u32::from_be_bytes(codestream[sot + 6..sot + 10].try_into().expect("Psot"));
    usize::try_from(psot).expect("Psot fits usize")
}

/// Each decode of the spliced image gives the columns of tile 0 as `left`
/// decodes them and the rest as `right` does.
#[test]
fn tiles_with_different_coding_styles_decode_as_their_own_encodes() {
    let no_mct = |style: Style| Style {
        mct: false,
        ..style
    };
    let two_levels = Style {
        levels: 2,
        ..REVERSIBLE
    };
    let cases = [
        (1, REVERSIBLE, IRREVERSIBLE),
        (1, IRREVERSIBLE, REVERSIBLE),
        (3, REVERSIBLE, IRREVERSIBLE),
        (3, IRREVERSIBLE, REVERSIBLE),
        (3, REVERSIBLE, no_mct(REVERSIBLE)),
        (3, no_mct(IRREVERSIBLE), IRREVERSIBLE),
        (3, REVERSIBLE, two_levels),
    ];
    for (components, left_style, right_style) in cases {
        let case = format!("{components} components, {left_style:?} then {right_style:?}");
        let left = encode_fixture(TWO_TILES, components, left_style, Some(TILE));
        let right = encode_fixture(TWO_TILES, components, right_style, Some(TILE));
        let spliced = splice_second_tile(&left, &right);
        let (width, height) = TWO_TILES;
        for mode in decode_modes(width, height) {
            let (left_image, right_image) = (
                decode_with_mode(&left, mode),
                decode_with_mode(&right, mode),
            );
            let decoded = decode_with_mode(&spliced, mode);
            let tile_width = if mode.half_resolution {
                TILE.0.div_ceil(2)
            } else {
                TILE.0
            };
            let x0 = mode.region.map_or(0, |(x, ..)| x);
            let pixel = usize::from(components);
            let mut expected = Vec::with_capacity(left_image.data.len());
            let rows = left_image
                .data
                .chunks_exact(left_image.width as usize * pixel);
            let right_rows = right_image
                .data
                .chunks_exact(right_image.width as usize * pixel);
            for (left_row, right_row) in rows.zip(right_rows) {
                for column in 0..left_image.width {
                    let source = if x0 + column < tile_width {
                        left_row
                    } else {
                        right_row
                    };
                    let at = column as usize * pixel;
                    expected.extend(&source[at..at + pixel]);
                }
            }
            assert_eq!(decoded.width, left_image.width, "{case} {mode:?}");
            assert!(decoded.data == expected, "{case} {mode:?}");
        }
    }
}

/// The planes of an image hold integers for exact-integer decodes and floats
/// otherwise, so tiles that differ in which they need cannot share them.
#[test]
fn tiles_that_differ_in_needing_exact_integer_decode_are_rejected() {
    let (width, height) = TWO_TILES;
    let pixels = (0..width * height)
        .flat_map(|index: u32| (index.wrapping_mul(2_654_435_761) >> 3).to_le_bytes())
        .collect::<Vec<_>>();
    let encode_29_bit = |reversible: bool| {
        let options = EncodeOptions {
            reversible,
            use_mct: false,
            tile_size: Some(TILE),
            ..EncodeOptions::default()
        };
        encode(&pixels, width, height, 1, 29, false, &options).expect("29-bit encode")
    };
    let (exact, float) = (encode_29_bit(true), encode_29_bit(false));
    for (left, right) in [(&exact, &float), (&float, &exact)] {
        let error = Image::new(&splice_second_tile(left, right), &DecodeSettings::default())
            .expect("parse spliced codestream")
            .decode_native()
            .err()
            .expect("mixed exact-integer tiles must be rejected");
        assert_eq!(
            error.to_string(),
            "unsupported decoding feature: tiles that differ in needing exact integer decode"
        );
    }
}
