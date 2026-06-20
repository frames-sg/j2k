// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use j2k_native::{encode, DecodeSettings, EncodeOptions, Image};

fn marker_offset(codestream: &[u8], marker: u8) -> usize {
    codestream
        .windows(2)
        .position(|window| window == [0xFF, marker])
        .unwrap_or_else(|| panic!("missing marker FF{marker:02X}"))
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
    let pixels: Vec<_> = (0..64 * 64).map(|idx| (idx % 251) as u8).collect();
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
fn tile_header_rgn_with_explicit_style_is_rejected() {
    let pixels: Vec<_> = (0..64 * 64).map(|idx| (idx % 251) as u8).collect();
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
    let pixels: Vec<_> = (0..64 * 64).map(|idx| (idx % 251) as u8).collect();
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
    let pixels: Vec<_> = (0..16 * 16).map(|idx| (idx % 251) as u8).collect();
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
                    i32::from(*byte as i8)
                } else {
                    i32::from(*byte)
                }
            })
            .collect()
    } else {
        payload
            .chunks_exact(2)
            .map(|chunk| {
                let raw = if big_endian {
                    u16::from_be_bytes([chunk[0], chunk[1]])
                } else {
                    u16::from_le_bytes([chunk[0], chunk[1]])
                };
                if signed {
                    i32::from(raw as i16)
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
