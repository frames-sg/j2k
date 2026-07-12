// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{encode, DecodeSettings, EncodeOptions, Image};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 8;

fn pixels() -> Vec<u8> {
    (0..WIDTH * HEIGHT)
        .map(|index| u8::try_from((index * 37 + index / 3) & 0xff).expect("masked sample"))
        .collect()
}

fn assert_roundtrip(codestream: &[u8], expected: &[u8]) {
    assert_roundtrip_with_context(codestream, expected, "roundtrip");
}

fn assert_roundtrip_with_context(codestream: &[u8], expected: &[u8], context: &str) {
    let image = Image::new(codestream, &DecodeSettings::default()).expect("parse codestream");
    let decoded = image.decode_native().expect("decode codestream");
    assert_eq!(decoded.data, expected, "{context}");
}

fn marker_count(codestream: &[u8], marker: u8) -> usize {
    codestream
        .windows(2)
        .filter(|bytes| *bytes == [0xff, marker])
        .count()
}

#[test]
fn multi_tile_packet_limit_splits_only_parent_tile_parts_and_round_trips() {
    let pixels = pixels();
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        tile_size: Some((8, 8)),
        tile_part_packet_limit: Some(1),
        ..EncodeOptions::default()
    };

    let codestream = encode(&pixels, WIDTH, HEIGHT, 1, 8, false, &options)
        .expect("multi-tile encode with parent tile-part splitting");
    let sot_count = marker_count(&codestream, 0x90);
    assert_eq!(sot_count, 4, "two packets in each of two image tiles");

    assert_roundtrip(&codestream, &pixels);
}

#[test]
fn multi_tile_direct_packets_preserve_parent_packet_marker_modes() {
    let pixels = pixels();
    for (name, marker, mode) in [
        ("PLT", 0x58, 0_u8),
        ("PLM", 0x57, 1),
        ("PPM", 0x60, 2),
        ("PPT", 0x61, 3),
    ] {
        let mut options = EncodeOptions {
            num_decomposition_levels: 1,
            reversible: true,
            tile_size: Some((8, 8)),
            ..EncodeOptions::default()
        };
        match mode {
            0 => options.write_plt = true,
            1 => options.write_plm = true,
            2 => options.write_ppm = true,
            3 => options.write_ppt = true,
            _ => unreachable!("fixed marker mode"),
        }

        let codestream = encode(&pixels, WIDTH, HEIGHT, 1, 8, false, &options)
            .unwrap_or_else(|error| panic!("multi-tile {name} encode failed: {error}"));
        assert!(
            marker_count(&codestream, marker) > 0,
            "multi-tile {name} marker missing"
        );
        assert_roundtrip_with_context(&codestream, &pixels, name);
    }
}
