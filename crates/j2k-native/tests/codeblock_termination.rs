// SPDX-License-Identifier: MIT OR Apache-2.0

//! T.800 B.10.7: "The sequence of bytes actually included for any given
//! code-block must not end in a 0xFF." Code-block data and packet headers
//! follow one another directly, so a trailing 0xFF would pair with the next
//! byte; the rule keeps every byte-aligned value in 0xFF90..=0xFFFF out of
//! packet data, where those values are markers (D.4, note). These tests check
//! it on whole encoded codestreams.

use j2k_native::{encode, DecodeSettings, EncodeOptions, Image};

/// Noise over a gradient: enough coding passes per code-block that most
/// MQ terminations would otherwise end in 0xFF.
fn pixels(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    (0..width * height)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let gradient = (index % width) * 255 / width.max(1);
            u8::try_from((gradient + (state >> 24)) & 0xff).expect("masked sample")
        })
        .collect()
}

fn marker_segment_end(codestream: &[u8], at: usize) -> usize {
    let length = u16::from_be_bytes([codestream[at + 2], codestream[at + 3]]);
    at + 2 + usize::from(length)
}

/// The data of each tile-part: from just after its SOD to the end its Psot
/// gives.
fn tile_part_data(codestream: &[u8]) -> Vec<(usize, usize)> {
    assert_eq!(codestream[..2], [0xff, 0x4f], "SOC");
    let mut at = 2;
    while codestream[at..at + 2] != [0xff, 0x90] {
        at = marker_segment_end(codestream, at);
    }
    let mut parts = Vec::new();
    while codestream[at..at + 2] == [0xff, 0x90] {
        let psot = u32::from_be_bytes(codestream[at + 6..at + 10].try_into().expect("Psot"));
        assert_ne!(psot, 0, "the encoder writes every Psot");
        let end = at + usize::try_from(psot).expect("Psot fits usize");
        let mut header = marker_segment_end(codestream, at);
        while codestream[header..header + 2] != [0xff, 0x93] {
            header = marker_segment_end(codestream, header);
        }
        parts.push((header + 2, end));
        at = end;
    }
    assert_eq!(
        codestream[at..],
        [0xff, 0xd9],
        "EOC follows the last tile-part"
    );
    parts
}

fn options(reversible: bool, ht: bool) -> EncodeOptions {
    EncodeOptions {
        reversible,
        use_ht_block_coding: ht,
        ..EncodeOptions::default()
    }
}

fn assert_lossless_round_trip(codestream: &[u8], expected: &[u8], context: &str) {
    let image = Image::new(codestream, &DecodeSettings::strict()).expect("parse codestream");
    let decoded = image.decode_native().expect("decode codestream");
    assert_eq!(decoded.data, expected, "{context}");
}

#[test]
fn tile_data_holds_no_marker_values() {
    for (width, height) in [(16_u32, 9_u32), (64, 64), (256, 256)] {
        for reversible in [true, false] {
            for ht in [false, true] {
                let context = format!("{width}x{height} reversible={reversible} ht={ht}");
                let samples = pixels(width, height, width * 31 + height);
                let codestream = encode(
                    &samples,
                    width,
                    height,
                    1,
                    8,
                    false,
                    &options(reversible, ht),
                )
                .expect("encode");
                for (start, end) in tile_part_data(&codestream) {
                    let data = &codestream[start..end];
                    let marker = data
                        .windows(2)
                        .position(|pair| pair[0] == 0xff && pair[1] > 0x8f);
                    assert_eq!(
                        marker,
                        None,
                        "{context}: byte-aligned 0xFF{:02X} in tile data",
                        marker.map_or(0, |at| data[at + 1])
                    );
                    assert_ne!(
                        data.last(),
                        Some(&0xff),
                        "{context}: tile data ends in 0xFF"
                    );
                }
                if reversible {
                    assert_lossless_round_trip(&codestream, &samples, &context);
                }
            }
        }
    }
}

#[test]
fn packet_bodies_never_end_in_0xff() {
    let (width, height) = (256_u32, 256_u32);
    let samples = pixels(width, height, 7);
    for reversible in [true, false] {
        let codestream = encode(
            &samples,
            width,
            height,
            1,
            8,
            false,
            &EncodeOptions {
                write_sop: true,
                write_eph: true,
                ..options(reversible, false)
            },
        )
        .expect("encode with SOP and EPH");
        let mut bodies = 0;
        for (start, end) in tile_part_data(&codestream) {
            let data = &codestream[start..end];
            let sops: Vec<usize> = (0..data.len().saturating_sub(3))
                .filter(|&at| data[at..at + 4] == [0xff, 0x91, 0x00, 0x04])
                .collect();
            assert_eq!(sops.first(), Some(&0), "tile data starts with an SOP");
            for (index, &sop) in sops.iter().enumerate() {
                let next = sops.get(index + 1).copied().unwrap_or(data.len());
                let eph = (sop + 6..next - 1)
                    .find(|&at| data[at..at + 2] == [0xff, 0x92])
                    .expect("every packet header ends with EPH");
                let body = &data[eph + 2..next];
                if !body.is_empty() {
                    bodies += 1;
                    assert_ne!(
                        body.last(),
                        Some(&0xff),
                        "reversible={reversible}: packet {index}'s body ends in 0xFF"
                    );
                }
            }
        }
        assert!(bodies > 0, "the image codes non-empty packets");
        if reversible {
            assert_lossless_round_trip(&codestream, &samples, "SOP/EPH");
        }
    }
}
