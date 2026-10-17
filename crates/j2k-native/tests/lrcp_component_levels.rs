// SPDX-License-Identifier: MIT OR Apache-2.0

//! An LRCP codestream whose components have different numbers of
//! decomposition levels, from a COC marker, decodes. The test builds it by
//! splicing two single-component codestreams from the encoder, one with 5
//! levels and one with 1, so every packet in it is valid; `OpenJPEG` 2.5.4
//! decodes the same construction exactly.

use j2k_native::{encode, DecodeSettings, EncodeOptions, EncodeProgressionOrder, Image};

const SIZE: u32 = 32;

/// A single-component LRCP codestream with one layer and PLT markers, and
/// its samples.
fn single_component(levels: u8, seed: u32) -> (Vec<u8>, Vec<u8>) {
    let pixels = (0..SIZE * SIZE)
        .map(|i| u8::try_from((i % SIZE * 5 + i / SIZE * 3 + i * seed % 17 * 4) % 256).unwrap())
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        num_decomposition_levels: levels,
        progression_order: EncodeProgressionOrder::Lrcp,
        write_plt: true,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, SIZE, SIZE, 1, 8, false, &options).expect("encode");
    (codestream, pixels)
}

/// The SIZ, COD and QCD bodies and the packets, one per resolution level.
struct Parts {
    siz: Vec<u8>,
    cod: Vec<u8>,
    qcd: Vec<u8>,
    packets: Vec<Vec<u8>>,
}

fn parts(codestream: &[u8]) -> Parts {
    let (mut siz, mut cod, mut qcd, mut lengths) = (vec![], vec![], vec![], vec![]);
    let mut at = 2;
    while codestream[at + 1] != 0x93 {
        let length = usize::from(u16::from_be_bytes([codestream[at + 2], codestream[at + 3]]));
        let body = &codestream[at + 4..at + 2 + length];
        match codestream[at + 1] {
            0x51 => siz = body.to_vec(),
            0x52 => cod = body.to_vec(),
            0x5c => qcd = body.to_vec(),
            // PLT: Zplt, then 7-bit groups with a continuation bit.
            0x58 => {
                let mut value = 0;
                for &byte in &body[1..] {
                    value = value << 7 | usize::from(byte & 0x7f);
                    if byte & 0x80 == 0 {
                        lengths.push(value);
                        value = 0;
                    }
                }
            }
            _ => {}
        }
        at += 2 + length;
    }
    let mut data = &codestream[at + 2..codestream.len() - 2];
    let packets = lengths
        .iter()
        .map(|&length| {
            let (packet, rest) = data.split_at(length);
            data = rest;
            packet.to_vec()
        })
        .collect();
    assert!(data.is_empty(), "PLT covers the tile data");
    Parts {
        siz,
        cod,
        qcd,
        packets,
    }
}

fn segment(out: &mut Vec<u8>, marker: u8, body: &[u8]) {
    out.extend_from_slice(&[0xff, marker]);
    out.extend_from_slice(&u16::try_from(body.len() + 2).unwrap().to_be_bytes());
    out.extend_from_slice(body);
}

/// Component 0 from `first` (COD, QCD) and component 1 from `second` (COC,
/// QCC), with packets in LRCP order.
fn splice(first: &Parts, second: &Parts) -> Vec<u8> {
    let mut siz = first.siz[..34].to_vec();
    siz.extend_from_slice(&2u16.to_be_bytes());
    siz.extend_from_slice(&first.siz[36..39]);
    siz.extend_from_slice(&second.siz[36..39]);
    // Ccoc, Scoc (the precinct flag), then SPcod from decomposition levels on.
    let mut coc = vec![1, second.cod[0] & 1];
    coc.extend_from_slice(&second.cod[5..]);
    let mut qcc = vec![1];
    qcc.extend_from_slice(&second.qcd);

    let levels = [first.cod[5], second.cod[5]];
    let mut data = vec![];
    for resolution in 0..=levels[0].max(levels[1]) {
        for (component, parts) in [first, second].into_iter().enumerate() {
            if resolution <= levels[component] {
                data.extend_from_slice(&parts.packets[usize::from(resolution)]);
            }
        }
    }

    let mut codestream = vec![0xff, 0x4f];
    segment(&mut codestream, 0x51, &siz);
    segment(&mut codestream, 0x52, &first.cod);
    segment(&mut codestream, 0x53, &coc);
    segment(&mut codestream, 0x5c, &first.qcd);
    segment(&mut codestream, 0x5d, &qcc);
    // SOT: Isot, Psot (SOT + SOD + data), TPsot, TNsot.
    let mut sot = 0u16.to_be_bytes().to_vec();
    sot.extend_from_slice(&u32::try_from(12 + 2 + data.len()).unwrap().to_be_bytes());
    sot.extend_from_slice(&[0, 1]);
    segment(&mut codestream, 0x90, &sot);
    codestream.extend_from_slice(&[0xff, 0x93]);
    codestream.extend_from_slice(&data);
    codestream.extend_from_slice(&[0xff, 0xd9]);
    codestream
}

#[test]
fn lrcp_components_with_different_decomposition_levels_decode() {
    let (five, five_pixels) = single_component(5, 7);
    let (one, one_pixels) = single_component(1, 11);
    let (five, one) = (parts(&five), parts(&one));

    // Fewer levels in the second component, then in the first.
    for (first, second, expected) in [
        (&five, &one, [&five_pixels, &one_pixels]),
        (&one, &five, [&one_pixels, &five_pixels]),
    ] {
        let codestream = splice(first, second);
        let image = Image::new(&codestream, &DecodeSettings::default()).expect("parse");
        let decoded = image.decode_native_components().expect("decode");
        let planes = decoded.planes();
        assert_eq!(planes.len(), 2);
        assert_eq!(planes[0].data(), expected[0].as_slice());
        assert_eq!(planes[1].data(), expected[1].as_slice());
    }
}
