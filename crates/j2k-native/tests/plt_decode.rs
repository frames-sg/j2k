// SPDX-License-Identifier: Apache-2.0

use j2k_native::{encode, DecodeSettings, EncodeOptions, Image};

fn marker_offset(codestream: &[u8], marker: u8) -> usize {
    codestream
        .windows(2)
        .position(|window| window == [0xFF, marker])
        .unwrap_or_else(|| panic!("missing marker FF{marker:02X}"))
}

fn packet_length_bytes(mut length: usize) -> Vec<u8> {
    let mut groups = vec![(length & 0x7F) as u8];
    length >>= 7;

    while length > 0 {
        groups.push((length & 0x7F) as u8);
        length >>= 7;
    }

    groups
        .iter()
        .rev()
        .enumerate()
        .map(|(idx, group)| {
            if idx + 1 == groups.len() {
                *group
            } else {
                *group | 0x80
            }
        })
        .collect()
}

fn insert_plt(mut codestream: Vec<u8>, packet_length: usize) -> Vec<u8> {
    let sod_offset = marker_offset(&codestream, 0x93);
    let mut plt = vec![0xFF, 0x58];
    let packet_length_bytes = packet_length_bytes(packet_length);
    let marker_len = u16::try_from(3 + packet_length_bytes.len()).expect("PLT marker length");
    plt.extend_from_slice(&marker_len.to_be_bytes());
    plt.push(0);
    plt.extend_from_slice(&packet_length_bytes);

    codestream.splice(sod_offset..sod_offset, plt.iter().copied());

    let sot_offset = marker_offset(&codestream, 0x90);
    let psot = u32::from_be_bytes([
        codestream[sot_offset + 6],
        codestream[sot_offset + 7],
        codestream[sot_offset + 8],
        codestream[sot_offset + 9],
    ]);
    codestream[sot_offset + 6..sot_offset + 10]
        .copy_from_slice(&(psot + u32::try_from(plt.len()).unwrap()).to_be_bytes());

    codestream
}

fn insert_plm(mut codestream: Vec<u8>, packet_length: usize) -> Vec<u8> {
    let sot_offset = marker_offset(&codestream, 0x90);
    let mut plm = vec![0xFF, 0x57];
    let packet_length_bytes = packet_length_bytes(packet_length);
    let marker_len = u16::try_from(7 + packet_length_bytes.len()).expect("PLM marker length");
    plm.extend_from_slice(&marker_len.to_be_bytes());
    plm.push(0);
    plm.extend_from_slice(
        &u32::try_from(packet_length_bytes.len())
            .unwrap()
            .to_be_bytes(),
    );
    plm.extend_from_slice(&packet_length_bytes);

    codestream.splice(sot_offset..sot_offset, plm);
    codestream
}

fn fixture_pixels() -> Vec<u8> {
    (0..64u8)
        .flat_map(|y| (0..64u8).map(move |x| x.wrapping_mul(3).wrapping_add(y)))
        .collect::<Vec<_>>()
}

fn one_packet_fixture() -> Vec<u8> {
    let pixels = fixture_pixels();
    encode(
        &pixels,
        64,
        64,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        },
    )
    .expect("one-packet fixture encode")
}

fn tile_body_len(codestream: &[u8]) -> usize {
    let sod_offset = marker_offset(codestream, 0x93);
    let eoc_offset = marker_offset(codestream, 0xD9);
    eoc_offset - (sod_offset + 2)
}

fn strict_decode(codestream: &[u8]) -> j2k_native::Result<j2k_native::RawBitmap> {
    Image::new(
        codestream,
        &DecodeSettings {
            strict: true,
            ..DecodeSettings::default()
        },
    )?
    .decode_native()
}

#[test]
fn strict_decode_accepts_matching_plt_packet_length() {
    let codestream = one_packet_fixture();
    let packet_length = tile_body_len(&codestream);
    let with_plt = insert_plt(codestream, packet_length);

    let decoded = strict_decode(&with_plt).expect("matching PLT packet length decodes");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.data, fixture_pixels());
}

#[test]
fn strict_decode_rejects_wrong_plt_packet_length() {
    let codestream = one_packet_fixture();
    let wrong_packet_length = tile_body_len(&codestream) + 1;
    let with_plt = insert_plt(codestream, wrong_packet_length);

    assert!(
        strict_decode(&with_plt).is_err(),
        "strict decode must reject PLT packet lengths that do not match consumed packet bytes"
    );
}

#[test]
fn strict_decode_accepts_matching_plm_packet_length() {
    let codestream = one_packet_fixture();
    let packet_length = tile_body_len(&codestream);
    let with_plm = insert_plm(codestream, packet_length);

    let decoded = strict_decode(&with_plm).expect("matching PLM packet length decodes");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.data, fixture_pixels());
}

#[test]
fn strict_decode_rejects_wrong_plm_packet_length() {
    let codestream = one_packet_fixture();
    let wrong_packet_length = tile_body_len(&codestream) + 1;
    let with_plm = insert_plm(codestream, wrong_packet_length);

    assert!(
        strict_decode(&with_plm).is_err(),
        "strict decode must reject PLM packet lengths that do not match consumed packet bytes"
    );
}
