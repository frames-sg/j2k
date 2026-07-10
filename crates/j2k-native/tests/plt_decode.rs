// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{encode, DecodeSettings, EncodeOptions, Image};

fn marker_offset(codestream: &[u8], marker: u8) -> usize {
    codestream
        .windows(2)
        .position(|window| window == [0xFF, marker])
        .unwrap_or_else(|| panic!("missing marker FF{marker:02X}"))
}

fn packet_length_bytes(mut length: usize) -> Vec<u8> {
    let mut groups = vec![u8::try_from(length & 0x7F).expect("7-bit group fits u8")];
    length >>= 7;

    while length > 0 {
        groups.push(u8::try_from(length & 0x7F).expect("7-bit group fits u8"));
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

fn insert_ppt(mut codestream: Vec<u8>, header_len: usize) -> Vec<u8> {
    let sod_offset = marker_offset(&codestream, 0x93);
    let body_start = sod_offset + 2;
    let header = codestream[body_start..body_start + header_len].to_vec();
    codestream.drain(body_start..body_start + header_len);

    let mut ppt = vec![0xFF, 0x61];
    let marker_len = u16::try_from(3 + header.len()).expect("PPT marker length");
    ppt.extend_from_slice(&marker_len.to_be_bytes());
    ppt.push(0);
    ppt.extend_from_slice(&header);
    codestream.splice(sod_offset..sod_offset, ppt.iter().copied());

    let sot_offset = marker_offset(&codestream, 0x90);
    let psot = u32::from_be_bytes([
        codestream[sot_offset + 6],
        codestream[sot_offset + 7],
        codestream[sot_offset + 8],
        codestream[sot_offset + 9],
    ]);
    let adjusted = psot
        .checked_add(u32::try_from(ppt.len()).unwrap())
        .and_then(|len| len.checked_sub(u32::try_from(header_len).unwrap()))
        .expect("adjusted Psot fits");
    codestream[sot_offset + 6..sot_offset + 10].copy_from_slice(&adjusted.to_be_bytes());

    codestream
}

fn insert_ppm(mut codestream: Vec<u8>, header_len: usize) -> Vec<u8> {
    let sot_offset = marker_offset(&codestream, 0x90);
    let sod_offset = marker_offset(&codestream, 0x93);
    let body_start = sod_offset + 2;
    let header = codestream[body_start..body_start + header_len].to_vec();
    codestream.drain(body_start..body_start + header_len);

    let mut ppm = vec![0xFF, 0x60];
    let marker_len = u16::try_from(5 + header.len()).expect("PPM marker length");
    ppm.extend_from_slice(&marker_len.to_be_bytes());
    ppm.push(0);
    ppm.extend_from_slice(&u16::try_from(header.len()).unwrap().to_be_bytes());
    ppm.extend_from_slice(&header);
    codestream.splice(sot_offset..sot_offset, ppm);

    let sot_offset = marker_offset(&codestream, 0x90);
    let psot = u32::from_be_bytes([
        codestream[sot_offset + 6],
        codestream[sot_offset + 7],
        codestream[sot_offset + 8],
        codestream[sot_offset + 9],
    ]);
    let adjusted = psot
        .checked_sub(u32::try_from(header_len).unwrap())
        .expect("adjusted Psot fits");
    codestream[sot_offset + 6..sot_offset + 10].copy_from_slice(&adjusted.to_be_bytes());

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

fn find_separated_packet_header_len(
    codestream: &[u8],
    transform: impl Fn(Vec<u8>, usize) -> Vec<u8>,
) -> usize {
    let expected = fixture_pixels();
    let body_len = tile_body_len(codestream);
    for header_len in 1..body_len {
        let candidate = transform(codestream.to_vec(), header_len);
        let Ok(decoded) = strict_decode(&candidate) else {
            continue;
        };
        if decoded.data == expected {
            return header_len;
        }
    }
    panic!("could not find valid separated packet header length");
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
fn strict_decode_accepts_ppt_separated_packet_header() {
    let codestream = one_packet_fixture();
    let header_len = find_separated_packet_header_len(&codestream, insert_ppt);
    let with_ppt = insert_ppt(codestream, header_len);

    let decoded = strict_decode(&with_ppt).expect("PPT separated packet header decodes");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.data, fixture_pixels());
}

#[test]
fn strict_decode_accepts_ppm_separated_packet_header() {
    let codestream = one_packet_fixture();
    let header_len = find_separated_packet_header_len(&codestream, insert_ppm);
    let with_ppm = insert_ppm(codestream, header_len);

    let decoded = strict_decode(&with_ppm).expect("PPM separated packet header decodes");

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
