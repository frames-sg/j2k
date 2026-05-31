// SPDX-License-Identifier: Apache-2.0

use signinum_j2k_native::{encode, DecodeSettings, EncodeOptions, EncodeProgressionOrder, Image};

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

fn force_cod_lrcp_and_insert_main_header_cprl_poc(mut codestream: Vec<u8>) -> Vec<u8> {
    let cod_offset = marker_offset(&codestream, 0x52);
    assert_eq!(
        codestream[cod_offset + 5],
        0x04,
        "fixture must start as CPRL"
    );
    codestream[cod_offset + 5] = 0x00;

    let poc = [
        0xFF, 0x5F, // POC
        0x00, 0x09, // Lpoc
        0x00, // RSpoc
        0x00, // CSpoc
        0x00, 0x01, // LYEpoc
        0x02, // REpoc: two resolutions
        0x03, // CEpoc: three components
        0x04, // Ppoc: CPRL
    ];
    codestream.splice(
        cod_marker_end(&codestream)..cod_marker_end(&codestream),
        poc,
    );
    codestream
}

fn force_cod_lrcp_and_insert_tile_header_cprl_poc(mut codestream: Vec<u8>) -> Vec<u8> {
    let cod_offset = marker_offset(&codestream, 0x52);
    assert_eq!(
        codestream[cod_offset + 5],
        0x04,
        "fixture must start as CPRL"
    );
    codestream[cod_offset + 5] = 0x00;

    let sod_offset = marker_offset(&codestream, 0x93);
    let poc = [
        0xFF, 0x5F, // POC
        0x00, 0x09, // Lpoc
        0x00, // RSpoc
        0x00, // CSpoc
        0x00, 0x01, // LYEpoc
        0x02, // REpoc: two resolutions
        0x03, // CEpoc: three components
        0x04, // Ppoc: CPRL
    ];
    codestream.splice(sod_offset..sod_offset, poc);

    let sot_offset = marker_offset(&codestream, 0x90);
    let psot = u32::from_be_bytes([
        codestream[sot_offset + 6],
        codestream[sot_offset + 7],
        codestream[sot_offset + 8],
        codestream[sot_offset + 9],
    ]);
    codestream[sot_offset + 6..sot_offset + 10]
        .copy_from_slice(&(psot + u32::try_from(poc.len()).unwrap()).to_be_bytes());

    codestream
}

fn force_cod_lrcp_and_insert_main_header_cprl_poc_with_sentinel_ends(
    mut codestream: Vec<u8>,
) -> Vec<u8> {
    let cod_offset = marker_offset(&codestream, 0x52);
    assert_eq!(
        codestream[cod_offset + 5],
        0x04,
        "fixture must start as CPRL"
    );
    codestream[cod_offset + 5] = 0x00;

    let poc = [
        0xFF, 0x5F, // POC
        0x00, 0x09, // Lpoc
        0x00, // RSpoc
        0x00, // CSpoc
        0x00, 0x02, // LYEpoc: clamp to actual layer count
        0x21, // REpoc: official vectors use 33 as an all-resolutions bound
        0xFF, // CEpoc: official profile 0 vectors use 255 as an all-components bound
        0x04, // Ppoc: CPRL
    ];
    codestream.splice(
        cod_marker_end(&codestream)..cod_marker_end(&codestream),
        poc,
    );
    codestream
}

fn cprl_fixture(pixels: &[u8]) -> Vec<u8> {
    encode(
        pixels,
        64,
        64,
        3,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 1,
            progression_order: EncodeProgressionOrder::Cprl,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("CPRL fixture encode")
}

fn fixture_pixels() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64u8 {
        for x in 0..64u8 {
            pixels.push(x.wrapping_mul(3).wrapping_add(y));
            pixels.push(y.wrapping_mul(5).wrapping_add(x / 2));
            pixels.push(x.wrapping_mul(7).wrapping_sub(y.wrapping_mul(2)));
        }
    }
    pixels
}

#[test]
fn main_header_poc_sentinel_end_bounds_are_clamped_to_tile_shape() {
    let pixels = fixture_pixels();
    let cprl = cprl_fixture(&pixels);
    let with_poc = force_cod_lrcp_and_insert_main_header_cprl_poc_with_sentinel_ends(cprl);

    let decoded = Image::new(&with_poc, &DecodeSettings::default())
        .expect("POC codestream with sentinel end bounds parses")
        .decode_native()
        .expect("POC codestream with sentinel end bounds decodes");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn main_header_poc_changes_packet_iteration_order() {
    let pixels = fixture_pixels();
    let cprl = cprl_fixture(&pixels);
    let with_poc = force_cod_lrcp_and_insert_main_header_cprl_poc(cprl);

    let decoded = Image::new(&with_poc, &DecodeSettings::default())
        .expect("POC codestream parses")
        .decode_native()
        .expect("POC codestream decodes");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn tile_header_poc_changes_packet_iteration_order() {
    let pixels = fixture_pixels();
    let cprl = cprl_fixture(&pixels);
    let with_poc = force_cod_lrcp_and_insert_tile_header_cprl_poc(cprl);

    let decoded = Image::new(&with_poc, &DecodeSettings::default())
        .expect("tile-header POC codestream parses")
        .decode_native()
        .expect("tile-header POC codestream decodes");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.data, pixels);
}
