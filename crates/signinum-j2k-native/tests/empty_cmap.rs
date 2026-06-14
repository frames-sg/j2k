// SPDX-License-Identifier: Apache-2.0

//! Regression scaffold for JP2 palette/component-map validation.

use signinum_j2k_native::{
    encode, ColorError, DecodeError, DecodeSettings, EncodeOptions, FormatError, Image,
};

fn jp2_box(box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(8u32 + payload.len() as u32).to_be_bytes());
    out.extend_from_slice(box_type);
    out.extend_from_slice(payload);
    out
}

#[test]
fn empty_cmap_with_palette_returns_error() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode fixture");
    let jp2 = jp2_with_empty_cmap(&codestream);

    let result = Image::new(&jp2, &DecodeSettings::default()).and_then(|image| image.decode());

    let err = result.expect_err("empty cmap must be rejected explicitly");
    assert!(
        matches!(
            err,
            DecodeError::Format(FormatError::InvalidBox)
                | DecodeError::Color(ColorError::PaletteResolutionFailed)
        ),
        "unexpected empty cmap error: {err:?}"
    );
}

fn jp2_with_empty_cmap(codestream: &[u8]) -> Vec<u8> {
    let ihdr = {
        let mut payload = Vec::new();
        payload.extend_from_slice(&4_u32.to_be_bytes());
        payload.extend_from_slice(&4_u32.to_be_bytes());
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&[7, 7, 0, 0]);
        jp2_box(b"ihdr", &payload)
    };
    let colr = jp2_box(b"colr", &[1, 0, 0, 0, 0, 0, 17]);
    let pclr = {
        let mut payload = Vec::new();
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.push(1);
        payload.push(7);
        payload.push(0);
        jp2_box(b"pclr", &payload)
    };
    let cmap = jp2_box(b"cmap", &[]);

    let mut jp2h_payload = Vec::new();
    jp2h_payload.extend_from_slice(&ihdr);
    jp2h_payload.extend_from_slice(&colr);
    jp2h_payload.extend_from_slice(&pclr);
    jp2h_payload.extend_from_slice(&cmap);

    let mut out = Vec::new();
    out.extend_from_slice(&jp2_box(b"jP  ", &[0x0d, 0x0a, 0x87, 0x0a]));
    out.extend_from_slice(&jp2_box(b"ftyp", b"jp2 \0\0\0\0jp2 "));
    out.extend_from_slice(&jp2_box(b"jp2h", &jp2h_payload));
    out.extend_from_slice(&jp2_box(b"jp2c", codestream));
    out
}
