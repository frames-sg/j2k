#![no_main]

use j2k::J2kDecoder;
use j2k_test_support::minimal_j2k_codestream;
use libfuzzer_sys::fuzz_target;

const MAX_ICC_BYTES: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_ICC_BYTES {
        return;
    }

    let jp2 = jp2_with_gray_icc(data);
    let Ok(mut decoder) = J2kDecoder::new(&jp2) else {
        return;
    };
    let _ = decoder.decode_srgb8();
});

fn jp2_with_gray_icc(profile: &[u8]) -> Vec<u8> {
    let codestream = minimal_j2k_codestream();
    let mut jp2h = Vec::new();
    push_box(
        &mut jp2h,
        *b"ihdr",
        &[0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 7, 7, 0, 0],
    );
    let mut color = Vec::with_capacity(profile.len().saturating_add(3));
    color.extend_from_slice(&[2, 0, 0]);
    color.extend_from_slice(profile);
    push_box(&mut jp2h, *b"colr", &color);

    let mut out = Vec::with_capacity(
        codestream
            .len()
            .saturating_add(jp2h.len())
            .saturating_add(48),
    );
    out.extend_from_slice(&[0, 0, 0, 12]);
    out.extend_from_slice(b"jP  ");
    out.extend_from_slice(&[0x0d, 0x0a, 0x87, 0x0a]);
    out.extend_from_slice(&[0, 0, 0, 20]);
    out.extend_from_slice(b"ftypjp2 \0\0\0\0jp2 ");
    push_box(&mut out, *b"jp2h", &jp2h);
    push_box(&mut out, *b"jp2c", &codestream);
    out
}

fn push_box(out: &mut Vec<u8>, box_type: [u8; 4], payload: &[u8]) {
    let Ok(length) = u32::try_from(payload.len().saturating_add(8)) else {
        return;
    };
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(&box_type);
    out.extend_from_slice(payload);
}
