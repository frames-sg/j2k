// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_native::{encode, encode_htj2k, EncodeOptions};
use j2k_test_support::wrap_codestream_jp2;

fn encode_rgb_codestream(htj2k: bool) -> Vec<u8> {
    let pixels = (0..16 * 16 * 3)
        .map(|idx| u8::try_from((idx * 11 + idx / 3) & 0xff).expect("masked fixture byte"))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    if htj2k {
        encode_htj2k(&pixels, 16, 16, 3, 8, false, &options).expect("encode HTJ2K")
    } else {
        encode(&pixels, 16, 16, 3, 8, false, &options).expect("encode J2K")
    }
}

#[test]
fn htj2k_eligibility_accepts_raw_and_jp2_wrapped_inputs() {
    let raw_htj2k = encode_rgb_codestream(true);
    let jp2_htj2k = wrap_codestream_jp2(&raw_htj2k, 16, 16, 3, 8, 16);
    let raw_classic = encode_rgb_codestream(false);
    let jp2_classic = wrap_codestream_jp2(&raw_classic, 16, 16, 3, 8, 16);

    assert!(input_declares_htj2k(&raw_htj2k));
    assert!(input_declares_htj2k(&jp2_htj2k));
    assert!(!input_declares_htj2k(&raw_classic));
    assert!(!input_declares_htj2k(&jp2_classic));

    let mut malformed_jp2 = jp2_htj2k;
    malformed_jp2[11] = 0;
    assert!(!input_declares_htj2k(&malformed_jp2));
    assert!(!input_declares_htj2k(&malformed_jp2[..8]));
}
