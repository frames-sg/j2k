// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Write;

use j2k_jpeg::transcode::{encode_baseline_dct_image, extract_dct_blocks, DctExtractOptions};
use j2k_test_support as fixtures;

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn lowercase_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(output, "{byte:02x}").expect("write to String");
    }
    output
}

fn assert_exact_hex(actual: &[u8], golden: &str) {
    let actual = lowercase_hex(actual);
    let expected = golden
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect::<String>();
    assert_eq!(actual, expected, "canonical re-emission bytes changed");
}

#[test]
fn valid_dct_reemission_bytes_remain_exact() {
    let cases = [
        (
            fixtures::grayscale_8x8_jpeg(),
            include_str!("golden/dct_reemit_grayscale.hex"),
        ),
        (
            fixtures::minimal_baseline_420_jpeg(),
            include_str!("golden/dct_reemit_420.hex"),
        ),
    ];
    let actual = cases.map(|(jpeg, golden)| {
        let image = extract_dct_blocks(&jpeg, DctExtractOptions::default())
            .expect("extract valid baseline DCT image");
        let encoded = encode_baseline_dct_image(&image).expect("re-emit valid baseline DCT image");
        assert_exact_hex(&encoded, golden);
        (encoded.len(), fnv1a64(&encoded))
    });

    assert_eq!(
        actual,
        [(323, 0xeb28_3c65_094a_b76a), (661, 0x44da_dc89_e927_2c00),]
    );
}

#[test]
fn valid_dct_reemission_preserves_quantized_component_parity() {
    for jpeg in [
        fixtures::grayscale_8x8_jpeg(),
        fixtures::minimal_baseline_420_jpeg(),
    ] {
        let expected = extract_dct_blocks(&jpeg, DctExtractOptions::default())
            .expect("extract source DCT image");
        let encoded =
            encode_baseline_dct_image(&expected).expect("re-emit valid baseline DCT image");
        let actual = extract_dct_blocks(&encoded, DctExtractOptions::default())
            .expect("extract re-emitted DCT image");

        assert_eq!(actual.width, expected.width);
        assert_eq!(actual.height, expected.height);
        assert_eq!(actual.components, expected.components);
    }
}
