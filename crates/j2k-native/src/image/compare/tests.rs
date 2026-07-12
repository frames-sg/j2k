// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{encode, DecodeError, DecodeSettings, EncodeOptions, Image, DEFAULT_MAX_DECODE_BYTES};

fn encode_gray(samples: &[u8]) -> alloc::vec::Vec<u8> {
    encode(
        samples,
        8,
        8,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 0,
            reversible: true,
            ..EncodeOptions::default()
        },
    )
    .expect("comparison fixture encodes")
}

#[test]
fn paired_decode_distinguishes_equal_and_different_pixels() {
    let source_samples = [42_u8; 64];
    let mut different_samples = source_samples;
    different_samples[31] = 43;
    let source_bytes = encode_gray(&source_samples);
    let equal_bytes = encode_gray(&source_samples);
    let different_bytes = encode_gray(&different_samples);
    let settings = DecodeSettings::default();
    let source = Image::new(&source_bytes, &settings).expect("source parses");
    let equal = Image::new(&equal_bytes, &settings).expect("equal image parses");
    let different = Image::new(&different_bytes, &settings).expect("different image parses");

    assert!(source
        .decoded_samples_equal_with_retained_bytes(&equal, &equal_bytes)
        .expect("equal paired decode"));
    assert!(!source
        .decoded_samples_equal(&different)
        .expect("different paired decode"));
}

#[test]
fn retained_parse_baseline_is_enforced_before_parser_growth() {
    let bytes = encode_gray(&[7_u8; 64]);
    let settings = DecodeSettings::default();
    let comfortable_baseline = DEFAULT_MAX_DECODE_BYTES - 1024 * 1024;
    Image::new_with_retained_baseline(&bytes, &settings, comfortable_baseline)
        .expect("small parser fits within one MiB of remaining headroom");
    assert!(
        Image::new_with_retained_baseline(&bytes, &settings, DEFAULT_MAX_DECODE_BYTES - 1).is_err(),
        "one remaining byte cannot hide parser-owned allocation growth"
    );
}

#[test]
fn retained_output_capacity_is_enforced_before_native_decode_growth() {
    let bytes = encode_gray(&[9_u8; 64]);
    let settings = DecodeSettings::default();
    let image = Image::new(&bytes, &settings).expect("fixture parses");

    image
        .decode_native_with_retained_capacity(DEFAULT_MAX_DECODE_BYTES - 1024 * 1024)
        .expect("small decode fits within one MiB of remaining headroom");
    let error = image
        .decode_native_with_retained_capacity(DEFAULT_MAX_DECODE_BYTES)
        .err()
        .expect("retained output capacity leaves no room for parsed metadata");
    assert!(matches!(
        error,
        DecodeError::AllocationTooLarge { requested, cap, .. }
            if requested > cap && cap == DEFAULT_MAX_DECODE_BYTES
    ));
}

#[test]
fn retained_output_capacity_is_enforced_for_owned_component_decode() {
    let bytes = encode_gray(&[11_u8; 64]);
    let settings = DecodeSettings::default();
    let image = Image::new(&bytes, &settings).expect("fixture parses");

    image
        .decode_native_components_with_retained_capacity(DEFAULT_MAX_DECODE_BYTES - 1024 * 1024)
        .expect("small component decode fits within one MiB of remaining headroom");
    let error = image
        .decode_native_components_with_retained_capacity(DEFAULT_MAX_DECODE_BYTES)
        .err()
        .expect("retained output capacity leaves no room for parsed metadata");
    assert!(matches!(
        error,
        DecodeError::AllocationTooLarge { requested, cap, .. }
            if requested > cap && cap == DEFAULT_MAX_DECODE_BYTES
    ));
}
