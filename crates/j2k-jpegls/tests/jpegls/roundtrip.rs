// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encoder round trips across the parameter space, and encoder input
//! validation.

use j2k_jpegls::{
    encode_jpegls, Decoder, InterleaveMode, JpegLsEncodeError, JpegLsEncodeOptions,
    JpegLsEncodeValidation, JpegLsSamples, PresetCodingParameters,
};

use super::support::decoded_samples;

/// Deterministic content with flat runs, edges, and noise.
fn content(width: u32, height: u32, components: u8, max_value: u16) -> Vec<u16> {
    let mut state = 0x9E37_79B9_u32 ^ (width * 31 + height * 17 + u32::from(components));
    let span = u32::from(max_value) + 1;
    let mut out = Vec::new();
    for y in 0..height {
        for x in 0..width {
            for c in 0..u32::from(components) {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = match (x / 5 + y / 3 + c) % 4 {
                    0 => span / 3,
                    1 => (x * 37 + y * 11 + c * 101) % span,
                    2 => (state >> 8) % span,
                    _ => {
                        if (x + y) % 2 == 0 {
                            0
                        } else {
                            span - 1
                        }
                    }
                };
                out.push(u16::try_from(value).expect("sample fits"));
            }
        }
    }
    out
}

/// Encode with round-trip validation, decode, and check every sample.
fn round_trip(
    samples: &[u16],
    width: u32,
    height: u32,
    components: u8,
    options: &JpegLsEncodeOptions,
) {
    let narrow: Vec<u8>;
    let input = if options.bit_depth > 8 {
        JpegLsSamples::U16 {
            data: samples,
            width,
            height,
            components,
        }
    } else {
        narrow = samples
            .iter()
            .map(|&sample| u8::try_from(sample).expect("8-bit sample"))
            .collect();
        JpegLsSamples::U8 {
            data: &narrow,
            width,
            height,
            components,
        }
    };
    let encoded = encode_jpegls(input, options).unwrap_or_else(|error| {
        panic!("encode {width}x{height}x{components} {options:?}: {error}")
    });
    let decoded = Decoder::new(&encoded.data)
        .and_then(|decoder| decoder.decode())
        .unwrap_or_else(|error| panic!("decode {options:?}: {error}"));
    let decoded = decoded_samples(&decoded.samples);
    assert_eq!(decoded.len(), samples.len());
    for (index, (&expected, &actual)) in samples.iter().zip(&decoded).enumerate() {
        assert!(
            expected.abs_diff(actual) <= u16::from(options.near),
            "{options:?}: sample {index} is {actual}, expected {expected}"
        );
    }
}

#[test]
fn every_bit_depth_round_trips_lossless_and_near_lossless() {
    for bits in 2..=16_u8 {
        let max_value = u16::try_from((1_u32 << bits) - 1).expect("max");
        for near in [0, 1, 4] {
            if u16::from(near) > max_value / 2 {
                continue;
            }
            let samples = content(19, 7, 1, max_value);
            round_trip(
                &samples,
                19,
                7,
                1,
                &JpegLsEncodeOptions::new(bits).with_near(near),
            );
        }
    }
}

#[test]
fn every_interleave_mode_and_component_count_round_trips() {
    for components in 1..=4_u8 {
        for interleave in [
            InterleaveMode::None,
            InterleaveMode::Line,
            InterleaveMode::Sample,
        ] {
            for (bits, near) in [(8, 0), (12, 0), (8, 2), (16, 9)] {
                let max_value = u16::try_from((1_u32 << bits) - 1).expect("max");
                let samples = content(13, 9, components, max_value);
                let options = JpegLsEncodeOptions::new(bits)
                    .with_near(near)
                    .with_interleave(interleave);
                round_trip(&samples, 13, 9, components, &options);
            }
        }
    }
    // Interleave mode 0 codes any number of components, one scan each.
    let samples = content(3, 2, 255, 255);
    round_trip(&samples, 3, 2, 255, &JpegLsEncodeOptions::new(8));
}

#[test]
fn edge_geometries_round_trip() {
    for (width, height) in [(1, 1), (1, 40), (40, 1), (2, 2), (300, 3)] {
        for interleave in [InterleaveMode::Line, InterleaveMode::Sample] {
            let samples = content(width, height, 3, 1023);
            let options = JpegLsEncodeOptions::new(10).with_interleave(interleave);
            round_trip(&samples, width, height, 3, &options);
        }
    }
    // Long flat lines exercise the longest run-length codes.
    let flat = std::iter::repeat_n(77_u16, 70_000).collect::<Vec<_>>();
    round_trip(&flat, 70_000 / 2, 2, 1, &JpegLsEncodeOptions::new(8));
}

#[test]
fn restart_intervals_round_trip() {
    for interval in [1, 2, 5, 100] {
        for interleave in [
            InterleaveMode::None,
            InterleaveMode::Line,
            InterleaveMode::Sample,
        ] {
            let samples = content(17, 11, 3, 4095);
            let options = JpegLsEncodeOptions::new(12)
                .with_near(1)
                .with_interleave(interleave)
                .with_restart_interval(interval);
            round_trip(&samples, 17, 11, 3, &options);
        }
    }
    // More than eight intervals wraps the RST numbering.
    let samples = content(5, 30, 1, 255);
    round_trip(
        &samples,
        5,
        30,
        1,
        &JpegLsEncodeOptions::new(8).with_restart_interval(2),
    );
}

/// A MAXVAL below `2^P - 1` sets RANGE and the prediction clamp, as T.87
/// A.2.1 and A.4.2 specify, including where samples jump between 0 and
/// MAXVAL.
#[test]
fn custom_maxval_round_trips() {
    for (bits, max_value, near) in [
        (10, 1000, 0),
        (10, 1000, 2),
        (8, 200, 0),
        (8, 200, 3),
        (16, 60000, 5),
        (12, 1, 0),
    ] {
        let samples = content(23, 13, 1, max_value);
        let preset = PresetCodingParameters {
            max_value,
            ..PresetCodingParameters::default()
        };
        let options = JpegLsEncodeOptions::new(bits)
            .with_near(near)
            .with_preset(preset);
        round_trip(&samples, 23, 13, 1, &options);
        let encoded = encode_jpegls(
            JpegLsSamples::U16 {
                data: &samples,
                width: 23,
                height: 13,
                components: 1,
            },
            &options,
        )
        .expect("encode");
        let info = Decoder::inspect(&encoded.data).expect("inspect");
        assert_eq!(info.scans[0].parameters.max_value, max_value);
    }
}

#[test]
fn custom_thresholds_and_reset_round_trip() {
    let samples = content(29, 17, 1, 255);
    for preset in [
        PresetCodingParameters {
            max_value: 0,
            t1: 1,
            t2: 1,
            t3: 1,
            reset: 3,
        },
        PresetCodingParameters {
            max_value: 0,
            t1: 9,
            t2: 40,
            t3: 200,
            reset: 255,
        },
        PresetCodingParameters {
            max_value: 255,
            t1: 3,
            t2: 7,
            t3: 21,
            reset: 64,
        },
    ] {
        round_trip(
            &samples,
            29,
            17,
            1,
            &JpegLsEncodeOptions::new(8).with_preset(preset),
        );
    }
}

fn has_lse(data: &[u8]) -> bool {
    data.windows(2).any(|pair| pair == [0xFF, 0xF8])
}

#[test]
fn lse_is_written_only_for_non_default_parameters() {
    let samples = content(9, 9, 1, 65535);
    let encode = |options: &JpegLsEncodeOptions| {
        encode_jpegls(
            JpegLsSamples::U16 {
                data: &samples,
                width: 9,
                height: 9,
                components: 1,
            },
            options,
        )
        .expect("encode")
        .data
    };
    assert!(!has_lse(&encode(&JpegLsEncodeOptions::new(16))));
    let defaults = PresetCodingParameters::defaults(65535, 3);
    assert!(!has_lse(&encode(
        &JpegLsEncodeOptions::new(16)
            .with_near(3)
            .with_preset(defaults)
    )));
    let custom = PresetCodingParameters {
        reset: 32,
        ..defaults
    };
    let data = encode(
        &JpegLsEncodeOptions::new(16)
            .with_near(3)
            .with_preset(custom),
    );
    assert!(has_lse(&data));
    assert_eq!(
        Decoder::inspect(&data).expect("inspect").preset,
        Some(custom)
    );
}

#[test]
fn sixteen_bit_output_is_little_endian() {
    let samples = [0x0102_u16, 0xFEDC, 0x8000, 0x00FF];
    let encoded = encode_jpegls(
        JpegLsSamples::U16 {
            data: &samples,
            width: 2,
            height: 2,
            components: 1,
        },
        &JpegLsEncodeOptions::new(16),
    )
    .expect("encode");
    let decoder = Decoder::new(&encoded.data).expect("parse");
    let mut out = [0_u8; 8];
    decoder.decode_into(&mut out, 4).expect("decode");
    assert_eq!(out, [0x02, 0x01, 0xDC, 0xFE, 0x00, 0x80, 0xFF, 0x00]);
}

#[test]
fn invalid_encoder_input_is_rejected() {
    let data = [0_u8; 12];
    let samples = |width, height, components| JpegLsSamples::U8 {
        data: &data,
        width,
        height,
        components,
    };
    let encode =
        |samples, options: JpegLsEncodeOptions| encode_jpegls(samples, &options).unwrap_err();
    let eight = JpegLsEncodeOptions::new(8);

    assert_eq!(
        encode(samples(0, 4, 1), eight),
        JpegLsEncodeError::EmptyDimensions
    );
    assert_eq!(
        encode(samples(70_000, 1, 1), eight),
        JpegLsEncodeError::DimensionsTooLarge {
            width: 70_000,
            height: 1
        }
    );
    assert_eq!(
        encode(samples(4, 3, 0), eight),
        JpegLsEncodeError::NoComponents
    );
    assert_eq!(
        encode(samples(4, 3, 1), JpegLsEncodeOptions::new(9)),
        JpegLsEncodeError::InvalidBitDepth {
            bit_depth: 9,
            sample_type: "u8"
        }
    );
    assert_eq!(
        encode(samples(4, 3, 1), JpegLsEncodeOptions::new(1)),
        JpegLsEncodeError::InvalidBitDepth {
            bit_depth: 1,
            sample_type: "u8"
        }
    );
    assert_eq!(
        encode(samples(4, 4, 1), eight),
        JpegLsEncodeError::SampleLength {
            expected: 16,
            actual: 12
        }
    );
    assert_eq!(
        encode(samples(4, 3, 1), eight.with_near(128)),
        JpegLsEncodeError::InvalidNear {
            near: 128,
            max: 127
        }
    );
}

#[test]
fn invalid_encoder_options_are_rejected() {
    let data = [0_u8; 12];
    let samples = |width, height, components| JpegLsSamples::U8 {
        data: &data,
        width,
        height,
        components,
    };
    let encode =
        |samples, options: JpegLsEncodeOptions| encode_jpegls(samples, &options).unwrap_err();
    let eight = JpegLsEncodeOptions::new(8);
    let three = [0_u8; 3];
    assert_eq!(
        encode_jpegls(
            JpegLsSamples::U8 {
                data: &three,
                width: 1,
                height: 1,
                components: 3
            },
            &eight.with_restart_interval(0)
        )
        .unwrap_err(),
        JpegLsEncodeError::InvalidRestartInterval
    );
    let five = [0_u8; 5];
    assert_eq!(
        encode_jpegls(
            JpegLsSamples::U8 {
                data: &five,
                width: 1,
                height: 1,
                components: 5
            },
            &eight.with_interleave(InterleaveMode::Sample)
        )
        .unwrap_err(),
        JpegLsEncodeError::InvalidInterleave {
            interleave: InterleaveMode::Sample,
            components: 5
        }
    );
    let threshold = PresetCodingParameters {
        t1: 300,
        ..PresetCodingParameters::default()
    };
    assert!(matches!(
        encode(samples(4, 3, 1), eight.with_preset(threshold)),
        JpegLsEncodeError::InvalidPresetParameters { .. }
    ));
    // T1 above the default T2: valid as written with T2 = 0, but the LSE
    // segment would carry T2 < T1, which no decoder accepts.
    let inverted = PresetCodingParameters {
        t1: 10,
        ..PresetCodingParameters::default()
    };
    assert!(matches!(
        encode(samples(4, 3, 1), eight.with_preset(inverted)),
        JpegLsEncodeError::InvalidPresetParameters { .. }
    ));
    let over = [0_u16, 5000];
    assert_eq!(
        encode_jpegls(
            JpegLsSamples::U16 {
                data: &over,
                width: 2,
                height: 1,
                components: 1
            },
            &JpegLsEncodeOptions::new(12)
        )
        .unwrap_err(),
        JpegLsEncodeError::SampleOutOfRange {
            index: 1,
            value: 5000,
            max_value: 4095
        }
    );
    let maxval = PresetCodingParameters {
        max_value: 100,
        ..PresetCodingParameters::default()
    };
    let high = [0_u8, 101];
    assert_eq!(
        encode_jpegls(
            JpegLsSamples::U8 {
                data: &high,
                width: 1,
                height: 2,
                components: 1
            },
            &eight.with_preset(maxval)
        )
        .unwrap_err(),
        JpegLsEncodeError::SampleOutOfRange {
            index: 1,
            value: 101,
            max_value: 100
        }
    );
}

#[test]
fn skip_validation_is_honored() {
    let samples = content(8, 8, 1, 255);
    let narrow: Vec<u8> = samples
        .iter()
        .map(|&s| u8::try_from(s).expect("8-bit"))
        .collect();
    let input = JpegLsSamples::U8 {
        data: &narrow,
        width: 8,
        height: 8,
        components: 1,
    };
    let validated = encode_jpegls(input, &JpegLsEncodeOptions::new(8)).expect("encode");
    let skipped = encode_jpegls(
        input,
        &JpegLsEncodeOptions::new(8).with_validation(JpegLsEncodeValidation::Skip),
    )
    .expect("encode");
    assert_eq!(validated, skipped);
}
