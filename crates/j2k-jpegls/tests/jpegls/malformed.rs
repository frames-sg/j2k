// SPDX-License-Identifier: MIT OR Apache-2.0

//! Truncated, corrupted, and hostile input returns errors, never panics or
//! silently wrong samples.

use j2k_jpegls::{DecodeOptions, Decoder, JpegLsError, UnsupportedFeature};

use super::support::{decoded_samples, matrix_case};

/// Fixtures that cover each interleave mode, restart markers, escape codes,
/// and 16-bit samples.
const TRUNCATION_CASES: &[&str] = &[
    "g8_mixed",
    "g16_noise",
    "rgb8_mixed_ilv0",
    "rgb8_mixed_ilv1_restart3",
    "rgb8_flat_ilv2_restart4",
    "g12_mixed_near3_restart6",
    "g8_preset_thresholds",
];

fn decode(data: &[u8]) -> Result<Vec<u16>, JpegLsError> {
    let decoder = Decoder::new(data)?;
    Ok(decoded_samples(&decoder.decode()?.samples))
}

/// Position of the first SOS marker.
fn sos_offset(codestream: &[u8]) -> usize {
    codestream
        .windows(2)
        .position(|pair| pair == [0xFF, 0xDA])
        .expect("SOS")
}

/// Every proper prefix is an error. With an EOI appended, a prefix cut in
/// the entropy-coded data either fails or, when only padding was cut,
/// decodes to the reference.
#[test]
fn every_truncation_is_an_error() {
    for name in TRUNCATION_CASES {
        let case = matrix_case(name);
        let reference = case.reference_samples();
        let data = case.codestream;
        for length in 0..data.len() {
            let prefix = &data[..length];
            assert!(
                Decoder::inspect(prefix).is_err(),
                "{name}: inspect accepted {length} bytes"
            );
            assert!(
                decode(prefix).is_err(),
                "{name}: decode accepted {length} bytes"
            );

            let mut closed = Vec::from(prefix);
            closed.extend_from_slice(&[0xFF, 0xD9]);
            if let Ok(samples) = decode(&closed) {
                assert_eq!(
                    samples, reference,
                    "{name}: {length} bytes + EOI decoded wrongly"
                );
            }
        }
    }
}

/// Flipping any single bit of the entropy-coded data never panics. Most
/// flips are errors; the rest decode to some image.
#[test]
fn bit_flips_in_entropy_data_do_not_panic() {
    for name in [
        "g8_mixed",
        "rgb8_mixed_ilv2",
        "g16_mixed_restart1",
        "c4_8_mixed_ilv1",
    ] {
        let case = matrix_case(name);
        let start = sos_offset(case.codestream);
        let mut errors = 0;
        let mut total = 0;
        for byte in start..case.codestream.len() - 2 {
            for bit in 0..8 {
                let mut data = Vec::from(case.codestream);
                data[byte] ^= 1 << bit;
                total += 1;
                errors += usize::from(decode(&data).is_err());
            }
        }
        assert!(errors > 0 && total > 0, "{name}: no flip was detected");
    }
}

/// Deterministic pseudo-random bytes after a valid header never panic.
#[test]
fn random_entropy_data_does_not_panic() {
    let case = matrix_case("rgb8_mixed_ilv1");
    let header = &case.codestream[..sos_offset(case.codestream) + 14];
    let mut state = 0x2545_F491_u32;
    for length in [0, 1, 7, 64, 500, 4000] {
        for _ in 0..20 {
            let mut data = Vec::from(header);
            for _ in 0..length {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                data.push(state.to_be_bytes()[0]);
            }
            data.extend_from_slice(&[0xFF, 0xD9]);
            let _ = decode(&data);
        }
    }
}

/// Patch the codestream of `name` at `offset` from the start of `marker`.
fn patched(name: &str, marker: [u8; 2], offset: usize, value: u8) -> Vec<u8> {
    let mut data = Vec::from(matrix_case(name).codestream);
    let at = data
        .windows(2)
        .position(|pair| pair == marker)
        .expect("marker present");
    data[at + offset] = value;
    data
}

const SOF55: [u8; 2] = [0xFF, 0xF7];
const SOS: [u8; 2] = [0xFF, 0xDA];
const LSE: [u8; 2] = [0xFF, 0xF8];

#[test]
fn corrupted_headers_are_errors() {
    let invalid_segment = |data: &[u8]| {
        matches!(
            Decoder::inspect(data),
            Err(JpegLsError::InvalidSegment { .. })
        )
    };
    // Bits per sample outside 2..=16.
    assert!(invalid_segment(&patched("g8_mixed", SOF55, 4, 1)));
    assert!(invalid_segment(&patched("g8_mixed", SOF55, 4, 17)));
    // Zero width.
    let mut zero_width = patched("g8_mixed", SOF55, 7, 0);
    let at = zero_width
        .windows(2)
        .position(|pair| pair == SOF55)
        .expect("SOF55");
    zero_width[at + 8] = 0;
    assert!(invalid_segment(&zero_width));
    // A component count that disagrees with the segment length.
    assert!(invalid_segment(&patched("g8_mixed", SOF55, 9, 2)));
    // An interleave mode above 2, and ILV 0 with three components.
    assert!(invalid_segment(&patched("g8_mixed", SOS, 8, 3)));
    assert!(invalid_segment(&patched("rgb8_mixed_ilv1", SOS, 12, 0)));
    // A scan naming a component that is not in the frame.
    assert!(invalid_segment(&patched("g8_mixed", SOS, 5, 9)));
    // A nonzero Ah nibble.
    assert!(invalid_segment(&patched("g8_mixed", SOS, 9, 0x10)));
    // An unknown LSE ID.
    assert!(invalid_segment(&patched("g8_preset_thresholds", LSE, 4, 9)));
    // A data byte where a marker should be.
    assert!(matches!(
        Decoder::inspect(&patched("g8_mixed", SOF55, 0, 0x12)),
        Err(JpegLsError::UnexpectedMarker { .. })
    ));
    // NEAR above MAXVAL / 2.
    assert!(matches!(
        Decoder::inspect(&patched("g2_mixed", SOS, 7, 2)),
        Err(JpegLsError::InvalidPresetParameters { .. })
    ));
    // T1 above MAXVAL.
    let mut threshold = patched("g8_preset_thresholds", LSE, 7, 0x01);
    let at = threshold
        .windows(2)
        .position(|pair| pair == LSE)
        .expect("LSE");
    threshold[at + 8] = 0x00;
    assert!(matches!(
        Decoder::inspect(&threshold),
        Err(JpegLsError::InvalidPresetParameters { .. })
    ));
    // Not JPEG at all.
    assert_eq!(
        Decoder::inspect(b"not a jpeg").unwrap_err(),
        JpegLsError::MissingSoi
    );
    assert_eq!(Decoder::inspect(&[]).unwrap_err(), JpegLsError::MissingSoi);
}

#[test]
fn structural_errors_are_reported() {
    let soi_sof_eoi = [
        0xFF, 0xD8, 0xFF, 0xF7, 0, 11, 8, 0, 1, 0, 1, 1, 1, 0x11, 0, 0xFF, 0xD9,
    ];
    assert_eq!(
        Decoder::inspect(&soi_sof_eoi).unwrap_err(),
        JpegLsError::MissingMarker { marker: "SOS" }
    );
    assert_eq!(
        Decoder::inspect(&[0xFF, 0xD8, 0xFF, 0xD9]).unwrap_err(),
        JpegLsError::MissingMarker { marker: "SOF55" }
    );
    // A second SOF55.
    let case = matrix_case("g8_mixed");
    let sof_end = 2 + 2 + 11;
    let mut twice = Vec::from(&case.codestream[..sof_end]);
    twice.extend_from_slice(&case.codestream[2..]);
    assert!(matches!(
        Decoder::inspect(&twice),
        Err(JpegLsError::UnexpectedMarker { marker: 0xF7, .. })
    ));
    // A frame with three components and only the first scan.
    let rgb = matrix_case("rgb8_mixed_ilv0").codestream;
    let second_scan = rgb
        .windows(2)
        .enumerate()
        .filter(|(_, pair)| *pair == SOS)
        .nth(1)
        .map(|(at, _)| at)
        .expect("second SOS");
    let mut one_scan = Vec::from(&rgb[..second_scan]);
    one_scan.extend_from_slice(&[0xFF, 0xD9]);
    assert_eq!(
        Decoder::inspect(&one_scan).unwrap_err(),
        JpegLsError::MissingComponent { component_id: 2 }
    );
    // The same component coded twice.
    let first_scan = sos_offset(rgb);
    let mut repeated = Vec::from(&rgb[..second_scan]);
    repeated.extend_from_slice(&rgb[first_scan..]);
    assert!(matches!(
        Decoder::inspect(&repeated),
        Err(JpegLsError::InvalidSegment { .. })
    ));
    // Bytes after the EOI are ignored, as DICOM pads fragments to even
    // lengths.
    let mut padded = Vec::from(case.codestream);
    padded.push(0);
    assert_eq!(decode(&padded).expect("padded"), case.reference_samples());
}

#[test]
fn unsupported_features_are_reported() {
    let unsupported = |data: &[u8]| match Decoder::inspect(data) {
        Err(JpegLsError::Unsupported(feature)) => feature,
        other => panic!("expected an unsupported feature, got {other:?}"),
    };
    assert_eq!(
        unsupported(&patched("g8_mixed", SOF55, 1, 0xC3)),
        UnsupportedFeature::NonJpegLsFrame { marker: 0xC3 }
    );
    assert!(matches!(
        unsupported(&patched("g8_mixed", SOF55, 11, 0x21)),
        UnsupportedFeature::SamplingFactors {
            horizontal: 2,
            vertical: 1,
            ..
        }
    ));
    assert_eq!(
        unsupported(&patched("g8_mixed", SOS, 9, 1)),
        UnsupportedFeature::PointTransform(1)
    );
    assert_eq!(
        unsupported(&patched("g8_mixed", SOS, 6, 1)),
        UnsupportedFeature::MappingTable
    );
    for (id, feature) in [
        (2, UnsupportedFeature::MappingTable),
        (3, UnsupportedFeature::MappingTable),
        (4, UnsupportedFeature::OversizeDimensions),
    ] {
        assert_eq!(
            unsupported(&patched("g8_preset_thresholds", LSE, 4, id)),
            feature
        );
    }
    // Height 0 (set by a later DNL marker).
    let mut deferred = patched("g8_mixed", SOF55, 5, 0);
    let at = deferred
        .windows(2)
        .position(|pair| pair == SOF55)
        .expect("SOF55");
    deferred[at + 6] = 0;
    assert_eq!(unsupported(&deferred), UnsupportedFeature::DeferredHeight);
    // HP color transform in APP8 "mrfx".
    let case = matrix_case("rgb8_mixed_ilv2");
    for (transform, expected) in [
        (0_u8, None),
        (1, Some(UnsupportedFeature::ColorTransform(1))),
    ] {
        let mut data = Vec::from(&case.codestream[..2]);
        data.extend_from_slice(&[0xFF, 0xE8, 0, 7, b'm', b'r', b'f', b'x', transform]);
        data.extend_from_slice(&case.codestream[2..]);
        match expected {
            None => assert_eq!(decode(&data).expect("mrfx 0"), case.reference_samples()),
            Some(feature) => assert_eq!(unsupported(&data), feature),
        }
    }
}

#[test]
fn restart_marker_errors_are_reported() {
    let case = matrix_case("rgb8_mixed_ilv1_restart3");
    let data = case.codestream;
    let restarts: Vec<usize> = (sos_offset(data)..data.len() - 1)
        .filter(|&at| data[at] == 0xFF && (0xD0..=0xD7).contains(&data[at + 1]))
        .collect();
    assert_eq!(restarts.len(), 7, "23 lines in intervals of 3");
    // Out of sequence.
    let mut wrong = Vec::from(data);
    wrong[restarts[1] + 1] = 0xD5;
    assert_eq!(
        decode(&wrong).unwrap_err(),
        JpegLsError::RestartMarker {
            expected: 1,
            offset: restarts[1]
        }
    );
    // Missing.
    let mut missing = Vec::from(data);
    missing.drain(restarts[2]..restarts[2] + 2);
    assert!(decode(&missing).is_err());
    // A restart marker where the scan should end.
    let plain = matrix_case("g8_mixed").codestream;
    let mut extra = Vec::from(&plain[..plain.len() - 2]);
    extra.extend_from_slice(&[0xFF, 0xD0, 0x00, 0xFF, 0xD9]);
    assert!(matches!(
        decode(&extra),
        Err(JpegLsError::UnexpectedMarker { marker: 0xD0, .. })
    ));
}

#[test]
fn leftover_entropy_data_is_an_error() {
    let case = matrix_case("g8_mixed");
    let mut data = Vec::from(&case.codestream[..case.codestream.len() - 2]);
    data.extend_from_slice(&[0x00, 0x00, 0xFF, 0xD9]);
    assert_eq!(
        decode(&data).unwrap_err(),
        JpegLsError::TrailingEntropyData { scan: 0 }
    );
}

/// Huge declared dimensions are rejected before allocation; `inspect` still
/// reports them.
#[test]
fn absurd_dimensions_hit_the_output_limit() {
    let header = [
        0xFF, 0xD8, 0xFF, 0xF7, 0, 11, 16, 0xFF, 0xFF, 0xFF, 0xFF, 1, 1, 0x11, 0, 0xFF, 0xDA, 0, 8,
        1, 1, 0, 0, 0, 0, 0xFF, 0xD9,
    ];
    let info = Decoder::inspect(&header).expect("inspect");
    assert_eq!(info.dimensions, (65535, 65535));
    assert!(matches!(
        Decoder::new(&header),
        Err(JpegLsError::OutputLimitExceeded {
            required: 8_589_672_450,
            ..
        })
    ));

    let mut wide = Vec::from([0xFF, 0xD8, 0xFF, 0xF7]);
    wide.extend_from_slice(&(8_u16 + 3 * 255).to_be_bytes());
    wide.extend_from_slice(&[16, 0xFF, 0xFF, 0xFF, 0xFF, 255]);
    for id in 0..=254_u8 {
        wide.extend_from_slice(&[id, 0x11, 0]);
    }
    for id in 0..=254_u8 {
        wide.extend_from_slice(&[0xFF, 0xDA, 0, 8, 1, id, 0, 0, 0, 0]);
    }
    wide.extend_from_slice(&[0xFF, 0xD9]);
    assert_eq!(Decoder::inspect(&wide).expect("inspect").scans.len(), 255);
    assert!(matches!(
        Decoder::new(&wide),
        Err(JpegLsError::OutputLimitExceeded { .. })
    ));

    // A caller-chosen limit applies too.
    let small = matrix_case("g8_mixed");
    let options = DecodeOptions::default().with_max_output_bytes(100);
    assert!(matches!(
        Decoder::with_options(small.codestream, options),
        Err(JpegLsError::OutputLimitExceeded {
            required: 1189,
            limit: 100
        })
    ));
}

#[test]
fn output_buffer_and_stride_are_checked() {
    let case = matrix_case("g16_mixed");
    let decoder = Decoder::new(case.codestream).expect("parse");
    let row_bytes = decoder.info().row_bytes().expect("row size");
    let mut short = std::iter::repeat_n(0, row_bytes * 23 - 1).collect::<Vec<_>>();
    assert!(matches!(
        decoder.decode_into(&mut short, row_bytes),
        Err(JpegLsError::OutputBufferTooSmall { .. })
    ));
    let mut out = std::iter::repeat_n(0, row_bytes * 23).collect::<Vec<_>>();
    assert!(matches!(
        decoder.decode_into(&mut out, row_bytes - 1),
        Err(JpegLsError::InvalidStride { .. })
    ));
}
