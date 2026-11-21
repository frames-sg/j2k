// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode and encode against the `CharLS` fixture matrix.

use super::support::{decoded_samples, matrix_case, matrix_cases, MatrixCase, Source};
use j2k_jpegls::{
    encode_jpegls, Decoder, InterleaveMode, JpegLsEncodeValidation, PresetCodingParameters,
};

#[test]
fn matrix_covers_the_dicom_parameter_space() {
    let cases = matrix_cases();
    assert!(cases.len() >= 50, "matrix has {} cases", cases.len());
    for bits in [2, 5, 8, 10, 12, 13, 16] {
        assert!(
            cases.iter().any(|case| case.bits == bits),
            "no {bits}-bit case"
        );
    }
    for near in [0, 1, 3, 7] {
        assert!(
            cases.iter().any(|case| case.near == near),
            "no NEAR {near} case"
        );
    }
    for interleave in [
        InterleaveMode::None,
        InterleaveMode::Line,
        InterleaveMode::Sample,
    ] {
        assert!(cases
            .iter()
            .any(|case| case.components == 3 && case.interleave == interleave));
    }
    assert!(cases.iter().any(|case| case.restart != 0));
    assert!(cases.iter().any(|case| case.source == Source::Dicom));
}

/// Every codestream decodes to `CharLS`'s reference exactly, through both the
/// owned and the byte-buffer API.
#[test]
fn decoder_matches_charls_for_every_case() {
    for case in matrix_cases() {
        let decoder = Decoder::new(case.codestream)
            .unwrap_or_else(|error| panic!("{}: parse failed: {error}", case.name));
        let info = decoder.info();
        assert_eq!(info.dimensions, (case.width, case.height), "{}", case.name);
        assert_eq!(info.components, case.components, "{}", case.name);
        assert_eq!(info.bit_depth, case.bits, "{}", case.name);
        assert_eq!(info.near, case.near, "{}", case.name);
        if case.components > 1 {
            assert_eq!(info.interleave, case.interleave, "{}", case.name);
        }
        assert_eq!(
            info.restart_interval,
            (case.restart != 0).then_some(u32::from(case.restart)),
            "{}",
            case.name
        );

        let image = decoder
            .decode()
            .unwrap_or_else(|error| panic!("{}: decode failed: {error}", case.name));
        assert_eq!(
            decoded_samples(&image.samples),
            case.reference_samples(),
            "{}: decoded samples differ from CharLS",
            case.name
        );

        // A padded stride leaves the padding untouched and the samples in
        // the same byte layout as the reference.
        let row_bytes = info.row_bytes().expect("row size");
        let stride = row_bytes + 3;
        let rows = usize::try_from(case.height).expect("height");
        let mut out = std::iter::repeat_n(0xA5_u8, stride * rows).collect::<Vec<_>>();
        decoder
            .decode_into(&mut out, stride)
            .unwrap_or_else(|error| panic!("{}: decode_into failed: {error}", case.name));
        for (row, expected) in out.chunks(stride).zip(case.reference.chunks(row_bytes)) {
            assert_eq!(&row[..row_bytes], expected, "{}", case.name);
            assert!(
                row[row_bytes..].iter().all(|&byte| byte == 0xA5),
                "{}",
                case.name
            );
        }
    }
}

/// Remove an LSE segment that only restates the default parameters. `CharLS`
/// writes one for more than 12 bits per sample by default (for Java Advanced
/// Imaging) and when a preset is passed with fields that resolve to the
/// defaults; j2k-jpegls writes LSE only for non-default parameters.
fn strip_default_preset(case: &MatrixCase, codestream: &[u8]) -> Vec<u8> {
    let Some(at) = codestream
        .windows(5)
        .position(|w| w == [0xFF, 0xF8, 0, 13, 1])
    else {
        return Vec::from(codestream);
    };
    let field = |index: usize| {
        u16::from_be_bytes([
            codestream[at + 5 + 2 * index],
            codestream[at + 6 + 2 * index],
        ])
    };
    let written = PresetCodingParameters {
        max_value: field(0),
        t1: field(1),
        t2: field(2),
        t3: field(3),
        reset: field(4),
    };
    let full_scale = u16::try_from((1_u32 << case.bits) - 1).expect("full scale");
    if written != PresetCodingParameters::defaults(full_scale, case.near) {
        return Vec::from(codestream);
    }
    [&codestream[..at], &codestream[at + 15..]].concat()
}

/// For the same samples and parameters, j2k-jpegls writes the same bytes
/// as `CharLS 2.4.2`, apart from LSE segments that restate the defaults.
#[test]
fn encoder_output_is_byte_identical_to_charls() {
    let mut compared = 0;
    for case in matrix_cases()
        .into_iter()
        .filter(|case| case.source != Source::Dicom)
    {
        let encoded = case
            .with_samples(|samples| encode_jpegls(samples, &case.options()))
            .unwrap_or_else(|error| panic!("{}: encode failed: {error}", case.name));
        let expected = strip_default_preset(&case, case.codestream);
        assert!(
            encoded.data == expected,
            "{}: encoded {} bytes, CharLS wrote {}; first difference at byte {:?}",
            case.name,
            encoded.data.len(),
            expected.len(),
            encoded.data.iter().zip(&expected).position(|(a, b)| a != b)
        );
        compared += usize::from(case.source == Source::Charls);
    }
    assert!(compared >= 45, "only {compared} CharLS cases compared");
}

/// The real-world DICOM vector decodes exactly, and re-encoding its samples
/// reproduces its codestream apart from the default-valued LSE segment.
#[test]
fn dicom_vector_round_trips() {
    let case = matrix_case("MR_small_jpeg_ls_lossless");
    let decoded = Decoder::new(case.codestream)
        .and_then(|decoder| decoder.decode())
        .expect("decode MR_small");
    let samples = decoded_samples(&decoded.samples);
    assert_eq!(samples, case.reference_samples());
    let encoded = case
        .with_samples(|samples| encode_jpegls(samples, &case.options()))
        .expect("re-encode MR_small");
    assert_eq!(encoded.data, strip_default_preset(&case, case.codestream));
}

#[test]
fn inspect_reports_scans_restarts_and_presets() {
    let five = Decoder::inspect(matrix_case("c5_8_mixed_ilv0").codestream).expect("inspect");
    assert_eq!(five.scans.len(), 5);
    assert_eq!(five.scans[4].components, [4]);

    let rgb = Decoder::inspect(matrix_case("rgb8_mixed_ilv2_near3").codestream).expect("inspect");
    assert_eq!(rgb.scans.len(), 1);
    assert_eq!(rgb.scans[0].components, [0, 1, 2]);
    assert_eq!(rgb.scans[0].near, 3);
    assert_eq!(
        rgb.scans[0].parameters,
        PresetCodingParameters::defaults(255, 3)
    );

    let restart =
        Decoder::inspect(matrix_case("rgb8_mixed_ilv1_restart3").codestream).expect("inspect");
    assert_eq!(restart.restart_interval, Some(3));

    let jai = Decoder::inspect(matrix_case("g16_mixed_jai").codestream).expect("inspect");
    assert_eq!(jai.preset, Some(PresetCodingParameters::defaults(65535, 0)));
    let plain = Decoder::inspect(matrix_case("g16_mixed").codestream).expect("inspect");
    assert_eq!(plain.preset, None);
    assert_eq!(plain.scans[0].parameters, jai.scans[0].parameters);

    let custom = Decoder::inspect(matrix_case("g8_preset_thresholds").codestream).expect("inspect");
    let expected = PresetCodingParameters {
        max_value: 255,
        t1: 4,
        t2: 8,
        t3: 22,
        reset: 32,
    };
    assert_eq!(custom.preset, Some(expected));
    assert_eq!(custom.scans[0].parameters, expected);
}

/// Skipping validation does not change the output bytes.
#[test]
fn skipping_validation_returns_the_same_codestream() {
    for name in ["g12_mixed_near3", "rgb8_mixed_ilv2", "g8_mixed_restart5"] {
        let case = matrix_case(name);
        let options = case.options().with_validation(JpegLsEncodeValidation::Skip);
        let skipped = case
            .with_samples(|samples| encode_jpegls(samples, &options))
            .expect("encode");
        let validated = case
            .with_samples(|samples| encode_jpegls(samples, &case.options()))
            .expect("encode");
        assert_eq!(skipped, validated, "{name}");
    }
}
