// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_jpeg::{DecodeOptions, Decoder, JpegError, JpegView, PixelFormat, Warning};
use j2k_test_support::progressive_8x8_jpeg;

fn decode_rgb(bytes: &[u8]) -> (Vec<u8>, Vec<Warning>) {
    let decoder = Decoder::new(bytes).expect("progressive decoder");
    let (width, height) = decoder.info().dimensions;
    let stride = usize::try_from(width * 3).expect("fixture stride");
    let mut pixels = vec![0; stride * usize::try_from(height).expect("fixture height")];
    let outcome = decoder
        .decode_into(&mut pixels, stride, PixelFormat::Rgb8)
        .expect("progressive decode");
    (pixels, outcome.warnings)
}

#[test]
fn valid_multiscan_and_missing_eoi_decode_identically_with_typed_warning() {
    let complete = progressive_8x8_jpeg();
    let mut missing_eoi = complete.clone();
    missing_eoi.truncate(missing_eoi.len() - 2);

    let (complete_pixels, complete_warnings) = decode_rgb(&complete);
    let (missing_pixels, missing_warnings) = decode_rgb(&missing_eoi);

    assert_eq!(missing_pixels, complete_pixels);
    assert!(!complete_warnings.contains(&Warning::MissingEoi));
    assert!(missing_warnings.contains(&Warning::MissingEoi));
}

#[test]
fn strict_progressive_eof_requires_all_bands_at_final_approximation() {
    let complete = progressive_8x8_jpeg();
    let scans: Vec<_> = complete
        .windows(2)
        .enumerate()
        .filter_map(|(offset, marker)| (marker == [0xff, 0xda]).then_some(offset))
        .collect();
    for cutoff in [scans[1], *scans.last().unwrap()] {
        let partial = &complete[..cutoff];
        assert!(
            Decoder::new(partial)
                .and_then(|decoder| {
                    decoder.decode_request(j2k_jpeg::DecodeRequest::full(PixelFormat::Rgb8))
                })
                .is_err(),
            "unverified EOF at {cutoff}"
        );
        // An actual EOI can intentionally terminate an otherwise legal partial
        // progressive script. EOF cannot establish that intent.
        let mut terminated = partial.to_vec();
        terminated.extend_from_slice(&[0xff, 0xd9]);
        let (expected, _) = decode_rgb(&terminated);
        let lenient = Decoder::from_view(
            JpegView::parse_with_options(partial, DecodeOptions::default().with_strict(false))
                .unwrap(),
        )
        .unwrap();
        let (pixels, outcome) = lenient
            .decode_request(j2k_jpeg::DecodeRequest::full(PixelFormat::Rgb8))
            .unwrap();
        assert_eq!(pixels, expected);
        assert!(outcome.recovered);
        assert_eq!(outcome.warnings, [Warning::MissingEoi]);
        assert!(matches!(
            j2k_jpeg::transcode::extract_dct_blocks(
                partial,
                j2k_jpeg::transcode::DctExtractOptions::default(),
            ),
            Err(JpegError::RecoveryRequired {
                warning: Warning::MissingEoi
            })
        ));
    }
}

#[test]
fn ambiguous_adobe_transform_requires_an_explicit_hint_in_both_policies() {
    let mut bytes = j2k_test_support::rgb_app14_8x8_jpeg();
    let adobe = bytes.windows(5).position(|part| part == b"Adobe").unwrap();
    bytes[adobe + 11] = 7;
    for strict in [true, false] {
        let options = DecodeOptions::default().with_strict(strict);
        let view = JpegView::parse_with_options(&bytes, options).unwrap();
        assert!(matches!(
            Decoder::from_view(view),
            Err(JpegError::AmbiguousColorTransform { raw_transform: 7 })
        ));
        let view = JpegView::parse_with_options(
            &bytes,
            options.with_color_transform(j2k_jpeg::ColorTransform::ForceRgb),
        )
        .unwrap();
        let decoder = Decoder::from_view(view).unwrap();
        let (pixels, outcome) = decoder
            .decode_request(j2k_jpeg::DecodeRequest::full(PixelFormat::Rgb8))
            .unwrap();
        assert_eq!(pixels, j2k_test_support::rgb_app14_8x8_rgb());
        assert!(!outcome.recovered);
        assert!(outcome
            .warnings
            .iter()
            .any(|w| matches!(w, Warning::AdobeApp14Ambiguous { raw_transform: 7 })));
    }
}

#[test]
fn ambiguous_four_component_transform_is_not_resolved_by_rgb_hints() {
    let mut bytes = j2k_test_support::cmyk_8x8_jpeg();
    let adobe = bytes.windows(5).position(|part| part == b"Adobe").unwrap();
    bytes[adobe + 11] = 7;
    for strict in [true, false] {
        for transform in [
            j2k_jpeg::ColorTransform::Auto,
            j2k_jpeg::ColorTransform::ForceRgb,
            j2k_jpeg::ColorTransform::ForceYCbCr,
        ] {
            let options = DecodeOptions::default()
                .with_strict(strict)
                .with_color_transform(transform);
            let view = JpegView::parse_with_options(&bytes, options).unwrap();
            assert!(
                matches!(
                    Decoder::from_view(view),
                    Err(JpegError::AmbiguousColorTransform { raw_transform: 7 })
                ),
                "a three-component hint cannot resolve a four-component frame"
            );
        }
    }
}

/// The Adobe APP14 segment with transform byte `transform`.
fn adobe_app14(transform: u8) -> [u8; 16] {
    let mut segment = [
        0xff, 0xee, 0x00, 0x0e, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0,
    ];
    segment[15] = transform;
    segment
}

/// `bytes` with an Adobe APP14 segment inserted after SOI.
fn with_adobe_app14(bytes: &[u8], transform: u8) -> Vec<u8> {
    let mut out = bytes[..2].to_vec();
    out.extend_from_slice(&adobe_app14(transform));
    out.extend_from_slice(&bytes[2..]);
    out
}

/// An 8x8 baseline frame with two components and all-zero blocks.
fn two_component_8x8_jpeg() -> Vec<u8> {
    let mut bytes = vec![0xff, 0xd8, 0xff, 0xdb, 0x00, 67, 0x00];
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc0, 0x00, 14, 8, 0, 8, 0, 8, 2, 1, 0x11, 0, 2, 0x11, 0,
    ]);
    for class in [0x00, 0x10] {
        bytes.extend_from_slice(&[0xff, 0xc4, 0x00, 20, class, 1]);
        bytes.extend(std::iter::repeat_n(0u8, 16));
    }
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 10, 2, 1, 0x00, 2, 0x00, 0, 63, 0, 0x00]);
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn decode_with(bytes: &[u8], strict: bool, format: PixelFormat) -> Result<(Vec<u8>, bool), String> {
    let options = DecodeOptions::default().with_strict(strict);
    let view =
        JpegView::parse_with_options(bytes, options).map_err(|error| format!("{error:?}"))?;
    let decoder = Decoder::from_view(view).map_err(|error| format!("{error:?}"))?;
    decoder
        .decode_request(j2k_jpeg::DecodeRequest::full(format))
        .map(|(pixels, outcome)| (pixels, outcome.recovered))
        .map_err(|error| format!("{error:?}"))
}

#[test]
fn reserved_adobe_transform_without_colour_conversion_decodes_as_before() {
    // The transform byte only selects a colour conversion, so a reserved value
    // cannot change grayscale or two-component samples. These frames decode
    // exactly as they do without the byte, in both policies; only three- and
    // four-component frames reject the ambiguity.
    let grayscale = j2k_test_support::grayscale_8x8_jpeg();
    let two_component = two_component_8x8_jpeg();
    for (source, format) in [
        (&grayscale, PixelFormat::Gray8),
        (&two_component, PixelFormat::Gray8),
    ] {
        for strict in [true, false] {
            let expected = decode_with(source, strict, format);
            let defined = decode_with(&with_adobe_app14(source, 0), strict, format);
            let reserved = decode_with(&with_adobe_app14(source, 7), strict, format);
            assert_eq!(defined, expected, "strict {strict}");
            assert_eq!(reserved, expected, "strict {strict}");
            if let Ok((_, recovered)) = &reserved {
                assert!(!recovered);
            }
        }
    }
    let reserved = with_adobe_app14(&grayscale, 7);
    let (pixels, recovered) = decode_with(&reserved, true, PixelFormat::Gray8)
        .expect("grayscale frame with a reserved Adobe transform");
    assert!(!recovered);
    assert_eq!(pixels.len(), 64);
    let view = JpegView::parse_with_options(&reserved, DecodeOptions::default()).unwrap();
    let (_, outcome) = Decoder::from_view(view)
        .unwrap()
        .decode_request(j2k_jpeg::DecodeRequest::full(PixelFormat::Gray8))
        .unwrap();
    assert!(outcome
        .warnings
        .iter()
        .any(|warning| matches!(warning, Warning::AdobeApp14Ambiguous { raw_transform: 7 })));
}
