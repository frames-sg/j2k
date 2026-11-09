// SPDX-License-Identifier: MIT OR Apache-2.0

//! A reserved Adobe APP14 transform byte under `ColorTransform::Auto`.
//!
//! Strict decode refuses it. Lenient decode reads the samples as
//! libjpeg-turbo's `default_decompress_parms` does, YCbCr for three
//! components and YCCK for four, keeps the warning and reports recovery.
//!
//! The references are the committed libjpeg-turbo matrix outputs.
//! libjpeg-turbo reads a reserved three-component frame as it reads a JFIF
//! one, and a reserved four-component frame as it reads transform 2, so a
//! matrix case keeps its reference when its marker is rewritten, except for
//! CMYK sources, whose samples libjpeg-turbo then reads as YCCK.
//! `libjpeg_turbo_compare` checks the same rewritten files against a live
//! libjpeg-turbo.

use j2k_jpeg::transcode::{extract_dct_blocks, DctExtractOptions};
use j2k_jpeg::{
    decode_tiles_into_with_options, ColorSpace, ColorTransform, DecodeOptions, DecodeOutcome,
    DecodeRequest, Decoder, Downscale, JpegBatchDecodeOptions, JpegBatchDecoder, JpegBatchInput,
    JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp, JpegError, JpegView, PixelFormat,
    Rect, TileBatchOptions, TileDecodeJob, Warning,
};
use j2k_test_support::{
    scaled_matrix_cases, scaled_rect_covering, with_adobe_transform, PixelRect, ScaledMatrixCase,
    RESERVED_ADOBE_TRANSFORMS, SCALED_MATRIX_DENOMINATORS,
};

fn lenient() -> DecodeOptions {
    DecodeOptions::default().with_strict(false)
}

fn decoder(jpeg: &[u8], options: DecodeOptions) -> Result<Decoder<'_>, JpegError> {
    Decoder::from_view(JpegView::parse_with_options(jpeg, options)?)
}

fn full_decode(
    jpeg: &[u8],
    options: DecodeOptions,
    fmt: PixelFormat,
) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
    decoder(jpeg, options)?.decode_request(DecodeRequest::full(fmt))
}

fn four_components(case: &ScaledMatrixCase) -> bool {
    case.layout.starts_with("cmyk") || case.layout.starts_with("ycck")
}

fn color_cases() -> Vec<ScaledMatrixCase> {
    scaled_matrix_cases()
        .into_iter()
        .filter(|case| !case.is_gray())
        .collect()
}

fn downscale(denominator: u32) -> Downscale {
    match denominator {
        1 => Downscale::None,
        2 => Downscale::Half,
        4 => Downscale::Quarter,
        8 => Downscale::Eighth,
        _ => unreachable!("matrix denominators are 1, 2, 4, 8"),
    }
}

fn format(case: &ScaledMatrixCase) -> PixelFormat {
    match (case.is_gray(), case.precision > 8) {
        (true, false) => PixelFormat::Gray8,
        (true, true) => PixelFormat::Gray16,
        (false, false) => PixelFormat::Rgb8,
        (false, true) => PixelFormat::Rgb16,
    }
}

/// Decoder output as sample values (16-bit output is little-endian).
fn samples(case: &ScaledMatrixCase, bytes: &[u8]) -> Vec<u16> {
    if case.precision > 8 {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect()
    } else {
        bytes.iter().map(|&sample| u16::from(sample)).collect()
    }
}

/// The samples libjpeg-turbo produces for `case` rewritten with a reserved
/// transform, at `1/denominator`.
fn turbo_reading(case: &ScaledMatrixCase, denominator: u32) -> Vec<u16> {
    if case.layout.starts_with("cmyk") {
        // CMYK samples read as YCCK: j2k's transform-2 decode of the same
        // bytes, which `scaled_matrix` and `libjpeg_turbo_compare` pin.
        let ycck = with_adobe_transform(case.jpeg, 2);
        let (bytes, _) = decoder(&ycck, DecodeOptions::default())
            .and_then(|decoder| {
                decoder.decode_request(DecodeRequest::scaled(format(case), downscale(denominator)))
            })
            .unwrap_or_else(|err| panic!("{} as YCCK: {err}", case.name));
        return samples(case, &bytes);
    }
    if case.precision > 8 {
        case.reference_u16(denominator)
    } else {
        case.reference(denominator)
            .iter()
            .map(|&sample| u16::from(sample))
            .collect()
    }
}

fn crop(samples: &[u16], width: u32, rect: PixelRect) -> Vec<u16> {
    let mut out = Vec::new();
    for y in rect.y..rect.y + rect.h {
        let start = (y * width + rect.x) as usize * 3;
        out.extend_from_slice(&samples[start..start + rect.w as usize * 3]);
    }
    out
}

fn check_lenient_outcome(label: &str, outcome: &DecodeOutcome, transform: u8) -> Option<String> {
    let warned = outcome.warnings.contains(&Warning::AdobeApp14Ambiguous {
        raw_transform: transform,
    });
    (!outcome.recovered || !warned).then(|| {
        format!(
            "{label}: recovered {}, warnings {:?}",
            outcome.recovered, outcome.warnings
        )
    })
}

fn assert_no_failures(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} reserved-transform decodes differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn lenient_decode_reads_reserved_transforms_as_libjpeg_turbo_does() {
    let mut failures = Vec::new();
    let mut compared = 0;
    for case in color_cases() {
        let expected_space = if four_components(&case) {
            ColorSpace::Ycck
        } else {
            ColorSpace::YCbCr
        };
        let references: Vec<Vec<u16>> = SCALED_MATRIX_DENOMINATORS
            .iter()
            .map(|&denominator| turbo_reading(&case, denominator))
            .collect();
        let roi = Rect {
            x: case.width / 3,
            y: case.height / 4,
            w: (case.width / 2).max(1),
            h: (case.height / 2).max(1),
        };
        for transform in RESERVED_ADOBE_TRANSFORMS {
            let jpeg = with_adobe_transform(case.jpeg, transform);
            let decoder = match decoder(&jpeg, lenient()) {
                Ok(decoder) => decoder,
                Err(err) => {
                    failures.push(format!("{} transform {transform}: {err}", case.name));
                    continue;
                }
            };
            if decoder.info().color_space != expected_space {
                failures.push(format!(
                    "{} transform {transform}: color space {:?}",
                    case.name,
                    decoder.info().color_space
                ));
            }
            for (&denominator, reference) in SCALED_MATRIX_DENOMINATORS.iter().zip(&references) {
                let label = format!("{} transform {transform} 1/{denominator}", case.name);
                let scale = downscale(denominator);
                let scaled_roi = scaled_rect_covering(
                    PixelRect {
                        x: roi.x,
                        y: roi.y,
                        w: roi.w,
                        h: roi.h,
                    },
                    denominator,
                );
                let scaled_width = case.scaled_dimensions(denominator).0;
                for (request, expected) in [
                    (
                        DecodeRequest::scaled(format(&case), scale),
                        reference.clone(),
                    ),
                    (
                        DecodeRequest::region_scaled(format(&case), roi, scale),
                        crop(reference, scaled_width, scaled_roi),
                    ),
                ] {
                    match decoder.decode_request(request) {
                        Ok((bytes, outcome)) => {
                            if samples(&case, &bytes) != expected {
                                failures.push(format!("{label} {request:?}: pixels differ"));
                            }
                            failures.extend(check_lenient_outcome(&label, &outcome, transform));
                            compared += 1;
                        }
                        Err(err) => failures.push(format!("{label} {request:?}: {err}")),
                    }
                }
            }
        }
    }
    assert_no_failures(&failures);
    assert!(compared > 5000, "compared {compared}");
}

#[test]
fn strict_decode_refuses_reserved_transforms_and_names_the_remedy() {
    for case in color_cases() {
        for transform in RESERVED_ADOBE_TRANSFORMS {
            let jpeg = with_adobe_transform(case.jpeg, transform);
            assert!(
                matches!(
                    decoder(&jpeg, DecodeOptions::default()),
                    Err(JpegError::AmbiguousColorTransform { raw_transform }) if raw_transform == transform
                ),
                "{} transform {transform}",
                case.name
            );
        }
    }
    let message = JpegError::AmbiguousColorTransform { raw_transform: 7 }.to_string();
    for remedy in [
        "transform 7",
        "strict(false)",
        "libjpeg-turbo",
        "YCbCr for three components",
        "YCCK for four",
        "ColorTransform::ForceRgb",
        "ColorTransform::ForceYCbCr",
    ] {
        assert!(message.contains(remedy), "{message:?} lacks {remedy:?}");
    }
}

#[test]
fn three_component_hints_resolve_and_four_component_frames_ignore_them() {
    for case in color_cases() {
        let fmt = format(&case);
        let jpeg = with_adobe_transform(case.jpeg, 7);
        let (automatic, _) = full_decode(&jpeg, lenient(), fmt).expect("lenient decode");
        for hint in [ColorTransform::ForceRgb, ColorTransform::ForceYCbCr] {
            for strict in [true, false] {
                let options = DecodeOptions::default()
                    .with_strict(strict)
                    .with_color_transform(hint);
                let decoded = full_decode(&jpeg, options, fmt);
                let label = format!("{} {hint:?} strict {strict}", case.name);
                if four_components(&case) {
                    // The hint is ignored: Auto's policy applies.
                    if strict {
                        assert!(
                            matches!(decoded, Err(JpegError::AmbiguousColorTransform { .. })),
                            "{label}"
                        );
                    } else {
                        let (pixels, outcome) = decoded.expect(&label);
                        assert_eq!(pixels, automatic, "{label}");
                        assert!(outcome.recovered, "{label}");
                    }
                    continue;
                }
                let (pixels, outcome) = decoded.expect(&label);
                assert!(!outcome.recovered, "{label}");
                assert!(
                    outcome
                        .warnings
                        .contains(&Warning::AdobeApp14Ambiguous { raw_transform: 7 }),
                    "{label}"
                );
                // ForceYCbCr reads the samples as lenient decode does; ForceRgb
                // as Adobe transform 0 declares them.
                let expected = if hint == ColorTransform::ForceYCbCr {
                    automatic.clone()
                } else {
                    full_decode(
                        &with_adobe_transform(case.jpeg, 0),
                        DecodeOptions::default(),
                        fmt,
                    )
                    .expect("RGB decode")
                    .0
                };
                assert_eq!(pixels, expected, "{label}");
            }
        }
    }
}

#[test]
fn defined_transforms_and_grayscale_frames_decode_as_before() {
    for case in scaled_matrix_cases() {
        let transforms: &[u8] = if case.is_gray() { &[7] } else { &[0, 1, 2] };
        for &transform in transforms {
            let jpeg = with_adobe_transform(case.jpeg, transform);
            let label = format!("{} transform {transform}", case.name);
            let strict = full_decode(&jpeg, DecodeOptions::default(), format(&case))
                .unwrap_or_else(|err| panic!("{label}: {err}"));
            let relaxed = full_decode(&jpeg, lenient(), format(&case))
                .unwrap_or_else(|err| panic!("{label} lenient: {err}"));
            assert_eq!(strict, relaxed, "{label}");
            assert!(!strict.1.recovered, "{label}");
            if case.is_gray() {
                assert_eq!(
                    samples(&case, &strict.0),
                    turbo_reading(&case, 1),
                    "{label}"
                );
            }
        }
    }
}

#[test]
fn tile_and_tensor_batches_apply_the_same_policy() {
    let cases: Vec<ScaledMatrixCase> = color_cases()
        .into_iter()
        .filter(|case| case.precision == 8 && case.width == 45)
        .collect();
    let inputs: Vec<Vec<u8>> = cases
        .iter()
        .map(|case| with_adobe_transform(case.jpeg, 7))
        .collect();
    let mut outputs: Vec<Vec<u8>> = cases
        .iter()
        .map(|case| vec![0; case.reference(1).len()])
        .collect();
    let mut jobs: Vec<TileDecodeJob<'_, '_>> = inputs
        .iter()
        .zip(outputs.iter_mut())
        .map(|(input, out)| TileDecodeJob {
            input: input.as_slice(),
            out: out.as_mut_slice(),
            stride: 45 * 3,
        })
        .collect();
    let refused = decode_tiles_into_with_options(
        &mut jobs,
        PixelFormat::Rgb8,
        DecodeOptions::default(),
        TileBatchOptions::default(),
    )
    .expect_err("strict tile batch");
    assert!(matches!(
        refused.tile_error().map(|error| &error.source),
        Some(JpegError::AmbiguousColorTransform { raw_transform: 7 })
    ));
    let outcomes = decode_tiles_into_with_options(
        &mut jobs,
        PixelFormat::Rgb8,
        lenient(),
        TileBatchOptions::default(),
    )
    .expect("lenient tile batch");
    drop(jobs);
    for ((case, out), outcome) in cases.iter().zip(&outputs).zip(&outcomes) {
        assert_eq!(samples(case, out), turbo_reading(case, 1), "{}", case.name);
        assert!(
            check_lenient_outcome(case.name, outcome, 7).is_none(),
            "{}",
            case.name
        );
    }

    let mut batch = JpegBatchDecoder::new(JpegBatchDecodeOptions::default());
    for strict in [true, false] {
        let batch_inputs: Vec<JpegBatchInput<'_>> = inputs
            .iter()
            .map(|input| {
                JpegBatchInput::new(input)
                    .with_options(DecodeOptions::default().with_strict(strict))
            })
            .collect();
        let result = batch.decode(&batch_inputs).expect("tensor batch");
        if strict {
            assert!(result.groups().is_empty());
            assert_eq!(result.errors().len(), inputs.len());
            assert!(result.errors().iter().all(|error| matches!(
                error.source,
                JpegError::AmbiguousColorTransform { raw_transform: 7 }
            )));
        } else {
            assert!(result.errors().is_empty());
            let rows: usize = result
                .groups()
                .iter()
                .map(|group| {
                    assert!(group.recovered().iter().all(|&recovered| recovered));
                    group.recovered().len()
                })
                .sum();
            assert_eq!(rows, inputs.len());
        }
    }
}

/// GPU routing reads the decoder's color space, which is the lenient
/// reading, so a reserved frame routes exactly as the defined transform with
/// the same reading: transform 1 (YCbCr) for three components and transform
/// 2 (YCCK) for four. The owned CUDA path and the Metal fast path take only
/// YCbCr, RGB or grayscale, so a four-component frame goes to the CPU or the
/// Metal hybrid decoder: never a third reading. `inspect` parses strictly,
/// so it refuses the reserved byte, as it did before lenient reading.
#[test]
fn gpu_routing_sees_the_lenient_reading() {
    let request = JpegCapabilityRequest {
        op: JpegDecodeOp::Full,
        fmt: PixelFormat::Rgb8,
    };
    for case in color_cases()
        .into_iter()
        .filter(|case| case.precision == 8 && case.coding == "baseline")
    {
        let jpeg = with_adobe_transform(case.jpeg, 7);
        assert!(
            matches!(
                JpegCapabilityReport::inspect(&jpeg, request),
                Err(JpegError::AmbiguousColorTransform { raw_transform: 7 })
            ),
            "{}",
            case.name
        );
        let relaxed = decoder(&jpeg, lenient()).expect("lenient decoder");
        let report = JpegCapabilityReport::for_decoder(&relaxed, request);
        let defined = with_adobe_transform(case.jpeg, if four_components(&case) { 2 } else { 1 });
        let reference = JpegCapabilityReport::inspect(&defined, request).expect("defined report");
        assert_eq!(
            report.info.color_space, reference.info.color_space,
            "{}",
            case.name
        );
        assert_eq!(report.cpu, reference.cpu, "{}", case.name);
        assert_eq!(report.owned_cuda, reference.owned_cuda, "{}", case.name);
        assert_eq!(report.metal_fast, reference.metal_fast, "{}", case.name);
        if four_components(&case) {
            assert_eq!(report.info.color_space, ColorSpace::Ycck, "{}", case.name);
            assert!(!report.owned_cuda.eligible, "{}", case.name);
            assert!(!report.metal_fast.eligible, "{}", case.name);
        } else {
            assert_eq!(report.info.color_space, ColorSpace::YCbCr, "{}", case.name);
        }
    }
}

#[test]
fn coefficient_extraction_stays_strict() {
    for case in color_cases()
        .into_iter()
        .filter(|case| case.precision == 8 && case.coding == "baseline")
    {
        let jpeg = with_adobe_transform(case.jpeg, 7);
        assert!(
            matches!(
                extract_dct_blocks(&jpeg, DctExtractOptions::default()),
                Err(JpegError::AmbiguousColorTransform { raw_transform: 7 })
            ),
            "{}",
            case.name
        );
    }
}

#[test]
fn lenient_progressive_eof_and_reserved_transform_both_report_recovery() {
    for case in color_cases()
        .into_iter()
        .filter(|case| case.coding == "progressive" && case.width == 45)
    {
        let jpeg = with_adobe_transform(case.jpeg, 7);
        let last_scan = jpeg
            .windows(2)
            .rposition(|pair| pair == [0xFF, 0xDA])
            .expect("progressive scans");
        let partial = &jpeg[..last_scan];
        assert!(
            matches!(
                decoder(partial, DecodeOptions::default()),
                Err(JpegError::AmbiguousColorTransform { raw_transform: 7 })
            ),
            "{}",
            case.name
        );
        let (_, outcome) = full_decode(partial, lenient(), format(&case))
            .unwrap_or_else(|err| panic!("{}: {err}", case.name));
        assert!(outcome.recovered, "{}", case.name);
        assert!(
            outcome.warnings.contains(&Warning::MissingEoi),
            "{}",
            case.name
        );
        assert!(
            outcome
                .warnings
                .contains(&Warning::AdobeApp14Ambiguous { raw_transform: 7 }),
            "{}",
            case.name
        );
    }
}
