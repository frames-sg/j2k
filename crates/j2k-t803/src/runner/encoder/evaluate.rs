// SPDX-License-Identifier: MIT OR Apache-2.0

mod quality;
mod route;
mod validation;

#[cfg(test)]
mod tests;

use std::borrow::Cow;

use j2k::{wrap_j2k_codestream, J2kEncodeDispatchReport, J2kFileWrapOptions};
use j2k_compare::openjpeg::{self, OpenJpegDecodedImage};

use crate::{CaseStatus, EncoderCaseReport, EncoderMode, EncoderQualityStatus, ExecutionLocation};

use super::reference::OpenHtj2kDecoder;
use super::{input::GeneratedInput, EncodedOutput};
use crate::encoder::{EncoderCase, EncoderPayload, EncoderReferenceDecoder};

use self::quality::{encoded_metrics, evaluate_quality, quality_requirement, EncodedMetrics};
use self::route::{route_evidence, EvaluatedRoute};
use self::validation::{validate_markers, validate_metadata};

pub(super) fn evaluate_case(
    case: &EncoderCase,
    input: &GeneratedInput,
    encoded: Result<EncodedOutput, String>,
    device: Option<ExecutionLocation>,
    openhtj2k: Option<&OpenHtj2kDecoder>,
) -> EncoderCaseReport {
    let encoded = match encoded {
        Ok(encoded) => encoded,
        Err(error) => {
            return error_report(case, error, J2kEncodeDispatchReport::default(), device);
        }
    };
    let dispatch = encoded.dispatch;
    let route = route_evidence(case, dispatch, device);
    let metrics = match encoded_metrics(case, encoded.codestream.len()) {
        Ok(metrics) => metrics,
        Err(error) => {
            return failed_report(case, route, None, false, None, error);
        }
    };

    if let Err(error) = validate_markers(case, &encoded.codestream) {
        return failed_report(case, route, Some(metrics), false, None, error);
    }

    let reference_input = match (encoded.reference_input, case.payload) {
        (Some(reference_input), _) => Cow::Owned(reference_input),
        (None, EncoderPayload::Codestream) => Cow::Borrowed(encoded.codestream.as_slice()),
        (None, EncoderPayload::Jph) => {
            match wrap_j2k_codestream(&encoded.codestream, J2kFileWrapOptions::jph()) {
                Ok(file) => Cow::Owned(file),
                Err(error) => {
                    return failed_report(
                        case,
                        route,
                        Some(metrics),
                        false,
                        None,
                        format!("wrap HTJ2K codestream as JPH: {error}"),
                    );
                }
            }
        }
        (None, EncoderPayload::Jp2) => {
            return failed_report(
                case,
                route,
                Some(metrics),
                false,
                None,
                "JP2 is not a supported encoder output payload".to_string(),
            );
        }
    };

    let decoded = match case.reference_decoder {
        EncoderReferenceDecoder::OpenJpeg => openjpeg::decode_components(&reference_input),
        EncoderReferenceDecoder::OpenHtj2k => openhtj2k
            .ok_or_else(|| "OpenHTJ2K decoder identity is unavailable".to_string())
            .and_then(|decoder| decoder.decode_components(&reference_input, case, input)),
    };
    let decoded = match decoded {
        Ok(decoded) => decoded,
        Err(error) => {
            return failed_report(
                case,
                route,
                Some(metrics),
                false,
                None,
                format!("{} reference decode failed: {error}", reference_name(case)),
            );
        }
    };
    if let Err(error) = validate_metadata(case, input, &decoded, reference_name(case)) {
        return failed_report(case, route, Some(metrics), true, None, error);
    }

    if case.mode == EncoderMode::Lossless {
        lossless_report(case, input, &decoded, route, metrics)
    } else {
        lossy_report(case, input, &decoded, route, metrics)
    }
}

fn lossless_report(
    case: &EncoderCase,
    input: &GeneratedInput,
    decoded: &OpenJpegDecodedImage,
    route: EvaluatedRoute,
    metrics: EncodedMetrics,
) -> EncoderCaseReport {
    let mismatch = input
        .components
        .iter()
        .zip(&decoded.components)
        .enumerate()
        .find_map(|(component, (expected, actual))| {
            expected
                .samples
                .iter()
                .zip(&actual.samples)
                .enumerate()
                .find(|(_, (expected, actual))| expected != actual)
                .map(|(sample, (&expected, &actual))| (component, sample, expected, actual))
        });
    if let Some((component, sample, expected, actual)) = mismatch {
        return failed_report(
            case,
            route,
            Some(metrics),
            true,
            Some(false),
            format!(
                "{} output first differs at component {component}, sample {sample}: expected {expected}, decoded {actual}",
                reference_name(case),
            ),
        );
    }
    let EvaluatedRoute {
        route,
        stages,
        dispatches,
    } = route;
    EncoderCaseReport {
        id: case.id.clone(),
        mode: case.mode,
        status: CaseStatus::Pass,
        route,
        reference_decoder: case.reference_decoder,
        reference_decode_success: true,
        lossless_exact: Some(true),
        encoded_bytes: Some(metrics.bytes),
        actual_bits_per_pixel: Some(metrics.bits_per_pixel),
        psnr_db: None,
        psnr_infinite: false,
        quality_status: EncoderQualityStatus::NotApplicable,
        quality_requirement: None,
        quality_error: None,
        error: None,
        stages,
        accelerator_dispatches: Some(dispatches),
    }
}

fn lossy_report(
    case: &EncoderCase,
    input: &GeneratedInput,
    decoded: &OpenJpegDecodedImage,
    route: EvaluatedRoute,
    metrics: EncodedMetrics,
) -> EncoderCaseReport {
    let psnr = quality::decoded_psnr(input, decoded);
    let quality = evaluate_quality(case, psnr, metrics);
    let EvaluatedRoute {
        route,
        stages,
        dispatches,
    } = route;
    EncoderCaseReport {
        id: case.id.clone(),
        mode: case.mode,
        status: CaseStatus::Pass,
        route,
        reference_decoder: case.reference_decoder,
        reference_decode_success: true,
        lossless_exact: None,
        encoded_bytes: Some(metrics.bytes),
        actual_bits_per_pixel: Some(metrics.bits_per_pixel),
        psnr_db: psnr.db,
        psnr_infinite: psnr.infinite,
        quality_status: quality.status,
        quality_requirement: Some(quality.requirement),
        quality_error: quality.error,
        error: None,
        stages,
        accelerator_dispatches: Some(dispatches),
    }
}

pub(super) fn generation_error(
    case: &EncoderCase,
    error: &str,
    device: Option<ExecutionLocation>,
) -> EncoderCaseReport {
    error_report(
        case,
        format!("generate deterministic encoder input: {error}"),
        J2kEncodeDispatchReport::default(),
        device,
    )
}

fn error_report(
    case: &EncoderCase,
    error: String,
    dispatch: J2kEncodeDispatchReport,
    device: Option<ExecutionLocation>,
) -> EncoderCaseReport {
    let EvaluatedRoute {
        route,
        stages,
        dispatches,
    } = route_evidence(case, dispatch, device);
    let (quality_status, quality_requirement, quality_error) = if case.mode == EncoderMode::Lossy {
        (
            EncoderQualityStatus::Fail,
            Some(quality_requirement(case)),
            Some("quality gate could not run because encoding failed".to_string()),
        )
    } else {
        (EncoderQualityStatus::NotApplicable, None, None)
    };
    EncoderCaseReport {
        id: case.id.clone(),
        mode: case.mode,
        status: CaseStatus::Error,
        route,
        reference_decoder: case.reference_decoder,
        reference_decode_success: false,
        lossless_exact: None,
        encoded_bytes: None,
        actual_bits_per_pixel: None,
        psnr_db: None,
        psnr_infinite: false,
        quality_status,
        quality_requirement,
        quality_error,
        error: Some(error),
        stages,
        accelerator_dispatches: Some(dispatches),
    }
}

fn failed_report(
    case: &EncoderCase,
    route: EvaluatedRoute,
    metrics: Option<EncodedMetrics>,
    reference_decode_success: bool,
    lossless_exact: Option<bool>,
    error: String,
) -> EncoderCaseReport {
    let (quality_status, quality_requirement, quality_error) = if case.mode == EncoderMode::Lossy {
        (
            EncoderQualityStatus::Fail,
            Some(quality_requirement(case)),
            Some("quality gate could not pass because the reference decode failed".to_string()),
        )
    } else {
        (EncoderQualityStatus::NotApplicable, None, None)
    };
    let EvaluatedRoute {
        route,
        stages,
        dispatches,
    } = route;
    EncoderCaseReport {
        id: case.id.clone(),
        mode: case.mode,
        status: CaseStatus::Fail,
        route,
        reference_decoder: case.reference_decoder,
        reference_decode_success,
        lossless_exact,
        encoded_bytes: metrics.map(|value| value.bytes),
        actual_bits_per_pixel: metrics.map(|value| value.bits_per_pixel),
        psnr_db: None,
        psnr_infinite: false,
        quality_status,
        quality_requirement,
        quality_error,
        error: Some(error),
        stages,
        accelerator_dispatches: Some(dispatches),
    }
}

const fn reference_name(case: &EncoderCase) -> &'static str {
    match case.reference_decoder {
        EncoderReferenceDecoder::OpenJpeg => "T.804 OpenJPEG",
        EncoderReferenceDecoder::OpenHtj2k => "OpenHTJ2K Part 15 interoperability decoder",
    }
}
