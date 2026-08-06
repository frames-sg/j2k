// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Write as _;

use j2k::J2kEncodeDispatchReport;
use j2k_compare::openjpeg::{self, OpenJpegDecodedImage};

use crate::compare::u64_as_f64;
use crate::{
    CaseStatus, EncodeRouteStage, EncodeRouteStageName, EncoderCaseReport, EncoderMarker,
    EncoderMode, EncoderQualityStatus, ExecutionLocation, RouteKind,
};

use super::{input::GeneratedInput, EncodedOutput};
use crate::encoder::{EncoderCase, EncoderInputKind, EncoderRateTarget};

pub(super) fn evaluate_case(
    case: &EncoderCase,
    input: &GeneratedInput,
    encoded: Result<EncodedOutput, String>,
    device: Option<ExecutionLocation>,
) -> EncoderCaseReport {
    let encoded = match encoded {
        Ok(encoded) => encoded,
        Err(error) => {
            return error_report(case, error, J2kEncodeDispatchReport::default(), device);
        }
    };
    let dispatch = encoded.dispatch;
    let (route, stages) = route_evidence(case, dispatch, device);
    let metrics = match encoded_metrics(case, encoded.codestream.len()) {
        Ok(metrics) => metrics,
        Err(error) => {
            return failed_report(case, route, stages, None, false, None, error);
        }
    };

    if let Err(error) = validate_markers(case, &encoded.codestream) {
        return failed_report(case, route, stages, Some(metrics), false, None, error);
    }

    let decoded = match openjpeg::decode_components(&encoded.codestream) {
        Ok(decoded) => decoded,
        Err(error) => {
            return failed_report(
                case,
                route,
                stages,
                Some(metrics),
                false,
                None,
                format!("T.804 OpenJPEG reference decode failed: {error}"),
            );
        }
    };
    if let Err(error) = validate_metadata(case, input, &decoded) {
        return failed_report(case, route, stages, Some(metrics), true, None, error);
    }

    if case.mode == EncoderMode::Lossless {
        lossless_report(case, input, &decoded, route, stages, metrics)
    } else {
        lossy_report(case, input, &decoded, route, stages, metrics)
    }
}

fn lossless_report(
    case: &EncoderCase,
    input: &GeneratedInput,
    decoded: &OpenJpegDecodedImage,
    route: RouteKind,
    stages: Vec<EncodeRouteStage>,
    metrics: EncodedMetrics,
) -> EncoderCaseReport {
    let exact = input
        .components
        .iter()
        .zip(&decoded.components)
        .all(|(expected, actual)| expected.samples == actual.samples);
    if !exact {
        return failed_report(
            case,
            route,
            stages,
            Some(metrics),
            true,
            Some(false),
            "T.804 OpenJPEG output does not exactly match the lossless input".to_string(),
        );
    }
    EncoderCaseReport {
        id: case.id.clone(),
        mode: case.mode,
        status: CaseStatus::Pass,
        route,
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
    }
}

fn lossy_report(
    case: &EncoderCase,
    input: &GeneratedInput,
    decoded: &OpenJpegDecodedImage,
    route: RouteKind,
    stages: Vec<EncodeRouteStage>,
    metrics: EncodedMetrics,
) -> EncoderCaseReport {
    let psnr = decoded_psnr(input, decoded);
    let quality = evaluate_quality(case, psnr, metrics);
    EncoderCaseReport {
        id: case.id.clone(),
        mode: case.mode,
        status: CaseStatus::Pass,
        route,
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
    let (route, stages) = route_evidence(case, dispatch, device);
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
    }
}

#[derive(Clone, Copy)]
struct EncodedMetrics {
    bytes: u64,
    bits_per_pixel: f64,
}

fn encoded_metrics(case: &EncoderCase, encoded_bytes: usize) -> Result<EncodedMetrics, String> {
    let bytes = u64::try_from(encoded_bytes)
        .map_err(|_| "encoded codestream size exceeds the report range".to_string())?;
    let bytes_as_f64 = u64_as_f64(bytes).map_err(|error| error.to_string())?;
    let pixel_count = f64::from(case.width) * f64::from(case.height);
    Ok(EncodedMetrics {
        bytes,
        bits_per_pixel: bytes_as_f64 * 8.0 / pixel_count,
    })
}

fn failed_report(
    case: &EncoderCase,
    route: RouteKind,
    stages: Vec<EncodeRouteStage>,
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
    EncoderCaseReport {
        id: case.id.clone(),
        mode: case.mode,
        status: CaseStatus::Fail,
        route,
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
    }
}

fn validate_metadata(
    case: &EncoderCase,
    expected: &GeneratedInput,
    actual: &OpenJpegDecodedImage,
) -> Result<(), String> {
    if actual.dimensions != (case.width, case.height) {
        return Err(format!(
            "T.804 OpenJPEG dimensions are {:?}, expected {}x{}",
            actual.dimensions, case.width, case.height
        ));
    }
    if actual.components.len() != expected.components.len() {
        return Err(format!(
            "T.804 OpenJPEG returned {} components, expected {}",
            actual.components.len(),
            expected.components.len()
        ));
    }
    for (index, (expected, actual)) in expected
        .components
        .iter()
        .zip(&actual.components)
        .enumerate()
    {
        let expected_dimensions = (expected.dimensions[0], expected.dimensions[1]);
        let expected_sampling = (
            u32::from(expected.sampling[0]),
            u32::from(expected.sampling[1]),
        );
        if actual.dimensions != expected_dimensions
            || actual.sampling != expected_sampling
            || actual.bit_depth != expected.bit_depth
            || actual.signed != expected.signed
            || actual.samples.len() != expected.samples.len()
        {
            return Err(format!(
                "T.804 OpenJPEG component {index} metadata differs from the encoder input"
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Psnr {
    db: Option<f64>,
    infinite: bool,
}

fn decoded_psnr(expected: &GeneratedInput, actual: &OpenJpegDecodedImage) -> Psnr {
    let mut squared_error = 0.0;
    let mut samples = 0.0_f64;
    let mut peak = 0.0_f64;
    for (expected, actual) in expected.components.iter().zip(&actual.components) {
        peak = peak.max(2_f64.powi(i32::from(expected.bit_depth)) - 1.0);
        for (&expected, &actual) in expected.samples.iter().zip(&actual.samples) {
            let error = f64::from(expected) - f64::from(actual);
            squared_error += error * error;
            samples += 1.0;
        }
    }
    if squared_error == 0.0 {
        return Psnr {
            db: None,
            infinite: true,
        };
    }
    let mse = squared_error / samples;
    Psnr {
        db: Some(10.0 * (peak * peak / mse).log10()),
        infinite: false,
    }
}

struct QualityResult {
    status: EncoderQualityStatus,
    requirement: String,
    error: Option<String>,
}

fn evaluate_quality(case: &EncoderCase, psnr: Psnr, metrics: EncodedMetrics) -> QualityResult {
    let requirement = quality_requirement(case);
    let minimum_psnr = case
        .minimum_psnr_db
        .expect("validated lossy case has minimum PSNR");
    let mut failures = Vec::new();
    if psnr.db.is_some_and(|psnr_db| psnr_db < minimum_psnr) {
        let psnr_db = psnr.db.expect("finite PSNR was compared");
        failures.push(format!(
            "PSNR {psnr_db:.6} dB is below {minimum_psnr:.6} dB"
        ));
    }
    if let Some((target, overshoot)) = rate_gate(case) {
        match target {
            EncoderRateTarget::BitsPerPixel(target) => {
                let actual = metrics.bits_per_pixel;
                let one_byte = 8.0 / (f64::from(case.width) * f64::from(case.height));
                let maximum = target * (1.0 + overshoot / 100.0) + one_byte;
                if actual > maximum {
                    failures.push(format!("rate {actual:.6} bpp exceeds {maximum:.6} bpp"));
                }
            }
            EncoderRateTarget::Bytes(target) => {
                let encoded_bytes = metrics.bytes;
                match (u64_as_f64(target), u64_as_f64(encoded_bytes)) {
                    (Ok(target), Ok(actual)) => {
                        let maximum = target * (1.0 + overshoot / 100.0) + 1.0;
                        if actual > maximum {
                            failures.push(format!(
                                "codestream {encoded_bytes} bytes exceeds {maximum:.0} bytes"
                            ));
                        }
                    }
                    _ => failures.push("rate gate numeric conversion failed".to_string()),
                }
            }
            EncoderRateTarget::PsnrDb(_) => {}
        }
    }
    if failures.is_empty() {
        QualityResult {
            status: EncoderQualityStatus::Pass,
            requirement,
            error: None,
        }
    } else {
        QualityResult {
            status: EncoderQualityStatus::Fail,
            requirement,
            error: Some(failures.join("; ")),
        }
    }
}

fn quality_requirement(case: &EncoderCase) -> String {
    let minimum_psnr = case.minimum_psnr_db.unwrap_or_default();
    let mut requirement = format!("PSNR >= {minimum_psnr:.6} dB");
    if let Some((target, overshoot)) = rate_gate(case) {
        let rate = match target {
            EncoderRateTarget::BitsPerPixel(value) => format!("{value:.6} bpp"),
            EncoderRateTarget::Bytes(value) => format!("{value} bytes"),
            EncoderRateTarget::PsnrDb(_) => return requirement,
        };
        let _ = write!(
            requirement,
            "; rate <= {rate} + {overshoot:.6}% + one-byte rounding"
        );
    }
    requirement
}

fn rate_gate(case: &EncoderCase) -> Option<(EncoderRateTarget, f64)> {
    let target = case
        .lossy_quality_layers
        .last()
        .copied()
        .or(case.lossy_rate_target)?;
    let overshoot = case.maximum_rate_overshoot_percent?;
    Some((target, overshoot))
}

fn validate_markers(case: &EncoderCase, codestream: &[u8]) -> Result<(), String> {
    for marker in [
        EncoderMarker::Soc,
        EncoderMarker::Siz,
        EncoderMarker::Cod,
        EncoderMarker::Qcd,
        EncoderMarker::Sot,
        EncoderMarker::Sod,
        EncoderMarker::Eoc,
    ]
    .into_iter()
    .chain(case.markers.iter().copied())
    {
        if !contains_marker(codestream, marker) {
            return Err(format!(
                "encoded codestream is missing requested {marker:?} marker"
            ));
        }
    }
    Ok(())
}

fn contains_marker(codestream: &[u8], marker: EncoderMarker) -> bool {
    let code = match marker {
        EncoderMarker::Soc => 0x4F,
        EncoderMarker::Cap => 0x50,
        EncoderMarker::Prf => 0x56,
        EncoderMarker::Cpf => 0x59,
        EncoderMarker::Sot => 0x90,
        EncoderMarker::Sod => 0x93,
        EncoderMarker::Eoc => 0xD9,
        EncoderMarker::Siz => 0x51,
        EncoderMarker::Cod => 0x52,
        EncoderMarker::Coc => 0x53,
        EncoderMarker::Rgn => 0x5E,
        EncoderMarker::Qcd => 0x5C,
        EncoderMarker::Qcc => 0x5D,
        EncoderMarker::Poc => 0x5F,
        EncoderMarker::Tlm => 0x55,
        EncoderMarker::Plm => 0x57,
        EncoderMarker::Plt => 0x58,
        EncoderMarker::Ppm => 0x60,
        EncoderMarker::Ppt => 0x61,
        EncoderMarker::Sop => 0x91,
        EncoderMarker::Eph => 0x92,
        EncoderMarker::Crg => 0x63,
        EncoderMarker::Com => 0x64,
    };
    codestream.windows(2).any(|bytes| bytes == [0xFF, code])
}

fn route_evidence(
    case: &EncoderCase,
    dispatch: J2kEncodeDispatchReport,
    device: Option<ExecutionLocation>,
) -> (RouteKind, Vec<EncodeRouteStage>) {
    let device = device
        .filter(|location| matches!(location, ExecutionLocation::Cuda | ExecutionLocation::Metal));
    let location = |required: bool, count: usize| {
        if count > 0 {
            device.unwrap_or(ExecutionLocation::Cpu)
        } else if required {
            ExecutionLocation::Cpu
        } else {
            ExecutionLocation::NotUsed
        }
    };
    let interleaved_colour =
        case.input == EncoderInputKind::Interleaved && matches!(case.components, 3 | 4);
    let transfer_location = if dispatch.any() {
        device.unwrap_or(ExecutionLocation::Cpu)
    } else {
        ExecutionLocation::NotUsed
    };
    let stages = Vec::from([
        EncodeRouteStage {
            stage: EncodeRouteStageName::InputPreparation,
            location: location(true, dispatch.deinterleave),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::ForwardRct,
            location: location(
                case.mode == EncoderMode::Lossless && interleaved_colour,
                dispatch.forward_rct,
            ),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::ForwardIct,
            location: location(
                case.mode == EncoderMode::Lossy && interleaved_colour,
                dispatch.forward_ict,
            ),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::ForwardDwt53,
            location: location(
                case.mode == EncoderMode::Lossless && case.decomposition_levels > 0,
                dispatch.forward_dwt53,
            ),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::ForwardDwt97,
            location: location(
                case.mode == EncoderMode::Lossy && case.decomposition_levels > 0,
                dispatch.forward_dwt97,
            ),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::Quantization,
            location: location(true, dispatch.quantize_subband),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::Tier1,
            location: location(true, dispatch.tier1_code_block),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::Packetization,
            location: location(true, dispatch.packetization),
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::HostToDevice,
            location: transfer_location,
        },
        EncodeRouteStage {
            stage: EncodeRouteStageName::DeviceToHost,
            location: transfer_location,
        },
    ]);
    let uses_cpu = stages
        .iter()
        .any(|stage| stage.location == ExecutionLocation::Cpu);
    let uses_device = stages.iter().any(|stage| {
        matches!(
            stage.location,
            ExecutionLocation::Cuda | ExecutionLocation::Metal
        )
    });
    let route = match (uses_cpu, uses_device) {
        (true, true) => RouteKind::Hybrid,
        (false, true) => RouteKind::DeviceNative,
        _ => RouteKind::Cpu,
    };
    (route, stages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::EncoderMatrix;

    #[test]
    fn planar_input_does_not_claim_an_interleaved_colour_transform() {
        let matrix = EncoderMatrix::parse(include_str!(
            "../../../../../corpus/j2k-conformance/encoder-matrix-v1.toml"
        ))
        .expect("valid committed matrix");
        let case = matrix
            .cases
            .iter()
            .find(|case| case.id == "planar-sampled")
            .expect("planar matrix case");

        let (_, stages) = route_evidence(case, J2kEncodeDispatchReport::default(), None);
        let rct = stages
            .iter()
            .find(|stage| stage.stage == EncodeRouteStageName::ForwardRct)
            .expect("RCT disclosure");

        assert_eq!(rct.location, ExecutionLocation::NotUsed);
    }
}
