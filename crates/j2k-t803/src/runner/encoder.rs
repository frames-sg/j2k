// SPDX-License-Identifier: MIT OR Apache-2.0

mod evaluate;
mod input;

use std::{fs, path::Path};

#[cfg(all(feature = "metal-runner", target_os = "macos"))]
use j2k::encode_j2k_lossless_with_accelerator;
use j2k::{
    encode_j2k_lossless, encode_j2k_lossless_components, encode_j2k_lossless_typed_components,
    encode_j2k_lossless_with_roi_regions, encode_j2k_lossy, EncodeBackendPreference,
    J2kBlockCodingMode, J2kEncodeDispatchReport, J2kEncodeValidation, J2kLosslessComponentPlane,
    J2kLosslessComponentSamples, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kLosslessTypedComponentPlane, J2kLosslessTypedComponentSamples, J2kLossyEncodeOptions,
    J2kLossySamples, J2kMarkerSegment, J2kProgressionOrder, J2kQualityLayer, J2kRateTarget,
    J2kRoiRegion, ReversibleTransform,
};
#[cfg(any(
    feature = "cuda-runner",
    all(feature = "metal-runner", target_os = "macos")
))]
use j2k::{encode_j2k_lossy_with_accelerator, BackendKind};
#[cfg(feature = "cuda-runner")]
use j2k_cuda::{CudaEncodeStageAccelerator, CudaLosslessEncoder};
#[cfg(all(feature = "metal-runner", target_os = "macos"))]
use j2k_metal::MetalEncodeStageAccelerator;
use sha2::{Digest, Sha256};

use crate::encoder::{
    ics_path, matrix_path, reference_decoder_identity, EncoderCase, EncoderIcs, EncoderInputKind,
    EncoderRateTarget,
};
use crate::{
    EncoderEvidence, EncoderIut, EncoderMatrix, EncoderReferenceIdentity, ExecutionLocation,
};

use self::evaluate::{evaluate_case, generation_error};
use self::input::{generate_input, GeneratedInput};

struct EncoderSources {
    matrix: EncoderMatrix,
    ics: EncoderIcs,
    ics_path: &'static str,
    ics_sha256: String,
}

struct EncodedOutput {
    codestream: Vec<u8>,
    dispatch: J2kEncodeDispatchReport,
}

pub(super) fn run_cpu() -> Result<EncoderEvidence, String> {
    run(EncoderIut::Cpu, None, encode_cpu_case)
}

#[cfg(feature = "cuda-runner")]
pub(super) fn run_cuda() -> Result<EncoderEvidence, String> {
    let mut lossless_encoder = CudaLosslessEncoder::new();
    run(
        EncoderIut::Cuda,
        Some(ExecutionLocation::Cuda),
        move |case, input| encode_cuda_case(&mut lossless_encoder, case, input),
    )
}

#[cfg(all(feature = "metal-runner", target_os = "macos"))]
pub(super) fn run_metal() -> Result<EncoderEvidence, String> {
    run(
        EncoderIut::Metal,
        Some(ExecutionLocation::Metal),
        encode_metal_case,
    )
}

fn run(
    iut: EncoderIut,
    device: Option<ExecutionLocation>,
    mut encode: impl FnMut(&EncoderCase, &GeneratedInput) -> Result<EncodedOutput, String>,
) -> Result<EncoderEvidence, String> {
    let sources = load_sources(iut)?;
    let (standard, implementation, expected_version) = reference_decoder_identity();
    let actual_version = j2k_compare::openjpeg::version();
    if actual_version != expected_version {
        return Err(format!(
            "T.804 OpenJPEG version is {actual_version}, expected {expected_version}"
        ));
    }
    let mut reports = Vec::new();
    reports
        .try_reserve_exact(sources.ics.matrix_case_count())
        .map_err(|error| format!("allocate encoder case reports: {error}"))?;
    for case in sources.matrix.selected_cases(iut) {
        let input = match generate_input(case) {
            Ok(input) => input,
            Err(error) => {
                reports.push(generation_error(case, &error, device));
                continue;
            }
        };
        reports.push(evaluate_case(case, &input, encode(case, &input), device));
    }
    EncoderEvidence::new(
        sources.ics_path.to_string(),
        sources.ics_sha256,
        matrix_path().to_string(),
        sources.ics.matrix_case_count(),
        sources.ics.matrix_case_sha256().to_string(),
        EncoderReferenceIdentity {
            standard: standard.to_string(),
            implementation: implementation.to_string(),
            version: actual_version,
        },
        reports,
    )
    .map_err(|error| error.to_string())
}

fn load_sources(iut: EncoderIut) -> Result<EncoderSources, String> {
    let ics_path = ics_path(iut);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "resolve j2k workspace root".to_string())?;
    let matrix_text = fs::read_to_string(root.join(matrix_path()))
        .map_err(|error| format!("read {}: {error}", matrix_path()))?;
    let ics_bytes =
        fs::read(root.join(ics_path)).map_err(|error| format!("read {ics_path}: {error}"))?;
    let ics_text = std::str::from_utf8(&ics_bytes)
        .map_err(|error| format!("read {ics_path} as UTF-8: {error}"))?;
    let matrix = EncoderMatrix::parse(&matrix_text).map_err(|error| error.to_string())?;
    let ics = EncoderIcs::parse(ics_text).map_err(|error| error.to_string())?;
    if ics.iut != iut {
        return Err(format!("{ics_path} identifies the wrong encoder IUT"));
    }
    ics.validate_against(&matrix)
        .map_err(|error| error.to_string())?;
    Ok(EncoderSources {
        matrix,
        ics,
        ics_path,
        ics_sha256: format!("{:x}", Sha256::digest(ics_bytes)),
    })
}

fn encode_cpu_case(case: &EncoderCase, input: &GeneratedInput) -> Result<EncodedOutput, String> {
    match case.mode {
        crate::EncoderMode::Lossless => encode_cpu_lossless(case, input),
        crate::EncoderMode::Lossy => {
            let samples = J2kLossySamples::new(
                &input.interleaved,
                case.width,
                case.height,
                case.components,
                case.bit_depth,
                case.signed,
            )
            .map_err(|error| error.to_string())?;
            let encoded = encode_j2k_lossy(
                samples,
                &lossy_options(case, EncodeBackendPreference::CpuOnly),
            )
            .map_err(|error| error.to_string())?;
            Ok(EncodedOutput {
                codestream: encoded.codestream,
                dispatch: encoded.dispatch_report,
            })
        }
    }
}

#[cfg(feature = "cuda-runner")]
fn encode_cuda_case(
    lossless_encoder: &mut CudaLosslessEncoder,
    case: &EncoderCase,
    input: &GeneratedInput,
) -> Result<EncodedOutput, String> {
    match case.mode {
        crate::EncoderMode::Lossless => {
            let samples = interleaved_lossless_samples(case, input)?;
            let options = lossless_options(case, EncodeBackendPreference::Auto);
            let encoded = lossless_encoder
                .encode(samples, &options)
                .map_err(|error| error.to_string())?
                .into_encoded();
            Ok(EncodedOutput {
                codestream: encoded.codestream,
                dispatch: encoded.dispatch_report,
            })
        }
        crate::EncoderMode::Lossy => {
            let samples = interleaved_lossy_samples(case, input)?;
            let options = lossy_options(case, EncodeBackendPreference::Auto);
            let mut accelerator = CudaEncodeStageAccelerator::for_auto_host_output();
            let encoded = encode_j2k_lossy_with_accelerator(
                samples,
                &options,
                BackendKind::Cuda,
                &mut accelerator,
            )
            .map_err(|error| error.to_string())?;
            Ok(EncodedOutput {
                codestream: encoded.codestream,
                dispatch: encoded.dispatch_report,
            })
        }
    }
}

#[cfg(all(feature = "metal-runner", target_os = "macos"))]
fn encode_metal_case(case: &EncoderCase, input: &GeneratedInput) -> Result<EncodedOutput, String> {
    // This lane tests the Metal adapter IUT itself. The separately benchmarked
    // public Auto policy may legitimately keep small matrix cases on the CPU.
    let mut accelerator = MetalEncodeStageAccelerator::for_host_output_benchmark();
    match case.mode {
        crate::EncoderMode::Lossless => {
            let samples = interleaved_lossless_samples(case, input)?;
            let options = lossless_options(case, EncodeBackendPreference::Auto);
            let encoded = encode_j2k_lossless_with_accelerator(
                samples,
                &options,
                BackendKind::Metal,
                &mut accelerator,
            )
            .map_err(|error| error.to_string())?;
            Ok(EncodedOutput {
                codestream: encoded.codestream,
                dispatch: encoded.dispatch_report,
            })
        }
        crate::EncoderMode::Lossy => {
            let samples = interleaved_lossy_samples(case, input)?;
            let options = lossy_options(case, EncodeBackendPreference::Auto);
            let encoded = encode_j2k_lossy_with_accelerator(
                samples,
                &options,
                BackendKind::Metal,
                &mut accelerator,
            )
            .map_err(|error| error.to_string())?;
            Ok(EncodedOutput {
                codestream: encoded.codestream,
                dispatch: encoded.dispatch_report,
            })
        }
    }
}

#[cfg(any(
    feature = "cuda-runner",
    all(feature = "metal-runner", target_os = "macos")
))]
fn interleaved_lossless_samples<'a>(
    case: &EncoderCase,
    input: &'a GeneratedInput,
) -> Result<J2kLosslessSamples<'a>, String> {
    J2kLosslessSamples::new(
        &input.interleaved,
        case.width,
        case.height,
        case.components,
        case.bit_depth,
        case.signed,
    )
    .map_err(|error| error.to_string())
}

#[cfg(any(
    feature = "cuda-runner",
    all(feature = "metal-runner", target_os = "macos")
))]
fn interleaved_lossy_samples<'a>(
    case: &EncoderCase,
    input: &'a GeneratedInput,
) -> Result<J2kLossySamples<'a>, String> {
    J2kLossySamples::new(
        &input.interleaved,
        case.width,
        case.height,
        case.components,
        case.bit_depth,
        case.signed,
    )
    .map_err(|error| error.to_string())
}

fn encode_cpu_lossless(
    case: &EncoderCase,
    input: &GeneratedInput,
) -> Result<EncodedOutput, String> {
    let options = lossless_options(case, EncodeBackendPreference::CpuOnly);
    let encoded = match case.input {
        EncoderInputKind::Interleaved => {
            let samples = J2kLosslessSamples::new(
                &input.interleaved,
                case.width,
                case.height,
                case.components,
                case.bit_depth,
                case.signed,
            )
            .map_err(|error| error.to_string())?;
            if let Some(roi) = case.roi {
                encode_j2k_lossless_with_roi_regions(
                    samples,
                    &options,
                    &[J2kRoiRegion {
                        component: roi.component,
                        x: roi.x,
                        y: roi.y,
                        width: roi.width,
                        height: roi.height,
                        shift: roi.shift,
                    }],
                )
            } else {
                encode_j2k_lossless(samples, &options)
            }
        }
        EncoderInputKind::ComponentPlanes => {
            let planes = input
                .components
                .iter()
                .map(|component| J2kLosslessComponentPlane {
                    data: &component.data,
                    x_rsiz: component.sampling[0],
                    y_rsiz: component.sampling[1],
                })
                .collect::<Vec<_>>();
            let samples = J2kLosslessComponentSamples::new(
                &planes,
                case.width,
                case.height,
                case.bit_depth,
                case.signed,
            )
            .map_err(|error| error.to_string())?;
            encode_j2k_lossless_components(samples, &options)
        }
        EncoderInputKind::TypedComponentPlanes => {
            let planes = input
                .components
                .iter()
                .map(|component| J2kLosslessTypedComponentPlane {
                    data: &component.data,
                    x_rsiz: component.sampling[0],
                    y_rsiz: component.sampling[1],
                    bit_depth: component.bit_depth,
                    signed: component.signed,
                })
                .collect::<Vec<_>>();
            let samples = J2kLosslessTypedComponentSamples::new(&planes, case.width, case.height)
                .map_err(|error| error.to_string())?;
            encode_j2k_lossless_typed_components(samples, &options)
        }
    }
    .map_err(|error| error.to_string())?;
    Ok(EncodedOutput {
        codestream: encoded.codestream,
        dispatch: encoded.dispatch_report,
    })
}

fn lossless_options(
    case: &EncoderCase,
    backend: EncodeBackendPreference,
) -> J2kLosslessEncodeOptions {
    let mut options = J2kLosslessEncodeOptions::default();
    options.backend = backend;
    options.block_coding_mode = J2kBlockCodingMode::Classic;
    options.progression = progression(case.progression);
    options.max_decomposition_levels = Some(case.decomposition_levels);
    options.tile_size = case.tile_size.map(|[width, height]| (width, height));
    options.tile_part_packet_limit = case.tile_part_packet_limit;
    options.quality_layers = case.lossless_quality_layers;
    options.write_tlm = case.markers.contains(&crate::EncoderMarker::Tlm);
    options.write_plt = case.markers.contains(&crate::EncoderMarker::Plt);
    options.write_plm = case.markers.contains(&crate::EncoderMarker::Plm);
    options.write_ppm = case.markers.contains(&crate::EncoderMarker::Ppm);
    options.write_ppt = case.markers.contains(&crate::EncoderMarker::Ppt);
    options.write_sop = case.markers.contains(&crate::EncoderMarker::Sop);
    options.write_eph = case.markers.contains(&crate::EncoderMarker::Eph);
    options.reversible_transform =
        if case.input == EncoderInputKind::Interleaved && matches!(case.components, 3 | 4) {
            ReversibleTransform::Rct53
        } else {
            ReversibleTransform::None53
        };
    options.validation = J2kEncodeValidation::External;
    options
}

fn lossy_options(case: &EncoderCase, backend: EncodeBackendPreference) -> J2kLossyEncodeOptions {
    let mut options = J2kLossyEncodeOptions::default();
    options.backend = backend;
    options.block_coding_mode = J2kBlockCodingMode::Classic;
    options.progression = progression(case.progression);
    options.max_decomposition_levels = Some(case.decomposition_levels);
    options.rate_target = case.lossy_rate_target.map(rate_target);
    options.quality_layers = case
        .lossy_quality_layers
        .iter()
        .copied()
        .map(rate_target)
        .map(J2kQualityLayer::new)
        .collect();
    options.tile_size = case.tile_size.map(|[width, height]| (width, height));
    options.tile_part_packet_limit = case.tile_part_packet_limit;
    options.precinct_exponents = case
        .precinct_exponents
        .iter()
        .map(|[width, height]| (*width, *height))
        .collect();
    options.marker_segments = marker_segments(case);
    options.validation = J2kEncodeValidation::External;
    options
}

fn progression(value: crate::EncoderProgression) -> J2kProgressionOrder {
    match value {
        crate::EncoderProgression::Lrcp => J2kProgressionOrder::Lrcp,
        crate::EncoderProgression::Rlcp => J2kProgressionOrder::Rlcp,
        crate::EncoderProgression::Rpcl => J2kProgressionOrder::Rpcl,
        crate::EncoderProgression::Pcrl => J2kProgressionOrder::Pcrl,
        crate::EncoderProgression::Cprl => J2kProgressionOrder::Cprl,
    }
}

fn rate_target(value: EncoderRateTarget) -> J2kRateTarget {
    match value {
        EncoderRateTarget::BitsPerPixel(value) => J2kRateTarget::BitsPerPixel(value),
        EncoderRateTarget::Bytes(value) => J2kRateTarget::Bytes(value),
        EncoderRateTarget::PsnrDb(value) => J2kRateTarget::PsnrDb(value),
    }
}

fn marker_segments(case: &EncoderCase) -> Vec<J2kMarkerSegment> {
    case.markers
        .iter()
        .filter_map(|marker| match marker {
            crate::EncoderMarker::Tlm => Some(J2kMarkerSegment::Tlm),
            crate::EncoderMarker::Plm => Some(J2kMarkerSegment::Plm),
            crate::EncoderMarker::Plt => Some(J2kMarkerSegment::Plt),
            crate::EncoderMarker::Ppm => Some(J2kMarkerSegment::Ppm),
            crate::EncoderMarker::Ppt => Some(J2kMarkerSegment::Ppt),
            crate::EncoderMarker::Sop => Some(J2kMarkerSegment::Sop),
            crate::EncoderMarker::Eph => Some(J2kMarkerSegment::Eph),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{CaseStatus, EncoderIut, EncoderQualityStatus, ReportStatus};

    #[cfg(all(feature = "metal-runner", target_os = "macos"))]
    use super::run_metal;
    use super::{generate_input, load_sources, run_cpu};

    #[test]
    fn generated_typed_samples_preserve_declared_component_metadata() {
        let sources = load_sources(EncoderIut::Cpu).expect("load CPU matrix and ICS");
        let case = sources
            .matrix
            .cases
            .iter()
            .find(|case| case.id == "planar-typed")
            .expect("typed case");

        let input = generate_input(case).expect("generate typed input");

        assert_eq!(input.components.len(), 3);
        assert_eq!(input.components[0].bit_depth, 8);
        assert_eq!(input.components[1].bit_depth, 12);
        assert!(input.components[1].signed);
        assert_eq!(input.components[1].sampling, [2, 1]);
        assert_eq!(input.components[2].dimensions, [41, 18]);
    }

    #[test]
    fn complete_cpu_matrix_decodes_with_t804_openjpeg() {
        let evidence = run_cpu().expect("run CPU encoder evidence");
        let failures = evidence
            .cases
            .iter()
            .filter(|case| {
                case.status != CaseStatus::Pass || case.quality_status == EncoderQualityStatus::Fail
            })
            .collect::<Vec<_>>();

        assert_eq!(evidence.cases.len(), 28);
        assert_eq!(
            evidence.standards_status,
            ReportStatus::Pass,
            "{failures:#?}"
        );
        assert_eq!(evidence.quality_status, ReportStatus::Pass, "{failures:#?}");
        assert!(evidence
            .cases
            .iter()
            .all(|case| case.status == CaseStatus::Pass));
        assert!(evidence
            .cases
            .iter()
            .all(|case| { case.quality_status != EncoderQualityStatus::Fail }));
        let exact_lossy = evidence
            .cases
            .iter()
            .find(|case| case.id == "pairwise-09")
            .expect("exact lossy fixture");
        assert!(exact_lossy.psnr_infinite);
        assert_eq!(exact_lossy.psnr_db, None);
    }

    #[cfg(all(feature = "metal-runner", target_os = "macos"))]
    #[test]
    fn complete_metal_adapter_matrix_records_routes_and_decodes_with_openjpeg() {
        let evidence = run_metal().expect("run Metal encoder evidence");
        let failures = evidence
            .cases
            .iter()
            .filter(|case| {
                case.status != CaseStatus::Pass || case.quality_status == EncoderQualityStatus::Fail
            })
            .collect::<Vec<_>>();

        assert_eq!(evidence.cases.len(), 25);
        assert_eq!(evidence.status, ReportStatus::Pass, "{failures:#?}");
        assert!(evidence.cases.iter().any(|case| {
            matches!(
                case.route,
                crate::RouteKind::Hybrid | crate::RouteKind::DeviceNative
            )
        }));
    }
}
