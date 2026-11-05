// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(feature = "metal-runner", target_os = "macos"))]
use j2k::encode_j2k_lossless_with_accelerator;
use j2k::{
    encode_j2k_lossless, encode_j2k_lossless_components, encode_j2k_lossless_typed_components,
    encode_j2k_lossless_with_roi_regions, encode_j2k_lossy, extract_j2k_codestream_payload,
    recode_j2k_to_htj2k_lossless, wrap_j2k_codestream, CompressedPayloadKind,
    EncodeBackendPreference, J2kEncodeDispatchReport, J2kEncodeValidation, J2kFileWrapOptions,
    J2kLosslessComponentPlane, J2kLosslessComponentSamples, J2kLosslessSamples,
    J2kLosslessTypedComponentPlane, J2kLosslessTypedComponentSamples, J2kLossySamples,
    J2kRoiRegion, J2kToHtj2kMode, J2kToHtj2kOptions,
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

use crate::encoder::{
    EncoderBlockCoding, EncoderCase, EncoderInputKind, EncoderOperation, EncoderPayload,
};
use crate::EncoderMode;

use super::input::GeneratedInput;
use super::options::{lossless_options, lossy_options, progression};
use super::EncodedOutput;

pub(super) fn encode_cpu_case(
    case: &EncoderCase,
    input: &GeneratedInput,
) -> Result<EncodedOutput, String> {
    if case.operation == EncoderOperation::Recode {
        return recode_cpu_case(case, input);
    }
    match case.mode {
        EncoderMode::Lossless => encode_cpu_lossless(case, input),
        EncoderMode::Lossy => {
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
                reference_input: None,
                dispatch: encoded.dispatch_report,
            })
        }
    }
}

#[cfg(feature = "cuda-runner")]
pub(super) fn encode_cuda_case(
    lossless_encoder: &mut CudaLosslessEncoder,
    case: &EncoderCase,
    input: &GeneratedInput,
) -> Result<EncodedOutput, String> {
    if case.operation != EncoderOperation::Encode {
        return Err("CUDA encoder adapter does not expose coefficient recoding".to_string());
    }
    match case.mode {
        EncoderMode::Lossless => {
            let samples = interleaved_lossless_samples(case, input)?;
            let options = lossless_options(case, EncodeBackendPreference::Auto);
            let encoded = lossless_encoder
                .encode(samples, &options)
                .map_err(|error| error.to_string())?
                .into_encoded();
            Ok(EncodedOutput {
                codestream: encoded.codestream,
                reference_input: None,
                dispatch: encoded.dispatch_report,
            })
        }
        EncoderMode::Lossy => {
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
                reference_input: None,
                dispatch: encoded.dispatch_report,
            })
        }
    }
}

#[cfg(all(feature = "metal-runner", target_os = "macos"))]
pub(super) fn encode_metal_case(
    case: &EncoderCase,
    input: &GeneratedInput,
) -> Result<EncodedOutput, String> {
    if case.operation != EncoderOperation::Encode {
        return Err("Metal encoder adapter does not expose coefficient recoding".to_string());
    }
    // This lane tests the Metal adapter IUT itself. The separately benchmarked
    // public Auto policy may legitimately keep small matrix cases on the CPU.
    let mut accelerator = MetalEncodeStageAccelerator::for_host_output_benchmark();
    match case.mode {
        EncoderMode::Lossless => {
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
                reference_input: None,
                dispatch: encoded.dispatch_report,
            })
        }
        EncoderMode::Lossy => {
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
                reference_input: None,
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
        reference_input: None,
        dispatch: encoded.dispatch_report,
    })
}

fn recode_cpu_case(case: &EncoderCase, input: &GeneratedInput) -> Result<EncodedOutput, String> {
    let mut source_case = case.clone();
    source_case.operation = EncoderOperation::Encode;
    source_case.block_coding = EncoderBlockCoding::Classic;
    source_case.payload = EncoderPayload::Codestream;
    source_case.source_payload = None;
    let source_codestream = encode_cpu_lossless(&source_case, input)?.codestream;
    let source = match case.source_payload {
        Some(EncoderPayload::Codestream) => source_codestream,
        Some(EncoderPayload::Jp2) => {
            wrap_j2k_codestream(&source_codestream, J2kFileWrapOptions::jp2())
                .map_err(|error| format!("wrap generated classic source as JP2: {error}"))?
        }
        Some(EncoderPayload::Jph) | None => {
            return Err("coefficient recode requires a raw J2K or JP2 source".to_string());
        }
    };
    let output_payload_kind = match case.payload {
        EncoderPayload::Codestream => CompressedPayloadKind::Jpeg2000Codestream,
        EncoderPayload::Jph => CompressedPayloadKind::JphFile,
        EncoderPayload::Jp2 => return Err("HTJ2K recode cannot emit JP2".to_string()),
    };
    let recoded = recode_j2k_to_htj2k_lossless(
        &source,
        J2kToHtj2kOptions::new(
            output_payload_kind,
            progression(case.progression),
            J2kEncodeValidation::External,
        ),
    )
    .map_err(|error| error.to_string())?;
    if recoded.report.mode != J2kToHtj2kMode::CoefficientPreserving {
        return Err(format!(
            "recode used {:?}, expected coefficient-preserving mode",
            recoded.report.mode
        ));
    }

    match case.payload {
        EncoderPayload::Codestream => Ok(EncodedOutput {
            codestream: recoded.bytes,
            reference_input: None,
            dispatch: J2kEncodeDispatchReport::default(),
        }),
        EncoderPayload::Jph => {
            let payload = extract_j2k_codestream_payload(&recoded.bytes)
                .map_err(|error| format!("extract recoded JPH codestream: {error}"))?
                .codestream();
            let mut codestream = Vec::new();
            codestream
                .try_reserve_exact(payload.len())
                .map_err(|_| "cannot allocate recoded JPH codestream".to_string())?;
            codestream.extend_from_slice(payload);
            Ok(EncodedOutput {
                codestream,
                reference_input: Some(recoded.bytes),
                dispatch: J2kEncodeDispatchReport::default(),
            })
        }
        EncoderPayload::Jp2 => Err("HTJ2K recode cannot emit JP2".to_string()),
    }
}
