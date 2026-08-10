// SPDX-License-Identifier: MIT OR Apache-2.0

mod backend;
mod evaluate;
mod input;
mod options;
pub(super) mod reference;
#[cfg(test)]
mod tests;

use std::{fs, path::Path};

use j2k::J2kEncodeDispatchReport;
#[cfg(feature = "cuda-runner")]
use j2k_cuda::CudaLosslessEncoder;
use sha2::{Digest, Sha256};

use crate::encoder::{
    ics_path, matrix_path, reference_decoder_identity, EncoderCase, EncoderIcs,
    EncoderReferenceDecoder,
};
use crate::{
    EncoderEvidence, EncoderIut, EncoderMatrix, EncoderReferenceIdentity, ExecutionLocation,
};

use self::backend::encode_cpu_case;
#[cfg(feature = "cuda-runner")]
use self::backend::encode_cuda_case;
#[cfg(all(feature = "metal-runner", target_os = "macos"))]
use self::backend::encode_metal_case;
use self::evaluate::{evaluate_case, generation_error};
use self::input::{generate_input, GeneratedInput};
use self::reference::OpenHtj2kDecoder;

struct EncoderSources {
    matrix: EncoderMatrix,
    ics: EncoderIcs,
    ics_path: &'static str,
    ics_sha256: String,
}

struct EncodedOutput {
    codestream: Vec<u8>,
    reference_input: Option<Vec<u8>>,
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
    let openhtj2k = if sources
        .matrix
        .selected_cases(iut)
        .any(|case| case.reference_decoder == EncoderReferenceDecoder::OpenHtj2k)
    {
        Some(OpenHtj2kDecoder::from_environment()?)
    } else {
        None
    };
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
        reports.push(evaluate_case(
            case,
            &input,
            encode(case, &input),
            device,
            openhtj2k.as_ref(),
        ));
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
        openhtj2k
            .as_ref()
            .map(OpenHtj2kDecoder::identity)
            .into_iter()
            .collect(),
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
