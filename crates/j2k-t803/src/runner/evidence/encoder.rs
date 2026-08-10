// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use sha2::{Digest, Sha256};

use crate::encoder::{ics_path, matrix_path, reference_decoder_identity};
use crate::{
    CaseStatus, EncoderEvidence, EncoderIcs, EncoderIut, EncoderMatrix, EncoderQualityStatus,
    ReportStatus,
};

use crate::runner::encoder::reference::verify_report_identity;

pub(super) fn verify_encoder_evidence(
    iut_name: &str,
    evidence: &EncoderEvidence,
) -> Result<(), String> {
    let iut = match iut_name {
        "j2k" => EncoderIut::Cpu,
        "j2k-cuda" => EncoderIut::Cuda,
        "j2k-metal" => EncoderIut::Metal,
        other => return Err(format!("unknown encoder IUT {other:?}")),
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "resolve j2k workspace root".to_string())?;
    let expected_matrix_path = matrix_path();
    let expected_ics_path = ics_path(iut);
    let matrix_text = fs::read_to_string(root.join(expected_matrix_path))
        .map_err(|error| format!("read {expected_matrix_path}: {error}"))?;
    let matrix = EncoderMatrix::parse(&matrix_text).map_err(|error| error.to_string())?;
    let ics_bytes = fs::read(root.join(expected_ics_path))
        .map_err(|error| format!("read {expected_ics_path}: {error}"))?;
    let ics_text = std::str::from_utf8(&ics_bytes)
        .map_err(|error| format!("read {expected_ics_path} as UTF-8: {error}"))?;
    let ics = EncoderIcs::parse(ics_text).map_err(|error| error.to_string())?;
    if ics.iut != iut {
        return Err(format!(
            "{expected_ics_path} identifies the wrong encoder IUT"
        ));
    }
    ics.validate_against(&matrix)
        .map_err(|error| error.to_string())?;

    let actual_ics_sha256 = format!("{:x}", Sha256::digest(&ics_bytes));
    if evidence.ics_path != expected_ics_path || evidence.ics_sha256 != actual_ics_sha256 {
        return Err(format!(
            "{iut_name} encoder ICS SHA-256 or path differs from {expected_ics_path}"
        ));
    }
    if evidence.matrix_path != expected_matrix_path
        || evidence.matrix_case_count != ics.matrix_case_count()
        || evidence.matrix_case_sha256 != ics.matrix_case_sha256()
    {
        return Err(format!(
            "{iut_name} encoder matrix identity differs from the committed ICS"
        ));
    }
    let (standard, implementation, version) = reference_decoder_identity();
    if evidence.reference_decoder.standard != standard
        || evidence.reference_decoder.implementation != implementation
        || evidence.reference_decoder.version != version
    {
        return Err(format!(
            "{iut_name} encoder reference decoder is not the pinned T.804 OpenJPEG build"
        ));
    }
    let selects_openhtj2k = matrix
        .selected_cases(iut)
        .any(|case| case.reference_decoder == crate::EncoderReferenceDecoder::OpenHtj2k);
    verify_report_identity(
        iut_name,
        selects_openhtj2k,
        &evidence.supplemental_reference_decoders,
    )?;
    if evidence.standards_status != ReportStatus::Pass
        || evidence.quality_status != ReportStatus::Pass
        || evidence.status != ReportStatus::Pass
    {
        return Err(format!("{iut_name} encoder evidence did not pass"));
    }

    let expected_cases = matrix.selected_cases(iut).collect::<Vec<_>>();
    if evidence.cases.len() != expected_cases.len() {
        return Err(format!(
            "{iut_name} encoder evidence contains {} cases, expected {}",
            evidence.cases.len(),
            expected_cases.len()
        ));
    }
    for (observed, expected) in evidence.cases.iter().zip(expected_cases) {
        if observed.id != expected.id
            || observed.mode != expected.mode
            || observed.reference_decoder != expected.reference_decoder
            || observed.status != CaseStatus::Pass
            || observed.quality_status == EncoderQualityStatus::Fail
        {
            return Err(format!(
                "{iut_name} encoder case {} differs from the committed encoder matrix",
                observed.id
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::encoder::reference::{
        OPENHTJ2K_IMPLEMENTATION, OPENHTJ2K_SCOPE, OPENHTJ2K_SOURCE_COMMIT, OPENHTJ2K_SOURCE_URL,
        OPENHTJ2K_VERSION,
    };
    use crate::{
        EncodeRouteStage, EncodeRouteStageName, EncoderCaseReport, EncoderDispatchEvidence,
        EncoderMode, EncoderReferenceDecoder, EncoderReferenceIdentity,
        EncoderSupplementalReferenceIdentity, ExecutionLocation, RouteKind,
    };

    #[test]
    fn verification_rejects_ics_case_or_decoder_tampering() {
        let mut evidence = committed_cpu_evidence();
        verify_encoder_evidence("j2k", &evidence).expect("committed CPU encoder evidence");

        evidence.ics_sha256 = "0".repeat(64);
        let error = verify_encoder_evidence("j2k", &evidence)
            .expect_err("the report must pin the exact committed ICS bytes");
        assert!(error.contains("ICS SHA-256"), "{error}");

        let mut evidence = committed_cpu_evidence();
        evidence.cases[0].mode = match evidence.cases[0].mode {
            EncoderMode::Lossless => EncoderMode::Lossy,
            EncoderMode::Lossy => EncoderMode::Lossless,
        };
        let error = verify_encoder_evidence("j2k", &evidence)
            .expect_err("the report must retain the selected matrix case modes");
        assert!(error.contains("encoder matrix"), "{error}");

        let mut evidence = committed_cpu_evidence();
        evidence.supplemental_reference_decoders[0].source_commit = "a".repeat(40);
        let error = verify_encoder_evidence("j2k", &evidence)
            .expect_err("the report must retain the pinned supplemental decoder source");
        assert!(error.contains("pinned OpenHTJ2K"), "{error}");
    }

    fn committed_cpu_evidence() -> EncoderEvidence {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let matrix_path = "corpus/j2k-conformance/encoder-matrix-v2.toml";
        let ics_path = "corpus/j2k-conformance/encoder-ics-cpu.toml";
        let matrix = EncoderMatrix::parse(
            &fs::read_to_string(root.join(matrix_path)).expect("read encoder matrix"),
        )
        .expect("valid encoder matrix");
        let ics_bytes = fs::read(root.join(ics_path)).expect("read CPU ICS");
        let ics = EncoderIcs::parse(std::str::from_utf8(&ics_bytes).expect("UTF-8 ICS"))
            .expect("valid CPU ICS");
        ics.validate_against(&matrix).expect("ICS matches matrix");
        let cases = matrix
            .selected_cases(EncoderIut::Cpu)
            .map(|case| passing_case(&case.id, case.mode, case.reference_decoder))
            .collect();
        let (standard, implementation, version) = crate::encoder::reference_decoder_identity();
        EncoderEvidence::new(
            ics_path.to_string(),
            format!("{:x}", Sha256::digest(ics_bytes)),
            matrix_path.to_string(),
            ics.matrix_case_count(),
            ics.matrix_case_sha256().to_string(),
            EncoderReferenceIdentity {
                standard: standard.to_string(),
                implementation: implementation.to_string(),
                version: version.to_string(),
            },
            Vec::from([EncoderSupplementalReferenceIdentity {
                decoder: EncoderReferenceDecoder::OpenHtj2k,
                scope: OPENHTJ2K_SCOPE.to_string(),
                implementation: OPENHTJ2K_IMPLEMENTATION.to_string(),
                version: OPENHTJ2K_VERSION.to_string(),
                source_url: OPENHTJ2K_SOURCE_URL.to_string(),
                source_commit: OPENHTJ2K_SOURCE_COMMIT.to_string(),
                executable_sha256: "4".repeat(64),
            }]),
            cases,
        )
        .expect("valid CPU encoder evidence")
    }

    fn passing_case(
        id: &str,
        mode: EncoderMode,
        reference_decoder: EncoderReferenceDecoder,
    ) -> EncoderCaseReport {
        let (lossless_exact, psnr_db, quality_status, quality_requirement) = match mode {
            EncoderMode::Lossless => (Some(true), None, EncoderQualityStatus::NotApplicable, None),
            EncoderMode::Lossy => (
                None,
                Some(40.0),
                EncoderQualityStatus::Pass,
                Some("test quality gate".to_string()),
            ),
        };
        EncoderCaseReport {
            id: id.to_string(),
            mode,
            status: CaseStatus::Pass,
            route: RouteKind::Cpu,
            reference_decoder,
            reference_decode_success: true,
            lossless_exact,
            encoded_bytes: Some(1),
            actual_bits_per_pixel: Some(1.0),
            psnr_db,
            psnr_infinite: false,
            quality_status,
            quality_requirement,
            quality_error: None,
            error: None,
            stages: cpu_stages(),
            accelerator_dispatches: Some(EncoderDispatchEvidence::default()),
        }
    }

    fn cpu_stages() -> Vec<EncodeRouteStage> {
        [
            EncodeRouteStageName::InputPreparation,
            EncodeRouteStageName::ForwardRct,
            EncodeRouteStageName::ForwardIct,
            EncodeRouteStageName::ForwardDwt53,
            EncodeRouteStageName::ForwardDwt97,
            EncodeRouteStageName::Quantization,
            EncodeRouteStageName::Tier1,
            EncodeRouteStageName::Packetization,
        ]
        .into_iter()
        .map(|stage| EncodeRouteStage {
            stage,
            location: ExecutionLocation::Cpu,
        })
        .chain(
            [
                EncodeRouteStageName::HostToDevice,
                EncodeRouteStageName::DeviceToHost,
            ]
            .into_iter()
            .map(|stage| EncodeRouteStage {
                stage,
                location: ExecutionLocation::NotUsed,
            }),
        )
        .collect()
    }
}
