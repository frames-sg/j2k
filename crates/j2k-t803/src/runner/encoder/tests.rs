// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{CaseStatus, EncoderIut, EncoderQualityStatus, EncoderReferenceDecoder};

#[cfg(all(feature = "metal-runner", target_os = "macos"))]
use crate::ReportStatus;

#[cfg(all(feature = "metal-runner", target_os = "macos"))]
use super::run_metal;
use super::{generate_input, load_sources};

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
fn cpu_ht_roi_matrix_case_roundtrips_through_production_decoder() {
    let sources = load_sources(EncoderIut::Cpu).expect("load CPU matrix and ICS");
    let case = sources
        .matrix
        .cases
        .iter()
        .find(|case| case.id == "part15-roi-lossless")
        .expect("HT ROI matrix case");
    let input = generate_input(case).expect("generate HT ROI input");
    let output = super::backend::encode_cpu_case(case, &input).expect("encode HT ROI case");
    let mut production_decoder =
        j2k::J2kDecoder::new(&output.codestream).expect("open encoded HT ROI case");
    let components = production_decoder
        .decode_native_components()
        .expect("decode encoded HT ROI case");

    assert_eq!(components.planes().len(), input.components.len());
    for (actual, expected) in components.planes().iter().zip(&input.components) {
        assert_eq!(actual.data(), expected.data);
    }
}

#[test]
fn all_t804_supported_cpu_matrix_cases_decode_with_openjpeg() {
    let sources = load_sources(EncoderIut::Cpu).expect("load CPU matrix and ICS");
    let reports = sources
        .matrix
        .selected_cases(EncoderIut::Cpu)
        .filter(|case| case.reference_decoder == EncoderReferenceDecoder::OpenJpeg)
        .map(|case| {
            let input = generate_input(case).expect("generate encoder input");
            super::evaluate::evaluate_case(
                case,
                &input,
                super::backend::encode_cpu_case(case, &input),
                None,
                None,
            )
        })
        .collect::<Vec<_>>();
    let failures = reports
        .iter()
        .filter(|case| {
            case.status != CaseStatus::Pass || case.quality_status == EncoderQualityStatus::Fail
        })
        .collect::<Vec<_>>();

    // OpenJPEG 2.5.3 rejects HT code-blocks with a nonzero ROI shift. The
    // remaining CPU case is exercised above and by the pinned OpenHTJ2K lane.
    assert_eq!(reports.len(), 55);
    assert!(
        reports.iter().all(|case| case.status == CaseStatus::Pass),
        "{failures:#?}"
    );
    assert!(
        reports
            .iter()
            .all(|case| case.quality_status != EncoderQualityStatus::Fail),
        "{failures:#?}"
    );
    let exact_lossy = reports
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

    assert_eq!(evidence.cases.len(), 35);
    assert_eq!(evidence.status, ReportStatus::Pass, "{failures:#?}");
    assert!(evidence.cases.iter().any(|case| {
        matches!(
            case.route,
            crate::RouteKind::Hybrid | crate::RouteKind::DeviceNative
        )
    }));
}
