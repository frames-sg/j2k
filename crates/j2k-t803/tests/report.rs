// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_t803::{
    CaseReport, CaseStatus, CorpusFile, EncodeRouteStage, EncodeRouteStageName, EncoderCaseReport,
    EncoderEvidence, EncoderMode, EncoderQualityStatus, EncoderReferenceIdentity,
    ExecutionLocation, IutIdentity, NativeComponentOracleEvidence, PlatformIdentity, ReportStatus,
    RouteKind, RouteStage, RouteStageName, T803Report,
};

fn report(cases: Vec<CaseReport>) -> T803Report {
    T803Report::new(
        IutIdentity {
            name: "j2k".to_string(),
            version: "0.8.0".to_string(),
            candidate_sha: "0123456789abcdef".to_string(),
            claim: "Profile-1 Cclass-1 candidate".to_string(),
        },
        PlatformIdentity {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            hardware: "test cpu".to_string(),
            driver: "not-applicable".to_string(),
        },
        "ac04b52e1fe38404912036c14f215099ea9a785f38644fbe76ae8f3d1523c86d".to_string(),
        ["parallel".to_string(), "simd".to_string()]
            .into_iter()
            .collect(),
        [CorpusFile {
            path: "files/codestreams_profile0/p0_13.j2k".to_string(),
            sha256: "0".repeat(64),
        }]
        .into_iter()
        .collect(),
        Vec::from([oracle_evidence()]),
        cases,
        encoder_evidence(),
    )
    .expect("valid report")
}

fn oracle_evidence() -> NativeComponentOracleEvidence {
    NativeComponentOracleEvidence {
        codestream_path: "files/codestreams_profile0/p0_13.j2k".to_string(),
        codestream_sha256: "0".repeat(64),
        selection: "COD MCT enabled with more than four codestream components".to_string(),
        implementation: "OpenJPEG".to_string(),
        version: "2.5.3".to_string(),
        library: "openjpeg-sys vendored openjp2".to_string(),
        component_count: 257,
        compared_sample_count: 257,
        production_components_sha256: "3".repeat(64),
        openjpeg_components_sha256: "3".repeat(64),
        exact: true,
    }
}

fn encoder_evidence() -> EncoderEvidence {
    EncoderEvidence::new(
        "corpus/j2k-conformance/encoder-ics-cpu.toml".to_string(),
        "1".repeat(64),
        "corpus/j2k-conformance/encoder-matrix-v1.toml".to_string(),
        1,
        "2".repeat(64),
        EncoderReferenceIdentity {
            standard: "ISO/IEC 15444-5 / ITU-T T.804".to_string(),
            implementation: "OpenJPEG".to_string(),
            version: "2.5.3".to_string(),
        },
        Vec::from([EncoderCaseReport {
            id: "pairwise-01".to_string(),
            mode: EncoderMode::Lossless,
            status: CaseStatus::Pass,
            route: RouteKind::Cpu,
            reference_decode_success: true,
            lossless_exact: Some(true),
            encoded_bytes: Some(123),
            actual_bits_per_pixel: Some(0.960_937_5),
            psnr_db: None,
            psnr_infinite: false,
            quality_status: EncoderQualityStatus::NotApplicable,
            quality_requirement: None,
            quality_error: None,
            error: None,
            stages: cpu_encode_stages(),
        }]),
    )
    .expect("valid encoder evidence")
}

fn cpu_encode_stages() -> Vec<EncodeRouteStage> {
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

fn passing_case(stages: Vec<RouteStage>) -> CaseReport {
    CaseReport {
        id: "c6-c1p0-01-0".to_string(),
        table: "C.6".to_string(),
        status: CaseStatus::Pass,
        route: RouteKind::Cpu,
        peak: Some(0),
        mse: Some(0.0),
        allowed_peak: 0,
        allowed_mse: Some(0.0),
        error: None,
        stages,
    }
}

fn cpu_stages() -> Vec<RouteStage> {
    [
        RouteStageName::Parsing,
        RouteStageName::Tier1,
        RouteStageName::Dequantization,
        RouteStageName::Idwt,
        RouteStageName::Mct,
        RouteStageName::ColorOutput,
    ]
    .into_iter()
    .map(|stage| RouteStage {
        stage,
        location: ExecutionLocation::Cpu,
    })
    .chain(
        [RouteStageName::HostToDevice, RouteStageName::DeviceToHost]
            .into_iter()
            .map(|stage| RouteStage {
                stage,
                location: ExecutionLocation::NotUsed,
            }),
    )
    .collect()
}

#[test]
fn report_json_is_deterministic_versioned_and_round_trips() {
    let report = report([passing_case(cpu_stages())].into_iter().collect());

    let first = report.to_json().expect("serialize report");
    let second = report.to_json().expect("serialize report again");

    assert_eq!(first, second);
    assert!(first.ends_with('\n'));
    assert!(first.contains("\"schema_version\": 3"));
    assert!(first.contains("ISO/IEC 15444-4:2024 / ITU-T T.803 v3"));
    assert!(first.contains("ISO/IEC 15444-5 / ITU-T T.804"));
    let reparsed = T803Report::from_json(&first).expect("parse report");
    assert_eq!(reparsed, report);
    assert_eq!(reparsed.to_json().expect("reserialize report"), first);
}

#[test]
fn report_json_round_trip_preserves_difficult_f64_values() {
    let mut case = passing_case(cpu_stages());
    case.mse = Some(0.247_063_802_083_333_34);
    case.allowed_mse = Some(1.0);
    let mut report = report(Vec::from([case]));
    report.encoder.cases[0].actual_bits_per_pixel = Some(13.244_893_054_554_193);

    let json = report.to_json().expect("serialize report");
    let reparsed = T803Report::from_json(&json).expect("parse report");

    assert_eq!(reparsed.to_json().expect("reserialize report"), json);
}

#[test]
fn report_status_fails_when_any_case_fails_or_errors() {
    let mut failed = passing_case(cpu_stages());
    failed.status = CaseStatus::Fail;
    failed.peak = Some(1);
    failed.error = None;
    assert_eq!(
        report([failed].into_iter().collect()).status,
        ReportStatus::Fail
    );

    let mut error = passing_case(cpu_stages());
    error.status = CaseStatus::Error;
    error.peak = None;
    error.mse = None;
    error.error = Some("decode failed".to_string());
    assert_eq!(
        report([error].into_iter().collect()).status,
        ReportStatus::Fail
    );
}

#[test]
fn report_rejects_route_labels_that_hide_cpu_assistance() {
    let stages = [
        (RouteStageName::Parsing, ExecutionLocation::Cpu),
        (RouteStageName::Tier1, ExecutionLocation::Cuda),
        (RouteStageName::Dequantization, ExecutionLocation::Cuda),
        (RouteStageName::Idwt, ExecutionLocation::Cuda),
        (RouteStageName::Mct, ExecutionLocation::Cuda),
        (RouteStageName::ColorOutput, ExecutionLocation::Cpu),
        (RouteStageName::HostToDevice, ExecutionLocation::Cuda),
        (RouteStageName::DeviceToHost, ExecutionLocation::Cuda),
    ]
    .into_iter()
    .map(|(stage, location)| RouteStage { stage, location })
    .collect();

    let error = T803Report::new(
        IutIdentity {
            name: "j2k-cuda".to_string(),
            version: "0.8.0".to_string(),
            candidate_sha: "abc".to_string(),
            claim: "adapter IUT candidate".to_string(),
        },
        PlatformIdentity {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            hardware: "test gpu".to_string(),
            driver: "test driver".to_string(),
        },
        "0".repeat(64),
        Vec::new(),
        [CorpusFile {
            path: "files/input.j2k".to_string(),
            sha256: "0".repeat(64),
        }]
        .into_iter()
        .collect(),
        Vec::from([{
            let mut oracle = oracle_evidence();
            oracle.codestream_path = "files/input.j2k".to_string();
            oracle
        }]),
        [{
            let mut case = passing_case(stages);
            case.route = RouteKind::DeviceNative;
            case
        }]
        .into_iter()
        .collect(),
        encoder_evidence(),
    )
    .expect_err("device-native label must reject CPU assistance");

    assert!(error.to_string().contains("device-native"));
}

#[test]
fn markdown_discloses_metrics_bounds_and_route_stages() {
    let report = report([passing_case(cpu_stages())].into_iter().collect());

    let markdown = report.to_markdown().expect("render Markdown");

    assert!(markdown.contains("Profile-1 Cclass-1 candidate"));
    assert!(markdown.contains("Device-native: 0 / 1"));
    assert!(markdown.contains("CPU-routed: 1 / 1"));
    assert!(markdown.contains("Native component oracle"));
    assert!(markdown.contains("257 components / 257 samples: exact"));
    assert!(markdown.contains("| c6-c1p0-01-0 | C.6 | pass | cpu | 0 / 0 | 0.000000 / 0.000000 |"));
    assert!(markdown.contains("color-output=cpu"));
    assert!(markdown.contains("Informative Annex D/F encoder evidence"));
    assert!(markdown.contains("OpenJPEG 2.5.3"));
    assert!(markdown.contains("Standards status: pass"));
    assert!(markdown.contains("Quality-gate status: pass"));
    assert!(markdown.contains(
        "Conformance does not establish robustness, security, adoption, or performance."
    ));
}

#[test]
fn one_report_can_disclose_cpu_and_hybrid_cases() {
    let cpu = passing_case(cpu_stages());
    let mut hybrid = passing_case(
        [
            (RouteStageName::Parsing, ExecutionLocation::Cpu),
            (RouteStageName::Tier1, ExecutionLocation::Cuda),
            (RouteStageName::Dequantization, ExecutionLocation::Cuda),
            (RouteStageName::Idwt, ExecutionLocation::Cuda),
            (RouteStageName::Mct, ExecutionLocation::Cuda),
            (RouteStageName::ColorOutput, ExecutionLocation::Cpu),
            (RouteStageName::HostToDevice, ExecutionLocation::Cuda),
            (RouteStageName::DeviceToHost, ExecutionLocation::Cuda),
        ]
        .into_iter()
        .map(|(stage, location)| RouteStage { stage, location })
        .collect(),
    );
    hybrid.id = "c6-c1p0-02-0".to_string();
    hybrid.route = RouteKind::Hybrid;

    let report = report([cpu, hybrid].into_iter().collect());

    assert_eq!(report.cases[0].route, RouteKind::Cpu);
    assert_eq!(report.cases[1].route, RouteKind::Hybrid);
    assert_eq!(report.decoder_routes.device_native, 0);
    assert_eq!(report.decoder_routes.hybrid, 1);
    assert_eq!(report.decoder_routes.cpu, 1);
}

#[test]
fn report_rejects_an_oracle_that_does_not_match_component_for_component() {
    let mut oracle = oracle_evidence();
    oracle.exact = false;
    oracle.openjpeg_components_sha256 = "4".repeat(64);

    let error = T803Report::new(
        IutIdentity {
            name: "j2k".to_string(),
            version: "0.8.0".to_string(),
            candidate_sha: "abc".to_string(),
            claim: "Profile-1 Cclass-1 candidate".to_string(),
        },
        PlatformIdentity {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            hardware: "test cpu".to_string(),
            driver: "not-applicable".to_string(),
        },
        "0".repeat(64),
        Vec::new(),
        Vec::from([CorpusFile {
            path: oracle.codestream_path.clone(),
            sha256: oracle.codestream_sha256.clone(),
        }]),
        Vec::from([oracle]),
        Vec::from([passing_case(cpu_stages())]),
        encoder_evidence(),
    )
    .expect_err("non-exact native component evidence must block the report");

    assert!(error.to_string().contains("component-for-component"));
}

#[test]
fn encoder_evidence_accepts_a_metadata_failure_after_reference_decode() {
    let mut case = encoder_evidence().cases.remove(0);
    case.mode = EncoderMode::Lossy;
    case.status = CaseStatus::Fail;
    case.reference_decode_success = true;
    case.lossless_exact = None;
    case.quality_status = EncoderQualityStatus::Fail;
    case.quality_requirement = Some("PSNR >= 30 dB".to_string());
    case.quality_error = Some("quality gate could not run".to_string());
    case.error = Some("decoded component metadata differs".to_string());

    EncoderEvidence::new(
        "corpus/j2k-conformance/encoder-ics-cpu.toml".to_string(),
        "1".repeat(64),
        "corpus/j2k-conformance/encoder-matrix-v1.toml".to_string(),
        1,
        "2".repeat(64),
        EncoderReferenceIdentity {
            standard: "ISO/IEC 15444-5 / ITU-T T.804".to_string(),
            implementation: "OpenJPEG".to_string(),
            version: "2.5.3".to_string(),
        },
        Vec::from([case]),
    )
    .expect("a completed reference decode can still expose a standards failure");
}
