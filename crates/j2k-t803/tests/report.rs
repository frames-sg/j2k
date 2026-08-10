// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_t803::{
    AcceleratorExecutionEvidence, CaseReport, CaseStatus, CorpusFile, EncodeRouteStage,
    EncodeRouteStageName, EncoderCaseReport, EncoderDispatchEvidence, EncoderEvidence, EncoderMode,
    EncoderQualityStatus, EncoderReferenceDecoder, EncoderReferenceIdentity,
    EncoderSupplementalReferenceIdentity, ExecutionLocation, IutIdentity,
    NativeComponentOracleEvidence, PlatformIdentity, ReportStatus, RouteKind, RouteStage,
    RouteStageName, T803Report, T803Suite,
};

fn report(cases: Vec<CaseReport>) -> T803Report {
    T803Report::new(
        T803Suite::Part1,
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
        "corpus/j2k-conformance/encoder-matrix-v2.toml".to_string(),
        1,
        "2".repeat(64),
        EncoderReferenceIdentity {
            standard: "ISO/IEC 15444-5 / ITU-T T.804".to_string(),
            implementation: "OpenJPEG".to_string(),
            version: "2.5.3".to_string(),
        },
        Vec::new(),
        Vec::from([EncoderCaseReport {
            id: "pairwise-01".to_string(),
            mode: EncoderMode::Lossless,
            status: CaseStatus::Pass,
            route: RouteKind::Cpu,
            reference_decoder: EncoderReferenceDecoder::OpenJpeg,
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
            accelerator_dispatches: Some(EncoderDispatchEvidence::default()),
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
        accelerator_execution: None,
        part15: None,
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

    assert_eq!(report.suite, T803Suite::Part1);

    let first = report.to_json().expect("serialize report");
    let second = report.to_json().expect("serialize report again");

    assert_eq!(first, second);
    assert!(first.ends_with('\n'));
    assert!(first.contains("\"schema_version\": 7"));
    assert!(first.contains("ISO/IEC 15444-4:2024 / ITU-T T.803 v3"));
    assert!(first.contains("ISO/IEC 15444-5 / ITU-T T.804"));
    let reparsed = T803Report::from_json(&first).expect("parse report");
    assert_eq!(reparsed, report);
    assert_eq!(reparsed.to_json().expect("reserialize report"), first);
}

#[test]
fn current_report_publishes_encoder_dispatch_counters() {
    let json = report(Vec::from([passing_case(cpu_stages())]))
        .to_json()
        .expect("serialize report");

    assert!(json.contains("\"accelerator_dispatches\""));
    assert!(json.contains("\"ht_code_block\": 0"));

    let mut missing = serde_json::from_str::<serde_json::Value>(&json).expect("report JSON");
    missing["encoder"]["cases"][0]
        .as_object_mut()
        .expect("encoder case object")
        .remove("accelerator_dispatches");
    let missing = serde_json::to_string_pretty(&missing).expect("report JSON");
    let error = T803Report::from_json(&missing)
        .expect_err("the current schema requires per-case dispatch counters");
    assert!(error.to_string().contains("dispatch counters"));

    let mut historical = serde_json::from_str::<serde_json::Value>(&missing).expect("report JSON");
    historical["schema_version"] = 6.into();
    let historical = serde_json::to_string_pretty(&historical).expect("historical report JSON");
    T803Report::from_json(&historical).expect("schema six remains readable");
}

#[test]
fn encoder_dispatch_counters_must_match_reported_stage_locations() {
    let json = report(Vec::from([passing_case(cpu_stages())]))
        .to_json()
        .expect("serialize report");
    let mut contradictory = serde_json::from_str::<serde_json::Value>(&json).expect("report JSON");
    contradictory["encoder"]["cases"][0]["accelerator_dispatches"]["deinterleave"] = 1.into();

    let contradictory =
        serde_json::to_string_pretty(&contradictory).expect("contradictory report JSON");
    let error = T803Report::from_json(&contradictory)
        .expect_err("a device dispatch cannot be reported as CPU work");

    assert!(error.to_string().contains("InputPreparation"));
}

#[test]
fn historical_schema_is_verifiable_but_cannot_be_reemitted() {
    let current = report(Vec::from([passing_case(cpu_stages())]))
        .to_json()
        .expect("current report");
    let mut historical = serde_json::from_str::<serde_json::Value>(&current).expect("report JSON");
    historical["schema_version"] = 3.into();
    for case in historical["cases"]
        .as_array_mut()
        .expect("report case array")
    {
        case.as_object_mut()
            .expect("report case object")
            .remove("accelerator_execution");
    }
    let historical = serde_json::to_string_pretty(&historical).expect("historical JSON");

    let parsed = T803Report::from_json(&historical).expect("read historical report");
    assert_eq!(parsed.schema_version, 3);
    let error = parsed
        .to_json()
        .expect_err("historical report schemas are read-only");
    assert!(error.to_string().contains("read-only"));
}

#[test]
fn schema_five_defaults_to_openjpeg_reference_evidence() {
    let current = report(Vec::from([passing_case(cpu_stages())]))
        .to_json()
        .expect("current report");
    let mut historical = serde_json::from_str::<serde_json::Value>(&current).expect("report JSON");
    historical["schema_version"] = 5.into();
    historical["encoder"]
        .as_object_mut()
        .expect("encoder evidence object")
        .remove("supplemental_reference_decoders");
    for case in historical["encoder"]["cases"]
        .as_array_mut()
        .expect("encoder case array")
    {
        case.as_object_mut()
            .expect("encoder case object")
            .remove("reference_decoder");
    }

    let historical = serde_json::to_string_pretty(&historical).expect("historical JSON");
    let parsed = T803Report::from_json(&historical).expect("read schema-five report");

    assert_eq!(parsed.schema_version, 5);
    assert_eq!(
        parsed.encoder.cases[0].reference_decoder,
        EncoderReferenceDecoder::OpenJpeg
    );
    assert!(parsed.encoder.supplemental_reference_decoders.is_empty());
}

#[test]
fn supplemental_reference_selection_requires_a_pinned_identity() {
    let mut evidence = encoder_evidence();
    evidence.cases[0].reference_decoder = EncoderReferenceDecoder::OpenHtj2k;

    let error = EncoderEvidence::new(
        evidence.ics_path,
        evidence.ics_sha256,
        evidence.matrix_path,
        evidence.matrix_case_count,
        evidence.matrix_case_sha256,
        evidence.reference_decoder,
        Vec::new(),
        evidence.cases,
    )
    .expect_err("supplemental decoder selection needs executable provenance");

    assert!(error.to_string().contains("OpenHTJ2K identity"));
}

#[test]
fn supplemental_reference_identity_is_accepted_when_selected() {
    let mut evidence = encoder_evidence();
    evidence.cases[0].reference_decoder = EncoderReferenceDecoder::OpenHtj2k;
    let supplemental = EncoderSupplementalReferenceIdentity {
        decoder: EncoderReferenceDecoder::OpenHtj2k,
        scope: "independent Part 15 interoperability evidence; not T.804".to_string(),
        implementation: "OpenHTJ2K".to_string(),
        version: "0.19.0".to_string(),
        source_url: "https://github.com/osamu620/OpenHTJ2K".to_string(),
        source_commit: "e0f7ae853220d1e359c438b0bb6ad6cb2b3899db".to_string(),
        executable_sha256: "4".repeat(64),
    };

    let rebuilt = EncoderEvidence::new(
        evidence.ics_path,
        evidence.ics_sha256,
        evidence.matrix_path,
        evidence.matrix_case_count,
        evidence.matrix_case_sha256,
        evidence.reference_decoder,
        Vec::from([supplemental]),
        evidence.cases,
    )
    .expect("selected supplemental decoder has complete provenance");

    assert_eq!(rebuilt.supplemental_reference_decoders.len(), 1);
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
        T803Suite::Part1,
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
            (RouteStageName::ColorOutput, ExecutionLocation::Cuda),
            (RouteStageName::HostToDevice, ExecutionLocation::Cuda),
            (RouteStageName::DeviceToHost, ExecutionLocation::Cuda),
        ]
        .into_iter()
        .map(|(stage, location)| RouteStage { stage, location })
        .collect(),
    );
    hybrid.id = "c6-c1p0-02-0".to_string();
    hybrid.route = RouteKind::Hybrid;
    hybrid.accelerator_execution = Some(AcceleratorExecutionEvidence {
        backend: ExecutionLocation::Cuda,
        ht_tier1_dispatches: 3,
        ht_refinement_dispatches: 1,
        classic_tier1_dispatches: 0,
        dequantization_dispatches: 2,
        idwt_dispatches: 1,
        mct_dispatches: 1,
        color_output_dispatches: 1,
        uploaded_payload_bytes: Some(4096),
        metal_host_inputs: None,
        device_to_host_completed: true,
    });

    let report = report([cpu, hybrid].into_iter().collect());

    assert_eq!(report.cases[0].route, RouteKind::Cpu);
    assert_eq!(report.cases[1].route, RouteKind::Hybrid);
    assert_eq!(report.decoder_routes.device_native, 0);
    assert_eq!(report.decoder_routes.hybrid, 1);
    assert_eq!(report.decoder_routes.cpu, 1);
    let json = report.to_json().expect("serialize observed execution");
    assert!(json.contains("\"ht_refinement_dispatches\": 1"));
    assert!(json.contains("\"uploaded_payload_bytes\": 4096"));
}

#[test]
fn hybrid_report_rejects_missing_or_mismatched_accelerator_observations() {
    let mut hybrid = passing_case(
        [
            (RouteStageName::Parsing, ExecutionLocation::Cpu),
            (RouteStageName::Tier1, ExecutionLocation::Cuda),
            (RouteStageName::Dequantization, ExecutionLocation::Cuda),
            (RouteStageName::Idwt, ExecutionLocation::Cuda),
            (RouteStageName::Mct, ExecutionLocation::NotUsed),
            (RouteStageName::ColorOutput, ExecutionLocation::Cuda),
            (RouteStageName::HostToDevice, ExecutionLocation::Cuda),
            (RouteStageName::DeviceToHost, ExecutionLocation::Cuda),
        ]
        .into_iter()
        .map(|(stage, location)| RouteStage { stage, location })
        .collect(),
    );
    hybrid.route = RouteKind::Hybrid;

    let error = T803Report::new(
        T803Suite::Part1,
        IutIdentity {
            name: "j2k-cuda".to_string(),
            version: "0.8.1".to_string(),
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
        Vec::from([CorpusFile {
            path: "files/input.j2k".to_string(),
            sha256: "0".repeat(64),
        }]),
        Vec::from([{
            let mut oracle = oracle_evidence();
            oracle.codestream_path = "files/input.j2k".to_string();
            oracle
        }]),
        Vec::from([hybrid]),
        encoder_evidence(),
    )
    .expect_err("hybrid routes require completed accelerator observations");

    assert!(error.to_string().contains("accelerator execution"));
}

#[test]
fn report_rejects_an_oracle_that_does_not_match_component_for_component() {
    let mut oracle = oracle_evidence();
    oracle.exact = false;
    oracle.openjpeg_components_sha256 = "4".repeat(64);

    let error = T803Report::new(
        T803Suite::Part1,
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
        "corpus/j2k-conformance/encoder-matrix-v2.toml".to_string(),
        1,
        "2".repeat(64),
        EncoderReferenceIdentity {
            standard: "ISO/IEC 15444-5 / ITU-T T.804".to_string(),
            implementation: "OpenJPEG".to_string(),
            version: "2.5.3".to_string(),
        },
        Vec::new(),
        Vec::from([case]),
    )
    .expect("a completed reference decode can still expose a standards failure");
}
