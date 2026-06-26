// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::adapter::adaptive_route::{
    J2kAdaptiveBackendRequest, J2kAdaptiveBenchmarkEvidence, J2kAdaptiveBenchmarks,
    J2kAdaptiveCodecMode, J2kAdaptiveOperation, J2kAdaptiveOutputResidency, J2kAdaptiveQualityMode,
    J2kAdaptiveRcaFinding, J2kAdaptiveRcaReason, J2kAdaptiveRouteKind, J2kAdaptiveRoutePlanner,
    J2kAdaptiveStage, J2kAdaptiveStageGateStatus, J2kAdaptiveStageOwner, J2kAdaptiveWorkload,
};
use j2k::{EncodeBackendPreference, J2kLosslessEncodeOptions, J2kLossyEncodeOptions};
use j2k_core::{BackendCapabilities, BackendKind, CpuFeatures};

fn metal_caps() -> BackendCapabilities {
    BackendCapabilities {
        cpu: CpuFeatures {
            avx2: false,
            sse41: false,
            neon: true,
        },
        metal: true,
        cuda: false,
    }
}

fn metal_cuda_caps() -> BackendCapabilities {
    BackendCapabilities {
        cpu: CpuFeatures {
            avx2: false,
            sse41: false,
            neon: true,
        },
        metal: true,
        cuda: true,
    }
}

fn cpu_caps() -> BackendCapabilities {
    BackendCapabilities {
        cpu: CpuFeatures::default(),
        metal: false,
        cuda: false,
    }
}

fn rgb_wsi_htj2k_encode() -> J2kAdaptiveWorkload {
    J2kAdaptiveWorkload::new(
        J2kAdaptiveOperation::Encode,
        J2kAdaptiveCodecMode::Htj2k,
        J2kAdaptiveQualityMode::Lossless,
        3,
        8,
        (512, 512),
        16,
    )
    .with_output_residency(J2kAdaptiveOutputResidency::Host)
}

fn approved_metal_benchmarks_for(workload: J2kAdaptiveWorkload) -> J2kAdaptiveBenchmarks {
    let mut benchmarks = J2kAdaptiveBenchmarks::default();
    for stage in J2kAdaptiveStage::ALL {
        if workload.logical_owner_for(stage) == J2kAdaptiveStageOwner::Gpu {
            benchmarks.push_stage(J2kAdaptiveBenchmarkEvidence::stage(
                stage,
                BackendKind::Metal,
                100_000,
                80_000,
                1.0,
            ));
        }
    }
    benchmarks.push_end_to_end(J2kAdaptiveBenchmarkEvidence::end_to_end(
        BackendKind::Metal,
        2_000_000,
        1_600_000,
        1.0,
    ));
    benchmarks
}

fn approved_cuda_benchmarks_for(workload: J2kAdaptiveWorkload) -> J2kAdaptiveBenchmarks {
    let mut benchmarks = J2kAdaptiveBenchmarks::default();
    for stage in J2kAdaptiveStage::ALL {
        if workload.logical_owner_for(stage) == J2kAdaptiveStageOwner::Gpu {
            benchmarks.push_stage(J2kAdaptiveBenchmarkEvidence::stage(
                stage,
                BackendKind::Cuda,
                100_000,
                80_000,
                1.0,
            ));
        }
    }
    benchmarks.push_end_to_end(J2kAdaptiveBenchmarkEvidence::end_to_end(
        BackendKind::Cuda,
        2_000_000,
        1_500_000,
        1.0,
    ));
    benchmarks
}

fn metal_stage_candidate_benchmarks_for(stage: J2kAdaptiveStage) -> J2kAdaptiveBenchmarks {
    let mut benchmarks = J2kAdaptiveBenchmarks::default();
    benchmarks.push_stage(J2kAdaptiveBenchmarkEvidence::stage(
        stage,
        BackendKind::Metal,
        100_000,
        70_000,
        1.0,
    ));
    benchmarks
}

#[test]
fn encode_backend_preference_helpers_select_clear_routes() {
    assert_eq!(
        J2kLosslessEncodeOptions::default()
            .with_accelerated_backend()
            .backend,
        EncodeBackendPreference::Auto
    );
    assert_eq!(
        J2kLosslessEncodeOptions::default()
            .with_cpu_only_backend()
            .backend,
        EncodeBackendPreference::CpuOnly
    );
    assert_eq!(
        J2kLosslessEncodeOptions::default()
            .with_strict_device_backend()
            .backend,
        EncodeBackendPreference::RequireDevice
    );
    assert_eq!(
        J2kLossyEncodeOptions::default()
            .with_accelerated_backend()
            .backend,
        EncodeBackendPreference::Auto
    );
}

#[test]
fn adaptive_planner_keeps_small_workloads_on_cpu_without_benchmark_gate() {
    let workload = J2kAdaptiveWorkload::new(
        J2kAdaptiveOperation::Encode,
        J2kAdaptiveCodecMode::Htj2k,
        J2kAdaptiveQualityMode::Lossless,
        3,
        8,
        (128, 128),
        1,
    );
    let report = J2kAdaptiveRoutePlanner::new(metal_caps())
        .plan(
            workload,
            J2kAdaptiveBackendRequest::Accelerated,
            &J2kAdaptiveBenchmarks::default(),
        )
        .expect("accelerated CPU route should plan");

    assert_eq!(report.route_kind, J2kAdaptiveRouteKind::CpuOnly);
    assert_eq!(report.selected_device, None);
    assert!(report.stage(J2kAdaptiveStage::MarkerParsing).is_some());
    assert!(report.stages.len() >= J2kAdaptiveStage::ALL.len());
    assert!(
        report
            .stages
            .iter()
            .all(|stage| stage.selected_backend == BackendKind::Cpu),
        "small ungated workload must stay CPU-only"
    );
}

#[test]
fn stage_candidate_remains_cpu_when_end_to_end_gate_is_missing() {
    let workload = rgb_wsi_htj2k_encode();
    let benchmarks = metal_stage_candidate_benchmarks_for(J2kAdaptiveStage::Dwt);

    let report = J2kAdaptiveRoutePlanner::new(metal_caps())
        .plan(
            workload,
            J2kAdaptiveBackendRequest::Accelerated,
            &benchmarks,
        )
        .expect("route should plan with stage evidence only");

    let dwt = report.stage(J2kAdaptiveStage::Dwt).expect("DWT decision");
    assert_eq!(report.route_kind, J2kAdaptiveRouteKind::CpuOnly);
    assert_eq!(report.selected_device, None);
    assert_eq!(dwt.logical_owner, J2kAdaptiveStageOwner::Gpu);
    assert_eq!(dwt.selected_backend, BackendKind::Cpu);
    assert_eq!(
        dwt.gate_status,
        J2kAdaptiveStageGateStatus::EndToEndGateBlocked
    );
    let improvement = dwt
        .improvement_percent
        .expect("stage candidate evidence should remain visible for RCA");
    assert!((improvement - 42.857_142_857_142_854).abs() < 1e-12);
}

#[test]
fn stage_candidate_remains_cpu_when_end_to_end_gate_fails() {
    let workload = rgb_wsi_htj2k_encode();
    let mut benchmarks = metal_stage_candidate_benchmarks_for(J2kAdaptiveStage::HtBlockCoding);
    benchmarks.push_end_to_end(J2kAdaptiveBenchmarkEvidence::end_to_end(
        BackendKind::Metal,
        2_000_000,
        1_950_000,
        1.0,
    ));

    let report = J2kAdaptiveRoutePlanner::new(metal_caps())
        .plan(
            workload,
            J2kAdaptiveBackendRequest::Accelerated,
            &benchmarks,
        )
        .expect("route should plan with a failing end-to-end gate");

    let ht = report
        .stage(J2kAdaptiveStage::HtBlockCoding)
        .expect("HT block decision");
    assert_eq!(report.route_kind, J2kAdaptiveRouteKind::CpuOnly);
    assert_eq!(report.selected_device, None);
    assert_eq!(ht.selected_backend, BackendKind::Cpu);
    assert_eq!(
        ht.gate_status,
        J2kAdaptiveStageGateStatus::EndToEndGateBlocked
    );
    assert!(ht.improvement_percent.is_some());
}

#[test]
fn approved_backend_is_not_masked_by_faster_stage_only_candidate() {
    let workload = rgb_wsi_htj2k_encode();
    let mut benchmarks = approved_cuda_benchmarks_for(workload);
    benchmarks.push_stage(J2kAdaptiveBenchmarkEvidence::stage(
        J2kAdaptiveStage::Dwt,
        BackendKind::Metal,
        100_000,
        70_000,
        1.0,
    ));

    let report = J2kAdaptiveRoutePlanner::new(metal_cuda_caps())
        .plan(
            workload,
            J2kAdaptiveBackendRequest::Accelerated,
            &benchmarks,
        )
        .expect("route should plan with mixed backend evidence");

    assert_eq!(report.route_kind, J2kAdaptiveRouteKind::Hybrid);
    assert_eq!(report.selected_device, Some(BackendKind::Cuda));
    assert_eq!(
        report
            .stage(J2kAdaptiveStage::Dwt)
            .expect("DWT decision")
            .selected_backend,
        BackendKind::Cuda
    );
    assert_eq!(
        report
            .stage(J2kAdaptiveStage::HtBlockCoding)
            .expect("HT block decision")
            .selected_backend,
        BackendKind::Cuda
    );
}

#[test]
fn rca_reclassification_is_exact_to_stage_and_backend() {
    let workload = rgb_wsi_htj2k_encode();
    let mut benchmarks = approved_metal_benchmarks_for(workload);
    benchmarks.push_stage(J2kAdaptiveBenchmarkEvidence::stage(
        J2kAdaptiveStage::Dwt,
        BackendKind::Metal,
        100_000,
        96_000,
        1.0,
    ));

    let report = J2kAdaptiveRoutePlanner::new(metal_caps())
        .with_rca_finding(J2kAdaptiveRcaFinding::reclassify_cpu(
            J2kAdaptiveStage::HtBlockCoding,
            BackendKind::Metal,
            J2kAdaptiveRcaReason::TransferSyncOverhead,
        ))
        .plan(
            workload,
            J2kAdaptiveBackendRequest::Accelerated,
            &benchmarks,
        )
        .expect("route should plan with non-matching RCA");

    let dwt = report.stage(J2kAdaptiveStage::Dwt).expect("DWT decision");
    assert_eq!(dwt.gate_status, J2kAdaptiveStageGateStatus::BlockedNeedsRca);
    assert_eq!(dwt.selected_backend, BackendKind::Cpu);
    assert!(report.has_unresolved_rca());

    let report = J2kAdaptiveRoutePlanner::new(metal_caps())
        .with_rca_finding(J2kAdaptiveRcaFinding::reclassify_cpu(
            J2kAdaptiveStage::Dwt,
            BackendKind::Cuda,
            J2kAdaptiveRcaReason::TransferSyncOverhead,
        ))
        .plan(
            workload,
            J2kAdaptiveBackendRequest::Accelerated,
            &benchmarks,
        )
        .expect("route should plan with backend-mismatched RCA");

    let dwt = report.stage(J2kAdaptiveStage::Dwt).expect("DWT decision");
    assert_eq!(dwt.gate_status, J2kAdaptiveStageGateStatus::BlockedNeedsRca);
    assert_eq!(dwt.selected_backend, BackendKind::Cpu);
    assert!(report.has_unresolved_rca());
}

#[test]
fn adaptive_planner_requires_stage_and_end_to_end_gates_before_default_gpu() {
    let workload = rgb_wsi_htj2k_encode();
    let planner = J2kAdaptiveRoutePlanner::new(metal_caps());

    let ungated = planner
        .plan(
            workload,
            J2kAdaptiveBackendRequest::Accelerated,
            &J2kAdaptiveBenchmarks::default(),
        )
        .expect("ungated route should still plan");
    assert_eq!(ungated.route_kind, J2kAdaptiveRouteKind::CpuOnly);
    assert!(ungated
        .stage(J2kAdaptiveStage::Dwt)
        .expect("DWT decision")
        .requires_rca());

    let gated = planner
        .plan(
            workload,
            J2kAdaptiveBackendRequest::Accelerated,
            &approved_metal_benchmarks_for(workload),
        )
        .expect("gated route should plan");

    assert_eq!(gated.route_kind, J2kAdaptiveRouteKind::Hybrid);
    assert_eq!(gated.selected_device, Some(BackendKind::Metal));
    assert_eq!(
        gated
            .stage(J2kAdaptiveStage::MarkerParsing)
            .expect("marker parsing decision")
            .selected_backend,
        BackendKind::Cpu
    );
    assert_eq!(
        gated
            .stage(J2kAdaptiveStage::Dwt)
            .expect("DWT decision")
            .selected_backend,
        BackendKind::Metal
    );
    assert_eq!(
        gated
            .stage(J2kAdaptiveStage::HtBlockCoding)
            .expect("HT block decision")
            .selected_backend,
        BackendKind::Metal
    );
}

#[test]
fn logical_gpu_loss_requires_rca_before_reclassification() {
    let workload = rgb_wsi_htj2k_encode();
    let mut benchmarks = approved_metal_benchmarks_for(workload);
    benchmarks.push_stage(J2kAdaptiveBenchmarkEvidence::stage(
        J2kAdaptiveStage::Dwt,
        BackendKind::Metal,
        100_000,
        96_000,
        1.0,
    ));

    let unresolved = J2kAdaptiveRoutePlanner::new(metal_caps())
        .plan(
            workload,
            J2kAdaptiveBackendRequest::Accelerated,
            &benchmarks,
        )
        .expect("route should plan with a blocked GPU stage");
    let dwt = unresolved
        .stage(J2kAdaptiveStage::Dwt)
        .expect("DWT decision");
    assert_eq!(dwt.gate_status, J2kAdaptiveStageGateStatus::BlockedNeedsRca);
    assert!(unresolved.has_unresolved_rca());

    let resolved = J2kAdaptiveRoutePlanner::new(metal_caps())
        .with_rca_finding(J2kAdaptiveRcaFinding::reclassify_cpu(
            J2kAdaptiveStage::Dwt,
            BackendKind::Metal,
            J2kAdaptiveRcaReason::TooSmallWorkload,
        ))
        .plan(
            workload,
            J2kAdaptiveBackendRequest::Accelerated,
            &benchmarks,
        )
        .expect("route should plan after RCA");
    let dwt = resolved.stage(J2kAdaptiveStage::Dwt).expect("DWT decision");
    assert_eq!(dwt.gate_status, J2kAdaptiveStageGateStatus::ReclassifiedCpu);
    assert_eq!(dwt.selected_backend, BackendKind::Cpu);
    assert!(!resolved.has_unresolved_rca());
}

#[test]
fn strict_device_request_fails_when_backend_is_unavailable() {
    let result = J2kAdaptiveRoutePlanner::new(cpu_caps()).plan(
        rgb_wsi_htj2k_encode(),
        J2kAdaptiveBackendRequest::StrictDevice(BackendKind::Metal),
        &J2kAdaptiveBenchmarks::default(),
    );

    assert!(result.is_err(), "strict Metal must not silently fall back");
}
