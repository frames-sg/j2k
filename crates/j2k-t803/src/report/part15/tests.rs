// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{HtClaimSet, HtComplianceClass, Part15CaseMetadata};

use super::{
    derive_native_ht_coverage, HtCodeBlockSetMode, NativeHtCoverageAxis, Part15CaseEvidence,
    Part15CodestreamEvidence, Part15EvidenceClassification,
};
use crate::report::{
    AcceleratorExecutionEvidence, CaseReport, CaseStatus, ExecutionLocation, ReportStatus,
    RouteKind,
};

#[test]
fn native_ht_coverage_is_derived_only_from_observed_passing_dispatches() {
    let cases = vec![
        part15_case(CaseFacts {
            id: "minimum-mixed",
            bmagb: 8,
            mode: HtCodeBlockSetMode::Mixed,
            reversible: true,
            component_count: 1,
            quality_layers: 8,
            refinement: true,
            observed: true,
            status: CaseStatus::Pass,
        }),
        part15_case(CaseFacts {
            id: "maximum-ht-only",
            bmagb: 15,
            mode: HtCodeBlockSetMode::HtOnly,
            reversible: false,
            component_count: 3,
            quality_layers: 1,
            refinement: false,
            observed: true,
            status: CaseStatus::Pass,
        }),
        part15_case(CaseFacts {
            id: "failed-is-ignored",
            bmagb: 12,
            mode: HtCodeBlockSetMode::HtOnly,
            reversible: true,
            component_count: 1,
            quality_layers: 1,
            refinement: false,
            observed: true,
            status: CaseStatus::Fail,
        }),
    ];

    let coverage = derive_native_ht_coverage(&cases).expect("Part 15 selection");

    assert_eq!(coverage.selected_bmagb_min, 8);
    assert_eq!(coverage.selected_bmagb_max, 15);
    assert_eq!(coverage.observed_bmagb, vec![8, 15]);
    assert!(coverage.missing_axes.is_empty());
    assert_eq!(coverage.status, ReportStatus::Pass);
    assert_eq!(
        coverage
            .coverage
            .iter()
            .find(|case| case.axis == NativeHtCoverageAxis::Mixed)
            .map(|case| case.case_id.as_str()),
        Some("minimum-mixed")
    );
}

#[test]
fn cpu_routed_selected_boundary_remains_a_missing_native_axis() {
    let cases = vec![
        part15_case(CaseFacts {
            id: "minimum-mixed-cpu",
            bmagb: 8,
            mode: HtCodeBlockSetMode::Mixed,
            reversible: true,
            component_count: 1,
            quality_layers: 8,
            refinement: false,
            observed: false,
            status: CaseStatus::Pass,
        }),
        part15_case(CaseFacts {
            id: "maximum-ht-only",
            bmagb: 15,
            mode: HtCodeBlockSetMode::HtOnly,
            reversible: false,
            component_count: 3,
            quality_layers: 2,
            refinement: true,
            observed: true,
            status: CaseStatus::Pass,
        }),
    ];

    let coverage = derive_native_ht_coverage(&cases).expect("Part 15 selection");

    assert_eq!(coverage.status, ReportStatus::Fail);
    assert_eq!(coverage.observed_bmagb, vec![15]);
    assert!(coverage.missing_axes.contains(&NativeHtCoverageAxis::Mixed));
    assert!(coverage
        .missing_axes
        .contains(&NativeHtCoverageAxis::MinimumBmagb));
}

#[derive(Clone, Copy)]
struct CaseFacts {
    id: &'static str,
    bmagb: u8,
    mode: HtCodeBlockSetMode,
    reversible: bool,
    component_count: u16,
    quality_layers: u8,
    refinement: bool,
    observed: bool,
    status: CaseStatus,
}

fn part15_case(facts: CaseFacts) -> CaseReport {
    CaseReport {
        id: facts.id.to_string(),
        table: "C.6".to_string(),
        status: facts.status,
        route: if facts.observed {
            RouteKind::Hybrid
        } else {
            RouteKind::Cpu
        },
        peak: Some(0),
        mse: Some(0.0),
        allowed_peak: 0,
        allowed_mse: Some(0.0),
        error: None,
        stages: Vec::new(),
        accelerator_execution: facts.observed.then(|| accelerator(facts.refinement)),
        part15: Some(Part15CaseEvidence {
            classification: Part15EvidenceClassification::Formal,
            selection: Part15CaseMetadata {
                bset: facts.id.to_string(),
                claim_set: HtClaimSet::Ds0Hm,
                cclass: HtComplianceClass::Cclass1h,
                mmagb: 15,
                bmagb: facts.bmagb,
                base_peak: 0,
                base_mse: 0.0,
                additional_peak: 0,
                additional_mse: 0.0,
            },
            codestream: Part15CodestreamEvidence {
                pcap: 1,
                ccap15: 1,
                mode: facts.mode,
                multiple_ht_sets: facts.mode == HtCodeBlockSetMode::Mixed,
                roi: false,
                heterogeneous: facts.mode == HtCodeBlockSetMode::Mixed,
                ht_irreversible: !facts.reversible,
                bmagb: facts.bmagb,
                quality_layers: facts.quality_layers,
                default_ht_block_coding: true,
                default_mixed_block_coding: facts.mode == HtCodeBlockSetMode::Mixed,
                corresponding_profile_words: Vec::new(),
                reversible: facts.reversible,
                component_count: facts.component_count,
            },
        }),
    }
}

fn accelerator(refinement: bool) -> AcceleratorExecutionEvidence {
    AcceleratorExecutionEvidence {
        backend: ExecutionLocation::Metal,
        ht_tier1_dispatches: 1,
        ht_refinement_dispatches: usize::from(refinement),
        classic_tier1_dispatches: 0,
        dequantization_dispatches: 1,
        idwt_dispatches: 1,
        mct_dispatches: 0,
        color_output_dispatches: 1,
        uploaded_payload_bytes: None,
        metal_host_inputs: Some(1),
        device_to_host_completed: true,
    }
}
