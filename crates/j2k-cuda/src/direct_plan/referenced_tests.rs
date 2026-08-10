// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    HtCodeBlockPayloadRanges, HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan,
    J2kClassicCodeBlockPayload, J2kCodeBlockSegment, J2kCodeBlockStyle, J2kCodestreamRange,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan,
    J2kRect, J2kSubBandType,
};

use super::*;

fn rect() -> J2kRect {
    J2kRect {
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 1,
    }
}

fn referenced_mixed_plan() -> J2kDirectGrayscalePlan {
    J2kDirectGrayscalePlan {
        dimensions: (1, 1),
        bit_depth: 8,
        steps: vec![
            J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
                band_id: 0,
                rect: rect(),
                width: 1,
                height: 1,
                irreversible_midpoint: false,
                jobs: vec![HtOwnedCodeBlockBatchJob {
                    output_x: 0,
                    output_y: 0,
                    data: Vec::new(),
                    cleanup_length: 1,
                    refinement_length: 0,
                    width: 1,
                    height: 1,
                    output_stride: 1,
                    missing_bit_planes: 7,
                    number_of_coding_passes: 1,
                    num_bitplanes: 8,
                    roi_shift: 0,
                    stripe_causal: false,
                    strict: true,
                    dequantization_step: 1.0,
                }],
            }),
            J2kDirectGrayscaleStep::ClassicSubBand(J2kOwnedSubBandPlan {
                band_id: 1,
                rect: rect(),
                width: 1,
                height: 1,
                irreversible_midpoint: false,
                jobs: vec![J2kOwnedCodeBlockBatchJob {
                    output_x: 0,
                    output_y: 0,
                    data: Vec::new(),
                    segments: vec![J2kCodeBlockSegment {
                        data_offset: 0,
                        data_length: 1,
                        start_coding_pass: 0,
                        end_coding_pass: 1,
                        use_arithmetic: true,
                    }],
                    width: 1,
                    height: 1,
                    output_stride: 1,
                    missing_bit_planes: 7,
                    number_of_coding_passes: 1,
                    total_bitplanes: 8,
                    roi_shift: 0,
                    sub_band_type: J2kSubBandType::LowLow,
                    style: J2kCodeBlockStyle {
                        selective_arithmetic_coding_bypass: false,
                        reset_context_probabilities: false,
                        termination_on_each_pass: false,
                        vertically_causal_context: false,
                        segmentation_symbols: false,
                    },
                    strict: true,
                    dequantization_step: 1.0,
                }],
            }),
        ],
    }
}

#[test]
fn referenced_htj2k_tile_accepts_observed_classic_and_ht_steps() {
    let encoded = [0xAA, 0xBB];
    let ht_payloads = [HtCodeBlockPayloadRanges {
        cleanup: J2kCodestreamRange {
            offset: 0,
            length: 1,
        },
        refinement: None,
    }];
    let classic_payloads = [J2kClassicCodeBlockPayload {
        first_range: 0,
        range_count: 1,
        combined_length: 1,
    }];
    let classic_ranges = [J2kCodestreamRange {
        offset: 1,
        length: 1,
    }];
    let mut shared_payload = Vec::new();
    let mut budget = HostPhaseBudget::new("mixed referenced CUDA plan test");

    let plan = CudaHtj2kDecodePlan::from_referenced_tile_grayscale_plan_into_shared(
        &referenced_mixed_plan(),
        &ht_payloads,
        &classic_payloads,
        &classic_ranges,
        &encoded,
        PixelFormat::Gray8,
        (0, 0),
        (1, 1),
        &mut shared_payload,
        &mut budget,
    )
    .expect("mixed referenced HTJ2K tile must retain both entropy coders");

    assert_eq!(shared_payload, encoded);
    assert_eq!(plan.code_blocks().len(), 1);
    assert_eq!(plan.classic_code_blocks().len(), 1);
    assert_eq!(plan.code_blocks()[0].payload_offset, 0);
    assert_eq!(plan.classic_code_blocks()[0].payload_offset, 1);
}
