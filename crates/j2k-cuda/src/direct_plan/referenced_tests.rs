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

#[test]
fn zero_pass_blocks_preserve_payload_order_without_entropy_jobs() {
    let mut direct = referenced_mixed_plan();
    for (step, zero_step) in direct.steps.iter_mut().zip(referenced_mixed_plan().steps) {
        match (step, zero_step) {
            (
                J2kDirectGrayscaleStep::HtSubBand(band),
                J2kDirectGrayscaleStep::HtSubBand(mut empty),
            ) => {
                let mut zero = empty.jobs.pop().unwrap();
                zero.number_of_coding_passes = 0;
                zero.cleanup_length = 0;
                band.jobs.insert(0, zero);
            }
            (
                J2kDirectGrayscaleStep::ClassicSubBand(band),
                J2kDirectGrayscaleStep::ClassicSubBand(mut empty),
            ) => {
                let mut zero = empty.jobs.pop().unwrap();
                zero.number_of_coding_passes = 0;
                zero.segments.clear();
                band.jobs.insert(0, zero);
            }
            _ => unreachable!(),
        }
    }
    let ht_payloads = [0, 1].map(|length| HtCodeBlockPayloadRanges {
        cleanup: J2kCodestreamRange { offset: 0, length },
        refinement: None,
    });
    let classic_payloads = [0, 1].map(|length| J2kClassicCodeBlockPayload {
        first_range: 0,
        range_count: length,
        combined_length: length,
    });
    let encoded = [0xaa, 0xbb];
    let mut payload = Vec::new();
    let referenced = CudaHtj2kDecodePlan::from_referenced_tile_grayscale_plan_into_shared(
        &direct,
        &ht_payloads,
        &classic_payloads,
        &[J2kCodestreamRange {
            offset: 1,
            length: 1,
        }],
        &encoded,
        PixelFormat::Gray8,
        (0, 0),
        (1, 1),
        &mut payload,
        &mut HostPhaseBudget::new("zero-pass referenced plan"),
    )
    .expect("referenced plan");
    assert_eq!(payload, encoded);
    assert_eq!(referenced.code_blocks().len(), 1);
    assert_eq!(referenced.classic_code_blocks().len(), 1);
    assert_eq!(referenced.classic_code_blocks()[0].payload_offset, 1);

    for step in &mut direct.steps {
        match step {
            J2kDirectGrayscaleStep::HtSubBand(band) => band.jobs[1].data = vec![encoded[0]],
            J2kDirectGrayscaleStep::ClassicSubBand(band) => band.jobs[1].data = vec![encoded[1]],
            _ => unreachable!(),
        }
    }
    let owned =
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
            .expect("owned plan");
    assert_eq!(owned.payload(), encoded);
    assert_eq!(owned.code_blocks(), referenced.code_blocks());
    assert_eq!(
        owned.classic_code_blocks(),
        referenced.classic_code_blocks()
    );
}
