// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_native::{
    HtOwnedCodeBlockBatchJob, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kRect,
};

#[test]
fn referenced_cuda_plan_concatenates_refinement_continuation_records() {
    let sub_band = HtOwnedSubBandPlan {
        band_id: 0,
        rect: J2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        width: 1,
        height: 1,
        irreversible_midpoint: false,
        jobs: vec![HtOwnedCodeBlockBatchJob {
            output_x: 0,
            output_y: 0,
            data: Vec::new(),
            cleanup_length: 2,
            refinement_length: 3,
            width: 1,
            height: 1,
            output_stride: 1,
            missing_bit_planes: 0,
            number_of_coding_passes: 3,
            num_bitplanes: 1,
            roi_shift: 0,
            stripe_causal: false,
            strict: true,
            dequantization_step: 1.0,
        }],
    };
    let plan = J2kDirectGrayscalePlan {
        dimensions: (1, 1),
        bit_depth: 1,
        steps: vec![J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
            band_id: sub_band.band_id,
            rect: sub_band.rect,
            width: sub_band.width,
            height: sub_band.height,
            irreversible_midpoint: sub_band.irreversible_midpoint,
            jobs: Vec::new(),
        })],
    };
    let input = [0x10, 0x11, 0xff, 0x20, 0xff, 0x21, 0x22];
    let payloads = [
        HtCodeBlockPayloadRanges {
            cleanup: J2kCodestreamRange {
                offset: 0,
                length: 2,
            },
            refinement: Some(J2kCodestreamRange {
                offset: 3,
                length: 1,
            }),
        },
        HtCodeBlockPayloadRanges {
            cleanup: J2kCodestreamRange {
                offset: 5,
                length: 0,
            },
            refinement: Some(J2kCodestreamRange {
                offset: 5,
                length: 2,
            }),
        },
    ];
    let (mut owners, _) = CudaPlanOwners::from_referenced_plan(&plan).unwrap();
    let mut records = payloads.iter();
    let mut shared = Vec::new();

    append_referenced_ht_subband(
        &mut owners,
        &sub_band,
        None,
        &mut records,
        &input,
        &mut shared,
    )
    .unwrap();

    assert!(records.next().is_none());
    assert_eq!(shared, [0x10, 0x11, 0x20, 0x21, 0x22]);
    assert_eq!(owners.code_blocks[0].payload_offset, 0);
    assert_eq!(owners.code_blocks[0].payload_len, 5);
}

#[test]
fn referenced_cuda_component_count_includes_refinement_continuation_records() {
    let plan = J2kDirectGrayscalePlan {
        dimensions: (1, 1),
        bit_depth: 1,
        steps: vec![J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
            band_id: 0,
            rect: J2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            width: 1,
            height: 1,
            irreversible_midpoint: false,
            jobs: vec![HtOwnedCodeBlockBatchJob {
                output_x: 0,
                output_y: 0,
                data: Vec::new(),
                cleanup_length: 2,
                refinement_length: 3,
                width: 1,
                height: 1,
                output_stride: 1,
                missing_bit_planes: 0,
                number_of_coding_passes: 3,
                num_bitplanes: 1,
                roi_shift: 0,
                stripe_causal: false,
                strict: true,
                dequantization_step: 1.0,
            }],
        })],
    };
    let input = [0x10, 0x11, 0xff, 0x20, 0xff, 0x21, 0x22, 0x30];
    let payloads = [
        HtCodeBlockPayloadRanges {
            cleanup: J2kCodestreamRange {
                offset: 0,
                length: 2,
            },
            refinement: Some(J2kCodestreamRange {
                offset: 3,
                length: 1,
            }),
        },
        HtCodeBlockPayloadRanges {
            cleanup: J2kCodestreamRange {
                offset: 5,
                length: 0,
            },
            refinement: Some(J2kCodestreamRange {
                offset: 5,
                length: 2,
            }),
        },
        HtCodeBlockPayloadRanges {
            cleanup: J2kCodestreamRange {
                offset: 7,
                length: 1,
            },
            refinement: None,
        },
    ];

    assert_eq!(
        referenced_ht_payload_record_count(&plan, &payloads, &input).unwrap(),
        2,
    );
}
