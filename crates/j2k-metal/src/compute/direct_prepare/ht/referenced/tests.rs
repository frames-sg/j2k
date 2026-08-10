// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_native::{HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan, J2kRect};

#[test]
fn fragmented_refinement_records_are_prepared_once_as_contiguous_payload() {
    let input = Arc::<[u8]>::from([0x10, 0x11, 0xff, 0x20, 0xff, 0x21, 0x22]);
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
    let plan = HtOwnedSubBandPlan {
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
    let mut cursor = 0;

    let prepared = prepare_referenced_ht_sub_band(&plan, &input, &payloads, &mut cursor).unwrap();

    assert_eq!(cursor, payloads.len());
    let PreparedHtPayloadSource::Contiguous(data) = prepared.payload_source else {
        panic!("fragmented records must be materialized once during preparation");
    };
    assert_eq!(data, [0x10, 0x11, 0x20, 0x21, 0x22]);
    assert_eq!(prepared.jobs[0].coded_offset, 0);
    assert_eq!(prepared.jobs[0].coded_len, 5);
}
