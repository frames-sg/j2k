// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_native::{HtCodeBlockPayloadRanges, J2kCodestreamRange};

fn test_job() -> J2kHtCleanupBatchJob {
    J2kHtCleanupBatchJob {
        coded_offset: 0,
        width: 1,
        height: 1,
        coded_len: 1,
        cleanup_length: 1,
        refinement_length: 0,
        missing_msbs: 0,
        num_bitplanes: 1,
        roi_shift: 0,
        number_of_coding_passes: 1,
        output_stride: 1,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: 0,
        irreversible_midpoint: 0,
    }
}

fn test_sub_band(band_id: u32, payload_source: PreparedHtPayloadSource) -> PreparedHtSubBand {
    PreparedHtSubBand {
        band_id,
        width: 1,
        height: 1,
        payload_source,
        jobs: vec![test_job()],
        execution_owner: Arc::new(PreparedHtExecutionOwner),
    }
}

#[test]
fn group_materializes_mixed_referenced_and_fragmented_payload_owners() {
    let input = Arc::<[u8]>::from([0x11]);
    let referenced = test_sub_band(
        0,
        PreparedHtPayloadSource::Referenced {
            input: input.clone(),
            ranges: vec![HtCodeBlockPayloadRanges {
                cleanup: J2kCodestreamRange {
                    offset: 0,
                    length: 1,
                },
                refinement: None,
            }],
        },
    );
    let fragmented = test_sub_band(1, PreparedHtPayloadSource::Contiguous(vec![0x22]));

    let group = prepare_ht_sub_band_group(
        0,
        2,
        &[&referenced, &fragmented],
        DirectTier1Mode::CpuUpload,
    )
    .unwrap();

    let PreparedHtPayloadSource::Contiguous(data) = group.payload_source else {
        panic!("a mixed group must materialize one stable contiguous payload arena");
    };
    assert_eq!(data, [0x11, 0x22]);
    assert_eq!(group.jobs[0].coded_offset, 0);
    assert_eq!(group.jobs[1].coded_offset, 1);
}
